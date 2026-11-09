"""Adapt advancing fire cells to the shared plume transport (GPL-3)."""

import numpy as np

from .contracts import UnportedFeature
from .cpu_plume_random import NativeCPUStreams
from .particle_ids import ParticleIDs
from .plume_config import PlumeConfig as SmokeConfig, parse_plume_config
from .plume_transport import PlumeModel, PlumeSample as SmokeSample
from .sources import Point, Source, StepRelease
from .timing import milliseconds
from .qes_xml import read_qes_xml


def legacy_smoke_sites(fire):
    """Collect and clear sites visited by the original Smoke::genSmoke diagnostic."""
    sites = []
    dx, dy = np.float32(fire.dx), np.float32(fire.dy)
    for j, i in np.argwhere(fire.smoke[1:-1, 1:-1] == 1)+1:
        sites.append((float(np.float32(i)*dx), float(np.float32(j)*dy),
                      float(np.float32(fire.terrain[j, i])+np.float32(1)), 20))
        fire.smoke[j, i] = 0
    return tuple(sites)


def read_smoke_config(path, *, seed=0):
    return parse_smoke_config(read_qes_xml(path), seed=seed)


def parse_smoke_config(root, *, seed=0, legacy_plume_xml=False):
    allowed = {'plumeParameters', 'collectionParameters', 'boundaryConditions', 'sourceParameters',
               'particleOutputParameters', 'simulationParameters', 'sources', 'particleParameters'}
    extra = {node.tag for node in root}-allowed
    if extra and not legacy_plume_xml:
        raise UnportedFeature(f'Smoke runner cannot apply plume XML sections {sorted(extra)}')
    return parse_plume_config(root, seed=seed, legacy_plume_xml=legacy_plume_xml)


class SmokePlume:
    def __init__(self, config, mesh=None, *, track_particles=False, native_collection_math=False,
                 specifications=None, native_threads=None, source_seed=None, native_release_indexing=False,
                 native_full_domain_bounds=False):
        specifications = {} if specifications is None else specifications
        # qesFire registers the XML sources before adding its own smoke model.
        self.first_smoke_source = sum(len(spec.sources) for spec in specifications.values())
        self.specifications = {tag: spec for tag, spec in specifications.items() if tag != 'smoke'}
        allocate_ids = ParticleIDs(native=native_threads is not None).take
        self.replay = None if native_threads is None else NativeCPUStreams(config.seed, native_threads)
        self.source_seed = config.seed if self.replay is not None and source_seed is None else source_seed
        self.native_full_domain_bounds = native_full_domain_bounds
        self.model = PlumeModel(config, mesh, rng=self.replay, track_particles=track_particles,
                                allocate_ids=allocate_ids,
                                native_collection_math=native_collection_math,
                                native_release_indexing=native_release_indexing)
        tags = sorted(self.specifications)
        rngs = np.random.SeedSequence(config.seed).spawn(len(tags))
        models = {tag: PlumeModel(config, mesh, rng=self.replay if self.replay is not None else np.random.default_rng(rng),
                                  track_particles=track_particles,
                                  allocate_ids=allocate_ids, native_collection_math=native_collection_math,
                                  native_release_indexing=native_release_indexing)
                  for tag, rng in zip(tags, rngs, strict=True)}
        models['smoke'] = self.model
        self.models = dict(sorted(models.items()))
        self.source_count = 0

    def _attach_sources(self, fire, start_ms):
        sources = []
        for j, i in np.argwhere(fire.smoke[1:-1, 1:-1] == 1)+1:
            if self.replay is None:
                position = (float(i*fire.dx), float(j*fire.dy), float(fire.terrain[j, i]+1))
            else:
                position = (np.float32(i)*np.float32(fire.dx), np.float32(j)*np.float32(fire.dy),
                            np.float32(fire.terrain[j, i])+np.float32(1))
            end_ms = start_ms+milliseconds(fire.properties.residence_time[j, i])
            sources.append(Source(Point(position), StepRelease(20, 0., start_ms/1000, end_ms/1000),
                                  source_id=self.first_smoke_source+self.source_count))
            self.source_count += 1
            fire.smoke[j, i] = 0
        self.model.simulation.sources += tuple(sources)

    def advance(self, fire, workspace, turbulence, start_ms, end_ms, spacing, on_step=None):
        callback = None if on_step is None else lambda _time, _models: on_step(self.model)
        return tuple(sample for tag, sample in self.advance_models(
            fire, workspace, turbulence, start_ms, end_ms, spacing, callback) if tag == 'smoke')

    def advance_models(self, fire, workspace, turbulence, start_ms, end_ms, spacing, on_step=None, *, progress=None):
        for tag, model in self.models.items():
            first = model.simulation is None
            model.set_fields(workspace, turbulence, spacing)
            if first and tag in self.specifications:
                boundaries = model.simulation.integrator.boundaries
                model.simulation.sources = tuple(spec.create(boundaries, source_seed=self.source_seed,
                                                             native_full_domain_bounds=self.native_full_domain_bounds)
                                                 for spec in self.specifications[tag].sources)
        self._attach_sources(fire, start_ms)
        if progress is not None:
            progress.begin(self.model.time_ms)
        while self.model.time_ms < end_ms:
            for tag, model in self.models.items():
                sample = model.advance_step(end_ms)
                if sample is not None:
                    yield tag, sample
            if on_step is not None:
                on_step(self.model.time_ms, self.models)
            if progress is not None:
                progress.step(self.model.time_ms, end_ms, self.models)

    def accounting(self):
        return dict(sources=self.source_count, **self.model.simulation.accounting())

    def model_accounting(self):
        return {tag: dict(sources=len(spec.sources), **self.models[tag].simulation.accounting())
                for tag, spec in self.specifications.items()}
