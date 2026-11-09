"""Standalone driver for the original experimental QES GPU plume stages."""
from itertools import islice
from operator import index
from pathlib import Path

import numpy as np

from .cuda_particle_buffer import ACTIVE, ROGUE, CudaParticleBuffer
from .cuda_plume_advection import advect_particles
from .cuda_plume_boundaries import boundary_ids
from .cuda_plume_collection import CudaPlumeCollection
from .cuda_plume_interpolation import CudaPlumeInterpolation
from .cuda_plume_random import CudaPlumeRandom
from .cuda_plume_release import release_particles
from .particle_ids import ParticleIDs


class CudaPlumeRun:
    """Advance fixed-duration steps with a bounded device particle pool."""

    def __init__(self, interpolation, collection, bounds, capacity, *, seed=1234, boundaries=('exiting',)*3):
        self.capacity = index(capacity)
        if self.capacity <= 0 or interpolation.device != collection.device:
            raise ValueError('Expected positive capacity and GPU stages on the same device')
        self.bounds = np.array(bounds, dtype=np.float32, copy=True)
        if (self.bounds.shape != (3, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('Invalid XYZ domain bounds')
        self.interpolation = interpolation
        self.collection = collection
        self.boundaries = tuple(boundaries)
        boundary_ids(self.boundaries)
        self.random = CudaPlumeRandom(seed=seed, device=interpolation.device)
        self.random.create('advect', 3*self.capacity)
        self.particles = CudaParticleBuffer(self.capacity, device=interpolation.device)
        self._scratch = CudaParticleBuffer(self.capacity, device=interpolation.device)
        self.steps = 0
        self.requested = 0
        self.inserted = 0

    def _allocate_ids(self, count):
        return ParticleIDs(native=True, start=self.requested).take(count).astype(np.uint32)

    def step(self, release_count, *, position=(20., 50., 70.), collect=True):
        release_count = index(release_count)
        with self.particles.cp.cuda.Device(self.particles.device):
            active = self.particles.compact_to(self._scratch)
            inserted = release_particles(self._scratch, active, release_count, self.interpolation, self.random,
                                         self._allocate_ids, position=position)
            count = active+inserted
            # New allocations start with zero history. Native insertion leaves
            # previous stress and current fluctuations intact when reusing slots.
            self.interpolation.sample(self._scratch, count)
            draws = self.random.generate('advect')[:3*count].reshape(3, count)
            advect_particles(self._scratch, draws, self.bounds, count, boundaries=self.boundaries)
            if collect:
                self.collection.collect(self._scratch.positions[:count], self._scratch.state[:count], 1.)
        self.particles, self._scratch = self._scratch, self.particles
        self.steps += 1
        self.requested += release_count
        self.inserted += inserted


def _wind_schedule(dates, first, steps):
    start = dates[first]
    offsets = [(date-start).total_seconds() for date in dates[first:]]
    if any(end < begin for begin, end in zip(offsets, offsets[1:])):
        raise ValueError('GPU plume wind timestamps must be nondecreasing')
    for record, (begin, end) in enumerate(zip(offsets, offsets[1:]+[steps]), first):
        if begin >= steps:
            break
        end = min(end, steps)
        if end != int(end):
            raise ValueError('GPU plume wind changes must fall on whole-second steps')
        if end > begin:
            yield record, int(end)


def _uniform_bounds(workspace, spacing):
    x, y, z, z_face = (workspace.coordinates[key] for key in ('x', 'y', 'z', 'z_face'))
    dx, dy, dz = spacing
    if (not np.allclose(x, (np.arange(len(x))+.5)*dx, rtol=1e-6)
            or not np.allclose(y, (np.arange(len(y))+.5)*dy, rtol=1e-6)
            or not np.allclose(z, (np.arange(len(z))-.5)*dz, rtol=1e-6)
            or not np.allclose(z_face, (np.arange(len(z_face))-1)*dz, rtol=1e-6)):
        raise ValueError('Experimental CUDA plume requires a uniform local grid with a ground ghost layer')
    return np.array([x[0], y[0], 0.]), np.array([x[-1], y[-1], z[-1]])


def _box_bounds(bounds, purpose):
    bounds = np.asarray(bounds, dtype=np.float32)
    if (bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError(f'Expected finite increasing XYZ {purpose} bounds')
    return bounds


def _source_specs(path, model):
    from .plume_sources import read_models
    from .qes_xml import read_qes_xml

    root = read_qes_xml(path)
    if root.tag != 'QESPlumeParameters':
        raise ValueError('Expected QESPlumeParameters root for GPU source XML')
    models = read_models(root, step=1.)
    if model is None and len(models) == 1:
        model = next(iter(models))
    if model not in models or not models[model].sources:
        raise ValueError('GPU source XML requires a model with sources; select its tag with --source-model')
    return model, models[model].sources


def _release_positions(sources, time):
    batches = []
    for source in sources:
        count, _ = source.release.count_and_mass(time, 1.)
        if count:
            batches.append(source.geometry.sample(count, None))
    if not batches:
        return np.empty((0, 3), dtype=np.float32)
    positions = np.concatenate(batches, dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError('GPU source geometry produced a non-finite position')
    return positions


def run_cuda_plume(winds, turbulence, output, *, steps, release_count=None, capacity, source=None, boxes, seed=1234,
                   record=0, collection_start=0, boundaries=('exiting',)*3, wind_series=False,
                   source_xml=None, source_model=None, source_seed=None, collection_bounds=None,
                   transport_bounds=None):
    """Advance fixed one-second GPU steps on a uniform NetCDF grid."""
    from .plume_fields import open_plume_fields

    steps, record, collection_start = map(index, (steps, record, collection_start))
    if steps <= 0 or record < 0 or not 0 <= collection_start < steps:
        raise ValueError('Invalid GPU plume steps, record, release count or collection start')
    output = Path(output)
    if output.suffix.lower() != '.npz':
        raise ValueError('Experimental GPU plume output must use the .npz extension')
    if source_xml is None:
        if release_count is None or source_model is not None or source_seed is not None:
            raise ValueError('Provide a fixed release count or source XML; model and source seed require XML')
        release_count = index(release_count)
        if release_count < 0:
            raise ValueError('GPU plume release count must be nonnegative')
        source = np.asarray((20., 50., 70.) if source is None else source, dtype=float)
    else:
        if release_count is not None or source is not None:
            raise ValueError('GPU source XML cannot be combined with a fixed release count or point')
        source_model, specifications = _source_specs(source_xml, source_model)
        source_seed = index(seed if source_seed is None else source_seed)
    if len(boxes) != 3 or min(boxes) <= 0:
        raise ValueError('Expected positive collection box counts')
    if collection_bounds is not None:
        collection_bounds = _box_bounds(collection_bounds, 'collection')
    if transport_bounds is not None:
        transport_bounds = _box_bounds(transport_bounds, 'transport')
    model = None
    next_record = 0
    used_records = []
    with open_plume_fields(winds, turbulence, include_tke=True) as (dates, fields):
        if record >= len(dates):
            raise ValueError('GPU plume wind record is outside the input series')
        schedule = tuple(_wind_schedule(dates, record, steps)) if wind_series else ((record, steps),)
        for field_record, end_step in schedule:
            workspace, turb, spacing = next(islice(fields, field_record-next_record, None))
            next_record = field_record+1
            if 'nuT' not in turb:
                turb['nuT'] = np.zeros_like(turb['CoEps'])
            if model is None:
                grid_lower, grid_upper = _uniform_bounds(workspace, spacing)
                if transport_bounds is None:
                    lower, upper = grid_lower, grid_upper
                else:
                    lower, upper = transport_bounds.T
                    if np.any(lower < grid_lower) or np.any(upper > grid_upper):
                        raise ValueError('GPU plume transport bounds must lie inside the interpolation domain')
                if source_xml is None:
                    if source.shape != (3,) or not np.all((source > grid_lower) & (source < grid_upper)):
                        raise ValueError('GPU plume point source must lie inside the interpolation domain')
                else:
                    from .boundaries import AxisBoundary, DomainBoundaries
                    policies = DomainBoundaries(tuple(AxisBoundary(lo, hi, mode)
                                                       for lo, hi, mode in zip(lower, upper, boundaries, strict=True)))
                    sources = tuple(spec.create(policies, source_seed=source_seed) for spec in specifications)
                if collection_bounds is None:
                    collection_lower, widths = lower, (upper-lower)/boxes
                else:
                    collection_lower, collection_upper = collection_bounds.T
                    widths = np.divide(collection_upper-collection_lower, boxes, dtype=np.float32)
                interpolation = CudaPlumeInterpolation(workspace.arrays, turb, spacing)
                collection = CudaPlumeCollection(collection_lower, widths, boxes)
                model = CudaPlumeRun(interpolation, collection, np.column_stack((lower, upper)), capacity,
                                    seed=seed, boundaries=boundaries)
            else:
                model.interpolation.load_fields(workspace.arrays, turb)
            used_records.append(field_record)
            while model.steps < end_step:
                if source_xml is None:
                    count, positions = release_count, source
                else:
                    positions = _release_positions(sources, model.steps)
                    count = len(positions)
                model.step(count, position=positions, collect=model.steps >= collection_start)
    particles = model.particles
    state = particles.state.get()
    counts = collection.copyback()
    coordinates = {axis: lo+(np.arange(n)+.5)*width
                   for axis, lo, n, width in zip('xyz', collection_lower, boxes, widths)}
    active = state == ACTIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream, **coordinates, p_count=counts, t_avg=collection.averaging_s,
                            number_concentration=counts/(collection.averaging_s*collection.volume),
                            active_ids=particles.ids.get()[active], active_positions=particles.positions.get()[active])
    report = dict(steps=model.steps, requested=model.requested, inserted=model.inserted, active=int(active.sum()),
                  rogue=int((state == ROGUE).sum()), samples=int(counts.sum()), averaging_s=collection.averaging_s)
    if wind_series:
        report['wind_records'] = used_records
    if source_xml is not None:
        report.update(source_model=source_model, sources=len(sources), source_seed=source_seed)
    return report
