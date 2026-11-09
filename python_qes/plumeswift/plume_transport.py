"""Persistent QES plume transport and concentration collection (GPL-3)."""
from dataclasses import dataclass, field

import numpy as np

from .boundaries import AxisBoundary, DomainBoundaries, NativeAxisBoundary, NativeDomainBoundaries
from .contracts import NumericPolicy
from .cpu_plume_random import NativeCPUStreams
from .deposition import DepositionField
from .interpolation import FieldSampler
from .io import netcdf_series
from .mesh_reflection import MeshReflection, NativeTriangleWalls, TriangleWalls
from .native_staircase import NativeStaircase
from .particles import ParticleIntegrator, ParticleSimulation
from .plume_collection import PlumeCollection, native_box_centers
from .powerlaw import NativePowerLawSampler, PowerLawSampler
from .statistics import StatisticsDirector
from .timing import milliseconds
from .walls import NoWalls, SolidExitWalls, VoxelWalls, cell_grid_mask, endpoint_cells, fluid_cell_mask


@dataclass(frozen=True)
class PlumeSample:
    time_ms: int
    fields: dict
    extra_statistics: dict = field(default_factory=dict)


def plume_collection_series(path, config, origin, *, native_math=False):
    axes = {axis: axis+'_c' for axis in 'xyz'}
    coordinates = native_box_centers(config) if native_math else config.coordinates
    return netcdf_series(path, coordinates, varying_fields=('t_avg',),
                         dimension_names=axes, coordinate_names=axes, time_origin=origin.isoformat())


class PlumeModel:
    def __init__(self, config, mesh=None, *, rng=None, track_particles=False, allocate_ids=None,
                 deposition=False, serial_slots=False, executor=None, native_collection_math=False,
                 native_arithmetic=False, native_release_indexing=False):
        self.config = config
        self.mesh = mesh
        self.simulation = None
        self.rng = np.random.default_rng(config.seed) if rng is None else rng
        self.native_math = native_arithmetic or isinstance(self.rng, NativeCPUStreams)
        self.serial_slots = serial_slots
        self.native_release_indexing = native_release_indexing
        self.executor = executor
        self.track_particles = track_particles
        self.allocate_ids = allocate_ids
        self.deposition_enabled = deposition
        self.time_ms = 0
        self.collection = PlumeCollection(config, native_math=native_collection_math,
                                          allow_zero_volume=config.allow_zero_collection_volume)
        self.statistics = StatisticsDirector(config.collection_start_ms, config.collection_period_ms)
        self.statistics.attach('concentration', self._collect, lambda _time_ms: self.collection.take())

    @property
    def next_sample(self):
        return self.statistics.next_output_ms

    def set_fields(self, workspace, turbulence, spacing):
        if turbulence is None:
            raise ValueError('Plume transport requires turbulent fields')
        c, a = workspace.coordinates, workspace.arrays
        native_math = self.native_math
        widths = np.asarray(a.get('dz_array', (spacing[2],)))
        if native_math and np.isnan(widths[-1]):
            widths = widths[:-1]
        spacing = (*spacing[:2], float(np.min(widths)))
        sampler_type = FieldSampler
        if self.config.interpolation == 'analyticalPowerLaw':
            sampler_type = NativePowerLawSampler if native_math else PowerLawSampler
        mode = 'nearest' if self.config.interpolation == 'nearestCell' else 'trilinear'
        sampler = sampler_type(c, a, turbulence, mode=mode, mixlength=turbulence['mixingLengths'],
                               native_spacing=spacing, native_math=native_math)
        solid = np.isin(a['icellflag'], (0, 2))

        def initial_active(positions):
            return fluid_cell_mask(sampler.candidate_cell_indices(positions), solid)

        def initial_in_grid(positions):
            return cell_grid_mask(sampler.candidate_cell_indices(positions), solid.shape)

        # In a stretched grid the original minimum-dz address can run out
        # of allocated cells before reaching the physical upper boundary.
        sampling_in_grid = initial_in_grid if native_math and np.any(widths != widths[0]) else None
        wall = self.config.wall
        if wall == 'setInactive':
            walls = SolidExitWalls(sampler, solid)
        elif wall == 'meshReflection' and self.mesh is not None:
            def endpoint_solid(positions):
                if not native_math:
                    indices, inside = endpoint_cells(sampler, positions)
                    return (inside & solid[indices]) | (~inside & (positions[:, 2] < sampler.zf[1]))
                k, j, i = sampler.candidate_cell_indices(positions)
                inside = (k >= 0) & (k < len(sampler.z))
                hit = np.ones(len(positions), dtype=bool)
                hit[inside] = solid[k[inside], j[inside], i[inside]]
                return hit
            wall_type = NativeTriangleWalls if native_math else TriangleWalls
            walls = wall_type(MeshReflection(self.mesh), endpoint_solid)
        elif wall in ('stairstepReflection', 'meshReflection'):
            walls = (NativeStaircase(sampler, solid) if sampler.native_math
                     else VoxelWalls(sampler, solid, endpoint_only=True))
        else:
            walls = NoWalls()
        if sampler.native_math:
            half_dx = .5*float(np.float32(spacing[0]))
            horizontal = [np.asarray(c[axis], dtype=np.float32) for axis in 'xy']
            bounds = [(float(axis[1])-half_dx, float(axis[-2])+half_dx) for axis in horizontal]
            boundary_type, axis_type = NativeDomainBoundaries, NativeAxisBoundary
        else:
            bounds = [(sampler.xf[1], sampler.xf[-2]), (sampler.yf[1], sampler.yf[-2])]
            boundary_type, axis_type = DomainBoundaries, AxisBoundary
        bounds.append((sampler.zf[1], sampler.zf[-2]))
        boundaries = boundary_type(tuple(axis_type(*bound, mode)
                                         for bound, mode in zip(bounds, self.config.boundaries, strict=True)))
        if native_math:
            # QES scans float32 standard deviations and skips NaNs because
            # their comparison with the current maximum is false.
            maximum_sigma = max(np.fmax.reduce(np.sqrt(np.abs(turbulence[name].astype(np.float32))).ravel(),
                                               initial=np.float32(-1e-9)) for name in ('txx', 'tyy', 'tzz'))
            rogue_velocity = np.float32(10*float(maximum_sigma))
        else:
            rogue_velocity = 10*np.sqrt(max(np.abs(turbulence[name]).max() for name in ('txx', 'tyy', 'tzz')))
        if self.simulation is None:
            policy = NumericPolicy(stress_invariant_tolerance=self.config.stress_tolerance)
            deposition = DepositionField(sampler, a['icellflag']) if self.deposition_enabled else None
            self.simulation = ParticleSimulation((), ParticleIntegrator(
                sampler, boundaries, walls, self.rng, policy=policy, courant=self.config.courant,
                main_step=self.config.step_s, min_spacing=min(spacing), rogue_velocity=rogue_velocity,
                deposition=deposition, executor=self.executor, native_arithmetic=native_math,
                sampling_in_grid=sampling_in_grid),
                initial_active=initial_active, track_particles=self.track_particles, allocate_ids=self.allocate_ids,
                serial_slots=self.serial_slots, native_release_indexing=self.native_release_indexing,
                initial_in_grid=initial_in_grid)
        else:
            self.simulation.integrator.fields = sampler
            self.simulation.integrator.walls = walls
            self.simulation.integrator.boundaries = boundaries
            self.simulation.integrator.rogue_velocity = rogue_velocity
            self.simulation.integrator.sampling_in_grid = sampling_in_grid
            self.simulation.initial_active = initial_active
            self.simulation.initial_in_grid = initial_in_grid
            if self.deposition_enabled:
                self.simulation.integrator.deposition.set_fields(sampler, a['icellflag'])

    def _collect(self, _time_ms, dt):
        batch = self.simulation.batch
        self.collection.collect(batch.positions, batch.mass, dt)

    def advance_step(self, end_ms):
        dt = float(np.float32(min(self.config.step_s, (end_ms-self.time_ms)/1000)))
        self.simulation.time_s = self.time_ms/1000
        self.simulation.step(dt)
        self.time_ms += milliseconds(dt)
        self.simulation.time_s = self.time_ms/1000
        results = self.statistics.compute(self.time_ms, dt)
        if results is None:
            return None
        return PlumeSample(self.time_ms, results.pop('concentration'), results)

    def advance(self, end_ms, on_step=None):
        while self.time_ms < end_ms:
            sample = self.advance_step(end_ms)
            if on_step is not None:
                on_step(self)
            if sample is not None:
                yield sample
