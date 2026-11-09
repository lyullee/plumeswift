"""Coupled fire and wind execution in the qesFireMain order (GPL-3)."""
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from numpy import ndarray

from .cpu_plume_random import resolve_cpu_seed
from .fire_case import FireConfig, _parse_fire_config
from .fire_front import cpu_front, cuda_front
from .fire_output import native_fire_series
from .fire_potential import HeatPotential
from .fire_smoke import SmokeConfig, SmokePlume, SmokeSample, parse_smoke_config
from .io import netcdf_series
from .numerics.native_cpu_rb import NativeCPUSOR
from .particle_output import ParticleOutput, ParticleOutputConfig, read_particle_output
from .concentration_output import native_collection_series
from .plume_sources import ModelSpec, read_models
from .plume_progress import PlumeProgress
from .plume_transport import plume_collection_series
from .qes_xml import read_qes_xml
from .timing import milliseconds
from .turbulence_output import native_turbulence_series
from .wind_case import WindCase, _read_wind_root
from .wind_run import legacy_wind_backend
from .wind_visualization import cell_fields
from .wind_xml import normalize_zero_wrf_startup


@dataclass(frozen=True)
class FireFrame:
    time: datetime
    fire_time_s: float
    dt_s: float
    fields: dict
    coordinates: dict
    turbulence: dict | None
    wind_coordinates: dict
    smoke_samples: tuple[SmokeSample, ...] = ()
    smoke_accounting: dict | None = None
    plume_samples: tuple[tuple[str, SmokeSample], ...] = ()
    plume_accounting: dict | None = None


@dataclass(frozen=True)
class CoupledFireCase:
    wind: WindCase
    fire: FireConfig
    potential: HeatPotential | None
    smoke: SmokeConfig | None = None
    particle_output: ParticleOutputConfig | None = None
    initialize_front: Callable[[ndarray], ndarray] = cpu_front
    plume_models: dict[str, ModelSpec] | None = None
    native_threads: int | None = None
    source_seed: int | None = None
    native_flame_sampling: bool = False
    native_release_indexing: bool = False
    native_fire_clock: bool = False
    native_full_domain_bounds: bool = False
    narrow_band: bool = True
    zero_wrf_steps: bool = False

    def _initialize_fire(self, workspace, spacing):
        return self.fire.initialize(workspace, spacing, initialize_front=self.initialize_front,
                                    native_flame_sampling=self.native_flame_sampling,
                                    native_fire_clock=self.native_fire_clock)

    def _smoke_mesh(self, workspace, diagnostics):
        if self.smoke is None or self.smoke.wall != 'meshReflection':
            return None
        mesh = workspace.terrain_mesh
        if mesh is None and diagnostics is not None:
            print('[WARRNING] mesh wall reflection method not available', file=diagnostics, flush=True)
        return mesh

    def frames(self, on_smoke_step=None, *, native_collection_math=False, include_gradients=False,
               native_fire_format=False, on_plume_step=None, progress=None, fire_progress_stream=None,
               plume_diagnostics=None):
        if self.zero_wrf_steps:
            # The original builds terrain and initializes fire before its wind loop.
            workspace = self.wind.initialize_geometry(prepare_mixing=True)
            self._initialize_fire(workspace, self.wind.domain.spacing)
            self._smoke_mesh(workspace, plume_diagnostics)
            return
        start = datetime.fromisoformat(self.wind.timestamp)
        final = start+timedelta(milliseconds=milliseconds(self.fire.duration))
        cases = tuple(self.wind.iter_steps())
        fire = None
        smoke = None
        previous = None
        previous_turbulence = None
        reuse_multiplier = isinstance(self.wind.poisson, NativeCPUSOR)
        if on_smoke_step is not None:
            def notify_plume_step(time_ms, models):
                on_smoke_step(models['smoke'])
                if on_plume_step is not None:
                    on_plume_step(time_ms, models)
        else:
            notify_plume_step = on_plume_step
        for index, case in enumerate(cases):
            current = datetime.fromisoformat(case.timestamp)
            end = datetime.fromisoformat(cases[index+1].timestamp) if index+1 < len(cases) else final
            workspace, parameterize = case.prepare()
            if fire is None:
                fire = self._initialize_fire(workspace, case.domain.spacing)
                if self.smoke is not None:
                    mesh = self._smoke_mesh(workspace, plume_diagnostics)
                    smoke = SmokePlume(self.smoke, mesh, track_particles=notify_plume_step is not None,
                                       native_collection_math=native_collection_math, specifications=self.plume_models,
                                       native_threads=self.native_threads, source_seed=self.source_seed,
                                       native_release_indexing=self.native_release_indexing,
                                       native_full_domain_bounds=self.native_full_domain_bounds)
            parameterize()
            solution = case.solve(workspace, initial_multiplier=previous, previous_turbulence=previous_turbulence)
            if reuse_multiplier:
                previous = solution['multiplier']
            previous_turbulence = case.nonlocal_turbulence_state(solution['turbulence'])
            initial = {name: workspace.arrays[name].copy() for name in ('u0', 'v0', 'w0')}
            if fire_progress_stream is not None:
                print(f'[QES-Fire]\t Fire simulation stating from {current.isoformat()} to {end.isoformat()}.',
                      file=fire_progress_stream, flush=True)
            while current < end:
                for name, values in initial.items():
                    workspace.arrays[name][:] = values
                if self.potential is not None:
                    fire.forcing(workspace, narrow=self.narrow_band)
                    self.potential.apply(fire, workspace)
                parameterize()
                solution = case.solve(workspace, include_gradients=include_gradients, initial_multiplier=previous,
                                      previous_turbulence=previous_turbulence)
                if reuse_multiplier:
                    previous = solution['multiplier']
                previous_turbulence = case.nonlocal_turbulence_state(solution['turbulence'])
                dt = fire.step(workspace, narrow=self.narrow_band)
                step_ms = milliseconds(dt)
                if step_ms <= 0:
                    raise FloatingPointError('Fire time step is below the native millisecond clock resolution')
                smoke_samples, smoke_accounting = (), None
                plume_samples, plume_accounting = (), None
                if smoke is not None:
                    begin_ms = round((current-start).total_seconds()*1000)
                    samples = tuple(smoke.advance_models(fire, workspace, solution['turbulence'], begin_ms,
                                                         begin_ms+step_ms, case.domain.spacing, notify_plume_step,
                                                         progress=progress))
                    smoke_samples = tuple(sample for tag, sample in samples if tag == 'smoke')
                    plume_samples = tuple((tag, sample) for tag, sample in samples if tag != 'smoke')
                    smoke_accounting = smoke.accounting()
                    plume_accounting = smoke.model_accounting()
                current += timedelta(milliseconds=step_ms)
                if fire_progress_stream is not None:
                    print(f'[QES-Fire]\t Fire step completed.\n[QES-Fire]\t Current time = {current.isoformat()}',
                          file=fire_progress_stream, flush=True)
                coordinates = {axis: workspace.coordinates[axis] for axis in ('x', 'y', 'z')}
                coordinates['z'] = coordinates['z'][1:]
                yield FireFrame(current, fire.time, dt, _fire_fields(fire, workspace, native_format=native_fire_format), coordinates,
                                solution['turbulence'], workspace.coordinates, smoke_samples, smoke_accounting,
                                plume_samples, plume_accounting)


def _fire_fields(fire, workspace, *, native_format=False):
    a = workspace.arrays
    fire_fields = {
        'terrain': fire.terrain,
        'fuel': fire.fuel_map,
        'burn': fire.burn_out.copy(),
        'Force': fire.force.copy(),
        'LS': fire.front.copy(),
    }
    if native_format:
        return fire_fields | cell_fields(workspace, ('u', 'v', 'w', 'icell'))
    u, v, w = (a[name] for name in ('u', 'v', 'w'))
    return fire_fields | {
        'u': .5*(u[1:-1, :-1, 1:]+u[1:-1, :-1, :-1]),
        'v': .5*(v[1:-1, 1:, :-1]+v[1:-1, :-1, :-1]),
        'w': .5*(w[2:, :-1, :-1]+w[1:-1, :-1, :-1]),
        'icell': a['icellflag'][1:].copy(),
    }


def read_coupled_fire(path, *, heat_potential=None, compute_turbulence=False, plume=None, seed=0, particle_csv=False,
                      poisson_backend='cpu', turbulence_gradient='cpu', heat_potential_backend='cpu',
                      front_backend='cpu', native_sensor_math=False, native_wake_math=False, native_turbulence_math=False,
                      native_building_math=False, native_serial=False, native_threads=None, source_seed=None,
                      wrf_timeout=None, native_flame_sampling=False, meteorology_backend='cpu',
                      native_release_indexing=False, legacy_turbulence_xml=False, legacy_plume_xml=False,
                      native_fire_clock=False, native_ignition_input=False, native_full_domain_bounds=False,
                      legacy_fire_xml=False, native_heat_potential_input=False, narrow_band=True):
    if native_serial and native_threads is not None:
        raise ValueError('Specify either native_serial or native_threads')
    workers = 1 if native_serial else native_threads
    if native_release_indexing and workers is None:
        raise ValueError('Original release indexing requires --native-serial or --native-threads')
    if workers is not None and workers < 1:
        raise ValueError('Native CPU replay needs at least one worker')
    if plume is None and (workers is not None or source_seed is not None):
        raise ValueError('Fire plume replay and source seed require --plume')
    if native_full_domain_bounds and plume is None:
        raise ValueError('Original full-domain bounds require a plume configuration')
    seed = resolve_cpu_seed(seed)
    # QES resolves inputs from the named XML's parent, even for a symlink.
    path = Path(path)
    root = read_qes_xml(path)
    section = root.find('fires')
    config = replace(_parse_fire_config(path, section, legacy_fire_xml=legacy_fire_xml),
                     native_ignition_input=native_ignition_input)
    root.remove(section)
    zero_wrf_steps = legacy_fire_xml and normalize_zero_wrf_startup(root)
    wind = _read_wind_root(path, root, poisson_backend=poisson_backend,
                           compute_turbulence=compute_turbulence or plume is not None,
                           turbulence_gradient=turbulence_gradient, native_sensor_math=native_sensor_math,
                           native_wake_math=native_wake_math, native_turbulence_math=native_turbulence_math,
                           native_building_math=native_building_math, wrf_startup_only=True, wrf_timeout=wrf_timeout,
                           meteorology_backend=meteorology_backend, legacy_turbulence_xml=legacy_turbulence_xml)
    # qesFire retains the initial face velocities after pressure projection.
    if wind.wall_log is not None:
        wind = replace(wind, wall_log=None)
    if heat_potential_backend == 'cuda' and heat_potential is not None:
        from .cuda_fire_potential import CudaHeatPotential
        potential = CudaHeatPotential.read(heat_potential, native_input=native_heat_potential_input)
    elif heat_potential_backend == 'cpu':
        potential = (HeatPotential.read(heat_potential, native_input=native_heat_potential_input)
                     if heat_potential is not None else None)
    else:
        raise ValueError('CUDA heat potential requires --heat-potential and a valid backend')
    plume_root = read_qes_xml(plume) if plume is not None else None
    smoke = parse_smoke_config(plume_root, seed=seed, legacy_plume_xml=legacy_plume_xml) if plume_root is not None else None
    if smoke is not None and wind.turbulence is None:
        raise ValueError('Smoke transport requires turbParams in the wind XML')
    if particle_csv and plume is None:
        raise ValueError('Fire particle CSV requires --plume')
    particle_output = read_particle_output(plume_root, legacy_plume_xml=legacy_plume_xml) if particle_csv else None
    plume_models = (read_models(plume_root, smoke.step_s, legacy_plume_xml=legacy_plume_xml)
                    if plume_root is not None and plume_root.find('particleParameters') is not None else None)
    front = {'cpu': cpu_front, 'cuda': cuda_front}.get(front_backend)
    if front is None:
        raise ValueError(f'Unknown fire-front backend {front_backend}')
    return CoupledFireCase(wind, config, potential, smoke, particle_output, front, plume_models, workers, source_seed,
                           native_flame_sampling, native_release_indexing, native_fire_clock,
                           native_full_domain_bounds=native_full_domain_bounds, narrow_band=narrow_band,
                           zero_wrf_steps=zero_wrf_steps)


def write_fire_run(case, output, *, native_collection_format=False, native_collection_math=False,
                   native_turbulence_format=False, native_fire_format=False, prefix='',
                   write_fire_output=True, write_turbulence=True, write_concentration=True, progress_stream=None,
                   native_particle_csv=False, fire_progress_stream=None, plume_diagnostics=None):
    output = Path(output)
    if (native_collection_format or native_collection_math) and case.smoke is None:
        raise ValueError('Native smoke collection output requires a plume configuration')
    if native_turbulence_format and case.wind.turbulence is None:
        raise ValueError('Native fire turbulence output requires turbulence computation and turbParams')
    count = 0
    tags = {'smoke', *(case.plume_models or ())} if case.smoke is not None else set()
    plume_counts = dict.fromkeys(sorted(tags), 0)
    origin = datetime.fromisoformat(case.wind.timestamp)
    progress = (PlumeProgress(origin, case.smoke.progress_period_ms, progress_stream)
                if case.smoke is not None and case.smoke.progress_period_ms is not None and progress_stream is not None
                else None)
    on_plume_step = None
    if case.particle_output is not None:
        particle_output = ParticleOutput(case.particle_output, output, origin, prefix=prefix,
                                         native_format=native_particle_csv)

        on_plume_step = particle_output.save

    first = frame = None
    with ExitStack() as files:
        plume_writers = {}
        if native_collection_format and write_concentration:
            for tag in plume_counts:
                plume_writers[tag] = files.enter_context(native_collection_series(
                    output/f'{prefix}{tag}_plumeOut.nc', case.smoke, origin))
        write_fire = write_turb = None
        for frame in case.frames(on_plume_step=on_plume_step, native_collection_math=native_collection_math,
                                 include_gradients=native_turbulence_format and write_turbulence,
                                 native_fire_format=native_fire_format, progress=progress,
                                 fire_progress_stream=fire_progress_stream, plume_diagnostics=plume_diagnostics):
            if count == 0:
                first = frame.time
                if write_fire_output:
                    path = output/f'{prefix}fireOutput.nc'
                    if native_fire_format:
                        series = native_fire_series(path, case.wind.domain, origin=first)
                    else:
                        series = netcdf_series(path, frame.coordinates, varying_fields=('burn', 'Force', 'LS'),
                                               dimension_names={axis: axis+'dim' for axis in ('x', 'y', 'z')},
                                               time_origin=first.isoformat())
                    write_fire = files.enter_context(series)
                if write_turbulence and frame.turbulence is not None:
                    if native_turbulence_format:
                        write_turb = files.enter_context(native_turbulence_series(
                            output/f'{prefix}turbOut.nc', case.wind.domain, origin=first,
                            allow_nan=case.wind.turbulence.native_zero_mixing_nan))
                    else:
                        write_turb = files.enter_context(netcdf_series(
                            output/f'{prefix}turbulence.nc', frame.wind_coordinates, time_origin=first.isoformat()))
            elapsed = (frame.time-first).total_seconds()
            stamp = frame.time.isoformat(timespec='seconds')
            if write_fire is not None:
                write_fire(frame.fields, stamp, elapsed)
            if write_turb is not None:
                write_turb(frame.turbulence, stamp, elapsed)
            samples = (*frame.plume_samples, *(('smoke', sample) for sample in frame.smoke_samples))
            for tag, sample in samples:
                if write_concentration:
                    if tag not in plume_writers:
                        plume_writers[tag] = files.enter_context(plume_collection_series(
                            output/f'{prefix}{tag}_plumeOut.nc', case.smoke, origin, native_math=native_collection_math))
                    elapsed_s = sample.time_ms/1000
                    stamp_smoke = (origin+timedelta(milliseconds=sample.time_ms)).isoformat(timespec='seconds')
                    plume_writers[tag](sample.fields, stamp_smoke, elapsed_s)
                plume_counts[tag] += 1
            count += 1
        if count == 0:
            if not native_fire_format:
                raise ValueError('Fire duration produces no output records')
            # The native driver declares these schemas before its fire loop.
            # FIREOutput and TURBOutput acquire their time origin on first save.
            if write_fire_output:
                files.enter_context(native_fire_series(output/f'{prefix}fireOutput.nc', case.wind.domain, origin=None))
            if write_turbulence and native_turbulence_format:
                files.enter_context(native_turbulence_series(output/f'{prefix}turbOut.nc', case.wind.domain, origin=None))
    result = dict(records=count, first_output=first.isoformat() if first is not None else None,
                  last_output=frame.time.isoformat() if frame is not None else None,
                  fire_time_s=frame.fire_time_s if frame is not None else 0., induced_winds=case.potential is not None)
    if not case.narrow_band:
        result['level_set'] = 'full'
    if case.smoke is not None:
        accounting = (frame.smoke_accounting if frame is not None
                      else dict(sources=0, released_particles=0, active_particles=0))
        result['smoke'] = dict(records=plume_counts['smoke'], seed=case.smoke.seed, **accounting)
        if case.native_threads is not None:
            result['smoke']['native_threads'] = case.native_threads
        if case.native_release_indexing:
            result['smoke']['native_release_indexing'] = True
        if case.source_seed is not None:
            result['smoke']['source_seed'] = case.source_seed
        if case.particle_output is not None:
            result['smoke']['particle_output_records'] = particle_output.records
        if frame is not None and frame.plume_accounting:
            result['plume_models'] = {tag: dict(records=plume_counts[tag], **accounting)
                                      for tag, accounting in frame.plume_accounting.items()}
    return result


def run_legacy_fire(wind_xml, plume_xml, *, basename=None, solver_type=1, fire_output=False, turbulence=False,
                    particle_csv=False, no_fire_winds=False, heat_potential=None, seed=0, native_threads=1,
                    source_seed=None, verbose=False, wrf_timeout=None, progress_stream=None,
                    native_release_indexing=None, native_particle_csv=True, native_full_domain_bounds=False,
                    cpu_serial=False):
    """Run the pinned fire driver with independent switches for each product."""
    backend = legacy_wind_backend(solver_type, cpu_serial=cpu_serial)
    cpu = solver_type == 1
    if no_fire_winds:
        heat_potential = None
    elif heat_potential is None:
        relative = Path('data/FireFiles/HeatPot.nc')
        package = Path(__file__).resolve().parent
        candidates = (Path('..')/relative, package.parents[1]/'external/QES-Public'/relative,
                      package/'fire_data'/relative.name)
        heat_potential = next((path for path in candidates if path.is_file()), None)
        if heat_potential is None:
            raise ValueError('Fire-induced winds require HeatPot.nc; specify --heat-potential')
    # Native appends the underscore before the filesystem interprets the path.
    prefix = Path(f'{basename}_') if basename else None
    compute_turbulence = turbulence or plume_xml is not None
    if native_release_indexing is None:
        native_release_indexing = plume_xml is not None
    case = read_coupled_fire(
        wind_xml, heat_potential=heat_potential, compute_turbulence=compute_turbulence, plume=plume_xml,
        seed=seed, particle_csv=prefix is not None and plume_xml is not None and particle_csv,
        poisson_backend=backend, native_sensor_math=cpu, native_building_math=cpu,
        native_turbulence_math=compute_turbulence, front_backend='cpu' if cpu else 'cuda',
        heat_potential_backend='cpu' if cpu or no_fire_winds else 'cuda',
        native_threads=native_threads if plume_xml is not None else None, source_seed=source_seed,
        native_release_indexing=native_release_indexing,
        native_full_domain_bounds=native_full_domain_bounds,
        wrf_timeout=wrf_timeout, native_flame_sampling=True, meteorology_backend='cpu' if cpu else 'cuda',
        legacy_turbulence_xml=True, legacy_plume_xml=True, native_fire_clock=True, native_ignition_input=True,
        legacy_fire_xml=True, native_heat_potential_input=True)
    case = replace(case, wind=replace(case.wind, native_staircase_metrics=True))
    if compute_turbulence and case.wind.turbulence is None:
        raise ValueError('Turbulence computation requires turbParams in the wind XML')
    if prefix is not None and case.plume_models and 'smoke' in case.plume_models:
        raise ValueError("The plume model tag 'smoke' is reserved for the qesFire smoke output")
    result = write_fire_run(
        case, prefix.parent if prefix is not None else Path('.'), prefix=prefix.name if prefix is not None else '',
        native_fire_format=True, native_turbulence_format=turbulence, native_collection_format=plume_xml is not None,
        native_collection_math=plume_xml is not None, write_fire_output=prefix is not None and fire_output,
        write_turbulence=prefix is not None and turbulence, write_concentration=prefix is not None,
        progress_stream=progress_stream, native_particle_csv=native_particle_csv,
        fire_progress_stream=progress_stream if verbose else None, plume_diagnostics=progress_stream)
    result.update(backend=backend, solvetype=solver_type, compute_turbulence=compute_turbulence)
    if verbose:
        result['configuration'] = case.wind.manifest()
        result['heat_potential'] = str(heat_potential) if heat_potential is not None else None
        result['native_flame_sampling'] = case.native_flame_sampling
    return result
