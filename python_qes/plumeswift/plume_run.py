"""Run the CPU QES wind-to-plume driver and export per-model concentrations."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta
from itertools import repeat
from pathlib import Path

import numpy as np

from .concentration_output import native_collection_series
from .contracts import UnportedFeature
from .cpu_plume_random import NativeCPUStreams, resolve_cpu_seed
from .io import netcdf_series
from .numerics.native_cpu_rb import NativeCPUSOR
from .particle_ids import ParticleIDs
from .particle_output import ParticleOutput, read_particle_output
from .plume_config import parse_plume_config
from .plume_fields import open_plume_fields
from .plume_progress import PlumeProgress, write_legacy_particle_summary
from .plume_sources import read_models
from .plume_transport import PlumeModel, plume_collection_series
from .qes_xml import read_qes_xml
from .runtime_info import report_warning
from .timing import milliseconds
from .turbulence_output import native_turbulence_series
from .wind_case import WindCase, _read_wind_root, read_wind_case
from .wind_run import legacy_wind_backend
from .wind_visualization import native_visualization_series
from .wind_workspace_output import native_workspace_fields, native_workspace_series
from .wind_xml import normalize_zero_wrf_startup


def _open_wind_products(wind, *, stack, output, products, origin):
    writers = {}
    if 'workspace' in products:
        writers['workspace'] = stack.enter_context(native_workspace_series(
            output/products['workspace'], wind.domain, origin=origin))
    if 'turbulence' in products:
        writers['turbulence'] = stack.enter_context(native_turbulence_series(
            output/products['turbulence'], wind.domain, origin=origin,
            allow_nan=wind.turbulence.native_zero_mixing_nan))
    if 'visualization' in products:
        writers['visualization'] = stack.enter_context(native_visualization_series(
            output/products['visualization'], wind.domain, wind.visualization_fields, origin=origin))
    return writers


def _empty_wind_outputs(wind, *, stack, output, products, terrain_output):
    workspace = wind.initialize_geometry(prepare_mixing=True)
    if terrain_output is not None:
        workspace.terrain_mesh.write_obj(terrain_output, indexed=True)
    _open_wind_products(wind, stack=stack, output=output, products=products, origin=None)


def _wind_fields(steps, *, stack, output, products, terrain_output=None):
    writers = {}
    previous = None
    previous_turbulence = None
    for index, step in enumerate(steps):
        result = step.run(include_gradients='turbulence' in products, initial_multiplier=previous,
                          previous_turbulence=previous_turbulence)
        if isinstance(step.poisson, NativeCPUSOR):
            previous = result['multiplier']
        previous_turbulence = step.nonlocal_turbulence_state(result['turbulence'])
        if index == 0:
            if terrain_output is not None:
                result['workspace'].terrain_mesh.write_obj(terrain_output, indexed=True)
            origin = datetime.fromisoformat(step.timestamp)
            writers = _open_wind_products(step, stack=stack, output=output, products=products, origin=origin)
        if writers:
            elapsed = (datetime.fromisoformat(step.timestamp)-origin).total_seconds()
            for name, write in writers.items():
                record = result['turbulence'] if name == 'turbulence' else result['workspace']
                if name == 'workspace':
                    record = native_workspace_fields(result)
                write(record, step.timestamp, elapsed)
        yield result['workspace'], result['turbulence'], step.domain.spacing


def run_plume(wind_path, plume_path, output, *, seed=0, turbulence_path=None, particle_csv=False,
              deposition=False, terrain_obj=None, poisson_backend='cpu', turbulence_gradient='cpu',
              native_serial=False, source_seed=None, native_threads=None, native_parallel=False,
              native_release_indexing=False, native_full_domain_bounds=False,
              native_collection_format=False, native_collection_math=False, native_workspace_format=False,
              native_turbulence_format=False, native_visualization_format=False, native_sensor_math=False,
              native_wake_math=False, native_turbulence_math=False, native_building_math=False,
              output_basename=None, allow_empty_models=False, write_concentration=True, terrain_output=None,
              wrf_timeout=None, progress_stream=None, native_timestamps=False, native_layout=False,
              native_arithmetic=False, legacy_plume_xml=False, native_particle_csv=False, _zero_wrf_steps=False):
    if native_serial and native_threads is not None:
        raise ValueError('Specify either native_serial or native_threads')
    workers = 1 if native_serial else native_threads
    if native_release_indexing and workers is None:
        raise ValueError('Original release indexing requires --native-serial or --native-threads')
    if native_parallel and (workers is None or workers < 2):
        raise ValueError('Parallel replay requires native_threads >= 2')
    seed = resolve_cpu_seed(seed)
    replay = None if workers is None else NativeCPUStreams(seed, workers)
    products = {name: filename for name, filename, selected in (
        ('workspace', 'windsWk.nc', native_workspace_format), ('turbulence', 'turbOut.nc', native_turbulence_format),
        ('visualization', 'windsOut.nc', native_visualization_format)) if selected}
    if turbulence_path is not None and products:
        raise ValueError('Writing coupled wind field outputs requires a wind XML input')
    if turbulence_path is not None and terrain_output is not None:
        raise ValueError('Writing a terrain mesh requires a wind XML input')
    if wrf_timeout is not None and (turbulence_path is not None or isinstance(wind_path, WindCase)):
        raise ValueError('Setting a WRF timeout requires a wind XML input')
    if turbulence_path is not None and poisson_backend != 'cpu':
        raise ValueError('Selecting a pressure solver requires a wind XML input')
    if turbulence_path is not None and turbulence_gradient != 'cpu':
        raise ValueError('Selecting a turbulence gradient backend requires a wind XML input')
    if turbulence_path is not None and native_sensor_math:
        raise ValueError('Selecting sensor arithmetic requires a wind XML input')
    if turbulence_path is not None and native_wake_math:
        raise ValueError('Selecting wake arithmetic requires a wind XML input')
    if turbulence_path is not None and native_building_math:
        raise ValueError('Selecting building arithmetic requires a wind XML input')
    if turbulence_path is not None and native_turbulence_math:
        raise ValueError('Selecting turbulence arithmetic requires a wind XML input')
    root = read_qes_xml(plume_path)
    config = parse_plume_config(root, seed=seed, legacy_plume_xml=legacy_plume_xml)
    if not legacy_plume_xml:
        allowed = {'plumeParameters', 'collectionParameters', 'boundaryConditions',
                   'sourceParameters', 'particleParameters', 'particleOutputParameters', 'simulationParameters', 'sources'}
        extra = {node.tag for node in root}-allowed
        if extra:
            raise UnportedFeature(f'Unsupported plume XML sections {sorted(extra)}')
    # Native can replace duplicate model keys only when there are no output
    # handles competing for the same basename and model tag.
    model_outputs = write_concentration or particle_csv or native_particle_csv or deposition
    specifications = read_models(root, config.step_s, legacy_plume_xml=legacy_plume_xml,
                                 allow_repeated_tags=not model_outputs)
    if not specifications and not allow_empty_models:
        raise ValueError('Plume run requires particleParameters with at least one baseParticle')
    deposition_tags = {tag for tag, spec in specifications.items() if deposition and spec.deposition}
    if deposition and not deposition_tags:
        raise ValueError('Deposition requires depositionFlag=true on at least one baseParticle model')
    for tag in deposition_tags:
        for source in specifications[tag].sources:
            if min(source.diameter_um, source.density_kg_m3) < 0:
                raise ValueError(f'Negative particle diameter or density for model {tag!r}')
    particle_csv = particle_csv or native_particle_csv
    particle_config = None
    if particle_csv or (legacy_plume_xml and root.find('particleOutputParameters') is not None):
        particle_config = read_particle_output(root, legacy_plume_xml=legacy_plume_xml)
    allocate_ids = ParticleIDs(native=workers is not None).take
    output = Path(output)
    prefix = ''
    if output_basename is not None:
        basename = Path(output_basename)
        literal_legacy_prefix = legacy_plume_xml and output_basename in ('', '.', '..')
        if not literal_legacy_prefix and (basename.parent != Path('.') or basename.name == '..' or not basename.name):
            raise ValueError('Output basename must be a single nonempty path component')
        prefix = (output_basename if literal_legacy_prefix else basename.name)+'_'
        products = {key: prefix+name for key, name in products.items()}

    with ExitStack() as stack:
        executor = (stack.enter_context(ThreadPoolExecutor(max_workers=workers, thread_name_prefix='qes-plume'))
                    if native_parallel else None)
        mesh = None
        if terrain_obj is not None:
            from .triangle_mesh import TriangleMesh
            if config.wall != 'meshReflection':
                raise ValueError('Terrain OBJ requires meshReflection in the plume XML')
            mesh = TriangleMesh.read_obj(terrain_obj)
        if turbulence_path is None:
            wind = wind_path if isinstance(wind_path, WindCase) else read_wind_case(
                wind_path, poisson_backend=poisson_backend, turbulence_gradient=turbulence_gradient,
                native_sensor_math=native_sensor_math, native_wake_math=native_wake_math,
                native_turbulence_math=native_turbulence_math, native_building_math=native_building_math,
                wrf_startup_only=True, wrf_timeout=wrf_timeout)
            # The post-projection wall correction belongs to qesWinds, not qes.
            if wind.wall_log is not None:
                wind = replace(wind, wall_log=None)
            if wind.turbulence is None:
                raise ValueError('Plume transport requires turbParams in the wind XML')
            if _zero_wrf_steps:
                timestamps = (datetime.fromisoformat(wind.timestamp),)
                _empty_wind_outputs(wind, stack=stack, output=output, products=products, terrain_output=terrain_output)
                fields = ()
            else:
                steps = tuple(wind.iter_steps())
                timestamps = tuple(datetime.fromisoformat(step.timestamp) for step in steps)
                fields = _wind_fields(steps, stack=stack, output=output, products=products, terrain_output=terrain_output)
        else:
            timestamps, fields = stack.enter_context(open_plume_fields(
                wind_path, turbulence_path, native_timestamps=native_timestamps, native_layout=native_layout,
                native_arithmetic=native_arithmetic or replay is not None,
                allow_nonfinite_turbulence=replay is not None and not deposition,
                diagnostics=progress_stream if legacy_plume_xml else None))
        if (legacy_plume_xml and progress_stream is not None and config.wall == 'meshReflection' and mesh is None
                and (turbulence_path is not None or wind.terrain is None)):
            print('[WARRNING] mesh wall reflection method not available', file=progress_stream, flush=True)
        native_file_clock = legacy_plume_xml and turbulence_path is not None
        if not timestamps:
            raise ValueError('Plume requires at least one wind timestamp')
        if not native_file_clock and any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError('Plume wind timestamps must be nondecreasing')
        origin = timestamps[0]
        progress = (PlumeProgress(origin, config.progress_period_ms, progress_stream)
                    if config.progress_period_ms is not None and progress_stream is not None else None)
        ticks = [(timestamp-origin) // timedelta(microseconds=1) for timestamp in timestamps]
        if any(tick % 1000 for tick in ticks):
            raise ValueError('Plume wind intervals must align to the millisecond transport clock')
        offsets = [tick//1000 for tick in ticks]
        last_interval = offsets[-1]-offsets[-2] if len(offsets) > 1 else milliseconds(config.duration_s)
        if legacy_plume_xml and len(offsets) > 1:
            # Both native drivers repeat the final observed interval as float32 seconds.
            last_interval = milliseconds(last_interval/1000)
        ends = [] if _zero_wrf_steps else offsets[1:]+[offsets[-1]+last_interval]
        particle_output = (ParticleOutput(particle_config, output, origin, prefix=prefix, native_format=native_particle_csv)
                           if particle_csv else None)
        if replay is None:
            rngs = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(len(specifications)))
        else:
            rngs = repeat(replay, len(specifications))
            if source_seed is None:
                source_seed = seed
        models = {}
        for tag, rng in zip(sorted(specifications), rngs, strict=True):
            models[tag] = PlumeModel(config, mesh, rng=rng, track_particles=particle_csv, allocate_ids=allocate_ids,
                                     deposition=tag in deposition_tags, executor=executor,
                                     native_collection_math=native_collection_math,
                                     native_arithmetic=native_arithmetic,
                                     native_release_indexing=native_release_indexing)
        clock = next(iter(models.values()), None)
        records = dict.fromkeys(models, 0)
        writers = {}
        if native_collection_format and write_concentration:
            writers = {tag: stack.enter_context(native_collection_series(output/f'{prefix}{tag}_plumeOut.nc', config, origin))
                       for tag in models}
        time_ms = 0
        for index, ((workspace, turbulence, spacing), end_ms) in enumerate(zip(fields, ends, strict=True)):
            for tag, model in models.items():
                if index == 0 and mesh is None and config.wall == 'meshReflection':
                    model.mesh = workspace.terrain_mesh
                model.set_fields(workspace, turbulence, spacing)
                if index == 0:
                    model.simulation.sources = tuple(spec.create(model.simulation.integrator.boundaries,
                                                                 source_seed=source_seed,
                                                                 native_full_domain_bounds=native_full_domain_bounds)
                                                     for spec in specifications[tag].sources)
            if progress is not None:
                progress.begin_interval(time_ms, end_ms, models)
            elif clock is None and not legacy_plume_xml:
                time_ms = max(time_ms, end_ms)
            while time_ms < end_ms:
                for tag, model in models.items():
                    sample = model.advance_step(end_ms)
                    if sample is None:
                        continue
                    if write_concentration:
                        if tag not in writers:
                            writers[tag] = stack.enter_context(plume_collection_series(
                                output/f'{prefix}{tag}_plumeOut.nc', config, origin, native_math=native_collection_math))
                        stamp = origin+timedelta(milliseconds=sample.time_ms)
                        writers[tag](sample.fields, stamp.isoformat(timespec='seconds'), sample.time_ms/1000)
                    records[tag] += 1
                if clock is not None:
                    time_ms = clock.time_ms
                else:
                    time_ms += milliseconds(min(config.step_s, (end_ms-time_ms)/1000))
                if particle_output is not None and clock is not None:
                    particle_output.save(time_ms, models)
                if progress is not None:
                    progress.step(time_ms, end_ms, models)
            if progress is not None:
                progress.end_interval(time_ms, models)
            if native_file_clock and progress_stream is not None:
                print('[QES-Plume] \t Finished.', file=progress_stream, flush=True)
        final_ms = time_ms if legacy_plume_xml or _zero_wrf_steps else ends[-1]
        elapsed = final_ms/1000
        stamp = (origin+timedelta(milliseconds=final_ms)).isoformat(timespec='seconds')
        if deposition_tags:
            coordinates = {axis: workspace.coordinates[axis] for axis in 'xyz'}
            for tag in sorted(deposition_tags):
                field = models[tag].simulation.integrator.deposition
                with netcdf_series(output/f'{prefix}{tag}_depositionOut.nc', coordinates, time_origin=origin.isoformat(),
                                   field_units={'depcvol': 'g'}) as write:
                    write({'depcvol': 1000*field.deposited_kg}, stamp, elapsed)
    empty = dict(released_mass_kg=0., active_mass_kg=0., escaped_mass_kg=0., mass_balance_error_kg=0.,
                 released_particles=0, active_particles=0, escaped_particles=0, pending_source_mass_kg=0.,
                 inactive_on_release_particles=0, inactive_on_release_mass_kg=0., rogue_particles=0, rogue_mass_kg=0.)
    result = dict(time_s=elapsed, wind_records=0 if _zero_wrf_steps else len(timestamps), seed=seed, models={
        tag: dict(records=records[tag], sources=len(specifications[tag].sources),
                  **(empty if _zero_wrf_steps else model.simulation.accounting()))
        for tag, model in models.items()})
    if particle_output is not None:
        result['particle_output_records'] = particle_output.records
    if replay is not None:
        result['transport_order'] = 'linux-qes-single-thread' if workers == 1 else 'linux-qes-openmp-static'
        result['native_threads'] = len(replay.streams)
    elif native_arithmetic:
        result['transport_order'] = 'vectorized-native-arithmetic'
    if native_parallel:
        result['native_parallel'] = True
    if native_release_indexing:
        result['native_release_indexing'] = True
    if source_seed is not None:
        result['source_seed'] = source_seed
    if products:
        result['field_outputs'] = products
    if legacy_plume_xml and progress_stream is not None:
        write_legacy_particle_summary(progress_stream, origin, final_ms, result['models'])
    return result


def run_legacy_qes(wind_xml, plume_xml, *, basename=None, solver_type=1, workspace=False, turbulence=False,
                   terrain=False, particle_csv=False, seed=0, native_threads=1, source_seed=None, verbose=False,
                   wrf_timeout=None, progress_stream=None, native_release_indexing=False, native_particle_csv=False,
                   native_full_domain_bounds=False, cpu_serial=False):
    """Apply the original coupled driver's solver and product-selection rules."""
    if native_release_indexing and plume_xml is None:
        raise ValueError('Original release indexing requires a plume configuration')
    if native_full_domain_bounds and plume_xml is None:
        raise ValueError('Original full-domain bounds require a plume configuration')
    if not basename and progress_stream is not None:
        report_warning('No output basename set -> output turned off', progress_stream)
    backend = legacy_wind_backend(solver_type, cpu_serial=cpu_serial)
    cpu = solver_type == 1
    compute_turbulence = turbulence or plume_xml is not None
    root = read_qes_xml(wind_xml)
    zero_wrf_steps = normalize_zero_wrf_startup(root)
    wind = _read_wind_root(wind_xml, root, poisson_backend=backend, compute_turbulence=compute_turbulence,
                           native_sensor_math=cpu, native_building_math=cpu, native_turbulence_math=compute_turbulence,
                           wrf_startup_only=True, wrf_timeout=wrf_timeout,
                           meteorology_backend='cpu' if cpu else 'cuda', legacy_turbulence_xml=True)
    wind = replace(wind, wall_log=None, native_staircase_metrics=True)
    if compute_turbulence and wind.turbulence is None:
        raise ValueError('Turbulence computation requires turbParams in the wind XML')
    prefix = Path(f'{basename}_') if basename else None
    basename_component = prefix.name.removesuffix('_') if prefix is not None else None
    terrain_output = None
    if prefix is not None and terrain:
        if wind.terrain is None:
            raise ValueError('Terrain output requires a terrain input')
        terrain_output = Path(str(prefix)+'terrainOut.obj')
    output = prefix.parent if prefix is not None else Path('.')
    workspace = prefix is not None and workspace
    turbulence = prefix is not None and turbulence
    if plume_xml is not None:
        result = run_plume(wind, plume_xml, output, seed=seed, native_threads=native_threads, source_seed=source_seed,
                           native_release_indexing=native_release_indexing,
                           native_full_domain_bounds=native_full_domain_bounds,
                           particle_csv=prefix is not None and particle_csv, allow_empty_models=True,
                           native_particle_csv=prefix is not None and native_particle_csv,
                           native_collection_format=True, native_collection_math=True,
                           native_workspace_format=workspace, native_turbulence_format=turbulence,
                           native_visualization_format=prefix is not None, terrain_output=terrain_output,
                           output_basename=basename_component,
                           write_concentration=prefix is not None, progress_stream=progress_stream, legacy_plume_xml=True,
                           _zero_wrf_steps=zero_wrf_steps)
    else:
        products = {key: prefix.name+suffix for key, suffix, selected in (
            ('workspace', 'windsWk.nc', workspace), ('turbulence', 'turbOut.nc', turbulence),
            ('visualization', 'windsOut.nc', prefix is not None),
        ) if selected}
        with ExitStack() as stack:
            if zero_wrf_steps:
                _empty_wind_outputs(wind, stack=stack, output=output, products=products, terrain_output=terrain_output)
                wind_records = 0
            else:
                fields = _wind_fields(wind.iter_steps(), stack=stack, output=output, products=products,
                                      terrain_output=terrain_output)
                wind_records = sum(1 for _ in fields)
            result = dict(wind_records=wind_records, models={}, field_outputs=products)
    result.update(backend=backend, solvetype=solver_type, compute_turbulence=compute_turbulence)
    if terrain_output is not None:
        result['terrain_output'] = str(terrain_output)
    if verbose:
        result['configuration'] = wind.manifest()
    return result
