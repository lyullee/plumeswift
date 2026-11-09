"""Write selected standalone wind products from the shared wind driver."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np

from .io import netcdf_series
from .turbulence_output import native_turbulence_series
from .wind_visualization import native_visualization_series, validate_output_fields, visualization_series
from .wind_workspace_output import native_workspace_fields, native_workspace_series


LEGACY_WIND_BACKENDS = {1: 'cpu-native-rb', 2: 'cuda-global-memory',
                        3: 'cuda-global-memory', 4: 'cuda-shared-memory'}


def legacy_wind_backend(solver_type, *, cpu_serial=False):
    if cpu_serial and solver_type != 1:
        raise ValueError('--cpu-serial requires --solvetype 1')
    return 'cpu-native-serial' if cpu_serial else LEGACY_WIND_BACKENDS[solver_type]


def inspect_wind_input(path, *, output=None):
    """Run the constructor stage exercised by the historical UIDugdTest utility."""
    from .wind_case import read_wind_case

    case = read_wind_case(path, compute_turbulence=False, poisson_backend='cpu-native-rb',
                          native_sensor_math=True, native_building_math=True, wrf_startup_only=True,
                          legacy_turbulence_xml=True)
    case = replace(case, native_staircase_metrics=True)
    workspace = case.initialize_geometry()
    flags, counts = np.unique(workspace.arrays['icellflag'], return_counts=True)
    if output is not None:
        stamp = datetime.fromisoformat(case.timestamp)
        arrays = workspace.arrays | {axis: workspace.arrays[axis+'0'] for axis in 'uvw'}
        with native_workspace_series(output, case.domain, origin=stamp) as append:
            append(arrays, case.timestamp, 0.)
    return dict(domain_cells=case.domain.cells, cell_flags={str(flag): int(count) for flag, count in zip(flags, counts)})


def write_wind_run(case, *, workspace_path=None, turbulence_path=None, visualization_path=None, terrain_obj=None,
                   native_workspace_format=False, native_turbulence_format=False, native_visualization_format=False,
                   initialize_only=False):
    if visualization_path is not None and native_visualization_format:
        validate_output_fields(case.visualization_fields)
    summaries = []
    with ExitStack() as output:
        if case.wrf_zero_steps:
            # Native output constructors declare their schemas even when the
            # solver loop has no records and never establishes a time origin.
            if workspace_path is not None:
                output.enter_context(native_workspace_series(workspace_path, case.domain, origin=None))
            if visualization_path is not None:
                output.enter_context(native_visualization_series(visualization_path, case.domain,
                                                                  case.visualization_fields, origin=None))
            if turbulence_path is not None and case.turbulence is not None:
                output.enter_context(native_turbulence_series(turbulence_path, case.domain, origin=None))
        write_wind = write_turbulence = write_visualization = None
        for step, result in case.iter_results(initialize_only=initialize_only,
                                              include_gradients=turbulence_path is not None and native_turbulence_format):
            workspace = result['workspace']
            timestamp = datetime.fromisoformat(step.timestamp)
            if not summaries:
                origin = timestamp
                if terrain_obj is not None:
                    workspace.terrain_mesh.write_obj(terrain_obj, indexed=True)
                if workspace_path is not None:
                    if native_workspace_format:
                        series = native_workspace_series(workspace_path, step.domain, origin=origin)
                    else:
                        varying = (('canopy_roughness', 'canopy_ustar', 'canopy_displacement')
                                   if case.schedule or case.wrf_exchange else ())
                        series = netcdf_series(workspace_path, workspace.coordinates, varying_fields=varying,
                                               time_origin=origin.isoformat())
                    write_wind = output.enter_context(series)
                if visualization_path is not None:
                    if native_visualization_format:
                        series = native_visualization_series(visualization_path, step.domain,
                                                             case.visualization_fields, origin=origin)
                    else:
                        series = visualization_series(visualization_path, workspace, case.visualization_fields,
                                                      time_origin=step.timestamp)
                    write_visualization = output.enter_context(series)
                if turbulence_path is not None and result['turbulence'] is not None:
                    if native_turbulence_format:
                        series = native_turbulence_series(turbulence_path, step.domain, origin=origin,
                                                          allow_nan=step.turbulence.native_zero_mixing_nan)
                    else:
                        series = netcdf_series(turbulence_path, workspace.coordinates, time_origin=origin.isoformat())
                    write_turbulence = output.enter_context(series)
            elapsed = (timestamp-origin).total_seconds()
            if write_wind is not None:
                write_wind(native_workspace_fields(result), step.timestamp, elapsed)
            if write_visualization is not None:
                write_visualization(workspace, step.timestamp, elapsed)
            if write_turbulence is not None and result['turbulence'] is not None:
                write_turbulence(result['turbulence'], step.timestamp, elapsed)
            summary = dict(timestamp=step.timestamp, elapsed_s=elapsed,
                           solver=asdict(result['solver']) if result['solver'] is not None else None,
                           turbulence=result['turbulence_metadata'])
            if 'wrf_frame' in result:
                summary['wrf_frame'] = result['wrf_frame']
            summaries.append(summary)
        if case.wrf_zero_steps and terrain_obj is not None:
            case.terrain.mesh(case.domain).write_obj(terrain_obj, indexed=True)
    return summaries
