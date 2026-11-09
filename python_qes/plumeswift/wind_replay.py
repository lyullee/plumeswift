"""Reproject stored QES wind fields with their original cut-cell stencil."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cutcell_walls import define_cutcell_walls
from .io import WindWorkspace, _open_netcdf, netcdf_dates, netcdf_series, workspace_field_timed, workspace_record
from .numerics.poisson import KrylovPoisson
from .wind_projection import project
from .wind_wall_log import WallLogCorrection


_WALL_GEOMETRY = ('ni', 'nj', 'nk', 'wall_distance', 'center_id')


def replay_wind(input_path, output_path, *, solver=None, wall_roughness=None, native_input=False):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('Wind replay output must differ from the input workspace')
    if solver is None:
        solver = KrylovPoisson()
    wall_log = None if wall_roughness is None else WallLogCorrection(wall_roughness)
    with ExitStack() as stack:
        source = stack.enter_context(_open_netcdf(input_path, raw=native_input))
        dates = netcdf_dates(source, native_timestamps=native_input)
        if not dates or len(dates) != len(source.dimensions['t']):
            raise ValueError('Wind replay requires timestamps for all workspace records')
        origin = dates[0]
        optional = ('terrain', 'mixlength', 'z0_u', 'z0_v', 'building_volume_frac', 'terrain_volume_frac',
                    'icellflag_initial', *_WALL_GEOMETRY)
        stencil = tuple('efghmn')
        keep = ('u0', 'v0', 'w0', *stencil, 'dz_array', 'icellflag', *optional)
        rebuild_stencil = native_input and 'e' not in source.variables
        read_fields = tuple(key for key in keep if key not in stencil) if rebuild_stencil else keep
        positional = {}
        face_axes = ('z_face', 'y_face', 'x_face')
        if all(axis in source.dimensions for axis in face_axes):
            face_shape = tuple(len(source.dimensions[axis]) for axis in face_axes)
            cell_shape = tuple(size-1 for size in face_shape)
            positional = dict.fromkeys(('u0', 'v0', 'w0'), face_shape)
            positional.update(dict.fromkeys(
                (*'efghmn', 'icellflag', 'icellflag_initial', 'mixlength',
                 'building_volume_frac', 'terrain_volume_frac', *_WALL_GEOMETRY), cell_shape))
        varying = tuple(key for key in read_fields if key in source.variables
                        and workspace_field_timed(source[key], positional.get(key)))
        static = tuple(key for key in read_fields if key in source.variables and key not in varying)
        fallback = None
        results = []
        for step, date in enumerate(dates):
            workspace = workspace_record(source, step, fields=read_fields, positional_fields=positional,
                                         native_input=native_input)
            arrays = workspace.arrays
            if rebuild_stencil:
                if fallback is None:
                    fallback = {key: np.ones(arrays['icellflag'].shape, dtype=np.float32) for key in stencil}
                # The original loader keeps the modified coefficients between records.
                define_cutcell_walls(WindWorkspace(fallback | {'icellflag': arrays['icellflag']}, workspace.coordinates))
                arrays.update(fallback)
            if wall_log is not None:
                flags = arrays.get('icellflag_initial', arrays['icellflag'])
                if flags.shape != arrays['icellflag'].shape or not np.isfinite(flags).all():
                    raise ValueError('Wind replay requires valid cell flags for wall correction')
                if np.isin(flags, (7, 8)).any():
                    invalid = [name for name in _WALL_GEOMETRY if name not in arrays or
                               arrays[name].shape != flags.shape or not np.isfinite(arrays[name]).all()]
                    if invalid:
                        raise ValueError(f'Cut-cell wall replay requires valid saved geometry: {", ".join(invalid)}')
                arrays['icellflag_initial'] = flags
            velocity, _, report = project(workspace, solver)
            if wall_log is not None:
                velocity = wall_log(workspace, velocity)
            if step == 0:
                write = stack.enter_context(netcdf_series(
                    output_path, workspace.coordinates, time_origin=origin.isoformat(),
                    varying_fields=varying, static_fields=static))
            fields = {name: arrays[name] for name in keep if name in arrays}
            elapsed = (date-origin).total_seconds()
            write(fields | velocity, date.isoformat(), elapsed)
            results.append(dict(timestamp=date.isoformat(), solver=asdict(report)))
    result = dict(records=len(results), steps=results)
    if wall_log is not None:
        result['wall_roughness_m'] = wall_log.roughness
    return result
