"""NetCDF layout of the original QES WINDSOutputWorkspace (GPL-3.0)."""

from contextlib import contextmanager
from datetime import datetime

import numpy as np

from .output_v2 import OutputField, open_output


_CELL = ('z', 'y', 'x')
_FACE = ('z_face', 'y_face', 'x_face')
_SURFACE = ('y', 'x')


def _schema():
    fields = {
        'building_volume_frac': OutputField(('t', *_CELL), 'f4', '--', 'building volume fraction'),
        'dz_array': OutputField(('z',), 'f4', 'm', 'dz size of the cells'),
        'icellflag': OutputField(('t', *_CELL), 'i4', '--', 'icell flag value'),
        'mixlength': OutputField(_CELL, 'f8', 'm', 'distance to nearest object'),
        'terrain': OutputField(_SURFACE, 'f4', 'm', 'terrain height'),
        'terrain_volume_frac': OutputField(('t', *_CELL), 'f4', '--', 'terrain volume fraction'),
    }
    for name in 'efghmn':
        fields[name] = OutputField(('t', *_CELL), 'f4', '--', f'{name} cut-cell coefficient')
    for axis, name in zip('xyz', 'uvw', strict=True):
        fields[name] = OutputField(('t', *_FACE), 'f4', 'm s-1', f'{axis}-component velocity')
        fields[name+'0'] = OutputField(('t', *_FACE), 'f4', 'm s-1', f'{axis}-component initial velocity')
    for name in 'uv':
        fields[f'z0_{name}'] = OutputField(_SURFACE, 'f4', 'm', f'terrain aerodynamic roughness, {name}')
    return dict(sorted(fields.items()))


def native_workspace_coordinates(domain):
    nx, ny, _ = domain.cells
    dx, dy = (np.float32(value) for value in domain.spacing[:2])
    x = (np.arange(nx, dtype=np.float32)+.5)*dx
    y = (np.arange(ny, dtype=np.float32)+.5)*dy
    dz = domain.dz.astype(np.float32)
    z_face = np.r_[-dz[0], np.float32(0), np.cumsum(dz[1:], dtype=np.float32)]
    return dict(x_face=np.r_[x[0]-dx, x], y_face=np.r_[y[0]-dy, y+dy], z_face=z_face,
                x=x, y=y, z=.5*(z_face[:-1]+z_face[1:]))


def native_workspace_fields(result):
    fields = result['workspace'].arrays
    if result['turbulence'] is not None and 'mixlength' not in fields:
        return fields | {'mixlength': result['turbulence']['mixingLengths']}
    return fields


@contextmanager
def native_workspace_series(path, domain, *, origin):
    """Write the native workspace schema and retain repeated record timestamps."""
    fields = {}
    schema = _schema()
    with open_output(path, start=origin) as output:
        # The original declares this dimension without publishing a variable.
        output.define_dimension('UTMx', 1)
        for name, values in native_workspace_coordinates(domain).items():
            long_name = name.replace('_', '-') if name.endswith('_face') else f'{name}-distance'
            output.define_axis(name, values, units='m', long_name=long_name)
        output.attach('workspace', schema, lambda: fields)
        ground_size = np.prod(domain.cell_shape[1:])
        defaults = {}
        default_values = {'building_volume_frac': 1., 'terrain_volume_frac': 1., 'mixlength': 0.}

        def append(arrays, timestamp, elapsed):
            nonlocal fields
            time = datetime.fromisoformat(timestamp)
            if (time-origin).total_seconds() != elapsed:
                raise ValueError('Workspace timestamp does not match elapsed time')
            for name, value in default_values.items():
                if name not in arrays and name not in defaults:
                    defaults[name] = np.full(domain.cell_shape, value, dtype=schema[name].dtype)
            fields = {name: arrays[name] if name in arrays else defaults[name] for name in schema}
            for name in ('z0_u', 'z0_v'):
                # Native writes a cell-sized prefix of its flattened face array.
                fields[name] = np.asarray(fields[name]).ravel()[:ground_size]
            if any(not np.isfinite(value).all() for value in fields.values()):
                raise ValueError('Native workspace fields must be finite')
            output.new_time_entry(time, allow_duplicate=True)
            output.push('workspace', time)

        yield append
