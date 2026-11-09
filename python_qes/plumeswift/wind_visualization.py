"""Cell-centred visualization NetCDF with the public QES field schema."""
from contextlib import contextmanager
from datetime import timedelta

import numpy as np

from .io import netcdf_series
from .output_v2 import OutputField, open_output
from .wind_workspace_output import native_workspace_coordinates


FIELDS = ('icell', 'icellInitial', 'mag', 'terrain', 'u', 'v', 'w')
_LONG_NAMES = dict(
    x='x-distance', y='y-distance', z='z-distance', t='time since start of simulation',
    timestamp='date time using format: YYYY-MM-DDThh:mm:ss', terrain='terrain height',
    u='x-component velocity', v='y-component velocity', w='z-component velocity',
    mag='velocity magnitude', icell='icell flag value', icellInitial='icell flag value',
)
_UNITS = dict(terrain='m', icell='--', icellInitial='--', timestamp='--',
              **dict.fromkeys(('u', 'v', 'w', 'mag'), 'm s-1'))


def validate_output_fields(fields):
    unknown = sorted(set(fields)-set(FIELDS))
    if unknown:
        raise ValueError(f'Unknown visualization output fields {unknown}')


def read_output_fields(options, *, validate_names=True):
    if options is None:
        return FIELDS
    int(options.findtext('outputFlag', '0'))
    for flag in ('massConservedFlag', 'sensorVelocityFlag', 'staggerdVelocityFlag'):
        if options.findtext(flag, 'false').strip().lower() not in ('0', '1', 'false', 'true'):
            raise ValueError(f'Invalid fileOptions/{flag} boolean')
    fields = [(node.text or '').strip() for node in options.findall('outputFields')]
    if not fields or fields[0] == 'all':
        return FIELDS
    if validate_names:
        validate_output_fields(fields)
    return tuple(sorted(set(fields)))


def cell_fields(workspace, fields):
    """Average neighbouring faces at each cell centre above the ground ghost layer."""
    arrays = workspace.arrays
    result = {}
    nz, ny, nx = arrays['icellflag'].shape
    lower = np.s_[1:nz, :ny, :nx]
    upper = {'u': np.s_[1:nz, :ny, 1:nx+1], 'v': np.s_[1:nz, 1:ny+1, :nx], 'w': np.s_[2:nz+1, :ny, :nx]}
    for name, face in upper.items():
        if name in fields or 'mag' in fields:
            velocity = arrays[name]
            result[name] = np.float32(.5)*(velocity[lower].astype(np.float32)+velocity[face].astype(np.float32))
    if 'mag' in fields:
        u, v, w = (result[name] for name in 'uvw')
        result['mag'] = np.sqrt(u*u+v*v+w*w)
    if 'icell' in fields:
        result['icell'] = arrays['icellflag'][1:]
    if 'icellInitial' in fields:
        result['icellInitial'] = arrays['icellflag_initial'][1:]
    if 'terrain' in fields:
        result['terrain'] = arrays['terrain']
    return {name: result[name] for name in fields}


@contextmanager
def visualization_series(path, workspace, fields, *, time_origin):
    coordinates = workspace.coordinates
    dx = np.float32(coordinates['x_face'][1]-coordinates['x_face'][0])
    # The upstream visualization accessor returns dx for both axes.
    xy = {name: (np.arange(len(coordinates[name]))+.5)*dx for name in ('x', 'y')}
    axes = dict(xy, z=coordinates['z'][1:])
    terrain = workspace.arrays['terrain'].astype(np.float32) if 'terrain' in fields else None
    with netcdf_series(path, axes, time_origin=time_origin, field_units=_UNITS, float_dtype='f4',
                       long_names=_LONG_NAMES) as append:
        def write(record, timestamp, elapsed):
            data = cell_fields(record, fields)
            if terrain is not None:
                data['terrain'] = terrain
            append(data, timestamp, elapsed)

        yield write


@contextmanager
def native_visualization_series(path, domain, fields, *, origin):
    """Write the original cell-centred visualization schema."""
    validate_output_fields(fields)
    coordinates = native_workspace_coordinates(domain)
    dx = np.float32(domain.spacing[0])
    # The native visualization accessor supplies dx for both axes.
    y = ((np.arange(domain.cells[1])+.5)*float(dx)).astype(np.float32)
    coordinates = dict(x=coordinates['x'], y=y, z=coordinates['z'][1:])
    values = {}
    schema = {
        name: OutputField(('y', 'x') if name == 'terrain' else ('t', 'z', 'y', 'x'),
                          'i4' if name in ('icell', 'icellInitial') else 'f4',
                          _UNITS[name], _LONG_NAMES[name])
        for name in sorted(fields)
    }
    with open_output(path, start=origin) as output:
        for axis, coordinate in coordinates.items():
            output.define_axis(axis, coordinate, units='m', long_name=_LONG_NAMES[axis])
        output.attach('visualization', schema, lambda: values)

        def append(workspace, timestamp, elapsed):
            nonlocal values
            time = origin+timedelta(seconds=elapsed)
            if timestamp not in (time.isoformat(), time.isoformat(timespec='seconds')):
                raise ValueError('Visualization timestamp does not match elapsed time')
            values = cell_fields(workspace, fields)
            if any(not np.isfinite(value).all() for value in values.values()):
                raise ValueError('Native visualization fields must be finite')
            output.new_time_entry(time, allow_duplicate=True)
            output.push('visualization', time)

        yield append
