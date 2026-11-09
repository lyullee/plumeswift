"""NetCDF layout of the original QES FIREOutput (GPL-3.0)."""

from contextlib import contextmanager
from datetime import timedelta

import numpy as np

from .output_v2 import OutputField, open_output
from .wind_workspace_output import native_workspace_coordinates


_SURFACE = ('ydim', 'xdim')
_VOLUME = ('t', 'zdim', *_SURFACE)
_FIELDS = {
    'icell': OutputField(_VOLUME, 'i4', '--', 'icell flag value'),
    'x': OutputField(('xdim',), 'f4', 'm', 'x-distance'),
    'y': OutputField(('ydim',), 'f4', 'm', 'y-distance'),
    'z': OutputField(('zdim',), 'f4', 'm', 'z-distance'),
    'u': OutputField(_VOLUME, 'f4', 'm s-1', 'x-component velocity'),
    'v': OutputField(_VOLUME, 'f4', 'm s-1', 'y-component velocity'),
    'w': OutputField(_VOLUME, 'f4', 'm s-1', 'z-component velocity'),
    'terrain': OutputField(_SURFACE, 'f4', 'm', 'terrain height'),
    'burn': OutputField(('t', *_SURFACE), 'f4', '--', 'burn flag value'),
    'fuel': OutputField(_SURFACE, 'f4', '--', 'fuel type'),
    'Force': OutputField(('t', *_SURFACE), 'f4', 'm/s', 'ROS Forcing'),
    'LS': OutputField(('t', *_SURFACE), 'f4', '--', 'Level Set Value'),
}


@contextmanager
def native_fire_series(path, domain, *, origin):
    """Write cell-centred fire and wind records without the ground ghost layer."""
    axes = native_workspace_coordinates(domain)
    coordinates = dict(x=axes['x'], y=axes['y'], z=axes['z'][1:])
    fields = {}
    with open_output(path, start=origin) as output:
        for axis, values in coordinates.items():
            output.define_dimension(axis+'dim', len(values))
        output.attach('fire', _FIELDS, lambda: fields)

        def append(arrays, timestamp, elapsed):
            nonlocal fields
            time = origin+timedelta(seconds=elapsed)
            if time.isoformat(timespec='seconds') != timestamp:
                raise ValueError('Fire timestamp does not match elapsed time')
            fields = arrays | coordinates
            if any(not np.isfinite(fields[key]).all() for key in _FIELDS):
                raise ValueError('Native fire output fields must be finite')
            output.new_time_entry(time, allow_duplicate=True)
            output.push('fire', time)

        yield append
