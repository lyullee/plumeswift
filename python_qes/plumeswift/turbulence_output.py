"""NetCDF layout of the original QES TURBOutput (GPL-3.0)."""

from contextlib import contextmanager
from datetime import timedelta

import numpy as np

from .output_v2 import OutputField, open_output
from .wind_workspace_output import native_workspace_coordinates


_LAYOUT = ('t', 'z', 'y', 'x')


def _schema():
    fields = {
        'CoEps': OutputField(_LAYOUT, 'f4', 'm2s-3', 'dissipation rate'),
        'L': OutputField(_LAYOUT, 'f4', 'm', 'mixing length'),
        'iturbflag': OutputField(_LAYOUT, 'i4', '--', 'icell turb flag'),
        'tke': OutputField(_LAYOUT, 'f4', 'm2s-2', 'turbulent kinetic energy'),
    }
    for axis, velocity in zip('xyz', 'uvw', strict=True):
        fields['div_tau_'+axis] = OutputField(_LAYOUT, 'f4', 'ms-2', f'{axis}-component of stress-tensor divergence')
        for direction in 'xyz':
            name = 'G'+axis+direction
            fields[name] = OutputField(_LAYOUT, 'f4', 's-1', f'velocity gradient tensor: {name} = d{velocity}d{direction}')
    for axes, velocity in zip(('xx', 'xy', 'xz', 'yy', 'yz', 'zz'), ('uu', 'uv', 'uw', 'vv', 'vw', 'ww'), strict=True):
        fields['t'+axes] = OutputField(_LAYOUT, 'f4', 'm2s-2', f'{velocity}-component of stress tensor')
    return dict(sorted(fields.items()))


@contextmanager
def native_turbulence_series(path, domain, *, origin, allow_nan=False):
    """Write the native turbulence schema, including all nine gradient components."""
    schema = _schema()
    fields = {}
    with open_output(path, start=origin) as output:
        coordinates = native_workspace_coordinates(domain)
        for axis in 'xyz':
            output.define_axis(axis, coordinates[axis], units='m', long_name=f'{axis}-distance')
        output.attach('turbulence', schema, lambda: fields)

        def append(arrays, timestamp, elapsed):
            nonlocal fields
            time = origin+timedelta(seconds=elapsed)
            if timestamp not in (time.isoformat(), time.isoformat(timespec='seconds')):
                raise ValueError('Turbulence timestamp does not match elapsed time')
            fields = {name: arrays['Lm' if name == 'L' else name] for name in schema}
            if any(np.isinf(value).any() if allow_nan else not np.isfinite(value).all()
                   for value in fields.values()):
                raise ValueError('Native turbulence fields must be finite')
            output.new_time_entry(time, allow_duplicate=True)
            output.push('turbulence', time)

        yield append
