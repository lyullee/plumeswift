"""NetCDF collection output shared by plume and fire smoke (GPL-3.0)."""

from contextlib import contextmanager
from datetime import timedelta

import numpy as np

from .output_v2 import OutputField, open_output
from .plume_collection import native_box_centers


def attach_concentration(output, coordinates, read, *, count_field='p_count'):
    for axis in 'xyz':
        output.define_axis(axis+'_c', np.asarray(coordinates[axis], dtype=np.float32),
                           units='m', long_name=f'{axis}-center collection box')
    grid = ('t', 'z_c', 'y_c', 'x_c')
    output.attach('concentration', {
        't_avg': OutputField(('t',), 'f4', 's', 'Averaging time'),
        count_field: OutputField(grid, 'i4', '#ofPar', 'number of particle per box'),
        'c': OutputField(grid, 'f4', 'g m-3', 'concentration'),
    }, read)


@contextmanager
def native_collection_series(path, config, origin):
    """Write the native collection layout, including a defined zero-record file."""
    fields = {}
    with open_output(path, start=origin) as output:
        attach_concentration(output, native_box_centers(config), lambda: fields)

        def append(arrays, timestamp, elapsed):
            nonlocal fields
            time = origin+timedelta(seconds=elapsed)
            if time.isoformat(timespec='seconds') != timestamp:
                raise ValueError('Collection timestamp does not match elapsed time')
            fields = arrays
            output.new_time_entry(time)
            output.push('concentration', time)

        yield append
