"""Python port of the QES util_output example (GPL-3.0)."""

from datetime import datetime, timedelta, timezone

import numpy as np

from .output_v2 import OutputField, open_output
from .concentration_output import attach_concentration


def write_output_example(path, *, start=None):
    if start is None:
        start = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    with open_output(path, start=start) as output:
        centers = {axis: np.arange(count, dtype=np.float32)+.5
                   for axis, count in (('x', 5), ('y', 5), ('z', 2))}
        concentration = dict(t_avg=1000, p=np.zeros((2, 5, 5), dtype=np.int32),
                             c=np.zeros((2, 5, 5), dtype=np.float32))
        attach_concentration(output, centers, lambda: concentration, count_field='p')

        for axis in 'xy':
            output.define_axis('k_'+axis, np.arange(5, dtype=np.float32)+.5,
                               units='m-1', long_name=f'{axis}-wavenumber')
        spectra = dict(t_collection=0, s=np.zeros((5, 5), dtype=np.float32))

        def prepare_spectra(_time):
            spectra['t_collection'] += 1000

        output.attach('spectra', {
            't_collection': OutputField(('t',), 'f4', 's', 'Collecting time'),
            's': OutputField(('t', 'k_y', 'k_x'), 'f4', 'm-3', 'spectra'),
        }, lambda: spectra, prepare=prepare_spectra)

        first = start+timedelta(seconds=120)
        output.new_time_entry(first)
        output.push('concentration', first, prepare=True)
        output.push('spectra', first, prepare=True)
        second = start+timedelta(seconds=240)
        output.new_time_entry(second)
        output.save(second)
