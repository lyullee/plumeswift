"""Positional NetCDF input diagnostic from scratch/wrfInputTest.cpp."""
from .netcdf_legacy import open_legacy_input


def pressure_dump(path):
    """Print the original 2 × 6 × 4 raw ``pr`` subset in both traversal orders."""
    with open_legacy_input(path) as source:
        pressure = source.read('pr', dtype='f8', start=(0, 0, 0), count=(2, 6, 4))
    lines = []
    for records in (pressure, pressure.transpose(0, 2, 1)):
        lines.extend(('[pr] CDL Data Dump', '=========================='))
        for time, record in enumerate(records):
            lines.append(f'Slice: (t={time})')
            lines.extend(''.join(f'{value:.6g} ' for value in row) for row in record)
    return '\n'.join(lines)+'\n'
