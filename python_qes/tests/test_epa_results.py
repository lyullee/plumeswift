import numpy as np
from netCDF4 import Dataset

from plumeswift.epa_results import analyze_epa_results


def test_epa_section_order_selection_and_normalization(tmp_path):
    path = tmp_path/'historical.nc'
    coords = dict(x=np.arange(49., 52.), y=np.arange(200., 219., 3.), z=np.arange(0., 23., 3.75))
    counts = np.arange(np.prod([len(axis) for axis in coords.values()]), dtype=np.int32).reshape(
        *(len(axis) for axis in coords.values()))
    with Dataset(path, 'w') as nc:
        for key, axis in coords.items():
            nc.createDimension(key, len(axis))
            nc.createVariable(key, 'f8', (key,))[:] = axis
        nc.createVariable('pBox', 'i4', ('x', 'y', 'z'))[:] = counts
    report, arrays = analyze_epa_results(path)
    assert report['source_m'] == (50, 200, 1)
    np.testing.assert_allclose(arrays['y_sections_over_h'], [0, .2, 1.2])
    np.testing.assert_allclose(arrays['z_sections_over_h'], [.25, .5, .75, 1.25, 1.25])
    factor = 2.83*15**2/(1800*200*1*3*3.75)
    np.testing.assert_allclose(arrays['vertical_cstar'], counts[:, [0, 1, 6], :].transpose(1, 2, 0)*factor)
    np.testing.assert_allclose(arrays['horizontal_cstar'], counts[:, :, [1, 2, 3, 5, 5]].transpose(2, 1, 0)*factor)
