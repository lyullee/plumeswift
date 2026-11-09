from pathlib import Path

import numpy as np
import pytest

from plumeswift.io import _open_netcdf, read_wind_workspace
from plumeswift.wind_case import read_wind_case


FIXTURES = Path(__file__).resolve().parents[2]/'results/python_port/roc'


@pytest.mark.parametrize('name, bleed, wake', [
    ('interior', 0, 0), ('thin_neutral', 834, 0), ('unstable', 0, 0),
    ('wake', 0, 59248), ('thin_wake', 834, 39514),
])
def test_row_canopy_matches_native_cpu(name, bleed, wake):
    folder = FIXTURES/name
    if not (folder/'native_turbOut.nc').exists():
        pytest.skip('Run scripts/check_python_roc.py --native')
    result = read_wind_case(folder/'winds.xml').run()
    actual = result['workspace'].arrays
    native = read_wind_workspace(folder/'native_windsWk.nc').arrays
    np.testing.assert_array_equal(actual['icellflag'], native['icellflag'])
    for field in 'efghmn':
        np.testing.assert_allclose(actual[field], native[field], atol=6e-7, rtol=0)
    assert np.count_nonzero(actual['icellflag'] == 28) == bleed
    assert np.count_nonzero(actual['icellflag'] == 33) == wake
    for field in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[field][:-1], native[field][:-1], atol=2e-5, rtol=0)
    turb = result['turbulence']
    interior = np.s_[1:-1, 1:-1, 1:-1]
    local_mixing = .4*turb['mixingLengths'][interior]
    with _open_netcdf(folder/'native_turbOut.nc') as data:
        np.testing.assert_allclose(turb['Lm'], data['L'][0], atol=4e-6, rtol=0)
        changes = np.abs(turb['Lm'][interior]-local_mixing) > 1e-5
        np.testing.assert_array_equal(changes, np.abs(data['L'][0][interior]-local_mixing) > 1e-5)
        assert bool(changes.any()) == bool(wake)
        for field in ('tke', 'CoEps', 'txx', 'tyy', 'tzz', 'txz', 'tyz', 'txy',
                      'div_tau_x', 'div_tau_y', 'div_tau_z'):
            np.testing.assert_allclose(turb[field], data[field][0], atol=2e-4, rtol=0)
