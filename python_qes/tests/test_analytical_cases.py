from datetime import datetime

import numpy as np
import pytest

from plumeswift.analytical_cases import CASES, prepare_analytical_case
from plumeswift.io import _open_netcdf, netcdf_dates
from plumeswift.plume_fields import open_plume_fields


@pytest.mark.parametrize('name', CASES)
def test_prescribed_legacy_fields(name, tmp_path):
    wind, turbulence, _ = prepare_analytical_case(name, tmp_path)
    uniform = name.startswith('uniform')
    x_direction = name.endswith('-x')
    with _open_netcdf(wind) as nc:
        assert nc['t'].dimensions == ()
        assert netcdf_dates(nc) == (datetime(2022, 1, 1),)
        assert nc['z_face'].dimensions == (('z_face',) if uniform else ('z',))
        assert np.all(nc['icellflag'][0, 0] == (1 if name == 'uniform-y' else 2))
        velocity = nc['u' if x_direction else 'v']
        assert velocity[0, 1, 0, 0] == pytest.approx(2 if uniform else 4.8*.25**.15)
        assert velocity[0, -1, 0, 0] == 0
        if not uniform:
            np.testing.assert_array_equal(velocity[0, -2], velocity[0, -3])
        far_edge = (-1, 0) if uniform or x_direction else (0, -1)
        assert velocity[(0, 1, *far_edge)] == 0
    with _open_netcdf(turbulence) as nc:
        assert nc['CoEps'][0, 0, 0, 0] < 0 < nc['CoEps'][0, 1, 0, 0]
        assert np.sign(nc['tke'][0, 0, 0, 0]) == (1 if x_direction else -1)
        shear = nc['txz' if x_direction else 'tyz']
        assert shear[0, 1, 0, 0] < 0
        assert np.sign(shear[0, 0, 0, 0]) == (1 if name == 'powerlaw-y' else -1)
        if not uniform:
            for field in ('CoEps', 'tke', 'txx', 'tyy', 'tzz', shear.name):
                np.testing.assert_array_equal(nc[field][0, -1], nc[field][0, -2])
    for native in (False, True):
        with open_plume_fields(wind, turbulence, native_layout=native) as (_, records):
            workspace, _, spacing = next(records)
            faces = workspace.coordinates['z_face']
            np.testing.assert_array_equal(np.diff(faces), spacing[2])
            assert faces[1] == 0
