from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from plumeswift import wind_case
from plumeswift.domain import Domain
from plumeswift.io import _open_netcdf
from plumeswift.wind_workspace_output import native_workspace_series


ROOT = Path(__file__).resolve().parents[2]


def test_native_workspace_keeps_duplicate_timestamps_and_packed_roughness(tmp_path):
    domain = Domain((4, 5, 6), (1.3, .8, .6), (.6, .7, .8, .9, 1., 1.1))
    arrays = domain.workspace().arrays
    for name in 'uvw':
        arrays[name] = arrays[name+'0'].copy()
    roughness = np.arange(np.prod(domain.face_shape[1:]), dtype=np.float32).reshape(domain.face_shape[1:])
    arrays['z0_u'] = arrays['z0_v'] = roughness
    origin = datetime(2022, 1, 1)
    path = tmp_path/'windsWk.nc'
    with native_workspace_series(path, domain, origin=origin) as append:
        append(arrays, origin.isoformat(), 0)
        append(arrays | {'u': np.ones(domain.face_shape)}, origin.isoformat(), 0)
    with _open_netcdf(path) as dataset:
        np.testing.assert_array_equal(dataset['t'][:], [0., 0.])
        np.testing.assert_array_equal(dataset['timestamp'][0], dataset['timestamp'][1])
        assert 'UTMx' in dataset.dimensions and 'UTMx' not in dataset.variables
        assert dataset['u'].dtype == np.dtype('f4')
        assert dataset['icellflag'].dtype == np.dtype('i4')
        assert dataset['mixlength'].dtype == np.dtype('f8')
        np.testing.assert_array_equal(dataset['z0_u'][:].ravel(), roughness.ravel()[:20])
        np.testing.assert_array_equal(dataset['x_face'][1:], dataset['x'][:])
        assert dataset['u'][0].max() == 0. and dataset['u'][1].min() == 1.


def test_initialize_only_skips_projection_and_turbulence(tmp_path, monkeypatch):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '12 12 12'
    ET.SubElement(tree.find('simulationParameters'), 'logLawFlag').text = '1'
    config = tmp_path/'winds.xml'
    tree.write(config)

    def unexpected(*args, **kwargs):
        raise AssertionError('Pressure projection called during initialization-only run')

    monkeypatch.setattr(wind_case, 'project', unexpected)
    case = wind_case.read_wind_case(config)
    assert case.wall_log is not None
    (_, result), = case.iter_results(initialize_only=True)
    assert result['solver'] is None and result['turbulence'] is None
    fields = result['workspace'].arrays
    for name in 'uvw':
        np.testing.assert_array_equal(fields[name], fields[name+'0'])
        assert not np.shares_memory(fields[name], fields[name+'0'])
