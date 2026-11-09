from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.io import _open_netcdf
from plumeswift.turbulence_output import native_turbulence_series
from plumeswift.wind_case import read_wind_case


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('ridge_column', [0, 8])
def test_stress_rotation_only_requires_clearance_over_active_terrain_columns(tmp_path, ridge_column):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin

    elevation = np.full((24, 24), 100., dtype=np.float32)
    elevation[:, ridge_column] = 130.
    with rasterio.open(tmp_path/'terrain.tif', 'w', driver='GTiff', width=24, height=24, count=1,
                       dtype='float32', transform=from_origin(1000, 2024, 1, 1)) as raster:
        raster.write(elevation, 1)
    xml = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    sim = xml.find('simulationParameters')
    sim.find('domain').text = '24 24 24'
    sim.find('maxIterations').text = '8'
    ET.SubElement(sim, 'DEM').text = 'terrain.tif'
    sensor = xml.find('metParams/sensor')
    sensor.find('site_xcoord').text = sensor.find('site_ycoord').text = '5'
    path = tmp_path/'wind.xml'
    xml.write(path)
    case = read_wind_case(path, poisson_backend='cpu-native-rb', native_turbulence_math=True,
                          native_sensor_math=True, legacy_turbulence_xml=True)
    if ridge_column == 0:
        result = case.run()
        assert result['workspace'].arrays['terrain'][:, 0].max() == 15.
        assert np.isfinite(result['turbulence']['txx']).all()
        assert np.any(result['turbulence']['txx'] > 0.)
    else:
        with pytest.raises(ValueError, match='15 m above terrain'):
            case.run()


@pytest.mark.parametrize('custom_widths', [False, True])
def test_legacy_stretching_switch_selects_turbulence_stencil_only(tmp_path, custom_widths):
    xml = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    sim = xml.find('simulationParameters')
    sim.find('domain').text = '10 11 28'
    sim.find('cellSize').text = '1 1 .7'
    stretch = sim.find('verticalStretching')
    path = tmp_path/'wind.xml'
    options = dict(legacy_turbulence_xml=True, native_turbulence_math=True, native_sensor_math=True)
    xml.write(path)
    uniform = read_wind_case(path, **options).run(include_gradients=True)
    stretch.text = '1'
    if custom_widths:
        for value in (.8, 1.1)*14:
            ET.SubElement(sim, 'dz_value').text = str(value)
    xml.write(path)
    case = read_wind_case(path, **options)
    assert case.domain.vertical_widths is None
    assert case.turbulence.force_nonuniform_z
    legacy = case.run(include_gradients=True)
    np.testing.assert_array_equal(legacy['workspace'].arrays['u'], uniform['workspace'].arrays['u'])
    assert not np.array_equal(legacy['turbulence']['Gxz'], uniform['turbulence']['Gxz'])
    assert not np.array_equal(legacy['turbulence']['div_tau_x'], uniform['turbulence']['div_tau_x'])
    if custom_widths:
        assert read_wind_case(path).domain.vertical_widths == (.8, 1.1)*14
    else:
        with pytest.raises(ValueError, match='Custom dz'):
            read_wind_case(path)


def test_native_turbulence_serializes_gradients_and_subsecond_times(tmp_path):
    xml = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    xml.find('simulationParameters/domain').text = '12 12 20'
    config = tmp_path/'wind.xml'
    xml.write(config)
    case = read_wind_case(config)
    baseline = case.run()['turbulence']
    result = case.run(include_gradients=True)
    fields = result['turbulence']
    names = {f'G{axis}{direction}' for axis in 'xyz' for direction in 'xyz'}
    assert fields.keys()-baseline.keys() == names
    for name in baseline:
        np.testing.assert_array_equal(baseline[name], fields[name])
    assert np.max(np.abs(fields['Gxz'])) > 0

    origin = datetime.fromisoformat('2022-01-01T00:00:00.125')
    path = tmp_path/'turbOut.nc'
    with native_turbulence_series(path, case.domain, origin=origin) as append:
        for elapsed in (0., .25, .25):
            append(fields, '2022-01-01T00:00:00', elapsed)
    with _open_netcdf(path) as file:
        assert file['t'].simulation_start == '2022-01-01T00:00:00.125000'
        np.testing.assert_array_equal(file['t'][:], [0., .25, .25])
        assert file['L'].units == 'm' and file['Gxz'].long_name.endswith('dudz')
        assert file['Gxz'].dtype == np.dtype('f4')
        assert file['iturbflag'].dtype == np.dtype('i4')
        np.testing.assert_array_equal(file['Gxz'][1], fields['Gxz'].astype(np.float32))
        np.testing.assert_array_equal(file['L'][1], fields['Lm'].astype(np.float32))
