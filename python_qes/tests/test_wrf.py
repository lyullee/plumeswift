from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from netCDF4 import Dataset
import numpy as np
import pytest

from plumeswift.io import _staged_netcdf_path, read_wind_workspace
from plumeswift.domain import Domain, HorizontalHalo
from plumeswift.wind_case import read_wind_case
from plumeswift.wrf import FireMeshInput, _fire_wind_fields, atmospheric_stations
from plumeswift.wrf_input_demo import pressure_dump


FIXTURES = Path(__file__).resolve().parents[2]/'results/python_port/wrf'


@pytest.mark.parametrize('terrain_source', ['HGT', 'ZSF'])
def test_wrf2dem_exports_atmospheric_and_sfire_terrain(tmp_path, terrain_source):
    from plumeswift.wrf_dem import convert_wrf_dem

    rasterio = pytest.importorskip('rasterio')
    source = FIXTURES.parent/'wrf_dem/wrf.nc'
    if not source.is_file():
        pytest.skip('Run scripts/check_python_wrf_dem.py to create the Lambert fixture')
    path = tmp_path/'wrf.nc'
    with _staged_netcdf_path(path) as staged:
        shutil.copyfile(source, staged)
        with Dataset(staged, 'r+') as nc:
            expected = np.asarray(nc[terrain_source][0], dtype=np.float32)[::-1]
            if terrain_source == 'HGT':
                for variable in ('ZSF', 'FXLONG', 'FXLAT'):
                    nc.renameVariable(variable, f'unused_{variable}')
    dem = tmp_path/'terrain.tif'

    report = convert_wrf_dem(path, dem, sample=2, nz=24)

    assert report['terrain_source'] == terrain_source
    assert report['cells'] == [expected.shape[1], expected.shape[0], 24]
    assert report['spacing_m'] == ([20., 20., 1.] if terrain_source == 'HGT' else [10., 10., 1.])
    with rasterio.open(dem) as actual:
        np.testing.assert_array_equal(actual.read(1), expected)
    case = read_wind_case(report['xml'], compute_turbulence=False)
    assert len(tuple(case.iter_steps())) == 2
    assert np.isfinite(case.initialize().arrays['u0']).all()


def test_original_wrf_pressure_dump_preserves_raw_subset_and_traversal(tmp_path):
    path = tmp_path/'pressure.nc'
    with Dataset(path, 'w') as nc:
        axes = ('time', 'lat', 'lon')
        shape = (3, 6, 5)
        for name, size in zip(axes, shape, strict=True):
            nc.createDimension(name, size)
        pressure = nc.createVariable('pr', 'f4', axes)
        pressure[:] = np.arange(np.prod(shape)).reshape(shape)
        pressure.scale_factor = 2.
    rows, columns = pressure_dump(path).split('[pr] CDL Data Dump\n')[1:]
    assert rows.splitlines()[2:4] == ['0 1 2 3 ', '5 6 7 8 ']
    assert columns.splitlines()[2:4] == ['0 5 10 15 20 25 ', '1 6 11 16 21 26 ']
    assert rows.count('Slice:') == columns.count('Slice:') == 2


@pytest.mark.parametrize('name, cells, stations', [
    ('square', (10, 10, 8), 0),
    ('rectangle_halo', (12, 16, 7), 0),
    ('atmosphere_dem', (18, 18, 24), 25),
    ('cutcell_halo', (12, 16, 7), 0),
])
def test_wrf_snapshot_against_corrected_cpu(name, cells, stations):
    folder = FIXTURES/name
    path = folder/'native_windsWk.nc'
    if not path.exists():
        pytest.skip('Run scripts/check_python_wrf.py --native')
    if name == 'atmosphere_dem':
        pytest.importorskip('rasterio')
    case = read_wind_case(folder/'winds.xml', compute_turbulence=False)
    assert case.domain.cells == cells
    assert len(case.stations) == stations
    actual = case.run()['workspace'].arrays
    expected = read_wind_workspace(path).arrays
    np.testing.assert_array_equal(actual['icellflag'], expected['icellflag'])
    np.testing.assert_allclose(actual['terrain'], expected['terrain'], atol=8e-6, rtol=0)
    if name == 'cutcell_halo':
        assert np.count_nonzero(actual['icellflag'] == 8) == 106
        for field in 'efghmn':
            np.testing.assert_allclose(actual[field], expected[field], atol=2e-9, rtol=0)
    for field in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[field], expected[field], atol=2.5e-6, rtol=0)
    if case.wrf is not None:
        assert case.wrf.record == 9
    else:
        assert case.stations[0].profile.heights == (12.5, 43.5, 88.5, 152.5)


@pytest.mark.parametrize('fixture', ['square', 'atmosphere_dem'])
def test_legacy_wrf_ignores_inactive_external_station_file(tmp_path, fixture):
    source = FIXTURES/fixture
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    if fixture == 'atmosphere_dem':
        pytest.importorskip('rasterio')
        shutil.copyfile(source/'terrain.tif', tmp_path/'terrain.tif')
    shutil.copyfile(source/'wrf.nc', tmp_path/'wrf.nc')
    tree = ET.parse(source/'winds.xml')
    ET.SubElement(tree.find('metParams'), 'sensorName').text = 'campaign_station_unavailable.xml'
    path = tmp_path/'winds.xml'
    tree.write(path)

    options = dict(legacy_turbulence_xml=True, compute_turbulence=False)
    original = read_wind_case(source/'winds.xml', **options).initialize()
    moved = read_wind_case(path, **options)
    assert 'ignored /metParams/sensorName/campaign_station_unavailable.xml while using WRF' in moved.xml_compatibility
    actual = moved.initialize()
    for name in ('u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual.arrays[name], original.arrays[name])
    with pytest.raises(FileNotFoundError):
        read_wind_case(path)


def test_wrf_frame_checksum_rejects_corruption(tmp_path):
    source = FIXTURES/'square/wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    path = tmp_path/'corrupt.nc'
    shutil.copyfile(source, path)
    with Dataset(path, 'a') as nc:
        nc['U0_FMW'][-1, 0, 1, 2] += np.float32(.5)
    with pytest.raises(ValueError, match='checksum mismatch'):
        FireMeshInput.read(path)


def test_wrf_fire_wind_reads_the_u_vertical_extent(tmp_path):
    with Dataset(tmp_path/'wrf.nc', 'w') as nc:
        for name, size in [('time', 1), ('z', 2), ('z_padded', 3), ('y', 2), ('x', 3)]:
            nc.createDimension(name, size)
        nc.createVariable('HT_FMW', 'f4', ('time', 'z_padded'))[0] = [4., 12., np.nan]
        for name, level in [('U0_FMW', 'z'), ('V0_FMW', 'z_padded'), ('W0_FMW', 'z_padded')]:
            field = nc.createVariable(name, 'f4', ('time', level, 'y', 'x'))
            field[0, :2] = np.arange(12).reshape(2, 2, 3)
        heights, winds = _fire_wind_fields(nc, 0, 3, 2)
        np.testing.assert_array_equal(heights, [4., 12.])
        for wind in winds:
            np.testing.assert_array_equal(wind, np.arange(12).reshape(2, 2, 3))
        with pytest.raises(ValueError, match='wind shapes disagree'):
            _fire_wind_fields(nc, 0, 4, 2)


def test_legacy_atmospheric_wrf_uses_stored_values(tmp_path):
    source = FIXTURES/'atmosphere_dem'
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    pytest.importorskip('rasterio')
    for name in ('terrain.tif', 'winds.xml'):
        shutil.copyfile(source/name, tmp_path/name)
    with _staged_netcdf_path(tmp_path/'wrf.nc') as temporary:
        shutil.copyfile(source/'wrf.nc', temporary)
        with Dataset(temporary, 'r+') as nc:
            nc['U'].scale_factor = .25
            nc['V'].add_offset = 1.25
            nc['HGT'].add_offset = 5.
    baseline = read_wind_case(source/'winds.xml', legacy_turbulence_xml=True)
    legacy = read_wind_case(tmp_path/'winds.xml', legacy_turbulence_xml=True)
    modern = read_wind_case(tmp_path/'winds.xml')
    assert legacy.stations == baseline.stations
    assert modern.stations[0].profile != baseline.stations[0].profile


@pytest.mark.parametrize('meridional,legacy_direction,modern_direction', [(3., 0., 180.), (-3., 180., 0.)])
def test_legacy_atmospheric_wrf_preserves_original_meridional_direction(tmp_path, meridional, legacy_direction,
                                                               modern_direction):
    source = FIXTURES/'atmosphere_dem'
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    pytest.importorskip('rasterio')
    for name in ('terrain.tif', 'winds.xml'):
        shutil.copyfile(source/name, tmp_path/name)
    with _staged_netcdf_path(tmp_path/'wrf.nc') as temporary:
        shutil.copyfile(source/'wrf.nc', temporary)
        with Dataset(temporary, 'r+') as nc:
            nc['U'][:] = 0.
            nc['V'][:] = meridional
    legacy = read_wind_case(tmp_path/'winds.xml', legacy_turbulence_xml=True)
    modern = read_wind_case(tmp_path/'winds.xml')
    assert all(station.profile.direction == legacy_direction for station in legacy.stations)
    assert all(station.profile.direction == modern_direction for station in modern.stations)


@pytest.mark.parametrize('first_level', [0., -5.])
def test_legacy_atmospheric_wrf_keeps_nonpositive_first_level(tmp_path, first_level):
    source = FIXTURES/'atmosphere_dem'
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    pytest.importorskip('rasterio')
    for name in ('terrain.tif', 'winds.xml'):
        shutil.copyfile(source/name, tmp_path/name)
    with _staged_netcdf_path(tmp_path/'wrf.nc') as temporary:
        shutil.copyfile(source/'wrf.nc', temporary)
        with Dataset(temporary, 'r+') as nc:
            nc['HGT'][:] = 0.
            nc['PHB'][:, 0] = (2*first_level-10.)*9.81
            nc['PHB'][:, 1] = 98.1
    legacy = read_wind_case(tmp_path/'winds.xml', legacy_turbulence_xml=True)
    modern = read_wind_case(tmp_path/'winds.xml')
    for original, filtered in zip(legacy.stations, modern.stations, strict=True):
        assert original.profile.heights[0] == pytest.approx(first_level, abs=1e-6)
        assert original.profile.heights[1:] == filtered.profile.heights


def test_wrf_fire_mesh_stores_atmospheric_resolution_as_native_float(tmp_path):
    source = FIXTURES/'square/wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    with Dataset(path, 'r+') as nc:
        nc.DX, nc.DY = np.float64(20.2000003), np.float64(13.3000001)

    mesh = FireMeshInput.read(path)
    expected = float(np.float32(10.10000015)), float(np.float32(6.65000005)), 12.
    assert mesh.spacing == expected
    assert mesh.domain(HorizontalHalo()).spacing == expected


@pytest.mark.parametrize('relief,levels', [(8., 1), (15., 2)])
def test_wrf_low_relief_uses_native_vertical_extent(tmp_path, relief, levels):
    source = FIXTURES/'square'
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source/'wrf.nc', path)
    with Dataset(path, 'r+') as nc:
        elevation = nc['ZSF'][0]
        core = elevation[:10, :10]
        nc['ZSF'][:] = 20.+relief*(elevation-core.min())/(core.max()-core.min())
    xml = tmp_path/'winds.xml'
    shutil.copyfile(source/'winds.xml', xml)

    native_case = read_wind_case(xml, compute_turbulence=False, poisson_backend='cpu-native-rb',
                                 native_sensor_math=True)
    assert native_case.domain.cells == (10, 10, levels)
    native = native_case.run()
    modern = read_wind_case(xml, compute_turbulence=False, native_sensor_math=True).run()
    assert modern['solver'].converged
    assert native['workspace'].arrays['u'].shape == (levels+2, 11, 11)
    assert np.isfinite(native['workspace'].arrays['u']).all()
    assert np.any(native['workspace'].arrays['u'] != 0)
    if levels == 1:
        assert native['solver'].converged and native['solver'].iterations == 1
        assert not np.any(native['multiplier'])
        assert not np.any(modern['multiplier'])
    else:
        # The original single-thread and synchronized OpenMP references both
        # reach the cap; an unsynchronized max_error reset can stop early.
        assert not native['solver'].converged and native['solver'].iterations == 3000
        assert native['solver'].update_linf == np.float32(2.**-17)

    reference = FIXTURES.parent/'wrf_shallow'/('one' if levels == 1 else 'two')/'native_windsWk.nc'
    if reference.is_file():
        expected = read_wind_workspace(reference).arrays
        for field in ('icellflag', 'u0', 'v0', 'w0'):
            np.testing.assert_array_equal(native['workspace'].arrays[field], expected[field])
        for field in ('u', 'v', 'w'):
            np.testing.assert_allclose(native['workspace'].arrays[field], expected[field], atol=3e-7, rtol=0)


def test_wrf_below_sea_level_mesh_uses_native_vertical_extent(tmp_path):
    source = FIXTURES/'square/wrf.nc'
    if not source.is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    with Dataset(path, 'r+') as nc:
        rows, columns = nc['ZSF'].shape[1:]
        nc['ZSF'][:] = np.linspace(-85., -80., rows*columns).reshape(rows, columns)

    grid = FireMeshInput.read(path)

    assert grid.cells == (10, 10, 8)
    assert grid.terrain.min() == -85.
    assert grid.terrain.max() < -80.


def test_ideal_wrf_metre_coordinates_through_wind_case(tmp_path):
    source = FIXTURES/'atmosphere_dem'
    if not (source/'wrf.nc').is_file():
        pytest.skip('Run scripts/check_python_wrf.py to create the fixture')
    pytest.importorskip('rasterio')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source/'wrf.nc', path)
    with Dataset(path, 'r+') as nc:
        j, i = np.indices(nc['XLONG'].shape[1:])
        nc['XLONG'][0] = 500+23+29*i
        nc['XLAT'][0] = 1000+21+28*j
    xml = ET.parse(source/'winds.xml')
    simulation = xml.find('simulationParameters')
    simulation.find('DEM').text = str(source/'terrain.tif')
    simulation.find('UTMx').text = '500'
    simulation.find('UTMy').text = '1000'
    simulation.find('UTMZone').text = '0'
    ET.SubElement(simulation, 'WRFCoordinates').text = 'ideal-local'
    config = tmp_path/'winds.xml'
    xml.write(config, encoding='utf-8')
    case = read_wind_case(config, compute_turbulence=False)
    assert case.wrf_atmosphere_coordinates == 'ideal-local'
    expected = read_wind_case(source/'winds.xml', compute_turbulence=False)
    assert [station.profile for station in case.stations] == [station.profile for station in expected.stations]
    assert [(station.x, station.y) for station in case.stations] == [
        (23+29*i, 21+28*j) for j in range(5) for i in range(5)]
    workspace = case.run()['workspace']
    assert np.isfinite(workspace.arrays['u']).all()
    assert np.any(workspace.arrays['u'] != 0)

    # A small QES domain reads just its WRF mass cells plus the final U/V
    # faces. A bad value outside that window must not affect the profiles.
    subset = Domain((3, 3, 5), (20., 20., 4.))
    args = (path, subset, HorizontalHalo(), (500., 1000.), 0, 1)
    cropped = atmospheric_stations(*args, coordinate_mode='ideal-local')
    assert cropped == tuple(case.stations[i] for i in (0, 1, 5, 6))
    with Dataset(path, 'r+') as nc:
        nc['U'][0, 0, 4, 4] = np.nan
        nc['V'][0, 0, 4, 4] = np.nan
    assert atmospheric_stations(*args, coordinate_mode='ideal-local') == cropped
    with Dataset(path, 'r+') as nc:
        nc['U'][0, 0, 1, 2] = np.nan
    with pytest.raises(ValueError, match='variable U.*nonfinite'):
        atmospheric_stations(*args, coordinate_mode='ideal-local')
