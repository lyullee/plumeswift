from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.io import _open_netcdf, _staged_netcdf_path, netcdf_dates, read_wind_workspace
from plumeswift.hrrr import HRRRInput, _data, _fields
from plumeswift.wind_case import read_wind_case


FIXTURES = Path(__file__).resolve().parents[2]/'results/python_port/hrrr'


def test_legacy_hrrr_barnes_keeps_station_at_float_domain_edge(tmp_path):
    from netCDF4 import Dataset
    from plumeswift.utm import to_utm

    source = FIXTURES/'barnes'
    if not (source/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    with _staged_netcdf_path(tmp_path/'hrrr.nc') as staged:
        shutil.copyfile(source/'hrrr.nc', staged)
        with Dataset(staged, 'r+') as nc:
            longitude = float(nc['longitude'][2, -1])-360.
            latitude = float(nc['latitude'][2, -1])
            east = float(to_utm(longitude, latitude, native_zone=True)[0])
            nc['UGRD_10maboveground'][:, 2, -1] = 10.
    tree = ET.parse(source/'winds.xml')
    tree.find('simulationParameters/cellSize').text = '199.9995 200 20'
    tree.find('simulationParameters/UTMx').text = str(east-4200.)
    path = tmp_path/'winds.xml'
    tree.write(path)

    legacy = read_wind_case(path, native_sensor_math=True, legacy_turbulence_xml=True)
    modern = read_wind_case(path, native_sensor_math=True)
    assert max(station.profile.speed for station in legacy.stations) > 9.
    assert max(station.profile.speed for station in modern.stations) < 9.


def test_legacy_hrrr_uses_float_zone_selection_at_boundary(tmp_path):
    from netCDF4 import Dataset
    from plumeswift.hrrr import _locations

    longitude = float(np.nextafter(np.float32(-60.), np.float32(np.inf)))
    with Dataset(tmp_path/'hrrr.nc', 'w', diskless=True) as nc:
        for axis in ('x', 'y'):
            nc.createDimension(axis, 1)
            nc.createVariable(axis, 'f8', (axis,))[:] = 0.
        nc.createVariable('longitude', 'f8', ('y', 'x'))[:] = longitude+360.
        nc.createVariable('latitude', 'f8', ('y', 'x'))[:] = 45.
        options = (nc, (734700., 4986950.), 20, (4200., 3600.), 6000.)

        legacy = _locations(*options, positional=True)
        modern = _locations(*options)

    assert legacy[-1].tolist() == [20]
    assert modern[-1].tolist() == [21]
    assert legacy[3][0]-modern[3][0] > 400_000.


def test_legacy_hrrr_pressure_terrain_uses_defined_face_stride(monkeypatch):
    from plumeswift.domain import Domain
    from plumeswift.hrrr import HRRRProfile, HRRRStation
    from plumeswift.hrrr_interpolation import initialize_hrrr

    workspace = Domain((6, 5, 8), (1., 1., 1.)).workspace()
    terrain = workspace.arrays['terrain']
    terrain[:] = np.arange(terrain.size).reshape(terrain.shape)
    profile = HRRRProfile(2., 270., .1, .01, 200., ((3., 270.),)*3)
    stations = (HRRRStation(1., 2., profile), HRRRStation(4., 4., profile))
    elevations = []

    def capture(_, heights, altitude, **options):
        elevations.append(altitude)
        return np.zeros((2, len(heights)))

    monkeypatch.setattr(HRRRProfile, 'evaluate', capture)
    initialize_hrrr(workspace, stations, (), 1, legacy_terrain_stride=True)
    assert elevations == [15., 28.]
    elevations.clear()
    initialize_hrrr(workspace, stations, (), 1)
    assert elevations == [13., 28.]


def test_legacy_hrrr_ignores_external_station_files(tmp_path):
    folder = FIXTURES/'barnes'
    if not (folder/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    shutil.copyfile(folder/'hrrr.nc', tmp_path/'hrrr.nc')
    tree = ET.parse(folder/'winds.xml')
    ET.SubElement(tree.find('metParams'), 'sensorName').text = 'old_sensor.xml'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    baseline = read_wind_case(folder/'winds.xml', legacy_turbulence_xml=True)
    actual = read_wind_case(xml, legacy_turbulence_xml=True)
    assert actual.schedule == baseline.schedule
    assert 'ignored /metParams/sensorName/old_sensor.xml while using HRRR' in actual.xml_compatibility


@pytest.mark.parametrize(('native_math', 'legacy'), [(False, False), (True, False), (True, True)])
def test_hrrr_nearest_map_is_built_before_halo_translation(tmp_path, native_math, legacy):
    folder = FIXTURES/'nearest'
    if not (folder/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    shutil.copyfile(folder/'hrrr.nc', tmp_path/'hrrr.nc')
    tree = ET.parse(folder/'winds.xml')
    simulation = tree.find('simulationParameters')
    simulation.find('halo_x').text = '450'
    simulation.find('halo_y').text = '350'
    path = tmp_path/'winds.xml'
    tree.write(path)

    options = dict(native_sensor_math=native_math, legacy_turbulence_xml=legacy, compute_turbulence=False)
    baseline = read_wind_case(folder/'winds.xml', **options)
    with_halo = read_wind_case(path, **options)
    assert with_halo.stations[0].x == baseline.stations[0].x+450
    assert with_halo.hrrr_nearest_map_positions == baseline.hrrr_nearest_map_positions
    assert with_halo.hrrr_nearest_map_positions
    expected = baseline.initialize().arrays
    actual = with_halo.initialize().arrays
    for field in ('u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual[field], expected[field])


def test_hrrr_bilinear_selects_original_cell_and_weights_translated_corners():
    from plumeswift.hrrr_interpolation import _quadrilaterals

    grid_ids = ((0, 0), (0, 1), (1, 1), (1, 0))
    original = np.array([[.5, .5], [2.5, .5], [2.5, 2.5], [.5, 2.5]], dtype=np.float32)
    translated = original+np.float32(1)
    jj, ii = np.indices((5, 5))
    ids, weights, covered = _quadrilaterals(grid_ids, translated, ii, jj, np.float32(1), np.float32(1),
                                            native_math=True, map_positions=original)

    assert covered[1, 1] and not covered[3, 3]
    np.testing.assert_array_equal(ids[:, 1, 1], [0, 1, 2, 3])
    np.testing.assert_array_equal(weights[:, 1, 1], [2, 2, -.5, 2.5, -.5, 2.5, 2, -.5, 2.5])


def test_hrrr_nearest_falls_back_if_the_startup_site_is_unavailable():
    from plumeswift.hrrr_interpolation import _nearest_sites

    positions = np.array([[-.5, 2.], [2., 1.], [2., 2.]], dtype=np.float32)
    jj, ii = np.indices((5, 5))
    active_only = _nearest_sites(positions[1:], ii, jj, 1., 1.)
    native = _nearest_sites(positions, ii, jj, 1., 1., available=np.array([0, 0, 1]))

    assert active_only[2, 0] == 1
    assert native[2, 0] == 0
    assert native[2, 2] == 1


@pytest.mark.parametrize('name', ['nearest', 'bilinear'])
def test_legacy_hrrr_host_interpolation_uses_native_precision_with_cuda_pressure(name):
    folder = FIXTURES/name
    if not (folder/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    options = dict(legacy_turbulence_xml=True, compute_turbulence=False)
    cpu = read_wind_case(folder/'winds.xml', native_sensor_math=True, **options)
    gpu = read_wind_case(folder/'winds.xml', poisson_backend='cuda-global-memory',
                         meteorology_backend='cuda', **options)

    for cpu_step, gpu_step in zip(cpu.iter_steps(), gpu.iter_steps(), strict=True):
        expected, actual = (step.initialize().arrays for step in (cpu_step, gpu_step))
        for field in ('u0', 'v0', 'w0'):
            np.testing.assert_array_equal(actual[field], expected[field])


def test_legacy_hrrr_uses_stored_values(tmp_path):
    from netCDF4 import Dataset

    folder = FIXTURES/'barnes'
    if not (folder/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    with _staged_netcdf_path(tmp_path/'hrrr.nc') as temporary:
        shutil.copyfile(folder/'hrrr.nc', temporary)
        with Dataset(temporary, 'r+') as nc:
            nc['time'].add_offset = 30.
            nc['UGRD_10maboveground'].scale_factor = .25
    shutil.copyfile(folder/'winds.xml', tmp_path/'winds.xml')
    baseline = read_wind_case(folder/'winds.xml', legacy_turbulence_xml=True)
    legacy = read_wind_case(tmp_path/'winds.xml', legacy_turbulence_xml=True)
    modern = read_wind_case(tmp_path/'winds.xml')
    assert legacy.schedule == baseline.schedule
    assert modern.timestamp != baseline.timestamp
    assert modern.stations[0].profile.speed != baseline.stations[0].profile.speed


def test_legacy_hrrr_reads_positional_extents(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path/'hrrr.nc', 'w') as nc:
        for name, size in [('time', 2), ('y', 2), ('x', 2), ('padded', 3), ('column', 2)]:
            nc.createDimension(name, size)
        time = nc.createVariable('time', 'f8', ('padded', 'column'))
        time[:2, 0] = [1600041600, 1600045200]
        wind = nc.createVariable('UGRD_10maboveground', 'f4', ('padded',)*3)
        wind[1, :2, :2] = [[1., 2.], [3., 4.]]
        np.testing.assert_array_equal(_data(nc, 'time', ('time',), positional=True), [1600041600, 1600045200])
        values, = _fields(nc, (wind.name,), 1, np.array([0, 1]), np.array([1, 0]), positional=True)
        np.testing.assert_array_equal(values, [2., 3.])
        with pytest.raises(ValueError, match='must use dimensions'):
            _data(nc, 'time', ('time',))


def test_legacy_yakima_hrrr_density_configuration(tmp_path):
    from netCDF4 import Dataset

    folder = FIXTURES/'barnes'
    if not (folder/'hrrr.nc').exists():
        pytest.skip('Run scripts/check_python_hrrr.py for the smoke density fixture')
    case = read_wind_case(folder/'winds.xml')
    example = Path(__file__).resolve().parents[2]/'external/QES-Public/data/InputFiles/Yakima_Plume.xml'
    path = tmp_path/'hrrr.nc'
    with _staged_netcdf_path(path) as staged:
        shutil.copyfile(folder/'hrrr.nc', staged)
        with Dataset(staged, 'r+') as nc:
            density = nc['MASSDEN_8maboveground']
            density.scale_factor = .5
            density.add_offset = 10.
    source = HRRRInput.from_smoke_xml(example, origin=case.hrrr.origin, zone=case.hrrr.zone,
                                     hrrr_path=path)
    for record in (0, 1):
        values = source.smoke_sources(case.domain, record, stored=True)
        np.testing.assert_array_equal(values['mass_density'], 1+record+values['y_index']+values['x_index'])
        assert values['mass_density'].dtype == np.float32
        decoded = source.smoke_sources(case.domain, record)
        np.testing.assert_array_equal(decoded['mass_density'], .5*values['mass_density']+10.)
    with pytest.raises(ValueError, match='outside the time axis'):
        source.smoke_sources(case.domain, -1)


@pytest.mark.parametrize('limit', [0, 1])
def test_hrrr_equal_whole_second_uses_last_source_record(tmp_path, limit):
    from netCDF4 import Dataset

    folder = FIXTURES/'barnes'
    if not (folder/'hrrr.nc').is_file():
        pytest.skip('Run scripts/check_python_hrrr.py to create the fixture')
    with _staged_netcdf_path(tmp_path/'hrrr.nc') as temporary:
        shutil.copyfile(folder/'hrrr.nc', temporary)
        with Dataset(temporary, 'r+') as nc:
            epoch = int(nc['time'][0])
            nc['time'][:] = (epoch+.9, epoch+.1)
    tree = ET.parse(folder/'winds.xml')
    tree.find('simulationParameters/totalTimeIncrements').text = str(limit)
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    original = tuple(read_wind_case(folder/'winds.xml', compute_turbulence=False,
                                    native_sensor_math=True).iter_steps())
    case = read_wind_case(xml, compute_turbulence=False, native_sensor_math=True)
    steps = tuple(case.iter_steps())
    assert len(steps) == (limit or 2)
    assert original[0].stations != original[1].stations
    assert all(step.timestamp == original[0].timestamp for step in steps)
    assert all(step.stations == original[1].stations for step in steps)
    expected = original[1].initialize().arrays
    for step in steps:
        actual = step.initialize().arrays
        for field in ('u0', 'v0', 'w0'):
            np.testing.assert_array_equal(actual[field], expected[field])


@pytest.mark.parametrize('name, stations', [('barnes', 16), ('nearest', 16), ('bilinear', 36), ('bilinear_dem', 36)])
def test_hrrr_native_cpu(name, stations):
    folder = FIXTURES/name
    reference = 'safe_terrain' if name.startswith('bilinear') else 'native'
    path = folder/(reference+'_windsWk.nc')
    if not path.exists():
        pytest.skip('Run scripts/check_python_hrrr.py --native')
    if name.endswith('_dem'):
        pytest.importorskip('rasterio')
    case = read_wind_case(folder/'winds.xml')
    with _open_netcdf(path) as fields:
        dates = netcdf_dates(fields)
    steps = tuple(case.iter_steps())
    assert len(steps) == len(dates) == 2
    assert steps[0].stations[0].profile.speed != steps[1].stations[0].profile.speed
    for index, (step, date) in enumerate(zip(steps, dates)):
        assert step.timestamp == date.isoformat(timespec='seconds')
        assert len(step.stations) == stations
        actual = step.run()['workspace'].arrays
        expected = read_wind_workspace(path, step=index).arrays
        np.testing.assert_array_equal(actual['icellflag'], expected['icellflag'])
        np.testing.assert_allclose(actual['terrain'], expected['terrain'], atol=5e-6, rtol=0)
        for field in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
            np.testing.assert_allclose(actual[field], expected[field], atol=1.3e-5, rtol=0)
    if name == 'barnes':
        smoke = case.hrrr.smoke_sources(case.domain, 1)
        assert len(smoke['mass_density']) == 16
        np.testing.assert_array_equal(smoke['mass_density'], 2+smoke['x_index']+smoke['y_index'])
