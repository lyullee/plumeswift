from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from plumeswift.domain import Domain
from plumeswift.geometry import PolygonBuilding, wall_stencil, zero_solid_faces
from plumeswift.meteorology import SensorProfile, single_sensor_flat
from plumeswift.turbulence import LocalTurbulence, cell_velocity
from plumeswift.io import read_wind_workspace, write_netcdf
from plumeswift.wind_case import WindCase, read_wind_case, UnsupportedFeatures

ROOT = Path(__file__).resolve().parents[2]


def test_geometry_initialization_leaves_wind_profile_unapplied():
    building = PolygonBuilding(((2., 2.), (5., 2.), (5., 5.), (2., 5.), (2., 2.)), 4.)
    case = WindCase(Domain((8, 8, 20), (1., 1., 1.)), SensorProfile(2., 225., 15.), buildings=(building,))
    geometry = case.initialize_geometry()
    workspace = case.initialize()
    assert np.any(geometry.arrays['icellflag'] == 0)
    assert not any(geometry.arrays[name].any() for name in ('u0', 'v0', 'w0'))
    assert np.any(workspace.arrays['u0'])
    for name in ('icellflag_initial', *'efghmn'):
        np.testing.assert_array_equal(geometry.arrays[name], workspace.arrays[name])


def test_native_barnes_accepts_station_in_east_half_cell():
    from plumeswift.meteorology import Station

    domain = Domain((24, 20, 28), (1., 1., 1.))
    west, east = SensorProfile(2., 270., 10.), SensorProfile(4., 236., 10.)
    case = WindCase(domain, west, stations=(Station(5., 5., west), Station(23.75, 11., east)),
                    native_sensor_math=True)

    arrays = case.initialize().arrays
    assert np.isfinite(arrays['u0']).all()
    assert arrays['u0'][10, 11, -1] > arrays['u0'][10, 11, 0]
    assert arrays['v0'][10, 11, -1] > arrays['v0'][10, 11, 0]


def test_native_barnes_uses_the_mean_distance_with_collocated_stations():
    from plumeswift.native_sensor import barnes_scale

    positions = np.array([[5., 5.], [5., 5.], [15., 5.]], dtype=np.float32)
    mean = float(np.float32(10)/np.float32(3))
    assert barnes_scale(positions) == np.float32(5.052*(2.*mean/np.pi)**2)
    with pytest.raises(ValueError, match='Zero mean nearest-station distance'):
        barnes_scale(positions[:2])


def test_partially_embedded_building_uses_roof_elevation_and_keeps_ground():
    domain = Domain((20, 18, 24), (1., 1., 1.))
    building = PolygonBuilding.rectangle(8.25, 7.25, 5.5, 4.5, 4.75, base_height=-.8)
    workspace = domain.workspace()
    building.flag(workspace, native_math=True)

    flags = workspace.arrays['icellflag']
    np.testing.assert_array_equal(flags[:6, 9, 10], [2, 0, 0, 0, 0, 1])


@pytest.mark.parametrize('x_start,wrapped_y,wrapped_x', [(-2.25, 7, 23), (22.25, 9, 0)])
def test_legacy_staircase_keeps_in_bounds_packed_edge_cells(x_start, wrapped_y, wrapped_x):
    domain = Domain((24, 22, 32), (1., 1., 1.))
    building = PolygonBuilding.rectangle(x_start, 8.25, 5.5, 4.5, 4.75)
    profile = SensorProfile(2., 270., 10.)
    for legacy, mesh_type, expected in ((True, 0, 0), (False, 0, 1), (True, 1, 1)):
        case = WindCase(domain, profile, (building,), mesh_type=mesh_type,
                        native_staircase_metrics=True, native_netcdf_inputs=legacy)
        arrays = case.initialize_geometry().arrays
        assert arrays['icellflag'][1, wrapped_y, wrapped_x] == expected
        assert arrays['ibuilding_flag'][1, wrapped_y, wrapped_x] == (-1 if expected else 0)


@pytest.mark.parametrize('mode,flag_count', [(1, 53), (2, 66)])
def test_native_rooftop_uses_geometry_bounds_on_anisotropic_grid(mode, flag_count):
    domain = Domain((36, 34, 32), (1., 1.5, 1.))
    building = PolygonBuilding.rectangle(11.25, 14.25, 7.5, 6.5, 5.25, native_math=True)
    case = WindCase(domain, SensorProfile(2., 270., 15.), (building,), wake_mode=2, rooftop_mode=mode,
                    native_building_math=True, native_sensor_math=True, native_staircase_metrics=True)

    flags = case.initialize().arrays['icellflag']
    assert np.count_nonzero(flags == 10) == flag_count


def test_native_wake_can_finish_before_the_first_downwind_face():
    domain = Domain((32, 32, 24), (1., 1., 1.))
    ring = PolygonBuilding.rectangle(10, 10, 8, 6, 7).vertices
    building = PolygonBuilding((ring[0], *ring), 7.)
    profile = SensorProfile(2., 270., 10.)
    actual = WindCase(domain, profile, (building,), wake_mode=2, native_wake_math=True).initialize()
    baseline = WindCase(domain, profile, (building,), wake_mode=0).initialize()
    assert actual.building_context.wakes[0].length == 0.
    for key in ('u0', 'v0', 'w0', 'icellflag'):
        np.testing.assert_array_equal(actual.arrays[key], baseline.arrays[key])


def test_native_staircase_resumes_after_a_rounded_ring_closure():
    domain = Domain((28, 28, 16), (1., 1., 1.))
    ring = PolygonBuilding.rectangle(10.75, 10, 8, 6, 4).vertices
    building = PolygonBuilding((ring[0], *ring), 4.)
    native, modern = domain.workspace(), domain.workspace()
    building.flag(native, native_math=True)
    building.flag(modern)
    np.testing.assert_array_equal(native.arrays['icellflag'][1, 12, 9:12], [1, 0, 0])
    np.testing.assert_array_equal(modern.arrays['icellflag'][1, 12, 9:12], [1, 1, 0])


def test_native_building_roof_height_rounds_before_cell_classification():
    domain = Domain((16, 16, 12), (1., 1., 1.))
    building = PolygonBuilding.rectangle(5, 5, 4, 4, 2.29999998, base_height=.2)
    case = WindCase(domain, SensorProfile(2., 270., 8.), buildings=(building,), native_building_math=True)
    workspace = case.initialize_geometry()
    roof = int(np.flatnonzero(domain.coordinates['z'] == 2.5)[0])
    assert np.count_nonzero(workspace.arrays['icellflag'][roof] == 0) == 16
    assert np.count_nonzero(workspace.arrays['icellflag'][roof+1] == 0) == 0


def test_native_polygon_edge_rounds_before_ray_crossing_comparison():
    ring = ((5.278073, 5.1207414), (6.1123905, 6.546534), (4., 6.1), (5.278073, 5.1207414))
    vertices = tuple(map(tuple, np.asarray(ring, dtype=np.float32)))
    building = PolygonBuilding(vertices, 3.)
    domain = Domain((16, 16, 12), (1., 1., 1.))
    workspace = domain.workspace()
    building.flag(workspace, native_math=True)
    np.testing.assert_array_equal(workspace.arrays['icellflag'][1:4, 5, 5], [1, 1, 1])


def test_stretched_domain_and_affine_velocity_gradient():
    domain = Domain((8, 9, 5), (2., 3., 1.), (1., 2., 4., 8., 16.))
    workspace = domain.workspace()
    np.testing.assert_array_equal(domain.dz, [1, 1, 2, 4, 8, 16])
    c = domain.coordinates
    expected = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
    velocity = {}
    for component, key in enumerate(['u', 'v', 'w']):
        # Use component-specific staggering for affine fields.
        x = c['x_face'] if component == 0 else np.r_[c['x'], c['x'][-1]+2]
        y = c['y_face'] if component == 1 else np.r_[c['y'], c['y'][-1]+3]
        z = c['z_face'] if component == 2 else np.r_[c['z'], c['z'][-1]+domain.dz[-1]]
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        velocity[key] = expected[component, 0]*xx+expected[component, 1]*yy+expected[component, 2]*zz+1
    gradient = LocalTurbulence(terrain_wall=0).velocity_gradient(workspace, velocity)
    np.testing.assert_allclose(gradient[domain.interior], np.broadcast_to(expected, gradient[domain.interior].shape), atol=1e-12)


def test_no_native_runtime_wind_and_netcdf_roundtrip(tmp_path):
    case = WindCase(Domain((8, 8, 20), (1., 1., 1.)), SensorProfile(2., 225., 15.),
                    turbulence=LocalTurbulence(background=.3))
    result = case.run()
    assert result['solver'].converged
    path = tmp_path/'한글'/'wind.nc'
    write_netcdf(path, result['workspace'].arrays, result['workspace'].coordinates)
    restored = read_wind_workspace(path)
    for key, value in result['workspace'].arrays.items():
        np.testing.assert_array_equal(restored.arrays[key], value)
    assert np.isfinite(result['turbulence']['CoEps']).all()


def test_xml_rejects_unported_features_before_computation(tmp_path):
    path = ROOT/'data/cases/block_west/winds.xml'
    if not path.exists():
        pytest.skip('Local upstream input fixture unavailable')
    root = ET.parse(path)
    root.getroot().find('turbParams/method').text = '5'
    changed = tmp_path/'method5.xml'; root.write(changed)
    with pytest.raises(UnsupportedFeatures, match='turbParams/method=5'):
        read_wind_case(changed)
    root = ET.parse(path)
    ET.SubElement(root.getroot(), 'unimplementedTerrain').text = 'data.tif'
    changed = tmp_path/'unknown.xml'; root.write(changed)
    with pytest.raises(UnsupportedFeatures, match='unimplementedTerrain'):
        read_wind_case(changed)


def test_wall_does_not_allow_flow_through_ground():
    ws = Domain((8, 8, 5), (1., 1., 1.)).workspace()
    ws.arrays['w0'].fill(2.)
    zero_solid_faces(ws); wall_stencil(ws)
    assert np.all(ws.arrays['w0'][1, 1:-1, 1:-1] == 0)
    assert np.all(ws.arrays['n'][1] == 0)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_profile_reference_and_direction(kind):
    profile = SensorProfile(3., 180., 15., kind=kind)
    u, v = profile.evaluate([15.])
    assert abs(u[0]) < 1e-14
    assert v[0] == pytest.approx(3.)


def test_native_zero_height_log_profile_uses_free_convection_limit():
    domain = Domain((14, 16, 30), (1., 1., 1.))
    log = SensorProfile(2., 317., 0., kind=1)
    uniform = SensorProfile(2., 317., 0., kind=0)
    actual, expected = (WindCase(domain, profile, native_sensor_math=True).initialize()
                        for profile in (log, uniform))
    for name in ('u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual.arrays[name], expected.arrays[name])
    with pytest.raises(ValueError):
        log.evaluate([1.])


def test_native_upwind_regressions():
    import json
    from plumeswift.upwind import upwind_cavity
    references = ROOT/'results/python_port/upwind'
    if not (references/'comparison.json').exists():
        pytest.skip('Run scripts/check_python_upwind.py --native for reference fixtures')
    for entry in json.loads((references/'comparison.json').read_text()):
        name = entry['case']
        case = read_wind_case(references/name/'winds.xml', compute_turbulence=False)
        workspace = case.initialize()
        native = read_wind_workspace(references/name/'native_windsWk.nc')
        for key in ['u0', 'v0', 'w0']:
            np.testing.assert_allclose(workspace.arrays[key], native.arrays[key], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(workspace.arrays['icellflag'], native.arrays['icellflag'])


@pytest.mark.parametrize('family,tolerance', [('meteorology', 2e-6), ('buildings', 2e-5), ('streets', 1e-5), ('streets_modified', 1e-5)])
def test_native_initialization_regressions(family, tolerance):
    import json
    references = ROOT/'results/python_port'/family
    if not (references/'comparison.json').exists():
        pytest.skip(f'Run scripts/check_python_{family}.py --native for fixtures')
    for entry in json.loads((references/'comparison.json').read_text()):
        path = references/entry['case']
        workspace = read_wind_case(path/'winds.xml', compute_turbulence=False).initialize()
        native = read_wind_workspace(path/'native_windsWk.nc')
        for key in ['u0', 'v0', 'w0']:
            np.testing.assert_allclose(workspace.arrays[key], native.arrays[key], atol=tolerance, rtol=0, err_msg=entry['case']+'/'+key)
        np.testing.assert_array_equal(workspace.arrays['icellflag'], native.arrays['icellflag'])


def test_native_local_turbulence_construction():
    from plumeswift.io import read_turbulence
    reference = ROOT/'results/qes/empty_west_s3'
    if not (reference/'run_turbOut.nc').exists():
        pytest.skip('Archived native local turbulence fixture unavailable')
    workspace = read_wind_workspace(reference/'run_windsWk.nc')
    generated, metadata = LocalTurbulence(sigma=(2.5, 1.8, 1.3), background=.3).build(
        workspace, {k: workspace.arrays[k] for k in ['u', 'v', 'w']})
    native = read_turbulence(reference/'run_turbOut.nc')
    for key in ['txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z']:
        np.testing.assert_allclose(generated[key], native[key], atol=5e-8, rtol=0, err_msg=key)


def test_zero_turbulence_bound_leaves_background_normal_stress():
    domain = Domain((10, 9, 25), (1., 1., 1.))
    profile = SensorProfile(2., 236., 10.)
    case = WindCase(domain, profile, turbulence=LocalTurbulence(upper_bound=0., background=.3))

    fields = case.run()['turbulence']
    assert np.any(fields['txx'] > 0)
    for name in ('txx', 'tyy', 'tzz'):
        np.testing.assert_array_equal(fields[name][fields['txx'] > 0], .3**2)
    for name in ('tke', 'CoEps', 'txy', 'txz', 'tyz'):
        assert not np.any(fields[name])
    with pytest.raises(ValueError, match='Invalid turbulence bound'):
        LocalTurbulence(upper_bound=-1.)


@pytest.mark.parametrize('name', ['empty', 'two_blocks'])
def test_native_serial_mixing_length(name):
    from plumeswift.mixing_length import SerialMixing
    folder = ROOT/'results/python_port/mixing_length'/name
    if not (folder/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_mixing_length.py --native')
    case = read_wind_case(folder/'winds.xml', compute_turbulence=False)
    workspace = case.domain.workspace()
    for number, building in enumerate(case.buildings):
        building.flag(workspace, number)
    generated = SerialMixing()(workspace)
    native = read_wind_workspace(folder/'native_windsWk.nc')
    np.testing.assert_allclose(generated, native.arrays['mixlength'], rtol=0, atol=2e-6)


def test_custom_mixing_file_and_native_method1_integration(tmp_path):
    from plumeswift.mixing_length import NetCDFMixing
    case = WindCase(Domain((8, 8, 20), (1., 1., 1.)), SensorProfile(2., 270., 15.),
                    turbulence=LocalTurbulence(background=.3), mixing_method=1,
                    mixing_file=str(tmp_path/'혼합길이.nc'), mixing_variable='custom_distance', save_mixing=True)
    result = case.run()
    restored = NetCDFMixing(case.mixing_file, 'custom_distance')(result['workspace'])
    np.testing.assert_array_equal(restored, result['workspace'].arrays['mixlength'])
    case.mixing_method = 4; case.save_mixing = False
    loaded = case.run()
    for key in ['txx', 'tyy', 'tzz', 'CoEps', 'div_tau_x']:
        np.testing.assert_array_equal(loaded['turbulence'][key], result['turbulence'][key])


def test_serial_mixing_stops_positive_wall_propagation_at_grid_edge():
    from plumeswift.mixing_length import SerialMixing

    workspace = Domain((12, 11, 30), (1., 1., 1.)).workspace()
    building = PolygonBuilding.rectangle(6., 5., 3., 3., 6., native_math=True)
    building.flag(workspace, 0)

    mixing = SerialMixing(native_math=True)(workspace)
    np.testing.assert_array_equal(mixing[6, 6, 9:], [.5, 1.5, 0.])
    np.testing.assert_array_equal(mixing[6, 8:, 7], [.5, 1.5, 0.])


def test_native_serial_mixing_keeps_decimal_grid_transition_and_scaling():
    from plumeswift.numerics.poisson import NativeWindPolicy, make_poisson

    domain = Domain((24, 24, 60), (1., 1., .3))
    building = PolygonBuilding.rectangle(8., 8., 4., 4., 2.2, native_math=True)
    solver = make_poisson('cpu-native-rb', NativeWindPolicy(max_iterations=8))
    case = WindCase(domain, SensorProfile(2., 270., 10.), buildings=(building,),
                    turbulence=LocalTurbulence(native_math=True), poisson=solver,
                    mixing_method=1, native_sensor_math=True, native_building_math=True,
                    native_staircase_metrics=True)
    result = case.run()
    distance = result['workspace'].arrays['mixlength']
    np.testing.assert_array_equal(distance[19:23, 8, 8], [3.450000762939453, 4.450001001358032,
                                                        5.450001239776611, 6.450001239776611])
    scaled = result['turbulence']['Lm'][19:23, 8, 8]
    np.testing.assert_array_equal(scaled.view(np.uint32), [1068540890, 1071896334, 1074496801, 1076174523])


@pytest.mark.parametrize('padded', [False, True])
def test_legacy_mixing_length_reads_stored_netcdf_hyperslab(tmp_path, padded):
    from netCDF4 import Dataset
    from plumeswift.io import _staged_netcdf_path

    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '8 9 20'
    turbulence = tree.find('turbParams')
    turbulence.find('method').text = '4'
    ET.SubElement(turbulence, 'LMfile').text = 'prescribed.nc'
    ET.SubElement(turbulence, 'varname').text = 'distances'
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    shape = (21, 9, 8)
    dims = ('z', 'y', 'x')
    values = (4+np.indices(shape).sum(axis=0)%7).astype(np.int16)
    with _staged_netcdf_path(tmp_path/'prescribed.nc') as temporary, Dataset(temporary, 'w') as nc:
        for axis, size in zip(dims, shape, strict=True):
            nc.createDimension(axis, size)
        if padded:
            dims = ('z_padded', 'y_padded', 'x_padded')
            for axis, size in zip(dims, shape, strict=True):
                nc.createDimension(axis, size+1)
        variable = nc.createVariable('distances', 'i2', dims)
        variable.set_auto_maskandscale(False)
        variable.scale_factor = .25
        variable.add_offset = 1.5
        if padded:
            variable[:] = 30000
        variable[:shape[0], :shape[1], :shape[2]] = values

    legacy = read_wind_case(xml, legacy_turbulence_xml=True).initialize()
    np.testing.assert_array_equal(legacy.arrays['mixlength'], values)
    modern = read_wind_case(xml)
    if padded:
        with pytest.raises(ValueError, match='static'):
            modern.initialize()
    else:
        np.testing.assert_array_equal(modern.initialize().arrays['mixlength'], .25*values+1.5)
