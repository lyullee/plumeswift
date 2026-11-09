from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from plumeswift.triangle_mesh import TriangleMesh
from plumeswift.wind_case import read_wind_case
from plumeswift.io import read_wind_workspace

ROOT = Path(__file__).resolve().parents[2]


def test_triangle_height_and_obj_preserve_discontinuous_vertices(tmp_path):
    # Coincident projected triangles with different heights must return the
    # higher hit, rather than merging vertices into a smoother surface.
    lower = np.array([[0., 0., 1.], [1., 0., 2.], [0., 1., 3.]])
    mesh = TriangleMesh([lower, lower+[0., 0., 5.]])
    got = mesh.height([.2, 0., -.1], [.3, 0., 0.])
    np.testing.assert_allclose(got, [6.8, 6., -1.])
    path = tmp_path/'terrain.obj'; mesh.write_obj(path)
    lines = path.read_text().splitlines()
    assert sum(line.startswith('v ') for line in lines) == 6
    assert sum(line.startswith('f ') for line in lines) == 2
    np.testing.assert_array_equal(TriangleMesh.read_obj(path).triangles, mesh.triangles)


def test_indexed_obj_round_trip_keeps_faces_and_shared_vertices(tmp_path):
    triangles = np.array([((0, 0, 0), (1, 0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 1), (0, 1, 0))])
    path = tmp_path/'indexed.obj'
    TriangleMesh(triangles).write_obj(path, indexed=True)
    lines = path.read_text().splitlines()
    assert sum(line.startswith('v ') for line in lines) == 4
    assert lines[-1] == 'f 1 3 4'
    np.testing.assert_array_equal(TriangleMesh.read_obj(path).triangles, triangles)


def test_native_triangle_height_uses_float32_intermediates():
    mesh = TriangleMesh([[[0.12, 0.23, 1.42], [8.07, 0.35, 2.11], [0.36, 7.31, 3.57]],
                         [[0.12, 0.23, 4.29], [8.07, 0.35, 7.11], [0.36, 7.31, 5.43]]])
    # Bit patterns from BVH::heightToTri in scripts/check_python_mesh.py.
    got = mesh.height([3.6, 4.1], [0.7, 2.4], native_math=True)
    np.testing.assert_array_equal(got.view(np.uint32), [1085457871, 1086358916])


def test_native_utm_crop_rounds_projected_bounds_before_choosing_pixels(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain
    from plumeswift.terrain import DEMTerrain

    rows, columns = np.indices((80, 80))
    elevation = (100+.25*columns+.0625*rows+.125*(columns % 4)).astype(np.float32)
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=80, height=80, count=1, dtype='float32',
                       transform=from_origin(500000.26, 4850040.26, .5, .5), crs='EPSG:32613') as raster:
        raster.write(elevation, 1)
    terrain = DEMTerrain(str(path), origin_flag=1, utm_x=500002.25, utm_y=4850002.)
    domain = Domain((24, 24, 24), (.5, .5, 1.))
    with rasterio.open(path) as raster:
        cropped, dx, dy = terrain._read_native_grid(raster, domain)
    np.testing.assert_array_equal(cropped, elevation[53:77, 4:28])
    assert (dx, dy) == (.5, .5)
    assert terrain.mesh(domain).height(.25, .25, native_math=True) == .1875


def test_native_decimal_dem_mesh_step_and_sampling_match_original(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain
    from plumeswift.terrain import DEMTerrain, apply_stairstep_terrain

    rows, columns = np.indices((32, 32))
    elevation = (100+.125*columns+.0625*rows+.375*(columns % 5)).astype(np.float32)
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=32, height=32, count=1, dtype='float32',
                       transform=from_origin(1000., 2000., .3, .3), crs='EPSG:32613') as raster:
        raster.write(elevation, 1)
    domain = Domain((24, 24, 24), (.3, .3, 1.))
    workspace = domain.workspace()
    apply_stairstep_terrain(workspace, DEMTerrain(str(path)).mesh(domain), native_sensor_math=True)
    np.testing.assert_array_equal(workspace.arrays['terrain'][0, :4].view(np.uint32),
                                  [1048576000, 1060634625, 1067188224, 1071382528])
    assert np.count_nonzero(workspace.arrays['icellflag'] == 2) == 1824


def test_scaled_dem_rounds_relative_height_before_applying_band_scale(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain, HorizontalHalo
    from plumeswift.terrain import DEMTerrain, apply_stairstep_terrain

    raw = np.full((30, 30), .2, dtype=np.float32)
    raw[6:19, 6:19] = 3.2
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=30, height=30, count=1, dtype='float32',
                       transform=from_origin(1000, 2030, 1, 1), nodata=-9999.) as raster:
        raster.scales = (.3,)
        raster.write(raw, 1)
    domain = Domain((24, 24, 36), (1., 1., .6))
    workspace = domain.workspace()
    terrain = DEMTerrain(str(path))
    apply_stairstep_terrain(workspace, terrain.mesh(domain), native_sensor_math=True)
    assert workspace.arrays['terrain'][15, 10] == np.float32(.9)
    assert workspace.arrays['icellflag'][2, 15, 10] == 1
    _, corners = terrain.cut_geometry(domain, HorizontalHalo())
    np.testing.assert_array_equal(corners[15, 10], np.full(4, .9, dtype=np.float32))


def test_dem_nan_nodata_preserves_finite_elevations(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain, HorizontalHalo
    from plumeswift.terrain import DEMTerrain, apply_stairstep_terrain

    raw = np.full((30, 30), .2, dtype=np.float32)
    raw[6:19, 6:19] = 3.2
    raw[15, 10] = np.nan
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=30, height=30, count=1, dtype='float32',
                       transform=from_origin(1000, 2030, 1, 1), nodata=np.nan) as raster:
        raster.write(raw, 1)
    domain = Domain((24, 24, 36), (1., 1., .6))
    terrain = DEMTerrain(str(path))
    mesh, corners = terrain.cut_geometry(domain, HorizontalHalo())
    assert mesh.height(10, 14, native_math=True) == 0.
    assert mesh.height(11, 14, native_math=True) == 3.
    np.testing.assert_array_equal(corners[14, 10], [0., 3., 3., 3.])
    workspace = domain.workspace()
    apply_stairstep_terrain(workspace, mesh, native_sensor_math=True)
    assert np.count_nonzero(workspace.arrays['icellflag'][1:] == 2) == 866


@pytest.mark.parametrize('nodata, start', [(None, 1), (0., 0)])
def test_dem_nodata_absence_preserves_positive_height_below_comparison_epsilon(tmp_path, nodata, start):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain
    from plumeswift.terrain import DEMTerrain, apply_stairstep_terrain

    raw = np.zeros((12, 12), dtype=np.float32)
    raw[3:8, 3:8] = 1e-8
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=12, height=12, count=1, dtype='float32',
                       transform=from_origin(1000, 2012, 1, 1), nodata=nodata) as raster:
        raster.write(raw, 1)

    domain = Domain((12, 12, 20), (1., 1., 1.))
    workspace = domain.workspace()
    apply_stairstep_terrain(workspace, DEMTerrain(str(path)).mesh(domain), native_sensor_math=True)
    assert workspace.arrays['terrain_face_id'][5, 5] == start
    assert workspace.arrays['icellflag'][1, 5, 5] == 1


def test_legacy_cuda_geometry_uses_native_terrain_height_comparison(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    from plumeswift.domain import Domain, native_vertical_coordinates
    from plumeswift.meteorology import SensorProfile
    from plumeswift.terrain import DEMTerrain
    from plumeswift.wind_case import WindCase

    domain = Domain((24, 24, 28), (1., 1., 1.3))
    z, _ = native_vertical_coordinates(domain.dz)
    assert z[2] < domain.coordinates['z'][2]
    height = np.zeros((30, 30), dtype=np.float32)
    height[6:19, 6:19] = z[2]
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=30, height=30, count=1, dtype='float32',
                       transform=from_origin(1000, 2030, 1, 1)) as raster:
        raster.write(height, 1)

    case = WindCase(domain, SensorProfile(2., 270., 15.), terrain=DEMTerrain(str(path)),
                    native_staircase_metrics=True, meteorology_backend='cuda')
    arrays = case.initialize_geometry().arrays
    assert arrays['terrain'][15, 10] == z[2]
    assert arrays['icellflag'][2, 15, 10] == 2


@pytest.mark.parametrize('name', ['ramp', 'descending', 'two_stations', 'building', 'utm_crop'])
@pytest.mark.parametrize('variant', ['native-cpu', 'native-gpu'])
def test_native_dem_initialization(name, variant):
    pytest.importorskip('rasterio')
    folder = ROOT/'results/python_port/terrain'/(name+'_cpu' if variant == 'native-cpu' else name)
    if not (folder/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_terrain.py --native for both solvers')
    case = read_wind_case(folder/'winds.xml', compute_turbulence=False)
    case.meteorology_variant = variant
    generated = case.initialize()
    native = read_wind_workspace(folder/'native_windsWk.nc')
    for field in ['terrain', 'icellflag', 'e', 'f', 'g', 'h', 'm', 'n']:
        np.testing.assert_array_equal(generated.arrays[field], native.arrays[field], err_msg=field)
    for field in ['u0', 'v0', 'w0']:
        got, expected = generated.arrays[field], native.arrays[field]
        if name == 'two_stations' and variant == 'native-cpu':
            # Native CPU reads domain.z[nz-1] one past its allocation at the
            # last top face. This is a defined-region comparison, not parity
            # for that native memory read. The report retains full differences.
            got, expected = got[:-1], expected[:-1]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6, err_msg=field)


def test_measured_profile_is_immutable_and_segment_state_is_local():
    from plumeswift.measured_profile import MeasuredProfile
    model = MeasuredProfile((3., 12., 22.), (1., 3., 2.), (350., 10., 320.))
    grid = np.array([.5, 3.5, 12.5, 22.5, 31.5])
    first = model.evaluate(grid)
    second = model.evaluate(grid)
    np.testing.assert_array_equal(first, second)
    assert model.directions == (350., 10., 320.)


@pytest.mark.parametrize('ground_speed', [0., 1.])
def test_wind_measured_profile_accepts_zero_height_anchor(tmp_path, ground_speed):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '8 8 10'
    series = tree.find('metParams/sensor/timeSeries')
    series.find('boundaryLayerFlag').text = '4'
    for key, values in {'height': (0., 4., 8.), 'speed': (ground_speed, 4., 8.),
                        'direction': (270., 270., 270.)}.items():
        for element in series.findall(key):
            series.remove(element)
        for value in values:
            ET.SubElement(series, key).text = str(value)
    path = tmp_path/'winds.xml'
    tree.write(path)
    for native_math in (False, True):
        case = read_wind_case(path, compute_turbulence=False, native_sensor_math=native_math)
        workspace = case.initialize()
        heights = workspace.coordinates['z'][1:5]
        expected = ground_speed + (1-3*ground_speed/8)*heights + ground_speed*heights**2/32
        np.testing.assert_allclose(workspace.arrays['u0'][1:5, 4, 4], expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(workspace.arrays['v0'], 0.)


def test_height_array_triangle_constructor_uses_native_diagonal_and_origin():
    from plumeswift.domain import Domain
    from plumeswift.terrain import GridTerrain
    yy, xx = np.indices((4, 4))
    terrain = GridTerrain(100+2*xx+3*yy, halo_x=5., halo_y=7.)
    mesh = terrain.mesh(Domain((4, 4, 20), (1., 1., 1.)))
    np.testing.assert_array_equal(mesh.triangles[0], [[0, 0, 0], [1, 0, 2], [0, 1, 3]])
    np.testing.assert_array_equal(mesh.triangles[1], [[0, 1, 3], [1, 0, 2], [1, 1, 5]])
    np.testing.assert_allclose(mesh.height([.5, 2.5], [.5, 2.5]), [2.5, 12.5])


def test_height_array_cut_cells_reach_wind_solver_with_asymmetric_halo():
    from plumeswift.domain import Domain, HorizontalHalo
    from plumeswift.meteorology import SensorProfile, Station
    from plumeswift.terrain import GridTerrain
    from plumeswift.wind_case import WindCase

    y, x = np.indices((6, 7))
    terrain = GridTerrain(10.+.25*x+.125*y)
    halo = HorizontalHalo(2., 4.)
    domain = Domain((9, 10, 8), (2., 2., 1.), (.5, .75, 1., 1., 1., 1., 1., 1.))
    mesh, corners = terrain.cut_geometry(domain, halo)
    np.testing.assert_array_equal(corners[2, 1], [0., .125, .375, .25])
    np.testing.assert_array_equal(corners[:2], 0.)
    np.testing.assert_array_equal(corners[:, 0], 0.)
    assert mesh.height(1., 1.) == .1875
    profile = SensorProfile(2., 270., 4., kind=0)
    result = WindCase(domain, profile, stations=(Station(6., 7., profile),),
                      terrain=terrain, halo=halo, mesh_type=1).run()
    fields = result['workspace'].arrays
    cut = fields['icellflag'] == 8
    assert cut.any() and result['solver'].converged
    assert not cut[:, :2].any() and not cut[:, :, 0].any()
    fraction = fields['terrain_volume_frac'][cut]
    assert np.all((fraction > .1) & (fraction <= 1.)) and np.any(fraction < 1.)
    assert np.isfinite(fields['u']).all()


@pytest.mark.parametrize('name', ['ramp', 'descending', 'building', 'utm_crop'])
def test_native_full_terrain_wind_and_turbulence(name):
    pytest.importorskip('rasterio')
    from plumeswift.io import read_turbulence
    folder = ROOT/'results/python_port/terrain'/(name+'_cpu_full')
    if not (folder/'native_turbOut.nc').exists():
        pytest.skip('Run scripts/check_python_terrain.py --native --full --solver 1')
    result = read_wind_case(folder/'winds.xml').run()
    native = read_wind_workspace(folder/'native_windsWk.nc')
    native_turb = read_turbulence(folder/'native_turbOut.nc')
    for key in ['u', 'v', 'w']:
        np.testing.assert_allclose(result['workspace'].arrays[key], native.arrays[key], atol=3e-6, rtol=0, err_msg=name+'/'+key)
    for key in ['txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z']:
        np.testing.assert_allclose(result['turbulence'][key], native_turb[key], atol=3e-5, rtol=0, err_msg=name+'/'+key)


@pytest.mark.parametrize('name, offsets, indices', [
    ('dem_integer', (2., 3.), (2, 3)),
    ('dem_fractional', (2.75, 1.25), (2, 1)),
    ('flat_objects', (1.25, 2.75), (1, 2)),
    ('dem_one_axis', (0., 2.), (0, 2)),
])
def test_native_wind_halo(name, offsets, indices):
    pytest.importorskip('rasterio')
    folder = ROOT/'results/python_port/halo'/name
    if not (folder/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_halo.py --native')
    case = read_wind_case(folder/'winds.xml', compute_turbulence=False)
    assert case.domain.cells == (24, 24, 24)
    assert case.halo.cells(case.domain.spacing) == indices
    assert (case.stations[0].x, case.stations[0].y) == tuple(5+offset for offset in offsets)
    assert (case.stations[1].x, case.stations[1].y) == (17.25, 15.75)
    actual = case.run()['workspace'].arrays
    expected = read_wind_workspace(folder/'native_windsWk.nc').arrays
    for key in ('terrain', 'icellflag', 'e', 'f', 'g', 'h', 'm', 'n'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[key][:-1], expected[key][:-1], atol=1e-4, rtol=0)
