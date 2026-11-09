from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from plumeswift.domain import Domain
from plumeswift.io import read_turbulence,read_wind_workspace
from plumeswift.meteorology import SensorProfile
from plumeswift.vegetation import HomogeneousCanopy,Vegetation
from plumeswift.wind_case import WindCase,read_wind_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['rectangle','rotated','raised_polygon','overlap','bare_attenuation','terrain_building',
    'windbreak_no_wake','windbreak_normal','windbreak_oblique',
    'tree_no_wake','tree_wake','tree_raised_oblique','tree_shapefile'])
def test_native_vegetation_pipeline(name):
    folder = ROOT/'results/python_port/vegetation'/name
    if not (folder/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_vegetation.py --native for native fixture')
    if name == 'terrain_building':
        pytest.importorskip('rasterio')
    if name == 'tree_shapefile':
        pytest.importorskip('shapefile')
    result = read_wind_case(folder/'winds.xml').run()
    a = result['workspace'].arrays
    ref = read_wind_workspace(folder/'native_windsWk.nc').arrays
    for field in ['icellflag','e','f','g','h','m','n']:
        np.testing.assert_array_equal(a[field], ref[field])
    windbreak = name.startswith('windbreak_')
    tree = name.startswith('tree_')
    velocity_tolerance = 3e-6 if windbreak else 1.6e-4 if tree else 1.3e-4
    for field in ['u0','v0','w0','u','v','w']:
        np.testing.assert_allclose(a[field], ref[field], atol=velocity_tolerance, rtol=0)
    turb = read_turbulence(folder/'native_turbOut.nc')
    turbulence_tolerance = 1e-5 if windbreak else 1.7e-4 if tree else 1e-3
    for field in ['txx','txy','txz','tyy','tyz','tzz','CoEps','div_tau_x','div_tau_y','div_tau_z']:
        np.testing.assert_allclose(result['turbulence'][field], turb[field], atol=turbulence_tolerance, rtol=0)
    if windbreak:
        assert np.any(a['icellflag_initial'] == 24)
        assert np.any(a['icellflag'] == 25) == (name != 'windbreak_no_wake')
        assert not np.any(a['icellflag_initial'] == 25)
    elif tree:
        assert np.any(a['icellflag_initial'] == 22)
        assert np.any(a['icellflag'] == 23) == (name != 'tree_no_wake')
        assert not np.any(a['icellflag_initial'] == 23)
    else:
        assert np.any(a['icellflag'] == 20)


def test_canopy_initialization_is_repeatable_and_column_state_is_local():
    canopy = HomogeneousCanopy.rectangle(4, 4, 3, 3, 4.2, 2.1)
    case = WindCase(Domain((14, 14, 20),(1.,1.,1.)), SensorProfile(2,270,10), vegetation=Vegetation((canopy,)))
    first = case.initialize().arrays
    second = case.initialize().arrays
    np.testing.assert_array_equal(first['u0'], second['u0'])
    np.testing.assert_array_equal(first['canopy_roughness'], second['canopy_roughness'])
    assert first['canopy_top'] is not second['canopy_top']
    assert case.manifest()['vegetation']['canopies'][0]['attenuation'] == 2.1


def test_native_regression_for_overlapping_trees_on_sloped_terrain():
    folder = ROOT/'results/python_port/wind_mixed_canopy_terrain'
    if not (folder/'native_turbOut.nc').exists():
        pytest.skip('Native mixed canopy and terrain output is unavailable')
    pytest.importorskip('rasterio')
    case = read_wind_case(folder/'winds.xml', poisson_backend='cpu-native-rb', native_sensor_math=True,
                          native_building_math=True, native_turbulence_math=True, legacy_turbulence_xml=True)
    result = replace(case, native_staircase_metrics=True).run()
    wind = read_wind_workspace(folder/'native_windsWk.nc').arrays
    turbulence = read_turbulence(folder/'native_turbOut.nc')
    np.testing.assert_array_equal(result['workspace'].arrays['icellflag'], wind['icellflag'])
    np.testing.assert_allclose(result['workspace'].arrays['u0'], wind['u0'], atol=7e-6, rtol=0)
    np.testing.assert_allclose(result['turbulence']['txx'], turbulence['txx'], atol=7e-6, rtol=0)


def test_native_canopy_geometry_rounds_height_before_classification():
    domain = Domain((14, 14, 20), (1., 1., 1.))
    canopy = HomogeneousCanopy.rectangle(4, 4, 4, 4, 2.49999998, 2.4)
    case = WindCase(domain, SensorProfile(2, 270, 10), vegetation=Vegetation((canopy,)),
                    native_staircase_metrics=True)
    flags = case.initialize_geometry().arrays['icellflag']
    assert np.count_nonzero(flags[3] == 20) == 16
    assert not np.any(flags[4] == 20)


def test_native_canopy_geometry_preserves_custom_setup_signature():
    class CustomVegetation(Vegetation):
        def setup(self, workspace):
            workspace.arrays['custom_canopy'] = True
            return super().setup(workspace)

    case = WindCase(Domain((14, 14, 20), (1., 1., 1.)), SensorProfile(2, 270, 10),
                    vegetation=CustomVegetation(), native_staircase_metrics=True)
    assert case.initialize_geometry().arrays['custom_canopy']


def test_singular_canopy_regression_reports_failure():
    canopy = HomogeneousCanopy.rectangle(4, 4, 3, 3, 4.2, 2.1)
    case = WindCase(Domain((14, 14, 20),(1.,1.,1.)), SensorProfile(2,270,10,kind=0), vegetation=Vegetation((canopy,)))
    with pytest.raises(FloatingPointError, match='Singular native canopy regression'):
        case.initialize()
