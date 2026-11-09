from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.io import _open_netcdf, read_wind_workspace
from plumeswift.wind_case import read_wind_case


FIXTURES = Path(__file__).resolve().parents[2]/'results/python_port/nonlocal'


def test_legacy_wind_series_preserves_previous_wake_mixing(tmp_path):
    tree = ET.parse(FIXTURES.parents[2]/'data/cases/block_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for name, text in {'domain': '44 36 25', 'maxIterations': '40', 'totalTimeIncrements': '2'}.items():
        simulation.find(name).text = text
    tree.find('turbParams/nonLocalMixing').text = 'true'
    buildings = tree.find('buildingsParams')
    for name in ('rooftopFlag', 'upwindCavityFlag', 'streetCanyonFlag', 'sidewallFlag'):
        buildings.find(name).text = '0'
    block = buildings.find('rectangularBuilding')
    for name, text in {'height': '4.2', 'xStart': '10.4', 'yStart': '12.4',
                       'width': '6.4', 'length': '5.5'}.items():
        block.find(name).text = text
    sensor = tree.find('metParams/sensor')
    later = deepcopy(sensor.find('timeSeries'))
    later.find('timeStamp').text = '2020-01-01T01:00:30'
    later.find('direction').text = '249'
    sensor.append(later)
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path, poisson_backend='cpu-native-rb', native_sensor_math=True,
                          native_turbulence_math=True, legacy_turbulence_xml=True)
    case.native_staircase_metrics = True
    first, second = [result['turbulence'] for _, result in case.iter_results()]
    independent = list(case.iter_steps())[1].run()['turbulence']
    retained = np.isin(first['iturbflag'], (11, 12)) & ~np.isin(independent['iturbflag'], (11, 12))
    assert np.count_nonzero(retained) == 230
    for field in ('iturbflag', 'Lm'):
        np.testing.assert_array_equal(second[field][retained], first[field][retained])


@pytest.mark.parametrize('case, horizontal, vertical', [
    ('normal', 391, 0), ('oblique', 438, 0), ('canyon', 625, 0),
    ('wake_off', 0, 0), ('wide', 543, 3),
])
def test_nonlocal_building_mixing_matches_cpu(case, horizontal, vertical):
    folder = FIXTURES/case
    reference = folder/'native_turbOut.nc'
    if not reference.exists():
        pytest.skip('Run scripts/check_python_nonlocal.py --native')
    actual = read_wind_case(folder/'winds.xml').run()['turbulence']
    flags = actual['iturbflag']
    assert np.count_nonzero(flags == 11) == horizontal
    assert np.count_nonzero(flags == 12) == vertical
    with _open_netcdf(reference) as fields:
        np.testing.assert_array_equal(flags, fields['iturbflag'][0])
        for name in ('tke', 'CoEps', 'txx', 'tyy', 'tzz', 'txz', 'tyz', 'txy',
                     'div_tau_x', 'div_tau_y', 'div_tau_z'):
            np.testing.assert_allclose(actual[name], fields[name][0], atol=3e-5, rtol=0)
        np.testing.assert_allclose(actual['Lm'], fields['L'][0], atol=3e-6, rtol=0)


@pytest.mark.parametrize('name', ['normal', 'canyon_roof', 'canyon_shifted_frame'])
def test_nonlocal_native_math_reproduces_independent_wind_and_turbulence(name):
    folder = FIXTURES/name
    reference = folder/'native_turbOut.nc'
    if not reference.exists():
        pytest.skip('Run scripts/check_python_nonlocal.py --native --native-math')
    case = read_wind_case(folder/'winds.xml', poisson_backend='cpu-native-rb', native_sensor_math=True,
                          native_building_math=True, native_turbulence_math=True)
    result = case.run()
    workspace, actual = result['workspace'], result['turbulence']
    if name == 'canyon_shifted_frame':
        wake = workspace.building_context.wakes[1]
        np.testing.assert_allclose(np.degrees(wake.mixing_frame.direction-wake.frame.direction), 4.88, atol=.001)
    wind = read_wind_workspace(folder/'native_windsWk.nc').arrays
    for field in ('u0', 'v0', 'w0', 'u', 'v', 'w', 'icellflag'):
        np.testing.assert_array_equal(workspace.arrays[field], wind[field])
    with _open_netcdf(reference) as fields:
        np.testing.assert_array_equal(actual['iturbflag'], fields['iturbflag'][0])
        np.testing.assert_array_equal(actual['Lm'], fields['L'][0])
        for name in ('tke', 'CoEps', 'txx', 'tyy', 'tzz', 'txz', 'tyz', 'txy',
                     'div_tau_x', 'div_tau_y', 'div_tau_z'):
            np.testing.assert_array_equal(actual[name], fields[name][0])
