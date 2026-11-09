import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.canopy_cionco import apply_cionco
from plumeswift.domain import Domain
from plumeswift.geometry import PolygonBuilding
from plumeswift.io import read_wind_workspace
from plumeswift.meteorology import SensorProfile
from plumeswift.vegetation import CanopyState
from plumeswift.wind_case import WindCase, read_wind_case


ROOT = Path(__file__).resolve().parents[2]


def test_qeswinds_cpu_serial_selects_the_non_openmp_solver(tmp_path):
    native = ROOT/'results/python_port/wind_cli_serial/native_windsWk.nc'
    if not native.is_file():
        native = ROOT/'results/python_port/native_cpu_serial/standalone_windsWk.nc'
    fixture = ROOT/'results/python_port/coupled_products/duplicate_outflow/reference/winds.xml'
    xml = fixture if native.is_file() and fixture.is_file() else ROOT/'data/cases/empty_west/winds.xml'
    basename = tmp_path/'serial'
    command = [sys.executable, '-m', 'plumeswift', 'qesWinds', '-q', str(xml), '--cpu-serial']
    result = subprocess.run(command+['-w', '-o', str(basename)], check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout)
    assert report['backend'] == 'cpu-native-serial'
    assert report['solvetype'] == 1
    if xml == fixture:
        assert [step['solver']['iterations'] for step in report['steps']] == [3000, 1355, 1355, 3000, 3000]
        for index in range(len(report['steps'])):
            actual = read_wind_workspace(basename.with_name('serial_windsWk.nc'), step=index).arrays
            expected = read_wind_workspace(native, step=index).arrays
            for name in ('u', 'v', 'w', 'icellflag'):
                np.testing.assert_array_equal(actual[name], expected[name])

    invalid = subprocess.run(command+['-s', '3'], capture_output=True, text=True, timeout=60)
    assert invalid.returncode == 2
    assert '--cpu-serial requires --solvetype 1' in invalid.stderr


def test_domain_rotation_is_recorded_and_preserves_native_wind(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '12 12 18'
    node = tree.find('simulationParameters/domainRotation')
    path = tmp_path/'wind.xml'
    tree.write(path)
    baseline = read_wind_case(path, compute_turbulence=False).run()['workspace'].arrays
    node.text = '110.5'
    tree.write(path)
    case = read_wind_case(path, compute_turbulence=False)
    assert case.manifest()['domain_rotation_deg'] == 110.5
    arrays = case.run()['workspace'].arrays
    for key in ('u0', 'v0', 'w0', *'uvw', 'icellflag'):
        np.testing.assert_array_equal(baseline[key], arrays[key])
    node.text = 'nan'
    tree.write(path)
    with pytest.raises(ValueError, match='domainRotation must be finite'):
        read_wind_case(path)


def test_native_building_closures_order_rounded_roof_height_ties():
    buildings = (
        PolygonBuilding.rectangle(9, 9, 5, 5, 3.4),
        PolygonBuilding.rectangle(16, 11, 5, 5, 3.2, base_height=.2),
    )
    case = WindCase(Domain((34, 30, 24), (1., 1., 1.)), SensorProfile(2, 225, 15),
                    buildings=buildings)
    modern = case.initialize()
    native = replace(case, native_building_math=True).initialize()
    assert tuple(index for index, _ in modern.building_context.ordered_buildings) == (0, 1)
    assert tuple(index for index, _ in native.building_context.ordered_buildings) == (1, 0)


@pytest.mark.parametrize('nx,ny', [(2, 7), (7, 2), (2, 2)])
def test_qeswinds_two_cell_horizontal_axis(tmp_path, nx, ny):
    from netCDF4 import Dataset

    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = f'{nx} {ny} 22'
    tree.find('metParams/sensor/site_xcoord').text = '1'
    tree.find('metParams/sensor/site_ycoord').text = '1'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    basename = tmp_path/'thin'
    result = subprocess.run([sys.executable, '-m', 'plumeswift', 'qesWinds', '-q', str(xml), '-w', '-t',
                             '-o', str(basename)], check=True, capture_output=True, text=True, timeout=20)
    step, = json.loads(result.stdout)['steps']
    assert step['solver']['converged'] and step['solver']['iterations'] == 1
    assert step['solver']['residual_linf'] == step['solver']['update_linf'] == 0.
    fields = read_wind_workspace(tmp_path/'thin_windsWk.nc').arrays
    assert fields['icellflag'].shape == (23, ny, nx)
    assert np.any(fields['u0'] > 0)
    for component in 'uvw':
        np.testing.assert_array_equal(fields[component], fields[component+'0'])
    for suffix, name in (('windsOut', 'u'), ('turbOut', 'tke')):
        with Dataset(tmp_path/f'thin_{suffix}.nc') as nc:
            assert len(nc.dimensions['t']) == 1
            assert np.isfinite(nc[name][:]).all()
    modern = read_wind_case(xml, compute_turbulence=False).run()
    assert modern['solver'].converged and not np.any(modern['multiplier'])


def test_cionco_zero_attenuation_keeps_in_canopy_and_updates_exposed_faces():
    domain = Domain((7, 7, 12), (1., 1., 1.))
    workspace = domain.workspace()
    state = CanopyState(workspace)
    state.top[3, 3] = state.height[3, 3] = 4.
    state.top_index[3, 3] = 5
    state.roughness[3, 3], state.ustar[3, 3] = .03, .3
    workspace.arrays['u0'][:] = 5.
    workspace.arrays['v0'][:] = 2.
    apply_cionco(workspace, state)
    displacement = .99*4
    assert state.displacement[3, 3] == displacement
    u, v = (workspace.arrays[key] for key in ('u0', 'v0'))
    np.testing.assert_array_equal(u[1:5, 3, 3], np.full(4, 5.))
    factor = np.log((4.5-displacement)/.03)/np.log(4.5/.03)
    np.testing.assert_allclose([u[5, 3, 3], u[5, 3, 4], v[5, 3, 3], v[5, 4, 3]],
                               np.array([5., 5., 2., 2.])*factor)


def test_native_tree_wake_excludes_a_face_on_the_rounded_ellipse_boundary():
    from plumeswift.canopy_tree import _wake_ellipse

    # This u-face intersects the outer edge of the FlatTerrain_wIsoTree wake.
    along, side, vertical, height = 39.+2.**-18, -25.5, 0., 15.
    _, radius, diameter = _wake_ellipse(along, side, vertical, height, native_math=True)
    assert radius == .5*diameter == 25.5
    _, wider_radius, wider_diameter = _wake_ellipse(along, side, vertical, height, native_math=False)
    assert wider_radius < .5*wider_diameter
    _, inside_radius, inside_diameter = _wake_ellipse(along+1., side, vertical, height, native_math=True)
    assert inside_radius < .5*inside_diameter


def test_native_turbulence_building_height_order_keeps_reference_on_the_lower_face(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    simulation.find('domain').text = '24 24 24'
    simulation.find('maxIterations').text = '8'
    buildings = tree.find('buildingsParams')
    for flag in ('upwindCavityFlag', 'wakeFlag', 'rooftopFlag', 'sidewallFlag', 'streetCanyonFlag'):
        buildings.find(flag).text = '0'
    for index, height in enumerate((4.78, 4.01, 1.21)):
        building = ET.SubElement(buildings, 'rectangularBuilding')
        fields = dict(height=height, xStart=3+6*index, yStart=8, length=3, width=3,
                      baseHeight=0, buildingRotation=0)
        for name, field in fields.items():
            ET.SubElement(building, name).text = str(field)
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    case = read_wind_case(xml, poisson_backend='cpu-native-rb', native_sensor_math=True,
                          native_turbulence_math=True, native_building_math=True, legacy_turbulence_xml=True)
    result = replace(case, native_staircase_metrics=True).run()
    assert result['turbulence_metadata']['z_reference'] == 10.
