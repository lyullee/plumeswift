from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.io import _open_netcdf, read_wind_workspace, write_netcdf
from plumeswift.wind_case import read_wind_case
from plumeswift.wind_replay import replay_wind
from plumeswift.wind_wall_log import WallLogCorrection


ROOT = Path(__file__).resolve().parents[2]


def test_axis_wall_tangent_log_profile():
    domain = Domain((6, 6, 6), (1., 1., 1.))
    workspace = domain.workspace()
    workspace.arrays['icellflag_initial'] = workspace.arrays['icellflag'].copy()
    velocity = {name: np.full(domain.face_shape, value) for name, value in zip('uvw', (3., 4., 2.))}
    result = WallLogCorrection(.01)(workspace, velocity)
    ratio = np.log(.5/.01)/np.log(1.5/.01)
    np.testing.assert_allclose([result[key][1, 3, 3] for key in 'uvw'], [3*ratio, 4*ratio, 0.], atol=5e-7, rtol=0)
    assert [result[key][2, 3, 3] for key in 'uvw'] == [3., 4., 2.]


def test_positive_grid_edge_has_no_wrapped_wall():
    domain = Domain((6, 6, 6), (1., 1., 1.))
    workspace = domain.workspace()
    flags = workspace.arrays['icellflag'].copy()
    flags[2, 0, 3] = 2  # the flattened +y neighbour of (1, 5, 3)
    workspace.arrays['icellflag_initial'] = flags
    velocity = {name: np.full(domain.face_shape, value) for name, value in zip('uvw', (3., 4., 2.))}
    result = WallLogCorrection(.01)(workspace, velocity)
    ratio = np.log(.5/.01)/np.log(1.5/.01)
    np.testing.assert_allclose([result[key][1, 5, 3] for key in 'uvw'], [3*ratio, 4*ratio, 0.], atol=5e-7, rtol=0)


@pytest.mark.parametrize('flag', [0, 1])
def test_qes_xml_post_projection_wall_log(tmp_path, flag):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    section = tree.find('simulationParameters')
    section.find('domain').text = '12 12 12'
    ET.SubElement(section, 'logLawFlag').text = str(flag)
    path = tmp_path/'winds.xml'
    tree.write(path)
    case = read_wind_case(path, compute_turbulence=False)
    assert (case.wall_log is not None) == bool(flag)
    original = case.initialize().arrays['u0']
    result = case.run()['workspace'].arrays
    if flag:
        np.testing.assert_array_equal(result['u0'], result['u'])
        assert result['u0'][1, 4, 4] != original[1, 4, 4]
    else:
        np.testing.assert_array_equal(result['u0'], original)


def test_cutcell_wall_replay_retains_geometry_and_initial_velocity(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('domain').text = '16 16 12'
    sim.find('meshTypeFlag').text = '1'
    ET.SubElement(sim, 'logLawFlag').text = '1'
    buildings = tree.find('buildingsParams')
    buildings.find('numBuildings').text = '1'
    building = ET.SubElement(buildings, 'rectangularBuilding')
    for name, value in dict(groupID=1, buildingType=1, height=4.3, baseHeight=0,
                            xStart=4.3, yStart=5.25, length=4.4, width=5.2, buildingRotation=0).items():
        ET.SubElement(building, name).text = str(value)
    config = tmp_path/'winds.xml'
    tree.write(config)

    case = read_wind_case(config, compute_turbulence=False)
    workspace = case.initialize()
    assert np.any(workspace.arrays['icellflag_initial'] == 7)
    source, output, repeated = (tmp_path/f'{name}.nc' for name in ('source', 'output', 'repeated'))
    write_netcdf(source, workspace.arrays, workspace.coordinates, static_fields=('center_id',))
    original = read_wind_workspace(source).arrays
    expected = case.solve(workspace)['workspace'].arrays
    options = dict(solver=case.poisson, wall_roughness=case.wall_log.roughness)
    assert replay_wind(source, output, **options)['wall_roughness_m'] == case.wall_log.roughness
    replay_wind(output, repeated, **options)
    actual = read_wind_workspace(output).arrays
    repeated_fields = read_wind_workspace(repeated).arrays
    for name in 'uvw':
        np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(repeated_fields[name], actual[name])
    for name in ('ni', 'nj', 'nk', 'wall_distance', 'center_id', 'icellflag_initial', 'u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual[name], original[name])
    with _open_netcdf(output) as result:
        assert result['center_id'].dimensions[0] != 't'
        assert result['ni'].dimensions[0] == 't'

    original.pop('wall_distance')
    write_netcdf(source, original, workspace.coordinates)
    invalid_output = tmp_path/'invalid.nc'
    with pytest.raises(ValueError, match='wall_distance'):
        replay_wind(source, invalid_output, **options)
    assert not invalid_output.exists()
