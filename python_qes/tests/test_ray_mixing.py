from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.geometry import PolygonBuilding
from plumeswift.ray_mixing import RayMixing, mixing_scene, ray_directions
from plumeswift.triangle_mesh import TriangleMesh
from plumeswift.wind_case import read_wind_case


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ('flat', 'obstacles', 'terrain', 'courtyard'))
def test_ray_mixing_matches_original_cuda_generator_and_triangle_reference(name):
    fixture = ROOT/'results/python_port/ray_mixing'/name
    if not fixture.with_suffix('.out').exists():
        pytest.skip('Run scripts/check_python_ray_mixing.py --native')
    with fixture.with_suffix('.in').open('rb') as stream:
        nx, ny, nz, samples, triangles, _ = np.fromfile(stream, dtype='<i4', count=6)
        spacing = np.fromfile(stream, dtype='<f4', count=3)
        flags = np.fromfile(stream, dtype='<i4', count=nx*ny*nz).reshape(nz, ny, nx)
        mesh = TriangleMesh(np.fromfile(stream, dtype='<f4', count=9*triangles).reshape(-1, 3, 3))
    domain = Domain((nx, ny, nz-1), tuple(spacing))
    workspace = domain.workspace()
    workspace.arrays['icellflag'] = flags
    directions, native = np.split(np.fromfile(fixture.with_suffix('.out'), dtype='<f4'), [3*samples])
    np.testing.assert_allclose(ray_directions(samples), directions.reshape(-1, 3), atol=3e-7, rtol=0)
    values = RayMixing(samples)(workspace, mesh).ravel()
    missed = native == np.finfo(np.float32).max
    np.testing.assert_allclose(values[~missed], native[~missed], atol=5e-5, rtol=0)
    heights = np.broadcast_to(np.maximum(domain.coordinates['z'], 0.)[:, None, None], flags.shape)
    np.testing.assert_allclose(values[missed], heights.ravel()[missed], atol=1e-6, rtol=0)


def test_method_three_xml_saves_and_reloads_the_same_mixing_length(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '10 10 19'
    mixing = tree.find('turbParams')
    mixing.find('method').text = '3'
    mixing.find('samples').text = '17'
    mixing.find('save').text = 'true'
    ET.SubElement(mixing, 'LMfile').text = 'distances.nc'
    xml = tmp_path/'wind.xml'
    tree.write(xml)
    actual = read_wind_case(xml).run()
    a = actual['workspace'].arrays
    np.testing.assert_allclose(a['mixlength'][1:, 0, 0], np.arange(1, 20)+.5, atol=1e-6, rtol=0)
    mixing.find('method').text = '4'
    mixing.find('save').text = 'false'
    tree.write(xml)
    reloaded = read_wind_case(xml).run()
    np.testing.assert_array_equal(reloaded['workspace'].arrays['mixlength'], a['mixlength'])
    np.testing.assert_array_equal(reloaded['turbulence']['Lm'], actual['turbulence']['Lm'])


def test_courtyard_ray_mesh_has_an_open_roof_and_inner_walls():
    domain = Domain((16, 16, 20), (1., 1., 1.))
    outer = PolygonBuilding.rectangle(3, 3, 8, 8, 4)
    inner = PolygonBuilding.rectangle(5, 5, 4, 4, 4)
    building = PolygonBuilding(outer.vertices, outer.height, holes=(inner.vertices,))
    mesh = mixing_scene(domain.workspace(), (building,))
    assert mesh.ray_hit((6., 7., 6.), (0., 0., -1.)).parameter == 6.
    assert mesh.ray_hit((4., 7., 6.), (0., 0., -1.)).parameter == 2.
    assert mesh.ray_hit((6., 7., 2.), (-1., 0., 0.)).parameter == 1.
