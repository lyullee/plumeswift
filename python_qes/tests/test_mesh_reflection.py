import json
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from plumeswift.triangle_mesh import TriangleMesh
from plumeswift.mesh_reflection import MeshReflection, TriangleWalls
from plumeswift.assembly import default_registry, ComponentChoice
from plumeswift.mesh_mixing import LegacyMeshMixing,sphere_directions


def plane():
    return TriangleMesh([[[-10, -10, 0], [10, -10, 0], [0, 10, 0]]])


def test_coupled_plume_uses_the_prepared_cut_cell_mesh(tmp_path, monkeypatch):
    from plumeswift.plume_run import run_plume
    from plumeswift.plume_transport import PlumeModel
    from plumeswift.terrain import GridTerrain
    from plumeswift.wind_case import read_wind_case

    class CutCellTerrain:
        def mesh(self, domain):
            raise AssertionError('Coupled plume should use the prepared wind mesh')

        def cut_geometry(self, domain, halo):
            self.prepared_mesh, corners = GridTerrain(np.zeros((8, 8))).cut_geometry(domain, halo)
            return self.prepared_mesh, corners

    source = Path(__file__).resolve().parents[2]/'data/cases/empty_west'
    tree = ET.parse(source/'winds.xml')
    tree.find('simulationParameters/domain').text = '8 8 20'
    path = tmp_path/'winds.xml'
    tree.write(path)
    terrain = CutCellTerrain()
    wind = replace(read_wind_case(path), terrain=terrain, mesh_type=1)
    tree = ET.parse(source/'plume.xml')
    for path, value in {'plumeParameters/simDur': '1', 'plumeParameters/updateFrequency_timeLoop': '1',
                        'collectionParameters/timeAvgStart': '0', 'collectionParameters/timeAvgFreq': '1',
                        'boundaryConditions/wallReflection': 'meshReflection',
                        './/particlePerTimestep': '1', './/posX': '4', './/posY': '4'}.items():
        tree.find(path).text = value
    path = tmp_path/'plume.xml'
    tree.write(path)
    meshes = []
    set_fields = PlumeModel.set_fields

    def observe(model, *args):
        meshes.append(model.mesh)
        return set_fields(model, *args)

    monkeypatch.setattr(PlumeModel, 'set_fields', observe)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = run_plume(wind, path, tmp_path, write_concentration=False)
    assert result['wind_records'] == 1
    assert meshes == [terrain.prepared_mesh]


def test_ray_parallel_on_slab_boundary_degenerate_and_zero():
    mesh = plane()
    assert mesh.ray_hit([0, 0, 2], [0, 0, -4]).parameter == .5
    assert mesh.ray_hit([0, 0, 2], [0, 0, -4], max_parameter=.4) is None
    assert mesh.ray_hit([0, 0, 0], [1, 0, 0]) is None
    assert mesh.ray_hit([0, 0, 0], [0, 0, 0]) is None
    assert mesh.ray_hit([0, 0, 0], [0, 0, 1]).parameter == 0
    assert mesh.ray_hit([20, 0, 2], [0, 0, -1]) is None
    degenerate = TriangleMesh([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
    assert degenerate.ray_hit([.5, 0, 2], [0, 0, -1]) is None
    with pytest.raises(ValueError):
        mesh.ray_hit([np.nan, 0, 0], [1, 0, 0])


def test_native_offset_and_endpoint_gate():
    reflection = MeshReflection(plane())
    end, fluct, count = reflection.trace([0, 0, 1], [0, 0, -2], [1, 2, -3])
    np.testing.assert_allclose(end, [0, 0, 1.2])
    np.testing.assert_array_equal(fluct, [1, 2, 3]); assert count == 1
    wall = TriangleWalls(reflection, lambda p: p[:, 2] < 0)
    p = np.array([[0, 0, 1.], [0, 0, 1.]])
    d = np.array([[0, 0, -2.], [0, 0, .5]])
    f = np.array([[1, 2, -3.], [3, 4, 5.]])
    q, v = wall.reflect(p, d, f)
    np.testing.assert_allclose(q[:, 2], [1.2, 1.5])
    np.testing.assert_allclose(v, [[1, 2, 3], [3, 4, 5]])
    np.testing.assert_allclose(p[:, 2], 1)


def test_bounce_limit_fails_explicitly():
    triangles = np.r_[plane().triangles, plane().triangles+np.array([0, 0, 2])]
    with pytest.raises(RuntimeError, match='bounce limit'):
        MeshReflection(TriangleMesh(triangles), max_bounces=1).trace([0, 0, 1], [0, 0, 10], [1, 2, 3])


def test_mesh_registry_composes_without_native_runtime():
    choice = ComponentChoice('mesh.triangles', {'triangles': plane().triangles})
    reflector = default_registry().create('reflection.mesh', mesh=choice)
    assert reflector.trace([0, 0, 1], [0, 0, -2], [0, 0, 0])[2] == 1


def test_ray_equal_distance_native_right_subtree_ties():
    triangle = plane().triangles[0]
    # Two leaves preserve input order; larger nodes reverse equal midpoints
    # during each merge, then choose the right subtree on a distance tie.
    for count, expected in [(2, 1), (3, 0), (8, 3)]:
        mesh = TriangleMesh(np.repeat(triangle[None], count, axis=0), leaf_size=1)
        assert mesh.ray_hit([0, 0, 1], [0, 0, -1]).triangle == expected


def test_actual_native_mesh_fixture():
    root = Path(__file__).resolve().parents[2]
    out = root/'results/python_port/mesh'
    if not (out/'comparison.json').exists():
        pytest.skip('Run scripts/check_python_mesh.py for compiled upstream reference')
    native = np.loadtxt(out/'native.csv', delimiter=',')
    rows = (out/'input.txt').read_text().splitlines()
    n = int(rows[0]); mesh = TriangleMesh(np.loadtxt(rows[1:n+1]).reshape(n, 3, 3))
    inputs = np.loadtxt(rows[n+2:]); reflection = MeshReflection(mesh)
    for row, ref in zip(inputs, native):
        x, u, f = row[:3], row[3:6], row[6:]
        hit = mesh.ray_hit(x, u)
        assert bool(hit) == bool(ref[1])
        if hit:
            np.testing.assert_allclose([hit.parameter, hit.distance, *hit.point, *hit.normal], ref[2:10], atol=1e-4, rtol=0)
        end, fluct, _ = reflection.trace(x, u, f)
        np.testing.assert_allclose(np.r_[end, fluct], ref[10:], atol=1e-4, rtol=0)
    mix_ref = np.loadtxt(out/'mixing_native.csv', delimiter=',')[:, 1].reshape(2, 2, 2)
    flags = np.array([0, 1, 2, 1, 1, 1, 0, 1]).reshape(2, 2, 2)
    np.testing.assert_allclose(LegacyMeshMixing(mesh).calculate(flags, [2, 2, 2], initial=np.full(flags.shape, -7.)), mix_ref, atol=1e-5, rtol=0)


def test_legacy_mesh_mixing_flags_grid_and_absence_of_hit():
    mixing = LegacyMeshMixing(plane(), directions=[[0, 0, -1]])
    flags = np.array([[[1, 0, 2]]])
    result = mixing.calculate(flags, [1, 1, 2], initial=np.full(flags.shape, 9.))
    np.testing.assert_allclose(result, [[[1, 9, 9]]])
    assert np.isinf(LegacyMeshMixing(plane(), directions=[[0, 0, 1]]).distance([[0, 0, 1]])[0])


def test_native_random_directions():
    out = Path(__file__).resolve().parents[2]/'results/python_port/mesh'
    if not (out/'directions_native.csv').exists():
        pytest.skip('Run scripts/check_python_mesh.py for native directions')
    directions = sphere_directions()
    np.testing.assert_allclose(directions, np.loadtxt(out/'directions_native.csv', delimiter=','), atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=2e-7)
