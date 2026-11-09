from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

from plumeswift.io import read_wind_workspace
from plumeswift.gis import polygon_layer_extents, resolve_vector_path
from plumeswift.input_paths import QESInputPaths
from plumeswift.wind_case import read_wind_case


FIXTURES = Path(__file__).resolve().parents[2]/'results/python_port/multipart_gis'


def test_polygon_layer_extents_use_native_float_coordinates(tmp_path, monkeypatch):
    shapefile = pytest.importorskip('shapefile')
    path = tmp_path/'buildings.shp'
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field('H', 'N')
        for x, y in ((500000.03, 4400000.01), (500010.98, 4400010.77)):
            writer.poly([[(x, y), (x, y+0.25), (x+1, y+0.25), (x+1, y), (x, y)]])
            writer.record(8)
    minimum, maximum, size = polygon_layer_extents(path, layer='BUILDINGS')
    assert minimum == (500000.03125, 4400000.)
    assert maximum == (500011.96875, 4400011.)
    assert size == (12, 11)
    archive = tmp_path/'footprints.zip'
    with ZipFile(archive, 'w') as zipped:
        for suffix in ('.shp', '.shx', '.dbf'):
            member = path.with_suffix(suffix)
            zipped.write(member, f'gis/{member.name}')
    assert polygon_layer_extents(archive, layer='BUILDINGS') == (minimum, maximum, size)
    monkeypatch.chdir(tmp_path)
    xml_folder = tmp_path/'project'
    xml_folder.mkdir()
    paths = QESInputPaths.from_input(xml_folder/'winds.xml')
    for filename in ('/vsizip/{footprints.zip}/gis/buildings.shp',
                     '/vsizip/footprints.zip/gis/buildings.shp',
                     f'/vsizip/{{{archive.as_posix()}}}/gis/buildings.shp'):
        assert polygon_layer_extents(resolve_vector_path(paths, filename)) == (minimum, maximum, size)


def test_native_shapefile_height_factor_can_change_solid_roof_cells(tmp_path):
    shapefile = pytest.importorskip('shapefile')
    with shapefile.Writer(str(tmp_path/'buildings'), shapeType=shapefile.POLYGON) as writer:
        writer.field('H', 'N')
        writer.poly([[(108, 208), (108, 214), (114, 214), (114, 208), (108, 208)]])
        writer.record(3)

    tree = ET.parse(FIXTURES.parents[2]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for name, value in {'domain': '24 24 20', 'cellSize': '1 1 .6', 'UTMx': '100', 'UTMy': '200'}.items():
        simulation.find(name).text = value
    buildings = tree.find('buildingsParams')
    ET.SubElement(buildings, 'SHPFile').text = 'buildings.shp'
    ET.SubElement(buildings, 'heightFactor').text = '.3'
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    native = read_wind_case(xml, compute_turbulence=False, legacy_turbulence_xml=True,
                            native_building_math=True)
    modern = read_wind_case(xml, compute_turbulence=False)
    assert native.buildings[0].height == float(np.float32(3)*np.float32(.3))
    assert modern.buildings[0].height == 3*.3
    assert native.initialize_geometry().arrays['icellflag'][2, 9, 9] == 0
    full_precision_height = replace(native, buildings=modern.buildings)
    assert full_precision_height.initialize_geometry().arrays['icellflag'][2, 9, 9] == 1


@pytest.mark.parametrize('tree_geometry', [False, True])
def test_legacy_gis_follows_ogr_exterior_and_record_order(tmp_path, tree_geometry):
    shapefile = pytest.importorskip('shapefile')
    path = tmp_path/('trees' if tree_geometry else 'buildings')

    def ring(x, y, width):
        return [(x, y), (x, y+width), (x+width, y+width), (x+width, y), (x, y)]

    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field('H', 'N')
        if tree_geometry:
            writer.field('D', 'N')
            writer.field('LAI', 'N')
        writer.poly([ring(104, 204, 3), ring(109, 204, 3)])
        writer.record(*((3, 4, 1) if tree_geometry else (3,)))
        writer.poly([ring(115, 208, 8), ring(117, 210, 4)[::-1]])
        writer.record(*((7, 8, 3) if tree_geometry else (7,)))
    tree = ET.parse(FIXTURES.parents[2]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for name, value in {'domain': '26 22 20', 'UTMx': '100', 'UTMy': '200'}.items():
        simulation.find(name).text = value
    if tree_geometry:
        options = ET.SubElement(tree.getroot(), 'vegetationParams')
        ET.SubElement(options, 'wakeFlag').text = '0'
    else:
        options = tree.find('buildingsParams')
    ET.SubElement(options, 'SHPFile').text = path.with_suffix('.shp').name
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    legacy = read_wind_case(xml, compute_turbulence=False, legacy_turbulence_xml=True)
    modern = read_wind_case(xml, compute_turbulence=False)

    legacy_geometry = legacy.vegetation.canopies if tree_geometry else legacy.buildings
    modern_geometry = modern.vegetation.canopies if tree_geometry else modern.buildings
    assert [(item.height, len(item.holes)) for item in legacy_geometry] == [(3., 0)]
    assert [(item.height, len(item.holes)) for item in modern_geometry] == [(3., 0), (3., 0), (7., 1)]
    if tree_geometry:
        assert legacy_geometry[0].leaf_area_index == 1.
    assert legacy.initialize_geometry().arrays['icellflag'][2, 11, 18] == (22 if tree_geometry else 0)
    assert modern.initialize_geometry().arrays['icellflag'][2, 11, 18] == 1


def test_shapefile_directory_supplies_building_and_tree_layers(tmp_path, monkeypatch):
    import sys

    shapefile = pytest.importorskip('shapefile')
    monkeypatch.setitem(sys.modules, 'fiona', None)
    dataset = tmp_path/'structures'
    dataset.mkdir()
    for name, x in [('buildings', 108), ('trees', 121)]:
        with shapefile.Writer(str(dataset/f'{name}.shp'), shapeType=shapefile.POLYGON) as writer:
            writer.field('H', 'N', decimal=1)
            if name == 'trees':
                writer.field('D', 'N', decimal=1)
                writer.field('LAI', 'N', decimal=1)
            writer.poly([[(x, 208), (x, 214), (x+5, 214), (x+5, 208), (x, 208)]])
            writer.record(*((4.2, 5., 2.5) if name == 'trees' else (6.,)))

    tree = ET.parse(FIXTURES.parents[2]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for name, value in {'domain': '36 30 26', 'UTMx': '100', 'UTMy': '200'}.items():
        simulation.find(name).text = value
    buildings = tree.find('buildingsParams')
    for name in ('upwindCavityFlag', 'wakeFlag', 'rooftopFlag', 'streetCanyonFlag', 'sidewallFlag'):
        buildings.find(name).text = '0'
    ET.SubElement(buildings, 'SHPFile').text = dataset.name
    vegetation = ET.SubElement(tree.getroot(), 'vegetationParams')
    ET.SubElement(vegetation, 'wakeFlag').text = '0'
    ET.SubElement(vegetation, 'SHPFile').text = dataset.name
    ET.SubElement(vegetation, 'SHPTreeLayer').text = 'TREES'
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    case = read_wind_case(xml, compute_turbulence=False)
    assert [item.height for item in case.buildings] == [6.]
    assert [item.leaf_area_index for item in case.vegetation.canopies] == [2.5]
    flags = case.initialize().arrays['icellflag']
    assert flags[2, 9, 9] == 0
    assert flags[2, 9, 22] == 22
    with pytest.raises(ValueError, match='Select exactly one shapefile layer'):
        polygon_layer_extents(dataset)


@pytest.mark.parametrize('name, parts, holes, wind_tolerance', [
    ('building_courtyard', 4, 1, 3e-6),
    ('multipart_wake', 3, 0, 1.2e-5),
    ('tree_courtyard', 4, 1, 6e-5),
])
def test_multipart_matches_single_ring_qes(name, parts, holes, wind_tolerance):
    pytest.importorskip('shapefile')
    folder = FIXTURES/name
    path = folder/'native_windsWk.nc'
    if not path.exists():
        pytest.skip('Run scripts/check_python_multipart_gis.py --native')
    case = read_wind_case(folder/'multipart.xml', compute_turbulence=False)
    definitions = case.vegetation.canopies if case.vegetation else case.buildings
    assert len(definitions) == parts
    assert sum(len(model.holes) for model in definitions) == holes
    assert [model.height for model in definitions] == [4.2]*(parts-1)+[5.1]
    actual = case.run()['workspace'].arrays
    native = read_wind_workspace(path).arrays
    for field in ('icellflag', *'efghmn'):
        np.testing.assert_array_equal(actual[field], native[field])
    for field in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[field], native[field], atol=wind_tolerance, rtol=0)
    if holes:
        assert actual['icellflag'][2, 12, 13] == 1
        assert actual['icellflag'][2, 14, 15] == (22 if case.vegetation else 0)


def test_courtyard_cutcell_has_exact_open_volume_and_wall_distance():
    pytest.importorskip('shapefile')
    path = FIXTURES/'building_courtyard/multipart.xml'
    if not path.exists():
        pytest.skip('Run scripts/check_python_multipart_gis.py')
    case = read_wind_case(path, compute_turbulence=False)
    model = next(building for building in case.buildings if building.holes)
    left = min(x for x, _ in model.holes[0])
    result = replace(case, mesh_type=1).run()
    assert result['solver'].converged
    a = result['workspace'].arrays
    index = (2, 12, 12)
    assert a['icellflag'][index] == 7
    assert a['wall_distance'][index] == pytest.approx(12.5-left)
    assert a['ni'][index] == 1
    assert a['building_volume_frac'][index] == pytest.approx(13-left)
    for face, expected in zip('efghmn', (1., 0., *([13-left]*4))):
        assert a[face][index] == pytest.approx(expected)
    assert a['icellflag'][2, 12, 13] == 1
    assert a['building_volume_frac'][2, 12, 13] == 1.


def test_tree_courtyard_wake_uses_an_occupied_reference_column():
    pytest.importorskip('shapefile')
    path = FIXTURES/'tree_courtyard/multipart.xml'
    if not path.exists():
        pytest.skip('Run scripts/check_python_multipart_gis.py')
    case = read_wind_case(path, compute_turbulence=False)
    wake = replace(case, vegetation=replace(case.vegetation, wake=True)).initialize().arrays
    assert (wake['icellflag'] == 23).any()
    assert wake['icellflag_initial'][2, 12, 13] == 1
    assert np.isfinite(wake['u0']).all()


def test_closed_vertex_rings_in_original_xml_fields(tmp_path):
    path = Path(__file__).resolve().parents[2]/'data/cases/empty_west/winds.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    root.find('simulationParameters/domain').text = '35 35 30'
    buildings = root.find('buildingsParams')
    buildings.remove(buildings.find('numBuildings'))
    for key in ('rooftopFlag', 'upwindCavityFlag', 'streetCanyonFlag', 'wakeFlag', 'sidewallFlag'):
        buildings.find(key).text = '0'
    building = ET.SubElement(buildings, 'QUICBuilding')
    vegetation = ET.SubElement(root, 'vegetationParams')
    ET.SubElement(vegetation, 'wakeFlag').text = '0'
    canopy = ET.SubElement(vegetation, 'Homogeneous')
    ET.SubElement(canopy, 'attenuationCoefficient').text = '1'
    exterior = ((4, 4), (4, 12), (12, 12), (12, 4), (4, 4))
    hole = ((6, 6), (6, 10), (10, 10), (10, 6), (6, 6))
    for node, offset in ((building, 0), (canopy, 13)):
        ET.SubElement(node, 'height').text = '4.2'
        ET.SubElement(node, 'baseHeight').text = '0'
        for x, y in (*exterior, *hole):
            ET.SubElement(node, 'xVertex').text = str(x+offset)
            ET.SubElement(node, 'yVertex').text = str(y)
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    case = read_wind_case(xml, compute_turbulence=False)
    assert len(case.buildings[0].holes) == len(case.vegetation.canopies[0].holes) == 1
    flags = case.initialize().arrays['icellflag']
    assert flags[2, 5, 5] == 0 and flags[2, 8, 8] == 1
    assert flags[2, 5, 18] == 20 and flags[2, 8, 21] == 1
