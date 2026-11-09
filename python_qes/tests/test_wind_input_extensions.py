from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.building_pipeline import BuildingPipeline
from plumeswift.io import read_turbulence, read_wind_workspace
from plumeswift.qes_xml import parse_vector, read_qes_xml
from plumeswift.wind_case import UnsupportedFeatures, read_wind_case

ROOT = Path(__file__).resolve().parents[2]/'results/python_port'


@pytest.mark.parametrize('explicitly_closed', [False, True])
def test_native_quic_xml_appends_first_vertex_to_single_ring(tmp_path, explicitly_closed):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '28 28 28'
    polygon = ET.SubElement(tree.find('buildingsParams'), 'QUICBuilding')
    ET.SubElement(polygon, 'height').text = '4.4'
    ET.SubElement(polygon, 'baseHeight').text = '.1'
    vertices = [(8.3, 8.7), (8.3, 14.1), (14.2, 14.1), (14.2, 8.7)]
    if explicitly_closed:
        vertices.append(vertices[0])
    for x, y in vertices:
        ET.SubElement(polygon, 'xVertex').text = str(x)
        ET.SubElement(polygon, 'yVertex').text = str(y)
    path = tmp_path/'winds.xml'
    tree.write(path)

    native = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern = read_wind_case(path, compute_turbulence=False)
    assert len(native.buildings[0].vertices) == len(vertices)+1
    assert len(modern.buildings[0].vertices) == 5


def test_native_quic_courtyard_preserves_xml_order_only_for_closures(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    tree.find('simulationParameters/domain').text = '34 30 30'
    tree.find('simulationParameters/cellSize').text = '1 1 1'
    tree.find('simulationParameters/halo_x').text = '3'
    tree.find('simulationParameters/halo_y').text = '2'
    buildings = tree.find('buildingsParams')
    buildings.remove(buildings.find('rectangularBuilding'))
    polygon = ET.SubElement(buildings, 'QUICBuilding')
    ET.SubElement(polygon, 'height').text = '13'
    ET.SubElement(polygon, 'baseHeight').text = '0'
    vertices = ((7, 8), (17, 8), (17, 20), (7, 20), (7, 8),
                (10, 11), (10, 17), (14, 17), (14, 11), (10, 11))
    for x, y in vertices:
        ET.SubElement(polygon, 'xVertex').text = str(x)
        ET.SubElement(polygon, 'yVertex').text = str(y)
    path = tmp_path/'winds.xml'
    tree.write(path)

    native = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern = read_wind_case(path, compute_turbulence=False)
    assert native.buildings[0].vertices == modern.buildings[0].vertices
    assert native.buildings[0].holes == modern.buildings[0].holes
    assert modern.buildings[0].native_closure_vertices is None
    raw = native.halo.translate(vertices+vertices[:1])
    assert native.buildings[0].native_closure_vertices == raw
    workspace, apply = native.prepare()
    assert workspace.arrays['icellflag'][1, 15, 15] == 1
    apply()
    assert workspace.building_context.ordered_buildings[0][1].vertices == raw

    tree.find('simulationParameters/meshTypeFlag').text = '1'
    tree.find('simulationParameters/halo_x').text = '0'
    tree.find('simulationParameters/halo_y').text = '0'
    tree.write(path)
    legacy_cut = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern_cut = read_wind_case(path, compute_turbulence=False)
    native_arrays = replace(legacy_cut, native_staircase_metrics=True).initialize_geometry().arrays
    modern_arrays = modern_cut.initialize_geometry().arrays
    assert native_arrays['icellflag'][1, 14, 10] == 7
    assert native_arrays['building_volume_frac'][1, 14, 10] == 1.
    assert native_arrays['building_volume_frac'][1, 11, 14] == 0.
    assert modern_arrays['icellflag'][1, 14, 10] == 1
    assert native_arrays['icellflag'][1, 14, 12] == modern_arrays['icellflag'][1, 14, 12] == 1


def test_native_quic_detached_exteriors_share_id_and_closure(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    for tag, value in {'domain': '38 38 34', 'cellSize': '1 1 1', 'meshTypeFlag': '0'}.items():
        tree.find(f'simulationParameters/{tag}').text = value
    buildings = tree.find('buildingsParams')
    buildings.find('numBuildings').text = '2'
    rectangle = buildings.find('rectangularBuilding')
    for tag, value in {'xStart': '21', 'yStart': '11', 'length': '4', 'width': '9', 'height': '17'}.items():
        rectangle.find(tag).text = value
    polygon = ET.SubElement(buildings, 'QUICBuilding')
    ET.SubElement(polygon, 'height').text = '13'
    ET.SubElement(polygon, 'baseHeight').text = '0'
    vertices = ((7, 8), (17, 8), (17, 20), (7, 20), (7, 8),
                (8, 25), (12, 25), (12, 30), (8, 30), (8, 25))
    for x, y in vertices:
        ET.SubElement(polygon, 'xVertex').text = str(x)
        ET.SubElement(polygon, 'yVertex').text = str(y)
    path = tmp_path/'winds.xml'
    tree.write(path)

    native = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern = read_wind_case(path, compute_turbulence=False)
    assert len(native.buildings) == 2
    assert len(modern.buildings) == 3
    workspace, apply = native.prepare()
    flags = workspace.arrays['icellflag'][1]
    owners = workspace.arrays['ibuilding_flag'][1]
    assert (flags[12, 8], flags[26, 8], flags[22, 8]) == (0, 0, 1)
    assert (owners[12, 8], owners[26, 8], owners[22, 8], owners[12, 22]) == (1, 1, -1, 0)
    apply()
    ordered = dict(workspace.building_context.ordered_buildings)
    assert set(ordered) == {0, 1}
    assert ordered[1].vertices == vertices+vertices[:1]

    tree.find('simulationParameters/meshTypeFlag').text = '1'
    tree.write(path)
    legacy_cut = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern_cut = read_wind_case(path, compute_turbulence=False)
    native_arrays = replace(legacy_cut, native_staircase_metrics=True).initialize_geometry().arrays
    modern_arrays = modern_cut.initialize_geometry().arrays
    # Original CutBuilding walks the join out and back. Both crossings have
    # no area inside the gap, but the partial boundary cell closes its south face.
    np.testing.assert_array_equal(native_arrays['icellflag'][1, 19:25, 7], 7)
    np.testing.assert_array_equal(native_arrays['building_volume_frac'][1, 19:25, 7],
                                  [np.float32(1.)-np.float32(1./6), 1., 1., 1., 1., 1.])
    assert native_arrays['h'][1, 19, 7] == 0.
    assert native_arrays['building_volume_frac'][1, 25, 8] == 0.
    assert modern_arrays['icellflag'][1, 22, 7] == 1


@pytest.mark.parametrize('layout', ['two_holes', 'reversed_holes', 'mixed'])
def test_legacy_cutcell_preserves_physical_openings_and_walks_multiple_ring_joins(tmp_path, layout):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    for tag, value in {'domain': '38 38 34', 'cellSize': '1 1 1', 'meshTypeFlag': '1'}.items():
        tree.find(f'simulationParameters/{tag}').text = value
    buildings = tree.find('buildingsParams')
    buildings.remove(buildings.find('rectangularBuilding'))
    polygon = ET.SubElement(buildings, 'QUICBuilding')
    ET.SubElement(polygon, 'height').text = '13'
    ET.SubElement(polygon, 'baseHeight').text = '0'
    exterior = ((7, 8), (17, 8), (17, 20), (7, 20), (7, 8))
    if layout == 'mixed':
        first = ((10, 11), (10, 17), (14, 17), (14, 11), (10, 11))
        second = ((8, 25), (12, 25), (12, 30), (8, 30), (8, 25))
        partial, zero_area, open_cell = (11, 9), (24, 7), (14, 12)
    else:
        first = ((11, 9), (11, 11), (15, 11), (15, 9), (11, 9))
        second = ((10, 13), (10, 17), (14, 17), (14, 13), (10, 13))
        if layout == 'reversed_holes':
            first, second = first[::-1], second[::-1]
        partial, zero_area, open_cell = (9, 10), (9, 11), (9, 12)
    for x, y in exterior+first+second:
        ET.SubElement(polygon, 'xVertex').text = str(x)
        ET.SubElement(polygon, 'yVertex').text = str(y)
    path = tmp_path/'winds.xml'
    tree.write(path)

    legacy = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=True)
    modern = read_wind_case(path, compute_turbulence=False)
    actual = replace(legacy, native_staircase_metrics=True).initialize_geometry().arrays
    physical = modern.initialize_geometry().arrays
    for cell, flag, fraction in ((partial, 7, .5), (zero_area, 7, 1.), (open_cell, 1, 1.)):
        idx = (1, *cell)
        assert actual['icellflag'][idx] == flag
        assert actual['building_volume_frac'][idx] == fraction
    assert physical['icellflag'][(1, *zero_area)] == physical['icellflag'][(1, *open_cell)] == 1


def test_shipped_wake_flag_three_uses_native_modified_factors(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    tree.find('simulationParameters/domain').text = '28 28 28'
    building = tree.find('buildingsParams/rectangularBuilding')
    for tag, value in {'height': '4', 'xStart': '8', 'yStart': '8', 'length': '5', 'width': '6'}.items():
        building.find(tag).text = value
    wake = tree.find('buildingsParams/wakeFlag')
    path = tmp_path/'winds.xml'
    wake.text = '2'
    tree.write(path)
    expected = read_wind_case(path, compute_turbulence=False, native_building_math=True).initialize().arrays
    wake.text = '3'
    tree.write(path)
    case = read_wind_case(path, compute_turbulence=False, native_building_math=True)
    assert case.wake_mode == 3
    actual = case.initialize().arrays
    assert np.any(np.isin(actual['icellflag'], (4, 5, 6)))
    for name in ('u0', 'v0', 'w0', 'icellflag'):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize('canyon_mode', [1, 2])
@pytest.mark.parametrize('legacy', [False, True])
def test_xml_wake_off_closures_keep_active_upwind_cavity(tmp_path, canyon_mode, legacy):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    tree.find('simulationParameters/domain').text = '32 30 30'
    tree.find('simulationParameters/cellSize').text = '1 1 1'
    for name, value in {'height': '9', 'xStart': '10', 'yStart': '10', 'length': '6', 'width': '9'}.items():
        tree.find(f'buildingsParams/rectangularBuilding/{name}').text = value
    tree.find('buildingsParams/wakeFlag').text = '0'
    tree.find('buildingsParams/streetCanyonFlag').text = str(canyon_mode)
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path, compute_turbulence=False, legacy_turbulence_xml=legacy)
    expected = replace(case, sidewall_mode=0, rooftop_mode=0, street_canyon_mode=0).initialize().arrays
    actual = case.initialize().arrays
    assert np.any(expected['icellflag'] == 3)
    for name in ('icellflag', 'u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual[name], expected[name])
    custom = BuildingPipeline.native_order(upwind=0, wake=0, sidewall=0, rooftop=0, street_canyon=canyon_mode)
    with pytest.raises(ValueError, match='requires polygon-wake state'):
        replace(case, building_pipeline=custom).initialize()


def test_legacy_cuda_case_uses_native_cpu_building_geometry_and_closures(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    tree.find('simulationParameters/domain').text = '34 30 32'
    tree.find('simulationParameters/cellSize').text = '1 1 1'
    building = tree.find('buildingsParams/rectangularBuilding')
    for tag, value in {'height': '5', 'xStart': '8.5', 'yStart': '8.5', 'length': '9.2',
                       'width': '6.1', 'buildingRotation': '45'}.items():
        building.find(tag).text = value
    path = tmp_path/'winds.xml'
    tree.write(path)

    legacy = read_wind_case(path, compute_turbulence=False, poisson_backend='cuda-global-memory',
                            meteorology_backend='cuda', legacy_turbulence_xml=True)
    modern = read_wind_case(path, compute_turbulence=False, poisson_backend='cuda-global-memory',
                            meteorology_backend='cuda')
    legacy.native_staircase_metrics = modern.native_staircase_metrics = True
    flags = legacy.initialize_geometry().arrays['icellflag']
    modern_flags = modern.initialize_geometry().arrays['icellflag']
    assert np.count_nonzero(flags != modern_flags) == 10
    assert legacy.native_building_math and not modern.native_building_math
    assert legacy.buildings[0].vertices != modern.buildings[0].vertices


@pytest.mark.parametrize('mesh_type', [0, 1])
def test_legacy_solver_stencil_rounds_decimal_grid_denominators(tmp_path, mesh_type):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for tag, value in {'domain': '8 8 12', 'cellSize': '3.2 3.2 3.2', 'meshTypeFlag': str(mesh_type)}.items():
        simulation.find(tag).text = value
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path, compute_turbulence=False)
    case.native_staircase_metrics = True
    coefficients = case.initialize_geometry().arrays
    spacing = np.float32(3.2)
    expected = np.float32(1)/(spacing*spacing)
    for name in 'efghmn':
        assert coefficients[name][3, 3, 3] == expected


def test_native_wind_solver_defaults_and_zero_sweeps(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    max_iterations = simulation.find('maxIterations')
    simulation.remove(max_iterations)
    simulation.remove(simulation.find('tolerance'))
    path = tmp_path/'winds.xml'
    tree.write(path)

    modern = read_wind_case(path, compute_turbulence=False)
    assert (modern.poisson.policy.max_iterations, modern.poisson.policy.relative_tolerance) == (10000, 1e-7)
    for backend in ('cpu-native-rb', 'cpu-historical-sor', 'cuda-global-memory', 'cuda-shared-memory'):
        native = read_wind_case(path, compute_turbulence=False, poisson_backend=backend)
        assert (native.poisson.policy.max_iterations, native.poisson.policy.relative_tolerance) == (500, 1e-9)

    max_iterations.text = '0'
    simulation.append(max_iterations)
    tree.write(path)
    native = read_wind_case(path, compute_turbulence=False, poisson_backend='cpu-native-rb')
    result = native.run()
    assert result['solver'].iterations == 0
    assert result['solver'].update_linf == 1.
    assert not result['solver'].converged
    assert not np.any(result['multiplier'])
    with pytest.raises(ValueError, match='Invalid numerical convergence policy'):
        read_wind_case(path, compute_turbulence=False)


@pytest.mark.parametrize('invalid', [('', ' \n '), ('disabled', '1 unit')])
def test_invalid_legacy_simulation_options_keep_native_defaults(tmp_path, invalid):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for index, name in enumerate(('halo_x', 'halo_y', 'verticalStretching', 'maxIterations',
                                  'tolerance', 'logLawFlag', 'meshTypeFlag', 'domainRotation',
                                  'readCoefficientsFlag')):
        node = simulation.find(name)
        if node is None:
            node = ET.SubElement(simulation, name)
        node.text = invalid[index % 2]
    path = tmp_path/'winds.xml'
    tree.write(path)

    native = read_wind_case(path, poisson_backend='cpu-native-rb', legacy_turbulence_xml=True)
    assert native.poisson.policy.max_iterations == 500
    assert native.poisson.policy.relative_tolerance == 1e-9
    assert native.halo.cells(native.domain.spacing) == (0, 0)
    assert native.mesh_type == native.domain_rotation_deg == 0
    assert native.wall_log is None
    assert not native.turbulence.force_nonuniform_z
    with pytest.raises(ValueError):
        read_wind_case(path)


@pytest.mark.parametrize('field', ['boundaryLayerFlag', 'timeEpoch'])
def test_blank_legacy_station_field_uses_native_default(tmp_path, field):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    record = tree.find('metParams/sensor/timeSeries')
    node = record.find(field)
    if node is None:
        node = ET.SubElement(record, field)
    node.text = ' \n '
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path, legacy_turbulence_xml=True)
    assert case.profile.kind == 1
    assert case.timestamp == '2020-01-01T01:00:00'
    with pytest.raises(ValueError):
        read_wind_case(path)


def test_native_turbulence_sigma_stops_at_an_annotation(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    sigma = tree.find('turbParams/sigmaConst')
    sigma.text = '2.6 1.9 1.4 # neutral calibration'
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path, legacy_turbulence_xml=True)
    assert case.turbulence.explicit_sigma
    assert case.turbulence.sigma == (2.6, 1.9, 1.4)
    with pytest.raises(ValueError):
        read_wind_case(path)

    sigma.text = '2.6 1.9 1.4 1.0 # extra coefficient'
    tree.write(path)
    with pytest.raises(ValueError, match='sigmaConst requires exactly 3 values'):
        read_wind_case(path, legacy_turbulence_xml=True)


@pytest.mark.parametrize('tolerance,sweeps', [(0, 5), (1, 0), (2, 0)])
def test_native_tolerance_uses_the_initial_change_and_exact_zero_stop(tmp_path, tolerance, sweeps):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('domain').text = '22 22 20'
    sim.find('maxIterations').text = '5'
    sim.find('tolerance').text = str(tolerance)
    buildings = tree.find('buildingsParams')
    for name in ('upwindCavityFlag', 'wakeFlag', 'rooftopFlag', 'sidewallFlag', 'streetCanyonFlag'):
        buildings.find(name).text = '0'
    rectangle = buildings.find('rectangularBuilding')
    for name, value in {'xStart': 8, 'yStart': 8, 'height': 5, 'length': 4, 'width': 4}.items():
        rectangle.find(name).text = str(value)
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    for backend in ('cpu-native-rb', 'cpu-native-serial', 'cpu-historical-sor'):
        case = read_wind_case(xml, compute_turbulence=False, poisson_backend=backend)
        result = case.run()
        assert result['solver'].iterations == sweeps
        assert np.any(result['multiplier']) == (sweeps != 0)
        if not sweeps:
            assert result['solver'].update_linf == 1.
            assert result['solver'].converged
    with pytest.raises(ValueError, match='Invalid numerical convergence policy'):
        read_wind_case(xml, compute_turbulence=False)


@pytest.mark.parametrize('upwind,canonical_upwind', [(7, 2), (-1, 0)])
def test_wind_closure_flags_follow_native_dispatch(tmp_path, upwind, canonical_upwind):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    simulation = tree.find('simulationParameters')
    simulation.find('domain').text = '36 36 24'
    flags = tree.find('buildingsParams')
    rectangle = flags.find('rectangularBuilding')
    for name, value in {'xStart': 12, 'yStart': 12, 'height': 6, 'length': 8, 'width': 6}.items():
        rectangle.find(name).text = str(value)
    log_law = ET.SubElement(simulation, 'logLawFlag')
    vegetation = ET.SubElement(tree.getroot(), 'vegetationParams')
    vegetation_wake = ET.SubElement(vegetation, 'wakeFlag')
    fields = ('upwindCavityFlag', 'wakeFlag', 'rooftopFlag', 'sidewallFlag', 'streetCanyonFlag')

    def initialize(values, log_law_value, vegetation_wake_value):
        for name, value in zip(fields, values, strict=True):
            flags.find(name).text = str(value)
        log_law.text = str(log_law_value)
        vegetation_wake.text = str(vegetation_wake_value)
        path = tmp_path/'winds.xml'
        tree.write(path)
        case = read_wind_case(path, compute_turbulence=False)
        assert case.wall_log is None
        assert not case.vegetation.wake
        return case.initialize().arrays

    expected = initialize((canonical_upwind, 0, 0, 0, 0), 0, 0)
    actual = initialize((upwind, -2, 3, -1, 3), 2, 2)
    for name in ('u0', 'v0', 'w0', 'icellflag'):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert np.any(actual['icellflag'] == 3) == (canonical_upwind > 0)


def test_legacy_turbulence_optix_method_falls_back_to_serial_mixing(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    tree.find('simulationParameters/domain').text = '22 22 24'
    buildings = tree.find('buildingsParams')
    for name in ('upwindCavityFlag', 'wakeFlag', 'rooftopFlag', 'sidewallFlag', 'streetCanyonFlag'):
        buildings.find(name).text = '0'
    rectangle = buildings.find('rectangularBuilding')
    for name, value in {'xStart': 8, 'yStart': 8, 'height': 5, 'length': 4, 'width': 4}.items():
        rectangle.find(name).text = str(value)
    turbulence = tree.find('turbParams')
    method = turbulence.find('method')
    path = tmp_path/'winds.xml'

    method.text = '1'
    tree.write(path)
    expected = read_wind_case(path, legacy_turbulence_xml=True).run()
    method.text = '3'
    tree.write(path)
    legacy = read_wind_case(path, legacy_turbulence_xml=True)
    assert legacy.mixing_method == 1
    assert 'turbParams/method 3 -> 1 (native build without OptiX)' in legacy.xml_compatibility
    actual = legacy.run()
    np.testing.assert_array_equal(actual['workspace'].arrays['mixlength'], expected['workspace'].arrays['mixlength'])
    for name in ('mixingLengths', 'tke', 'CoEps'):
        np.testing.assert_array_equal(actual['turbulence'][name], expected['turbulence'][name])
    assert read_wind_case(path).mixing_method == 3

    ET.SubElement(turbulence, 'rayBackend').text = 'cpu'
    tree.write(path)
    assert read_wind_case(path, legacy_turbulence_xml=True).mixing_method == 3


def test_qes_vector_parser_distinguishes_grid_triplets_from_repeated_widths():
    root = ET.fromstring('<simulationParameters><domain>10\n 11 12</domain><domain>1</domain>'
                         '<dz_value>1.25</dz_value><dz_value>1.5 2</dz_value></simulationParameters>')
    assert parse_vector(root, 'domain', convert=int, size=3) == (10, 11, 12)
    assert parse_vector(root, 'dz_value', repeated=True) == (1.25, 1.5, 2.)
    with pytest.raises(ValueError, match='exactly 3'):
        parse_vector(root, 'cellSize', size=3)
    root.find('domain').text = '10 11 12 legacy 13'
    assert parse_vector(root, 'domain', convert=int, size=3, stop_on_invalid=True) == (10, 11, 12)
    with pytest.raises(ValueError):
        parse_vector(root, 'domain', convert=int, size=3)
    root.find('domain').text = '10 11 12 13'
    with pytest.raises(ValueError, match='exactly 3'):
        parse_vector(root, 'domain', convert=int, size=3, stop_on_invalid=True)


def test_qes_legacy_vectors_extract_attached_numeric_prefixes():
    root = ET.fromstring('<sim><domain>10+11+12cells</domain><cellSize>1 1.5 2e0m</cellSize></sim>')
    assert parse_vector(root, 'domain', convert=int, size=3, stop_on_invalid=True) == (10, 11, 12)
    assert parse_vector(root, 'cellSize', size=3, stop_on_invalid=True) == (1., 1.5, 2.)
    with pytest.raises(ValueError):
        parse_vector(root, 'cellSize', size=3)
    root.find('cellSize').text = '1 1.5 2e+m'
    with pytest.raises(ValueError, match='exactly 3'):
        parse_vector(root, 'cellSize', size=3, stop_on_invalid=True)


@pytest.mark.parametrize('value', [None, 'use model default'])
def test_legacy_optional_turbulence_scalars_use_native_defaults(tmp_path, value):
    source = Path(__file__).resolve().parents[2]/'data/cases/empty_west/winds.xml'
    tree = ET.parse(source)
    for name in ('buildingWallFlag', 'terrainWallFlag', 'turbUpperBound', 'backgroundMixing'):
        tree.find(f'turbParams/{name}').text = value
    path = tmp_path/'wind.xml'
    tree.write(path)

    model = read_wind_case(path, legacy_turbulence_xml=True).turbulence

    assert (model.building_wall, model.terrain_wall, model.upper_bound, model.background) == (2, 2, 20., 0.)
    with pytest.raises(ValueError):
        read_wind_case(path)


def test_legacy_wind_xml_maps_fields_and_keeps_unknowns_strict(tmp_path):
    source = Path(__file__).resolve().parents[2]/'data/cases/empty_west/winds.xml'
    tree = ET.parse(source)
    root = tree.getroot()
    simulation = root.find('simulationParameters')
    simulation.find('domain').text = '10 10 19'
    buildings = root.find('buildingsParams')
    turbulence = root.find('turbParams')
    stamp = root.find('metParams/sensor/timeSeries/timeStamp')
    stamp.text = '2012-11-11T18:07:00'
    modern = tmp_path/'modern.xml'
    tree.write(modern)
    buildings.tag = 'buildings'
    for name in ('wakeFlag', 'rooftopFlag', 'upwindCavityFlag'):
        item = buildings.find(name)
        buildings.remove(item)
        simulation.append(item)
    mixing = ET.SubElement(root, 'localMixingParam')
    for name in ('method', 'samples', 'save'):
        node = turbulence.find(name)
        turbulence.remove(node)
        mixing.append(node)
    ET.SubElement(simulation, 'useDiffusion').text = '0'
    stamp.text = '2012-11-11T18:7:0'
    sensor = root.find('metParams/sensor')
    series = sensor.find('timeSeries')
    sensor.extend(series)
    sensor.remove(series)
    legacy = tmp_path/'legacy.xml'
    tree.write(legacy)
    actual, expected = read_wind_case(legacy), read_wind_case(modern)
    assert actual.timestamp == expected.timestamp
    assert actual.wake_mode == expected.wake_mode
    assert 'buildings -> buildingsParams' in actual.xml_compatibility
    assert 'ignored simulationParameters/useDiffusion' in actual.xml_compatibility
    assert 'sensor profile fields -> sensor/timeSeries' in actual.xml_compatibility
    a, b = actual.run(), expected.run()
    for name in ('u', 'v', 'w', 'icellflag'):
        np.testing.assert_array_equal(a['workspace'].arrays[name], b['workspace'].arrays[name])
    np.testing.assert_array_equal(a['turbulence']['txx'], b['turbulence']['txx'])
    ET.SubElement(mixing, 'unknownControl').text = '9'
    tree.write(legacy)
    with pytest.raises(UnsupportedFeatures, match='unknownControl'):
        read_wind_case(legacy)
    compatible = read_wind_case(legacy, legacy_turbulence_xml=True)
    assert 'ignored /localMixingParam' in compatible.xml_compatibility


def test_qes_lexical_recovery_is_limited_to_known_variants(tmp_path):
    path = tmp_path/'input.xml'
    path.write_text('''<QESPlumeParameters><!~- old note --><source>
        <!-- invalid -- comment -->
        <sourceGeometry_FullDomain></SourceGeometry_FullDomain>
        <value><![CDATA[<!~- preserved -->]]></value>
        </source></QESPlumeParameters>''')
    root = read_qes_xml(path)
    assert [node.tag for node in root.find('source')] == ['sourceGeometry_FullDomain', 'value']
    assert root.findtext('source/value') == '<!~- preserved -->'
    path.write_text('<QESPlumeParameters><!~- old note --><source></SOURCE></QESPlumeParameters>')
    with pytest.raises(ET.ParseError):
        read_qes_xml(path)


@pytest.mark.parametrize('layout', ['padded', 'standard'])
def test_native_solver_coefficient_reload(layout):
    folder = ROOT/'coefficients'
    native_path = folder/'native_windsWk.nc'
    if not native_path.exists():
        pytest.skip('Run scripts/check_python_coefficients.py --native for native fixture')
    reference = read_wind_workspace(native_path).arrays
    actual = read_wind_case(folder/f'{layout}.xml').run()['workspace'].arrays
    for key in ('icellflag', 'e', 'f', 'g', 'h', 'm', 'n'):
        np.testing.assert_array_equal(actual[key], reference[key])
    for key in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[key], reference[key], atol=3e-6, rtol=0)
    assert not np.any(actual['icellflag'] == 0)


@pytest.mark.parametrize('alias_axes', [False, True])
def test_legacy_coefficient_reload_uses_stored_record_zero(tmp_path, alias_axes):
    from netCDF4 import Dataset
    from plumeswift.io import _staged_netcdf_path

    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('domain').text = '8 9 12'
    sim.find('maxIterations').text = '20'
    sim.find('readCoefficientsFlag').text = '1'
    ET.SubElement(sim, 'COEFF').text = 'coefficients.nc'
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    shape = (13, 9, 8)
    dimensions = ('t', 'z', 'y', 'x')
    sizes = (2, *(n+1 for n in shape))
    region = tuple(slice(0, size) for size in shape)
    values = (1 + np.indices(shape).sum(axis=0) % 4).astype(np.int16)
    with _staged_netcdf_path(tmp_path/'coefficients.nc') as temporary, Dataset(temporary, 'w') as nc:
        for axis, size in zip(dimensions, sizes, strict=True):
            nc.createDimension(axis, size)
        if alias_axes:
            dimensions = tuple(f'field_{axis}' for axis in dimensions)
            for axis, size in zip(dimensions, sizes, strict=True):
                nc.createDimension(axis, size)
        nc.createVariable('icellflag', 'i4', dimensions)[:] = 0
        for name in 'efghmn':
            variable = nc.createVariable(name, 'i2', dimensions)
            variable.set_auto_maskandscale(False)
            variable.scale_factor = .25
            variable.add_offset = 2.
            variable[:] = 30000
            variable[(0, *region)] = values

    legacy = read_wind_case(xml, legacy_turbulence_xml=True, compute_turbulence=False).initialize_geometry()
    for name in 'efghmn':
        np.testing.assert_array_equal(legacy.arrays[name], values)
    assert not np.any(legacy.arrays['icellflag'] == 0)
    modern = read_wind_case(xml, compute_turbulence=False)
    if alias_axes:
        with pytest.raises(ValueError, match='Invalid coefficient dimension order'):
            modern.initialize_geometry()
    else:
        np.testing.assert_array_equal(modern.initialize_geometry().arrays['e'], .25*values+2.)


@pytest.mark.parametrize('layout', ['standard', 'mixed', 'oversized'])
def test_legacy_unpadded_coefficients_replace_only_the_flat_prefix(tmp_path, layout):
    from netCDF4 import Dataset
    from plumeswift.io import _staged_netcdf_path

    tree = ET.parse(ROOT.parents[1]/'data/cases/empty_west/winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('domain').text = '8 9 20'
    sim.find('cellSize').text = '1 1 2'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    legacy = dict(legacy_turbulence_xml=True, compute_turbulence=False)
    defaults = read_wind_case(xml, **legacy).initialize_geometry().arrays
    shape = defaults['e'].shape
    nz, ny, nx = shape
    stored_shape = {'standard': shape, 'mixed': (nz, ny+2, nx-2),
                    'oversized': (nz+2, ny+2, nx+2)}[layout]
    dimensions = ('t', 'z', 'y', 'x')
    values = (2+np.indices(stored_shape).sum(axis=0) % 4).astype(np.float32)
    with _staged_netcdf_path(tmp_path/'coefficients.nc') as temporary, Dataset(temporary, 'w') as nc:
        for axis, size in zip(dimensions, (2, *stored_shape), strict=True):
            nc.createDimension(axis, size)
        nc.createVariable('icellflag', 'i4', dimensions)[:] = 0
        for name in 'efghmn':
            variable = nc.createVariable(name, 'f4', dimensions)
            variable[0] = values
            variable[1] = 99.
    sim.find('readCoefficientsFlag').text = '1'
    ET.SubElement(sim, 'COEFF').text = 'coefficients.nc'
    tree.write(xml)
    if layout == 'oversized':
        with pytest.raises(ValueError, match='does not fit cell grid'):
            read_wind_case(xml, **legacy).initialize_geometry()
        return
    prefix = values[:-1, :-1, :-1].ravel()
    old = read_wind_case(xml, **legacy).initialize_geometry().arrays
    for name in 'efghmn':
        np.testing.assert_array_equal(old[name].ravel()[:prefix.size], prefix)
        np.testing.assert_array_equal(old[name].ravel()[prefix.size:], defaults[name].ravel()[prefix.size:])
    assert np.any(old['m'].ravel()[prefix.size:] == .25)
    modern = read_wind_case(xml, compute_turbulence=False)
    if layout == 'standard':
        for name in 'efghmn':
            np.testing.assert_array_equal(modern.initialize_geometry().arrays[name], values)
    else:
        with pytest.raises(ValueError, match='does not match cell grid'):
            modern.initialize_geometry()


@pytest.mark.parametrize('text_field', ['icellflag', 'e'])
def test_legacy_coefficient_reload_rejects_character_fields(tmp_path, text_field):
    from netCDF4 import Dataset
    from plumeswift.io import read_solver_coefficients

    path = tmp_path/'coefficients.nc'
    dimensions = ('t', 'z', 'y', 'x')
    with Dataset(path, 'w') as nc:
        for axis, size in zip(dimensions, (1, 3, 3, 3), strict=True):
            nc.createDimension(axis, size)
        for field in ('icellflag', *'efghmn'):
            dtype = 'S1' if field == text_field else 'i4'
            nc.createVariable(field, dtype, dimensions)[:] = b'1' if field == text_field else 1
    with pytest.raises(TypeError, match='between text and numeric types'):
        read_solver_coefficients(path, (2, 2, 2), native_hyperslab=True)


def test_legacy_noncanonical_coefficient_flag_skips_reload_and_solid_marking(tmp_path):
    tree = ET.parse(ROOT.parents[1]/'data/cases/block_west/winds.xml')
    simulation = tree.find('simulationParameters')
    simulation.find('domain').text = '24 24 20'
    simulation.find('readCoefficientsFlag').text = '2'
    ET.SubElement(simulation, 'COEFF').text = 'unread.nc'
    building = tree.find('buildingsParams/rectangularBuilding')
    building.find('xStart').text = '8'
    building.find('yStart').text = '8'
    path = tmp_path/'winds.xml'
    tree.write(path)

    options = dict(compute_turbulence=False, legacy_turbulence_xml=True)
    case = read_wind_case(path, **options)
    assert case.coefficient_file is None and case.coefficient_flag == 2
    assert not np.any(case.initialize_geometry().arrays['icellflag'] == 0)
    simulation.find('readCoefficientsFlag').text = '0'
    tree.write(path)
    assert np.any(read_wind_case(path, **options).initialize_geometry().arrays['icellflag'] == 0)


@pytest.mark.parametrize('name', ['utm', 'split_roughness', 'utm_split'])
def test_native_station_coordinates_and_roughness(name):
    folder = ROOT/'station_geometry'/name
    native_path = folder/'native_windsWk.nc'
    if not native_path.exists():
        pytest.skip('Run scripts/check_python_station_geometry.py --native for native fixture')
    case = read_wind_case(folder/'winds.xml')
    assert [(s.x, s.y) for s in case.stations] == [(5., 5.), (17.5, 15.5)]
    result = case.run()
    actual = result['workspace'].arrays
    reference = read_wind_workspace(native_path).arrays
    for key in ('z0_u', 'z0_v'):
        assert actual[key].shape == case.domain.face_shape[1:]
        np.testing.assert_array_equal(actual[key].ravel()[:reference[key].size].astype(np.float32),
                                      reference[key].ravel())
    for key in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
        np.testing.assert_allclose(actual[key], reference[key], atol=2e-6, rtol=0)
    reference_turb = read_turbulence(folder/'native_turbOut.nc')
    for key in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z'):
        np.testing.assert_allclose(result['turbulence'][key], reference_turb[key], atol=4e-5, rtol=0)
