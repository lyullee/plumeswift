from pathlib import Path

import numpy as np
import pytest

from plumeswift.cutcell_terrain import apply_cutcell_terrain
from plumeswift.cutcell_walls import cutcell_wall_stencil
from plumeswift.domain import Domain, native_vertical_coordinates
from plumeswift.geometry import PolygonBuilding
from plumeswift.io import _open_netcdf, read_wind_workspace, write_netcdf
from plumeswift.meteorology import SensorProfile
from plumeswift.wind_case import WindCase, read_wind_case
from plumeswift.wind_replay import replay_wind


FIXTURE = Path(__file__).resolve().parents[2]/'results/python_port/cutcell'


def test_native_cutcell_applies_five_percent_opening_threshold_after_3d_face_sum():
    vertices = ((598.375, 763.75), (630.625, 763.25), (630.6875, 765.),
                (639.5625, 765.), (639.1875, 735.), (598., 735.5), (598.375, 763.75))
    building = PolygonBuilding(vertices, 46.183101654052734)
    case = WindCase(Domain((66, 79, 18), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (building,), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    column = np.s_[1:16, 75, 63]
    np.testing.assert_array_equal(arrays['icellflag'][column], 0)
    np.testing.assert_array_equal(arrays['building_volume_frac'][column], np.float32(.039583325))


def test_native_cutcell_revisits_closing_vertices_between_north_and_south_faces():
    origin = np.array([940., 360.])
    vertices = np.array([
        (7.125, 9.5), (.625, 9.25), (.8125, -1.5), (-9.5, -1.75),
        (-10.4375, 47.75), (4.8125, 48.), (5., 38.), (21.1875, 38.25),
        (21.0625, 44.75), (49.1875, 45.25), (49.9375, 6.25), (38., 6.),
        (37.875, 13.), (7.0625, 12.25), (7.125, 9.5),
    ])+origin
    building = PolygonBuilding(tuple(map(tuple, vertices)), 10.748700141906738)
    case = WindCase(Domain((103, 45, 7), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (building,), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    column = np.s_[1:5, 36, 94]
    np.testing.assert_array_equal(arrays['icellflag'][column], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][column],
                               [0.80890185, 0.80890185, 0.80890185, 0.96335763], rtol=0, atol=2e-7)


def test_boundary_buildings_keep_ids_and_do_not_wrap_to_opposite_edge():
    domain = Domain((24, 24, 12), (1., 1., 1.))
    outside = PolygonBuilding.rectangle(-30, 8, 2, 2, 2)
    west = PolygonBuilding.rectangle(-3, 8, 4, 3, 4.35)
    east = PolygonBuilding.rectangle(22, 14, 6, 3, 4.35)
    options = dict(mesh_type=1, upwind_mode=2, wake_mode=2, sidewall_mode=1, rooftop_mode=2)
    profile = SensorProfile(2., 270., 10.)
    actual = WindCase(domain, profile, (outside, west, east), **options).initialize().arrays
    expected = WindCase(domain, profile, (west, east), **options).initialize().arrays
    for key in ('icellflag', 'building_volume_frac', *'efghmn', 'u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual[key], expected[key])
    owners = actual['ibuilding_flag']
    assert set(np.unique(owners)) == {-1, 1, 2}
    assert np.all(np.argwhere(owners == 1)[:, 2] == 0)
    assert np.all(np.argwhere(owners == 2)[:, 2] >= 22)


def test_native_cutcell_retains_horizontal_gridline_contact():
    domain = Domain((14, 16, 24), (10., 10., 3.))
    vertices = ((69.1875, 70.), (69.3125, 27.), (55.0625, 27.), (55., 70.), (69.1875, 70.))
    building = PolygonBuilding(vertices, 25.46540069580078)
    case = WindCase(domain, SensorProfile(5., 270., 10., .1), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:10, 6:8, 5:7], 7)
    np.testing.assert_array_equal(arrays['building_volume_frac'][1:10, 7, 5:7], 1.)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:9, 6, 6], .84639049, rtol=0, atol=1e-7)
    np.testing.assert_allclose(arrays['m'][1:9, 6, 6], .05990795, rtol=0, atol=1e-7)


def test_native_cutcell_collects_reentrant_face_points():
    vertices = ((168.625, 62.75), (167.4375, 18.5), (148.6875, 19.), (149.3125, 43.),
                (142.375, 43.), (141.6875, 18.75), (12.5625, 22.), (13.625, 63.75),
                (49.1875, 62.75), (49.0625, 58.25), (67.0625, 57.75), (67.3125, 68.),
                (12.875, 69.5), (14., 112.75), (169.6875, 108.75), (168.5625, 67.5),
                (151.125, 68.), (151., 63.25), (168.625, 62.75))
    case = WindCase(Domain((22, 16, 28), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 41.12200164794922),), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    columns = (((4, 14), .051041603, .292666137), ((6, 15), .025989532, .292666137),
               ((6, 4), .031106830, .550446987))
    for (j, i), fraction, roof_fraction in columns:
        np.testing.assert_array_equal(arrays['icellflag'][1:14, j, i], 0)
        assert arrays['icellflag'][14, j, i] == 7
        np.testing.assert_allclose(arrays['building_volume_frac'][1:14, j, i], fraction, atol=1e-7, rtol=0)
        assert arrays['building_volume_frac'][14, j, i] == pytest.approx(roof_fraction, abs=1e-7)
    # A west-to-south vertex is followed by a second edge crossing west to east.
    np.testing.assert_allclose([arrays[face][14, 6, 4] for face in 'efghmn'],
                               [.00892276, .00292666, .00292666, .00355540, .11111111, 0.],
                               atol=1e-7, rtol=0)
    # Two separated vertices arrive at the west face across different edges.
    np.testing.assert_array_equal(arrays['icellflag'][1:15, 6, 16], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:14, 6, 16], .14245105, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][14, 6, 16] == pytest.approx(.59602833, abs=1e-7)
    assert arrays['f'][14, 6, 16] == pytest.approx(.00292666, abs=1e-7)
    # The opposite side shares the east face between two distinct vertices.
    np.testing.assert_array_equal(arrays['icellflag'][1:15, 6, 1], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:14, 6, 1], .33692992, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][14, 6, 1] == pytest.approx(.68767893, abs=1e-7)
    assert arrays['n'][14, 6, 1] == pytest.approx(.03727007, abs=1e-7)


def test_native_cutcell_retains_earlier_edge_reference_at_east_reentry():
    vertices = ((46.625, 19.5), (46.8125, 28.), (40.3125, 28.), (40.625, 50.25),
                (50.0625, 50.25), (49.9375, 43.), (53.625, 43.), (53.25, 19.25), (46.625, 19.5))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 7.183),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:4, 4, 4], [0, 0, 7])
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 4, 4],
                               [.05543911, .05543911, .75160325], atol=1e-7, rtol=0)
    # The adjacent vertex arrives horizontally and follows an earlier sentinel crossing.
    np.testing.assert_array_equal(arrays['icellflag'][1:4, 4, 5], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 4, 5],
                               [.44583333, .44583333, .77916670], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(arrays['h'][1:4, 4, 5], 0.)
    assert arrays['f'][3, 4, 5] == pytest.approx(.00605667, abs=1e-7)


def test_native_cutcell_collects_west_and_north_face_points():
    vertices = ((49.375, 48.75), (49.5625, 53.75), (58.8125, 53.25), (58.0625, 32.25),
                (31.75, 33.25), (32.3125, 49.25), (49.375, 48.75))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 8.1194),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:4, 4, 4], [0, 0, 7])
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 4, 4],
                               [.09546661, .09546661, .57828772], atol=1e-7, rtol=0)
    np.testing.assert_allclose([arrays[face][3, 4, 4] for face in 'efghmn'],
                               [.00293533, .00362433, .00959157, .00293533, .11111111, 0.],
                               atol=1e-7, rtol=0)


def test_native_cutcell_collects_south_and_west_face_points():
    vertices = ((36.6875, 49.), (37.375, 63.), (63.875, 61.5), (61.875, 21.),
                (46., 21.75), (47.3125, 48.5), (36.6875, 49.))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 39.84299850463867),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:15, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:14, 4, 4], .53758645, atol=2e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:14, 4, 4], .04221340, atol=1e-7, rtol=0)
    assert arrays['n'][14, 4, 4] == pytest.approx(.04221340, abs=1e-7)


def test_native_cutcell_collects_orthogonal_south_and_west_face_points():
    vertices = ((44.4375, 23.75), (44.4375, 49.5), (13.4375, 49.5), (13.4375, 100.),
                (90., 100.), (90., 23.75), (44.4375, 23.75))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 15.589799880981445),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:7, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:6, 4, 4], .35963535, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:6, 4, 4], .02619792, atol=1e-7, rtol=0)
    assert arrays['n'][6, 4, 4] == pytest.approx(.02619792, abs=1e-7)


def test_native_cutcell_collects_orthogonal_north_and_east_face_points():
    vertices = ((220.5, 217.75), (220.5, 203.5), (241.5625, 203.5), (241.5625, 170.),
                (190., 170.), (190., 217.75), (220.5, 217.75))
    case = WindCase(Domain((30, 30, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 15.589799880981445),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:7, 20, 22], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:6, 20, 22], .52291667, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:6, 20, 22], .03708334, atol=1e-7, rtol=0)
    assert arrays['n'][6, 20, 22] == pytest.approx(.03708334, abs=1e-7)


def test_native_cutcell_collects_oblique_north_and_east_face_points():
    vertices = ((5., 17.25), (6., 52.), (34.875, 51.), (35.125, 57.75), (87.375, 56.25),
                (86.9375, 41.), (96.6875, 40.5), (95.9375, 14.5), (5., 17.25))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 21.925199508666992),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:9, 4, 8], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:8, 4, 8], .23707265, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:8, 4, 8], .01982176, atol=1e-7, rtol=0)
    assert arrays['n'][8, 4, 8] == pytest.approx(.01982176, abs=1e-7)


def test_native_cutcell_collects_orthogonal_west_and_north_face_points():
    vertices = ((13.4375, 201.), (74.125, 201.), (74.125, 210.25), (100., 210.25),
                (100., 170.), (13.4375, 170.), (13.4375, 201.))
    case = WindCase(Domain((16, 26, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 15.589799880981445),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:7, 20, 7], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:6, 20, 7], .32604164, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:6, 20, 7], .02618056, atol=1e-7, rtol=0)
    assert arrays['n'][6, 20, 7] == pytest.approx(.02618056, abs=1e-7)


def test_native_cutcell_retains_endpoints_on_nonsupporting_gridline():
    vertices = ((40.125, 15.), (40.125, 19.75), (10.8125, 19.5), (10.75, 40.5),
                (25.6875, 60.), (40.6875, 60.), (40.625, 72.75), (51.1875, 83.),
                (55.75, 83.), (55.8125, 63.25), (93.5, 63.5), (93.6875, 15.25), (40.125, 15.))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 7.7812),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:4, 6, [2, 4]], 7)
    np.testing.assert_array_equal(arrays['building_volume_frac'][1:4, 6, 2], 1.)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 6, 4],
                               [.53437501, .53437501, .81569529], atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:4, 6, 4], [.059375, .059375, .11111111], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(arrays['icellflag'][1:4, 5, 4], [0, 0, 7])
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 5, 4],
                               [.02291667, .02291667, .51882863], atol=1e-7, rtol=0)
    assert arrays['f'][3, 5, 4] == pytest.approx(.01, abs=1e-7)


def test_native_cutcell_keeps_east_face_vertex_before_paired_south_crossings():
    vertices = ((6.6875, 105.5), (162.5, 100.25), (159.5, 9.), (117.5625, 10.25),
                (120., 83.25), (111.4375, 83.5), (109.9375, 37.75), (71.5625, 39.),
                (73.0625, 84.), (63.9375, 84.25), (63.1875, 60.75), (28.5625, 62.),
                (29.4375, 88.5), (23.0625, 88.75), (22.125, 59.75), (5.25, 60.25), (6.6875, 105.5))
    case = WindCase(Domain((24, 18, 24), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 32.5273),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:11, 8, 11], 0)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:11, 8, 11], .02395833, atol=1e-7, rtol=0)
    assert arrays['icellflag'][11, 8, 11] == 7
    assert arrays['building_volume_frac'][11, 8, 11] == pytest.approx(.15756607, abs=1e-7)


def test_native_cutcell_keeps_south_vertex_as_the_paired_exit():
    vertices = ((27.4375, 20.5), (28.375, 60.5), (67.25, 59.5), (67., 49.), (75.5, 49.),
                (75.6875, 58.), (83.3125, 57.75), (82.5, 25.5), (69.4375, 26.), (69.375, 22.5),
                (45.9375, 23.25), (45.875, 20.), (27.4375, 20.5))
    case = WindCase(Domain((24, 18, 24), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 47.7373),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:17, 2, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:16, 2, 4], .50073129, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][16, 2, 4] == pytest.approx(.70969403, abs=1e-7)
    assert arrays['f'][16, 2, 4] == pytest.approx(.01, abs=1e-7)


def test_native_cutcell_sorts_horizontal_pair_with_opposite_y_crossings():
    vertices = ((48.4375, 166.25), (49., 189.5), (41.9375, 189.5), (42.5, 211.25),
                (80., 210.), (80., 160.), (48.4375, 166.25))
    case = WindCase(Domain((18, 28, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 20.132999420166016),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:8, 18, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:7, 18, 4], .76045841, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['m'][1:7, 18, 4], .05982863, atol=1e-7, rtol=0)
    assert arrays['n'][7, 18, 4] == pytest.approx(.05982863, abs=1e-7)


def test_native_cutcell_sorts_oblique_pairs_with_opposite_y_crossings():
    vertices = ((77.875, 30.), (46.125, 30.75), (46.5625, 49.5), (41.875, 49.75),
                (42.0625, 57.25), (46.6875, 57.), (47.0625, 72.), (78.875, 71.25), (77.875, 30.))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 19.490100860595703),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:8, 4:6, 4], 7)
    for j, fraction, coefficient in ((4, .56451643, .04727934), (5, .28049213, .03476301)):
        np.testing.assert_allclose(arrays['building_volume_frac'][1:7, j, 4], fraction, atol=1e-7, rtol=0)
        assert arrays['n'][7, j, 4] == pytest.approx(coefficient, abs=1e-7)


def test_native_cutcell_assigns_paired_south_vertex_to_face():
    vertices = ((47.625, 17.5), (46.9375, 40.25), (41.5625, 40.), (41.0625, 58.5),
                (51.6875, 59.), (52., 49.), (62., 49.25), (62.9375, 18.), (47.625, 17.5))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 27.423599243164062),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:11, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:10, 4, 4], .32340121, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays['h'][1:10, 4, 4], .00156250, atol=1e-7, rtol=0)
    assert arrays['h'][10, 4, 4] == pytest.approx(.00880863, abs=1e-7)


def test_native_cutcell_retains_south_slot_for_later_west_north_crossing():
    vertices = ((159.3125, 128.75), (166.3125, 124.5), (170.5, 103.5), (171.25, 87.5),
                (164.375, 76.), (164.625, 62.75), (144.5625, 29.25), (121., 16.75),
                (100.3125, 13.75), (78.3125, 26.75), (74.125, 19.75), (50.5, 34.), (30., 60.),
                (35.3125, 69.), (26.125, 84.), (24.625, 96.75), (20.375, 99.25), (19.0625, 111.75),
                (27.9375, 147.5), (30.5, 151.75), (52.3125, 138.75), (59.625, 151.),
                (81.4375, 155.5), (98.4375, 153.25), (121.1875, 139.5), (125.8125, 147.25),
                (134.0625, 142.5), (142.75, 157.), (152.75, 151.), (154.8125, 141.5),
                (163.8125, 136.25), (159.3125, 128.75))
    case = WindCase(Domain((24, 18, 24), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 24.1644),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:10, 12, 16], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:9, 12, 16], .30624998, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][9, 12, 16] == pytest.approx(.85102260, abs=1e-7)
    np.testing.assert_array_equal(arrays['h'][1:9, 12, 16], 0.)
    assert arrays['h'][9, 12, 16] == pytest.approx(.00083690, abs=1e-7)


def test_native_cutcell_retains_south_slot_for_later_north_east_crossing():
    short = ((243.4375, 706.75), (242.375, 701.), (220.0625, 705.25),
             (221.125, 711.), (243.4375, 706.75))
    tall = ((323.8125, 673.75), (215.4375, 675.), (215.75, 701.), (222.125, 701.),
            (222.0625, 695.25), (294.0625, 694.25), (294., 687.25), (323.9375, 686.75),
            (323.8125, 673.75))
    buildings = (PolygonBuilding(short, 11.712499618530273), PolygonBuilding(tall, 21.976299285888672))
    case = WindCase(Domain((35, 73, 10), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    buildings, mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:9, 70, 22], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:4, 70, 22], .44829795, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][4, 70, 22] == pytest.approx(.80052710, abs=1e-7)
    np.testing.assert_array_equal(arrays['m'][1:4, 70, 22], 0.)
    np.testing.assert_array_equal(arrays['n'][1:5, 70, 22], 0.)


def test_native_cutcell_uses_earlier_west_crossing_before_north_vertex():
    vertices = ((35.1875, 49.5), (45.375, 50.5), (44., 45.5), (36., 41.),
                (29.5625, 41.5), (29.1875, 44.75), (35.1875, 49.5))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 18.063499450683594),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:8, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:7, 4, 4], .66666663, atol=1e-7, rtol=0)
    assert arrays['building_volume_frac'][7, 4, 4] == 1.
    np.testing.assert_array_equal(arrays['m'][1:7, 4, 4], 0.)
    np.testing.assert_array_equal(arrays['n'][1:8, 4, 4], 0.)


def test_native_cutcell_flags_zero_area_east_endpoint():
    vertices = ((50.25, 67.75), (13.25, 68.25), (13.5, 92.), (81.125, 91.25),
                (80.625, 48.25), (50., 48.5), (50.25, 67.75))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 5.0195),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:3, 4, 4], 7)
    np.testing.assert_array_equal(arrays['building_volume_frac'][1:3, 4, 4], 1.)


def test_native_cutcell_revisits_east_crossing_after_north_vertex():
    vertices = ((44.875, 37.5), (45.25, 50.), (53.5, 49.75), (54.3125, 75.75),
                (70.3125, 75.25), (69.5625, 51.75), (115., 50.25), (114.5, 35.25), (44.875, 37.5))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 4.973100185394287),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:3, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:3, 4, 4],
                               [.17499995, .56402970], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(arrays['h'][1:3, 4, 4], 0.)
    np.testing.assert_array_equal(arrays['n'][1:3, 4, 4], 0.)


def test_native_cutcell_keeps_later_sentinel_after_two_west_vertex_cuts():
    vertices = ((44.6875, 31.5), (46.625, 41.75), (37.6875, 43.25), (38.5, 47.75),
                (44.5625, 46.5), (45.4375, 50.75), (57.75, 48.5), (54.125, 29.75), (44.6875, 31.5))
    case = WindCase(Domain((16, 16, 20), (10., 10., 3.)), SensorProfile(5., 270., 10., .1),
                    (PolygonBuilding(vertices, 3.873500108718872),), mesh_type=1, native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:3, 4, 4], 7)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:3, 4, 4],
                               [.39471740, .87974316], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(arrays['h'][1:3, 4, 4], 0.)
    np.testing.assert_array_equal(arrays['n'][1:3, 4, 4], 0.)


def test_native_cutcell_skips_gridline_edge_already_consumed_in_adjacent_cell():
    domain = Domain((14, 16, 24), (10., 10., 3.))
    vertices = ((43.8125, 43.5), (30.0625, 43.5), (30.25, 68.), (38.3125, 68.),
                (38.375, 73.5), (46.6875, 73.5), (46.75, 80.), (58.4375, 80.),
                (58.3125, 61.), (43.9375, 61.), (43.8125, 43.5))
    building = PolygonBuilding(vertices, 5.264800071716309)
    case = WindCase(domain, SensorProfile(5., 270., 10., .1), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    # The first contact is owned by the lower cell's earlier vertex walk.
    np.testing.assert_array_equal(arrays['icellflag'][1:3, 8, 4:6], [[1, 7], [1, 7]])
    # The lower cell keeps both consecutive vertices and the west entry.
    np.testing.assert_allclose(arrays['building_volume_frac'][1:3, 7, 4],
                               [.63552082, .85302389], rtol=0, atol=1e-7)
    np.testing.assert_allclose(arrays['f'][1:3, 7, 4], [.0065, .00735773], rtol=0, atol=1e-7)


def test_native_cutcell_combines_vertex_with_later_through_edge():
    domain = Domain((22, 14, 28), (10., 10., 3.))
    vertices = ((161.3125, 6.), (118.875, 7.), (119.1875, 18.5), (130.3125, 18.25),
                (130.5, 25.75), (114.5, 26.25), (114., 5.75), (8.4375, 8.75),
                (9.4375, 48.5), (31., 48.), (30.875, 44.), (37.1875, 44.),
                (37.3125, 50.25), (120.3125, 48.25), (121., 75.), (163.125, 73.75),
                (161.3125, 6.))
    building = PolygonBuilding(vertices, 49.1349)
    case = WindCase(domain, SensorProfile(5., 270., 10., .1), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:17, 1, 11], 0)
    np.testing.assert_allclose(arrays['building_volume_frac'][1:17, 1, 11], .032374084, rtol=0, atol=1e-7)
    assert arrays['icellflag'][17, 1, 11] == 7


def test_native_cutcell_preserves_vertex_face_samples_in_adjacent_columns():
    domain = Domain((17, 16, 28), (10., 10., 3.))
    vertices = ((108.8125, 69.75), (109.625, 35.), (118.625, 35.), (119.1875, 11.),
                (15., 8.75), (14.3125, 37.25), (51.25, 38.), (51.25, 36.25),
                (60.375, 36.25), (59.625, 70.5), (14.5625, 69.5), (13.9375, 97.25),
                (117.125, 99.75), (117.75, 70.), (108.8125, 69.75))
    building = PolygonBuilding(vertices, 45.0175)
    case = WindCase(domain, SensorProfile(5., 270., 10., .1), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    np.testing.assert_array_equal(arrays['icellflag'][1:16, 6, 10:12], np.tile([0, 7], (15, 1)))
    expected = np.tile([.08972228, .86803341], (15, 1))
    np.testing.assert_allclose(arrays['building_volume_frac'][1:16, 6, 10:12], expected, rtol=0, atol=2e-7)


def test_decimal_cutcell_keeps_intersections_on_their_grid_faces():
    domain = Domain((28, 28, 28), (1.3, 1.3, 1.3))
    vertices = ((8.3, 8.7), (8.3, 14.1), (14.2, 14.1), (14.2, 8.7), (8.3, 8.7))
    building = PolygonBuilding(vertices, 4.4, .1)
    case = WindCase(domain, SensorProfile(2., 247., 15.), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    # CutBuilding.cpp reference values for the south and east prism faces.
    np.testing.assert_allclose(arrays['building_volume_frac'][2, 6, 8], .69230783, rtol=0, atol=1e-6)
    np.testing.assert_allclose(arrays['building_volume_frac'][2, 8, 10], .07692277, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(arrays['icellflag'][1:4, 7:10, 10], 0)


def test_native_raised_cutcell_uses_float32_vertical_faces():
    domain = Domain((28, 28, 28), (1.3, 1.3, 1.3))
    vertices = ((8.3, 8.7), (8.3, 14.1), (14.2, 14.1), (14.2, 8.7), (8.3, 8.7))
    building = PolygonBuilding(vertices, 4.4, 3.9)
    case = WindCase(domain, SensorProfile(2., 247., 15.), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    # Native float32 face 4 falls just below the specified building base.
    np.testing.assert_array_equal(arrays['icellflag'][3:5, 8, 10], [1, 0])
    np.testing.assert_allclose(arrays['building_volume_frac'][3:5, 8, 10],
                               [1., .07692277], rtol=0, atol=1e-6)


def test_native_building_volume_subtracts_in_float32_before_solid_threshold():
    domain = Domain((10, 10, 12), (10., 10., 3.))
    building = PolygonBuilding.rectangle(22.5, 21, 37, 39, 15)
    case = WindCase(domain, SensorProfile(2., 270., 10.), (building,), mesh_type=1,
                    native_staircase_metrics=True)
    arrays = case.initialize_geometry().arrays

    # The footprint blocks exactly 90 m² of each 100 m² cell in this column.
    column = (slice(1, 5), 2, 3)
    np.testing.assert_array_equal(arrays['building_volume_frac'][column], np.float32(1.)-np.float32(.9))
    np.testing.assert_array_equal(arrays['icellflag'][column], 7)


def test_terrain_exactly_on_accumulated_float32_face_is_not_a_cut_cell():
    domain = Domain((4, 4, 8), (1., 1., 1.), (.37,)*4+(.91,)*4)
    workspace = domain.workspace()
    for name in ('ni', 'nj', 'nk', 'wall_distance', 'terrain_volume_frac', 'center_id'):
        workspace.arrays[name] = np.ones(domain.cell_shape)
    _, faces = native_vertical_coordinates(domain.dz)
    face = 7
    assert np.float32(domain.coordinates['z_face'][face]) < faces[face]
    corners = np.full((3, 3, 4), faces[face])
    apply_cutcell_terrain(workspace, corners)
    assert workspace.arrays['icellflag'][face, 1, 1] == 1
    assert workspace.arrays['icellflag'][face-1, 1, 1] == 2


def test_native_cutcell_wall_coefficients():
    if not (FIXTURE/'wall_native.csv').exists():
        pytest.skip('Run scripts/check_python_cutcell.py --native')
    records = np.loadtxt(FIXTURE/'wall_inputs.txt').astype(np.float32).astype(float)
    expected = np.loadtxt(FIXTURE/'wall_native.csv', delimiter=',')
    domain = Domain((7, 6, 6), (.7, .9, .6), (.6, .8, 1.1, 1.2, 1.6, 2.))
    for block, reference in zip(np.split(records, 4), np.split(expected, 4), strict=True):
        workspace = domain.workspace()
        for index, key in enumerate(('icellflag', 'building_volume_frac', *'efghmn')):
            workspace.arrays[key] = block[:, index].reshape(domain.cell_shape)
        cutcell_wall_stencil(workspace)
        np.testing.assert_array_equal(workspace.arrays['icellflag'].ravel(), reference[:, 0])
        for index, key in enumerate('efghmn', start=1):
            np.testing.assert_allclose(workspace.arrays[key].ravel(), reference[:, index], atol=1e-8, rtol=5e-7)


@pytest.mark.parametrize('kind, cut_flag, count', [('terrain', 8, 509), ('building', 7, 62)])
def test_cutcell_replay_and_coefficient_reload(kind, cut_flag, count, tmp_path):
    folder = FIXTURE/kind
    reference = folder/'native_windsWk.nc'
    if not reference.exists():
        pytest.skip('Run scripts/check_python_cutcell.py --native')
    expected = read_wind_workspace(reference).arrays
    assert np.count_nonzero(expected['icellflag'] == cut_flag) == count
    output = tmp_path/'winds.nc'
    assert replay_wind(reference, output)['records'] == 1
    actual = read_wind_workspace(output).arrays
    for key in ('icellflag', 'terrain_volume_frac', 'building_volume_frac', *'efghmn'):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in 'uvw':
        np.testing.assert_allclose(actual[key], expected[key], atol=4e-6, rtol=0)
    loaded = read_wind_case(folder/'reload.xml', compute_turbulence=False).initialize().arrays
    assert not np.isin(loaded['icellflag'], (7, 8)).any()
    for key in 'efghmn':
        np.testing.assert_array_equal(loaded[key], expected[key])


def test_wind_replay_ignores_sparse_auxiliary_fields_but_requires_initial_wind(tmp_path):
    from netCDF4 import Dataset

    workspace = Domain((6, 6, 6), (1., 1., 1.), (1., 1.5, 2., 2.5, 3., 3.5)).workspace()
    source, output = (tmp_path/f'{name}.nc' for name in ('source', 'replay'))
    write_netcdf(source, workspace.arrays, workspace.coordinates)
    with Dataset(source, 'a') as nc:
        nc.createVariable('quality_diagnostic', 'i4', ('t', 'z', 'y', 'x'))[0, 0, 0, 0] = 1
        for name in ('x_face', 'y_face', 'z_face'):
            nc.renameVariable(name, 'archived_'+name)
    assert replay_wind(source, output)['records'] == 1
    with _open_netcdf(output) as nc:
        assert 'quality_diagnostic' not in nc.variables
        np.testing.assert_array_equal(nc['u'][0], workspace.arrays['u0'])
        for name in ('x_face', 'y_face', 'z_face'):
            np.testing.assert_array_equal(nc[name][:], workspace.coordinates[name])
    with Dataset(source, 'a') as nc:
        nc['u0'][0, 0, 0, 0] = np.ma.masked
    previous = output.read_bytes()
    with pytest.raises(ValueError, match='Missing values in workspace field u0'):
        replay_wind(source, output)
    assert output.read_bytes() == previous


@pytest.mark.parametrize('uniform', [True, False])
def test_wind_replay_infers_only_uniform_missing_vertical_widths(tmp_path, uniform):
    from netCDF4 import Dataset

    widths = None if uniform else (1., 1.5, 2., 2.5, 3., 3.5)
    workspace = Domain((6, 6, 6), (1., 1., 1.), widths).workspace()
    source, output = (tmp_path/f'{name}.nc' for name in ('source', 'replay'))
    write_netcdf(source, workspace.arrays, workspace.coordinates)
    with Dataset(source, 'a') as nc:
        nc.renameVariable('dz_array', 'archived_dz_array')
    if not uniform:
        with pytest.raises(ValueError, match='nonuniform vertical centers'):
            replay_wind(source, output)
        assert not output.exists()
        return
    assert replay_wind(source, output)['records'] == 1
    restored = read_wind_workspace(output)
    np.testing.assert_array_equal(restored.arrays['dz_array'], workspace.arrays['dz_array'])
    np.testing.assert_array_equal(restored.coordinates['z_face'], workspace.coordinates['z_face'])


def test_native_wind_replay_keeps_rebuilt_stencil_between_records(tmp_path):
    from plumeswift.io import netcdf_series

    workspace = Domain((6, 6, 6), (1., 1., 1.)).workspace()
    arrays = workspace.arrays
    fields = {name: arrays[name] for name in ('u0', 'v0', 'w0', 'dz_array', 'icellflag')}
    obstructed = arrays['icellflag'].copy()
    obstructed[2, 2, 2] = 0
    source, output = (tmp_path/f'{name}.nc' for name in ('source', 'replay'))
    with netcdf_series(source, workspace.coordinates) as write:
        write(fields | {'icellflag': obstructed}, '2020-01-01T00:00:00', 0.)
        write(fields, '2020-01-01T00:00:01', 1.)

    with pytest.raises(ValueError, match='Workspace missing e'):
        replay_wind(source, output)
    assert replay_wind(source, output, native_input=True)['records'] == 2
    with _open_netcdf(output) as nc:
        np.testing.assert_array_equal(nc['icellflag'][:, 2, 2, 2], [0, 1])
        np.testing.assert_array_equal(nc['e'][:, 2, 2, 1], [0., 0.])
        np.testing.assert_array_equal(nc['e'][:, 2, 3, 1], [1., 1.])


@pytest.mark.parametrize('kind, cut_count', [('terrain', 509), ('steep', 1304), ('saddle', 1760)])
def test_generated_dem_cutcell_wind_and_turbulence(kind, cut_count):
    pytest.importorskip('rasterio')
    folder = FIXTURE/kind
    if not (folder/'native_turbOut.nc').exists():
        pytest.skip('Run scripts/check_python_cutcell_terrain.py --native')
    result = read_wind_case(folder/'winds.xml').run()
    actual = result['workspace'].arrays
    reference = read_wind_workspace(folder/'native_windsWk.nc').arrays
    assert np.count_nonzero(actual['icellflag'] == 8) == cut_count
    for key in ('terrain', 'icellflag'):
        np.testing.assert_array_equal(actual[key], reference[key])
    for key in ('terrain_volume_frac', *'efghmn'):
        np.testing.assert_allclose(actual[key], reference[key], atol=1.2e-6, rtol=0)
    for key in 'uvw':
        np.testing.assert_allclose(actual[key][:-1], reference[key][:-1], atol=3e-6, rtol=0)
    with _open_netcdf(folder/'native_turbOut.nc') as fields:
        expected = np.ma.filled(fields['CoEps'][0], np.nan)
        finite = np.isfinite(expected)
        assert not finite.all()
        actual_eps = result['turbulence']['CoEps']
        assert np.isfinite(actual_eps).all()
        np.testing.assert_allclose(actual_eps[finite], expected[finite], atol=2e-6, rtol=0)


def test_generated_cutcell_supports_repeated_vertical_widths():
    pytest.importorskip('rasterio')
    folder = FIXTURE/'stretch'
    if not (folder/'winds.xml').exists():
        pytest.skip('Run scripts/check_python_cutcell_terrain.py --native')
    case = read_wind_case(folder/'winds.xml')
    assert case.domain.vertical_widths == (.5,)*4+(1.25,)*20
    workspace = case.run()['workspace']
    assert np.count_nonzero(workspace.arrays['icellflag'] == 8) == 521


@pytest.mark.parametrize('name', ('building', 'rotated', 'raised', 'on_dem', 'aligned', 'concave', 'overlap'))
def test_generated_building_cutcell_matches_native(name):
    folder = FIXTURE/name
    if not (folder/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_cutcell_building.py --native')
    if name == 'on_dem':
        pytest.importorskip('rasterio')
    result = read_wind_case(folder/'winds.xml').run()
    actual = result['workspace'].arrays
    reference = read_wind_workspace(folder/'native_windsWk.nc').arrays
    assert np.any(actual['icellflag'] == 7)
    np.testing.assert_array_equal(actual['icellflag'], reference['icellflag'])
    for key in ('building_volume_frac', 'terrain_volume_frac', *'efghmn'):
        np.testing.assert_allclose(actual[key], reference[key], atol=4e-6, rtol=0)
    for key in 'uvw':
        np.testing.assert_allclose(actual[key][:-1], reference[key][:-1],
                                   atol=4e-5 if name == 'building' else 5e-6, rtol=0)
    assert np.isfinite(result['turbulence']['CoEps']).all()
