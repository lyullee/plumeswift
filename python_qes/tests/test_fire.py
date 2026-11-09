from dataclasses import replace
from io import StringIO
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.contracts import UnportedFeature
from plumeswift.cuda_fire_potential import CudaHeatPotential
from plumeswift.domain import Domain
from plumeswift.fire_case import read_fire_config
from plumeswift.fire_fuels import read_fuel_raster
from plumeswift.fire_fuels_legacy import HISTORICAL_FUELS
from plumeswift.fire_grid import FireGrid, Ignition, TimedIgnition, read_timed_ignitions
from plumeswift.fire_potential import HeatPotential
from plumeswift.fire_run import read_coupled_fire, run_legacy_fire, write_fire_run
from plumeswift.fire_spread import balbi
from plumeswift.terrain import GridTerrain
from plumeswift.turbulence import LocalTurbulence

ROOT = Path(__file__).resolve().parents[2]


def test_historical_fuel_lookup_preserves_native_types_and_names():
    assert len(HISTORICAL_FUELS) == 18
    chaparral = HISTORICAL_FUELS['Chaparral']
    assert type(chaparral.fgi) is np.float32 and chaparral.fgi == np.float32(2.468)
    assert type(chaparral.cmbcnst) is int and chaparral.cmbcnst == 17433000
    assert HISTORICAL_FUELS['TimberClosedLitter'].savr == 1889
    assert all(value == 0 for value in vars(HISTORICAL_FUELS['Water']).values())
    with pytest.raises(KeyError):
        HISTORICAL_FUELS['GR1']


@pytest.mark.parametrize('section', ['.', './boundaryConditions', './particleParameters/baseParticle/source/sourceGeometry_Point'])
def test_legacy_fire_plume_ignores_xml_metadata(tmp_path, section):
    source = ROOT/'data/cases/empty_west'
    winds = ET.parse(source/'winds.xml')
    winds.getroot().append(ET.fromstring('''<fires><fireDur>1</fireDur><fuelType>1</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    wind_path, plume_path = tmp_path/'winds.xml', tmp_path/'plume.xml'
    winds.write(wind_path)
    plume = ET.parse(source/'plume.xml')
    expected = read_coupled_fire(wind_path, plume=source/'plume.xml', legacy_plume_xml=True)
    ET.SubElement(plume.find(section), 'caseNote').text = 'diagnostic metadata'
    plume.write(plume_path)
    actual = read_coupled_fire(wind_path, plume=plume_path, legacy_plume_xml=True)
    assert actual.smoke == expected.smoke
    assert actual.plume_models == expected.plume_models
    with pytest.raises(UnportedFeature, match='caseNote'):
        read_coupled_fire(wind_path, plume=plume_path)


@pytest.mark.parametrize('section, label', [('.', 'fires'), ('./ignition', 'ignition')])
def test_legacy_fire_config_ignores_xml_metadata(tmp_path, section, label):
    path = tmp_path/'fire.xml'
    root = ET.fromstring('''<fires><fireDur>1</fireDur><fuelType>106</fuelType>
      <fmc>.06</fmc><cure>.68</cure><courant>.05</courant>
      <ignition><xStart>4</xStart><yStart>5</yStart><length>2</length><width>2</width>
        <height>2</height><baseHeight>0</baseHeight></ignition></fires>''')
    ET.ElementTree(root).write(path)
    expected = read_fire_config(path)
    ET.SubElement(root.find(section), 'caseNote').text = 'diagnostic metadata'
    ET.ElementTree(root).write(path)
    assert read_fire_config(path, legacy_fire_xml=True) == expected
    with pytest.raises(ValueError, match=f'Unknown {label} XML field'):
        read_fire_config(path)


def test_verbose_legacy_fire_reports_interval_and_step_on_progress_stream(tmp_path):
    winds = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    winds.find('simulationParameters/domain').text = '8 8 20'
    winds.find('.//timeStamp').text = '2020-01-01T00:00:00'
    winds.getroot().append(ET.fromstring('''<fires><fireDur>.25</fireDur><fuelType>99</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    progress = StringIO()
    result = run_legacy_fire(path, None, no_fire_winds=True, verbose=True, progress_stream=progress)
    assert result['records'] == 1
    assert progress.getvalue().splitlines() == [
        '[QES-Fire]\t Fire simulation stating from 2020-01-01T00:00:00 to 2020-01-01T00:00:00.250000.',
        '[QES-Fire]\t Fire step completed.',
        '[QES-Fire]\t Current time = 2020-01-01T00:00:01',
    ]
    progress = StringIO()
    run_legacy_fire(path, None, no_fire_winds=True, progress_stream=progress)
    assert progress.getvalue() == ''


def test_qesfire_preserves_trailing_separator_in_output_basename(tmp_path):
    winds = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    winds.find('simulationParameters/domain').text = '8 8 20'
    winds.getroot().append(ET.fromstring('''<fires><fireDur>.25</fireDur><fuelType>99</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    output = tmp_path/'out'
    output.mkdir()
    result = run_legacy_fire(path, None, basename=f'{output}/', fire_output=True, no_fire_winds=True)
    assert result['records'] == 1
    assert {item.name for item in output.iterdir()} == {'_fireOutput.nc'}
    assert not (tmp_path/'out_fireOutput.nc').exists()


def test_legacy_fire_accepts_signed_duration_for_zero_steps(tmp_path):
    winds = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    winds.getroot().append(ET.fromstring('''<fires><fireDur>-.25</fireDur><fuelType>1</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    with pytest.raises(ValueError, match='Invalid fire duration'):
        read_fire_config(path)

    summary = run_legacy_fire(path, None, basename=str(tmp_path/'out'), fire_output=True, no_fire_winds=True)
    assert summary['records'] == 0 and summary['first_output'] is None
    assert (tmp_path/'out_fireOutput.nc').is_file()


def test_legacy_fire_defaults_to_native_release_indexing_with_plume(tmp_path):
    source = ROOT/'data/cases/empty_west'
    winds = ET.parse(source/'winds.xml')
    winds.getroot().append(ET.fromstring('''<fires><fireDur>0</fireDur><fuelType>1</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    plume_path = source/'plume.xml'
    default = run_legacy_fire(path, plume_path, no_fire_winds=True)
    corrected = run_legacy_fire(path, plume_path, no_fire_winds=True, native_release_indexing=False)
    assert default['smoke']['native_release_indexing'] is True
    assert 'native_release_indexing' not in corrected['smoke']
    assert not read_coupled_fire(path, plume=plume_path).native_release_indexing
    assert 'smoke' not in run_legacy_fire(path, None, no_fire_winds=True)


def test_legacy_fire_warns_once_when_mesh_reflection_has_no_terrain(tmp_path):
    source = ROOT/'data/cases/empty_west'
    winds = ET.parse(source/'winds.xml')
    winds.getroot().append(ET.fromstring('''<fires><fireDur>0</fireDur><fuelType>1</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    wind_path, plume_path = tmp_path/'winds.xml', tmp_path/'plume.xml'
    winds.write(wind_path)
    plume = ET.parse(source/'plume.xml')
    plume.find('boundaryConditions/wallReflection').text = 'meshReflection'
    plume.write(plume_path)
    output = StringIO()
    run_legacy_fire(wind_path, plume_path, no_fire_winds=True, progress_stream=output)
    assert output.getvalue().count('[WARRNING] mesh wall reflection method not available') == 1
    plume.find('boundaryConditions/wallReflection').text = 'stairstepReflection'
    plume.write(plume_path)
    output = StringIO()
    run_legacy_fire(wind_path, plume_path, no_fire_winds=True, progress_stream=output)
    assert 'mesh wall reflection method not available' not in output.getvalue()


@pytest.mark.parametrize(('flags', 'enabled', 'native'), [
    ([], False, True),
    (['-l'], True, True),
    (['--doParticleDataOutput'], True, True),
    (['--native-particle-csv'], True, True),
    (['--precise-particle-csv'], True, False),
    (['-l', '--precise-particle-csv'], True, False),
])
def test_legacy_fire_cli_defaults_to_native_particle_csv(monkeypatch, flags, enabled, native):
    from plumeswift import fire_run
    from plumeswift.__main__ import main

    def run(wind, plume, **options):
        assert options['particle_csv'] is enabled
        assert options['native_particle_csv'] is native
        return {}

    monkeypatch.setattr(fire_run, 'run_legacy_fire', run)
    main(['qesFire', '-q', 'wind.xml', '-p', 'plume.xml', '-o', 'case', *flags])


def test_legacy_fire_rejects_reserved_smoke_output_tag(tmp_path):
    source = ROOT/'data/cases/empty_west'
    winds = ET.parse(source/'winds.xml')
    winds.getroot().append(ET.fromstring('''<fires><fireDur>1</fireDur><fuelType>1</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant></fires>'''))
    wind_path, plume_path = tmp_path/'winds.xml', tmp_path/'plume.xml'
    winds.write(wind_path)
    plume = ET.parse(source/'plume.xml')
    plume.find('particleParameters/baseParticle/tag').text = 'smoke'
    plume.write(plume_path)

    with pytest.raises(ValueError, match="'smoke' is reserved"):
        run_legacy_fire(wind_path, plume_path, basename=str(tmp_path/'out'), fire_output=True, no_fire_winds=True)
    assert not list(tmp_path.glob('out_*.nc'))


def test_coupled_full_level_set_applies_spread_outside_narrow_band(tmp_path):
    winds = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    winds.find('simulationParameters/domain').text = '16 16 20'
    winds.getroot().append(ET.fromstring('''<fires><fireDur>.25</fireDur><fuelType>106</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant>
        <ignition><xStart>6</xStart><yStart>7</yStart><length>1</length><width>1</width>
          <height>2</height><baseHeight>0</baseHeight></ignition></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    full = read_coupled_fire(path, narrow_band=False)
    narrow = read_coupled_fire(path)
    assert next(full.frames()).fields['Force'][1, 1] > 0
    assert next(narrow.frames()).fields['Force'][1, 1] == 0
    assert write_fire_run(full, tmp_path/'full')['level_set'] == 'full'


def test_qesfire_zero_wrf_steps_writes_empty_outputs_without_solving(tmp_path, monkeypatch):
    from netCDF4 import Dataset
    from plumeswift.wind_case import WindCase

    fixture = ROOT/'results/python_port/wrf_startup'
    if not (fixture/'wrf.nc').is_file():
        pytest.skip('Run the WRF startup fixture to prepare the coupled input')
    winds = ET.parse(fixture/'winds.xml')
    winds.find('simulationParameters/WRF').text = str(fixture/'wrf.nc')
    winds.find('simulationParameters/totalTimeIncrements').text = '0'
    path = tmp_path/'winds.xml'
    winds.write(path)

    def unexpected_solve(*args, **kwargs):
        pytest.fail('Zero wind steps should not invoke the pressure solver')

    monkeypatch.setattr(WindCase, 'solve', unexpected_solve)
    summary = run_legacy_fire(path, None, basename=tmp_path/'case', fire_output=True,
                              turbulence=True, no_fire_winds=True, wrf_timeout=3)
    assert summary['records'] == 0
    assert summary['first_output'] is None and summary['last_output'] is None
    for name in ('fireOutput', 'turbOut'):
        with Dataset(tmp_path/f'case_{name}.nc') as output:
            assert len(output.dimensions['t']) == 0


def test_fire_smoke_reflects_against_prepared_cut_cell_mesh(tmp_path):
    class CutCellTerrain:
        def mesh(self, domain):
            raise AssertionError('Coupled smoke should use the prepared wind mesh')

        def cut_geometry(self, domain, halo):
            self.prepared_mesh, corners = GridTerrain(np.zeros((8, 8))).cut_geometry(domain, halo)
            return self.prepared_mesh, corners

    source = ROOT/'data/cases/empty_west'
    winds = ET.parse(source/'winds.xml')
    winds.find('simulationParameters/domain').text = '8 8 20'
    winds.getroot().append(ET.fromstring('''<fires><fireDur>1</fireDur><fuelType>106</fuelType>
        <fmc>.06</fmc><cure>.68</cure><courant>.05</courant>
        <ignition><xStart>4</xStart><yStart>4</yStart><length>1</length><width>1</width>
          <height>2</height><baseHeight>0</baseHeight></ignition></fires>'''))
    path = tmp_path/'winds.xml'
    winds.write(path)
    case = read_coupled_fire(path, plume=source/'plume.xml')
    terrain = CutCellTerrain()
    case = replace(case, wind=replace(case.wind, terrain=terrain, mesh_type=1),
                   smoke=replace(case.smoke, wall='meshReflection'), plume_models=None)
    meshes = []
    with np.errstate(divide='ignore', invalid='ignore'):
        frames = tuple(case.frames(on_smoke_step=lambda model: meshes.append(model.mesh)))
    assert frames and meshes
    assert all(mesh is terrain.prepared_mesh for mesh in meshes)


def test_legacy_fire_cut_cell_zero_mixing_keeps_native_nan():
    domain = Domain((5, 5, 20), (1., 1., 1.))
    workspace = domain.workspace()
    velocity = {name: np.full(domain.face_shape, value) for name, value in zip('uvw', (1., 0., 0.))}
    distance = np.ones(domain.cell_shape)
    center = (1, 2, 2)
    distance[center] = 0.
    conventional = LocalTurbulence(background=.3, native_math=True)
    legacy = replace(conventional, native_zero_mixing_nan=True)
    modern_fields, _ = conventional.build(workspace, velocity, mixing_distance=distance)
    native_fields, _ = legacy.build(workspace, velocity, mixing_distance=distance)
    for name in ('CoEps', 'tke', 'txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz'):
        assert np.isfinite(modern_fields[name]).all()
        np.testing.assert_array_equal(np.argwhere(np.isnan(native_fields[name])), [center])
        finite = np.isfinite(native_fields[name])
        np.testing.assert_array_equal(native_fields[name][finite], modern_fields[name][finite])


@pytest.mark.parametrize('spacing', [(.75, .75), (1.25, 1.25)])
def test_fire_fuel_raster_samples_rotated_band_by_pixel(tmp_path, spacing):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import Affine

    codes = np.arange(101, 126, dtype=np.uint8).reshape(5, 5)
    paths = tmp_path/'north_up.tif', tmp_path/'rotated.tif'
    for path, (b, d) in zip(paths, ((0., 0.), (.25, .25)), strict=True):
        transform = Affine(1., b, 1000., d, -1., 2000.)
        with rasterio.open(path, 'w', driver='GTiff', height=5, width=5, count=1,
                           dtype='uint8', transform=transform) as raster:
            raster.write(codes, 1)
    np.testing.assert_array_equal(read_fuel_raster(paths[1], (4, 4), *spacing),
                                  read_fuel_raster(paths[0], (4, 4), *spacing))


def test_fire_fuel_raster_preserves_defined_south_up_sampling(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import Affine

    codes = np.arange(101, 117, dtype=np.uint8).reshape(4, 4)
    path = tmp_path/'south_up.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=4, width=4, count=1,
                       dtype='uint8', transform=Affine(1., 0., 1000., 0., 2., 2000.)) as raster:
        raster.write(codes, 1)
    np.testing.assert_array_equal(read_fuel_raster(path, (4, 4), 1., 1.), codes[::-1])
    np.testing.assert_array_equal(read_fuel_raster(path, (4, 4), .5, 1.), np.zeros((4, 4)))
    with pytest.raises(ValueError, match='negative native histogram'):
        read_fuel_raster(path, (4, 4), 1., 2.)


def test_fire_fuel_raster_preserves_defined_westward_sampling(tmp_path):
    import rasterio
    from rasterio.transform import Affine

    path = tmp_path/'westward.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=4, height=4, count=1, dtype='uint16',
                       transform=Affine(-2, 0, 1000, 0, -1, 2000)) as raster:
        raster.write(np.full((4, 4), 106, dtype=np.uint16), 1)
    np.testing.assert_array_equal(read_fuel_raster(path, (4, 4), 1., 1.), np.zeros((4, 4)))
    with pytest.raises(ValueError, match='negative native histogram size'):
        read_fuel_raster(path, (4, 4), 2., 1.)


def test_fire_potential_near_axis_uses_zero_radial_index():
    table = HeatPotential(np.array([[4., 5.], [6., 7.]]), np.array([[1., 2.], [3., 4.]]),
                          np.zeros(2), np.zeros(2), np.array([0., 1e-6]), np.array([0., 1.]))
    velocity = table._sample(np.array([0., 1.5e-6, 9e-6]), np.zeros(3), 1., 2.)
    np.testing.assert_array_equal(velocity, [[0.]*3, [0.]*3, [6.]*3])


def test_native_serial_far_field_rounds_trigonometric_coordinate():
    table = HeatPotential(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(2), np.zeros(2),
                          np.arange(2.), np.arange(2.))
    x, y, z = np.array([0., 1.]), np.zeros(2), 70.
    native = replace(table, native_math=True)
    velocity = native._sample(x, y, z, 1.)
    distance = np.hypot(x[1], z)
    angle = np.arctan(np.float32(x[1]/z))
    coordinate = float((np.float32(1)+np.cos(angle))*np.float32(.5))
    expected = 1/(2*np.pi*distance**3) + distance**(-1/3)*(5/6)*(
        (1-2*coordinate)/np.sqrt(coordinate*(1-coordinate)))
    assert velocity[0][1] == pytest.approx(expected, abs=1e-12)
    assert abs(table._sample(x, y, z, 1.)[0][1]-expected) > .001
    assert velocity[0][0] == 0
    assert np.isfinite(native._sample(np.array([30.]), np.zeros(1), 0., 1.)).all()


def test_native_serial_potential_rounds_dimensionless_lookup_coordinates(monkeypatch):
    from types import SimpleNamespace

    table = HeatPotential(np.zeros((5, 5)), np.arange(25.).reshape(5, 5), np.zeros(2), np.zeros(2),
                          np.arange(5.), .5*np.arange(5.))
    native = replace(table, native_math=True)
    fire = SimpleNamespace(terrain=np.zeros((3, 3)), dx=.2, dy=.3, dz=.2)
    monkeypatch.setattr(HeatPotential, '_merged_sources',
                        lambda self, fire, workspace: ([(1, 2, 0, 0, .1, 1., .2000000001)], None))

    def potential_for(potential):
        arrays = {axis+'0': np.zeros((5, 4, 4)) for axis in 'uvw'}
        arrays['icellflag'] = np.ones((4, 3, 3), dtype=int)
        fields, _ = potential.apply(fire, SimpleNamespace(arrays=arrays))
        return fields[2, 1, 0, 1]

    assert potential_for(table) == 0
    assert potential_for(native) == 6
    x = np.array([np.nextafter(1., 0.)])
    assert table._sample(x, np.zeros(1), .75, 1.)[2][0] == 5
    assert native._sample(x, np.zeros(1), .75, 1.)[2][0] == 6


@pytest.mark.parametrize('reader', [HeatPotential, CudaHeatPotential])
def test_native_heat_potential_reads_raw_values_and_paired_hyperslabs(tmp_path, reader):
    from netCDF4 import Dataset

    path = tmp_path/'heat.nc'
    with Dataset(path, 'w') as nc:
        nc.createDimension('lookup', 2)
        nc.createDimension('padded', 3)
        for key, dims, values in (
                ('u_r', ('lookup', 'lookup'), [[.1, 2], [3, 4]]),
                ('u_z', ('padded', 'padded'), [[5, 6, 90], [7, 8, 90], [90, 90, 90]]),
                ('G', ('lookup',), [.25, .5]), ('Gprime', ('padded',), [2, 3, 90]),
                ('rStar', ('padded',), [4, 5, 6]), ('zStar', ('padded',), [10, 11, 12])):
            variable = nc.createVariable(key, 'f8', dims)
            variable[:] = values
            variable.scale_factor = 2.
    native = reader.read(path, native_input=True)
    assert isinstance(native, reader)
    assert native.native_math
    assert native.radial.dtype == np.float32 and native.radial[0, 0] == np.float32(.1)
    np.testing.assert_array_equal(native.vertical, [[5, 6], [7, 8]])
    np.testing.assert_array_equal(native.g_prime, [2, 3])
    np.testing.assert_array_equal(native._sample(np.array([0.]), np.zeros(1), 1., 1.), [[0.], [0.], [7.]])
    with pytest.raises(ValueError, match='lookup dimensions'):
        reader.read(path)


def test_native_fire_kernels_and_trajectories():
    directory = ROOT/'results/python_port/fire'
    reference = directory/'reference.npz'
    if not reference.exists():
        pytest.skip('Run scripts/check_python_fire.py to compile the reference')
    pytest.importorskip('rasterio')
    comparison = runpy.run_path(str(ROOT/'scripts/check_python_fire.py'))['compare']
    with np.load(reference) as records:
        report = comparison(directory, records)
    assert report['fuel_codes'] == 58
    assert report['raster_cases'] == 4
    assert report['trajectory_max_absolute'][4] == report['trajectory_max_absolute'][6] == 0


def test_fire_xml_initializes_rectangle_and_timed_ignition(tmp_path):
    from netCDF4 import Dataset

    with Dataset(tmp_path/'times.nc', 'w') as nc:
        nc.createDimension('record', 1)
        for key, value in [('time', .2), ('y1', 4), ('x1', 746)]:
            nc.createVariable(key, 'f4', ('record',))[:] = value
    path = tmp_path/'fire.xml'
    path.write_text('''<QESWindsParameters><fires>
      <fireDur>5</fireDur><fuelType>1</fuelType><fmc>.06</fmc><cure>.68</cure><courant>.3</courant>
      <igTimes>times.nc</igTimes>
      <ignition><xStart>1</xStart><yStart>1</yStart><length>1</length><width>1</width>
      <height>2</height><baseHeight>0</baseHeight></ignition>
    </fires></QESWindsParameters>''')
    domain = Domain((8, 8, 10), (1., 1., 1.))
    ws = domain.workspace()
    ws.arrays['u'] = np.full(domain.face_shape, 1.2)
    ws.arrays['v'] = np.full(domain.face_shape, .2)
    fire = read_fire_config(path).initialize(ws, domain.spacing)
    assert fire.burn_flag[1, 1] == 1
    assert fire.burn_flag[4, 4] == 0
    fire.step(ws)
    assert fire.burn_flag[4, 4] == 1
    assert fire.smoke[4, 4] == 1


def test_timed_ignition_at_outer_face_uses_last_allocated_cell():
    terrain = np.zeros((4, 5))
    options = dict(moisture=.06, cure=.68, courant=.3)
    events = (TimedIgnition(1., 10., 7.), TimedIgnition(2., 6., 12.), TimedIgnition(3., 9.5, 11.5))
    fire = FireGrid(terrain, (2., 3., 1.), 1, timed_ignitions=events, **options)
    assert fire.timed_ignitions == [(1., 2, 4), (2., 3, 3), (3., 3, 4)]
    native = FireGrid(terrain, (2., 3., 1.), 1, timed_ignitions=events, native_ignition_input=True, **options)
    assert native.timed_ignitions == [(1., 3, 0), (2., 3, 3), (3., 3, 4)]
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (2., 3., 1.), 1, timed_ignitions=(TimedIgnition(1., 10.1, 7.),), **options)


def test_native_timed_ignition_clamps_beyond_east_face_before_flattening():
    terrain = np.zeros((4, 5))
    options = dict(moisture=.06, cure=.68, courant=.3, native_ignition_input=True)
    event = TimedIgnition(1., 99., 7.)
    fire = FireGrid(terrain, (2., 3., 1.), 1, timed_ignitions=(event,), **options)
    assert fire.timed_ignitions == [(1., 3, 0)]
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (2., 3., 1.), 1, timed_ignitions=(replace(event, y=10.),), **options)


def test_native_timed_ignition_uses_allocated_index_past_west_face():
    terrain = np.zeros((4, 5))
    event = TimedIgnition(1., -2., 6.)
    options = dict(moisture=.06, cure=.68, courant=.3)
    fire = FireGrid(terrain, (1., 3., 1.), 1, timed_ignitions=(event,), native_ignition_input=True, **options)
    assert fire.timed_ignitions == [(1., 1, 3)]
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 3., 1.), 1, timed_ignitions=(event,), **options)
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 3., 1.), 1, timed_ignitions=(replace(event, y=0.),),
                 native_ignition_input=True, **options)


def test_native_timed_ignition_checks_combined_corner_offsets():
    terrain = np.zeros((4, 5))
    options = dict(moisture=.06, cure=.68, courant=.3)
    events = (TimedIgnition(1., 6., -1.), TimedIgnition(2., -2., 5.))
    fire = FireGrid(terrain, (1., 1., 1.), 1, timed_ignitions=events, native_ignition_input=True, **options)
    assert fire.timed_ignitions == [(1., 0, 0), (2., 3, 3)]
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 1, timed_ignitions=events, **options)


def test_native_ignition_rectangle_uses_float32_corner_arithmetic():
    terrain = np.zeros((8, 8))
    ignition = Ignition(3.5999999, 1.4999999, .8, 1.)
    options = dict(moisture=.06, cure=.68, courant=.3)
    native = FireGrid(terrain, (.8, 1., 1.), 106, (ignition,), native_ignition_input=True, **options)
    modern = FireGrid(terrain, (.8, 1., 1.), 106, (ignition,), **options)
    np.testing.assert_array_equal(np.argwhere(native.burn), [[1, 5], [2, 5]])
    np.testing.assert_array_equal(np.argwhere(modern.burn), [[1, 4]])


def test_native_ignition_rectangle_crosses_east_face_within_storage():
    terrain = np.zeros((8, 8))
    options = dict(moisture=.06, cure=.68, courant=.3)
    rectangle = Ignition(7., 3., 3., 2.)
    native = FireGrid(terrain, (1., 1., 1.), 106, (rectangle,), native_ignition_input=True, **options)
    np.testing.assert_array_equal(np.argwhere(native.burn), [[3, 7], [4, 0], [4, 1], [4, 7], [5, 0], [5, 1]])
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 106, (rectangle,), **options)
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 106, (Ignition(7., 7., 3., 1.),), native_ignition_input=True, **options)


def test_native_ignition_rectangle_crosses_west_face_within_storage():
    terrain = np.zeros((8, 8))
    options = dict(moisture=.06, cure=.68, courant=.3)
    rectangle = Ignition(-2., 3., 3., 2.)
    native = FireGrid(terrain, (1., 1., 1.), 106, (rectangle,), native_ignition_input=True, **options)
    np.testing.assert_array_equal(np.argwhere(native.burn), [[2, 6], [2, 7], [3, 0], [3, 6], [3, 7], [4, 0]])
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 106, (rectangle,), **options)
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 106, (replace(rectangle, y=0.),), native_ignition_input=True, **options)


def test_native_ignition_rectangle_without_cells_does_not_address_storage():
    terrain = np.zeros((8, 8))
    options = dict(moisture=.06, cure=.68, courant=.3)
    actual_ignition = Ignition(2., 3., 1., 2.)
    empty_placeholder = Ignition(40., 8., .2, 2.)
    actual = FireGrid(terrain, (1., 1., 1.), 106, (actual_ignition, empty_placeholder),
                      native_ignition_input=True, **options)
    expected = FireGrid(terrain, (1., 1., 1.), 106, (actual_ignition,), native_ignition_input=True, **options)
    np.testing.assert_array_equal(actual.burn, expected.burn)
    np.testing.assert_array_equal(actual.front, expected.front)
    with pytest.raises(ValueError, match='outside the fire grid'):
        FireGrid(terrain, (1., 1., 1.), 106, (empty_placeholder,), **options)


def test_native_timed_ignition_divides_stored_coordinates_in_float32():
    terrain = np.zeros((8, 8))
    events = (TimedIgnition(.25, float(np.float32(3.6)), 1.5),)
    options = dict(moisture=.06, cure=.68, courant=.3, timed_ignitions=events)
    native = FireGrid(terrain, (.8, 1., 1.), 106, native_ignition_input=True, **options)
    modern = FireGrid(terrain, (.8, 1., 1.), 106, **options)
    assert native.timed_ignitions == [(.25, 2, 5)]
    assert modern.timed_ignitions == [(.25, 2, 4)]


def test_native_fire_burns_out_on_the_float32_residence_boundary():
    domain = Domain((8, 8, 10), (1., 1., 1.))
    workspace = domain.workspace()
    for field in ('u', 'v'):
        workspace.arrays[field] = np.zeros(domain.face_shape)
    fire = FireGrid(workspace.arrays['terrain'], domain.spacing, 99, (Ignition(3., 3., 1., 1.),),
                    moisture=.06, cure=.68, courant=35.714287, native_fire_clock=True)
    for _ in range(6):
        fire.step(workspace)
        assert fire.burn_out[3, 3] == 1
    fire.step(workspace)
    assert fire.time < 5000
    assert fire.burn_time[3, 3] == fire.properties.residence_time[3, 3] == 5000
    assert fire.burn_out[3, 3] == 2
    assert workspace.arrays['icellflag'][0, 3, 3] == 1


@pytest.mark.parametrize('extra_coordinates', [False, True])
def test_native_timed_ignition_reads_stored_values(tmp_path, extra_coordinates):
    from netCDF4 import Dataset

    path = tmp_path/'times.nc'
    with Dataset(path, 'w') as nc:
        nc.createDimension('record', 1)
        nc.createDimension('locations', 2 if extra_coordinates else 1)
        for key, dim, values, scale in (
                ('time', 'record', [.25], 4.),
                ('y1', 'locations', [6., 8.], 2.),
                ('x1', 'locations', [745., 740.], .5)):
            var = nc.createVariable(key, 'f4', (dim,))
            var[:] = values[:len(var)]
            var.scale_factor = scale
    assert read_timed_ignitions(path, native_input=True) == (TimedIgnition(.25, 6., 5.),)
    if extra_coordinates:
        with pytest.raises(ValueError, match='matching finite one-dimensional'):
            read_timed_ignitions(path)
    else:
        assert read_timed_ignitions(path) == (TimedIgnition(1., 12., 377.5),)


def test_bundled_ffii_ignition_track_initializes_on_north_edge():
    source = ROOT/'external/QES-Public/data/InputFiles/FFII.xml'
    if not source.is_file():
        pytest.skip('Bundled FFII case requires the original QES checkout')
    case = read_coupled_fire(source)
    workspace, _ = case.wind.prepare()
    fire = case.fire.initialize(workspace, case.wind.domain.spacing)
    assert len(fire.timed_ignitions) == 830
    assert fire.timed_ignitions[-1] == (215., 149, 26)


@pytest.mark.parametrize('native_sampling, initial_level', [(False, 2), (True, 0)])
def test_initial_flame_wind_sampling_on_elevated_terrain(native_sampling, initial_level):
    domain = Domain((8, 8, 10), (1., 1., 1.))
    ws = domain.workspace()
    ws.arrays['terrain'].fill(2.)
    ws.arrays['u'] = np.broadcast_to(np.arange(domain.face_shape[0])[:, None, None], domain.face_shape)
    ws.arrays['v'] = np.zeros(domain.face_shape)
    levels = []

    def sample(fuel, u, *args, **kwargs):
        levels.append(np.unique(u).tolist())
        return balbi(fuel, u, *args, **kwargs)

    fire = FireGrid(ws.arrays['terrain'], domain.spacing, 106, (Ignition(3, 3, 1, 1),),
                    moisture=.06, cure=.68, courant=.3, spread=sample, native_flame_sampling=native_sampling)
    fire.forcing(ws)
    assert levels == [[initial_level]]
    assert fire.properties.residence_time[3, 3] > 0
    _, sample_level, _ = fire._flame_levels(3, 3)
    assert sample_level >= 2
    fire.forcing(ws)
    assert levels[-1] == [sample_level]


def test_native_flame_layer_division_preserves_float32_half_cell():
    terrain = np.full((8, 8), np.float32(.45))
    options = dict(moisture=.06, cure=.68, courant=.3)
    native = FireGrid(terrain, (1., 1., .9), 99, native_flame_sampling=True, **options)
    modern = FireGrid(terrain, (1., 1., .9), 99, **options)
    assert native._flame_levels(3, 3) == (1, 1, 1)
    assert modern._flame_levels(3, 3) == (0, 0, 0)


@pytest.mark.parametrize('name', ['prescribed', 'coupled', 'coupled_series', 'coupled_building'])
def test_coupled_fire_netcdf(name, tmp_path):
    from netCDF4 import Dataset

    folder = ROOT/'results/python_port/fire_coupled'/name
    native = folder/'bounded_fireOutput.nc'
    if not native.exists():
        pytest.skip('Run scripts/check_python_coupled_fire.py --native for native fixture')
    heat = None if name == 'prescribed' else ROOT/'external/QES-Public/data/FireFiles/HeatPot.nc'
    case = read_coupled_fire(folder/'winds.xml', heat_potential=heat, compute_turbulence=name == 'coupled_series')
    native_format = name == 'coupled_building'
    summary = write_fire_run(case, tmp_path, native_fire_format=native_format)
    with Dataset('reference', memory=native.read_bytes()) as expected, Dataset(tmp_path/'fireOutput.nc') as actual:
        assert actual['u'].dimensions == ('t', 'zdim', 'ydim', 'xdim')
        assert actual['LS'].dimensions == ('t', 'ydim', 'xdim')
        assert actual['t'].simulation_start == expected['t'].simulation_start
        assert summary['records'] == len(expected['t'])
        if native_format:
            assert actual.__dict__ == expected.__dict__
            assert tuple(actual.variables) == tuple(expected.variables)
            for key, variable in expected.variables.items():
                field = actual[key]
                assert (field.dtype, field.dimensions, field.__dict__) == (
                    variable.dtype, variable.dimensions, variable.__dict__)
        for key in ('timestamp', 'fuel', 'burn', 'icell'):
            np.testing.assert_array_equal(actual[key][:], expected[key][:])
        for key in ('t', 'u', 'v', 'w', 'LS', 'Force'):
            np.testing.assert_allclose(actual[key][:], expected[key][:], atol=3e-6, rtol=0)
    if name == 'coupled_series':
        with Dataset('reference', memory=(folder/'bounded_turbOut.nc').read_bytes()) as expected, Dataset(tmp_path/'turbulence.nc') as actual:
            for key in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z'):
                np.testing.assert_allclose(actual[key][:], expected[key][:], atol=1e-4, rtol=0)


@pytest.mark.parametrize('name, emitted', [('nearest', 720), ('series', 1540), ('mixed_models', 720)])
def test_coupled_fire_smoke_collection(name, emitted, tmp_path):
    from netCDF4 import Dataset

    folder = ROOT/'results/python_port/fire_smoke'/name
    native = folder/'native_smoke_plumeOut.nc'
    if not native.exists():
        pytest.skip('Run scripts/check_python_fire_smoke.py --native for the CPU fixture')
    heat = ROOT/'external/QES-Public/data/FireFiles/HeatPot.nc'
    case = read_coupled_fire(folder/'winds.xml', heat_potential=heat, plume=folder/'plume.xml',
                             particle_csv=name != 'series')
    native_collection = name != 'series'
    result = write_fire_run(case, tmp_path, native_collection_format=native_collection,
                            native_collection_math=native_collection)
    smoke = result['smoke']
    assert smoke['released_particles'] == smoke['active_particles'] == emitted
    with Dataset('reference', memory=native.read_bytes()) as expected, Dataset(tmp_path/'smoke_plumeOut.nc') as actual:
        assert actual['p_count'].dimensions == ('t', 'z_c', 'y_c', 'x_c')
        assert actual['t'].simulation_start == expected['t'].simulation_start
        for key in ('x_c', 'y_c', 'z_c', 'p_count', 'c', 'timestamp'):
            np.testing.assert_array_equal(actual[key][:], expected[key][:])
        for key in ('t', 't_avg'):
            np.testing.assert_allclose(actual[key][:], expected[key][:], atol=.001001, rtol=0)
        if native_collection:
            assert actual['t_avg'].dtype == expected['t_avg'].dtype
    if name != 'series':
        snapshots = sorted(tmp_path.glob('smoke_particleData_*.csv'))
        assert smoke['particle_output_records'] == len(snapshots) == 3
        for snapshot, count in zip(snapshots, [160, 400, 640]):
            assert len(snapshot.read_text().splitlines()) == 5+count
    if name == 'mixed_models':
        for tag, count in (('beacon', 18), ('tracer', 27)):
            assert result['plume_models'][tag]['released_particles'] == count
            assert result['plume_models'][tag]['records'] == 3
            with (Dataset('reference', memory=(folder/f'native_{tag}_plumeOut.nc').read_bytes()) as expected,
                  Dataset(tmp_path/f'{tag}_plumeOut.nc') as actual):
                for key in ('t', 'p_count', 'timestamp'):
                    np.testing.assert_array_equal(actual[key][:], expected[key][:])
                np.testing.assert_allclose(actual['c'][:], expected['c'][:], atol=1e-9, rtol=0)
