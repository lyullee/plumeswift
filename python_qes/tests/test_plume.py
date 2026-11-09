from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv
import json
from shutil import copyfile
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from netCDF4 import Dataset
import numpy as np
import pytest

from plumeswift.boundaries import AxisBoundary, DomainBoundaries, NativeAxisBoundary
from plumeswift.__main__ import main
from plumeswift.contracts import UnportedFeature
from plumeswift.cpu_plume_random import LinuxQESRandom, NativeCPUStreams
from plumeswift.native_staircase import NativeStaircase
from plumeswift.particle_output import ParticleOutput, read_particle_output
from plumeswift.plume_config import PlumeConfig, parse_plume_config
from plumeswift.plume_fields import open_plume_fields
from plumeswift.plume_run import run_legacy_qes, run_plume
from plumeswift.plume_sources import read_models
from plumeswift.plume_transport import PlumeModel
from plumeswift.powerlaw import NativePowerLawSampler, PowerLawSampler
from plumeswift.qes_xml import read_qes_xml


FIXTURE = Path(__file__).resolve().parents[2]/'results/python_port/plume'


@pytest.mark.parametrize('include_plume', [False, True])
def test_coupled_qes_zero_wrf_steps_skips_solver_and_writes_empty_outputs(tmp_path, monkeypatch, include_plume):
    from plumeswift.wind_case import WindCase

    root = Path(__file__).resolve().parents[2]
    fixture = root/'results/python_port/wrf_startup'
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
    plume = root/'data/cases/empty_west/plume.xml' if include_plume else None
    result = run_legacy_qes(path, plume, basename=tmp_path/'case', workspace=True, turbulence=True, wrf_timeout=3)
    assert result['wind_records'] == 0
    outputs = ['windsOut', 'windsWk', 'turbOut']
    if include_plume:
        assert result['models']['passive_tracer']['released_particles'] == 0
        outputs.append('passive_tracer_plumeOut')
    for name in outputs:
        with Dataset(tmp_path/f'case_{name}.nc') as output:
            assert len(output.dimensions['t']) == 0


def test_upstream_plume_input_model_definitions():
    path = FIXTURE.parents[2]/'external/QES-Public/tests/unitTests/plume_input_parameters.xml'
    root = read_qes_xml(path)
    config = parse_plume_config(root, legacy_plume_xml=True)
    models = read_models(root, config.step_s, legacy_plume_xml=True)
    assert list(models) == ['tracer', 'pollen']
    assert [len(model.sources) for model in models.values()] == [2, 1]
    assert [source.release.particles for model in models.values() for source in model.sources] == [10, 20, 400]
    assert models['pollen'].deposition


def test_upstream_yakima_legacy_metadata_does_not_create_a_source():
    path = FIXTURE.parents[2]/'external/QES-Public/data/InputFiles/Yakima_Plume.xml'
    root = read_qes_xml(path)
    config = parse_plume_config(root, legacy_plume_xml=True)
    assert read_models(root, config.step_s, legacy_plume_xml=True) == {}
    with pytest.raises(UnportedFeature, match='Unsupported sourceParameters fields'):
        read_models(root, config.step_s)


def test_native_duplicate_model_uses_later_sources_with_original_source_ids():
    root = ET.fromstring('''<QESPlumeParameters><particleParameters><baseParticle>
        <tag>cloud</tag><source><releaseType_instantaneous><numPar>30</numPar>
        </releaseType_instantaneous><sourceGeometry_FullDomain/></source>
        </baseParticle></particleParameters></QESPlumeParameters>''')
    section = root.find('particleParameters')
    replacement = deepcopy(section[0])
    replacement.find('.//numPar').text = '2'
    section.append(replacement)
    with pytest.raises(ValueError, match='duplicate'):
        read_models(root, 1., legacy_plume_xml=True)
    with pytest.raises(ValueError, match='duplicate'):
        read_models(root, 1., allow_repeated_tags=True)
    models = read_models(root, 1., legacy_plume_xml=True, allow_repeated_tags=True)
    source, = models['cloud'].sources
    assert source.source_id == 1
    assert source.release.particles == 2


def test_native_model_tags_keep_printable_spaces():
    root = ET.fromstring('''<QESPlumeParameters><particleParameters>
        <baseParticle><tag>point A</tag></baseParticle>
        <baseParticle><tag>  point A  </tag></baseParticle>
        </particleParameters></QESPlumeParameters>''')
    with pytest.raises(ValueError, match='duplicate'):
        read_models(root, 1.)
    assert list(read_models(root, 1., legacy_plume_xml=True)) == ['point A', '  point A  ']


def test_legacy_plume_validates_xml_after_float32_conversion():
    root = ET.fromstring('''<QESPlumeParameters>
      <plumeParameters><timeStep>.250000001</timeStep><simDur>.25</simDur><CourantNumber>1</CourantNumber>
        <invarianceTol>1e-5</invarianceTol></plumeParameters>
      <boundaryConditions><xBCtype>exiting</xBCtype><yBCtype>exiting</yBCtype><zBCtype>exiting</zBCtype></boundaryConditions>
      <collectionParameters><timeAvgStart>0</timeAvgStart><timeAvgFreq>.25</timeAvgFreq>
        <boxBoundsX1>1.00000001</boxBoundsX1><boxBoundsX2>1</boxBoundsX2><nBoxesX>1</nBoxesX>
        <boxBoundsY1>0</boxBoundsY1><boxBoundsY2>1</boxBoundsY2><nBoxesY>1</nBoxesY>
        <boxBoundsZ1>0</boxBoundsZ1><boxBoundsZ2>1</boxBoundsZ2><nBoxesZ>1</nBoxesZ>
      </collectionParameters></QESPlumeParameters>''')
    legacy = parse_plume_config(root, legacy_plume_xml=True)
    assert legacy.step_s == legacy.duration_s == .25
    assert legacy.lower[0] == legacy.upper[0] == 1.
    with pytest.raises(ValueError, match='Invalid plume'):
        parse_plume_config(root)


def test_plume_statistics_share_native_window_and_finalize_in_key_order():
    config = PlumeConfig(2., .25, 1., 1e-6, 'triLinear', ('exiting',)*3, 'doNothing',
                         250, 100, (0.,)*3, (1.,)*3, (1,)*3, 0)
    model = PlumeModel(config, native_collection_math=True)
    model.simulation = SimpleNamespace(step=lambda _dt: None, batch=SimpleNamespace(
        positions=np.array([[.5, .5, .5]]), mass=np.array([.001])))
    events = []
    for key in ('z_extra', 'a_extra'):
        model.statistics.attach(key,
                                lambda time, dt, key=key: events.append((key, 'collect', time, dt)),
                                lambda time, key=key: events.append((key, 'finalize', time)) or time)
    with pytest.raises(ValueError, match='Duplicate plume statistics source'):
        model.statistics.attach('concentration', None, None)
    assert model.advance_step(250) is None
    assert events == []
    sample = model.advance_step(500)
    assert sample.time_ms == 500
    assert sample.fields['c'].item() == 1.
    assert sample.fields['p_count'].item() == 1
    assert sample.fields['t_avg'] == .25
    assert list(sample.extra_statistics.items()) == [('a_extra', 500), ('z_extra', 500)]
    assert events == [('a_extra', 'collect', 500, .25), ('z_extra', 'collect', 500, .25),
                      ('a_extra', 'finalize', 500), ('z_extra', 'finalize', 500)]
    assert model.next_sample == 450
    assert model.advance_step(750).fields['p_count'].item() == 1
    assert model.next_sample == 550


@pytest.mark.parametrize('project', [True, False])
@pytest.mark.parametrize('trailing_separator', [False, True])
def test_native_standalone_cli_basename(project, trailing_separator, tmp_path, monkeypatch, capsys):
    if not (FIXTURE/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_plume.py --native for the field fixture')
    base = tmp_path/'test.case'
    if trailing_separator:
        base.mkdir()
    raw_basename = f'{base}/' if trailing_separator else str(base)
    prefix = Path(raw_basename+'_')
    if project:
        for suffix in ('windsWk.nc', 'turbOut.nc'):
            copyfile(FIXTURE/f'native_{suffix}', Path(str(prefix)+suffix))
        flags = ('-f', raw_basename, '-w', 'missing.nc', '-t', 'missing.nc', '-o', tmp_path/'ignored')
    else:
        flags = ('-w', FIXTURE/'native_windsWk.nc', '-t', FIXTURE/'native_turbOut.nc', '-o', raw_basename)
    tree = ET.parse(FIXTURE/'plume.xml')
    root = tree.getroot()
    if project:
        root.find('boundaryConditions/wallReflection').text = 'meshReflection'
    for node in (root, root.find('plumeParameters'), *root.findall('.//baseParticle'),
                 *root.findall('.//source'), *root.findall('.//releaseType_continuous')):
        ET.SubElement(node, 'annotation').text = 'ignored by native QES'
    config = tmp_path/'plume.xml'
    tree.write(config, encoding='utf-8')
    with pytest.raises(UnportedFeature, match='Unsupported plumeParameters fields'):
        parse_plume_config(root)
    with pytest.raises(UnportedFeature, match='Unsupported baseParticle fields'):
        read_models(root, .25)
    monkeypatch.setattr(sys, 'argv', ['plumeswift', 'qesPlume', '-p', str(config), '-l', *map(str, flags)])
    main()
    captured = capsys.readouterr()
    assert captured.err.count('[WARRNING] mesh wall reflection method not available') == int(project)
    result = json.loads(captured.out)
    assert result['wind_records'] == 2
    assert set(result['models']) == {'continuous', 'pulse'}
    for tag in result['models']:
        with Dataset(Path(f'{prefix}{tag}_plumeOut.nc')) as output:
            assert output['c'].dtype == np.float32
        assert len(list(prefix.parent.glob(f'{prefix.name}{tag}_particleData_*.csv'))) == result['particle_output_records']
    assert not list(tmp_path.glob('ignored_*'))


def test_native_offline_preserves_nonfinite_wind_faces_until_sampled(tmp_path):
    fixture_wind = FIXTURE/'native_windsWk.nc'
    if not fixture_wind.exists():
        pytest.skip('Run scripts/check_python_plume.py --native for the field fixture')
    wind = tmp_path/'wind.nc'
    turbulence = FIXTURE/'native_turbOut.nc'
    copyfile(fixture_wind, wind)
    with Dataset(wind, 'r+') as data:
        data['u'][:, 0, 0, 0] = np.nan
    with open_plume_fields(wind, turbulence, native_layout=True) as (_, fields):
        workspace, _, _ = next(fields)
        assert np.isnan(workspace.arrays['u'][0, 0, 0])
    with open_plume_fields(wind, turbulence) as (_, fields):
        with pytest.raises(ValueError, match='non-finite field u'):
            next(fields)
    result = run_plume(wind, FIXTURE/'plume.xml', tmp_path/'unused', turbulence_path=turbulence,
                       native_layout=True, native_threads=1)
    assert result['models']['continuous']['active_particles'] > 0
    with Dataset(wind, 'r+') as data:
        data['u'][:] = np.nan
    with pytest.raises(FloatingPointError, match='Non-finite wind sampled'):
        run_plume(wind, FIXTURE/'plume.xml', tmp_path/'sampled', turbulence_path=turbulence,
                  native_layout=True, native_threads=1)


def test_native_particle_output_negative_period_remains_due_after_start(tmp_path):
    root = ET.fromstring('''<QESPlumeParameters><particleOutputParameters>
        <outputStartTime>.75</outputStartTime><outputFrequency>-.25</outputFrequency>
        </particleOutputParameters></QESPlumeParameters>''')
    with pytest.raises(ValueError, match='positive millisecond output frequency'):
        read_particle_output(root)
    config = read_particle_output(root, legacy_plume_xml=True)
    output = ParticleOutput(config, tmp_path, datetime(2020, 1, 1))
    for time_ms in range(250, 2001, 250):
        output.save(time_ms, {})
        assert output.records == max(0, time_ms//250-2)
    assert config.period_ms == -250


def test_legacy_particle_output_ignores_extra_xml_metadata():
    root = ET.fromstring('''<QESPlumeParameters><particleOutputParameters>
        <outputStartTime>.5</outputStartTime><outputFrequency>.25</outputFrequency>
        <diagnosticNotes>metadata</diagnosticNotes>
        </particleOutputParameters></QESPlumeParameters>''')
    with pytest.raises(NotImplementedError, match='diagnosticNotes'):
        read_particle_output(root)
    config = read_particle_output(root, legacy_plume_xml=True)
    assert (config.first_ms, config.period_ms) == (500, 250)


@pytest.mark.parametrize('count', [1, 32])
def test_native_staircase_reflects_before_out_of_grid_endpoint(count):
    centers = np.arange(4, dtype=np.float32)+.5
    sampler = SimpleNamespace(x=centers, y=centers, zf=np.arange(-1, 7, dtype=np.float32),
                              native_spacing=(1., 1., 1.))
    solid = np.zeros((7, 4, 4), dtype=bool)
    positions = np.tile([1.5, 1.5, 4.4], (count, 1))
    displacement = np.tile([0, 0, 2.5], (count, 1))
    fluctuations = np.tile([0, 0, .2], (count, 1))
    assert not NativeStaircase(sampler, solid).reflect_native(positions, displacement, fluctuations)[2].any()
    solid[-1] = True
    endpoints, velocity, active = NativeStaircase(sampler, solid).reflect_native(
        positions, displacement, fluctuations)
    assert active.all()
    np.testing.assert_allclose(endpoints[:, 2], 3.1)
    np.testing.assert_allclose(velocity[:, 2], -.2)


@pytest.mark.parametrize('mode', ['exiting', 'periodic', 'reflection'])
def test_native_boundary_accepts_coincident_faces(mode):
    with pytest.raises(ValueError, match='Invalid boundary'):
        AxisBoundary(1., 1., mode)
    position = np.array([.5, 1., 1.5])
    fluctuation = np.ones(3)
    end, velocity, active = NativeAxisBoundary(1., 1., mode).apply(position, fluctuation)
    np.testing.assert_array_equal(end, position)
    np.testing.assert_array_equal(velocity, fluctuation)
    np.testing.assert_array_equal(active, np.full(3, mode != 'exiting'))


def test_native_powerlaw_interpolation_preserves_nonpositive_height_singularities():
    position = np.array([[2., 3., 0.], [2., 3., -.3], [2., 3., .5]])
    modern = PowerLawSampler.__new__(PowerLawSampler)
    with pytest.raises(ValueError, match='singular at nonpositive height'):
        modern.sample_wind(position)
    native = NativePowerLawSampler.__new__(NativePowerLawSampler)
    native.mixlength = None
    wind = native.sample_wind(position)
    np.testing.assert_array_equal(wind[0], np.zeros(3))
    assert np.isnan(wind[1, 0]) and np.isfinite(wind[2]).all()
    stress, divergence, dissipation, _, _ = native.sample_turbulence_and_distance(position)
    np.testing.assert_array_equal(stress[0], np.zeros((3, 3)))
    assert np.isneginf(divergence[0, 0]) and np.isposinf(divergence[0, 2])
    assert np.isnan(stress[1, 0, 0]) and np.isnan(divergence[1, 0])
    assert np.isnan(dissipation[:2]).all() and np.isfinite(dissipation[2])


def test_native_cpu_workers_keep_holes_and_skip_inactive_release_draws():
    streams = NativeCPUStreams(17, 3)
    slots = np.array([0, 4, 7, 10])
    np.testing.assert_array_equal(streams.owners(slots, 11), [0, 1, 1, 2])
    np.testing.assert_array_equal(streams.owners(np.arange(2), 2), [0, 1])
    expected = [LinuxQESRandom(17 ^ worker) for worker in range(3)]
    initial = streams.initial_draws(slots, 11)
    reference = np.concatenate([rng.normal(size=(count, 3)) for rng, count in zip(expected, (1, 2, 1))])
    np.testing.assert_array_equal(initial, reference)
    for actual, rng in zip(streams.streams, expected):
        assert actual.normal() == rng.normal()


def test_stretched_plume_uses_physical_cells_and_derives_vertical_stress_gradient(tmp_path):
    from plumeswift.domain import Domain
    from plumeswift.interpolation import FieldSampler
    from plumeswift.io import write_netcdf
    from plumeswift.plume_fields import open_plume_fields

    domain = Domain((12, 12, 8), (1., 1.5, 1.), (.5,)*3+(2.,)*5)
    workspace = domain.workspace()
    fields = {name: np.zeros(domain.face_shape) for name in 'uvw'}
    fields['icellflag'] = workspace.arrays['icellflag']
    coords = workspace.coordinates
    wind, turbulence = tmp_path/'wind.nc', tmp_path/'turb.nc'
    # z_face is sufficient to recover the layer widths in the physical path.
    write_netcdf(wind, fields, coords)
    stresses = {name: np.zeros(domain.cell_shape) for name in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'CoEps')}
    stresses['tzz'] = np.broadcast_to(coords['z'][:, None, None]**2, domain.cell_shape)
    write_netcdf(turbulence, stresses, coords)
    with open_plume_fields(wind, turbulence) as (_, records):
        actual, turb, spacing = next(records)
    np.testing.assert_array_equal(turb['div_tau_z'][1:-1, 3, 4], 2*coords['z'][1:-1])
    distance = np.broadcast_to(np.arange(domain.cell_shape[0])[:, None, None], domain.cell_shape)
    sampler = FieldSampler(actual.coordinates, actual.arrays, turb, mode='nearest',
                           mixlength=distance, native_spacing=spacing)
    points = [[4.5, 5.25, 3.25], [7.5, 16.1, 8.25]]
    np.testing.assert_array_equal(sampler.cell_indices(points), [[4, 7], [3, 10], [4, 7]])
    np.testing.assert_array_equal(sampler.sample_distance(points), [4, 7])
    with open_plume_fields(wind, turbulence, native_arithmetic=True) as (_, records):
        _, native_turb, _ = next(records)
    assert native_turb['div_tau_z'][5, 3, 4] != turb['div_tau_z'][5, 3, 4]


def test_plume_reloads_saved_wall_distance_by_record_and_keeps_native_fallback(tmp_path):
    from plumeswift.domain import Domain
    from plumeswift.io import netcdf_series
    from plumeswift.plume_fields import open_plume_fields

    domain = Domain((5, 5, 5), (1., 1., 1.))
    workspace = domain.workspace()
    coords = workspace.coordinates
    wind, turbulence, without_distance = (tmp_path/f'{name}.nc' for name in ('wind', 'turb', 'legacy_turb'))
    wind_fields = {name: np.zeros(domain.face_shape) for name in 'uvw'}
    wind_fields.update(icellflag=workspace.arrays['icellflag'], mixlength=np.full(domain.cell_shape, 2.))
    stresses = {name: np.zeros(domain.cell_shape) for name in
                ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z')}
    stresses['L'] = np.full(domain.cell_shape, 99.)
    with (netcdf_series(wind, coords) as write_wind,
          netcdf_series(turbulence, coords) as write_turbulence,
          netcdf_series(without_distance, coords) as write_legacy):
        for second, distance in enumerate((3., 7.)):
            time = f'2020-01-01T00:00:0{second}'
            write_wind(wind_fields, time, second)
            write_turbulence(stresses | {'mixingLengths': np.full(domain.cell_shape, distance)}, time, second)
            write_legacy(stresses, time, second)
    for path, native, distances in ((turbulence, False, (3., 7.)), (without_distance, False, (2., 2.)),
                                    (turbulence, True, (0., 0.))):
        with open_plume_fields(wind, path, native_arithmetic=native) as (_, records):
            for (_, fields, _), distance in zip(records, distances, strict=True):
                np.testing.assert_array_equal(fields['mixingLengths'], distance)


@pytest.mark.parametrize('source_seed', [None, 17])
def test_full_domain_source_uses_each_axis_bounds(source_seed):
    root = ET.fromstring("""<QESPlumeParameters><particleParameters><baseParticle>
      <tag>cloud</tag><source><releaseType_instantaneous><numPar>30</numPar>
      </releaseType_instantaneous><sourceGeometry_FullDomain/></source>
      </baseParticle></particleParameters></QESPlumeParameters>""")
    spec, = read_models(root, .1)['cloud'].sources
    boundaries = DomainBoundaries(tuple(AxisBoundary(lo, hi) for lo, hi in [(1, 3), (2, 7), (0, 5)]))
    source = spec.create(boundaries, source_seed=source_seed)
    points, mass = source.emit(0, .1, np.random.default_rng(0))
    assert len(points) == 30
    assert np.all((points > (1, 2, 0)) & (points < (3, 7, 5)))
    assert mass.sum() == 0
    assert len(source.emit(.1, .1, np.random.default_rng(0))[0]) == 0
    native = spec.create(boundaries, source_seed=source_seed, native_full_domain_bounds=True)
    native_points, _ = native.emit(0, .1, np.random.default_rng(0))
    assert np.all((native_points > (1, 0, 2)) & (native_points < (7, 3, 5)))
    assert np.any(native_points[:, 0] > 3)
    thin = DomainBoundaries((boundaries.axes[0], NativeAxisBoundary(2., 2., 'periodic'), boundaries.axes[2]))
    planar = spec.create(thin, source_seed=source_seed)
    assert np.all(planar.emit(0, .1, np.random.default_rng(0))[0][:, 1] == 2.)
    if source_seed is not None:
        repeat = spec.create(boundaries, source_seed=source_seed)
        other_rng = np.random.default_rng(123)
        np.testing.assert_array_equal(repeat.emit(0, .1, other_rng)[0], points)


def test_point_does_not_advance_native_geometry_seed():
    root = ET.fromstring("""<QESPlumeParameters><particleParameters><baseParticle>
      <tag>cloud</tag><source><releaseType_instantaneous><numPar>1</numPar></releaseType_instantaneous>
      <sourceGeometry_Point><posX>1</posX><posY>1</posY><posZ>1</posZ></sourceGeometry_Point></source>
      <source><releaseType_instantaneous><numPar>8</numPar></releaseType_instantaneous>
      <sourceGeometry_FullDomain/></source></baseParticle></particleParameters></QESPlumeParameters>""")
    bounds = DomainBoundaries(tuple(AxisBoundary(0, 3) for _ in range(3)))
    _, with_point = read_models(root, .1)['cloud'].sources
    model = root.find('particleParameters/baseParticle')
    model.remove(model.find('source'))
    without_point, = read_models(root, .1)['cloud'].sources
    assert (with_point.source_id, without_point.source_id) == (1, 0)
    rng = np.random.default_rng(0)
    actual = with_point.create(bounds, source_seed=233).emit(0, .1, rng)[0]
    expected = without_point.create(bounds, source_seed=233).emit(0, .1, rng)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('layout', ['sourceParameters', 'sources'])
def test_legacy_plume_sources_run_through_normal_entry(layout, tmp_path):
    if not (FIXTURE/'native_windsWk.nc').exists():
        pytest.skip('Run scripts/check_python_plume.py --native for the wind fixture')
    tree = ET.parse(FIXTURE/'plume.xml')
    root = tree.getroot()
    root.remove(root.find('particleParameters'))
    section = ET.SubElement(root, layout)
    if layout == 'sources':
        root.find('plumeParameters').tag = 'simulationParameters'
        geometry = source = ET.SubElement(section, 'SourcePoint')
        release = ET.SubElement(source, 'ReleaseType_continuous')
    else:
        source = ET.SubElement(section, 'source')
        geometry = ET.SubElement(source, 'sourceGeometry_Point')
        release = ET.SubElement(source, 'releaseType_continuous')
    ET.SubElement(release, 'parPerTimestep').text = '3'
    for axis, value in zip('XYZ', (6, 11, 8)):
        ET.SubElement(geometry, 'pos'+axis).text = str(value)
    assert read_models(root, .25, legacy_plume_xml=True) == {}
    path = tmp_path/'legacy.xml'
    tree.write(path)
    result = run_plume(FIXTURE/'native_windsWk.nc', path, tmp_path, turbulence_path=FIXTURE/'native_turbOut.nc')
    assert result['models']['base']['released_particles'] == 48
    assert (tmp_path/'base_plumeOut.nc').exists()


@pytest.mark.parametrize('offline', [False, True])
def test_native_plume_model_collection(offline, tmp_path):
    reference = 'offline' if offline else 'native'
    if not (FIXTURE/f'{reference}_continuous_plumeOut.nc').exists():
        pytest.skip('Run scripts/check_python_plume.py --native for the CPU fixture')
    wind = FIXTURE/('native_windsWk.nc' if offline else 'winds.xml')
    turb = FIXTURE/'native_turbOut.nc' if offline else None
    result = run_plume(wind, FIXTURE/'plume.xml', tmp_path, turbulence_path=turb, particle_csv=True)
    assert result['time_s'] == 4 and result['wind_records'] == 2
    assert result['particle_output_records'] == 5
    for tag, released in [('continuous', 128), ('pulse', 59)]:
        assert result['models'][tag]['released_particles'] == released
        assert result['models'][tag]['active_particles'] == released
        with Dataset(tmp_path/f'{tag}_plumeOut.nc') as actual, Dataset(
                'reference', memory=(FIXTURE/f'{reference}_{tag}_plumeOut.nc').read_bytes()) as expected:
            assert actual['c'].dimensions == ('t', 'z_c', 'y_c', 'x_c')
            for key in ('timestamp', 't', 'p_count'):
                np.testing.assert_array_equal(actual[key][:], expected[key][:])
            np.testing.assert_allclose(actual['c'][:], expected['c'][:], atol=1e-9, rtol=0)
        snapshots = sorted(tmp_path.glob(f'{tag}_particleData_*.csv'))
        assert len(snapshots) == 5
        for snapshot in snapshots:
            native = FIXTURE/f'{reference}_{snapshot.name}'
            with snapshot.open(newline='') as output, native.open(newline='') as raw:
                for _ in range(5):
                    assert next(output).strip() == next(raw).strip()
                generated = {int(row[0]): row[1:6] for row in csv.reader(output)}
                baseline = {int(row[0]): row[1:6] for row in csv.reader(raw, skipinitialspace=True)}
                assert generated.keys() == baseline.keys()
                for particle_id, row in generated.items():
                    assert row[:2] == baseline[particle_id][:2]
                    np.testing.assert_allclose(np.array(row[2:], dtype=float),
                                               np.array(baseline[particle_id][2:], dtype=float), atol=1e-7)


def test_coupled_plume_exports_duplicate_wind_records_without_advancing(tmp_path):
    if not (FIXTURE/'winds.xml').exists():
        pytest.skip('Run scripts/check_python_plume.py for the coupled fixture')
    tree = ET.parse(FIXTURE/'winds.xml')
    tree.find('simulationParameters/totalTimeIncrements').text = '3'
    sensor = tree.find('metParams/sensor')
    sensor.append(deepcopy(sensor.find('timeSeries')))
    wind = tmp_path/'duplicate.xml'
    tree.write(wind)

    plume = FIXTURE/'plume.xml'
    baseline = tmp_path/'baseline'
    output = tmp_path/'coupled'
    expected = run_plume(FIXTURE/'winds.xml', plume, baseline)
    actual = run_plume(wind, plume, output, native_workspace_format=True,
                       native_turbulence_format=True, native_visualization_format=True)
    assert actual['wind_records'] == 3 and actual['time_s'] == expected['time_s'] == 4
    assert actual['models'] == expected['models']
    assert set(actual['field_outputs'].values()) == {'windsWk.nc', 'turbOut.nc', 'windsOut.nc'}
    for filename in actual['field_outputs'].values():
        with Dataset(output/filename) as series:
            np.testing.assert_array_equal(series['t'][:], [0, 0, 2])
            assert np.array_equal(series['timestamp'][0], series['timestamp'][1])
    for tag in actual['models']:
        filename = f'{tag}_plumeOut.nc'
        with Dataset(output/filename) as series, Dataset(baseline/filename) as reference:
            for name in reference.variables:
                np.testing.assert_array_equal(series[name][:], reference[name][:])


def test_rogue_particles_are_accounted_separately(tmp_path):
    fixture = FIXTURE/'rogue'
    if not (fixture/'stress.nc').exists():
        pytest.skip('Run scripts/check_python_plume_rogue.py for the CPU fixture')
    result = run_plume(FIXTURE/'native_windsWk.nc', fixture/'plume.xml', tmp_path,
                       turbulence_path=fixture/'stress.nc', particle_csv=True)
    for tag, count in [('determinant', 4), ('velocity', 4), ('regular', 0)]:
        model = result['models'][tag]
        assert model['released_particles'] == 4
        assert model['rogue_particles'] == count
        assert model['active_particles'] == 4-count
        assert model['escaped_particles'] == 0
        assert model['rogue_mass_kg'] == pytest.approx(.001*count)
        assert model['mass_balance_error_kg'] == pytest.approx(0.)
        for snapshot in tmp_path.glob(f'{tag}_particleData_*.csv'):
            with snapshot.open(newline='') as output:
                for _ in range(5):
                    next(output)
                particles = list(csv.reader(output))
            assert len(particles) == 4-count
            for particle in particles:
                assert particle[1] == '2020-01-01T00:00:00'
                assert tuple(map(float, particle[6:9])) == (6., 11., 8.)
