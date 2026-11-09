from dataclasses import replace
import errno
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from netCDF4 import Dataset
import numpy as np
import pytest

from plumeswift.wind_case import read_wind_case
from plumeswift.io import _open_netcdf
from plumeswift.wind_run import write_wind_run
from plumeswift.wrf import FireMeshInput, frame_checksum
from plumeswift.wrf_exchange import WRFExchange, _live_netcdf, fire_height_winds


FIXTURE = Path(__file__).resolve().parents[2]/'results/python_port/wrf_exchange'


def test_plain_wrf_fire_output_is_distinguished_from_qes_exchange(tmp_path):
    path = tmp_path/'wrfout.nc'
    with Dataset(path, 'w', format='NETCDF3_CLASSIC') as nc:
        for name in ('U', 'V', 'PH', 'PHB', 'HGT', 'ZSF', 'FXLONG', 'UF', 'VF'):
            nc.createVariable(name, 'f4')
    original = path.read_bytes()
    with pytest.raises(ValueError, match=r'fire-mesh input.*U0_FMW.*WRF \+ DEM'):
        FireMeshInput.read(path)
    with pytest.raises(ValueError, match=r'live frame exchange.*FRAME0_FMW.*WRF \+ DEM'):
        WRFExchange.open(path, 1)
    assert path.read_bytes() == original


@pytest.mark.parametrize('shared_records', [False, True])
def test_legacy_exchange_waits_for_unwritten_geometry_records(tmp_path, monkeypatch, shared_records):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    snapshots = {}
    with Dataset(path, 'r+') as nc:
        nc['FRAME0_FMW'][0] = 1
        for name in ('ZSF', 'FWH'):
            field = nc[name]
            snapshots[name] = np.array(field[0])
            nc.renameVariable(name, name+'_original')
            record_dim = nc['U0_FMW'].dimensions[0] if shared_records else name+'_records'
            if not shared_records:
                nc.createDimension(record_dim, None)
            nc.createVariable(name, field.dtype, (record_dim, *field.dimensions[1:]))

    def publish_next_record(_delay):
        name = next(iter(snapshots))
        with Dataset(path, 'r+') as nc:
            nc[name][0] = snapshots.pop(name)

    monkeypatch.setattr('plumeswift.wrf_exchange.time.sleep', publish_next_record)
    exchange = WRFExchange.open(path, 1, timeout=1., legacy_protocol=True)
    assert list(snapshots) == ['FWH']
    snapshot, stamp = exchange.receive(1)
    assert not snapshots and snapshot.record == 9 and stamp == '2022-01-01T00:00:00'


def test_legacy_exchange_accepts_single_record_pnetcdf_peer(tmp_path):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    with _open_netcdf(source, raw=True) as original, Dataset(str(path), 'w', format='NETCDF3_64BIT_OFFSET') as peer:
        original.set_auto_chartostring(False)
        time = original['U0_FMW'].dimensions[0]
        peer.setncatts(original.__dict__)
        for name, dimension in original.dimensions.items():
            peer.createDimension(name, None if name == time else len(dimension))
        for name, field in original.variables.items():
            peer.createVariable(name, field.dtype, field.dimensions)[:] = field[:1] if time in field.dimensions else field[:]
        for name in ('U0_FMW', 'V0_FMW', 'W0_FMW', 'HT_FMW', 'CHSUM0_FMW'):
            peer[name][0] = original[name][-1]
        peer['FRAME0_FMW'][0] = 1

    tree = ET.parse(FIXTURE/'winds.xml')
    tree.find('simulationParameters/totalTimeIncrements').text = '1'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    case = read_wind_case(xml, legacy_wrf_protocol=True, wrf_timeout=1.)
    stream = case.iter_results()
    _, result = next(stream)
    assert result['wrf_frame']['input_record'] == 0
    with Dataset(str(path), 'r+') as peer:
        assert peer['FRAME_FMW'][0] == 1
        assert peer['CHSUM_FMW'][0] == result['wrf_frame']['output_checksum']
        peer['FRAME0_FMW'][0] = 2
    with pytest.raises(StopIteration):
        next(stream)
    with Dataset(str(path)) as peer:
        assert peer['FRAME_FMW'][0] == -99


def test_exchange_retries_a_marked_wind_record_with_unwritten_values(tmp_path, monkeypatch):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    with Dataset(path, 'r+') as nc:
        field = nc['V0_FMW']
        expected = np.array(field[1])
        field[1] = np.ma.masked_all(expected.shape, dtype=expected.dtype)

    def publish_wind(_delay):
        with Dataset(path, 'r+') as nc:
            nc['V0_FMW'][1] = expected

    monkeypatch.setattr('plumeswift.wrf_exchange.time.sleep', publish_wind)
    exchange = WRFExchange.open(path, 1, timeout=1.)
    snapshot, _ = exchange.receive(1)
    nx, ny, _ = snapshot.cells
    np.testing.assert_array_equal(snapshot.v, expected[:, :ny, :nx])


def test_exchange_waits_for_a_partial_timestamp(tmp_path, monkeypatch):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    expected = np.frombuffer(b'2020-09-14_00:00:00', dtype='S1')
    with Dataset(path, 'r+') as nc:
        nc['Times'].set_auto_chartostring(False)
        nc['Times'][1] = np.ma.array(expected, mask=np.arange(len(expected)) >= 10)

    def publish_timestamp(_delay):
        with Dataset(path, 'r+') as nc:
            nc['Times'][1] = expected

    monkeypatch.setattr('plumeswift.wrf_exchange.time.sleep', publish_timestamp)
    exchange = WRFExchange.open(path, 1, timeout=1.)
    snapshot, stamp = exchange.receive(1)
    assert snapshot.record == 1 and stamp == '2020-09-14T00:00:00'


@pytest.mark.parametrize('peer_finishes, packed, empty, resets_response', [
    (False, False, False, False), (True, False, False, False), (True, True, False, False),
    (True, False, True, False), (True, False, False, True),
])
def test_exchange_completes_only_after_final_peer_frame(tmp_path, peer_finishes, packed, empty, resets_response):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    tree = ET.parse(FIXTURE/'winds.xml')
    tree.find('simulationParameters/totalTimeIncrements').text = '1'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    with Dataset(path, 'r+') as nc:
        if empty:
            for name in ('UF', 'VF', 'CHSUM_FMW', 'FRAME_FMW'):
                nc.renameVariable(name, name+'_old')
            nc.createDimension('response', None)
            for name in ('UF', 'VF', 'CHSUM_FMW', 'FRAME_FMW'):
                old = nc[name+'_old']
                nc.createVariable(name, old.dtype, ('response', *old.dimensions[1:]))
        else:
            nc['FRAME_FMW'][0] = -1 if resets_response else 0
        nc['FRAME0_FMW'][4] = 0
        if packed:
            for name in ('FRAME0_FMW', 'FRAME_FMW', 'CHSUM0_FMW', 'CHSUM_FMW'):
                nc[name].add_offset = 5
            for name in ('U0_FMW', 'V0_FMW', 'W0_FMW', 'HT_FMW', 'ZSF', 'FWH', 'UF', 'VF'):
                nc[name].add_offset = .12345
                nc[name].scale_factor = .25
    case = read_wind_case(xml, wrf_timeout=.15)
    with pytest.raises(ValueError, match='iter_results'):
        case.run()
    stream = case.iter_results()
    step, result = next(stream)
    assert step.timestamp == '2020-09-14T00:00:00'
    assert result['wrf_frame']['input_record'] == 1
    nx, ny, _ = step.wrf.cells
    with Dataset(path, 'r+') as nc:
        nc.set_auto_scale(False)
        assert nc['FRAME_FMW'][0] == 1
        uf, vf = (np.asarray(nc[name][0, :ny, :nx]) for name in ('UF', 'VF'))
        assert nc['CHSUM_FMW'][0] == frame_checksum(uf, vf)
        if resets_response:
            nc['FRAME_FMW'][0] = -1
        elif peer_finishes:
            nc['FRAME0_FMW'][4] = 2
    if resets_response:
        with pytest.raises(ValueError, match='completion conflicts'):
            case.wrf_exchange.finish()
        with Dataset(path, 'r+') as nc:
            nc['FRAME0_FMW'][4] = 2
    if peer_finishes:
        with pytest.raises(StopIteration):
            next(stream)
    else:
        with pytest.raises(TimeoutError, match='input frame 2'):
            next(stream)
    with Dataset(path) as nc:
        nc.set_auto_scale(False)
        assert nc['FRAME_FMW'][0] == (-99 if peer_finishes else 1)


def test_live_exchange_reopens_unicode_path(tmp_path, monkeypatch):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    folder = tmp_path/'WRF 한글'
    folder.mkdir()
    path = folder/'wrf.nc'
    shutil.copyfile(source, path)
    monkeypatch.chdir(tmp_path)
    with _live_netcdf(path, 'r+') as nc:
        nc['FRAME_FMW'][0] = 23
    with _live_netcdf(path, 'r') as nc:
        assert nc['FRAME_FMW'][0] == 23

    if os.name == 'nt':
        from plumeswift import wrf_exchange

        temporary = tmp_path/'other_volume'
        temporary.mkdir()
        monkeypatch.setattr(wrf_exchange, 'gettempdir', lambda: str(temporary))
        link = os.link
        attempts = []

        def link_same_volume(source, alias):
            attempts.append(alias)
            if alias.is_relative_to(temporary):
                raise OSError(errno.EXDEV, 'cross-volume hard link')
            link(source, alias)
            assert os.path.samefile(source, alias)

        monkeypatch.setattr(wrf_exchange.os, 'link', link_same_volume)
        with _live_netcdf(path, 'r+') as nc:
            nc['FRAME_FMW'][0] = 24
        assert len(attempts) == 2
        assert attempts[1].parent.parent == tmp_path
        assert not any(alias.exists() for alias in attempts)
        with _live_netcdf(path, 'r') as nc:
            assert nc['FRAME_FMW'][0] == 24


def test_offline_unicode_netcdf_stays_disk_backed(tmp_path, monkeypatch):
    import netCDF4

    source = tmp_path/'source.nc'
    with Dataset(source, 'w') as nc:
        nc.createVariable('value', 'i4').assignValue(27)
    directory = tmp_path/'한글'
    directory.mkdir()
    path = directory/source.name
    source.rename(path)
    monkeypatch.chdir(tmp_path)
    aliases = []

    def open_dataset(filename, **kwargs):
        assert 'memory' not in kwargs
        if os.name == 'nt' and not str(filename).isascii():
            raise FileNotFoundError(filename)
        if os.name == 'nt':
            assert os.path.samefile(path, filename)
            aliases.append(Path(filename))
        return Dataset(filename, **kwargs)

    monkeypatch.setattr(netCDF4, 'Dataset', open_dataset)
    with _open_netcdf(path) as nc:
        assert nc['value'].getValue() == 27
    if os.name == 'nt':
        alias, = aliases
        assert not alias.exists()


def test_legacy_exchange_publishes_iteration_limit_and_uses_latest_wrf_record(tmp_path):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    tree = ET.parse(FIXTURE/'winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('maxIterations').text = '1'
    sim.find('tolerance').text = '1e-12'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    stamps = ('2020-09-14_00:02:00', '2020-09-14_00:02:30')
    with Dataset(path, 'r+') as nc:
        nc['FRAME_FMW'][0] = 0
        nc['FRAME0_FMW'][0] = 1
        nc['FRAME0_FMW'][-1] = 0
        nc['Times'][-1] = np.frombuffer(stamps[0].encode('ascii'), dtype='S1')
    modern = read_wind_case(xml, poisson_backend='cpu-native-rb', wrf_timeout=.1)
    with pytest.raises(RuntimeError, match='did not converge'):
        next(modern.iter_results())
    with Dataset(path, 'r+') as nc:
        nc['FRAME_FMW'][0] = 17
    legacy = read_wind_case(xml, poisson_backend='cpu-native-rb', legacy_wrf_protocol=True, wrf_timeout=.1)
    stream = legacy.iter_results()
    step, first = next(stream)
    assert step.timestamp == stamps[0].replace('_', 'T')
    assert first['wrf_frame']['input_record'] == 9 and not first['solver'].converged
    nx, ny, _ = step.wrf.cells
    with pytest.raises(TimeoutError, match='input frame 2'):
        legacy.wrf_exchange.receive(2)
    with Dataset(path, 'r+') as nc:
        assert nc['FRAME_FMW'][0] == 1
        nc['FRAME_FMW'][0] = 0
        startup_height = np.array(nc['FWH'][0, :ny, :nx])
        nc['FWH'][0, :ny, :nx] = startup_height + 24
        # An older frame 2 is already present in the fixture, but only the
        # newest wind record is authoritative for the native-style exchange.
        nc['FRAME0_FMW'][-1] = 2
        nc['Times'][-1] = np.frombuffer(stamps[1].encode('ascii'), dtype='S1')
    step, second = next(stream)
    assert step.timestamp == stamps[1].replace('_', 'T')
    assert second['wrf_frame']['input_record'] == 9 and not second['solver'].converged
    expected_winds = fire_height_winds(second['workspace'], startup_height, step.halo.cells(step.wrf.spacing),
                                       np.float32(step.wrf.spacing[2]))
    with Dataset(path, 'r+') as nc:
        assert nc['FRAME_FMW'][0] == 2
        for field, expected in zip(('UF', 'VF'), expected_winds, strict=True):
            np.testing.assert_array_equal(nc[field][0, :ny, :nx], expected)
        nc['FRAME_FMW'][0] = 0
        nc['FRAME0_FMW'][-1] = 3
        nc['Times'].set_auto_chartostring(False)
        nc['Times'][-1] = np.frombuffer(b'2020-09-14'+bytes(9), dtype='S1')
    with pytest.raises(StopIteration):
        next(stream)
    with Dataset(path) as nc:
        assert nc['FRAME_FMW'][0] == -99


def test_legacy_exchange_keeps_marker_index_when_peer_appends(tmp_path, monkeypatch):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    names = ('U0_FMW', 'V0_FMW', 'W0_FMW', 'HT_FMW', 'CHSUM0_FMW', 'FRAME0_FMW', 'Times')
    with Dataset(path, 'r+') as nc:
        nc.createDimension('incoming', None)
        for name in names:
            old = nc[name]
            nc.renameVariable(name, 'old_'+name)
            nc.createVariable(name, old.dtype, ('incoming', *old.dimensions[1:]))[:] = old[:]
        last = len(nc.dimensions['incoming'])-1
        nc['FRAME0_FMW'][last] = 0

    exchange = WRFExchange.open(path, 2, timeout=1., legacy_protocol=True)
    sleeps = []

    def append_response(seconds):
        assert not sleeps
        sleeps.append(seconds)
        with Dataset(path, 'r+') as nc:
            for name in names:
                nc[name][last+1] = nc[name][last]
            nc['FRAME0_FMW'][last] = 2
            nc['FRAME0_FMW'][last+1] = 3
            nc['Times'][last+1] = np.frombuffer(b'2020-09-14_00:03:00', dtype='S1')

    monkeypatch.setattr('plumeswift.wrf_exchange.time.sleep', append_response)
    snapshot, timestamp = exchange.receive(2)
    assert sleeps and snapshot.record == last+1
    assert timestamp == '2020-09-14T00:03:00'


def test_legacy_exchange_reads_input_markers_as_native_int32(tmp_path):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    with Dataset(path, 'r+') as nc:
        old = nc['FRAME0_FMW']
        nc.renameVariable('FRAME0_FMW', 'old_FRAME0_FMW')
        marks = nc.createVariable('FRAME0_FMW', 'f8', old.dimensions)
        marks[0], marks[-1] = 1.75, 2.75
        marks.add_offset = 40.
        marks.missing_value = np.array([1.75, 2.75])

    legacy = WRFExchange.open(path, 2, timeout=1., legacy_protocol=True)
    first, _ = legacy.receive(1)
    second, _ = legacy.receive(2)
    assert first.record == second.record == 9
    with pytest.raises(ValueError, match='one-dimensional integer field'):
        WRFExchange.open(path, 2).receive(2)

    with Dataset(path, 'r+') as nc:
        nc['FRAME0_FMW'].set_auto_scale(False)
        nc['FRAME0_FMW'][-1] = 2**31
    with pytest.raises(OverflowError, match='numeric conversion not representable'):
        legacy.receive(2)


@pytest.mark.parametrize('coupled', [False, True])
def test_qeswinds_wrf_zero_steps_create_empty_outputs_and_skip_publication(tmp_path, coupled):
    source = FIXTURE/'wrf.nc'
    if not source.exists():
        pytest.skip('Run scripts/check_python_wrf_exchange.py')
    path = tmp_path/'wrf.nc'
    shutil.copyfile(source, path)
    xml = tmp_path/'winds.xml'
    tree = ET.parse(FIXTURE/'winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('totalTimeIncrements').text = '0'
    sim.find('WRFCoupling').text = str(coupled).lower()
    tree.write(xml)
    with Dataset(path, 'r+') as nc:
        nc['FRAME0_FMW'][:] = 0
        nc['FRAME0_FMW'][0] = 1
        nc['FRAME_FMW'][0] = 19
        nc['CHSUM_FMW'][0] = 31415
        nc['UF'][0] = 8.
        nc['VF'][0] = 9.

    case = read_wind_case(xml, legacy_wrf_protocol=True, wrf_timeout=.2 if coupled else None)
    outputs = [tmp_path/name for name in ('workspace.nc', 'visualization.nc', 'turbulence.nc')]
    assert write_wind_run(case, workspace_path=outputs[0], visualization_path=outputs[1],
                          turbulence_path=outputs[2]) == []
    for path_output in outputs:
        with Dataset(path_output) as nc:
            assert len(nc.dimensions['t']) == 0 and 'simulation_start' not in nc['t'].ncattrs()
            assert not np.ma.is_masked(nc['x'][:])
    with Dataset(path) as nc:
        assert nc['FRAME_FMW'][0] == (-99 if coupled else 19)
        assert nc['CHSUM_FMW'][0] == 31415
        np.testing.assert_array_equal(nc['UF'][0], 8.)
        np.testing.assert_array_equal(nc['VF'][0], 9.)
