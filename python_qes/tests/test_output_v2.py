from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from plumeswift.io import _open_netcdf, netcdf_dates
from plumeswift.concentration_output import native_collection_series
from plumeswift.output_example import write_output_example
from plumeswift.output_v2 import OutputField, open_output


def test_numeric_clock_uses_archived_origin_for_modern_reads(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path/'numeric_clock.nc'
    start = datetime(2025, 3, 2, microsecond=125000)
    with Dataset(path, 'w') as nc:
        nc.createDimension('t', 2)
        time = nc.createVariable('t', 'f8', ('t',))
        time.simulation_start = start.isoformat()
        time[:] = [0., .123456]
    with _open_netcdf(path) as nc:
        assert netcdf_dates(nc) == (start, start+timedelta(microseconds=123456))
        fallback = datetime(2022, 1, 1)
        assert netcdf_dates(nc, native_timestamps=True) == (fallback, fallback+timedelta(milliseconds=123))


def test_native_numeric_clock_is_read_even_when_timestamp_is_present(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path/'native_clock.nc'
    start = datetime(2025, 3, 2)
    with Dataset(path, 'w') as nc:
        nc.createDimension('t', 1)
        nc.createDimension('dateStrLen', 19)
        nc.createVariable('t', 'f8', ('t',))[0] = 1e100
        nc.createVariable('timestamp', 'S1', ('t', 'dateStrLen'))[0] = np.frombuffer(
            start.isoformat().encode('ascii'), dtype='S1')
    with _open_netcdf(path, raw=True) as nc:
        assert netcdf_dates(nc) == (start,)
        with pytest.raises(OverflowError, match='numeric conversion not representable'):
            netcdf_dates(nc, native_timestamps=True)


def test_output_example_records(tmp_path):
    path = tmp_path/'example.nc'
    start = datetime(2022, 1, 1)
    write_output_example(path, start=start)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['t'][:], [120, 240])
        assert output['t'].simulation_start == start.isoformat()
        np.testing.assert_array_equal(output['t_collection'][:], [1000, 1000])
        assert output['p'].dimensions == ('t', 'z_c', 'y_c', 'x_c')
        assert output['p'].dtype == np.dtype('i4')
        assert output['c'].dtype == np.dtype('f4')
        assert output['s'].dimensions == ('t', 'k_y', 'k_x')
        assert output['s'].units == 'm-3'
        np.testing.assert_array_equal(output['p'][:], np.zeros((2, 2, 5, 5)))
        np.testing.assert_array_equal(output['s'][:], np.zeros((2, 5, 5)))


def test_multi_source_output_deduplicates_push_and_keeps_static_values(tmp_path):
    path = tmp_path/'fields.nc'
    first = datetime(2022, 1, 1)
    second = first+timedelta(seconds=1)
    values = dict(count=1, grid=[3., 4.])
    with open_output(path) as output:
        output.define_axis('x', np.arange(2, dtype=np.int32), units='m', long_name='x')
        output.attach('test', {'count': OutputField(('t',), 'i4', '--', 'count'),
                               'grid': OutputField(('x',), 'f8', '--', 'grid')}, lambda: values)
        output.new_time_entry(first)
        output.push('test', first)
        values.update(count=2, grid=[5., 6.])
        output.new_time_entry(first)
        output.save(first)
        with pytest.raises(ValueError, match='timestamp'):
            output.save(second)
        output.new_time_entry(second)
        output.save(second)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['t'][:], [0, 1])
        np.testing.assert_array_equal(output['count'][:], [1, 2])
        np.testing.assert_array_equal(output['grid'][:], [3., 4.])

    with pytest.raises(ValueError, match='shape'):
        with open_output(path, start=first) as output:
            output.attach('bad', {'count': OutputField(('t',), 'i4', '--', 'count')},
                          lambda: {'count': [1, 2]})
            output.new_time_entry(first)
            output.save(first)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['count'][:], [1, 2])


def test_native_collection_output_has_schema_before_first_sample(tmp_path):
    path = tmp_path/'plume.nc'
    config = SimpleNamespace(lower=(.35, .45, .25), upper=(23.75, 23.65, 19.6), boxes=(3, 2, 1))
    start = datetime(2022, 1, 1)
    with native_collection_series(path, config, start):
        pass
    with _open_netcdf(path) as output:
        assert output['c'].shape == (0, 1, 2, 3)
        assert output['c'].dtype == np.dtype('f4')
        assert output['c'].units == 'g m-3'
        assert output['t'].simulation_start == start.isoformat()

    fields = dict(c=np.full((1, 2, 3), .125), p_count=np.ones((1, 2, 3), dtype=np.int32), t_avg=.25)
    with native_collection_series(path, config, start) as write:
        write(fields, start.isoformat(), .75)
        write(fields, start.isoformat(), .875)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['t'][:], [.75, .875])
        np.testing.assert_array_equal(output['t_avg'][:], [.25, .25])
        np.testing.assert_array_equal(output['c'][:], np.full((2, 1, 2, 3), .125))


def test_select_and_remove_fields_preserves_schema_and_source_push(tmp_path):
    path = tmp_path/'selection.nc'
    first = datetime(2022, 1, 1)
    second = first+timedelta(seconds=1)
    values = dict(count=1, dropped=2)
    field = OutputField('time', 'i4', '--', 'counter')
    with open_output(path) as output:
        with pytest.raises(ValueError, match='Incompatible'):
            output.define_dimension_set('time', ('t', 't'))
        output.attach('test', {'count': field, 'dropped': field}, lambda: values)
        output.new_time_entry(first)
        output.push_fields(first, ('count',))
        values['count'] = 3
        output.push('test', first)
        values['count'] = 4
        output.save(first)
        output.new_time_entry(second)
        output.remove_field('dropped')
        output.remove_field('dropped')
        del values['dropped']
        with pytest.raises(ValueError, match='not active'):
            output.push_fields(second, ('count', 'dropped'))
        output.save(second)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['count'][:], [3, 4])
        assert output['dropped'][0] == 2
        assert np.ma.is_masked(output['dropped'][1])


def test_legacy_character_records_and_static_field_removal(tmp_path):
    path = tmp_path/'characters.nc'
    timestamp = datetime(2022, 1, 1)
    values = dict(code=b'AB', fixed=[3, 4])
    with open_output(path) as output:
        output.define_dimension('width', 2)
        output.attach('example', {
            'code': OutputField(('t', 'width'), 'S1', '--', 'code'),
            'fixed': OutputField(('width',), 'i4', '--', 'fixed'),
        }, lambda: values)
        output.new_time_entry(timestamp)
        output.save(timestamp)
        output.remove_static_fields()
        del values['fixed']
        with pytest.raises(ValueError, match='not active'):
            output.push_fields(timestamp, ('fixed',))
        output.new_time_entry(timestamp, allow_duplicate=True)
        values['code'] = [b'CD']
        with pytest.raises(ValueError, match='Character field'):
            output.save(timestamp)
        values['code'] = np.array([b'C', b'D'], dtype='S1')
        output.save(timestamp)
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['t'][:], [0, 0])
        np.testing.assert_array_equal(output['fixed'][:], [3, 4])
        np.testing.assert_array_equal(output['code'][:], [[b'A', b'B'], [b'C', b'D']])


def test_static_only_reader_is_used_only_at_first_record(tmp_path):
    path = tmp_path/'static.nc'
    first = datetime(2022, 1, 1)
    second = first+timedelta(seconds=1)
    reads = []
    values = {'fixed': [3, 4]}

    def read():
        reads.append(True)
        return {'fixed': values['fixed']}

    with open_output(path, start=first) as output:
        output.define_dimension('width', 2)
        output.attach('static', {'fixed': OutputField(('width',), 'i4', '--', 'fixed')}, read)
        output.new_time_entry(first)
        output.save(first)
        values.clear()
        output.new_time_entry(second)
        output.save(second)
    assert reads == [True]
    with _open_netcdf(path) as output:
        np.testing.assert_array_equal(output['fixed'][:], [3, 4])


def test_disabled_output_prepares_source_without_reading_or_touching_file(tmp_path, capsys):
    path = tmp_path/'disabled.nc'
    path.write_bytes(b'existing file')
    first = datetime(2022, 1, 1)
    second = first+timedelta(seconds=1)
    prepared = []

    def read():
        pytest.fail('Disabled output read its source payload')

    with open_output(path, start=first, enabled=False) as output:
        output.attach('test', {'count': OutputField('time', 'i4', '--', 'count')}, read, prepare=prepared.append)
        output.new_time_entry(first)
        output.save(first)
        output.push('test', first, prepare=True)
        output.push('test', first, prepare=True)
        output.new_time_entry(second)
        output.push('test', second, prepare=True)
    assert prepared == [first, first, second]
    assert path.read_bytes() == b'existing file'
    assert capsys.readouterr().err == f'[!!!WARNING!!!]\toutput disabled for file: {path}\n'
