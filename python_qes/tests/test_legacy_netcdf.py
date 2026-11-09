from types import SimpleNamespace

import numpy as np
import pytest

from plumeswift.mixing_length import NetCDFMixing
from plumeswift.netcdf_legacy import open_legacy_input, open_legacy_output


def test_legacy_hyperslab_preserves_unscaled_numeric_data(tmp_path):
    path = tmp_path/'저장.nc'
    with open_legacy_output(path) as writer:
        writer.define_dimension('t')
        writer.define_dimension('x', 2)
        assert writer.dimension('t').isunlimited()
        field = writer.define_field('packed', ('t', 'x'), 'i4', units='m', long_name='packed')
        field.setncattr('scale_factor', .25)
        writer.write('packed', [8, 12], start=(1, 0), count=(1, 2))
    with open_legacy_input(path) as reader:
        assert reader.dimension_size('t') == 2
        np.testing.assert_array_equal(reader.read('packed', dtype='f4', start=(1, 0), count=(1, 2)), [[8, 12]])
        with pytest.raises(IndexError, match='dimension x'):
            reader.read('packed', start=(1, 1), count=(1, 2))
    with open_legacy_input(path, raw=False) as reader:
        np.testing.assert_array_equal(reader.read('packed')[1], [2., 3.])
        assert np.ma.getmaskarray(reader.read('packed'))[0].all()


def test_legacy_failed_write_preserves_previous_output(tmp_path):
    path = tmp_path/'existing.nc'
    path.write_bytes(b'previous artifact')
    with pytest.raises(ValueError, match='Data size'):
        with open_legacy_output(path) as writer:
            writer.define_dimension('x', 3)
            writer.define_field('values', ('x',), 'f8', units='m', long_name='values')
            writer.write('values', [1., 2.])
    assert path.read_bytes() == b'previous artifact'


def test_legacy_numeric_reads_check_native_conversion_range(tmp_path):
    path = tmp_path/'range.nc'
    with open_legacy_output(path) as writer:
        writer.define_dimension('x', 4)
        writer.define_field('double', ('x',), 'f8', units='1', long_name='double')
        writer.define_field('single', ('x',), 'f4', units='1', long_name='single')
        writer.define_field('text', ('x',), 'S1', units='1', long_name='text')
        writer.write('double', [2147483647., 2147483647.75, 1e100, np.nan])
        writer.write('single', np.array([2147483520., 2147483648., np.inf, np.nan], dtype=np.float32))
        writer.write('text', b'1234')
    with open_legacy_input(path) as reader:
        assert reader.read('double', dtype='i4', start=(0,), count=(1,)).item() == 2147483647
        assert reader.read('single', dtype='i4', start=(0,), count=(1,)).item() == 2147483520
        for name in ('double', 'single'):
            with pytest.raises(OverflowError, match='not representable'):
                reader.read(name, dtype='i4', start=(1,), count=(1,))
            assert reader.read(name, dtype='i4', start=(3,), count=(1,)).item() == -2147483648
        with pytest.raises(OverflowError, match='not representable'):
            reader.read('double', dtype='f4')
        assert np.isposinf(reader.read('single', dtype='f4')[2])
        with pytest.raises(TypeError, match='text and numeric'):
            reader.read('text', dtype='i4')


def test_legacy_write_range_failure_preserves_previous_file(tmp_path):
    path = tmp_path/'range.nc'
    path.write_bytes(b'previous artifact')
    for dtype, value in (('f4', 1e100), ('i4', 2147483647.75)):
        with pytest.raises(OverflowError, match='not representable'):
            with open_legacy_output(path) as writer:
                writer.define_dimension('x', 2)
                writer.define_field('values', ('x',), dtype, units='1', long_name='values')
                writer.write('values', 1.9, start=(0,))
                writer.write('values', value, start=(1,))
        assert path.read_bytes() == b'previous artifact'


def test_legacy_mixing_reader_rejects_digit_only_text_field(tmp_path):
    path = tmp_path/'mixing.nc'
    shape = (2, 3, 4)
    with open_legacy_output(path) as writer:
        for axis, size in zip('zyx', shape, strict=True):
            writer.define_dimension(axis, size)
        writer.define_field('mixlength', ('z', 'y', 'x'), 'S1', units='m', long_name='distance')
        writer.write('mixlength', b'1'*24)
    workspace = SimpleNamespace(arrays={'icellflag': np.zeros(shape, dtype=np.int32)})
    with pytest.raises(TypeError, match='text and numeric'):
        NetCDFMixing(path, native_hyperslab=True)(workspace)
