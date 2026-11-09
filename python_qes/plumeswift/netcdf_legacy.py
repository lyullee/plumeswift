"""Direct legacy QES NetCDF dimensions, fields and hyperslabs."""
from contextlib import contextmanager

import numpy as np

from .io import _open_netcdf, _staged_netcdf_path


def _hyperslab(variable, start, count, *, writing=False):
    if len(start) != variable.ndim or len(count) != variable.ndim:
        raise ValueError('Hyperslab rank must match the NetCDF field')
    if any(int(value) != value or value < 0 for value in (*start, *count)):
        raise ValueError('Hyperslab indices and counts must be nonnegative integers')
    for dimension, first, length in zip(variable.get_dims(), start, count, strict=True):
        if first+length > len(dimension) and not (writing and dimension.isunlimited()):
            raise IndexError(f'Hyperslab exceeds NetCDF dimension {dimension.name}')
    return tuple(slice(int(first), int(first+length)) for first, length in zip(start, count, strict=True))


def _native_numeric_cast(values, dtype):
    """Apply the range checks of NetCDF's typed int, float and double conversion."""
    target = np.dtype(dtype)
    if target.type is values.dtype.type or target.type not in (np.int32, np.float32, np.float64):
        return values.astype(target, copy=False)
    if values.dtype.kind not in 'iuf':
        raise TypeError('NetCDF cannot convert between text and numeric types')
    if target.type is np.int32:
        limits = np.iinfo(target)
        if values.dtype.kind == 'f':
            # Compare float32 values without rounding INT_MAX to 2**31.
            values = values.astype(np.float64, copy=False)
        if np.any((values < limits.min) | (values > limits.max)):
            raise OverflowError('NetCDF numeric conversion not representable')
        if values.dtype.kind == 'f':
            values = np.where(np.isnan(values), limits.min, values)
    elif target.type is np.float32 and values.dtype.type is np.float64:
        if np.any(np.abs(values) > float(np.finfo(target).max)):
            raise OverflowError('NetCDF numeric conversion not representable')
    return values.astype(target, copy=False)


@contextmanager
def open_legacy_input(path, *, raw=True):
    """Read NetCDF directly, with native unscaled values by default."""
    with _open_netcdf(path, raw=raw) as dataset:
        yield LegacyNetCDFInput(dataset, raw=raw)


class LegacyNetCDFInput:
    def __init__(self, dataset, *, raw=True):
        self._dataset = dataset
        self._raw = raw

    def dimension_size(self, name):
        return len(self._dataset.dimensions[name])

    def field(self, name):
        return self._dataset.variables[name]

    def read(self, name, *, dtype=None, start=None, count=None):
        variable = self.field(name)
        if (start is None) != (count is None):
            raise ValueError('Reading a hyperslab requires both start and count')
        index = (...,) if start is None else _hyperslab(variable, tuple(start), tuple(count))
        values = np.asanyarray(variable[index])
        if dtype is not None and self._raw:
            return _native_numeric_cast(values, dtype)
        return np.asanyarray(values, dtype=dtype)


@contextmanager
def open_legacy_output(path):
    """Publish an output atomically after the legacy direct writes finish."""
    from netCDF4 import Dataset

    with _staged_netcdf_path(path) as temporary:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            yield LegacyNetCDFOutput(dataset)


class LegacyNetCDFOutput:
    def __init__(self, dataset):
        self._dataset = dataset

    def define_dimension(self, name, size=0):
        if int(size) != size or size < 0:
            raise ValueError('NetCDF dimension size must be a nonnegative integer')
        return self._dataset.createDimension(name, int(size) if size else None)

    def dimension(self, name):
        return self._dataset.dimensions[name]

    def define_field(self, name, dimensions, dtype, *, units, long_name):
        variable = self._dataset.createVariable(name, dtype, tuple(dimensions))
        variable.set_auto_maskandscale(False)
        variable.units = units
        variable.long_name = long_name
        return variable

    def add_attribute(self, field, name, value):
        self._dataset.variables[field].setncattr(name, value)

    def write(self, name, values, *, start=None, count=None):
        variable = self._dataset.variables[name]
        values = np.frombuffer(values, dtype='S1') if isinstance(values, bytes) else np.asanyarray(values)
        if start is None:
            if count is not None:
                raise ValueError('A write count requires a starting index')
            index, shape = (...,), variable.shape
        elif count is None:
            index = tuple(start)
            _hyperslab(variable, index, (1,)*len(index), writing=True)
            index, shape = tuple(map(int, index)), ()
        else:
            shape = tuple(count)
            index = _hyperslab(variable, tuple(start), shape, writing=True)
            shape = tuple(map(int, shape))
        if values.size != np.prod(shape, dtype=np.int64):
            raise ValueError('Data size does not match the NetCDF field selection')
        if values.size:
            if getattr(variable.dtype, 'type', None) in (np.int32, np.float32, np.float64) and not np.ma.is_masked(values):
                values = _native_numeric_cast(values, variable.dtype)
            variable[index] = values.reshape(shape)
            self._dataset.sync()
