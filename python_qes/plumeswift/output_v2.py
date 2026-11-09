"""Multi-source QES NetCDF output with independent field layouts (GPL-3.0)."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import sys

import numpy as np

from .io import _staged_netcdf_path
from .timing import elapsed_milliseconds


_TYPES = (np.dtype('i4'), np.dtype('f4'), np.dtype('f8'))
_CHARACTER = np.dtype('S1')


@dataclass(frozen=True)
class OutputField:
    dimensions: tuple[str, ...] | str
    dtype: str | np.dtype
    units: str
    long_name: str


@contextmanager
def open_output(path, *, start=None, enabled=True):
    """Open a NetCDF output, or a disabled sink that only prepares sources."""
    if not enabled:
        print(f'[!!!WARNING!!!]\toutput disabled for file: {path}', file=sys.stderr)
        yield NullOutput()
        return

    from netCDF4 import Dataset

    with _staged_netcdf_path(path) as temporary:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            yield MultiSourceOutput(dataset, start)


class NullOutput:
    """Keep source preparation available when NetCDF output is disabled."""

    def __init__(self):
        self._prepare = {}

    def attach(self, source, fields, read, *, prepare=None):
        self._prepare[source] = prepare

    def push(self, source, time, *, prepare=False):
        callback = self._prepare[source]
        if prepare and callback is not None:
            callback(time)

    def _ignore(self, *args, **kwargs):
        pass

    define_axis = define_dimension = define_dimension_set = _ignore
    remove_field = remove_static_fields = set_start_time = _ignore
    new_time_entry = push_fields = save = _ignore


class MultiSourceOutput:
    def __init__(self, dataset, start):
        self._dataset = dataset
        self._start = None
        self._current = None
        self._sources = {}
        self._field_sources = {}
        self._dimension_sets = {'time': ('t',)}
        self._pushed = set()
        dataset.createDimension('t', None)
        dataset.createDimension('dateStrLen', 19)
        self._time = self._variable('t', OutputField(('t',), 'f8', 's', 'time since start of simulation'))
        self._timestamp = self._variable('timestamp', OutputField(
            ('t', 'dateStrLen'), 'S1', '--', 'date time using format: YYYY-MM-DDThh:mm:ss'))
        if start is not None:
            self.set_start_time(start)

    def _variable(self, name, field):
        dims = self._dimensions(field)
        variable = self._dataset.createVariable(name, field.dtype, dims)
        variable.units = field.units
        variable.long_name = field.long_name
        return variable

    def _dimensions(self, field):
        dims = field.dimensions
        if isinstance(dims, str):
            if dims not in self._dimension_sets:
                raise ValueError(f'Unknown output dimension set: {dims}')
            return self._dimension_sets[dims]
        return dims

    def define_axis(self, name, values, *, units, long_name):
        values = np.asarray(values)
        if values.ndim != 1 or not values.size or values.dtype not in _TYPES:
            raise ValueError(f'Axis {name} must be a nonempty int32, float32 or float64 vector')
        self.define_dimension(name, len(values))
        self._variable(name, OutputField((name,), values.dtype, units, long_name))[:] = values

    def define_dimension(self, name, size):
        if name in self._dataset.dimensions or name in self._dataset.variables:
            raise ValueError(f'Output dimension already defined: {name}')
        self._dataset.createDimension(name, size)

    def define_dimension_set(self, name, dimensions):
        """Register an ordered dimension layout that attached fields can reuse."""
        dims = tuple(dimensions)
        if not dims or any(dim not in self._dataset.dimensions for dim in dims):
            raise ValueError(f'Invalid output dimension set: {name}')
        if name in self._dimension_sets and self._dimension_sets[name] != dims:
            raise ValueError(f'Incompatible output dimension set: {name}')
        self._dimension_sets[name] = dims

    def attach(self, source, fields, read, *, prepare=None):
        """Attach a reader returning field values and an optional preparation callback."""
        if source in self._sources:
            raise ValueError(f'Output source already attached: {source}')
        for name, field in fields.items():
            dims = self._dimensions(field)
            dtype = np.dtype(field.dtype)
            if name in self._dataset.variables:
                raise ValueError(f'Output field already defined: {name}')
            if (not dims or any(dim not in self._dataset.dimensions for dim in dims)
                    or 't' in dims[1:] or dtype not in (*_TYPES, _CHARACTER)
                    or ('dateStrLen' in dims and dtype != _CHARACTER)
                    or (dtype == _CHARACTER and dims == ('t',))):
                raise ValueError(f'Invalid output field layout: {name}')
        variables = {name: self._variable(name, field) for name, field in fields.items()}
        self._sources[source] = (variables, read, prepare)
        self._field_sources.update(dict.fromkeys(variables, source))

    def remove_field(self, name):
        """Stop writing an attached field and preserve its NetCDF variable and data."""
        if name in self._field_sources:
            source = self._field_sources.pop(name)
            del self._sources[source][0][name]

    def remove_static_fields(self):
        """Stop writing time-independent fields while retaining their saved data."""
        for name in tuple(self._field_sources):
            if self._dataset.variables[name].dimensions[0] != 't':
                self.remove_field(name)

    def set_start_time(self, start):
        """Set the simulation origin used for subsequent records."""
        if not isinstance(start, datetime) or start.utcoffset() is not None:
            raise ValueError('QES output requires a start datetime without a timezone offset')
        self._start = start
        self._time.simulation_start = start.isoformat()

    def new_time_entry(self, time, *, allow_duplicate=False):
        if not isinstance(time, datetime) or time.utcoffset() is not None:
            raise ValueError('QES output requires a datetime without a timezone offset')
        if time == self._current and not allow_duplicate:
            return
        if self._start is None:
            print('[!!!WARNING!!!]\tstart time not defined in output -> use first time entry as start time',
                  file=sys.stderr)
            self.set_start_time(time)
        index = len(self._dataset.dimensions['t'])
        stamp = time.isoformat(timespec='seconds').encode('ascii')
        self._time[index] = elapsed_milliseconds(time, self._start)/1000
        self._timestamp[index] = np.frombuffer(stamp, dtype='S1')
        self._current = time
        self._pushed.clear()

    def push(self, source, time, *, prepare=False):
        """Write one source once for the current record, optionally preparing it first."""
        self._check_current_time(time)
        variables, _, prepare_data = self._sources[source]
        if prepare and prepare_data is not None:
            prepare_data(time)
        if source in self._pushed:
            return
        self._write_fields(((source, name) for name in variables))
        self._pushed.add(source)

    def push_fields(self, time, fields=None):
        """Write named fields directly, or all active fields when names are omitted."""
        self._check_current_time(time)
        names = sorted(self._field_sources) if fields is None else tuple(fields)
        bindings = []
        for name in names:
            if name not in self._field_sources:
                raise ValueError(f'Output field is not active: {name}')
            bindings.append((self._field_sources[name], name))
        self._write_fields(bindings)

    def _check_current_time(self, time):
        if self._current is None or time != self._current:
            raise ValueError('Output timestamp does not match the current entry')

    def _write_fields(self, bindings):
        index = len(self._dataset.dimensions['t'])-1
        values_by_source = {}
        record = []
        for source, name in bindings:
            variables, read, _ = self._sources[source]
            variable = variables[name]
            timed = variable.dimensions[0] == 't'
            if not timed and index:
                continue
            if source not in values_by_source:
                values_by_source[source] = read()
            shape = variable.shape[1:] if timed else variable.shape
            raw = values_by_source[source][name]
            if variable.dtype == _CHARACTER and isinstance(raw, bytes):
                value = np.frombuffer(raw, dtype=_CHARACTER)
            else:
                value = np.asarray(raw)
            if variable.dtype == _CHARACTER and value.dtype != _CHARACTER:
                raise ValueError(f'Character field {name} requires bytes or an S1 array')
            if value.shape != shape and not (value.ndim == 1 and value.size == np.prod(shape)):
                raise ValueError(f'Invalid output shape for {name}: {value.shape}, expected {shape}')
            record.append((variable, index if timed else slice(None), value.reshape(shape)))
        for variable, key, value in record:
            variable[key] = value

    def save(self, time):
        """Push attached sources without invoking their preparation callbacks."""
        for source in self._sources:
            self.push(source, time)
