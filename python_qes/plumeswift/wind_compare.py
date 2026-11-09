"""Wind workspace and solver diagnostics ported from QES comparison tools."""

from dataclasses import asdict, replace

import numpy as np

from .io import _open_netcdf, netcdf_dates
from .netcdf_legacy import _native_numeric_cast


_CELL_AXES = ('z', 'y', 'x')
_FACE_AXES = ('z_face', 'y_face', 'x_face')
SOLVER_BACKENDS = ('cpu-sor', 'cpu-native-rb', 'cpu-native-serial', 'cpu-historical-sor', 'cuda-sor',
                   'cuda-dynamic-legacy', 'cuda-global-memory', 'cuda-shared-memory', 'cuda-device-parallel')


def _array(variable, index, *, allow_nonfinite=False, native_dtype=None):
    values = variable[index]
    if np.ma.is_masked(values):
        raise ValueError(f'Missing or nonfinite comparison values in {variable.name}')
    values = np.asarray(values)
    if native_dtype is not None:
        values = _native_numeric_cast(values, native_dtype)
    if not allow_nonfinite and not np.isfinite(values).all():
        raise ValueError(f'Missing or nonfinite comparison values in {variable.name}')
    return values


def _read(file, name, dimensions, index=None, *, native_layout=False, allow_nonfinite=False, native_dtype=None,
          allow_short_vector=False):
    variable = file.variables.get(name)
    if variable is None:
        raise ValueError(f'Missing wind comparison variable: {name}')
    if native_layout:
        expected = tuple(len(file.dimensions[axis+'_face'])-1 if axis in _CELL_AXES
                         else len(file.dimensions[axis]) for axis in dimensions)
        if len(expected) == 1:
            tail = expected[0]-variable.size
            if tail < 0 or (tail and not allow_short_vector):
                raise ValueError(f'{name} must match the stored shape of {dimensions}')
            values = _array(variable, ..., allow_nonfinite=allow_nonfinite, native_dtype=native_dtype).ravel()
            # Native full-variable reads leave the rest of the allocated vector at zero.
            return np.pad(values, (0, tail)) if tail else values
        if variable.ndim != len(expected) or any(size < need for size, need in zip(variable.shape, expected)):
            raise ValueError(f'{name} must cover the stored shape of {dimensions}')
        # Native terrain, flags and winds use the domain-sized leading slab.
        slab = tuple(slice(length) for length in expected)
        return _array(variable, slab if index is None else (index, *slab[1:]),
                      allow_nonfinite=allow_nonfinite, native_dtype=native_dtype)
    if variable.dimensions != dimensions:
        raise ValueError(f'{name} must use NetCDF dimensions {dimensions}')
    return _array(variable, slice(None) if index is None else index)


def _total(values):
    # Native accumulates in storage order using a float accumulator.
    if values.dtype == np.dtype('f4'):
        return np.cumsum(values, dtype=np.float32)[-1]
    return values.sum(dtype=np.float64)


def _abs_stats(delta):
    total = _total(delta)
    # The original updates its zero-initialized maximum only when delta > max.
    maximum = np.fmax.reduce(delta.ravel(), initial=0)
    return dict(total_abs=float(total), max_abs=float(maximum), mean_abs=float(total/delta.dtype.type(delta.size)))


def _difference(first, second):
    with np.errstate(invalid='ignore'):
        return _abs_stats(np.abs(first-second))


def _legacy_prefix(reference, candidate, name):
    # compareVar traverses the reference vector's length in storage order.
    if candidate.size < reference.size:
        raise ValueError(f'Candidate is shorter than the reference for {name}')
    return reference.ravel(), candidate.ravel()[:reference.size]


def _pearson_squared(first, second):
    if first.size < 2 or first.max() == first.min() or second.max() == second.min():
        return None
    count = first.dtype.type(first.size)
    left = first-_total(first)/count
    right = second-_total(second)/count
    degrees = first.dtype.type(first.size-1)
    left_std = np.sqrt(_total(left*left)/degrees)
    right_std = np.sqrt(_total(right*right)/degrees)
    denominator = left_std*right_std
    if denominator == 0:
        return None
    correlation = (_total(left*right)/degrees)/denominator
    squared = float(correlation*correlation)
    return min(squared, 1.) if first.dtype == np.dtype('f8') else squared


def compare_velocity(reference, candidate, *, precision='native'):
    """Compare raw face arrays, pairing the three components by stored index."""
    dtype = {'native': np.float32, 'float64': np.float64}.get(precision)
    if dtype is None:
        raise ValueError(f'Unknown comparison precision {precision!r}')
    left = [np.asarray(reference[name], dtype=dtype) for name in 'uvw']
    right = [np.asarray(candidate[name], dtype=dtype) for name in 'uvw']
    shape = left[0].shape
    if any(values.shape != shape or not values.size or not np.isfinite(values).all() for values in (*left, *right)):
        raise ValueError('Wind components require equal, nonempty shapes and finite values')
    with np.errstate(over='raise', invalid='raise'):
        fields = {key: _abs_stats(np.abs(a-b)) for key, a, b in zip('uvw', left, right, strict=True)}
        u, v, w = left
        first = np.sqrt(u*u+v*v+w*w)
        u, v, w = right
        second = np.sqrt(u*u+v*v+w*w)
        speed = _abs_stats(np.abs(first-second))
        speed['pearson_r_squared'] = _pearson_squared(first, second)
    return dict(fields=fields, speed=speed)


def compare_wind_files(reference, candidate, *, precision='native', by_index=False, native_layout=False,
                       include_speed=True):
    """Compare two workspace series with bounded memory and validated layouts."""
    dtype = {'native': np.float32, 'float64': np.float64}.get(precision)
    if dtype is None:
        raise ValueError(f'Unknown comparison precision {precision!r}')
    axes = (*_CELL_AXES, *_FACE_AXES)
    by_index = by_index or native_layout
    legacy_prefix = native_layout and not include_speed
    with _open_netcdf(reference, raw=native_layout) as first, _open_netcdf(candidate, raw=native_layout) as second:
        for name in _FACE_AXES if native_layout else axes:
            left_dim, right_dim = first.dimensions.get(name), second.dimensions.get(name)
            if (left_dim is None or right_dim is None or not len(left_dim) or not len(right_dim)
                    or (not legacy_prefix and len(left_dim) != len(right_dim))):
                raise ValueError(f'Wind workspace dimensions differ or are empty: {name}')
        left_time, right_time = (file.dimensions.get('t') for file in (first, second))
        if left_time is None or right_time is None:
            raise ValueError('Wind workspace requires a time dimension')
        record_count, candidate_count = len(left_time), len(right_time)
        if (candidate_count < record_count if native_layout else not 0 < record_count == candidate_count):
            raise ValueError('Wind workspace record counts do not match')
        for center, face in zip(_CELL_AXES, _FACE_AXES, strict=True):
            face_count = len(first.dimensions[face])
            if (min(face_count, len(second.dimensions[face])) < 3 if native_layout
                    else face_count != len(first.dimensions[center])+1):
                raise ValueError(f'Invalid staggered workspace layout on {center}')
        grid_errors = {}
        if legacy_prefix:
            for file in (first, second):
                if 'dz_array' in file.variables:
                    _read(file, 'dz_array', ('z',), native_layout=True, allow_nonfinite=True,
                          native_dtype=np.float32, allow_short_vector=True)
        for name in axes:
            # Domain's native reader gets the horizontal face sizes from the
            # dimensions without loading their optional coordinate variables.
            if legacy_prefix and name in ('x_face', 'y_face'):
                continue
            coordinates = [
                None if native_layout and name in _FACE_AXES and name not in file.variables
                else _read(file, name, (name,), native_layout=native_layout, allow_nonfinite=legacy_prefix,
                           native_dtype=dtype, allow_short_vector=legacy_prefix).astype(np.float64)
                for file in (first, second)
            ]
            left, right = coordinates
            if left is None or right is None:
                continue
            if legacy_prefix and left.shape != right.shape:
                continue
            grid_errors[name] = float(np.max(np.abs(left-right)))
            if not by_index and not np.allclose(left, right, rtol=0, atol=1e-6):
                raise ValueError(f'Wind workspace coordinates differ on {name}; use --by-index to pair stored entries')
        first_dates, second_dates = (netcdf_dates(file, native_timestamps=native_layout) if count else ()
                                    for file, count in ((first, record_count), (second, candidate_count)))
        if len(first_dates) != record_count or len(second_dates) != candidate_count:
            raise ValueError('Wind workspace timestamps do not cover all records')
        if not by_index and first_dates != second_dates:
            raise ValueError('Wind workspace timestamps differ; use --by-index to pair stored records')
        terrain = []
        for file in (first, second):
            if 'terrain' in file.variables:
                ground = _read(file, 'terrain', ('y', 'x'), native_layout=native_layout,
                               allow_nonfinite=legacy_prefix, native_dtype=dtype).astype(dtype, copy=False)
            else:
                ground_shape = (tuple(len(file.dimensions[name])-1 for name in ('y_face', 'x_face'))
                                if native_layout else tuple(len(file.dimensions[name]) for name in ('y', 'x')))
                ground = np.zeros(ground_shape, dtype=dtype)
            terrain.append(ground)
        if legacy_prefix:
            terrain = _legacy_prefix(*terrain, 'terrain')
        terrain_error = _difference(*terrain)
        records = []
        for index, (left_date, right_date) in enumerate(zip(first_dates, second_dates[:record_count], strict=True)):
            if legacy_prefix:
                for file in (first, second):
                    # The native record loader reads the full stencil when e is present.
                    if 'e' in file.variables:
                        for name in 'efghmn':
                            _read(file, name, ('t', *_CELL_AXES), index, native_layout=True,
                                  allow_nonfinite=True, native_dtype=np.float32)
            first_fields, second_fields = ({name: _read(file, name, ('t', *_FACE_AXES), index,
                                                       native_layout=native_layout, allow_nonfinite=legacy_prefix,
                                                       native_dtype=dtype)
                                             for name in 'uvw'}
                                           for file in (first, second))
            if include_speed:
                comparison = compare_velocity(first_fields, second_fields, precision=precision)
            else:
                fields = {}
                for name in 'uvw':
                    left = first_fields[name].astype(dtype, copy=False)
                    right = second_fields[name].astype(dtype, copy=False)
                    if legacy_prefix:
                        left, right = _legacy_prefix(left, right, name)
                    fields[name] = _difference(left, right)
                comparison = dict(fields=fields)
            left, right = (_read(file, 'icellflag', ('t', *_CELL_AXES), index, native_layout=native_layout,
                                 native_dtype=np.int32)
                           for file in (first, second))
            if legacy_prefix:
                left, right = _legacy_prefix(left, right, 'icellflag')
            comparison['fields']['icellflag'] = _abs_stats(np.abs(left.astype(np.int64)-right.astype(np.int64)).astype(dtype))
            records.append(dict(index=index, reference_timestamp=left_date.isoformat(),
                                candidate_timestamp=right_date.isoformat(), **comparison))
    return dict(precision=precision, by_index=by_index, coordinate_max_abs_m=grid_errors,
                terrain=terrain_error, records=records)


def compare_solvers(path, *, backends=('cpu-sor',), precision='native'):
    """Run solvers from the same initialized wind and pressure stencil."""
    from .numerics.poisson import RedBlackSOR, make_poisson
    from .wind_case import read_wind_case
    from .wind_projection import project

    case = read_wind_case(path, compute_turbulence=False)
    solvers = {name: RedBlackSOR(case.poisson.policy) if name == 'cpu-sor' else make_poisson(name, case.poisson.policy)
               for name in backends}
    records = []
    for step in case.iter_steps():
        workspace = step.initialize()
        reference, _, report = project(workspace, step.poisson)
        if step.wall_log is not None:
            reference = step.wall_log(workspace, reference)
        candidates = {}
        for name, solver in solvers.items():
            wind, _, candidate_report = project(workspace, solver)
            if step.wall_log is not None:
                wind = step.wall_log(workspace, wind)
            candidates[name] = dict(solver=asdict(candidate_report), **compare_velocity(reference, wind, precision=precision))
        records.append(dict(timestamp=step.timestamp, reference_solver=asdict(report), candidates=candidates))
    return dict(precision=precision, reference_backend='cpu', records=records)


def compare_legacy_solvers(path, *, backends, precision='native'):
    """Run the historical first-record CPU and CUDA comparison."""
    from .wind_case import read_wind_case

    def solve(backend, *, cpu):
        case = read_wind_case(path, compute_turbulence=False, poisson_backend=backend,
                              native_sensor_math=cpu, native_building_math=cpu,
                              meteorology_backend='cpu' if cpu else 'cuda',
                              legacy_turbulence_xml=True, wrf_startup_only=True)
        case = replace(case, wall_log=None, native_staircase_metrics=True)
        step = next(case.iter_steps())
        result = step.run()
        return step.timestamp, result['workspace'].arrays, asdict(result['solver'])

    reference_backend = 'cpu-historical-sor'
    timestamp, reference, report = solve(reference_backend, cpu=True)
    candidates = {}
    for backend in backends:
        _, wind, candidate_report = solve(backend, cpu=False)
        candidates[backend] = dict(solver=candidate_report, **compare_velocity(reference, wind, precision=precision))
    return dict(precision=precision, reference_backend=reference_backend,
                records=[dict(timestamp=timestamp, reference_solver=report, candidates=candidates)])
