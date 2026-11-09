"""Stream staggered QES wind and turbulence files for standalone plume."""
from contextlib import contextmanager
from math import prod

import numpy as np

from .io import WindWorkspace, _open_netcdf, netcdf_dates
from .domain import native_vertical_coordinates
from .cutcell_walls import define_cutcell_walls
from .netcdf_legacy import _native_numeric_cast

_STRESSES = ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz')
_DIVERGENCE = ('div_tau_x', 'div_tau_y', 'div_tau_z')
_STRESS_ROWS = (('txx', 'txy', 'txz'), ('txy', 'tyy', 'tyz'), ('txz', 'tyz', 'tzz'))


def _read(dataset, key, dimensions, index=None, *, allow_nonfinite=False, native_shape=None, native_dtype=np.float32):
    variable = dataset[key]
    if native_shape is not None and index is None and len(native_shape) == 1:
        if variable.size != prod(native_shape):
            raise ValueError(f'Field {key} cannot supply native coordinate shape {native_shape}')
        raw = variable[:].reshape(native_shape)
    elif native_shape is not None:
        shape = variable.shape
        prefix = () if index is None else (index,)
        field_shape = shape[len(prefix):]
        if (len(field_shape) != len(native_shape) or (prefix and index >= shape[0])
                or any(actual < needed for actual, needed in zip(field_shape, native_shape, strict=True))):
            raise ValueError(f'Field {key} cannot supply native data with shape {native_shape} at record {index}')
        raw = variable[(*prefix, *(slice(0, size) for size in native_shape))]
    elif index is not None and variable.dimensions == ('t', *dimensions):
        raw = variable[index]
    elif variable.dimensions == dimensions:
        raw = variable[:]
    else:
        raise ValueError(f'Unexpected dimensions for {key}: {variable.dimensions}')
    if np.ma.is_masked(raw) or (not allow_nonfinite and not np.isfinite(raw).all()):
        raise ValueError(f'Missing or non-finite field {key}')
    values = np.asarray(raw)
    if native_shape is None or values.dtype == native_dtype:
        return values
    if native_dtype == np.int32:
        try:
            return _native_numeric_cast(values, np.int32)
        except OverflowError as exc:
            raise ValueError(f'Field {key} exceeds the native integer range') from exc
    with np.errstate(over='raise'):
        return values.astype(native_dtype)


def _read_mixing_distance(dataset, key, cell_axes, index=None):
    distance = _read(dataset, key, cell_axes, index)
    if np.any(distance < 0):
        raise ValueError(f'Negative wall distance in {key}')
    return distance


def _standalone_stress_divergence(fields, spacing, *, vertical_centers=None):
    # qesPlume constructs TURB before loading the first wind record. Its
    # interior cell list therefore includes solids and it has no wall stencil.
    stress = {key: np.asarray(fields[key], dtype=np.float32) for key in _STRESSES}
    dx, dy, dz = np.float32(2) * spacing
    if vertical_centers is not None:
        separation = np.diff(vertical_centers)[:, None, None]
        below, above = separation[:-1], separation[1:]
    for key, (x, y, z) in zip(_DIVERGENCE, _STRESS_ROWS):
        div = np.zeros_like(stress[x])
        low = stress[z][:-2, 1:-1, 1:-1]
        high = stress[z][2:, 1:-1, 1:-1]
        if vertical_centers is None:
            vertical = (high-low)/dz
        else:
            center = stress[z][1:-1, 1:-1, 1:-1]
            vertical = ((below/above)*(high-center)+(above/below)*(center-low))/(below+above)
        div[1:-1, 1:-1, 1:-1] = (
            (stress[x][1:-1, 1:-1, 2:] - stress[x][1:-1, 1:-1, :-2]) / dx
            + (stress[y][1:-1, 2:, 1:-1] - stress[y][1:-1, :-2, 1:-1]) / dy
            + vertical
        )
        fields[key] = div


@contextmanager
def open_plume_fields(winds_path, turbulence_path, *, native_timestamps=False,
                      allow_nonfinite_turbulence=False, native_layout=False, native_arithmetic=False,
                      diagnostics=None, include_tke=False):
    with _open_netcdf(winds_path, raw=native_layout) as wind, _open_netcdf(turbulence_path, raw=native_layout) as turb:
        if diagnostics is not None:
            from .runtime_info import report_warning

            for dataset in (wind, turb):
                if 'timestamp' not in dataset.variables:
                    report_warning('No timestamp in NetCDF file', diagnostics)
        dates = netcdf_dates(wind, native_timestamps=native_timestamps)
        if native_layout and 't' not in turb.variables:
            raise ValueError('Native turbulence archive requires the numeric t variable')
        turbulence_count = len(turb.dimensions['t'])
        # Native qesPlume selects each field slice by wind record number.
        # A field's positional record axis can differ from the nominal t axis.
        if (len(dates) != len(wind.dimensions['t'])
                or (not native_layout and turbulence_count != len(dates))):
            raise ValueError('Wind and turbulence time record counts differ')
        if (not native_layout and 'timestamp' in turb.variables
                and dates != netcdf_dates(turb, native_timestamps=native_timestamps)):
            raise ValueError('Wind and turbulence timestamps differ')
        cell_axes = ('z', 'y', 'x')
        face_axes = tuple(axis+'_face' for axis in cell_axes)
        coordinate_type = np.float32 if native_layout else float

        def coordinate(key, dimension, size=None, *, allow_nonfinite=False):
            shape = (size,) if native_layout else None
            missing_outer_center = (native_layout and key in ('x', 'y') and size >= 3
                                    and wind[key].size == size-1)
            if missing_outer_center:
                # The native coordinate read leaves the final unused center
                # zero when the archived vector only contains physical cells.
                shape = (size-1,)
            values = _read(wind, key, (dimension,), native_shape=shape,
                           allow_nonfinite=allow_nonfinite).astype(coordinate_type, copy=False)
            return np.pad(values, (0, 1)) if missing_outer_center else values

        if native_layout:
            cell_shape = tuple(len(wind.dimensions[axis])-1 for axis in face_axes)
            if tuple(len(turb.dimensions[axis]) for axis in cell_axes) != cell_shape:
                raise ValueError('Wind and turbulence dimensions do not match the native staggered grid')
            coords = {axis: coordinate(axis, axis, size, allow_nonfinite=axis in ('x', 'y'))
                      for axis, size in zip(cell_axes, cell_shape, strict=True)}
        else:
            coords = {axis: coordinate(axis, axis) for axis in cell_axes}
            cell_shape = tuple(len(coords[axis]) for axis in cell_axes)
            for axis in cell_axes:
                if not np.array_equal(coords[axis], _read(turb, axis, (axis,))):
                    raise ValueError(f'Wind and turbulence {axis} coordinates differ')
        minimum_centers = 2 if native_layout else 3
        for name, axis in coords.items():
            defined = axis
            if native_layout and name in ('x', 'y') and len(axis) > 2:
                # Native horizontal interpolation uses spacing from the first
                # two centers, and its upper bound uses the penultimate one.
                defined = axis[:-1]
            if (len(axis) < minimum_centers or np.isinf(axis[-1]) or not np.isfinite(defined).all()
                    or not np.all(np.diff(defined) > 0)):
                raise ValueError('Invalid plume field axes')
        face_shape = tuple(size+1 for size in cell_shape)
        dx, dy = (float(np.diff(coords[axis])[0]) for axis in ('x', 'y'))
        has_widths = 'dz_array' in wind.variables
        if has_widths:
            dz_array = coordinate('dz_array', 'z', cell_shape[0], allow_nonfinite=native_layout)
        else:
            dz_array = np.full(cell_shape[0], np.diff(coords['z'])[0])
        # The original archive loader skips the unused top width when it is
        # NaN: it cannot replace a finite minimum or alter a physical face.
        spacing_widths = dz_array[:-1] if native_layout and np.isnan(dz_array[-1]) else dz_array
        if not np.isfinite(spacing_widths).all() or np.any(spacing_widths <= 0):
            raise ValueError('Nonpositive vertical cell width')
        face_variable = wind.variables.get('z_face')
        if face_variable is not None and (face_variable.size == cell_shape[0] if native_layout
                                          else face_variable.dimensions == ('z',)):
            # The power-law examples store upper faces on the cell axis.
            # Restore the missing lower ghost face. Reading the stored
            # array as complete shifts both physical boundaries by one
            # cell and allows lookups above the last turbulence center.
            faces = coordinate('z_face', 'z', cell_shape[0])
            if faces[0] != 0 or not np.array_equal(faces, coords['z']+dz_array[0]/2):
                raise ValueError('Invalid legacy vertical face coordinates')
            coords['z_face'] = np.insert(faces, 0, faces[0]-dz_array[0])
        elif face_variable is not None:
            coords['z_face'] = coordinate('z_face', 'z_face', face_shape[0], allow_nonfinite=native_layout)
        elif native_layout:
            coords['z_face'] = native_vertical_coordinates(dz_array)[1]
        else:
            coords['z_face'] = np.r_[0., np.cumsum(dz_array)]-dz_array[0]
        faces = coords['z_face']
        # The native upper transport boundary is the penultimate face. A
        # trailing NaN still compares false in the in-domain lower_bound.
        defined_faces = faces[:-1] if native_layout and np.isnan(faces[-1]) else faces
        if (len(faces) != face_shape[0] or not np.isfinite(defined_faces).all()
                or np.any(np.diff(defined_faces) <= 0)):
            raise ValueError('Invalid vertical faces')
        physical_grid = not (native_layout or native_arithmetic)
        if physical_grid and not has_widths:
            dz_array = np.diff(coords['z_face'])
            spacing_widths = dz_array
        vertical_centers = coords['z'] if physical_grid and np.any(dz_array != dz_array[0]) else None
        spacing = dx, dy, float(np.min(spacing_widths))
        has_divergence = _DIVERGENCE[0] in turb.variables
        if has_divergence and not all(key in turb.variables for key in _DIVERGENCE):
            raise ValueError('Incomplete turbulence stress divergence')
        if native_layout or not has_divergence:
            # Native Domain reads the NetCDF coordinates as floats before
            # subtracting, and uses the minimum width on stretched grids.
            native_spacing = np.array([np.diff(coords[axis][:2].astype(np.float32))[0]
                                       for axis in ('x', 'y', 'z')])
            if has_widths:
                native_spacing[2] = np.min(spacing_widths.astype(np.float32))
            if np.any(~np.isfinite(native_spacing) | (native_spacing <= 0)):
                raise ValueError('Invalid plume grid spacing in native precision')
            if native_layout:
                spacing = tuple(map(float, native_spacing))
        if native_layout:
            if 'terrain' in wind.variables:
                terrain = _read(wind, 'terrain', cell_axes[1:], native_shape=cell_shape[1:], allow_nonfinite=True)
            else:
                terrain = np.zeros(cell_shape[1:], dtype=np.float32)
                if diagnostics is not None:
                    print('[WINDS Data] \t no terrain data found -> assumed flat', file=diagnostics)

        # The original standalone loader leaves wall distance zero. The
        # physical-grid path restores the recorded distance, not the
        # turbulence closure's scaled mixing length L.
        distance_source = None
        if physical_grid:
            if 'mixingLengths' in turb.variables:
                distance_source = turb, 'mixingLengths'
            elif 'mixlength' in wind.variables:
                distance_source = wind, 'mixlength'
        if distance_source is None:
            shared_distance = np.zeros(cell_shape)
        else:
            dataset, key = distance_source
            shared_distance = (_read_mixing_distance(dataset, key, cell_axes)
                               if dataset[key].dimensions == cell_axes else None)

        def records():
            previous_coefficients = None
            has_coefficients = native_layout and 'e' in wind.variables
            for index, timestamp in enumerate(dates):
                if diagnostics is not None:
                    stamp = timestamp.isoformat()
                    print(f'[QES-WINDS] \t Loading data at step {index} ({stamp})', file=diagnostics, flush=True)
                # The native file loader keeps unused halo and solid-face values as stored.
                arrays = {key: _read(wind, key, face_axes, index, allow_nonfinite=native_layout,
                                     native_shape=face_shape if native_layout else None)
                          for key in 'uvw'}
                arrays['icellflag'] = _read(wind, 'icellflag', cell_axes, index,
                                            native_shape=cell_shape if native_layout else None, native_dtype=np.int32,
                                            allow_nonfinite=native_layout)
                if arrays['icellflag'].shape != cell_shape or any(arrays[key].shape != face_shape for key in 'uvw'):
                    raise ValueError('Wind arrays do not match the staggered grid')
                workspace = WindWorkspace(arrays, coords)
                if native_layout:
                    arrays.update(terrain=terrain, dz_array=dz_array)
                    if has_coefficients:
                        # The native wall pass applies comparisons to raw
                        # apertures, including an archived IEEE NaN.
                        coefficients = {key: _read(wind, key, cell_axes, index, native_shape=cell_shape,
                                                   allow_nonfinite=True) for key in 'efghmn'}
                    elif previous_coefficients is None:
                        coefficients = {key: np.ones(cell_shape, dtype=np.float32) for key in 'efghmn'}
                    else:
                        # Without a stored stencil, native retains the prior record's modified coefficients.
                        coefficients = {key: values.copy() for key, values in previous_coefficients.items()}
                    if not has_coefficients and diagnostics is not None:
                        print('[WINDS Data] \t no SORcoeff data found -> assumed e,f,g,h,m,n=1', file=diagnostics)
                    arrays.update(coefficients)
                    define_cutcell_walls(workspace)
                    if not has_coefficients:
                        previous_coefficients = coefficients
                if diagnostics is not None:
                    print(f'[QES-TURB]\t Loading data at step {index} ({stamp})', file=diagnostics, flush=True)

                def read_turbulence(key):
                    return _read(turb, key, cell_axes, index, allow_nonfinite=allow_nonfinite_turbulence,
                                 native_shape=cell_shape if native_layout else None)

                fields = {key: read_turbulence(key) for key in (*_STRESSES, 'CoEps')}
                if native_layout or (include_tke and 'tke' in turb.variables):
                    fields['tke'] = read_turbulence('tke')
                if not native_layout and 'nuT' in turb.variables:
                    fields['nuT'] = read_turbulence('nuT')
                if has_divergence:
                    fields.update({key: read_turbulence(key) for key in _DIVERGENCE})
                else:
                    if diagnostics is not None:
                        print('[QES-TURB]\t Computing Divergence of Stess Tensor...', file=diagnostics, flush=True)
                    _standalone_stress_divergence(fields, native_spacing, vertical_centers=vertical_centers)
                fields['mixingLengths'] = (shared_distance if shared_distance is not None else
                                           _read_mixing_distance(*distance_source, cell_axes, index))
                yield workspace, fields, spacing

        yield dates, records()
