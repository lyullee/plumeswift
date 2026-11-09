"""Portable QES field and workspace NetCDF I/O."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import errno
from math import prod
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import numpy as np
from .numerics.poisson import Stencil
from .timing import milliseconds, parse_timestamp

@dataclass
class WindWorkspace:
    arrays: dict
    coordinates: dict
    building_context: object | None = None
    row_canopy_flows: tuple = ()
    terrain_mesh: object | None = None
    def stencil(self, *, eliminate_solid=False):
        a=self.arrays
        # Native CPU/CUDA SOR update all internal multiplier cells, including
        # solid cells. Fluid-only elimination is equivalent only when no later
        # empirical flag changes reconnect those cells (terrain canyons do).
        active=~np.isin(a['icellflag'],[0,2]) if eliminate_solid else np.ones(a['icellflag'].shape,dtype=bool)
        return Stencil(*(a[k].astype(float)for k in ['e','f','g','h','m','n']),active)

@contextmanager
def _ascii_netcdf_link(path, *, temporary_parent=None):
    """Temporarily address the same NetCDF file without a Unicode pathname."""
    temporary_parent = Path(gettempdir()) if temporary_parent is None else Path(temporary_parent)
    for parent in dict.fromkeys((temporary_parent, *Path(path).absolute().parents)):
        if not str(parent).isascii():
            continue
        try:
            directory = TemporaryDirectory(prefix='qes-netcdf-', dir=parent)
        except PermissionError:
            continue
        with directory as temporary:
            alias = Path(temporary)/'data.nc'
            try:
                os.link(path, alias)
            except OSError as exc:
                if isinstance(exc, PermissionError) or exc.errno == errno.EXDEV or getattr(exc, 'winerror', None) == 17:
                    continue
                raise
            yield alias
            return
    yield None


@contextmanager
def _open_netcdf(path, *, raw=False):
    from netCDF4 import Dataset
    filename = str(path)
    if os.name == 'nt' and not filename.isascii():
        try:
            relative = os.path.relpath(filename)
        except ValueError:
            relative = filename  # The file may be on another drive.
        if relative.isascii():
            filename = relative
    with ExitStack() as stack:
        try:
            nc = Dataset(filename)
        except OSError:
            # Keep large inputs disk-backed if Windows NetCDF cannot open their path.
            alias = stack.enter_context(_ascii_netcdf_link(path)) if os.name == 'nt' and not filename.isascii() else None
            nc = Dataset(alias) if alias else Dataset('workspace', memory=Path(path).read_bytes())
        stack.enter_context(nc)
        if raw:
            nc.set_auto_maskandscale(False)
            nc.set_auto_chartostring(False)
        yield nc


def workspace_field_timed(variable, spatial_shape=None):
    return (variable.dimensions[:1] == ('t',)
            or (spatial_shape is not None and variable.ndim == len(spatial_shape)+1))


def workspace_record(nc, step, *, fields=None, positional_fields=None, native_input=False):
    from .netcdf_legacy import _native_numeric_cast

    step = range(len(nc.dimensions['t']))[step]
    positional_fields = {} if positional_fields is None else positional_fields
    stored_coords = {}
    for key in ('x', 'y', 'z', 'x_face', 'y_face', 'z_face'):
        if key.endswith('_face') and key not in nc.variables:
            continue
        values = nc[key][:]
        if native_input:
            values = _native_numeric_cast(np.asarray(values), np.float32)
        if positional_fields:
            is_face = key.endswith('_face')
            axis = key if is_face else key+'_face'
            expected = len(nc.dimensions[axis])-int(not is_face)
            if values.size != expected:
                raise ValueError(f'Invalid workspace coordinate vector {key}')
            values = values.reshape(-1)
        stored_coords[key] = values
    coords = {key: np.array(values, dtype=float) for key, values in stored_coords.items()}
    for axis in ('x', 'y'):
        face = axis+'_face'
        if face in coords:
            continue
        centers = coords[axis]
        dimension = nc.dimensions.get(face)
        if centers.ndim != 1 or centers.size < 2 or dimension is None or len(dimension) != centers.size+1:
            raise ValueError(f'Cannot reconstruct {face} coordinates from the stored workspace')
        steps = np.diff(centers)
        if np.ma.is_masked(stored_coords[axis]) or not np.isfinite(steps).all() or not np.all(steps > 0):
            raise ValueError(f'Cannot reconstruct {face} coordinates from invalid centers')
        # Legacy files may declare horizontal face dimensions without variables.
        coords[face] = np.r_[centers[0]-steps[0]/2, centers[:-1]+steps/2, centers[-1]+steps[-1]/2]
    arrays={}
    for key,var in nc.variables.items():
        if key in coords or key in ['timestamp','t']:continue
        if fields is not None and key not in fields:
            continue
        spatial_shape = positional_fields.get(key)
        timed = workspace_field_timed(var, spatial_shape)
        if spatial_shape is None:
            values = var[step] if timed else var[:]
        else:
            slab = tuple(slice(length) for length in spatial_shape)
            if var.ndim != len(spatial_shape)+int(timed) or (timed and step >= var.shape[0]):
                raise ValueError(f'Invalid positional layout for workspace field {key}')
            values = var[(step, *slab) if timed else slab]
            if values.shape != spatial_shape:
                raise ValueError(f'Incomplete positional layout for workspace field {key}')
        if positional_fields and key == 'dz_array':
            if values.size != coords['z'].size:
                raise ValueError('Invalid workspace cell width vector')
            values = values.reshape(-1)
        if np.ma.is_masked(values):
            raise ValueError(f'Missing values in workspace field {key}')
        values = np.asarray(values)
        if native_input:
            if key in ('icellflag', 'icellflag_initial'):
                values = _native_numeric_cast(values, np.int32)
            elif key in ('u0', 'v0', 'w0', *'efghmn', 'dz_array', 'terrain', 'z0_u', 'z0_v',
                         'building_volume_frac', 'terrain_volume_frac'):
                values = _native_numeric_cast(values, np.float32)
        arrays[key] = values
    if 'dz_array' not in nc.variables:
        stored_centers = stored_coords['z']
        centers = coords['z']
        if (centers.ndim != 1 or centers.size < 2 or np.ma.is_masked(stored_centers)
                or not np.isfinite(centers).all()):
            raise ValueError('Cannot reconstruct dz_array from invalid vertical centers')
        # Original QES uses the first center spacing for archives without cell widths.
        spacing = float(stored_centers[1] - stored_centers[0])
        expected = centers[0] + spacing*np.arange(centers.size)
        if spacing <= 0 or not np.allclose(centers, expected, rtol=1e-5, atol=1e-6):
            raise ValueError('Cannot reconstruct dz_array from nonuniform vertical centers')
        if 'z_face' in coords:
            faces = coords['z_face']
            expected_faces = expected[0] - spacing/2 + spacing*np.arange(centers.size+1)
            if (faces.shape != expected_faces.shape or np.ma.is_masked(stored_coords['z_face'])
                    or not np.allclose(faces, expected_faces, rtol=1e-5, atol=1e-6)):
                raise ValueError('Stored vertical faces disagree with the inferred cell widths')
        arrays['dz_array'] = np.full(centers.shape, spacing)
    required = ('u0', 'v0', 'w0', 'dz_array', 'icellflag')
    if not native_input or 'e' in nc.variables:
        required += tuple('efghmn')
    for key in required:
        if key not in arrays:raise ValueError(f"Workspace missing {key}")
        if not np.isfinite(arrays[key]).all():raise ValueError(f"Non-finite workspace field {key}")
    if 'z_face' not in coords:
        centers = coords['z']
        widths = arrays['dz_array'].astype(float, copy=False)
        dimension = nc.dimensions.get('z_face')
        if (centers.ndim != 1 or not centers.size or widths.shape != centers.shape
                or dimension is None or len(dimension) != centers.size+1 or np.any(widths <= 0)
                or np.ma.is_masked(stored_coords['z'])):
            raise ValueError('Cannot reconstruct z_face coordinates from the stored workspace')
        faces = centers[0]-widths[0]/2+np.r_[0., np.cumsum(widths)]
        if not np.allclose(faces[:-1]+widths/2, centers, rtol=1e-5, atol=1e-6):
            raise ValueError('Stored vertical centers disagree with the cell widths')
        coords['z_face'] = faces
    return WindWorkspace(arrays,coords)


def read_wind_workspace(path: str|Path, *, step=0)->WindWorkspace:
    with _open_netcdf(path) as nc:
        return workspace_record(nc, step)


def netcdf_dates(dataset, *, native_timestamps=False):
    times = None
    if native_timestamps:
        from .netcdf_legacy import _native_numeric_cast

        # QES reads the numeric clock as float32 before checking for strings.
        raw = dataset['t'][:]
        if np.ma.is_masked(raw):
            raise ValueError('Missing values in the native numeric clock')
        times = _native_numeric_cast(np.asarray(raw), np.float32)
        if 't' in dataset.dimensions:
            record_count = len(dataset.dimensions['t'])
            if times.size > record_count:
                raise ValueError('Numeric wind timestamps must match the native record count')
            # The original MATLAB files include scalar time for a sole record.
            times = times.reshape(-1)
            if times.size < record_count:
                # Native reads into a zero-initialized vector sized to t.
                times = np.pad(times, (0, record_count-times.size))
    if 'timestamp' in dataset.variables:
        variable = dataset['timestamp']
        if variable.ndim == 2 and native_timestamps:
            # Read the same rows and columns as the native positional clock.
            count = len(dataset.dimensions['t']) if 't' in dataset.dimensions else variable.shape[0]
            stamp = variable[:count, :19]
        else:
            stamp = variable[:]
        if np.ma.is_masked(stamp) or stamp.ndim != 2 or stamp.shape[1] != 19:
            raise ValueError('Expected one 19-character timestamp per field record')
        parse = parse_timestamp if native_timestamps else datetime.fromisoformat
        dates = tuple(parse(row.tobytes().decode('ascii')) for row in stamp)
        # The original standalone reader only uses the 19-character strings.
        # The numeric axis can recover the fractions when its origin and
        # whole seconds agree with those strings.
        time = dataset.variables.get('t')
        origin = getattr(time, 'simulation_start', None)
        if native_timestamps or origin is None:
            return dates
        offsets = time[:]
        if offsets.shape != (len(dates),) or np.ma.is_masked(offsets) or not np.isfinite(offsets).all():
            return dates
        try:
            start = datetime.fromisoformat(origin)
            precise = tuple(start+timedelta(seconds=float(value)) for value in offsets)
        except (ValueError, TypeError, OverflowError):
            return dates
        return precise if tuple(value.replace(microsecond=0) for value in precise) == dates else dates
    if times is None:
        time = dataset['t']
        times = time[:]
        if times.ndim == 0 and 't' in dataset.dimensions and len(dataset.dimensions['t']) == 1:
            times = times.reshape(1)
    if np.ma.is_masked(times) or times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Invalid NetCDF time offsets')
    if not native_timestamps:
        try:
            origin = datetime.fromisoformat(getattr(time, 'simulation_start', None))
        except (TypeError, ValueError):
            origin = None
        if origin is not None and origin.tzinfo is None:
            return tuple(origin+timedelta(seconds=float(value)) for value in times)
    origin = datetime(2022, 1, 1)
    return tuple(origin+timedelta(milliseconds=milliseconds(value)) for value in times)

def write_arrays(path,arrays,coordinates):
    """Self-describing Python result; no upstream code or native executable."""
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays,**{f'coord_{k}':v for k,v in coordinates.items()})

def read_turbulence(path, *, step=0):
    with _open_netcdf(path) as nc:
        step = range(len(nc.dimensions['t']))[step]
        fields={k:np.array(v[step] if v.dimensions[:1] == ('t',) else v[:])for k,v in nc.variables.items()
                if k not in ['t','timestamp','x','y','z']}
    required=['txx','txy','txz','tyy','tyz','tzz','CoEps','div_tau_x','div_tau_y','div_tau_z']
    for key in required:
        if key not in fields or not np.isfinite(fields[key]).all():raise ValueError(f"Missing/non-finite turbulence field {key}: {path}")
    return fields


def read_solver_coefficients(path, cell_shape, *, native_hyperslab=False, prepared_stencil=None):
    """Load record zero in standard cell layout or the legacy padded layout.

    The native reader writes its positional slab into the prepared stencil's flat prefix.
    """
    from .netcdf_legacy import LegacyNetCDFInput, _native_numeric_cast

    fields = {}
    cell_shape = tuple(cell_shape)
    with _open_netcdf(path, raw=native_hyperslab) as nc:
        dimensions = ('z', 'y', 'x')
        if any(axis not in nc.dimensions for axis in dimensions):
            raise ValueError('Coefficient file requires z, y and x dimensions')
        stored_shape = tuple(len(nc.dimensions[axis]) for axis in dimensions)
        if native_hyperslab:
            read_shape = tuple(size-1 for size in stored_shape)
            if min(read_shape) <= 0 or prod(read_shape) > prod(cell_shape):
                raise ValueError(f'Native coefficient slab {read_shape} does not fit cell grid {cell_shape}')
            if 't' not in nc.dimensions or 'icellflag' not in nc.variables:
                raise ValueError('Native coefficient file requires the t dimension and icellflag')
            # Native reload reads the flags as integers before restoring the prepared mask.
            LegacyNetCDFInput(nc).read('icellflag', dtype=np.int32, start=(0, 0, 0, 0),
                                       count=(1, *read_shape))
        else:
            if stored_shape not in (cell_shape, tuple(size+1 for size in cell_shape)):
                raise ValueError(f'Coefficient grid {stored_shape} does not match cell grid {cell_shape}')
            read_shape = cell_shape
        region = tuple(slice(0, size) for size in read_shape)
        for name in ('e', 'f', 'g', 'h', 'm', 'n'):
            if name not in nc.variables:
                raise ValueError(f'Missing solver coefficient {name}')
            variable = nc[name]
            if variable.dimensions == ('t', *dimensions) or (native_hyperslab and variable.ndim == 4):
                if variable.shape[0] == 0:
                    raise ValueError('Coefficient file has no time records')
                data = variable[(0, *region)]
            elif variable.dimensions == dimensions:
                data = variable[region]
            else:
                raise ValueError(f'Invalid coefficient dimension order for {name}: {variable.dimensions}')
            if np.ma.is_masked(data):
                raise ValueError(f'Missing data in solver coefficient {name}')
            values = (_native_numeric_cast(np.asarray(data), np.float32) if native_hyperslab
                      else np.asarray(data, dtype=float))
            if values.shape != read_shape:
                raise ValueError(f'Incomplete solver coefficient hyperslab in {name}')
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'Invalid solver coefficient values in {name}')
            if read_shape != cell_shape and values.size == prod(cell_shape):
                values = values.reshape(cell_shape)
            elif read_shape != cell_shape:
                if prepared_stencil is None:
                    raise ValueError('Partial native coefficient reload requires the prepared stencil')
                initialized = np.array(prepared_stencil[name], dtype=np.float32, copy=True, order='C')
                if initialized.shape != cell_shape:
                    raise ValueError(f'Prepared stencil shape does not match solver coefficient {name}')
                initialized.ravel()[:values.size] = values.ravel()
                values = initialized
            fields[name] = values
    return fields


def write_netcdf(path, arrays, coordinates, *, timestamp='2020-01-01T00:00:00', elapsed=0., static_fields=()):
    """Write one QES field record and preserve its static variable layout."""
    date = datetime.fromisoformat(timestamp)
    origin = (date-timedelta(seconds=elapsed)).isoformat() if date.microsecond and np.isfinite(elapsed) else None
    with netcdf_series(path, coordinates, static_fields=static_fields, time_origin=origin) as append:
        append(arrays, timestamp, elapsed)


@contextmanager
def _staged_netcdf_path(path):
    """Write through an ASCII-safe temporary path and atomically publish."""
    import os
    import shutil
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix='.nc')
    os.close(descriptor)
    try:
        yield temporary
        # os.replace needs its source and destination on the same volume.
        staged_descriptor, staged = tempfile.mkstemp(prefix='.plumeswift-', suffix='.nc', dir=path.parent)
        os.close(staged_descriptor)
        try:
            shutil.copyfile(temporary, staged)
            os.replace(staged, path)
        finally:
            if os.path.exists(staged):
                os.unlink(staged)
    finally:
        os.unlink(temporary)


@contextmanager
def netcdf_series(path, coordinates, *, static_fields=(), varying_fields=(), dimension_names=None,
                  coordinate_names=None, time_origin=None, field_units=None, float_dtype='f8', long_names=None):
    """Append one field record at a time and publish only after successful close."""
    from netCDF4 import Dataset

    dims = {k: len(coordinates[k]) for k in ['x', 'y', 'z']}
    dims.update({k: len(coordinates[k]) for k in ['x_face', 'y_face', 'z_face'] if k in coordinates})
    dimension_names = {} if dimension_names is None else dimension_names
    coordinate_names = {} if coordinate_names is None else coordinate_names
    field_units = {} if field_units is None else field_units
    long_names = {} if long_names is None else long_names
    cell_shape = tuple(dims[k] for k in ['z', 'y', 'x'])
    face_shape = tuple(dims.get(k, -1) for k in ['z_face', 'y_face', 'x_face'])
    shape_dimensions = {
        (): (),
        cell_shape: ('z', 'y', 'x'),
        face_shape: ('z_face', 'y_face', 'x_face'),
        cell_shape[1:]: ('y', 'x'),
        face_shape[1:]: ('y_face', 'x_face'),
    }
    with _staged_netcdf_path(path) as temporary:
        with Dataset(temporary, 'w', format='NETCDF4') as nc:
            nc.createDimension('t', None); nc.createDimension('dateStrLen', 19)
            for name, size in dims.items():
                dimension = dimension_names.get(name, name)
                nc.createDimension(dimension, size)
                coordinate = nc.createVariable(coordinate_names.get(name, name), float_dtype, (dimension,))
                coordinate[:] = coordinates[name]; coordinate.units = 'm'
                if name in long_names:
                    coordinate.long_name = long_names[name]
            time = nc.createVariable('t', 'f8', ('t',))
            time.units = 's'
            if time_origin is not None:
                time.simulation_start = time_origin
            dates = nc.createVariable('timestamp', 'S1', ('t', 'dateStrLen'))
            for key, variable in (('t', time), ('timestamp', dates)):
                if key in long_names:
                    variable.long_name = long_names[key]
                if key in field_units:
                    variable.units = field_units[key]
            nc.plumeswift_precision = np.dtype(float_dtype).name
            nc.plumeswift_scope = 'Field output; feature coverage is recorded in component_status.json'
            fields = {}
            index = 0

            def append(arrays, timestamp, elapsed):
                nonlocal index
                parsed = datetime.fromisoformat(timestamp)
                if (parsed.tzinfo is not None or parsed.isoformat(timespec='seconds') != timestamp[:19]
                        or not np.isfinite(elapsed)):
                    raise ValueError('Expected a QES ISO timestamp and finite elapsed seconds')
                if index and fields.keys() != arrays.keys():
                    raise ValueError('NetCDF record fields differ from the first record')
                record = {}
                for key, raw in arrays.items():
                    value = np.asarray(raw)
                    if not np.isfinite(value).all():
                        raise ValueError(f'Cannot write non-finite field {key}')
                    dimensions = shape_dimensions.get(value.shape)
                    if key == 'dz_array' and value.shape == (dims['z'],):
                        dimensions = ('z',)
                    if dimensions is None:
                        raise ValueError(f'Unsupported NetCDF field shape {key}: {value.shape}')
                    timed = key in varying_fields or (len(dimensions) == 3 and key != 'mixlength' and key not in static_fields)
                    dtype = 'i4' if np.issubdtype(value.dtype, np.integer) else float_dtype
                    dimensions = tuple(dimension_names.get(name, name) for name in dimensions)
                    variable_dims = ('t',)+dimensions if timed else dimensions
                    if index:
                        variable = fields[key]
                        if variable.dimensions != variable_dims or variable.dtype != np.dtype(dtype):
                            raise ValueError(f'NetCDF record schema changed for {key}')
                        if not timed and not np.array_equal(variable[:], value):
                            raise ValueError(f'Static NetCDF field changed: {key}')
                    record[key] = (value, dtype, variable_dims, timed)
                for key, (value, dtype, variable_dims, timed) in record.items():
                    if index == 0:
                        fields[key] = nc.createVariable(key, dtype, variable_dims, zlib=True)
                        if key in field_units:
                            fields[key].units = field_units[key]
                        if key in long_names:
                            fields[key].long_name = long_names[key]
                    variable = fields[key]
                    if timed:
                        variable[index] = value
                    elif index == 0:
                        variable[:] = value
                time[index] = elapsed
                dates[index] = np.frombuffer(parsed.isoformat(timespec='seconds').encode('ascii'), dtype='S1')
                index += 1

            yield append
            if index == 0:
                raise ValueError('No records written to NetCDF series')
