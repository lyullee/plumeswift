"""QES-Winds file handshake with a running WRF/SFIRE peer."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import os
from tempfile import gettempdir
import time

from netCDF4 import Dataset
import numpy as np

from .io import _ascii_netcdf_link
from .netcdf_legacy import _native_numeric_cast
from .wrf import (FireMeshGrid, FireMeshInput, WRFChecksumError, _field, frame_checksum,
                  _FIRE_MESH_VARIABLES, _FIRE_WIND_VARIABLES, _WRFFieldMissingData,
                  _require_fire_mesh_variables)


_EXCHANGE_VARIABLES = ('FRAME0_FMW', 'UF', 'VF', 'FWH', 'CHSUM_FMW', 'FRAME_FMW')


@contextmanager
def _live_netcdf(path, mode):
    """Open a live file by path, with an ASCII alias for Windows NetCDF."""
    path = str(path)
    if os.name != 'nt' or path.isascii():
        with Dataset(path, mode) as nc:
            yield nc
        return
    try:
        relative = os.path.relpath(path)
    except ValueError:
        relative = path
    if relative.isascii():
        with Dataset(relative, mode) as nc:
            yield nc
        return
    # A hard link addresses the same file and file lock. Creating it for
    # each open also allows the peer to replace the file between frames.
    with _ascii_netcdf_link(path, temporary_parent=gettempdir()) as alias:
        if alias is not None:
            with Dataset(alias, mode) as nc:
                yield nc
            return
    raise ValueError('Live WRF requires a writable ASCII alias directory on the same volume as the input file')


def fire_height_winds(workspace, fire_height, halo_cells, dz):
    """Port the face-address arithmetic in WRFInput::extractWind."""
    ny, nx = fire_height.shape
    hx, hy = halo_cells
    arrays, coordinates = workspace.arrays, workspace.coordinates
    z = np.asarray(coordinates['z'], dtype=np.float32)
    j, i = np.indices((ny-1, nx-1))
    i += hx
    j += hy
    height = np.asarray(fire_height[:-1, :-1], dtype=np.float32).astype(float)
    target = arrays['terrain'][j, i].astype(np.float32)+np.where(height < dz, 1.5*dz, height)
    k = np.minimum(np.searchsorted(z[:-1], target.astype(np.float32), side='right'), len(z)-2)
    weight = ((z[k]-target)/dz).astype(np.float32)
    result = []
    for name, stride in (('u', 1), ('v', arrays['u'].shape[2])):
        field = np.asarray(arrays[name], dtype=np.float32)
        address = (k*field.shape[1]+j)*field.shape[2]+i
        flat = field.ravel()
        # Native face offsets are horizontal at the selected vertical level.
        # Use the current face if the shifted address precedes allocation.
        neighbor = np.where(address >= stride, address-stride, address)
        output = np.zeros((ny, nx), dtype=np.float32)
        output[:-1, :-1] = weight*flat[neighbor]+(1-weight)*flat[address]
        result.append(output)
    return tuple(result)


def _can_write_record(field, record):
    return record < field.shape[0] or field.get_dims()[0].isunlimited()


def _output_variables(nc, nx, ny):
    names = ('UF', 'VF', 'CHSUM_FMW', 'FRAME_FMW')
    try:
        u, v, checksum, frame = (nc[name] for name in names)
    except (IndexError, KeyError) as exc:
        raise ValueError(f'WRF exchange requires variables {names}') from exc
    if any(field.ndim != 3 or not _can_write_record(field, 0) or field.shape[1] < ny or field.shape[2] < nx
           or field.dtype.kind != 'f' or field.dtype.itemsize < 4 for field in (u, v)):
        raise ValueError('WRF UF and VF must contain a writable fire-mesh record')
    if any(field.ndim != 1 or not _can_write_record(field, 0) or field.dtype.kind != 'i' or field.dtype.itemsize < 4
           for field in (checksum, frame)):
        raise ValueError('WRF output checksum and frame must contain signed integer records')
    for field in (u, v, checksum, frame):
        field.set_auto_scale(False)
    return u, v, checksum, frame


def _timestamp(nc, record, frame):
    if 'Times' in nc.variables:
        times = nc['Times']
        if times.ndim == 2 and record < times.shape[0]:
            times.set_auto_chartostring(False)
            row = times[record]
            value = np.ma.filled(row, b'').tobytes().decode('ascii').rstrip('\x00 ')
            if value:
                if np.ma.is_masked(row) and ('\x00' in value or len(value) < 19):
                    raise _WRFFieldMissingData(f'WRF Times contains a partial timestamp at record {record}')
                return datetime.strptime(value, '%Y-%m-%d_%H:%M:%S').isoformat(timespec='seconds')
    return (datetime(2022, 1, 1)+timedelta(seconds=frame-1)).isoformat(timespec='seconds')


def _await(path, label, read, timeout, poll_interval):
    deadline = time.monotonic()+timeout
    error = None
    while True:
        try:
            result = read()
            if result is not None:
                return result
            error = None
        except (OSError, WRFChecksumError, _WRFFieldMissingData) as exc:
            # Close and reopen for each attempt so the peer can finish
            # writing and its NetCDF buffers become visible.
            error = exc
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Timed out waiting for WRF {label} in {path}') from error
        time.sleep(min(poll_interval, remaining))


def _fire_height(nc, grid):
    nx, ny, _ = grid.cells
    height = _field(nc, 'FWH', 3, (0, slice(ny), slice(nx)), stored=True)
    if height.shape != (ny, nx):
        raise ValueError('WRF FWH does not cover the fire mesh')
    return height


def _initial_record_pending(nc, name):
    field = nc.variables.get(name)
    return field is not None and field.ndim == 3 and field.get_dims()[0].isunlimited() and field.shape[0] == 0


def _input_markers(nc, *, legacy):
    marks = nc['FRAME0_FMW']
    if legacy:
        # The native typed NetCDF read uses the stored value even when marked missing.
        marks.set_auto_maskandscale(False)
    else:
        marks.set_auto_scale(False)
    kinds = 'iuf' if legacy else 'iu'
    if marks.ndim != 1 or marks.dtype.kind not in kinds:
        expected = 'numeric' if legacy else 'integer'
        raise ValueError(f'WRF FRAME0_FMW must be a one-dimensional {expected} field')
    return marks


def _legacy_marker(marks, record):
    value = np.asarray(marks[record])
    return int(_native_numeric_cast(value, np.int32))


def _peer_cleared_response(nc, response, frame):
    # The PnetCDF producer clears its response slot after publishing each input.
    return response == -1 and np.any(np.ma.filled(_input_markers(nc, legacy=False)[:] == frame, False))


def _read_startup_frame(path, timeout, *, record_time, include_fire_height=False):
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('WRF timeout must be finite and positive')
    marker_record = None

    def read():
        nonlocal marker_record
        with _live_netcdf(path, 'r') as nc:
            _require_fire_mesh_variables(nc, _FIRE_MESH_VARIABLES+_FIRE_WIND_VARIABLES+('FRAME0_FMW',),
                                         'startup frame')
            records = nc['U0_FMW'].shape[0]
            marks = _input_markers(nc, legacy=True)
            if marker_record is None:
                if records == 1 and len(marks) and _legacy_marker(marks, 0) == 1:
                    # The public WRF-SFIRE PnetCDF peer writes its first record in place.
                    marker_record = 0
                elif records >= 10:
                    # The native constructor retains this index while it waits.
                    marker_record = records-10
                else:
                    return None
            if marker_record >= min(records, len(marks)) or _legacy_marker(marks, marker_record) != 1:
                return None
            if _initial_record_pending(nc, 'ZSF') or (include_fire_height and _initial_record_pending(nc, 'FWH')):
                return None
            grid = FireMeshGrid._read_grid(nc, path)
            snapshot = FireMeshInput.read_record(nc, grid, records-1)
            stamp = _timestamp(nc, records-1, 1) if record_time else '2022-01-01T00:00:00'
            height = _fire_height(nc, grid) if include_fire_height else None
            return snapshot, stamp, height

    return _await(path, 'startup frame', read, timeout, .25)


def read_wrf_startup(path, *, timeout=60., record_time=False):
    """Wait for the coupled qes/qesFire startup marker, then load the last wind record."""
    snapshot, stamp, _ = _read_startup_frame(path, timeout, record_time=record_time)
    return (snapshot, stamp) if record_time else snapshot


@dataclass
class WRFExchange:
    grid: FireMeshGrid
    steps: int
    timeout: float = 60.
    poll_interval: float = .25
    legacy_protocol: bool = False
    _startup_fire_height: np.ndarray | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        if (self.steps < int(not self.legacy_protocol) or not np.isfinite((self.timeout, self.poll_interval)).all()
                or min(self.timeout, self.poll_interval) <= 0):
            raise ValueError('Live WRF requires a valid step count and positive timeout and poll interval')

    def describe(self):
        return dict(path=self.grid.path, steps=self.steps, timeout_seconds=self.timeout,
                    poll_seconds=self.poll_interval, legacy_protocol=self.legacy_protocol)

    @classmethod
    def open(cls, path, steps, *, timeout=60., legacy_protocol=False):
        if steps < int(not legacy_protocol):
            raise ValueError('Live WRF requires a positive totalTimeIncrements')
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Live WRF timeout must be finite and positive')
        def read_grid():
            with _live_netcdf(path, 'r') as nc:
                _require_fire_mesh_variables(nc, _FIRE_MESH_VARIABLES+_FIRE_WIND_VARIABLES+_EXCHANGE_VARIABLES,
                                             'live frame exchange')
                if _initial_record_pending(nc, 'ZSF'):
                    return None
                grid = FireMeshGrid._read_grid(nc, path)
                _output_variables(nc, *grid.cells[:2])
                return grid
        grid = _await(path, 'fire mesh', read_grid, timeout, .25)
        return cls(grid, steps, timeout=timeout, legacy_protocol=legacy_protocol)

    def _await(self, label, operation):
        return _await(self.grid.path, label, operation, self.timeout, self.poll_interval)

    def receive(self, frame, *, record_time=True):
        if self.legacy_protocol and frame == 1:
            # Native WRF-only provides one fixed timestamp. Use record
            # times for multi-frame output, including its first frame.
            snapshot, stamp, height = _read_startup_frame(
                self.grid.path, self.timeout, record_time=record_time and self.steps > 1, include_fire_height=True)
            self._startup_fire_height = height
            return snapshot, stamp

        marker_record = None

        def read():
            nonlocal marker_record
            with _live_netcdf(self.grid.path, 'r') as nc:
                marks = _input_markers(nc, legacy=self.legacy_protocol)
                if self.legacy_protocol:
                    record = nc['U0_FMW'].shape[0]-1
                    if record < 0:
                        return None
                    if marker_record is None:
                        # Native updateFromWRF keeps the marker index from
                        # the first poll, then reads the latest wind payload.
                        marker_record = record
                    if marker_record >= len(marks) or _legacy_marker(marks, marker_record) != frame:
                        return None
                else:
                    records = np.flatnonzero(np.ma.filled(marks[:] == frame, False))
                    if not len(records):
                        return None
                    record = int(records[-1])
                snapshot = FireMeshInput.read_record(nc, self.grid, record)
                return snapshot, _timestamp(nc, record, frame) if record_time else None
        return self._await(f'input frame {frame}', read)

    def publish(self, frame, workspace, halo):
        nx, ny, _ = self.grid.cells
        dz = np.float32(self.grid.spacing[2])

        def write():
            with _live_netcdf(self.grid.path, 'r+') as nc:
                u, v, checksum, marker = _output_variables(nc, nx, ny)
                if not self.legacy_protocol:
                    previous = int(np.ma.filled(marker[0], 0)) if len(marker) else 0
                    valid = (-99, 0, 1) if frame == 1 else (frame-1, frame)
                    if previous not in valid and not _peer_cleared_response(nc, previous, frame):
                        raise ValueError(f'WRF output marker {previous} conflicts with frame {frame}')
                    if _initial_record_pending(nc, 'FWH'):
                        return None
                # The legacy constructor reads FWH once after the startup marker.
                height = self._startup_fire_height if self.legacy_protocol else _fire_height(nc, self.grid)
                if height is None:
                    raise RuntimeError('WRF startup frame must be received before publishing winds')
                uf, vf = fire_height_winds(workspace, height, halo.cells(self.grid.spacing), dz)
                if not np.isfinite(uf).all() or not np.isfinite(vf).all():
                    raise FloatingPointError('Cannot publish nonfinite QES wind to WRF')
                digest = frame_checksum(uf, vf)
                u[0, :ny, :nx], v[0, :ny, :nx] = uf, vf
                checksum[0] = digest
                nc.sync()
                marker[0] = frame
                nc.sync()
                return digest
        return self._await(f'output frame {frame}', write)

    def finish(self):
        def write():
            nx, ny, _ = self.grid.cells
            with _live_netcdf(self.grid.path, 'r+') as nc:
                *_, marker = _output_variables(nc, nx, ny)
                if self.steps and not self.legacy_protocol:
                    previous = int(np.ma.filled(marker[0], 0)) if len(marker) else 0
                    if previous not in (self.steps, -99) and not _peer_cleared_response(nc, previous, self.steps+1):
                        raise ValueError(f'WRF completion conflicts with output marker {previous}')
                marker[0] = -99
                nc.sync()
            return True
        self._await('completion acknowledgement', write)
