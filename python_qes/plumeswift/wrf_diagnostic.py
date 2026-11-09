"""Round-trip diagnostic from the upstream WRF coupling test."""
from operator import index

import numpy as np

from .wrf import WRFChecksumError, _field, _fire_wind_fields, _require_fire_mesh_variables, frame_checksum
from .wrf_exchange import (_await, _can_write_record, _input_markers, _legacy_marker, _live_netcdf,
                           _output_variables)


def run_coupling_diagnostic(path, *, rounds=10, timeout=60.):
    """Check raw WRF wind records and acknowledge them with unit UF/VF arrays."""
    rounds = index(rounds)
    if rounds < 1 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('WRF diagnostic needs positive rounds and a finite positive timeout')
    marker_record = None
    expected = None

    def ready(nc):
        nonlocal marker_record
        _require_fire_mesh_variables(nc, ('U0_FMW', 'FRAME0_FMW'), 'coupling diagnostic')
        wind = nc['U0_FMW']
        if wind.ndim != 4:
            raise ValueError('WRF diagnostic requires rank-4 U0_FMW')
        records = wind.shape[0]
        marker = _input_markers(nc, legacy=True)
        if not records:
            return None
        if marker_record is None:
            marker_record = records-1
        if marker_record >= min(records, len(marker)):
            return None
        frame = _legacy_marker(marker, marker_record)
        if frame < 0 or (expected is not None and frame != expected):
            return None
        return frame, records-1

    def respond():
        with _live_netcdf(path, 'r+') as nc:
            received = ready(nc)
            if received is None:
                return None
            frame, record = received
            _require_fire_mesh_variables(nc, ('FXLONG', 'V0_FMW', 'W0_FMW', 'CHSUM0_FMW', 'HT_FMW'),
                                         'coupling diagnostic')
            if nc['FXLONG'].ndim != 3:
                raise ValueError('WRF diagnostic requires rank-3 FXLONG')
            ny, nx = nc['FXLONG'].shape[1:]
            _, winds = _fire_wind_fields(nc, record, nx, ny)
            checksum = frame_checksum(*winds)
            if checksum != int(_field(nc, 'CHSUM0_FMW', 1, record, stored=True)):
                raise WRFChecksumError(f'WRF diagnostic wind checksum mismatch at record {record}')
            u, v, output_checksum, marker = _output_variables(nc, nx, ny)
            if (not all(_can_write_record(field, record) for field in (output_checksum, marker))
                    or frame > np.iinfo(marker.dtype).max):
                raise ValueError('WRF diagnostic output arrays cannot represent the acknowledgement')
            unit_winds = np.ones((ny, nx), dtype=np.float32)
            digest = frame_checksum(unit_winds, unit_winds)
            u[0, :ny, :nx], v[0, :ny, :nx] = unit_winds, unit_winds
            output_checksum[record] = digest
            nc.sync()
            marker[record] = frame
            nc.sync()
            return dict(frame=frame, record=record, input_checksum=checksum, output_checksum=digest)

    reports = []
    for _ in range(rounds):
        label = 'initial diagnostic frame' if expected is None else f'diagnostic frame {expected}'
        report = _await(path, label, respond, timeout, .25)
        reports.append(report)
        expected = report['frame']+1

    def final():
        with _live_netcdf(path, 'r') as nc:
            return ready(nc)

    _await(path, f'diagnostic continuation frame {expected}', final, timeout, .25)
    return dict(frames=reports, continuation_frame=expected)
