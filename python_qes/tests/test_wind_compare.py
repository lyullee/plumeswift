from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.io import netcdf_series
from plumeswift.wind_compare import compare_solvers, compare_velocity, compare_wind_files


ROOT = Path(__file__).resolve().parents[2]


def test_wind_metrics_reproduce_native_precision_and_constant_correlation():
    reference = {'u': [1., 2., 3.], 'v': [0., 0., 0.], 'w': [0., 0., 0.]}
    candidate = reference | {'u': [1.+2e-8, 3., 5.]}
    native = compare_velocity(reference, candidate)
    assert native['fields']['u'] == dict(total_abs=3., max_abs=2., mean_abs=1.)
    assert native['speed']['pearson_r_squared'] == pytest.approx(1.)
    assert compare_velocity(reference, candidate, precision='float64')['speed']['total_abs'] > 3.
    constant = candidate | {'u': [1., 1., 1.]}
    assert compare_velocity(reference, constant)['speed']['pearson_r_squared'] is None


def test_workspace_comparison_requires_alignment_unless_using_stored_indices(tmp_path):
    ws = Domain((4, 4, 4), (1., 1., 1.)).workspace()
    reference, candidate = (tmp_path/f'{name}.nc' for name in ('reference', 'candidate'))
    fields = {name: ws.arrays[name+'0'] for name in 'uvw'} | {'icellflag': ws.arrays['icellflag']}
    altered = fields | {'u': fields['u']+1, 'terrain': np.ones_like(ws.arrays['terrain'])}
    with netcdf_series(reference, ws.coordinates) as write:
        write(fields, '2020-01-01T00:00:00', 0)
        write(fields, '2020-01-01T00:01:00', 60)
    with netcdf_series(candidate, ws.coordinates | {'x_face': ws.coordinates['x_face']+.25}) as write:
        write(altered, '2020-01-01T00:00:00', 0)
        write(altered, '2020-01-01T00:02:00', 120)
    with pytest.raises(ValueError, match='coordinates differ on x_face'):
        compare_wind_files(reference, candidate)
    result = compare_wind_files(reference, candidate, by_index=True)
    assert result['coordinate_max_abs_m']['x_face'] == .25
    assert result['terrain']['mean_abs'] == 1.
    assert result['records'][1]['candidate_timestamp'].endswith('00:02:00')
    for record in result['records']:
        assert record['fields']['u']['mean_abs'] == 1.
        assert record['fields']['icellflag']['max_abs'] == 0.


def test_legacy_comparison_uses_declared_extents_for_padded_fields(tmp_path):
    from netCDF4 import Dataset

    reference, candidate = (tmp_path/f'{name}.nc' for name in ('reference', 'candidate'))
    for path, padded in ((reference, False), (candidate, True)):
        with Dataset(path, 'w') as nc:
            nc.createDimension('t', 1)
            nc.createVariable('t', 'f4', ('t',))[:] = 0.
            if padded:
                nc.createDimension('column', 1)
            for axis in 'xyz':
                centers = axis+'_centers' if padded else axis
                nc.createDimension(centers, 2)
                nc.createDimension(axis+'_face', 3)
                coordinate_axes = (centers, 'column') if padded else (centers,)
                coordinate = np.array([.5, 1.5])
                nc.createVariable(axis, 'f4', coordinate_axes)[:] = coordinate[:, None] if padded else coordinate
            nc.createDimension('extra', 4)
            cells = ('extra',)*3 if padded else ('z', 'y', 'x')
            faces = ('extra',)*3 if padded else ('z_face', 'y_face', 'x_face')
            nc.createVariable('icellflag', 'i4', ('t', *cells))[:, :2, :2, :2] = 1
            for name in 'uvw':
                nc.createVariable(name, 'f4', ('t', *faces))[:, :3, :3, :3] = int(padded and name == 'u')
    with pytest.raises(ValueError, match='Wind workspace dimensions differ or are empty'):
        compare_wind_files(reference, candidate)
    record, = compare_wind_files(reference, candidate, native_layout=True)['records']
    assert record['fields']['u'] == dict(total_abs=27., max_abs=1., mean_abs=1.)
    assert record['fields']['icellflag']['max_abs'] == 0.
    with Dataset(candidate, 'r+') as nc:
        nc['u'][0, 1, 1, 1] = np.nan
    with pytest.raises(ValueError, match='nonfinite comparison values in u'):
        compare_wind_files(reference, candidate, native_layout=True)
    record, = compare_wind_files(reference, candidate, native_layout=True, include_speed=False)['records']
    u = record['fields']['u']
    assert np.isnan(u['total_abs']) and np.isnan(u['mean_abs']) and u['max_abs'] == 1.
    for path in (reference, candidate):
        with Dataset(path, 'r+') as nc:
            nc.createVariable('x_face', 'f4', ('x_face',))[:] = np.nan
            nc.createVariable('y_face', 'f4', ('extra',))[:] = 0.
    result = compare_wind_files(reference, candidate, native_layout=True, include_speed=False)
    assert 'x_face' not in result['coordinate_max_abs_m'] and 'y_face' not in result['coordinate_max_abs_m']
    assert result['records'][0]['fields']['u']['max_abs'] == 1.


def test_legacy_comparison_pairs_different_grids_by_flat_prefix(tmp_path):
    paths = [tmp_path/f'{name}.nc' for name in ('reference', 'candidate')]
    for nx, path in zip((4, 5), paths, strict=True):
        ws = Domain((nx, 4, 4), (1., 1., 1.)).workspace()
        u = ws.arrays['u0']
        if nx == 5:
            u[:] = np.arange(u.size).reshape(u.shape)
        with netcdf_series(path, ws.coordinates) as write:
            fields = {'u': u, 'v': ws.arrays['v0'], 'w': ws.arrays['w0'],
                      'icellflag': ws.arrays['icellflag']}
            write(fields, '2020-01-01T00:00:00', 0)
        if nx == 4:
            reference_size = u.size
    result = compare_wind_files(*paths, native_layout=True, include_speed=False)
    assert result['records'][0]['fields']['u']['total_abs'] == sum(range(reference_size))
    with pytest.raises(ValueError, match='Candidate is shorter than the reference for terrain'):
        compare_wind_files(*reversed(paths), native_layout=True, include_speed=False)
    with pytest.raises(ValueError, match='dimensions differ'):
        compare_wind_files(*paths)


def test_solver_comparison_uses_initialized_wind_and_reports_both_backends(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '12 12 12'
    config = tmp_path/'winds.xml'
    tree.write(config)
    result = compare_solvers(config, precision='float64')
    record, = result['records']
    assert record['reference_solver']['converged']
    candidate = record['candidates']['cpu-sor']
    assert candidate['solver']['converged']
    assert candidate['solver']['method'] == 'numpy.red-black-SOR'
    assert candidate['speed']['max_abs'] < 1e-6


def test_legacy_comparison_rejects_out_of_range_numeric_fields(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path/'workspace.nc'
    with Dataset(path, 'w') as nc:
        nc.createDimension('t', 1)
        nc.createVariable('t', 'f4', ('t',))[:] = 0.
        for axis in 'xyz':
            nc.createDimension(axis, 2)
            nc.createDimension(axis+'_face', 3)
            nc.createVariable(axis, 'f4', (axis,))[:] = [.5, 1.5]
        for axis in 'uvw':
            nc.createVariable(axis, 'f8', ('t', 'z_face', 'y_face', 'x_face'))[:] = 0.
        nc.createVariable('icellflag', 'i8', ('t', 'z', 'y', 'x'))[:] = 1
    for name, value in (('u', 1e100), ('icellflag', 2**31)):
        with Dataset(path, 'r+') as nc:
            nc[name][0, 0, 0, 0] = value
        with pytest.raises(OverflowError, match='numeric conversion not representable'):
            compare_wind_files(path, path, native_layout=True, include_speed=False)
        with Dataset(path, 'r+') as nc:
            nc[name][0, 0, 0, 0] = 0


def test_legacy_comparison_reads_optional_vertical_grid_and_stencil(tmp_path):
    from netCDF4 import Dataset

    ws = Domain((4, 4, 4), (1., 1., 1.)).workspace()
    reference, candidate = (tmp_path/f'{name}.nc' for name in ('reference', 'candidate'))
    axes = ws.coordinates
    fields = {name: ws.arrays[name+'0'] for name in 'uvw'} | {'icellflag': ws.arrays['icellflag']}
    for path in (reference, candidate):
        with netcdf_series(path, axes) as write:
            write(fields, '2020-01-01T00:00:00', 0)
    with Dataset(reference, 'a') as nc:
        nc.renameVariable('z_face', 'old_z_face')
    options = dict(native_layout=True, include_speed=False)
    with Dataset(candidate, 'a') as nc:
        nc.createVariable('dz_array', 'f8', ('z',))[:] = 1e100
    with pytest.raises(OverflowError, match='numeric conversion not representable'):
        compare_wind_files(reference, candidate, **options)
    with Dataset(candidate, 'a') as nc:
        nc['dz_array'][:] = 1.
        nc['z_face'][:] = 1e100
    with pytest.raises(OverflowError, match='numeric conversion not representable'):
        compare_wind_files(reference, candidate, **options)
    with Dataset(candidate, 'a') as nc:
        nc['z_face'][:] = axes['z_face']
        nc['z_face'][-1] = np.nan
        for name in 'efghmn':
            nc.createVariable(name, 'f8', ('t', 'z', 'y', 'x'))[:] = 1.
        nc['n'][0, 0, 0, 0] = 1e100
    with pytest.raises(OverflowError, match='numeric conversion not representable'):
        compare_wind_files(reference, candidate, **options)
    with Dataset(candidate, 'a') as nc:
        nc['n'][0, 0, 0, 0] = 1.
    assert compare_wind_files(reference, candidate, **options)['records'][0]['fields']['u']['total_abs'] == 0.


def test_legacy_comparison_zero_fills_unstored_coordinate_tail(tmp_path):
    from netCDF4 import Dataset

    ws = Domain((4, 4, 4), (1., 1., 1.)).workspace()
    paths = [tmp_path/f'{name}.nc' for name in ('reference', 'candidate')]
    fields = {name: ws.arrays[name+'0'] for name in 'uvw'} | {'icellflag': ws.arrays['icellflag']}
    for path in paths:
        with netcdf_series(path, ws.coordinates) as write:
            write(fields, '2020-01-01T00:00:00', 0)
    with Dataset(paths[1], 'a') as nc:
        nc.renameVariable('x', 'stored_x')
        nc.createDimension('short_x', 3)
        nc.createVariable('x', 'f4', ('short_x',))[:] = ws.coordinates['x'][:3]
    result = compare_wind_files(*paths, native_layout=True, include_speed=False)
    assert result['coordinate_max_abs_m']['x'] == ws.coordinates['x'][-1]
    assert result['records'][0]['fields']['u']['total_abs'] == 0.
    with pytest.raises(ValueError, match='must match the stored shape'):
        compare_wind_files(*paths, native_layout=True)


def test_legacy_solver_cli_renders_ordered_tables_and_preserves_nonconvergence_notice(monkeypatch, capsys):
    from plumeswift import wind_compare
    from plumeswift.__main__ import main

    candidate = dict(solver={'converged': False},
                     fields={axis: dict(max_abs=.25, mean_abs=.125) for axis in 'uvw'},
                     speed=dict(max_abs=.5, mean_abs=.25, pearson_r_squared=None))

    def compare(path, *, backends, precision):
        assert backends == ('cuda-dynamic-legacy', 'cuda-global-memory', 'cuda-shared-memory')
        assert precision == 'native'
        return {'records': [
            dict(timestamp=stamp, candidates=dict.fromkeys(backends, candidate))
            for stamp in ('2020-01-01T00:00:00', '2020-01-01T00:00:30')
        ]}

    monkeypatch.setattr(wind_compare, 'compare_legacy_solvers', compare)
    monkeypatch.setattr(sys, 'argv', ['plumeswift', 'compare-solvers', '-q', 'winds.xml', '-s', '1',
                                     '--format', 'legacy-table'])
    main()
    captured = capsys.readouterr()
    assert captured.out.count('Table 3 of 3\n') == 2
    assert captured.out.index('|Dynamic') < captured.out.index('|Global') < captured.out.index('|Shared')
    assert '|nan      |' in captured.out
    assert captured.out.count('Running time step: ') == 2
    assert 'did not converge' not in captured.out
    assert captured.err.count('did not converge') == 6
