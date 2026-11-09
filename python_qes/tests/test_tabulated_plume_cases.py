import json

from netCDF4 import Dataset
import numpy as np
import pytest

from plumeswift.__main__ import main
from plumeswift.io import _open_netcdf
from plumeswift.tabulated_plume_cases import _les_field, run_tabulated_case


@pytest.mark.parametrize('key,staggered', [('txz', None), ('u', 2), ('v', 1), ('w', 0)])
def test_les_preserves_matlab_axis_order_and_staggering(key, staggered, tmp_path):
    z, y, x = np.indices((32, 32, 32))
    original = 100*z+10*y+x
    np.savetxt(tmp_path/f'LES_{key}.txt', original.ravel(), fmt='%d')
    field = _les_field(tmp_path, key, velocity=staggered is not None, staggered_axis=staggered)
    loc = np.array([8., 9., 10.])-.5
    if staggered is not None:
        loc[staggered] -= .5
    assert field[8, 9, 10] == np.dot(loc, [100, 10, 1])
    if staggered is None:
        assert field.shape == (33,)*3
        np.testing.assert_array_equal(field[0], field[1])
        np.testing.assert_array_equal(field[-1], field[-2])
        assert field[8, 9, 0] == 100*7.5+10*8.5+31/2
    else:
        assert field.shape == (34,)*3
        if staggered == 0:
            np.testing.assert_array_equal(field[[0, -1]], 0.)
        else:
            np.testing.assert_array_equal(field[0], -field[1])
            np.testing.assert_array_equal(field[-2], field[-3])


def test_channel_profile_wall_dissipation_and_fractional_timestep(tmp_path):
    profiles = tmp_path/'profiles'
    profiles.mkdir()
    for name, values in {'u': [0., 1., 2.], 'sigma2': [1., 1., 1.], 'epps': [1., 2., 3.]}.items():
        np.savetxt(profiles/f'channel_{name}.txt', values)
    report = run_tabulated_case('channel', profiles, tmp_path/'output', steps=[.00183],
                                duration=.006, particles=24, seed=17)
    with _open_netcdf(report['wind']) as nc:
        u = nc['u'][0]
        np.testing.assert_array_equal(u[0], -u[1])
        np.testing.assert_array_equal(u[-1], 0)
    with _open_netcdf(report['turbulence']) as nc:
        np.testing.assert_array_equal(nc['CoEps'][0, :2], 8.)
    case = report['cases']['0.00183']
    assert case['transport_step_s'] == .00183 and case['clock_step_ms'] == 2
    assert case['plume']['models']['base']['active_particles'] == 24
    assert case['plume']['particle_output_records'] == 1
    with np.load(case['diagnostics']['arrays']) as arrays:
        np.testing.assert_allclose(arrays['relative_probability'].sum(), 25.)


def test_archived_particle_results_use_all_slots_for_probability_and_active_moments(tmp_path, capsys):
    snapshot = tmp_path/'BaileyLES_particleInfo.nc'
    turbulence = tmp_path/'BaileyLES_turbOut.nc'
    with Dataset(snapshot, 'w') as nc:
        nc.createDimension('particle', 4)
        nc.createDimension('t', 2)
        for name, values in {
            'zPos': [[.02, .020], [.03, .021], [.82, .82], [1.2, 1.2]],
            'isActive': [[1, 1], [0, 1], [0, 0], [0, 0]],
            'uFluct': [[2, 2], [4, 4], [90, 90], [90, 90]],
            'wFluct': [[10, 2], [4, 4], [90, 90], [90, 90]],
            'delta_wFluct': [[2, 2], [4, 4], [90, 90], [90, 90]],
        }.items():
            nc.createVariable(name, 'f8', ('particle', 't'))[:] = values
    with Dataset(turbulence, 'w') as nc:
        for name, size in {'t': 1, 'z': 3, 'y': 1, 'x': 1}.items():
            nc.createDimension(name, size)
        nc.createVariable('z', 'f8', ('z',))[:] = [0, .5, 1]
        for name in ('tzz', 'txz', 'CoEps'):
            nc.createVariable(name, 'f8', ('t', 'z', 'y', 'x'))[:] = 1

    for record, expected in [(0, [10, 0, 4, 0]), (-1, [3, 2, 6, 4])]:
        output = tmp_path/f'diagnostics_{record}.npz'
        main(['plume-particle-results', str(snapshot), str(turbulence), '--output', str(output),
              '--step', '.5', '--height', '1', '--record', str(record), '--les'])
        report = json.loads(capsys.readouterr().out)
        assert report['occupied_statistics_bins'] == report['occupied_les_statistics_bins'] == 1
        with np.load(output) as arrays:
            np.testing.assert_array_equal(arrays['relative_probability'][[0, 20]], [12.5, 6.25])
            np.testing.assert_array_equal(arrays['observed'][:, 0], expected)
            np.testing.assert_array_equal(arrays['les_observed'][:, 2], [0, 0, 0] if record == 0 else [2, 2, 4])
