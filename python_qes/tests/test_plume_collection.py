from types import SimpleNamespace

import numpy as np

from plumeswift.plume_collection import PlumeCollection


def test_native_collection_rounds_bin_inputs_and_accumulates_in_particle_order():
    config = SimpleNamespace(lower=(.35, 0., 0.), upper=(23.75, 1., 1.), boxes=(9, 1, 1))
    positions = np.array([[2.949999991, .5, .5], [3.25, .5, .5], [30., .5, .5]])
    masses = np.array([.123456789, .0000000001, 1.])
    native = PlumeCollection(config, native_math=True)
    default = PlumeCollection(config)
    time_sum = np.float32(0)
    mass_sum = np.float32(0)
    for dt in (.123, .025):
        native.collect(positions, masses, dt)
        default.collect(positions, masses, dt)
        dt = np.float32(dt)
        time_sum += dt
        for mass in masses[:2]:
            mass_sum += np.float32(mass*1000)*dt

    fields = native.take()
    np.testing.assert_array_equal(fields['p_count'].ravel(), [0, 4, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(default.take()['p_count'].ravel(), [2, 2, 0, 0, 0, 0, 0, 0, 0])
    volume = (np.float32(23.75)-np.float32(.35))/np.float32(9)
    assert fields['c'].dtype == np.dtype('f4')
    assert fields['c'][0, 0, 1] == mass_sum/(time_sum*volume)
    assert fields['t_avg'] == time_sum

    native.collect(np.empty((0, 3)), np.empty(0), .125)
    second = native.take()
    assert not second['p_count'].any()
    assert not second['c'].any()
    assert second['t_avg'] == .125
