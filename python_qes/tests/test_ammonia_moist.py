"""Conservation and limiting cases, independent of FLADIS concentrations."""
import numpy as np
import pytest

from plumeswift.thermodynamics.ammonia_moist import MoistAmmoniaVapour, _at_temperature, RN, RA, RW
from plumeswift.thermodynamics.moisture import equilibrium_water_pressure


@pytest.mark.parametrize('rh', [0., 53.6, 62., 86.])
def test_recover_prescribed_temperature_and_pressure(rh):
    model = MoistAmmoniaVapour(289.65, 102000., rh)
    c = np.geomspace(1e-6, .12, 40)
    for temperature in (204., 260., 273.15, 273.17, 285., 289.65, 310.):
        raw = [_at_temperature(v, temperature, 289.65, 102000., model.water_loading) for v in c]
        energy, dry, vapor, condensed = np.array(raw).T
        state = model.evaluate(c, energy)
        np.testing.assert_allclose(state.temperature, temperature, atol=2e-8)
        np.testing.assert_allclose(c*RN*temperature+dry*RA*temperature+vapor*RW*temperature,
                                   102000., atol=1e-8)
        np.testing.assert_allclose(vapor+condensed, dry*model.water_loading, atol=1e-14)
        np.testing.assert_allclose(state.density, c+dry+vapor+condensed, atol=1e-12)
        assert np.max(np.abs(state.energy_residual)) < 1e-6
        assert np.all(vapor*RW*temperature <= float(equilibrium_water_pressure(temperature))+1e-6)


def test_ambient_and_dry_gas_limit():
    m = MoistAmmoniaVapour(289.65, 102000., 86.)
    s = m.evaluate(0., 0.)
    assert float(s.temperature) == pytest.approx(289.65)
    assert float(s.density) == pytest.approx(m.ambient_density)
    dry = MoistAmmoniaVapour(289.65, 102000., 0.)
    c = np.geomspace(1e-8, .15, 100)
    e = -1.24e6*c
    k = 102000.*1005./RA
    a = c*(2100.-1005.*RN/RA)
    b = k-a*289.65-e
    expected = 2*k*289.65/(b+np.sqrt(b*b+4*a*k*289.65))
    np.testing.assert_allclose(dry.evaluate(c, e).temperature, expected, atol=1e-8)


def test_ice_liquid_coexistence_and_input_rejection():
    m = MoistAmmoniaVapour(289.65, 102000., 86.)
    c = .02
    ei = _at_temperature(c, 273.16-1e-9, 289.65, 102000., m.water_loading)[0]
    el = _at_temperature(c, 273.16, 289.65, 102000., m.water_loading)[0]
    assert el > ei
    assert float(m.evaluate(c, .5*(ei+el)).temperature) == pytest.approx(273.16)
    with pytest.raises(ValueError):
        m.evaluate(-1., 0.)
    with pytest.raises(ValueError):
        m.evaluate(.001, -1e9)
