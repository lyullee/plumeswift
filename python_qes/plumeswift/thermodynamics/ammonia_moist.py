"""Gas NH3 with equilibrium ambient water, conserving cell mass and enthalpy.

This deliberately separates water condensation from NH3 dissolution. NH3 is
all gaseous; liquid-water NH3 activity and heat of solution are NOT represented.
The closure is suitable for testing the missing moisture contribution, rather
than claiming a complete ammonia-water solution EOS. Condensate volume is
neglected, but its mass, sensible energy, vaporization and fusion are retained.
Water saturation follows Buck (liquid) and IAPWS R14-08 (ice), as in moisture.py.
No parameters are fitted to dispersion concentrations.
"""
from types import SimpleNamespace

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function

R = 8.314462618
RN, RA, RW = R/.01703052, R/.02897, R/.01801528
CPN, CPA = 2100., 1005.
TF = 273.16


@njit(cache=True)
def _saturation(t):
    if t < TF:
        theta = max(t, 50.)/TF
        terms = (-21.2144006*theta**.00333333333
                 + 27.3203819*theta**1.20666667-6.10598130*theta**1.70333333)
        return 611.657*np.exp(terms/theta)
    delta = t-273.15
    return 611.21*np.exp((18.678-delta/234.5)*delta/(257.14+delta))


@njit(cache=True)
def _at_temperature(c, t, ta, pressure, loading, liquid=False):
    remaining_pressure = max(0., pressure-c*RN*t)
    dry = remaining_pressure/(t*(RA+loading*RW))
    vapor = loading*dry
    saturated = _saturation(t)/(RW*t)
    if vapor > saturated:
        vapor = saturated
        dry = max(0., remaining_pressure-saturated*RW*t)/(RA*t)
    condensed = max(0., loading*dry-vapor)
    delta = t-TF
    hv = 2501000.+1860.*delta
    hc = 4180.*delta if liquid or t >= TF else -333500.+2100.*delta
    energy = (c*CPN*(t-ta)+dry*CPA*(t-ta)+vapor*hv+condensed*hc
              - loading*dry*(2501000.+1860.*(ta-TF)))
    return energy, dry, vapor, condensed


@njit(cache=True)
def _solve(c, energy, ta, pressure, loading):
    temperature = np.empty(c.size)
    density = np.empty(c.size)
    fraction = np.empty(c.size)
    condensate = np.empty(c.size)
    residual = np.empty(c.size)
    for i in range(c.size):
        lo = 160.
        hi = min(400., pressure/(max(c[i], 1e-30)*RN)*(1.-1e-12))
        minimum = _at_temperature(c[i], lo, ta, pressure, loading)[0]
        maximum = _at_temperature(c[i], hi, ta, pressure, loading)[0]
        if hi < lo or energy[i] < minimum-1e-7 or energy[i] > maximum+1e-7:
            raise ValueError('Moist NH3 state outside 160--400 K gas closure')
        # At freezing, finite fusion energy is carried by a phase fraction.
        ei = _at_temperature(c[i], TF-1e-9, ta, pressure, loading)[0]
        el = _at_temperature(c[i], TF, ta, pressure, loading, True)[0]
        coexist = ei <= energy[i] <= el and el > ei+1e-8
        if coexist:
            t = TF
        else:
            for _ in range(44):
                mid = .5*(lo+hi)
                if _at_temperature(c[i], mid, ta, pressure, loading)[0] < energy[i]:
                    lo = mid
                else:
                    hi = mid
            t = .5*(lo+hi)
        computed, dry, vapor, condensed = _at_temperature(c[i], t, ta, pressure, loading)
        temperature[i] = t
        density[i] = c[i]+dry*(1.+loading)
        fraction[i] = c[i]*RN*t/pressure
        condensate[i] = condensed
        residual[i] = 0. if coexist else computed-energy[i]
    return temperature, density, fraction, condensate, residual


class MoistAmmoniaVapour:
    """Inputs: NH3 kg/m³ and excess enthalpy J/m³ relative to ambient gas.

    Ambient entrainment carries a fixed water/dry-air mass ratio from measured
    RH (%). Pressure is Pa, temperatures K. evaluate returns gas volume fraction
    (dimensionless), total density kg/m³, condensate kg/m³ and energy residual.
    Water mass is closed locally with inferred dry-air mass; an independently
    transported droplet or water tracer is not provided by this adapter.
    """
    def __init__(self, temperature, pressure, relative_humidity_pct=0.):
        if not 0. <= relative_humidity_pct <= 100. or not 160. < temperature < 400. or pressure <= 0.:
            raise ValueError('Invalid ambient temperature, pressure or humidity')
        self.ambient_temperature = float(temperature)
        self.pressure = float(pressure)
        self.relative_humidity_pct = float(relative_humidity_pct)
        pw = relative_humidity_pct/100.*_saturation(temperature)
        if pw >= pressure:
            raise ValueError('Water partial pressure must be below total pressure')
        self.water_loading = RA/RW*pw/(pressure-pw)
        self.ambient_density = pressure*(1.+self.water_loading)/(
            temperature*(RA+self.water_loading*RW))

    def evaluate(self, concentration, energy):
        c, e = np.broadcast_arrays(np.asarray(concentration, dtype=float), np.asarray(energy, dtype=float))
        if not np.isfinite(c).all() or not np.isfinite(e).all() or np.any(c < 0):
            raise ValueError('NH3 mass must be nonnegative and energy finite')
        state = _solve(c.ravel(), e.ravel(), self.ambient_temperature, self.pressure, self.water_loading)
        return SimpleNamespace(concentration=c, **{key: value.reshape(c.shape) for key, value in zip(
            ('temperature', 'density', 'volume_fraction', 'water_condensate_density', 'energy_residual'), state)})

    def heating_limit(self, concentration, target):
        c = np.asarray(concentration, dtype=float)
        return np.array([_at_temperature(value, target, self.ambient_temperature,
                        self.pressure, self.water_loading)[0] for value in c.ravel()]).reshape(c.shape)
