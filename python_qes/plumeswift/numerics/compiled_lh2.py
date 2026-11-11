"""Compiled equilibrium bisection for table-backed cryogenic gas states."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _interpolate(x, grid, values):
    i = np.searchsorted(grid, x)
    if i == 0:
        return values[0]
    if i == len(grid):
        return values[-1]
    slope = (values[i] - values[i - 1]) / (grid[i] - grid[i - 1])
    return values[i - 1] + slope * (x - grid[i - 1])


@njit(cache=True, nogil=True)
def warm_temperatures(concentration, energy, temperatures, volumes, water_t, water_p,
                      ambient_t, pressure, water_loading, ambient_water_enthalpy,
                      gas_r_air, gas_r_water, cp_air, cp_h2, water_triple_t):
    boiling_t = temperatures[0]
    highest_t = temperatures[-1]
    wet = water_loading > 0.
    unsaturated = gas_r_water * water_loading / (gas_r_air + gas_r_water * water_loading)
    result = np.empty_like(concentration)
    for i in range(len(result)):
        c, target = concentration[i], energy[i]
        lo, hi = boiling_t, highest_t
        for _ in range(25):
            t = (lo + hi) / 2
            remaining = 1 - c * _interpolate(t, temperatures, volumes)
            if remaining < 0.:
                hi = t
                continue
            air_energy = cp_air * (t - ambient_t)
            water_volume = 0.
            if wet:
                saturation = _interpolate(t, water_t, water_p)
                water_volume = min(remaining * unsaturated, saturation / pressure)
            dry_mass = (remaining - water_volume) * pressure / (gas_r_air * t)
            if wet:
                water_mass = water_volume * pressure / (gas_r_water * t)
                vapor_loading = water_mass / dry_mass if dry_mass > 0. else 0.
                delta = t - water_triple_t
                vapor_h = 2501000. + 1860. * delta
                condensed_h = -333500. + 2100. * delta if delta < 0. else 4180. * delta
                air_energy += (vapor_loading * vapor_h + (water_loading - vapor_loading) * condensed_h
                               - ambient_water_enthalpy)
            cell_energy = c * cp_h2 * (t - boiling_t) + dry_mass * air_energy
            if cell_energy < target:
                lo = t
            else:
                hi = t
        result[i] = lo
    return result
