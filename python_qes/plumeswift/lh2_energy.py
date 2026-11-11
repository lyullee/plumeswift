"""H₂ and ambient moisture equilibrium with a transported enthalpy tracer."""

from dataclasses import dataclass, field
from importlib.util import find_spec

import numpy as np

from .contracts import ModelDomainError, ThermoState
from .thermodynamics.moisture import R_WATER, WATER_TRIPLE_K, equilibrium_water_pressure, water_enthalpy

if find_spec('numba') is not None:
    from .numerics.compiled_lh2 import warm_temperatures
else:
    warm_temperatures = None

R_AIR = 8.314462618 / .028965
R_H2 = 8.314462618 / .002016
CP_H2 = 14300.
CP_AIR = 1005.


@dataclass(frozen=True)
class CryogenicMixture:
    """Specific enthalpy is relative to saturated H₂ vapor at ambient pressure.

    Ambient air enters at zero reference enthalpy. Gas volumes are ideal and
    additive, with incompressible liquid H₂ and constant sensible heat capacities.
    """

    ambient_temperature: float
    pressure: float
    boiling_temperature: float
    latent_heat: float
    liquid_density: float
    vapor_density: float
    vapor_volume_table: tuple[np.ndarray, np.ndarray] | None = None
    relative_humidity_pct: float = 0.
    use_compiled: bool = True
    water_loading: float = field(init=False)
    water_pressure_table: tuple[np.ndarray, np.ndarray] | None = field(init=False, repr=False)

    def __post_init__(self):
        if not 0 <= self.relative_humidity_pct <= 100:
            raise ValueError('Relative humidity must be between 0 and 100 percent')
        pw = self.relative_humidity_pct / 100 * float(equilibrium_water_pressure(self.ambient_temperature))
        if pw >= self.pressure:
            raise ValueError('Ambient water partial pressure must be below total pressure')
        object.__setattr__(self, 'water_loading', R_AIR / R_WATER * pw / (self.pressure - pw))
        temperatures = np.linspace(self.boiling_temperature, self.maximum_temperature, 1024)
        table = (temperatures, equilibrium_water_pressure(temperatures)) if pw else None
        object.__setattr__(self, 'water_pressure_table', table)

    @property
    def maximum_temperature(self):
        return self.ambient_temperature if self.vapor_volume_table is None else self.vapor_volume_table[0][-1]

    def vapor_specific_volume(self, temperature):
        if self.vapor_volume_table is not None:
            return np.interp(temperature, *self.vapor_volume_table)
        # Direct construction uses the ideal-gas limit away from saturation.
        t = np.asarray(temperature)
        tb = self.boiling_temperature
        departure = 1 / self.vapor_density - R_H2 * tb / self.pressure
        return R_H2 * t / self.pressure + departure * (tb / t)**2

    @property
    def ambient_density(self):
        return self.pressure * (1 + self.water_loading) / (
            (R_AIR + R_WATER * self.water_loading) * self.ambient_temperature)

    def _water_energy(self, temperature, vapor_per_dry_air):
        liquid_or_ice = self.water_loading - vapor_per_dry_air
        return (vapor_per_dry_air * water_enthalpy(temperature)
                + liquid_or_ice * water_enthalpy(temperature, condensed=True)
                - self.water_loading * water_enthalpy(self.ambient_temperature))

    def _air_state(self, temperature, gas_volume, h2_volume):
        """Return air/water mass and energy per cell volume, including condensate."""
        remaining = np.maximum(0., gas_volume - h2_volume)
        air_energy = CP_AIR * (temperature - self.ambient_temperature)
        if self.water_pressure_table is None:
            dry_mass = remaining * self.pressure / (R_AIR * temperature)
            return dry_mass, dry_mass * air_energy
        saturation = np.interp(temperature, *self.water_pressure_table)
        unsaturated_fraction = R_WATER * self.water_loading / (R_AIR + R_WATER * self.water_loading)
        water_volume = np.minimum(remaining * unsaturated_fraction, gas_volume * saturation / self.pressure)
        dry_mass = (remaining - water_volume) * self.pressure / (R_AIR * temperature)
        vapor_mass = water_volume * self.pressure / (R_WATER * temperature)
        vapor_loading = np.divide(vapor_mass, dry_mass, out=np.zeros_like(dry_mass), where=dry_mass > 0)
        return dry_mass * (1 + self.water_loading), dry_mass * (air_energy + self._water_energy(temperature, vapor_loading))

    def vapor_enthalpy_density(self, concentration, temperature):
        """Equilibrium cell energy at a prescribed temperature with vapor H₂."""
        h2_volume = concentration * self.vapor_specific_volume(temperature)
        _, air_energy = self._air_state(temperature, 1., h2_volume)
        return concentration * CP_H2 * (temperature - self.boiling_temperature) + air_energy

    def heating_limit(self, concentration, target):
        """Maximum energy density at the target, subject to the cell volume."""
        c = np.asarray(concentration)
        tb = self.boiling_temperature
        if self.vapor_volume_table is not None:
            temperatures, volumes = self.vapor_volume_table
            target = np.minimum(target, np.interp(1 / c, volumes, temperatures))
        q = np.clip((1 / c - 1 / self.liquid_density)
                    / (1 / self.vapor_density - 1 / self.liquid_density), 0., 1.)
        v_liquid = c * (1 - q) / self.liquid_density
        v_h2 = c * q * self.vapor_specific_volume(target)
        _, air_energy = self._air_state(target, 1 - v_liquid, v_h2)
        return c * (q * CP_H2 * (target - tb) - (1 - q) * self.latent_heat) + air_energy

    def _warm_temperatures(self, concentration, energy):
        if (self.use_compiled and warm_temperatures is not None and self.vapor_volume_table is not None
                and self.vapor_volume_table[0][0] == self.boiling_temperature):
            temperatures, volumes = self.vapor_volume_table
            water_t, water_p = self.water_pressure_table or (temperatures[:0], volumes[:0])
            return warm_temperatures(concentration, energy, temperatures, volumes, water_t, water_p,
                                     self.ambient_temperature, self.pressure, self.water_loading,
                                     self.water_loading * float(water_enthalpy(self.ambient_temperature)),
                                     R_AIR, R_WATER, CP_AIR, CP_H2, WATER_TRIPLE_K)

        lo = np.full(concentration.shape, self.boiling_temperature)
        hi = np.full(concentration.shape, self.maximum_temperature)
        for _ in range(25):
            mid = (lo + hi) / 2
            h2_volume = concentration * self.vapor_specific_volume(mid)
            _, air_energy = self._air_state(mid, 1., h2_volume)
            cell_energy = concentration * CP_H2 * (mid - self.boiling_temperature) + air_energy
            below = (cell_energy < energy) & (h2_volume <= 1.)
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return lo

    def evaluate(self, concentration, enthalpy_density):
        c = np.asarray(concentration, dtype=float)
        e = np.asarray(enthalpy_density, dtype=float)
        tb = self.boiling_temperature
        v_liq = 1 / self.liquid_density
        v_vap = 1 / self.vapor_density
        if (np.any(c < 0) or not np.isfinite(c).all() or not np.isfinite(e).all()
                or np.any(c * v_liq > 1.)):
            raise ModelDomainError('Invalid H₂ mass or energy in cryogenic mixture')
        _, air_energy_b = self._air_state(tb, 1., 0.)
        energy_liquid = -c * self.latent_heat + (1 - c * v_liq) * air_energy_b
        energy_vapor = (1 - c * v_vap) * air_energy_b
        phase_slope = energy_vapor - energy_liquid
        vapor = np.divide(e - energy_liquid, phase_slope, out=np.ones_like(c), where=c > 0)
        vapor = np.clip(vapor, 0., 1.)
        vapor_limit = np.divide(1 - c * v_liq, c * (v_vap - v_liq),
                                out=np.ones_like(c), where=c > 0)
        vapor = np.minimum(vapor, vapor_limit)

        warm = (vapor == 1.) & (e > energy_vapor)
        temperature = np.full(c.shape, tb)
        if np.any(warm):
            temperature[warm] = self._warm_temperatures(c[warm], e[warm])
        vapor_volume = c * vapor * self.vapor_specific_volume(temperature)
        liquid_volume = c * (1 - vapor) * v_liq
        gas_volume = 1 - liquid_volume
        air_density, _ = self._air_state(temperature, gas_volume, vapor_volume)
        density = c + air_density
        fraction = np.divide(vapor_volume, gas_volume, out=np.zeros_like(c), where=gas_volume > 0)
        return ThermoState(c, temperature, density, fraction, vapor)

    def jet_state(self, hydrogen_rate, air_rate, specific_enthalpy):
        """Return temperature, H₂ vapor fraction and volume flow at ambient pressure."""
        tb = self.boiling_temperature
        dry_air_rate = air_rate / (1 + self.water_loading)
        air_energy_b = CP_AIR * (tb - self.ambient_temperature) + self._water_energy(tb, 0.)
        heat = hydrogen_rate * specific_enthalpy - dry_air_rate * air_energy_b
        vapor = min(1., max(0., 1. + heat / (hydrogen_rate * self.latent_heat)))
        temperature = tb + max(0., heat) / (hydrogen_rate * CP_H2 + dry_air_rate * CP_AIR)
        water_vapor_rate = 0.
        if self.water_pressure_table is not None and temperature > tb:
            lo, hi = tb, self.ambient_temperature
            for _ in range(25):
                t = (lo + hi) / 2
                ps = min(float(np.interp(t, *self.water_pressure_table)), self.pressure * (1 - 1e-12))
                dry_gas_volume = hydrogen_rate * self.vapor_specific_volume(t) + dry_air_rate * R_AIR * t / self.pressure
                water_vapor_rate = min(dry_air_rate * self.water_loading,
                                       ps * self.pressure * dry_gas_volume / ((self.pressure - ps) * R_WATER * t))
                water_energy = self._water_energy(t, water_vapor_rate / dry_air_rate) if dry_air_rate else 0.
                energy = hydrogen_rate * CP_H2 * (t - tb) + dry_air_rate * (CP_AIR * (t - self.ambient_temperature) + water_energy)
                if energy < hydrogen_rate * specific_enthalpy:
                    lo = t
                else:
                    hi = t
            temperature = t
        gas_volume = hydrogen_rate * vapor * self.vapor_specific_volume(temperature)
        liquid_volume = hydrogen_rate * (1 - vapor) / self.liquid_density
        air_volume = (dry_air_rate * R_AIR + water_vapor_rate * R_WATER) * temperature / self.pressure
        return temperature, vapor, gas_volume + liquid_volume + air_volume
