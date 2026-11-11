"""Sensible heat transfer from flat ground to the contacting gas cloud."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CloudGroundHeat:
    shape: tuple[int, int, int]
    ground_cell_index: int
    cell_area_m2: float
    ground_temperature_k: float
    heat_transfer_w_m2_k: float

    def __post_init__(self):
        values = (self.cell_area_m2, self.ground_temperature_k, self.heat_transfer_w_m2_k)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Ground heat requires positive area, temperature and heat-transfer coefficient')

    def energy(self, thermo, cells, volume, mass, energy, state, dt):
        """Return heat per cell [J], bounded by equilibrium at ground temperature."""
        level = cells // (self.shape[1] * self.shape[2])
        selected = ((level == self.ground_cell_index) & (mass > 0)
                    & (state.temperature < self.ground_temperature_k))
        heat = np.zeros_like(mass)
        if np.any(selected):
            requested = (self.cell_area_m2 * self.heat_transfer_w_m2_k * dt
                         * (self.ground_temperature_k - state.temperature[selected]))
            limit = thermo.heating_limit(state.concentration[selected], self.ground_temperature_k)
            capacity = np.maximum(0., limit * volume[selected] - energy[selected])
            heat[selected] = np.minimum(requested, capacity)
        return heat
