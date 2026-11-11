"""Transient, fixed-footprint LH₂ pool with semi-infinite ground conduction."""

from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np


@dataclass
class PoolRelease:
    """Emit saturated vapor while tracking the liquid inventory and heat use."""

    liquid_rate_kg_s: float
    end_s: float
    radius_m: float
    ground_temperature_k: float
    boiling_temperature_k: float
    latent_heat_j_kg: float
    ground_effusivity: float
    particles_per_second: float
    background_flux_w_m2: float = 0.
    inventory_kg: float = field(default=0., init=False)
    fed_kg: float = field(default=0., init=False)
    evaporated_kg: float = field(default=0., init=False)
    wet_age_s: float = field(default=0., init=False)
    peak_inventory_kg: float = field(default=0., init=False)

    def __post_init__(self):
        positive = [self.liquid_rate_kg_s, self.radius_m, self.ground_temperature_k,
                    self.boiling_temperature_k, self.latent_heat_j_kg, self.particles_per_second]
        nonnegative = [self.ground_effusivity, self.background_flux_w_m2]
        if (not np.isfinite(positive + nonnegative).all() or min(positive) <= 0
                or min(nonnegative) < 0 or not self.end_s > 0):
            raise ValueError('Invalid LH₂ pool or substrate properties')

    def count_and_mass(self, t, dt):
        feed_time = min(dt, max(0., self.end_s - t))
        feed = self.liquid_rate_kg_s * feed_time
        before = self.evaporated_kg
        self.fed_kg += feed
        self._evaporate(feed_time, feed)
        self._evaporate(dt - feed_time, 0.)
        mass = self.evaporated_kg - before
        return (max(1, round(dt * self.particles_per_second)) if mass > 0 else 0), mass

    def _evaporate(self, dt, feed):
        available = self.inventory_kg + feed
        if dt <= 0 or available <= 0:
            return
        age = self.wet_age_s
        delta_t = max(0., self.ground_temperature_k - self.boiling_temperature_k)
        integrated_flux = (2 * self.ground_effusivity * delta_t * dt
                           / (sqrt(pi) * (sqrt(age + dt) + sqrt(age))))
        integrated_flux += self.background_flux_w_m2 * dt
        evaporated = min(available, pi * self.radius_m**2 * integrated_flux / self.latent_heat_j_kg)
        self.inventory_kg = available - evaporated
        self.evaporated_kg += evaporated
        self.peak_inventory_kg = max(self.peak_inventory_kg, self.inventory_kg)
        # Retain the accumulated substrate cooling throughout this one spill.
        self.wet_age_s += dt
