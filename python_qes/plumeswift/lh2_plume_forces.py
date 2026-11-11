"""Vertical force closure for a bent-over plume with elliptical sections."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlumeSectionForces:
    origin: tuple[float, float, float]
    horizontal_cell_m: float
    vertical_cell_m: float
    downwind: np.ndarray = field(default_factory=lambda: np.array([1., 0.]), init=False)
    active_steps: int = field(default=0, init=False)
    peak_width_m: float = field(default=0., init=False)
    peak_depth_m: float = field(default=0., init=False)
    width_sum_m: float = field(default=0., init=False)
    depth_sum_m: float = field(default=0., init=False)

    def set_wind(self, sampler):
        """Use the resolved horizontal wind at the nozzle to orient sections."""
        wind = sampler.sample_wind(np.asarray([self.origin]))[0, :2]
        speed = np.linalg.norm(wind)
        if speed > 1e-6:
            self.downwind = wind / speed

    def vertical_velocity(self, positions, h2_mass, inverse, cell_mass, density,
                          ambient_density, vertical, original, gravity, drag, dt):
        """Return vertical drift using local buoyancy and section-dependent inertia and drag."""
        x = positions[:, 0] - self.origin[0]
        y = positions[:, 1] - self.origin[1]
        # Leave injection and the ground-impact source to the near-field model.
        start = 2 * self.horizontal_cell_m
        ramp = np.clip((np.hypot(x, y) - start) / self.horizontal_cell_m, 0., 1.)
        active = (ramp > 0) & (h2_mass > 0)
        if not np.any(active):
            return original

        u, v = self.downwind
        s = x[active] * u + y[active] * v
        transverse = -x[active] * v + y[active] * u
        height = positions[active, 2]
        _, section = np.unique(np.floor(s / self.horizontal_cell_m).astype(np.intp), return_inverse=True)
        weights = h2_mass[active]
        section_mass = np.bincount(section, weights=weights)

        def semiaxis(coordinate, spacing):
            centre = np.bincount(section, weights=weights * coordinate) / section_mass
            variance = np.bincount(section, weights=weights * (coordinate - centre[section])**2) / section_mass
            # A uniform ellipse has standard deviations equal to half its semiaxes.
            return 2 * np.sqrt(np.maximum(variance, spacing**2 / 12))

        half_width = semiaxis(transverse, self.horizontal_cell_m)
        half_depth = semiaxis(height, self.vertical_cell_m)
        self.active_steps += 1
        self.peak_width_m = max(self.peak_width_m, float(2 * half_width.max()))
        self.peak_depth_m = max(self.peak_depth_m, float(2 * half_depth.max()))
        self.width_sum_m += float(2 * np.average(half_width, weights=section_mass))
        self.depth_sum_m += float(2 * np.average(half_depth, weights=section_mass))

        cells = inverse[active]
        count = len(cell_mass)
        blend_weights = weights * ramp[active]
        cell_weight = np.bincount(cells, weights=blend_weights, minlength=count)
        use = cell_weight > 0
        width = np.bincount(cells, weights=blend_weights * half_width[section], minlength=count)[use] / cell_weight[use]
        depth = np.bincount(cells, weights=blend_weights * half_depth[section], minlength=count)[use] / cell_weight[use]

        # Per unit length, the ellipse displaces pi*b*c and has vertical
        # projected width 2*b. Potential-flow added mass is rho_air*pi*b^2.
        effective_density = density[use] + ambient_density * width / depth
        acceleration = gravity * (ambient_density - density[use]) / effective_density
        drag_rate = ambient_density * drag / (np.pi * depth * effective_density)
        forced = vertical[use] + dt * acceleration
        # Solve w_new + dt*k*|w_new|*w_new = forced without cancellation.
        updated = 2 * forced / (1 + np.sqrt(1 + 4 * dt * drag_rate * np.abs(forced)))
        result = original.copy()
        blend = np.minimum(1., cell_weight[use] / cell_mass[use])
        result[use] += blend * (updated - result[use])
        return result

    def diagnostics(self):
        return dict(active_steps=self.active_steps,
                    mean_width_m=self.width_sum_m / max(1, self.active_steps),
                    mean_depth_m=self.depth_sum_m / max(1, self.active_steps),
                    peak_width_m=self.peak_width_m, peak_depth_m=self.peak_depth_m)
