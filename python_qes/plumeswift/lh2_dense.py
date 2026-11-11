"""Hydrostatic shallow-layer pressure correction on the existing 3D grid."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShallowLayerPressure:
    sampler: object
    solid: np.ndarray
    ground_z: float
    depth_m: float
    minimum_density_excess: float = .02

    def __post_init__(self):
        if (not np.isfinite([self.ground_z, self.depth_m, self.minimum_density_excess]).all()
                or self.depth_m <= 0 or self.minimum_density_excess < 0):
            raise ValueError('Invalid dense layer depth or density threshold')

    def pressure_head(self, z, y, x, density, ambient_density, gravity):
        zf = self.sampler.zf
        thickness = np.maximum(0., np.minimum(zf[z + 1], self.ground_z + self.depth_m)
                               - np.maximum(zf[z], self.ground_z))
        reduced_gravity = gravity * np.maximum(0., density / ambient_density - 1. - self.minimum_density_excess)
        head = np.zeros(self.solid.shape[1:])
        np.add.at(head, (y, x), reduced_gravity * thickness * ~self.solid[z, y, x])
        return head

    def increment(self, positions, flat_ids, cells, inverse, density, ambient_density, gravity, dt):
        """Integrate the excess density above ground and interpolate face forces."""
        nz, ny, nx = self.solid.shape
        z, y, x = np.unravel_index(cells, (nz, ny, nx))
        head = self.pressure_head(z, y, x, density, ambient_density, gravity)
        in_layer = ((positions[:, 2] >= self.ground_z)
                    & (positions[:, 2] < self.ground_z + self.depth_m))
        active = in_layer & (head[y, x][inverse] > 0)
        out = np.zeros((len(positions), 2))
        if not np.any(active):
            return out
        k, j, i = np.unravel_index(flat_ids[active], (nz, ny, nx))
        centre = head[j, i]
        for axis, indices, limit, faces in ((0, i, nx, self.sampler.xf), (1, j, ny, self.sampler.yf)):
            width = faces[indices + 1] - faces[indices]
            fraction = np.clip((positions[active, axis] - faces[indices]) / width, 0., 1.)
            acceleration = np.zeros_like(centre)
            for offset, weight in ((-1, 1 - fraction), (1, fraction)):
                neighbor = np.clip(indices + offset, 0, limit - 1)
                jj, ii = (j, neighbor) if axis == 0 else (neighbor, i)
                blocked = (neighbor == indices) | self.solid[k, jj, ii]
                acceleration += weight * offset * np.where(blocked, 0., centre - head[jj, ii]) / width
            out[active, axis] = acceleration * dt
        speed = np.linalg.norm(out[active], axis=1)
        wave_speed = np.sqrt(centre)
        out[active] *= (wave_speed / np.maximum(wave_speed, speed))[:, None]
        return out


@dataclass(frozen=True)
class DenseLayerExchange(ShallowLayerPressure):
    """Resist mean updrafts in a dense ground layer using its buoyancy head."""

    minimum_density_excess: float = 0.
    support_depth_m: float = 0.
    wake_influence: np.ndarray | None = field(default=None, repr=False, compare=False)
    head: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.support_depth_m) or self.support_depth_m < 0.:
            raise ValueError("Dense ground support depth must be finite and nonnegative")
        if self.wake_influence is not None and (
                self.wake_influence.shape != self.solid.shape[1:]
                or not np.isfinite(self.wake_influence).all()
                or np.any((self.wake_influence < 0.) | (self.wake_influence > 1.))):
            raise ValueError("Wake influence must be a horizontal field between zero and one")
        object.__setattr__(self, 'head', np.zeros(self.solid.shape[1:]))

    def pressure_head(self, z, y, x, density, ambient_density, gravity):
        if self.wake_influence is not None:
            density = density - ambient_density * .02 * (1. - self.wake_influence[y, x])
        self.head[:] = super().pressure_head(z, y, x, density, ambient_density, gravity)
        return self.head

    def ground_weight(self, positions):
        if self.wake_influence is None:
            return 1.
        _, j, i = self.sampler.cell_indices(positions)
        return self.wake_influence[j, i]

    def vertical_wind(self, positions, wind, variance):
        height = positions[:, 2] - self.ground_z
        active = (height >= 0.) & (height < self.depth_m) & (wind > 0.)
        result = wind.copy()
        if active.any():
            _, j, i = self.sampler.cell_indices(positions[active])
            available = wind[active]**2 + np.maximum(variance[active], 0.)
            barrier = self.head[j, i] * (1. - height[active]/self.depth_m)
            energy = available + barrier
            if self.wake_influence is None:
                result[active] *= np.divide(available, energy, out=np.ones_like(energy), where=energy > 0.)
            else:
                fraction = np.divide(barrier, energy, out=np.zeros_like(energy), where=energy > 0.)
                result[active] *= 1. - self.ground_weight(positions[active]) * fraction
        return result

    def vertical_correction(self, positions, wind, slip, variance):
        correction = self.vertical_wind(positions, wind, variance) - wind
        if self.support_depth_m > 0.:
            fraction = np.clip((positions[:, 2] - self.ground_z) / self.support_depth_m, 0., 1.)
            support = fraction**2 * (3. - 2.*fraction)
            # Taper the transported slip only. The momentum state still
            # carries the history of descent into and through the layer.
            correction += (support - 1.) * np.minimum(slip, 0.)
        return correction
