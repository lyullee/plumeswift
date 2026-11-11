"""Observation-driven surface-layer turbulence for flat, open sites."""

from dataclasses import dataclass

import numpy as np

from .turbulence import LocalTurbulence, _fluid


@dataclass(frozen=True, kw_only=True)
class SurfaceLayerTurbulence(LocalTurbulence):
    """Supply MOST velocity variances and dissipation to the existing GLE.

    The lateral velocity standard deviation is a meteorological observation.
    Vertical variance combines the mechanical surface limit and the free
    convection limit from EPA AERMOD formulation, equation 35. Dissipation
    assumes local balance of shear production and buoyancy.
    """

    friction_velocity: float
    crosswind_std: float
    reciprocal_length: float = 0.
    flow_from_deg: float = 270.
    native_quirks: bool = False

    def __post_init__(self):
        super().__post_init__()
        inputs = [self.friction_velocity, self.reciprocal_length,
                  self.crosswind_std, self.flow_from_deg]
        if not np.isfinite(inputs).all() or min(self.friction_velocity, self.crosswind_std) <= 0:
            raise ValueError("Surface-layer turbulence requires positive velocity scales and finite inputs")
        if self.native_math or self.nonlocal_mixing:
            raise ValueError("Surface-layer turbulence is a separate flat-site closure")

    def profiles(self, height):
        z = np.maximum(height, self.terrain_roughness)
        zeta = z * self.reciprocal_length
        ustar = self.friction_velocity
        shear_factor = np.where(zeta >= 0, 1 + 4.7*zeta,
                                (1 - 15*np.minimum(zeta, 0))**(-.25))
        dissipation = ustar**3 / (.4*z) * (shear_factor - zeta)
        vertical_variance = ustar**2 * (1.3**2 + 1.6*(np.maximum(-zeta, 0)/.4)**(2/3))
        return vertical_variance, dissipation

    def build(self, workspace, velocity, *, building_heights=(), **kwargs):
        if len(building_heights) or np.any(workspace.arrays['terrain'] != 0):
            raise ValueError("Surface-layer turbulence requires flat terrain without buildings")
        fields, metadata = super().build(workspace, velocity, building_heights=building_heights, **kwargs)
        fluid = _fluid(workspace)
        height = workspace.coordinates['z']
        vertical, dissipation = self.profiles(height)
        angle = np.deg2rad(270 - self.flow_from_deg)
        along = np.array([np.cos(angle), np.sin(angle), 0.])
        across = np.array([-along[1], along[0], 0.])
        horizontal = (2.5*self.friction_velocity)**2 * np.outer(along, along)
        horizontal += self.crosswind_std**2 * np.outer(across, across)
        vertical_axis = np.array([0., 0., 1.])
        flux = -self.friction_velocity**2 * np.outer(along, vertical_axis)
        horizontal += flux + flux.T
        stress = np.zeros(fluid.shape + (3, 3))
        stress[fluid] = horizontal
        for plane in range(len(height)):
            stress[plane, ..., 2, 2][fluid[plane]] = vertical[plane]
            fields['CoEps'][plane][fluid[plane]] = 5.7*dissipation[plane]
            fields['tke'][plane][fluid[plane]] = .5*(np.trace(horizontal) + vertical[plane])
        fields.update({key: stress[..., i, j] for key, i, j in
                       [('txx', 0, 0), ('txy', 0, 1), ('txz', 0, 2),
                        ('tyy', 1, 1), ('tyz', 1, 2), ('tzz', 2, 2)]})
        divergence = self.stress_divergence(workspace, stress)
        fields.update({f'div_tau_{axis}': divergence[..., i] for i, axis in enumerate('xyz')})
        metadata.update(u_star=self.friction_velocity, closure='surface-layer')
        return fields, metadata
