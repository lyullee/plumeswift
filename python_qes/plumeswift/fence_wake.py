"""Optional wake closure for grid-aligned solid fences on flat ground."""

from dataclasses import dataclass, field

import numpy as np

from .meteorology import SensorProfile
from .turbulence import LocalTurbulence, _fluid


@dataclass(frozen=True)
class SolidFenceWake:
    """Build a mean recirculation and an elevated, growing shear layer.

    The streamfunction increment is differenced on staggered faces so that
    its divergence cancels before solid-face masking and pressure projection.
    Lengths are given in fence heights. The profiles are an engineering
    parameterization inspired by solid-fence wake experiments.
    """

    inflow: SensorProfile
    normal_axis: str = 'y'
    length_h: float = 10.
    deficit: float = 1.2
    confined_gap: bool = False

    def __post_init__(self):
        if (self.normal_axis not in ('x', 'y') or not np.isfinite([self.length_h, self.deficit]).all()
                or self.length_h <= 0 or self.deficit <= 0):
            raise ValueError("Invalid solid-fence wake parameters")

    def apply(self, workspace, ordered_buildings, context):
        a, c = workspace.arrays, workspace.coordinates
        if np.any(a['terrain'] != 0):
            raise ValueError("SolidFenceWake requires flat ground")
        axis = self.normal_axis
        across = 'x' if axis == 'y' else 'y'
        normal_index = int(axis == 'y')
        theta = np.deg2rad(270 - self.inflow.direction)
        incidence = (np.cos(theta), np.sin(theta))[normal_index]
        if abs(incidence) < 1e-8:
            return
        sign = np.sign(incidence)
        fences = []
        for _, building in ordered_buildings:
            vertices = np.asarray(building.vertices[:-1])
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            is_rectangle = len(vertices) == 4 and np.all((vertices == low) | (vertices == high))
            if (not is_rectangle or building.base_height != 0 or building.holes or building.detached_parts
                    or high[normal_index] - low[normal_index] > building.height):
                raise ValueError("SolidFenceWake requires thin, ground-mounted rectangular fences")
            face = (high if sign > 0 else low)[normal_index]
            fences.append((sign * face, low[1-normal_index], high[1-normal_index], building.height))
        fences.sort()
        variance = np.zeros_like(a['icellflag'], dtype=float)
        dissipation = np.zeros_like(variance)
        influence = np.zeros_like(variance[0])
        for index, (face, left, right, height) in enumerate(fences):
            # A second fence spanning the first wake limits the circulation
            # between them. Do not truncate the wake for a staggered end gap.
            length = min((other - face for other, lo, hi, _ in fences[index+1:]
                          if other > face and lo <= left and hi >= right),
                         default=self.length_h*height)
            length = min(length, self.length_h*height)
            confined = self.confined_gap and length < self.length_h*height
            speed = abs(self.inflow.evaluate(height)[normal_index])
            side = np.clip(np.minimum(c[across]-left, right-c[across])/height, 0., 1.)
            side = side**2*(3.-2.*side)

            def envelope(distance):
                downwind = np.maximum(distance, 0.)
                return (-np.expm1(-downwind/(.18*height)) * np.exp(-downwind/(3.*height))
                        * np.maximum(1. - (downwind/length)**2, 0.)**2)

            z_face = np.maximum(c['z_face'], 0.)
            depth = .65*height
            vertical = depth*(-np.expm1(-z_face/depth))*np.exp(-(z_face/(3.*height))**2)
            if confined:
                # Close the shortened circulation below the crest. Otherwise
                # shortening the envelope accelerates its deep return flow
                # and pulls too much over-crest fluid into the gap.
                crest_fraction = np.clip(z_face/height, 0., 1.)
                vertical *= 1. - crest_fraction**2*(3. - 2.*crest_fraction)
            stream = -self.deficit*speed*vertical[:, None]*envelope(sign*c[axis+'_face']-face)[None, :]
            normal = sign*np.diff(stream, axis=0)/np.diff(c['z_face'])[:, None]
            up = -sign*np.diff(stream, axis=1)/np.diff(c[axis+'_face'])[None, :]

            def orient(plane):
                values = plane[:, :, None]*side[None, None, :]
                return values if axis == 'y' else values.swapaxes(1, 2)

            a['uv'[normal_index]+'0'][:-1, :len(c['y'])+normal_index, :len(c['x'])+1-normal_index] += orient(normal)
            a['w0'][:, :-1, :-1] += orient(up)
            distance = np.maximum(sign*c[axis]-face, 0.)
            footprint = envelope(distance)[:, None]*side[None, :]
            influence = np.maximum(influence, footprint if axis == 'y' else footprint.T)
            width = .15*height + .10*distance
            shear_center = height + .06*distance
            gaussian = np.exp(-.5*((c['z'][:, None]-shear_center)/width)**2)
            sigma_squared = (.18*speed)**2 * gaussian * envelope(distance)
            variance = np.maximum(variance, orient(sigma_squared))
            dissipation = np.maximum(dissipation, orient(5.7*sigma_squared**1.5/width))
        if fences:
            context.fence_variance = variance
            context.fence_dissipation = dissipation
            context.fence_influence = influence


@dataclass(frozen=True)
class FenceShearTurbulence(LocalTurbulence):
    """Add the unresolved shear-layer variance to the local closure."""

    native_quirks: bool = field(default=False, kw_only=True)
    vertical_shear_std: float = field(default=0., kw_only=True)
    shear_height: float = field(default=1., kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if (not np.isfinite([self.vertical_shear_std, self.shear_height]).all()
                or self.vertical_shear_std < 0. or self.shear_height <= 0.):
            raise ValueError("Invalid wake vertical mixing parameters")

    def build(self, workspace, velocity, **kwargs):
        fields, metadata = super().build(workspace, velocity, **kwargs)
        context = workspace.building_context
        if not hasattr(context, 'fence_variance'):
            return fields, metadata
        fluid = _fluid(workspace)
        added = np.where(fluid, context.fence_variance, 0.)
        for key, factor in zip(('txx', 'tyy', 'tzz'), (1., 1., .65), strict=True):
            fields[key] += factor*added
        fields['tke'] += 1.325*added
        fields['CoEps'] += np.where(fluid, context.fence_dissipation, 0.)
        if self.vertical_shear_std:
            z = workspace.coordinates['z']
            width = .5*self.shear_height
            band = np.exp(-.5*((z-self.shear_height)/width)**2)
            # The wall-to-crest shear drives extra vertical fluctuations.
            # Use the same horizontal wake envelope as the mean closure.
            extra = self.vertical_shear_std**2 * band[:, None, None] * context.fence_influence
            extra = np.where(fluid, extra, 0.)
            fields['tzz'] += extra
            fields['tke'] += .5*extra
            fields['CoEps'] += 5.7*extra**1.5/width
            metadata['fence_peak_vertical_variance'] = float(extra.max())
        stress = np.zeros(fluid.shape+(3, 3))
        for name, i, j in (('txx', 0, 0), ('txy', 0, 1), ('txz', 0, 2),
                           ('tyy', 1, 1), ('tyz', 1, 2), ('tzz', 2, 2)):
            stress[..., i, j] = stress[..., j, i] = fields[name]
        divergence = self.stress_divergence(workspace, stress)
        for i, axis in enumerate('xyz'):
            fields['div_tau_'+axis] = divergence[..., i]
        metadata['fence_peak_added_variance'] = float(added.max())
        return fields, metadata
