"""QES deposition constitutive equations in SI units (Deposition.cpp, GPL-3.0).

The pinned native ParticleModel.cpp comments out the call to Deposition::deposit.
The CPU plume can opt into that kernel before wall reflection. The original
base particle path defaults to inactive deposition.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DepositionResult:
    remaining_kg: np.ndarray
    deposited_kg: np.ndarray
    cutoff_discard_kg: np.ndarray
    inactive: np.ndarray


def vegetation_survival(displacement_m, velocity_m_s, density_kg_m3, diameter_m, stress, coeps):
    """Bailey/Price vegetation mass-survival factor, original constants."""
    nu = 1.506e-5; rho_air = 1.225
    displacement = np.asarray(displacement_m, dtype=float)
    speed = np.linalg.norm(velocity_m_s, axis=-1)
    distance = np.linalg.norm(displacement, axis=-1)
    trace = np.trace(stress, axis1=-2, axis2=-1)
    coeps = np.asarray(coeps)
    if np.any(trace < 0) or np.any(coeps <= 0) or not np.isfinite(coeps).all():
        raise ValueError('Vegetation deposition requires nonnegative stress trace and positive CoEps')
    rms = np.sqrt(trace/3)
    microscale = np.sqrt(15*nu*5*rms*rms/coeps)
    stokes = density_kg_m3*np.asarray(diameter_m)**2*speed/(18*rho_air*nu*.1)
    re_lambda = rms*microscale/nu
    efficiency = 1-1/(2.049*(re_lambda**.3*stokes)**1.19+1)
    re_leaf = np.clip(.1*speed/nu, 400., 6000.)
    gamma = -6.5e-5*re_leaf+.43
    return np.exp(-efficiency*5.57*(1+gamma)*distance*.7)


def ground_survival(displacement_m, velocity_m_s, stress, eddy_viscosity_m2_s, settling_m_s, layer_width_m):
    """Original aerodynamic/quasi-laminar resistance ground model."""
    nu = 1.506e-5
    speed = np.linalg.norm(velocity_m_s, axis=-1)
    distance = np.linalg.norm(displacement_m, axis=-1)
    viscosity = np.asarray(eddy_viscosity_m2_s, dtype=float)
    settling = np.asarray(settling_m_s, dtype=float)
    layer_width = np.asarray(layer_width_m, dtype=float)
    if (not np.isfinite(viscosity).all() or not np.isfinite(settling).all()
            or np.any(viscosity < 0) or np.any(layer_width <= 0) or np.any(settling < 0)):
        raise ValueError('Ground deposition requires positive layer width and nonnegative nuT and settling')
    if np.any((speed == 0) & (distance != 0)):
        raise ValueError('Nonzero displacement with zero particle speed')
    dt = np.divide(distance, speed, out=np.zeros_like(speed), where=speed > 0)
    tau = np.asarray(stress)
    ustar = (tau[..., 0, 2]**2+tau[..., 1, 2]**2)**.25
    inverse_sc = viscosity/nu
    positive = ustar > 0
    safe_u = np.where(positive, ustar, 1.)
    ra = np.log1p((4900*safe_u/nu)/(100*safe_u/nu+inverse_sc))/(.4*safe_u)
    stokes = settling*safe_u**2/(9.81*nu)
    impaction = np.zeros_like(stokes)
    np.power(10., np.divide(-3., stokes, out=np.full_like(stokes, -np.inf), where=stokes > 0), out=impaction)
    # Use 1/rb directly so zero eddy viscosity or impaction has a finite limit.
    conductance = safe_u*(inverse_sc**(2/3)+impaction)
    deposition_velocity = np.where(positive, conductance/(ra*conductance+1+ra*settling)+settling, settling)
    return np.exp(-deposition_velocity*dt/layer_width)


def apply_survival(mass_kg, survival, density_kg_m3, diameter_m):
    mass = np.asarray(mass_kg, dtype=float)
    fraction = np.asarray(survival, dtype=float)
    if np.any(mass < 0) or not np.isfinite(mass).all() or not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Invalid mass or survival fraction')
    remaining = mass*fraction
    one_particle = np.asarray(density_kg_m3)*np.pi/6*np.asarray(diameter_m)**3
    inactive = remaining < one_particle
    discarded = np.where(inactive, remaining, 0.)
    return DepositionResult(np.where(inactive, 0., remaining), mass-remaining, discarded, inactive)


class DepositionField:
    """Accumulate the original deposition buffer in the departure grid cell."""

    def __init__(self, sampler, flags):
        self.deposited_kg = np.zeros(flags.shape)
        self.cutoff_discard_kg = 0.
        self.cutoff_particles = 0
        self.set_fields(sampler, flags)

    def set_fields(self, sampler, flags):
        if flags.shape != self.deposited_kg.shape:
            raise ValueError('Deposition grid changed during plume transport')
        self.sampler = sampler
        self.canopy = np.isin(flags, (20, 22, 24, 28))
        self.terrain = flags == 2

    def _cells(self, position):
        indices = self.sampler.candidate_cell_indices(position)
        bounds = self.deposited_kg.shape
        inside = np.logical_and.reduce([(axis >= 0) & (axis < n) for axis, n in zip(indices, bounds)])
        return tuple(np.clip(axis, 0, n-1) for axis, n in zip(indices, bounds)), inside

    def record(self, cells, result):
        np.add.at(self.deposited_kg, cells, result.deposited_kg)
        self.cutoff_discard_kg += float(result.cutoff_discard_kg.sum())
        self.cutoff_particles += int(result.inactive.sum())

    def apply(self, mass_kg, material, start, displacement, velocity, stress, coeps, viscosity, *,
              settling_m_s=0., record=None):
        """Apply deposition to the supplied trajectory with optional ground settling speed."""
        if getattr(self.sampler, 'native_math', False):
            # QES updates ParticleCore.pos before deposition, then recovers
            # the departure cell by subtracting the displacement in float32.
            movement = np.asarray(displacement, dtype=np.float32)
            endpoint = np.asarray(start, dtype=np.float32)+movement
            start = endpoint-movement
        else:
            endpoint = start+displacement
        old, old_inside = self._cells(start)
        new, new_inside = self._cells(endpoint)
        k, j, i = new
        canopy = old_inside & self.canopy[old]
        ground = old_inside & ~canopy & new_inside & (k > 0) & self.terrain[np.maximum(k-1, 0), j, i]
        selected = (canopy | ground) & (mass_kg > 0)
        if not selected.any():
            return mass_kg, np.zeros(len(mass_kg), dtype=bool)
        diameter, density = material.T
        survival = np.ones(len(mass_kg))
        vegetated = canopy & selected
        if vegetated.any():
            s = vegetated
            survival[s] = vegetation_survival(displacement[s], velocity[s], density[s],
                                              diameter[s], stress[s], coeps[s])
        s = ground & selected
        if s.any():
            width = self.sampler.zf[k[s]]-self.sampler.zf[k[s]-1]
            settling = np.broadcast_to(settling_m_s, mass_kg.shape)[s]
            nu_t = np.zeros_like(settling) if viscosity is None else np.maximum(viscosity[s], 0.)
            survival[s] = ground_survival(displacement[s], velocity[s], stress[s], nu_t, settling, width)
        s = selected
        result = apply_survival(mass_kg[s], survival[s], density[s], diameter[s])
        (self.record if record is None else record)(tuple(axis[s] for axis in old), result)
        remaining = mass_kg.copy()
        remaining[s] = result.remaining_kg
        inactive = np.zeros(len(mass_kg), dtype=bool)
        inactive[s] = result.inactive
        return remaining, inactive
