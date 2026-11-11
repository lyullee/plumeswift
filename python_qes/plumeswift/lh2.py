"""Original research LH2 momentum coupling, thermodynamics supplied separately.

Same buoyancy, quadratic drag and ground reaction as the executed C++
prototype. The concentration reconstruction conserves H2 mass. Replacing
the Thermodynamics object does not replace these force equations.
"""
from dataclasses import dataclass,field
from typing import Callable
import numpy as np
from .contracts import PhysicalCoefficients,Thermodynamics
from .lh2_dense import DenseLayerExchange, ShallowLayerPressure
from .lh2_energy import CryogenicMixture
from .lh2_ground_heat import CloudGroundHeat
from .lh2_plume_forces import PlumeSectionForces

@dataclass
class LH2Momentum:
    thermo:Thermodynamics|CryogenicMixture
    coefficients:PhysicalCoefficients=field(default_factory=PhysicalCoefficients)
    shallow_layer:ShallowLayerPressure|None=None
    ground_heat:CloudGroundHeat|None=None
    plume_forces:PlumeSectionForces|None=None
    ground_contact_z:float|None=None
    drag_length:Callable[[np.ndarray], np.ndarray]|None=None
    force_density_reconstruction:object|None=None
    vertical_buoyancy:bool=True
    vertical_force_density:Callable|None=None
    vertical_drift_update:Callable|None=None
    pressure_active_steps:int=field(default=0,init=False)
    peak_pressure_increment_m_s:float=field(default=0.,init=False)
    ground_heat_j:float=field(default=0.,init=False)

    def __post_init__(self):
        if self.ground_contact_z is not None and not np.isfinite(self.ground_contact_z):
            raise ValueError('Ground contact elevation must be finite')

    def update(self,positions,masses,drift,cell_ids,cell_volumes,dt,*,enthalpy=None,
               carrier_air_mass=None, mixture_mass_ratio=None):
        """Advance the LH₂ drift shared by markers in each occupied cell.

        Positions are (N, 3) in m, masses are (N,) in kg and drift is (N, 3)
        in m/s. Cell volumes are a scalar or one m³ value per sorted unique
        cell ID. dt is seconds. Return updated drift and mass diagnostics.
        The optional enthalpy array holds J/kg of H₂ for each marker.
        """
        if len(positions)==0:return drift.copy(),{"active_mass_kg":0.,"cold_mass_kg":0.}
        ids,inverse=np.unique(cell_ids,return_inverse=True)
        mass=np.bincount(inverse,weights=masses,minlength=len(ids))
        inertia = masses if carrier_air_mass is None else masses + carrier_air_mass
        cell_inertia = mass if carrier_air_mass is None else np.bincount(inverse, weights=inertia)
        velocity=np.column_stack([np.divide(np.bincount(inverse,weights=inertia*drift[:,d],minlength=len(ids)),cell_inertia,
                    out=np.zeros(len(ids)),where=cell_inertia>0)for d in range(3)])
        volume=np.broadcast_to(cell_volumes,(len(ids),))
        energy = None
        if enthalpy is None:
            state=self.thermo.evaluate(mass/volume)
        else:
            energy=np.bincount(inverse,weights=masses*enthalpy,minlength=len(ids))
            state=self.thermo.evaluate(mass/volume,energy/volume)
            if self.ground_heat is not None:
                heat = self.ground_heat.energy(self.thermo, ids, volume, mass, energy, state, dt)
                if np.any(heat):
                    specific = np.divide(heat, mass, out=np.zeros_like(heat), where=mass > 0)
                    enthalpy += specific[inverse]
                    self.ground_heat_j += float(heat.sum())
                    energy = energy + heat
                    state = self.thermo.evaluate(mass / volume, energy / volume)
        # Evaluate buoyancy from the cell mixture, then distribute the
        # mass-weighted drift back to the individual markers.
        if mixture_mass_ratio is not None:
            self.mix_vertical_momentum(
                velocity, drift, masses, inverse, mass, state.density * volume, mixture_mass_ratio)
        density = state.density
        if self.force_density_reconstruction is not None:
            # Correct only the force reconstruction: heat and all mass/energy
            # ledgers continue to use the entire occupied-cell inventory.
            counts = np.bincount(inverse, minlength=len(ids))
            representative = mass[inverse]/counts[inverse]
            if not np.allclose(masses, representative, rtol=1e-10, atol=1e-15):
                raise ValueError('Cell-average force reconstruction requires equal marker masses')
            density = self.force_density_reconstruction.density(self.thermo, mass, energy, volume, counts)
        buoyancy_density = density
        a = np.zeros_like(density)
        if self.vertical_buoyancy:
            if self.vertical_force_density is not None:
                buoyancy_density = self.vertical_force_density(
                    self.thermo, ids, mass, energy, volume, np.bincount(inverse))
            a = self.coefficients.gravity*(self.thermo.ambient_density-buoyancy_density)/self.thermo.ambient_density
        length = self.coefficients.lh2_drag_length_m
        if self.drag_length is not None:
            particle_length = self.drag_length(positions)
            length_sum = np.bincount(inverse, weights=inertia*particle_length)
            length = np.divide(length_sum, cell_inertia, out=np.full_like(length_sum, length), where=cell_inertia > 0.)
        denominator=1+.5*self.coefficients.lh2_drag_coefficient*dt*np.linalg.norm(velocity,axis=1)/length
        if self.plume_forces is None or not self.vertical_buoyancy:
            if self.vertical_drift_update is None:
                velocity[:, 2] += a * dt
            else:
                velocity[:, 2] = self.vertical_drift_update(
                    positions, inverse, mass, masses, velocity[:, 2], a, dt)
            velocity /= denominator[:, None]
        else:
            original_vertical = (velocity[:, 2] + a * dt) / denominator
            updated_vertical = self.plume_forces.vertical_velocity(
                positions, masses, inverse, mass, buoyancy_density, self.thermo.ambient_density,
                velocity[:, 2], original_vertical, self.coefficients.gravity,
                self.coefficients.lh2_drag_coefficient, dt)
            velocity[:, :2] /= denominator[:, None]
            velocity[:, 2] = updated_vertical
        result=drift.copy();occupied=mass[inverse]>0;result[occupied]=velocity[inverse[occupied]]
        if self.shallow_layer is not None:
            increment=self.shallow_layer.increment(positions,cell_ids,ids,inverse,density,
                self.thermo.ambient_density,self.coefficients.gravity,dt)
            result[occupied,:2]+=increment[occupied]
            peak=float(np.max(np.abs(increment),initial=0.))
            self.pressure_active_steps+=int(peak>0.)
            self.peak_pressure_increment_m_s=max(self.peak_pressure_increment_m_s,peak)
        if self.ground_contact_z is None:
            ground=occupied&(positions[:,2]<self.coefficients.lh2_ground_reaction_height_m)&(result[:,2]<0)
            result[ground,2]=0
        else:
            # Limit the dense slip by the distance available before contact.
            # The particle integrator still resolves wind and wall reflection.
            contact_velocity = -np.maximum(positions[occupied, 2]-self.ground_contact_z, 0.) / dt
            result[occupied, 2] = np.maximum(result[occupied, 2], contact_velocity)
            if isinstance(self.shallow_layer, DenseLayerExchange) and self.shallow_layer.wake_influence is not None:
                near = occupied & (positions[:, 2] < self.ground_contact_z + self.coefficients.lh2_ground_reaction_height_m)
                near &= result[:, 2] < 0.
                result[near, 2] *= self.shallow_layer.ground_weight(positions[near])
        return result,{"active_mass_kg":float(mass.sum()),"cold_mass_kg":float(mass[state.temperature<80].sum())}

    @staticmethod
    def mix_vertical_momentum(velocity, drift, masses, inverse, cell_mass, mixture_mass, ratios):
        """Share vertical momentum over the inferred mixture plus newly entrained air.

        Each marker starts from its first occupied-cell estimate. Subsequent
        increases in mixture mass per tracer mass represent entrainment of
        air with zero relative drift. A concentration rebound cannot restore
        the lost drift. This ratio is a closure for inertia, not an air tracer.
        """
        inferred = np.divide(mixture_mass, cell_mass, out=np.ones_like(cell_mass),
                             where=cell_mass > 0.)
        inferred = np.maximum(1., inferred[inverse])
        prior = np.where(ratios > 0., ratios, inferred)
        np.maximum(prior, inferred, out=ratios)
        momentum = np.bincount(inverse, weights=masses * prior * drift[:, 2],
                               minlength=len(cell_mass))
        mixed_mass = np.bincount(inverse, weights=masses * ratios, minlength=len(cell_mass))
        velocity[:, 2] = np.divide(momentum, mixed_mass, out=np.zeros_like(momentum),
                                  where=mixed_mass > 0.)
