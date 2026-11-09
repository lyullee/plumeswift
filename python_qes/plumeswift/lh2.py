"""Original research LH2 momentum coupling, thermodynamics supplied separately.

Same buoyancy, quadratic drag and ground reaction as the executed C++
prototype. The concentration reconstruction conserves H2 mass. Replacing
the Thermodynamics object does not replace these force equations.
"""
from dataclasses import dataclass,field
import numpy as np
from .contracts import PhysicalCoefficients,Thermodynamics

@dataclass
class LH2Momentum:
    thermo:Thermodynamics
    coefficients:PhysicalCoefficients=field(default_factory=PhysicalCoefficients)
    def update(self,positions,masses,drift,cell_ids,cell_volumes,dt):
        """Advance the LH₂ drift shared by markers in each occupied cell.

        Positions are (N, 3) in m, masses are (N,) in kg and drift is (N, 3)
        in m/s. Cell volumes are a scalar or one m³ value per sorted unique
        cell ID. dt is seconds. Return updated drift and mass diagnostics.
        """
        if len(positions)==0:return drift.copy(),{"active_mass_kg":0.,"cold_mass_kg":0.}
        ids,inverse=np.unique(cell_ids,return_inverse=True)
        mass=np.bincount(inverse,weights=masses,minlength=len(ids))
        velocity=np.column_stack([np.divide(np.bincount(inverse,weights=masses*drift[:,d],minlength=len(ids)),mass,
                    out=np.zeros(len(ids)),where=mass>0)for d in range(3)])
        volume=np.broadcast_to(cell_volumes,(len(ids),))
        state=self.thermo.evaluate(mass/volume)
        # Evaluate buoyancy from the cell mixture, then distribute the
        # mass-weighted drift back to the individual markers.
        a=self.coefficients.gravity*(self.thermo.ambient_density-state.density)/self.thermo.ambient_density
        denominator=1+.5*self.coefficients.lh2_drag_coefficient*dt*np.linalg.norm(velocity,axis=1)/self.coefficients.lh2_drag_length_m
        velocity[:,2]+=a*dt;velocity/=denominator[:,None]
        result=drift.copy();occupied=mass[inverse]>0;result[occupied]=velocity[inverse[occupied]]
        ground=occupied&(positions[:,2]<self.coefficients.lh2_ground_reaction_height_m)&(result[:,2]<0)
        result[ground,2]=0
        return result,{"active_mass_kg":float(mass.sum()),"cold_mass_kg":float(mass[state.temperature<80].sum())}
