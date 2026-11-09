"""Independent mixture-density rule; thermodynamic approximation is explicit."""
import numpy as np

class AmagatMixing:
    def density_and_volume_fraction(self,hydrogen_mass_fraction,rho_hydrogen,rho_air):
        y=np.asarray(hydrogen_mass_fraction);vh=y/rho_hydrogen;va=(1-y)/rho_air
        if np.any(vh+va<=0):raise ValueError("Mixture specific volume must be positive")
        return 1/(vh+va),vh/(vh+va)
    def describe(self):return {'name':'Amagat additive specific volumes','full_mixture_EOS':False}
