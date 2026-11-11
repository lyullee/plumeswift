"""Pure-fluid property adapter; no dispersion, force, or mixture equations."""
import numpy as np

class CoolPropProperties:
    def __init__(self,hydrogen_fluid='Hydrogen'):
        if hydrogen_fluid not in ['Hydrogen','ParaHydrogen','OrthoHydrogen']:raise ValueError("Unknown hydrogen spin model")
        from CoolProp.CoolProp import PropsSI
        import CoolProp
        self._query=PropsSI;self.version=CoolProp.__version__;self.hydrogen_fluid=hydrogen_fluid
    def enthalpy(self,fluid,temperature,pressure,phase=None):
        return np.asarray(self._query('Hmass','T'if phase is None else 'T|'+phase,temperature,'P',pressure,fluid))
    def density(self,fluid,temperature,pressure,phase=None):
        return np.asarray(self._query('Dmass','T'if phase is None else 'T|'+phase,temperature,'P',pressure,fluid))
    def saturation(self,fluid,pressure,quality):
        return {key:float(self._query(output,'P',pressure,'Q',quality,fluid))for key,output in [('temperature','T'),('enthalpy','Hmass')]}
    def saturation_density(self, fluid, pressure, quality):
        return float(self._query('Dmass', 'P', pressure, 'Q', quality, fluid))
    def describe(self):return {'provider':'CoolProp HEOS','version':self.version,'hydrogen_fluid':self.hydrogen_fluid}
