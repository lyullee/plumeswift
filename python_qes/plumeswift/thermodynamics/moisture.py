"""Trace-water energy module, independently replaceable from hydrogen EOS."""
import numpy as np

def saturation_pressure_water(T):
    T=np.asarray(T,dtype=float);t=np.maximum(T,173.)-273.15
    value=611.21*np.exp(np.clip((18.678-t/234.5)*t/(257.14+t),-100,30))
    return np.where(T>173.,value,0.)

class TraceWaterEnergy:
    def enthalpy_change(self,temperatures,ambient_temperature,relative_humidity,pressure,properties,liquid_fraction_at_freezing=None):
        T=np.asarray(temperatures);Ta=ambient_temperature;p=pressure
        pw=relative_humidity/100*float(saturation_pressure_water(Ta));wa=.622*pw/(p-pw)
        ps=saturation_pressure_water(T);wv=np.minimum(wa,.622*ps/np.maximum(p-ps,1))
        hwv0=float(properties.enthalpy('Water',273.15,100.,phase='gas'))
        hwl0=float(properties.enthalpy('Water',273.15,p,phase='liquid'))
        hvapor=hwv0+1860*(T-273.15)
        hcond=np.where(T<273.15,hwl0-333500+2100*(T-273.15),hwl0+4180*(T-273.15))
        if liquid_fraction_at_freezing is not None:
            f=np.asarray(liquid_fraction_at_freezing);coexist=np.isfinite(f)
            hcond[coexist]=hwl0-333500*(1-f[coexist])
        return wv*hvapor+(wa-wv)*hcond-wa*(hwv0+1860*(Ta-273.15))
    def describe(self):return {'name':'trace-water sensible/latent heat with ice-liquid coexistence',
          'water_mass_loading':False,'vapor_cp_J_kgK':1860,'ice_cp_J_kgK':2100,'liquid_cp_J_kgK':4180,'fusion_J_kg':333500}
