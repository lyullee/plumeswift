"""Trace-water energy module, independently replaceable from hydrogen EOS."""
import numpy as np

def saturation_pressure_water(T):
    T=np.asarray(T,dtype=float);t=np.maximum(T,173.)-273.15
    value=611.21*np.exp(np.clip((18.678-t/234.5)*t/(257.14+t),-100,30))
    return np.where(T>173.,value,0.)


WATER_TRIPLE_K = 273.16
R_WATER = 8.314462618 / .01801528


def equilibrium_water_pressure(temperature):
    """Water saturation pressure over liquid and IAPWS R14-08 ice Ih."""
    temperature = np.asarray(temperature)
    theta = np.maximum(temperature, 50.) / WATER_TRIPLE_K
    terms = (-21.2144006 * theta**.00333333333 + 27.3203819 * theta**1.20666667
             - 6.10598130 * theta**1.70333333)
    ice = np.where(temperature >= 50., 611.657 * np.exp(terms / theta), 0.)
    return np.where(temperature < WATER_TRIPLE_K, ice, saturation_pressure_water(temperature))


def water_enthalpy(temperature, *, condensed=False):
    """Specific enthalpy relative to liquid water at the triple point."""
    delta = np.asarray(temperature) - WATER_TRIPLE_K
    if condensed:
        return np.where(delta < 0., -333500. + 2100. * delta, 4180. * delta)
    return 2501000. + 1860. * delta

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
