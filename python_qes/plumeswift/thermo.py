"""Interchangeable homogeneous-mixture thermodynamic providers.

The legacy provider preserves the previously executed constant-cp closure.
The optional real-property provider replaces ONLY property/enthalpy closure,
not jet, pool, entrainment, buoyancy or drag equations. Its mixture density
uses Amagat's rule, not an exact H2/air mixture EOS. Air phase equilibrium
is not supplied; the default gas-only range starts at 85 K and out-of-domain
queries fail explicitly.
"""
import numpy as np
from .contracts import ThermoState,ModelDomainError
from .thermodynamics.moisture import saturation_pressure_water

class EquilibriumTable:
    name="abstract"
    def _finish(self,table,metadata):
        self.table=np.asarray(table,dtype=float)
        if self.table.ndim!=2 or self.table.shape[1]!=5 or not np.isfinite(self.table).all():raise ValueError("Invalid property table")
        if not np.all(np.diff(self.table[:,0])>0):raise ValueError("Concentration inversion is not unique")
        if np.any(self.table[:,1:3]<=0)or np.any((self.table[:,3:]<0)|(self.table[:,3:]>1)):raise ValueError("Invalid thermodynamic states")
        self.table.flags.writeable=False
        self.ambient_density=float(self.table[0,2]);self.metadata=metadata
    def evaluate(self,concentration):
        """Map H₂ mass concentration (kg/m³) to mixture state on the same axes.

        Temperature is K, density is kg/m³, and volume and vapor fractions
        are dimensionless. The provider's range policy controls extrapolation.
        """
        c=np.asarray(concentration,dtype=float)
        if not np.isfinite(c).all() or np.any(c<0):raise ModelDomainError("H2 mass concentration must be finite and nonnegative")
        if np.any(c>self.table[-1,0]) and self.out_of_range=="raise":
            raise ModelDomainError(f"{self.name}: C exceeds {self.table[-1,0]:.6g} kg/m3; unresolved cold-mixture/air phase state")
        values=[np.interp(c,self.table[:,0],self.table[:,j])for j in range(1,5)]
        return ThermoState(c,*values)
    def describe(self):
        return {"name":self.name,"ambient_density_kg_m3":self.ambient_density,"out_of_range":self.out_of_range,
                "concentration_range_kg_m3":[0,float(self.table[-1,0])],**self.metadata}

class LegacyEquilibrium(EquilibriumTable):
    name="constant-cp equilibrium (legacy research closure)"
    def __init__(self,ambient_temperature=276.45,liquid_fraction=.559,relative_humidity=75.,
                 pressure=101325.,warm=False,points=10000,out_of_range="clip"):
        if not 0<=liquid_fraction<=1 or not 0<=relative_humidity<=100:raise ValueError("Invalid source phase/RH")
        if pressure<=0 or ambient_temperature<=20.37:raise ValueError("Invalid ambient state")
        if out_of_range not in {"clip","raise"}:raise ValueError(out_of_range)
        self.out_of_range=out_of_range;Ta=ambient_temperature;p=pressure
        y=np.r_[0,np.geomspace(1e-9,.999999,points),1.]
        pw=relative_humidity/100*saturation_pressure_water(Ta);wa=.622*pw/(p-pw)
        Tb=20.37;L=445000.;q=np.ones_like(y)
        def heat(T):
            ps=saturation_pressure_water(T);cond=np.maximum(0,wa-.622*ps/np.maximum(p-ps,1))
            water=cond*(2500000+np.where(T<273.15,333500,0))
            return (1-y)*(1005*(T-Ta)-water)+y*(14300*(T-Tb)+L*liquid_fraction)
        if warm:T=np.full_like(y,Ta)
        else:
            liquid=heat(np.full_like(y,Tb))>0
            lo=np.full_like(y,Tb);hi=np.full_like(y,Ta)
            for _ in range(45):
                mid=(lo+hi)/2;positive=heat(mid)>0;hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
            T=(lo+hi)/2;T[0]=Ta;T[liquid]=Tb
            q[liquid]=np.clip(1-liquid_fraction+(1-y[liquid])*(1005*(Ta-Tb)+wa*2833500)/(y[liquid]*L),0,1)
        vh=y*q*8.314462618*T/(p*.002016);va=(1-y)*8.314462618*T/(p*.028965)
        vl=y*(1-q)/70.8;rho=1/(vh+va+vl)
        xv=np.divide(vh,vh+va,out=np.zeros_like(y),where=vh+va>0)
        self._finish(np.c_[y*rho,T,rho,xv,q],{"ambient_temperature_K":Ta,"liquid_fraction":liquid_fraction,
          "relative_humidity_percent":relative_humidity,"pressure_Pa":p,"warm":warm,"provider":"Python/NumPy"})

class CoolPropEquilibrium(EquilibriumTable):
    name="real pure-fluid properties + Amagat mixture, variable enthalpy"
    def __init__(self,ambient_temperature=276.45,liquid_fraction=.559,relative_humidity=75.,
                 pressure=101325.,minimum_temperature=85.,points=4096,out_of_range="raise",
                 properties=None,mixing_rule=None,moisture=None):
        from .thermodynamics import CoolPropProperties,AmagatMixing,TraceWaterEnergy
        properties=properties or CoolPropProperties();mixing_rule=mixing_rule or AmagatMixing();moisture=moisture or TraceWaterEnergy()
        hydrogen=properties.hydrogen_fluid
        if not 0<=liquid_fraction<=1 or not 0<=relative_humidity<=100:raise ValueError("Invalid source phase/RH")
        if not 80<=minimum_temperature<ambient_temperature:raise ModelDomainError("Provider requires 80K <= Tmin < Tamb")
        if out_of_range!="raise":raise ValueError("Real-property provider requires explicit out-of-domain failure")
        self.out_of_range=out_of_range;Ta=float(ambient_temperature);p=float(pressure)
        dewpoint=properties.saturation('Air',p,1)['temperature']
        if minimum_temperature<=dewpoint:raise ModelDomainError(f"Air gas EOS requires T > dew point {dewpoint:.6g} K at this pressure")
        sat_liq=properties.saturation(hydrogen,p,0);sat_vap=properties.saturation(hydrogen,p,1)
        Tb=sat_liq['temperature'];hl=sat_liq['enthalpy'];hv=sat_vap['enthalpy']
        hsource=liquid_fraction*hl+(1-liquid_fraction)*hv
        T=np.linspace(minimum_temperature,Ta,points)
        phase_fraction=np.full_like(T,np.nan)
        if minimum_temperature<273.15<Ta:
            T=np.r_[T[T<273.15],np.full(65,273.15),T[T>273.15]]
            phase_fraction=np.full_like(T,np.nan);phase_fraction[T==273.15]=np.linspace(0,1,65)
        hh=properties.enthalpy(hydrogen,T,p);ha=properties.enthalpy('Air',T,p)
        rh=properties.density(hydrogen,T,p);ra=properties.density('Air',T,p)
        if not all(np.isfinite(v).all()for v in [hh,ha,rh,ra]):
            raise ModelDomainError("Property provider returned non-finite gas states; no rows were dropped")
        hair0=properties.enthalpy('Air',Ta,p)
        water_energy=moisture.enthalpy_change(T,Ta,relative_humidity,p,properties,phase_fraction)
        dhair=ha-hair0+water_energy;dhhydrogen=hh-hsource
        y=np.clip(-dhair/(dhhydrogen-dhair),0,1)
        rho,xv=mixing_rule.density_and_volume_fraction(y,rh,ra);c=y*rho
        residual=y*dhhydrogen+(1-y)*dhair
        order=np.argsort(c);table=np.c_[c,T,rho,xv,np.ones_like(c)][order]
        unique=np.r_[True,np.diff(table[:,0])>1e-14];table=table[unique]
        self._finish(table,{"ambient_temperature_K":Ta,"liquid_fraction":liquid_fraction,"relative_humidity_percent":relative_humidity,
          "pressure_Pa":p,"minimum_temperature_K":minimum_temperature,"provider":"CoolProp HEOS Hydrogen/Air + Python mixture closure",
          "property_component":properties.describe(),"mixing_component":mixing_rule.describe(),"moisture_component":moisture.describe(),
          "table_energy_residual_max_J_kg":float(np.max(abs(residual))),"source_saturation_temperature_K":Tb,
          "source_specific_enthalpy_J_kg":hsource,"latent_heat_J_kg":hv-hl,
          "limitations":["Amagat mixture approximation, not full H2-N2-O2 mixture equilibrium",
                         "No air condensation below 80K; no pool or droplet transport",
                         "Trace water mass loading neglected; water heat capacities piecewise constant"]})
