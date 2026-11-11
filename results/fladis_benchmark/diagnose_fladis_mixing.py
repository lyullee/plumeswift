"""Analytical and EOS diagnostics, without concentration-based calibration."""
import json

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from CoolProp.CoolProp import PropsSI

from run_fladis import ROOT, TRIALS, R, MW, CP, AIR_CP, AIR_MW, write_csv
from fladis_sensitivity import HeatedAmmonia, VerticalTurbulence, VERTICAL_RATIO
from plumeswift.numerics.langevin import implicit_gle
from plumeswift.thermodynamics.ammonia_moist import MoistAmmoniaVapour


OUT = ROOT/'mixing_thermo'
HUMIDITY = {9:86., 16:62., 24:53.6}


def main():
    OUT.mkdir(exist_ok=True)
    covariance, states, sources, floors = [], [], [], []
    for trial, cfg in TRIALS.items():
        turbulence = VerticalTurbulence(terrain_roughness=.04, friction_velocity=cfg['ustar'],
            crosswind_std=cfg['ustar']*cfg['sigma_v_ratio'], reciprocal_length=1/cfg['length'],
            measured_ratio=VERTICAL_RATIO[trial])
        for z in (.125,.5,1.5,10.):
            var, eps = turbulence.profiles(z)
            stress = np.diag([(2.5*cfg['ustar'])**2,
                              (cfg['ustar']*cfg['sigma_v_ratio'])**2, float(var)])
            stress[0,2] = stress[2,0] = -cfg['ustar']**2
            for dt in (.01,.05,.25):
                for scheme in ('backward-euler','midpoint'):
                    # Derive the exact discrete map using basis vectors, no MC noise.
                    tau = np.repeat(stress[None],3,axis=0)
                    transition = implicit_gle(np.eye(3),tau,tau,np.zeros((3,3)),
                        5.7*float(eps),dt,np.zeros((3,3)),scheme=scheme)[0].T
                    noise = implicit_gle(np.zeros((3,3)),tau,tau,np.zeros((3,3)),
                        5.7*float(eps),dt,np.eye(3),scheme=scheme)[0].T
                    stationary = solve_discrete_lyapunov(transition,noise@noise.T)
                    lag_sum = np.linalg.solve(np.eye(3)-transition,transition@stationary)
                    diffusivity = .5*dt*(stationary+lag_sum+lag_sum.T)
                    continuous_diffusivity = 2.*stress@stress/(5.7*float(eps))
                    # The lower BE variance alone does not imply a lower
                    # homogeneous long-time diffusivity: lag correlations matter.
                    np.testing.assert_allclose(diffusivity,continuous_diffusivity,rtol=1e-8,atol=1e-9)
                    covariance.append(dict(trial=trial,z_m=z,dt_s=dt,scheme=scheme,
                        prescribed_sigma_w_m_s=np.sqrt(var),
                        stationary_sigma_w_m_s=np.sqrt(stationary[2,2]),
                        vertical_variance_ratio=stationary[2,2]/var,
                        vertical_long_time_diffusivity_m2_s=diffusivity[2,2],
                        homogeneous_diffusivity_ratio=diffusivity[2,2]/continuous_diffusivity[2,2]))
                    if scheme=='midpoint':
                        np.testing.assert_allclose(stationary,stress,rtol=1e-9,atol=1e-10)
        dry = HeatedAmmonia(cfg['air_k'], cfg['pressure'])
        moist = MoistAmmoniaVapour(cfg['air_k'],cfg['pressure'],HUMIDITY[trial])
        h = (cfg['jet_k']-cfg['air_k'])*(CP+(1-.12)/.12*AIR_MW/MW*AIR_CP)
        source_c = .12*cfg['pressure']*MW/(R*cfg['jet_k'])
        humid_h = float(moist.heating_limit(source_c,cfg['jet_k'])/source_c)
        recovered = moist.evaluate(source_c,source_c*humid_h)
        assert np.isclose(recovered.temperature,cfg['jet_k'],atol=1e-8)
        assert np.isclose(recovered.volume_fraction,.12,atol=1e-9)
        sources.append(dict(trial=trial,rh_pct=HUMIDITY[trial],dry_source_h_j_kg=h,
            humid_source_h_same_temperature_j_kg=humid_h,
            humid_source_T_at_old_energy_K=float(moist.evaluate(source_c,source_c*h).temperature),
            prescribed_source_T_K=cfg['jet_k'],water_loading=moist.water_loading))
        for concentration_ppm in (70,140,1000,5000,120000):
            c = concentration_ppm*1e-6*cfg['pressure']*MW/(R*cfg['air_k'])
            for mode, thermo, enthalpy in (('dry',dry,h),('moist_fixed_energy',moist,h),
                                           ('moist_fixed_source_T',moist,humid_h)):
                state=thermo.evaluate(c,c*enthalpy)
                temp=float(state.temperature)
                partial=float(state.volume_fraction)*cfg['pressure']
                B=PropsSI('Bvirial','T',max(200.,temp),'P',cfg['pressure'],'Ammonia')
                states.append(dict(trial=trial,mode=mode,nominal_ambient_ppmv=concentration_ppm,
                    concentration_kg_m3=c,temperature_k=temp,density_kg_m3=float(state.density),
                    gas_ppmv=float(state.volume_fraction)*1e6,
                    buoyancy_acceleration_m_s2=9.80665*(thermo.ambient_density-float(state.density))/thermo.ambient_density,
                    water_condensate_kg_m3=float(getattr(state,'water_condensate_density',0.)),
                    nh3_self_virial_partial_pressure_abs_delta_Z=abs(B*partial/(R*temp))))
        mass_per_marker=cfg['rate']/120
        single=dry.evaluate(mass_per_marker,mass_per_marker*h)
        floors.append(dict(trial=trial,marker_mass_kg=mass_per_marker,force_cell_volume_m3=1.,
                          single_marker_force_cell_ppmv=float(single.volume_fraction)*1e6,
                          single_marker_force_cell_buoyancy_m_s2=9.80665*(dry.ambient_density-float(single.density))/dry.ambient_density))
    for name,rows in (('stationary_covariance',covariance),('thermodynamic_states',states),
                       ('source_conventions',sources),('marker_resolution',floors)):
        write_csv(OUT/f'{name}.csv',rows)
    print(json.dumps({'source_conventions':sources,'marker_resolution':floors},indent=2))


if __name__=='__main__':
    main()
