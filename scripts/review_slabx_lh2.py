"""Re-run only the PUBLIC six-trial arc-maximum route. Not the private 210-record model."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from slabx.core.plume import run_dispersion
from slabx.core.source import HorizontalJet
from slabx.post.concentration import concentration_field
from slabx.submodels.atmosphere import Atmosphere
from slabx.thermo.base import Substance
from slabx.thermo.coolprop import CoolPropThermo, coolprop_water
from slabx_lh2.trials import FFI
from slabx_lh2.plume_width import plume_width_coupling
from slabx_lh2.water_ice import with_sublimation
from slabx_lh2.lfl import flammable_distance,brackets_from_arcs,verdict
from slabx_lh2.diagnostics import premise_summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/slabx_review';OUT.mkdir(exist_ok=True)
H2=Substance(name='H2',mw=.002016,cp_vapour=14300.,cp_liquid=9800.,dh_vap=445000.,T_boil=20.3,rho_liquid=70.8)

def main():
    observations=list(csv.DictReader((ROOT/'data/FFI/lh2_ffi_sensors.csv').open(encoding='utf-8')))
    concentrations=[r for r in observations if r['kind']=='concentration']
    result=[]
    for rh in [0,25,50,75,100]:
        for trial,tr in FFI.items():
            start=time.perf_counter()
            atm=Atmosphere(u_ref=tr.wind_m_s,z_ref=10,T=tr.temperature_C+273.15,rh=rh,z0=.01,stability='D')
            src=HorizontalJet(substance=H2,rate=tr.rate_kg_s,area=tr.area_m2,duration=tr.duration_s,
                              liquid_fraction=tr.liquid_fraction,height=tr.release_height_m,T_source=20.37)
            with plume_width_coupling():
                trajectory,_=run_dispersion(src,atm,CoolPropThermo(H2,fluid='Hydrogen'),
                    with_sublimation(coolprop_water()),x_max=300,n_puff_steps=40)
            predicted=[];measured=[]
            for radius in [30,50,100]:
                p=[]
                for z in [.1,1,1.8]:
                    field=concentration_field(trajectory,atm,z=z,t_avg=tr.averaging_window_s,t_release=tr.duration_s)
                    p.append(float(np.interp(radius,field.x,field.peak))*100)
                predicted.append(max(p))
                measured.append(max(float(r['max']) for r in concentrations if int(r['test'])==trial and float(r['R_m'])==radius))
            distance=flammable_distance(trajectory,atm,t_avg=tr.averaging_window_s,t_release=tr.duration_s,safety_factor=1.)['raw']
            bracket=brackets_from_arcs([30,50,100],measured)
            premise=premise_summary(trajectory)
            row={'trial':trial,'assumed_RH_pct':rh,'predicted_arc_c_volpct':predicted,'observed_temporal_max_arc_c_volpct':measured,
                 'c30_model_over_observed_max':predicted[0]/measured[0],
                 'LFL_raw_m':float(distance),'observed_LFL_bracket_m':[float(bracket[0]),float(bracket[1]) if math.isfinite(bracket[1]) else None],
                 'verdict':verdict(distance,bracket),'max_wc_over_u':float(premise['max']),
                 'wall_s':time.perf_counter()-start,'orientation':tr.orientation}
            result.append(row)
            print(trial,rh,round(row['LFL_raw_m'],3),row['verdict'],flush=True)
    distribution={stat:{str(level):sum(float(r[stat])>=level for r in concentrations) for level in [2,4]} for stat in ['average','max']}
    summary={'scope':'Public legacy six-test route; not the private seven-test 210-record research configuration.',
        'versions':{'slabx':'1.0.8 git source','slabx-lh2':'0.2.0 git source'},
        'FFI_pdf_sha256':hashlib.sha256((ROOT/'data/FFI/fetch_1.pdf').read_bytes()).hexdigest(),
        'report_rows':len(observations),'concentration_sensor_records':len(concentrations),
        'observed_threshold_positive_counts':distribution,
        'nominal_test_windows_s':{str(t): sorted({(r['window_start_s'],r['window_end_s']) for r in concentrations if int(r['test'])==t}) for t in range(1,8)},
        'by_RH':{str(rh):{'LFL_bracket_count':sum(r['verdict']=='in' for r in result if r['assumed_RH_pct']==rh),
                          'horizontal_c30_ratios':[r['c30_model_over_observed_max'] for r in result if r['assumed_RH_pct']==rh and r['orientation']=='horizontal']} for rh in [0,25,50,75,100]},
        'limitations':['No QES LH2 prediction has been validated on these same tests.',
          'Public legacy operator compares spatial centerline time-averaged concentrations with arc maxima of reported temporal maxima; not a same-statistic pointwise concentration validation.',
          'Public report does not give RH. Humidity sweep is sensitivity, not tuning or selection of a validated value.',
          'Experimental LFL brackets reflect sensor coverage and a summary window, not an exact continuous cloud boundary.',
          'Report PDF differs in byte hash from the author-audited copy; extracted counts and key Table A values were checked.'],
        'runs':result}
    (OUT/'public_ffi_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT/'public_ffi_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)

if __name__=='__main__':main()
