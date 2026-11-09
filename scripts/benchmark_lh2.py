"""Matched public-code FFI benchmark. Run --prepare on host, --run in Docker.

Frozen BEFORE evaluation: RH75%, z0 .01m, neutral, 1m source drag length,
Cd .5, dt .2s, 2x2x.5m wind/force cells, 2x2x.2m receptor boxes.
This is a quasi-steady benchmark, not a reconstruction of unprovided raw
wind histories or valve/ignition timing. No fitted receptor parameters.
"""
import argparse,csv,json,math,os,subprocess,time,sys,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from qes_suite import setval,plume
CASES=ROOT/'data/cases/lh2'; OUT=ROOT/'results/lh2_benchmark'

def rows():
    obs=list(csv.DictReader((ROOT/'data/FFI/lh2_ffi_sensors.csv').open(encoding='utf-8')))
    return [r for r in obs if r['kind']=='concentration']

def thermotable(Ta,liquid,rh=75,warm=False):
    """Homogeneous equilibrium, fixed pressure, constant-cp enthalpy balance.

    No air condensation, substrate heat transfer, separate droplets or pool
    inventory. States below 80 K are diagnostic OUTSIDE the reliable closure.
    Trace moisture latent heat is included, water mass loading is neglected.
    """
    p=101325.;R=8.314462618;mh=.002016;ma=.028965;Tb=20.37;L=445000.
    def psat(T):
        t=T-273.15
        return 611.21*np.exp(np.clip((18.678-t/234.5)*t/(257.14+t),-100,30)) if T>173 else 0.
    pw=rh/100*psat(Ta);wa=.622*pw/(p-pw)
    Y=np.r_[0,np.geomspace(1e-9,.999999,10000),1.]
    T=np.full_like(Y,Ta);q=np.ones_like(Y)
    if not warm:
        for i,y in enumerate(Y[1:],1):
            def heat(t,quality=1):
                condensed=max(0.,wa-.622*psat(t)/max(p-psat(t),1))
                water_heat=condensed*(2500000+(333500 if t<273.15 else 0))
                return (1-y)*(1005*(t-Ta)-water_heat)+y*(14300*(t-Tb)+L*(quality-(1-liquid)))
            if heat(Tb)>0:
                T[i]=Tb;q[i]=np.clip((1-liquid)+(1-y)*(1005*(Ta-Tb)+wa*2833500)/(max(y,1e-15)*L),0,1)
            else:
                lo,hi=Tb,Ta
                for _ in range(45):
                    mid=(lo+hi)/2
                    if heat(mid)>0:hi=mid
                    else:lo=mid
                T[i]=(lo+hi)/2
    vg=Y*q*R*T/(p*mh)+(1-Y)*R*T/(p*ma)
    vl=Y*(1-q)/70.8
    rho=1/(vg+vl);c=Y*rho
    xv=np.divide(Y*q*R*T/(p*mh),vg,out=np.zeros_like(vg),where=vg>0)
    arr=np.c_[c,T,rho,xv,q]
    assert np.all(np.diff(c)>0), 'concentration inversion must be monotone'
    assert np.isfinite(arr).all()
    return arr

def prepare():
    from slabx_lh2.trials import FFI
    CASES.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    conditions=list(csv.DictReader((ROOT/'data/FFI/lh2_ffi_conditions.csv').open()))
    configs=[]
    for condition in conditions:
        t=int(condition['test']);obs=[r for r in rows() if int(r['test'])==t]
        d=CASES/f'test{t}';d.mkdir(exist_ok=True)
        Ta=float(condition['T_amb_C'])+273.15;rate=float(condition['outflow_kg_s'])
        liquid=FFI[t].liquid_fraction if t in FFI else .70 # predeclared interpolation, NOT fitted
        window=float(obs[0]['window_end_s'])-float(obs[0]['window_start_s'])
        # Preserve original site axes: x east, y north. Bearings are clockwise from north.
        receptors=np.array([[120+float(r['R_m'])*math.sin(math.radians(float(r['bearing_deg']))),
                              120+float(r['R_m'])*math.cos(math.radians(float(r['bearing_deg']))),float(r['z_m'])] for r in obs])
        np.savetxt(d/'receptors.txt',receptors,fmt='%.8g')
        for warm in [False,True]:
            np.savetxt(d/('warm.txt' if warm else 'cold.txt'),thermotable(Ta,liquid,warm=warm),fmt='%.10g')
        tree=ET.parse(ROOT/'data/cases/empty_west/winds.xml');r=tree.getroot()
        for path,val in {'simulationParameters/domain':'120 120 50','simulationParameters/cellSize':'2 2 .5',
            'metParams/sensor/timeSeries/height':10,'metParams/sensor/timeSeries/siteZ0':.01,
            'metParams/sensor/timeSeries/direction':condition['wind_dir_deg'],
            'metParams/sensor/timeSeries/speed':condition['u_mean_m_s'],
            'buildingsParams/streetCanyonFlag':1,'turbParams/method':1}.items():setval(r,path,val)
        tree.write(d/'winds.xml')
        plume(d/'plume.xml',source=(120,120,.5),duration=100+window,particles=100,bounds=(240,240,25))
        tree=ET.parse(d/'plume.xml');r=tree.getroot()
        for path,val in {'plumeParameters/timeStep':.2,'plumeParameters/CourantNumber':.25,
            'collectionParameters/timeAvgStart':100,'collectionParameters/timeAvgFreq':window,
            'particleParameters/baseParticle/source/releaseType_continuous/particlePerTimestep':20,
            'particleParameters/baseParticle/source/releaseType_continuous/massPerSec':rate*1000}.items():setval(r,path,val)
        tree.write(d/'plume.xml')
        cfg={'test':t,'rate_kg_s':rate,'Ta':Ta,'liquid_fraction':liquid,'wind_speed':float(condition['u_mean_m_s']),
             'wind_from':float(condition['wind_dir_deg']),'window_s':window,'sim_s':100+window,
             'orientation':condition['orientation'],'RH_assumed':75,'spinup_s':100}
        (d/'config.json').write_text(json.dumps(cfg,indent=2));configs.append(cfg)
    (OUT/'protocol.json').write_text(json.dumps({'frozen':True,'cases':configs,'drag_Cd':.5,'drag_length_m':1,
       'wind_grid_m':[2,2,.5],'receptor_box_m':[2,2,.2],'time_step_s':.2,'particles_s':100,
       'note':'Quasi-steady; actual wind and source time histories are not available. Nominal reported sensor averaging duration; no ignition simulated.'},indent=2))

def run(t,variant,seed=20261107,suffix=''):
    d=CASES/f'test{t}{suffix}';cfg=json.loads((d/'config.json').read_text())
    out=OUT/f'test{t}{suffix}_{variant}_{seed}';out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'4','QES_RNG_SEED':str(seed),'QES_LH2_TABLE':str(d/('warm.txt' if variant=='passive' else 'cold.txt')),
         'QES_LH2_FORCE':'0' if variant=='passive' else '1','QES_LH2_CELL_VOLUME':str(cfg.get('force_cell_volume',2)),'QES_LH2_DRAG_LENGTH':'1',
         'QES_RECEPTORS':str(d/'receptors.txt'),'QES_RECEPTOR_OUTPUT':str(out/'receptors.csv'),
         'QES_SAMPLE_START':'100','QES_SAMPLE_END':str(cfg['sim_s'])}
    # Incompressible liquid-nozzle momentum hypothesis; unresolved impingement removes
    # vertical source momentum. Horizontal nozzle points east (site release axis).
    u=cfg['rate_kg_s']/(70.8*math.pi*(.0254/2)**2)
    if cfg['orientation']=='horizontal' and variant!='passive':env['QES_LH2_UX']=str(u)
    cmd=['/opt/QES/build/qes/qes','-q',str(d/'winds.xml'),'-s','3','-t','-w','-p',str(d/'plume.xml'),'-o',str(out/'run')]
    ts=time.perf_counter()
    with (out/'run.log').open('w')as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    result={'test':t,'suffix':suffix,'variant':variant,'seed':seed,'wall_s':time.perf_counter()-ts,'returncode':p.returncode,'command':cmd}
    (out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if p.returncode:raise RuntimeError(result)

def extras():
    """Predetermined numerical sensitivity and independent geometry-function checks."""
    for suffix in ['_south','_halfwind','_block','_pps400','_dz025','_dt01']:
        source=CASES/'test4';d=CASES/f'test4{suffix}';shutil.copytree(source,d,dirs_exist_ok=True)
        cfg=json.loads((d/'config.json').read_text());w=ET.parse(d/'winds.xml');p=ET.parse(d/'plume.xml')
        if suffix=='_south':setval(w.getroot(),'metParams/sensor/timeSeries/direction',180)
        if suffix=='_halfwind':setval(w.getroot(),'metParams/sensor/timeSeries/speed',cfg['wind_speed']/2)
        if suffix=='_block':
            bp=w.getroot().find('buildingsParams');setval(bp,'numBuildings',1)
            b=ET.SubElement(bp,'rectangularBuilding')
            for k,v in {'groupID':1,'buildingType':1,'height':5,'baseHeight':0,'xStart':135,'yStart':115,'length':10,'width':10,'buildingRotation':0}.items():ET.SubElement(b,k).text=str(v)
        if suffix=='_pps400':setval(p.getroot(),'particleParameters/baseParticle/source/releaseType_continuous/particlePerTimestep',80)
        if suffix=='_dz025':
            setval(w.getroot(),'simulationParameters/domain','120 120 100');setval(w.getroot(),'simulationParameters/cellSize','2 2 .25');cfg['force_cell_volume']=1
        if suffix=='_dt01':
            setval(p.getroot(),'plumeParameters/timeStep',.1)
            setval(p.getroot(),'particleParameters/baseParticle/source/releaseType_continuous/particlePerTimestep',10)
        w.write(d/'winds.xml');p.write(d/'plume.xml');(d/'config.json').write_text(json.dumps(cfg,indent=2))
        run(4,'lh2',suffix=suffix)
    for seed in [20261108,20261109]:run(4,'lh2',seed=seed)

def slab():
    from slabx.core.plume import run_dispersion
    from slabx.core.source import HorizontalJet
    from slabx.post.concentration import concentration_field
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.coolprop import CoolPropThermo,coolprop_water
    from slabx_lh2.plume_width import plume_width_coupling
    from slabx_lh2.water_ice import with_sublimation
    from review_slabx_lh2 import H2
    results=[]
    for t in range(1,8):
        d=CASES/f'test{t}';cfg=json.loads((d/'config.json').read_text());obs=[r for r in rows() if int(r['test'])==t]
        ts=time.perf_counter()
        atm=Atmosphere(u_ref=cfg['wind_speed'],z_ref=10,T=cfg['Ta'],rh=75,z0=.01,stability='D')
        src=HorizontalJet(substance=H2,rate=cfg['rate_kg_s'],area=math.pi*(.0254/2)**2,duration=cfg['sim_s'],
                          liquid_fraction=cfg['liquid_fraction'],height=.5,T_source=20.37)
        with plume_width_coupling():
            traj,_=run_dispersion(src,atm,CoolPropThermo(H2,fluid='Hydrogen'),with_sublimation(coolprop_water()),x_max=160,n_puff_steps=40)
        pred=[]
        # Volume-average the same 2x2x.2 m boxes. Native SLAB coordinates are downwind/crosswind.
        a=math.radians((cfg['wind_from']+180)%360);ux,uy=math.sin(a),math.cos(a)
        for row in obs:
            b=math.radians(float(row['bearing_deg']));R=float(row['R_m']);east,north=R*math.sin(b),R*math.cos(b)
            values=[]
            for ez in [-.0666667,0,.0666667]:
                field=concentration_field(traj,atm,z=float(row['z_m'])+ez,t_avg=cfg['window_s'],t_release=cfg['sim_s'])
                for ex in [-2/3,0,2/3]:
                    for ey in [-2/3,0,2/3]:
                        x=(east+ex)*ux+(north+ey)*uy;y=(east+ex)*uy-(north+ey)*ux
                        values.append(float(np.interp(x,field.x,field.at_y(y),left=0,right=0))*100)
            pred.append(float(np.mean(values)))
        result={'test':t,'wall_s':time.perf_counter()-ts,'pred_volpct':pred,'scope':'public 1.0.8 / 0.2.0; matched legacy source, not private research route'}
        results.append(result);print('SLAB',t,result['wall_s'],flush=True)
    (OUT/'slab.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--slab',action='store_true');p.add_argument('--run',type=int);p.add_argument('--suite',action='store_true');p.add_argument('--extras',action='store_true');p.add_argument('--variant',default='lh2');p.add_argument('--seed',type=int,default=20261107);a=p.parse_args()
    if a.prepare:prepare()
    if a.slab:slab()
    if a.run:run(a.run,a.variant,a.seed)
    if a.extras:extras()
    if a.suite:
        for t in range(1,8):
            for variant in ['passive','lh2']:
                if not (OUT/f'test{t}_{variant}_{a.seed}'/'execution.json').exists():run(t,variant,a.seed)
