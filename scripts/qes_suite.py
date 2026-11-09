"""Run the actual upstream QES executables (not a substitute plume implementation)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'external/QES-Public'
CASES = ROOT/'data/cases'
OUT = ROOT/'results/qes'

def setval(root,path,value):
    e=root.find(path)
    if e is None:
        parent,_,tag=path.rpartition('/')
        e=ET.SubElement(root.find(parent) if parent else root,tag)
    e.text=str(value)

def plume(path, source=(20,50,15), duration=240, particles=100, bounds=(100,100,50)):
    tree=ET.parse(SRC/'testCases/EPA_7x11array/QES-files/EPA_7x11array_Plume.xml')
    r=tree.getroot()
    for key,v in {'simDur':duration,'timeStep':1,'CourantNumber':0.25,
                  'updateFrequency_timeLoop':60}.items(): setval(r,'plumeParameters/'+key,v)
    for axis,length,spacing in zip('XYZ',bounds,(2,2,1)):
        setval(r,'collectionParameters/boxBounds'+axis+'1',0)
        setval(r,'collectionParameters/boxBounds'+axis+'2',length)
        setval(r,'collectionParameters/nBoxes'+axis,int(length/spacing))
    setval(r,'collectionParameters/timeAvgStart',60)
    setval(r,'collectionParameters/timeAvgFreq',duration-60)
    old=r.find('sourceParameters')
    src=old.find('source')
    r.remove(old)
    pp=ET.SubElement(r,'particleParameters')
    base=ET.SubElement(pp,'baseParticle')
    ET.SubElement(base,'tag').text='passive_tracer'
    base.append(src)
    setval(src,'releaseType_continuous/particlePerTimestep',particles)
    setval(src,'releaseType_continuous/massPerSec',1.0)
    setval(src,'particleDensity',0.0)
    setval(src,'particleDiameter',0.0)
    for axis,value in zip('XYZ',source): setval(src,'sourceGeometry_Point/pos'+axis,value)
    tree.write(path,encoding='utf-8',xml_declaration=True)

def prepare():
    CASES.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,direction,speed,obstacle in [('empty_west',270,2,False),('block_west',270,2,True),
                                         ('block_south',180,2,True),('block_fast',270,5,True)]:
        folder=CASES/name; folder.mkdir(exist_ok=True)
        t=ET.parse(SRC/'testCases/EPA_7x11array/QES-files/EPA_7x11array_Winds.xml'); r=t.getroot()
        for path,val in {'simulationParameters/domain':'100 100 50',
                         'simulationParameters/cellSize':'1 1 1',
                         'simulationParameters/tolerance':'1e-6',
                         'simulationParameters/maxIterations':3000,
                         'metParams/sensor/site_xcoord':5,'metParams/sensor/site_ycoord':5,
                         'metParams/sensor/timeSeries/direction':direction,
                         'metParams/sensor/timeSeries/speed':speed,
                         'metParams/sensor/timeSeries/height':15,
                         'turbParams/method':0}.items(): setval(r,path,val)
        bp=r.find('buildingsParams')
        for tag in ['SHPFile','SHPBuildingLayer','SHPHeightField']:
            for e in list(bp.findall(tag)): bp.remove(e)
        setval(bp,'numBuildings',int(obstacle))
        if obstacle:
            b=ET.SubElement(bp,'rectangularBuilding')
            for key,val in {'groupID':1,'buildingType':1,'height':15,'baseHeight':0,
                            'xStart':40,'yStart':40,'length':15,'width':20,
                            'buildingRotation':0}.items(): ET.SubElement(b,key).text=str(val)
        t.write(folder/'winds.xml',encoding='utf-8',xml_declaration=True)
        plume(folder/'plume.xml',source=(25,50,2),duration=240)
    # Keep published EPA geometry, meteorology, and source; coarse mesh and shorter averaging for a pilot.
    folder=CASES/'epa_array'; folder.mkdir(exist_ok=True)
    t=ET.parse(SRC/'testCases/EPA_7x11array/QES-files/EPA_7x11array_Winds.xml'); r=t.getroot()
    for path,val in {'simulationParameters/domain':'200 200 60',
                     'simulationParameters/cellSize':'2 2 1',
                     'simulationParameters/tolerance':'1e-6',
                     'simulationParameters/maxIterations':3000,
                     'turbParams/method':1}.items(): setval(r,path,val)
    shutil.copytree(SRC/'testCases/EPA_7x11array/QES-files/shapefile',folder/'shapefile',dirs_exist_ok=True)
    t.write(folder/'winds.xml',encoding='utf-8',xml_declaration=True)
    plume(folder/'plume.xml',source=(66,200.5,1),duration=600,particles=200,bounds=(400,400,60))
    config={'model':'QES upstream 2.3.1', 'physics':'passive tracer; no LH2 thermodynamics or density feedback',
            'epa_changes':{'cell_m':[2,2,1],'duration_s':600,'averaging_s':540,'particles_per_s':200,
                           'mixing_length':'serial nearest wall (method 1), no OptiX'},
            'stochastic_note':'Upstream seeds RNG from wall clock. Plume runs are independent realizations.',
            'quic':{'status':'requires_licensed_distribution','official_guide':'https://cdn.lanl.gov/files/quic-startguide_7c182.pdf'}}
    (CASES/'configuration.json').write_text(json.dumps(config,indent=2))
    for option in [0,1]:
        variant=CASES/f'epa_canyon{option}'
        shutil.copytree(folder,variant,dirs_exist_ok=True)
        tree=ET.parse(variant/'winds.xml')
        setval(tree.getroot(),'buildingsParams/streetCanyonFlag',option)
        tree.write(variant/'winds.xml',encoding='utf-8',xml_declaration=True)
    sampling=CASES/'epa_sampling'
    shutil.copytree(CASES/'epa_canyon1',sampling,dirs_exist_ok=True)
    tree=ET.parse(sampling/'plume.xml')
    for path,value in {'collectionParameters/nBoxesZ':300,
                       'collectionParameters/timeAvgStart':300,
                       'collectionParameters/timeAvgFreq':900,
                       'plumeParameters/simDur':1200,
                       'plumeParameters/CourantNumber':.1}.items():setval(tree.getroot(),path,value)
    tree.write(sampling/'plume.xml',encoding='utf-8',xml_declaration=True)

def run_case(name,solver=3,with_plume=True):
    folder=CASES/name; output=OUT/f'{name}_s{solver}'; output.mkdir(exist_ok=True)
    cmd=['/opt/QES/build/qes/qes','-q',str(folder/'winds.xml'),'-s',str(solver),'-t','-w','-o',str(output/'run')]
    if with_plume: cmd+=['-p',str(folder/'plume.xml')]
    start=time.perf_counter()
    with (output/'run.log').open('w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800,env={**os.environ,'OMP_NUM_THREADS':'4'})
    result={'case':name,'solver':solver,'plume':with_plume,'returncode':p.returncode,
            'wall_s':time.perf_counter()-start,'command':cmd,'output':str(output)}
    (output/'execution.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

def main():
    a=argparse.ArgumentParser(); a.add_argument('--prepare',action='store_true'); a.add_argument('--case'); a.add_argument('--solver',type=int,default=3); a.add_argument('--wind-only',action='store_true'); args=a.parse_args()
    if args.prepare: prepare()
    if args.case:
        result=run_case(args.case,args.solver,not args.wind_only)
        if result['returncode']:raise SystemExit(result['returncode'])

if __name__=='__main__': main()
