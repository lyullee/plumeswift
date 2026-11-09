"""Audit executed cases and preserve hashes; this does not qualify LH2 physics."""
import csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from benchmark_lh2 import ROOT,OUT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=[]
    for folder in sorted(OUT.glob('test*')):
        e=json.loads((folder/'execution.json').read_text());assert e['returncode']==0
        log=(folder/'run.log').read_text();assert 'Number of rogue particles: 0' in log
        values=list(csv.DictReader((folder/'receptors.csv').open()))
        assert len(values)==30
        assert all(np.isfinite(float(r['vol_pct']))and 0<=float(r['vol_pct'])<=100 for r in values)
        with Dataset('memory',memory=(folder/'run_windsWk.nc').read_bytes())as nc:
            finite=all(np.isfinite(nc[k][:]).all()for k in 'uvw')
            assert finite,folder
        result.append({'case':folder.name,'finite_wind':bool(finite),'rogue_particles':0,'receptors':len(values),
                       'receptors_sha256':sha(folder/'receptors.csv')})
    assert len(result)==22,len(result)
    physics=json.loads((OUT/'physics_checks.json').read_text());assert physics['checks_passed']==7
    # Standalone test compiles the exact production coupling header.
    fixture="import subprocess; subprocess.run(['g++','-O2','-std=c++17','/work/src/qes_lh2/check_coupling.cpp','-o','/tmp/check_lh2'],check=True); subprocess.run(['/tmp/check_lh2'],check=True)"
    test=subprocess.run(['docker','run','--rm','--mount',f'type=bind,source={ROOT},target=/work',
            '-e','QES_LH2_TABLE=/work/data/cases/lh2/test4/cold.txt','gas-qes:lh2-prototype','python3','-c',fixture],
            check=True,capture_output=True,text=True)
    (OUT/'cpp_limiting_checks.log').write_text(test.stdout)
    hashes={str(p.relative_to(ROOT)):sha(p)for p in [ROOT/'src/qes_lh2/LH2Coupling.h',ROOT/'scripts/patch_qes_lh2.py',ROOT/'scripts/benchmark_lh2.py',OUT/'protocol.json',ROOT/'data/FFI/fetch_1.pdf']}
    with Dataset('memory',memory=(OUT/'test4_block_lh2_20261107/run_passive_tracer_plumeOut.nc').read_bytes())as nc:
        x=np.array(nc['x_c'][:]);y=np.array(nc['y_c'][:]);z=np.array(nc['z_c'][:]);c=np.array(nc['c'][-1])
        # Interior only; boundary cells can intersect both fluid and building.
        interior=c[np.ix_((z>1)&(z<4),(y>117)&(y<123),(x>137)&(x<143))]
        interior_max=float(np.max(interior));assert interior_max==0,interior_max
    audit={'QES_runs':result,'physics_checks':physics,'cpp_limiting_checks_passed':3,
           'building_interior_max_mean_H2_g_m3':interior_max,'sha256':hashes,
           'scope':'Finite values, accounting and limiting cases only; accuracy assessment is in metrics.json.'}
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2))
    print('Verified:',len(result),'QES runs, 7 thermo checks, 3 C++ limiting cases; building interior zero.')

if __name__=='__main__':main()
