"""Compare frozen FLADIS baseline with independent mixing/EOS diagnostics."""
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_fladis_mast import read, cross_arc_centroid
from run_fladis import ROOT, TRIALS, write_csv

OUT=ROOT/'mixing_thermo'
MODES=('before','midpoint','moist','moist_midpoint','source_moist_midpoint','loo_moist_midpoint')


def main():
    reference={(int(r['trial']),int(r['arc_m'])):r for r in read(ROOT/'lateral_wind'/'arc_comparison.csv')}
    arcs, sensors, runs = [], [], []
    for mode in MODES:
        folder=ROOT/'lateral_wind'/'pps120' if mode=='before' else OUT/'pps120'/mode
        for trial,cfg in TRIALS.items():
            case=json.loads((folder/f'fladis{trial}_run.json').read_text())
            budget=case['accounting']
            assert abs(budget['mass_balance_error_kg'])<1e-6
            assert np.isclose(budget['released_mass_kg'],cfg['rate']*cfg['window'])
            assert case['particles_per_s']==120 and case['dt_s']==.25
            assert 0. <= budget['heat_j'] <= -case['source_enthalpy_j_kg']*budget['released_mass_kg']+1e-5
            runs.append(dict(trial=trial,mode=mode,wall_s=case['wall_s'],
                released_mass_kg=budget['released_mass_kg'],mass_balance_error_kg=budget['mass_balance_error_kg'],
                source_enthalpy_j_kg=case['source_enthalpy_j_kg'],heat_j=budget['heat_j']))
            points=read(folder/f'fladis{trial}_sensors.csv')
            old=read(ROOT/'lateral_wind'/'pps120'/f'fladis{trial}_sensors.csv')
            for a,b in zip(old,points,strict=True):
                for key in ('trial','nominal_arc_m','x_m','y_m','z_m','observed_ppmv'):
                    assert float(a[key])==float(b[key])
                sensors.append(dict(mode=mode,**b))
            for row in read(folder/f'fladis{trial}_arcs.csv'):
                arc=int(row['nominal_arc_m'])
                ref=reference[trial,arc]
                line=[r for r in points if int(r['nominal_arc_m'])==arc and
                      float(r['z_m'])==float(row['height_m'])]
                arcs.append(dict(mode=mode,trial=trial,arc_m=arc,
                    observed_peak_ppmv=float(ref['observed_peak_ppmv']),
                    model_peak_ppmv=float(row['modeled_arcmax_ppmv']),
                    flacs_D_peak_ppmv=float(ref['flacs_D_peak_ppmv']),
                    flacs_min_peak_ppmv=float(ref['flacs_min_peak_ppmv']),
                    flacs_max_peak_ppmv=float(ref['flacs_max_peak_ppmv']),
                    observed_center_m=float(ref['observed_center_m']),model_center_m=cross_arc_centroid(line,'model_ppmv'),
                    observed_width_m=float(ref['observed_width_m']),model_width_m=float(row['modeled_arc_width_m']),
                    observed_matched_width_m=float(row['observed_matched_width_m']),
                    model_matched_width_m=float(row['modeled_matched_width_m']),
                    sensor_mae_ppmv=float(row['mae'])))
    metrics=[]
    for mode in MODES:
        for scope in ('all','70_plus','20','70','240'):
            dist=(20,70,240) if scope=='all' else (70,240) if scope=='70_plus' else (int(scope),)
            a=[r for r in arcs if r['mode']==mode and r['arc_m'] in dist]
            p=[r for r in sensors if r['mode']==mode and int(r['nominal_arc_m']) in dist]
            obs=np.array([float(r['observed_ppmv']) for r in p])
            pred=np.array([float(r['model_ppmv']) for r in p])
            metrics.append(dict(mode=mode,scope=scope,n=len(p),sensor_mae_ppmv=float(np.mean(abs(pred-obs))),
                nmse=float(np.mean((pred-obs)**2)/(np.mean(pred)*np.mean(obs))),
                fractional_bias=float(2*(np.mean(obs)-np.mean(pred))/(np.mean(obs)+np.mean(pred))),
                fac2_positive_observations=float(np.mean((pred[obs>0]>=.5*obs[obs>0])&(pred[obs>0]<=2*obs[obs>0]))),
                width_mae_m=float(np.mean([abs(r['model_width_m']-r['observed_width_m']) for r in a])),
                center_mae_m=float(np.mean([abs(r['model_center_m']-r['observed_center_m']) for r in a])),
                peak_geometric_error_factor=float(np.exp(np.mean([abs(np.log(r['model_peak_ppmv']/r['observed_peak_ppmv'])) for r in a])))))
    for name,rows in (('arc_comparison',arcs),('sensor_comparison',sensors),('metrics',metrics),('runtime_and_budget',runs)):
        write_csv(OUT/f'{name}.csv',rows)
    vertical=[]
    for r in sensors:
        if int(r['nominal_arc_m'])==70 and abs(float(r['y_m']))<.01:
            vertical.append(r)
    write_csv(OUT/'vertical_tower_comparison.csv',vertical)
    plot(arcs,sensors)
    report(arcs,metrics,runs,vertical)
    print(json.dumps([r for r in metrics if r['scope'] in ('70_plus','all')],indent=2))


def report(arcs,metrics,runs,vertical):
    lines=['FLADIS 9/16/24 controlled diagnostic results -- 2026-11-11','',
        'See METHOD.txt for controlled inputs, source-state ambiguity and model limits.',
        f'{3*(len(MODES)-1)} new calculations + 3 archived reference cases in this report; no concentration-fitted constants.',
        'Source-consistent means matching the NOMINAL dry-derived T and 12 mol%, not a verified upstream enthalpy.',
        'Leave-one-out is an experimental equal-weight sampling correction; default remains disabled.','',
        'SENSOR MAE [ppmv]; all receptors including zero observations',
        'mode                         all        20 m        70 m       240 m      >=70 m']
    for mode in MODES:
        byscope={r['scope']:r for r in metrics if r['mode']==mode}
        lines.append(f'{mode:26s}'+''.join(f"{byscope[s]['sensor_mae_ppmv']:12.3f}" for s in ('all','20','70','240','70_plus')))
    lines+=['','ARC MAXIMA [ppmv] (published observation vs fixed-height model scan; see METHOD)',
            'trial arc observed FLACS_D '+' '.join(MODES)]
    for trial in TRIALS:
        for arc in (20,70,240):
            section={r['mode']:r for r in arcs if r['trial']==trial and r['arc_m']==arc}
            ref=section['before']
            lines.append(f"{trial:2d} {arc:3d} {ref['observed_peak_ppmv']:.3f} {ref['flacs_D_peak_ppmv']:.3f} "+
                         ' '.join(f"{section[m]['model_peak_ppmv']:.3f}" for m in MODES))
    lines+=['','70 m CENTRAL VERTICAL TOWER: observed, before, fixed-source-T moist/midpoint, also LOO [ppmv]']
    for trial in TRIALS:
        section=[r for r in vertical if int(r['trial'])==trial]
        for z in sorted({float(r['z_m']) for r in section}):
            p={r['mode']:r for r in section if float(r['z_m'])==z}
            lines.append(f"trial {trial} z={z:g} m: "+' '.join([
                f"observed={float(p['before']['observed_ppmv']):.3f}"]+
                [f"{m}={float(p[m]['model_ppmv']):.3f}" for m in ('before','source_moist_midpoint','loo_moist_midpoint')]))
    lines+=['',f'Checks: all {len(runs)} cases matched receptor coordinates, dt and release duration;',
        f"maximum absolute mass error {max(abs(r['mass_balance_error_kg']) for r in runs):.4g} kg;",
        'ground heat is nonnegative and below total injected enthalpy deficit in every case.',
        'See tests.log for conservation/limiting-case regressions.',
        'All timings in runtime_and_budget.csv were taken under varying concurrent load.',
        'No controlled runtime speedup or statistical confidence interval is claimed.',
        'Published FLACS provides arc maxima only, not paired receptor MAE, vertical profiles or widths.',
        'NH3-water solution VLE/excess heat, droplets and deposition remain outside the water-only closure.',
        'Marker-count/grid/time-step convergence is still required before adopting the new force estimator.']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(arcs,sensors):
    styles={'before':('Before','C0--'),'source_moist_midpoint':('Moist + midpoint, fixed source T','C1-.'),
            'loo_moist_midpoint':('Also exclude self-density (diagnostic)','C2-')}
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for column,trial in enumerate(TRIALS):
        for row,(key,unit) in enumerate((('peak','ppmv'),('width','m'),('center','m'))):
            ax=axes[row,column]
            section=[r for r in arcs if r['trial']==trial and r['mode']=='before']
            x=[r['arc_m'] for r in section]
            ax.plot(x,[r[f'observed_{key}_{unit}'] for r in section],'ko-',label='Observed')
            for mode,(label,style) in styles.items():
                current=[r for r in arcs if r['trial']==trial and r['mode']==mode]
                ax.plot(x,[r[f'model_{key}_{unit}'] for r in current],style,marker='s',label=label)
            if row==0:
                ax.plot(x,[r['flacs_D_peak_ppmv'] for r in section],'k^:',label='Published FLACS D')
                ax.set_yscale('log');ax.set_title(f'FLADIS {trial}')
            if column==0:
                ax.set_ylabel(('Arc maximum (ppmv)','Lateral width (m)','Paired-sensor center (m)')[row])
            ax.set_xticks(x);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel('Nominal distance (m)')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,trial in zip(axes,TRIALS,strict=True):
        for mode,(label,style) in styles.items():
            p=[r for r in sensors if int(r['trial'])==trial and r['mode']==mode
               and int(r['nominal_arc_m'])==70 and abs(float(r['y_m']))<.01]
            p.sort(key=lambda r:float(r['z_m']))
            ax.plot([float(r['model_ppmv']) for r in p],[float(r['z_m']) for r in p],style,marker='s',label=label)
            if mode=='before':
                ax.plot([float(r['observed_ppmv']) for r in p],[float(r['z_m']) for r in p],'ko-',label='Observed')
        ax.set_title(f'FLADIS {trial}: 70 m, y=0 m')
        ax.set_xscale('symlog',linthresh=1);ax.grid(alpha=.25);ax.set_xlabel('NH3 (ppmv)')
    axes[0].set_ylabel('Height (m)');axes[0].legend(fontsize=7)
    fig.savefig(OUT/'vertical_profiles.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    arguments=parser.parse_args()
    MODES=tuple(arguments.modes)
    if not {'before','source_moist_midpoint','loo_moist_midpoint'} <= set(MODES):
        parser.error('Plot/report require before and both source-temperature-controlled modes')
    main()
