"""Compare a mast-direction rerun with the previous FLADIS benchmark."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from run_fladis import ROOT, TRIALS, write_csv
from compare_fladis_sensitivity import cross_arc_centroid


OUT = ROOT/'mast_wind'
AFTER = OUT/'pps120'


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    before = {(int(row['trial']), int(row['nominal_arc_m'])): row
              for row in read(ROOT/'sensitivity'/'arc_sensitivity.csv')
              if row['particles_per_s']=='120' and row['mode']=='combined'}
    original = read(ROOT/'sensitivity'/'sensor_sensitivity.csv')
    results, paired, met = [], [], []
    for trial, cfg in TRIALS.items():
        run = json.loads((AFTER/f'fladis{trial}_run.json').read_text())
        budget = run['accounting']
        assert run['particles_per_s'] == 120 and run['dt_s'] == .25
        assert abs(budget['mass_balance_error_kg']) < 1e-6
        assert np.isclose(budget['released_mass_kg'], cfg['rate']*cfg['window'])
        axis = 110.
        x = np.array(budget['mast_station_x_m'])
        bearing = np.array(budget['mast_mean_direction_deg'])
        for x_m, y_m, z_m, mean, sigma in budget['mast_station_xy_z_fromdeg_sigma']:
            met.append(dict(trial=trial, mast_x_m=x_m, mast_y_m=y_m, height_m=z_m,
                            measured_from_deg=mean, measured_std_deg=sigma,
                            assigned_from_deg=float(np.interp(x_m,x,bearing)),
                            prior_from_deg=cfg['direction'], survey_axis_bearing_deg=axis))
        old = [r for r in original if int(r['trial'])==trial and
               r['particles_per_s']=='120' and r['mode']=='combined']
        new = read(AFTER/f'fladis{trial}_sensors.csv')
        for a,b in zip(old,new,strict=True):
            for key in ('trial','nominal_arc_m','x_m','y_m','z_m','observed_ppmv'):
                assert float(a[key])==float(b[key])
            paired.append(dict(trial=trial, arc_m=b['nominal_arc_m'], x_m=b['x_m'],
                               y_m=b['y_m'], z_m=b['z_m'], observed_ppmv=b['observed_ppmv'],
                               before_ppmv=a['model_ppmv'], after_ppmv=b['model_ppmv']))
        for row in read(AFTER/f'fladis{trial}_arcs.csv'):
            arc = int(row['nominal_arc_m'])
            a = before[trial,arc]
            height = float(row['height_m'])
            line = [r for r in new if int(r['nominal_arc_m'])==arc and float(r['z_m'])==height]
            obs_y = cross_arc_centroid(line,'observed_ppmv')
            old_y = float(a['modeled_sensor_centroid_y_m'])
            new_y = cross_arc_centroid(line,'model_ppmv')
            start = cfg['x0']+cfg['handoff']
            end = cfg['x0']+float(row['protocol_arc_m'])
            line_x = np.linspace(start,end,1001)
            old_drift = (end-start)*np.tan(np.deg2rad(axis+180.-cfg['direction']))
            new_drift = trapezoid(np.tan(np.deg2rad(axis+180.-np.interp(line_x,x,bearing))),line_x)
            results.append(dict(
                trial=trial, arc_m=arc, height_m=height, sensor_n=row['n'],
                observed_center_m=obs_y, before_center_m=old_y, after_center_m=new_y,
                geometric_drift_change_m=float(new_drift-old_drift),
                observed_width_m=row['published_width_sigma_y_m'],
                before_width_m=a['modeled_arc_width_m'], after_width_m=row['modeled_arc_width_m'],
                observed_matched_width_m=row['observed_matched_width_m'],
                before_matched_width_m=a['modeled_matched_width_m'],
                after_matched_width_m=row['modeled_matched_width_m'],
                observed_peak_ppmv=row['published_arcmax_ppmv'],
                before_peak_ppmv=a['modeled_arcmax_ppmv'], after_peak_ppmv=row['modeled_arcmax_ppmv'],
                flacs_D_peak_ppmv=a['flacs_cerc_D_ppmv'],
                flacs_min_peak_ppmv=a['flacs_cerc_min_ppmv'], flacs_max_peak_ppmv=a['flacs_cerc_max_ppmv'],
                observed_matched_peak_ppmv=row['observed_sensor_peak_at_height_ppmv'],
                before_matched_peak_ppmv=a['modeled_sensor_peak_at_height_ppmv'],
                after_matched_peak_ppmv=row['modeled_sensor_peak_at_height_ppmv'],
                before_sensor_mae_ppmv=a['mae'], after_sensor_mae_ppmv=row['mae']))
    assert len(results)==9 and len(paired)==153 and len(met)==12
    write_csv(OUT/'arc_comparison.csv',results)
    write_csv(OUT/'sensor_comparison.csv',paired)
    write_csv(OUT/'mast_inputs.csv',met)
    plot(results)


def plot(rows, *, output=OUT, labels=("Site mean", "Mast means")):
    fig, axes = plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for column,trial in enumerate(TRIALS):
        section = [r for r in rows if r['trial']==trial]
        x = [r['arc_m'] for r in section]
        for j,(key,unit) in enumerate((('center','m'),('width','m'),('peak','ppmv'))):
            ax = axes[j,column]
            for series,label,style in (('observed','Observed','ko-'),('before',labels[0],'x--'),
                                       ('after',labels[1],'s-')):
                ax.plot(x,[float(r[f'{series}_{key}_{unit}']) for r in section],style,label=label)
            if j==2:
                ax.plot(x,[float(r['flacs_D_peak_ppmv']) for r in section],'k^:',label='FLACS (D)')
                ax.set_yscale('log')
            if j==0:
                ax.set_title(f'FLADIS {trial}')
            if column==0:
                ax.set_ylabel(('Paired-sensor center (m)','Lateral width (m)','Arc maximum (ppmv)')[j])
            ax.set_xticks(x)
            ax.grid(alpha=.25)
    axes[2,0].legend(fontsize=8)
    fig.supxlabel('Nominal distance (m)')
    fig.savefig(output/'mast_comparison.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
