"""Summarize the controlled FLADIS simulations using the frozen benchmark sensors."""
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from run_fladis import ROOT, TRIALS, write_csv
from fladis_sensitivity import MODES, OUTPUT


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def cross_arc_centroid(rows, value):
    ordered = sorted(rows, key=lambda r: float(r['y_m']))
    y = np.array([float(r['y_m']) for r in ordered])
    c = np.array([float(r[value]) for r in ordered])
    integral = trapezoid(c, y)
    return float(trapezoid(c*y, y)/integral) if integral > 0. else None


def main(screening_only=False):
    archived = read(ROOT/'arc_comparison.csv')
    sensors = read(ROOT/'sensor_comparison.csv')
    index = {(int(r['trial']), int(r['nominal_arc_m'])): r for r in archived}
    results, paired, audits = [], [], []
    experiments = [(40, mode) for mode in MODES]
    if not screening_only:
        experiments.append((120, 'combined'))
    for trial in TRIALS:
        reference = [r for r in sensors if int(r['trial']) == trial]
        for rate, mode in experiments:
            location = OUTPUT/f'pps{rate}'/mode
            current = read(location/f'fladis{trial}_sensors.csv')
            sensor_key = ('trial', 'nominal_arc_m', 'x_m', 'y_m', 'z_m', 'observed_ppmv')
            for a, b in zip(reference, current, strict=True):
                assert all(a[k] == b[k] for k in sensor_key)
                paired.append(dict(**b, particles_per_s=rate, mode=mode, archived_model_ppmv=a['model_ppmv']))
            run = json.loads((location/f'fladis{trial}_run.json').read_text())
            accounting = run['accounting']
            assert abs(accounting['mass_balance_error_kg']) < 1e-6
            assert run['particles_per_s'] == rate
            assert (accounting['heat_j'] > 0) == (mode in ('heat', 'combined'))
            assert (accounting['direction_residual_std_deg'] > 0) == (mode in ('wind', 'combined'))
            audits.append(dict(trial=trial, particles_per_s=rate, mode=mode, wall_s=run['wall_s'], **accounting))
            for row in read(location/f'fladis{trial}_arcs.csv'):
                old = index[trial, int(row['nominal_arc_m'])]
                def at_height(sensor):
                    return (sensor['nominal_arc_m'] == row['nominal_arc_m']
                            and float(sensor['z_m']) == float(row['height_m']))
                measured_line = [r for r in reference if at_height(r)]
                model_line = [r for r in current if at_height(r)]
                results.append(dict(particles_per_s=rate, mode=mode, **row,
                    archived_arcmax_ppmv=old['modeled_arcmax_ppmv'],
                    archived_width_m=old['modeled_arc_width_m'],
                    archived_sensor_mae_ppmv=old['mae'],
                    observed_sensor_centroid_y_m=cross_arc_centroid(measured_line, 'observed_ppmv'),
                    archived_sensor_centroid_y_m=cross_arc_centroid(measured_line, 'model_ppmv'),
                    modeled_sensor_centroid_y_m=cross_arc_centroid(model_line, 'model_ppmv'),
                    flacs_cerc_D_ppmv=old['flacs_cerc_D_ppmv'],
                    flacs_cerc_min_ppmv=old['flacs_cerc_min_ppmv'],
                    flacs_cerc_max_ppmv=old['flacs_cerc_max_ppmv']))
    write_csv(OUTPUT/'arc_sensitivity.csv', results)
    write_csv(OUTPUT/'sensor_sensitivity.csv', paired)
    write_csv(OUTPUT/'run_audit.csv', audits)
    summarize(results, paired)
    plot(results)


def summarize(rows, paired):
    out = []
    experiments = [(40, mode) for mode in MODES]
    if any(r['particles_per_s'] == 120 for r in rows):
        experiments.extend(((120, 'archived'), (120, 'combined')))
    for rate, mode in experiments:
        current = [r for r in rows if r['particles_per_s'] == rate and
                   r['mode'] == ('combined' if mode == 'archived' else mode)]
        receptors = [r for r in paired if r['particles_per_s'] == rate and
                     r['mode'] == ('combined' if mode == 'archived' else mode)]
        peak = 'archived_arcmax_ppmv' if mode == 'archived' else 'modeled_arcmax_ppmv'
        width = 'archived_width_m' if mode == 'archived' else 'modeled_arc_width_m'
        estimate = 'archived_model_ppmv' if mode == 'archived' else 'model_ppmv'
        for scope, arcs in (('20 m', {20}), ('70–240 m', {70, 240}), ('all', {20, 70, 240})):
            chosen = [r for r in current if int(r['nominal_arc_m']) in arcs]
            sensors = [r for r in receptors if int(r['nominal_arc_m']) in arcs]
            concentration_ratio = np.array([float(r[peak])/float(r['published_arcmax_ppmv']) for r in chosen])
            observed_width = np.array([float(r['published_width_sigma_y_m']) for r in chosen])
            predicted_width = np.array([float(r[width]) for r in chosen])
            sensor_error = np.array([float(r[estimate])-float(r['observed_ppmv']) for r in sensors])
            out.append(dict(particles_per_s=rate, mode=mode, scope=scope, arcs=len(chosen),
                            sensors=len(sensors), sensor_mae_ppmv=float(np.mean(np.abs(sensor_error))),
                            peak_geometric_ratio=float(np.exp(np.mean(np.log(concentration_ratio)))),
                            peak_geometric_error_factor=float(np.exp(np.mean(np.abs(np.log(concentration_ratio))))),
                            width_mae_m=float(np.mean(np.abs(predicted_width-observed_width))),
                            width_bias_m=float(np.mean(predicted_width-observed_width))))
    write_csv(OUTPUT/'effect_summary.csv', out)


def plot(rows):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    styles = dict(control='0.55', heat='#bc5435', wind='#3580ba',
                  vertical='#985cac', combined='#007755')
    for column, trial in enumerate(TRIALS):
        data = [r for r in rows if int(r['trial']) == trial and r['particles_per_s']==40]
        for row_index, key, obs in (
            (0, 'modeled_arcmax_ppmv', 'published_arcmax_ppmv'),
            (1, 'modeled_arc_width_m', 'published_width_sigma_y_m'),
        ):
            axis = axes[row_index, column]
            base = [r for r in data if r['mode']=='control']
            x = [int(r['nominal_arc_m']) for r in base]
            axis.plot(x, [float(r[obs]) for r in base], 'ko-', label='Observed')
            if row_index == 0:
                axis.plot(x, [float(r['flacs_cerc_D_ppmv']) for r in base], 'k^:',
                          label='FLACS (D)')
                axis.set_yscale('log')
            full = [r for r in rows if int(r['trial']) == trial and r['particles_per_s']==120]
            old = 'archived_arcmax_ppmv' if row_index == 0 else 'archived_width_m'
            if full:
                axis.plot(x, [float(r[old]) for r in full], 'x--', color='0.4', label='Archived 120/s')
                axis.plot(x, [float(r[key]) for r in full], 'x--', color='#132057', label='Combined 120/s')
            for mode, color in styles.items():
                series = [float(r[key]) for r in data if r['mode']==mode]
                axis.plot(x, series, '.-', color=color, label=mode.capitalize())
            axis.set_xticks(x)
            axis.grid(alpha=.25)
            if row_index==0:
                axis.set_title(f'FLADIS {trial}')
    axes[0,0].set_ylabel('Arc maximum (ppmv)')
    axes[1,0].set_ylabel('Lateral width σᵧ (m)')
    axes[0,0].legend(fontsize=8)
    fig.supxlabel('Nominal distance (m)')
    fig.savefig(OUTPUT/'effect_comparison.png', dpi=165)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screening-only', action='store_true')
    main(parser.parse_args().screening_only)
