"""Compare the archived reference and the nine controlled FLADIS runs."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fladis_force_study import CASES, OUTPUT
from run_fladis import ROOT, TRIALS, write_csv


REFERENCE = ROOT / 'mixing_thermo' / 'pps120' / 'loo_moist_midpoint'
LABELS = {
    'reference': 'Reference',
    'no_buoyancy': 'Zero vertical buoyancy',
    'more_particles': '240 particles/s',
    'coarse_buoyancy': '0.50 m buoyancy layers',
}
SENSOR_KEYS = ('nominal_arc_m', 'x_m', 'y_m', 'z_m', 'observed_ppmv')


def read(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def main(output=OUTPUT, labels=LABELS, cases=CASES, *, profile_errors=False):
    arcs, sensors, budgets = [], [], []
    for trial in TRIALS:
        stem = f'fladis{trial}'
        control = json.loads((REFERENCE / f'{stem}_run.json').read_text())
        control_sensors = read(REFERENCE / f'{stem}_sensors.csv')
        for case in labels:
            folder = REFERENCE if case == 'reference' else output / case
            run = json.loads((folder / f'{stem}_run.json').read_text())
            account = run['accounting']
            for key in control.keys() - {'particles_per_s', 'accounting', 'wall_s'}:
                assert run[key] == control[key], (trial, case, key)
            for key in ('relative_humidity_pct', 'gle_scheme', 'force_density_reconstruction',
                        'mast_station_xy_z_fromdeg_sigma', 'vertical_sigma_reference_m_s',
                        'direction_residual_std_deg', 'lateral_slope_gradient_per_m'):
                assert account[key] == control['accounting'][key], (trial, case, key)
            rate, buoyancy, coarse = cases.get(case, (120, True, False))
            assert run['particles_per_s'] == rate
            assert account['released_particles'] == rate * run['window']
            assert np.isclose(account['released_mass_kg'], run['rate'] * run['window'])
            assert abs(account['mass_balance_error_kg']) < 1e-6
            assert account.get('vertical_buoyancy', True) == buoyancy
            assert account.get('vertical_force_cell_height_m', .25) == (.5 if coarse else .25)
            budgets.append(dict(trial=trial, case=case, particles_per_s=rate,
                                vertical_buoyancy=buoyancy, coarse_vertical_grid=coarse,
                                released_mass_kg=account['released_mass_kg'],
                                mass_error_kg=account['mass_balance_error_kg'],
                                ground_heat_j=account['heat_j'], wall_s=run['wall_s']))
            for row in read(folder / f'{stem}_arcs.csv'):
                observed = float(row['observed_sensor_peak_at_height_ppmv'])
                predicted = float(row['modeled_sensor_peak_at_height_ppmv'])
                arcs.append(dict(case=case, **row,
                                 matched_peak_error_pct=100 * (predicted / observed - 1)))
            for expected, row in zip(control_sensors, read(folder / f'{stem}_sensors.csv'), strict=True):
                assert all(float(expected[key]) == float(row[key]) for key in SENSOR_KEYS)
                sensors.append(dict(case=case, **row))

    profiles = [row.copy() for row in sensors
                if int(row['nominal_arc_m']) == 70 and abs(float(row['y_m'])) < .01]
    assert (len(arcs), len(sensors), len(profiles)) == tuple(
        count * len(labels) for count in (9, 153, 12))
    for row in profiles:
        observed = float(row['observed_ppmv'])
        row['relative_error_pct'] = 100 * (float(row['model_ppmv']) / observed - 1) if observed else ''
    write_csv(output / 'arc_comparison.csv', arcs)
    write_csv(output / 'sensor_comparison.csv', sensors)
    write_csv(output / 'vertical_profiles.csv', profiles)
    write_csv(output / 'run_checks.csv', budgets)
    metrics = []
    for case in labels:
        for trial in ('all', *TRIALS):
            for distance in (20, 70, 240, '70+'):
                distances = (70, 240) if distance == '70+' else (distance,)
                rows = [r for r in sensors if r['case'] == case
                        and (trial == 'all' or int(r['trial']) == trial)
                        and int(r['nominal_arc_m']) in distances]
                observed = np.array([float(r['observed_ppmv']) for r in rows])
                predicted = np.array([float(r['model_ppmv']) for r in rows])
                metrics.append(dict(case=case, trial=trial, distance_m=distance, n=len(rows),
                                    mae_ppmv=np.mean(np.abs(predicted - observed)),
                                    normalized_absolute_error_pct=100 * np.abs(predicted - observed).sum() / observed.sum()))
    write_csv(output / 'metrics.csv', metrics)
    plot_profiles(profiles, output, labels, profile_errors)


def plot_profiles(profiles, output, labels, profile_errors):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, layout='constrained')
    for trial, ax in zip(TRIALS, axes, strict=True):
        for case, label in labels.items():
            rows = sorted((r for r in profiles if r['case'] == case and int(r['trial']) == trial),
                          key=lambda row: float(row['z_m']))
            height = [float(r['z_m']) for r in rows]
            if profile_errors:
                ax.plot([float(r['relative_error_pct']) for r in rows], height, 'o--', label=label)
            elif case == 'reference':
                ax.plot([float(r['observed_ppmv']) for r in rows], height, 'ko-', label='Observed')
            if not profile_errors:
                ax.plot([float(r['model_ppmv']) for r in rows], height, 'o--', label=label)
        ax.set_title(f'FLADIS {trial}, 70 m, y = 0 m')
        if profile_errors:
            ax.axvline(0, color='k', linewidth=.8)
            ax.set_xlabel('Concentration error (%)')
        else:
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlabel('Ammonia concentration (ppmv)')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Height (m)')
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'vertical_profiles.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
