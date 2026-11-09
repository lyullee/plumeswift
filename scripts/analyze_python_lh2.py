"""Score fresh FFI predictions and matched published CFD probes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version

import numpy as np

from analyze_lh2_benchmark import metrics
from benchmark_lh2 import CASES, rows as observation_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/python_lh2_validation'
METHODS = ('qes', 'slab')
CFD = ('pseudo', 'hybrid', 'pool')


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def write_rows(path, records):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def score(records, models):
    observed = [float(row['observed_volpct']) for row in records]
    return {name: metrics(observed, [float(row[name]) for row in records]) for name in models}


def timing(execution):
    wall = np.array([row['wall_s'] for row in execution])
    return dict(mean_s=float(wall.mean()), median_s=float(np.median(wall)),
                min_s=float(wall.min()), max_s=float(wall.max()))


def slab_revisions():
    return {name: subprocess.check_output(['git', '-C', str(ROOT/'external'/name), 'rev-parse', 'HEAD'],
                                         text=True).strip() for name in ('slabx', 'slabx-lh2')}


def input_hashes(tests):
    paths = [ROOT/'data/FFI/lh2_ffi_sensors.csv', ROOT/'data/FFI/lh2_ffi_conditions.csv',
             ROOT/'results/cfd_comparison/matched_sensor_values.csv']
    paths.extend(CASES/f'test{test}'/filename for test in tests
                 for filename in ('winds.xml', 'config.json', 'receptors.txt'))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def clustered_rmse_difference(records):
    tests = sorted({int(row['test']) for row in records})
    error = np.array([[[float(row[model])-float(row['observed_volpct'])
                        for row in records if int(row['test']) == test] for test in tests] for model in METHODS])
    selection = np.random.default_rng(20261109).integers(len(tests), size=(10_000, len(tests)))
    draws = np.sqrt(np.mean((error**2)[:, selection, :], axis=(2, 3)))
    return np.quantile(draws[0]-draws[1], (.025, .975)).tolist()


def read_cfd(records):
    by_sensor = {(int(row['test']), row['sensor']): row for row in records}
    table = read_rows(ROOT/'results/cfd_comparison/matched_sensor_values.csv')
    matched = []
    for row in table:
        key = int(row['test']), row['sensor']
        if key not in by_sensor:
            continue
        current = by_sensor[key]
        assert float(row['average']) == float(current['observed_volpct'])
        assert all(float(row[name]) == float(current[name]) for name in ('R_m', 'bearing_deg', 'z_m'))
        match = current | {'source_figure': row['source_figure']}
        for mode in CFD:
            match.update({mode: float(row[mode+'_volpct']),
                          mode+'_pixel_interval_volpct': float(row[mode+'_uncertainty_volpct']),
                          mode+'_occluded': row[mode+'_occluded'] == 'True'})
        matched.append(match)
    return matched


def cfd_report(records):
    truth = np.array([float(row['observed_volpct']) for row in records])
    scores = {}
    disagreement = {}
    for mode in CFD:
        reference = np.array([row[mode] for row in records])
        error = np.array([row[mode+'_pixel_interval_volpct'] for row in records])
        low, high = np.maximum(0., reference-error), reference+error
        scores[mode] = {
            'pixel_interval_RMSE_min': float(np.sqrt(np.mean(np.maximum(np.maximum(low-truth, truth-high), 0.)**2))),
            'pixel_interval_RMSE_max': float(np.sqrt(np.mean(np.maximum(abs(low-truth), abs(high-truth))**2))),
            'occluded_count': sum(row[mode+'_occluded'] for row in records),
        }
        disagreement[mode] = {model: metrics(reference, [row[model] for row in records])
                              for model in METHODS}
    return dict(all=score(records, METHODS+CFD), by_test={
        str(test): score([row for row in records if int(row['test']) == test], METHODS+CFD)
        for test in sorted({int(row['test']) for row in records})},
        pixel_intervals=scores, model_to_cfd_disagreement=disagreement,
        reference='Choi et al. 2023, Figures 6 and 8. Approximate digitized probe means.',
        source='https://www.mdpi.com/2076-3417/13/23/12838',
        reported_CFD_wall_hours=dict(chosen_grid=76, all_grids=[48, 132], cells=713934, simulated_s=800,
                                     reported_cores=40, cpu='Intel Xeon Silver 4214'),
        limits=['The FLACS raw workspace and case setup were not obtained. Published point probes '
                'are compared with box averages.',
                'The paper used dry air, a container and three distinct source models. The shared FFI '
                'baseline uses RH 75% and open terrain.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', nargs='+', type=int, default=list(range(1, 8)))
    parser.add_argument('--thermo', default='legacy')
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    names = {'qes': f'qes_{args.thermo}_{args.seed}', 'slab': 'slab'}
    records, by_test = [], []
    execution = {model: [] for model in METHODS}
    observation = {(row['test'], row['sensor']): row for row in observation_rows()}
    for test in sorted(args.tests):
        values = {}
        for model, name in names.items():
            path = OUT/f'test{test}_{name}'
            result = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            execution[model].append(result)
            values[model] = read_rows(path.with_suffix('.csv'))
        qes_input, slab_input = (execution[model][-1]['configuration'] for model in METHODS)
        assert qes_input == slab_input
        batch = []
        for qes, slab in zip(values['qes'], values['slab'], strict=True):
            keys = ('test', 'sensor', 'observed_volpct')
            assert all(qes[key] == slab[key] for key in keys)
            original = observation[qes['test'], qes['sensor']]
            assert float(qes['observed_volpct']) == float(original['average'])
            locations = ('R_m', 'bearing_deg', 'z_m', 'window_start_s', 'window_end_s')
            batch.append({key: qes[key] for key in keys}
                         | {key: original[key] for key in locations}
                         | dict(qes=float(qes['predicted_volpct']), slab=float(slab['predicted_volpct'])))
        assert len(batch) == 30
        records.extend(batch)
        by_test.append(dict(test=test, scores=score(batch, METHODS),
                            timing_s={name: execution[name][-1]['wall_s'] for name in METHODS}))
    report = dict(tests=sorted(args.tests), seed=args.seed, thermo=args.thermo,
                  host=dict(platform=platform.platform(), python=platform.python_version(),
                            packages={name: version(name) for name in
                                      ('plumeswift-research', 'numpy', 'scipy', 'CoolProp', 'slabx', 'slabx-lh2')}),
                  all=score(records, METHODS), by_test=by_test,
                  timing_s={model: timing(execution[model]) for model in METHODS},
                  clustered_rmse_diff_interval=clustered_rmse_difference(records),
                  max_qes_mass_balance_error_kg=max(abs(row['accounting']['mass_balance_error_kg'])
                                                   for row in execution['qes']),
                  slab_git=slab_revisions(), input_sha256=input_hashes(args.tests),
                  protocol=dict(wind_reference_height_m=10, neutral=True, roughness_length_m=.01,
                                receptor_box_m=[2, 2, .2], spinup_s=100, qes_wind_cell_m=[2, 2, .5],
                                qes_dt_s=.2, qes_particles_per_s=100,
                                both_geometry='Open site, shared ambient and nominal time-averaging durations',
                                timing='Wall clock on one host. QES includes wind, turbulence and particle integration. '
                                       'SLAB includes properties, trajectory and sensor averaging, median of three runs. '
                                       'Result file writes and program startup are excluded.'))
    quality_audit = OUT/'preslhy_trial10_data_audit.json'
    if quality_audit.is_file():
        report['preslhy_trial10_audit'] = json.loads(quality_audit.read_text(encoding='utf-8'))
    failure = OUT/f'test3_qes_coolprop_{args.seed}_failure.json'
    if failure.is_file():
        report['coolprop_test3_failure'] = json.loads(failure.read_text(encoding='utf-8'))
    unignited = [row for row in records if int(row['test']) not in (5, 6)]
    if unignited:
        report['unignited'] = score(unignited, METHODS)
    matched = read_cfd(records)
    if matched:
        report['published_cfd'] = cfd_report(matched)
        write_rows(OUT/'matched_cfd.csv', matched)
        layout_predictions = {}
        for test in (1, 3):
            paths = {model: OUT/f'test{test}_{name}_cfd_layout.csv' for model, name in names.items()}
            if all(path.is_file() for path in paths.values()):
                layout_predictions[str(test)] = {
                    model: {value['sensor']: value for value in read_rows(path)} for model, path in paths.items()}
        layout = []
        for row in matched:
            if row['test'] in layout_predictions:
                values = {model: index[row['sensor']] for model, index in layout_predictions[row['test']].items()}
                assert all(float(value['observed_volpct']) == float(row['observed_volpct'])
                           for value in values.values())
                layout.append(row | {model: float(value['predicted_volpct']) for model, value in values.items()})
        if layout:
            sensitivity = cfd_report(layout)
            sensitivity['limits'][-1] = ('Both reruns use dry air. QES includes a 13 × 2.5 × 5.2 m cut-cell '
                                        'container placed from the schematic. SLAB has no explicit obstacle. '
                                        'The FLACS source models and grid were not reproduced.')
            report['published_cfd_sensitivity'] = sensitivity
            write_rows(OUT/'matched_cfd_layout.csv', layout)
    write_rows(OUT/'sensor_predictions.csv', records)
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
