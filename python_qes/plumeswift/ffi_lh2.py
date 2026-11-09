"""Run the FFI open-field LH₂ concentration benchmark without external inputs."""

from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
from importlib.resources import files
from importlib.util import find_spec
import json
import math
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from . import __version__
from .assembly import assemble, AssemblyConfig, ComponentChoice
from .boundaries import AxisBoundary, DomainBoundaries
from .ffi_report import REPORT, REPORT_URL, read_ffi_report
from .interpolation import FieldSampler
from .io import write_arrays
from .lh2 import LH2Momentum
from .lh2_visual import GridAverage, visual_dependencies, write_figure
from .particles import ParticleIntegrator, ParticleSimulation
from .sources import ContinuousRelease, Point, Source
from .statistics import SensorBoxes, TimeAveragedSensors
from .walls import VoxelWalls
from .wind_case import read_wind_case


ORIGIN = (120., 120., .5)
DT = .2
SEED = 20261107
PARTICLES_PER_S = 100
NOZZLE_AREA = math.pi * (.0254 / 2)**2
LEVELS = (.5, 2., 4.)


def _resource(name):
    return files('plumeswift').joinpath('example_data', name).read_text(encoding='utf-8')


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _metrics(rows):
    observed = np.array([r['observed_volpct'] for r in rows])
    predicted = np.array([r['predicted_volpct'] for r in rows])
    error = predicted - observed
    actual4 = observed >= 4.
    return dict(sensor_count=len(rows), rmse_pp=float(np.sqrt(np.mean(error**2))),
                mae_pp=float(np.mean(np.abs(error))), bias_pp=float(np.mean(error)),
                measured_at_least_4=int(actual4.sum()),
                detected_at_least_4=int(np.count_nonzero(actual4 & (predicted >= 4.))))


def _write_sensors(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _simulate(cfg, wind_path, centres, *, render):
    """Keep the original benchmark wind, release, transport and receptor settings."""
    wind = read_wind_case(wind_path, poisson_backend='cpu', legacy_turbulence_xml=True,
                          native_sensor_math=True).run()
    if not wind['solver'].converged:
        raise RuntimeError(f'FFI Test {cfg["test"]}: wind solver did not converge')
    workspace = wind['workspace']
    compiled = find_spec('numba') is not None
    sampler = FieldSampler(workspace.coordinates, {k: workspace.arrays[k] for k in 'uvw'},
                           wind['turbulence'], mixlength=workspace.arrays['mixlength'],
                           compiled=compiled)
    solid = np.isin(workspace.arrays['icellflag'], (0, 2))
    boundaries = DomainBoundaries([AxisBoundary(axis[1], axis[-2])
                                   for axis in (sampler.xf, sampler.yf, sampler.zf)])
    model = assemble(AssemblyConfig(ComponentChoice('thermo.legacy', {
        'ambient_temperature': cfg['Ta'], 'liquid_fraction': cfg['liquid_fraction'],
        'relative_humidity': cfg['RH_assumed'], 'out_of_range': 'raise',
    }), ComponentChoice('poisson.krylov'), seed=SEED))
    speed = cfg['rate_kg_s'] / (70.8 * NOZZLE_AREA) if cfg['orientation'] == 'horizontal' else 0.
    source = Source(Point(ORIGIN), ContinuousRelease(cfg['rate_kg_s'], PARTICLES_PER_S),
                    initial_drift=(speed, 0., 0.))
    start, end = cfg['spinup_s'], cfg['sim_s']
    receptors = TimeAveragedSensors(SensorBoxes(centres), start, end, model.thermo)
    volume = float(sampler.dx * sampler.dy * np.diff(sampler.zf)[0])
    grid = GridAverage(sampler, solid.shape, model.thermo, volume, start, end) if render else None
    integrator = ParticleIntegrator(sampler, boundaries, VoxelWalls(sampler, solid, compiled=compiled),
                                    np.random.default_rng(SEED), model.config.numeric, compiled_gle=compiled)
    observers = [receptors, grid] if render else [receptors]
    run = ParticleSimulation([source], integrator, observers,
                             LH2Momentum(model.thermo, model.config.physics), volume)
    accounting = run.run(end, DT)
    prediction = 100 * receptors.mean()
    if (not np.isfinite(prediction).all()
            or not math.isclose(receptors.sampled_s, cfg['window_s'], abs_tol=1e-7)
            or not math.isclose(accounting['released_mass_kg'], cfg['rate_kg_s'] * end, rel_tol=1e-8)
            or abs(accounting['mass_balance_error_kg']) >= 1e-6):
        raise RuntimeError(f'FFI Test {cfg["test"]}: invalid sensor or mass accounting result')
    return prediction, accounting, sampler, solid, grid, compiled


def _run_case(cfg, *, output, render):
    started = perf_counter()
    test = cfg['test']
    folder = output / f'test{test}'
    folder.mkdir(exist_ok=True)
    _write_json(folder / 'case.json', cfg)
    winds = ET.fromstring(_resource('ffi_winds.xml'))
    series = winds.find('metParams/sensor/timeSeries')
    series.find('speed').text = str(cfg['wind_speed'])
    series.find('direction').text = str(cfg['wind_from'])
    wind_path = folder / 'winds.xml'
    ET.ElementTree(winds).write(wind_path, encoding='utf-8')

    # Match the eight significant figures of the archived benchmark receptors.
    centres = []
    for sensor in cfg['sensors']:
        radius, bearing = sensor['radius_m'], math.radians(sensor['bearing_deg'])
        centres.append([float(format(value, '.8g')) for value in (
            ORIGIN[0] + radius * math.sin(bearing), ORIGIN[1] + radius * math.cos(bearing),
            sensor['height_m'])])
    prediction, accounting, sampler, solid, grid, compiled = _simulate(
        cfg, wind_path, centres, render=render)
    rows = [dict(test=test, sensor=sensor['id'], x_m=x-ORIGIN[0], y_m=y-ORIGIN[1], z_m=z,
                 observed_volpct=obs, predicted_volpct=float(pred), error_pp=float(pred-obs),
                 observed_start_s=cfg['observed_window_s'][0], observed_end_s=cfg['observed_window_s'][1],
                 model_start_s=cfg['spinup_s'], model_end_s=cfg['sim_s'])
            for sensor, (x, y, z), pred in zip(cfg['sensors'], centres, prediction, strict=True)
            for obs in [sensor['observed_volpct']]]
    result = dict(test=test, **_metrics(rows), accounting=accounting, accelerated=compiled,
                  compute_s=perf_counter()-started, rendered_isosurfaces_volpct=[])
    _write_sensors(folder / 'sensor_comparison.csv', rows)
    if render:
        concentration = (100 * grid.integral[1:] / grid.sampled_s).astype(np.float32)
        concentration[solid[1:]] = np.nan
        coords = dict(x=sampler.x - ORIGIN[0], y=sampler.y - ORIGIN[1], z=sampler.z[1:])
        write_arrays(folder / 'concentration.npz',
                     {'h2_mean_volpct': concentration, 'solid': solid[1:]}, coords)
        title = f'PlumeSwift · FFI Test {test} · Mean H₂ concentration ({cfg["spinup_s"]:g}–{cfg["sim_s"]:g} s)'
        result['rendered_isosurfaces_volpct'] = write_figure(
            folder / 'plume.html', concentration, coords, LEVELS, (0., 0., ORIGIN[2]), title,
            sensors=rows)
    result['total_s'] = perf_counter()-started
    _write_json(folder / 'summary.json', result)
    return rows, result


def _write_index(output, summary, render):
    aggregate = summary['overall']
    cells = []
    for case in summary['cases']:
        test = case['test']
        plume = f'<a href="test{test}/plume.html">3D plume</a>' if render else '—'
        cells.append(f'<tr><td>{test}</td><td>{case["sensor_count"]}</td>'
                     f'<td>{case["rmse_pp"]:.4f}</td><td>{case["mae_pp"]:.4f}</td>'
                     f'<td>{case["bias_pp"]:+.4f}</td><td>{plume}</td>'
                     f'<td><a href="test{test}/sensor_comparison.csv">CSV</a></td></tr>')
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>PlumeSwift · FFI LH₂ results</title>
<style>body{{font:16px system-ui;max-width:52rem;margin:3rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{padding:.55rem;border-bottom:1px solid #ddd}}</style>
<h1>PlumeSwift · FFI LH₂ results</h1>
<p>Across {aggregate['sensor_count']} sensor records: RMSE {aggregate['rmse_pp']:.4f} pp,
MAE {aggregate['mae_pp']:.4f} pp, bias {aggregate['bias_pp']:+.4f} pp.</p>
<p>Bias is prediction minus measurement. The 3D surfaces use model grid means;
hover over a sensor to see its measured mean and the separate sensor-box prediction.</p>
<table><thead><tr><th>Test</th><th>Sensors</th><th>RMSE (pp)</th><th>MAE (pp)</th>
<th>Bias (pp)</th><th>View</th><th>Sensor data</th></tr></thead><tbody>
{''.join(cells)}</tbody></table>
<p><a href="sensor_comparison.csv">All selected sensors (CSV)</a> ·
<a href="summary.json">Metrics and run settings (JSON)</a></p></html>
"""
    (output / 'index.html').write_text(content, encoding='utf-8')


def run_ffi(tests, output, workers=1, *, report, render=True):
    """Run selected FFI cases against the user's local copy of the official report."""
    if workers < 1 or not tests or len(set(tests)) != len(tests) or not set(tests) <= set(range(1, 8)):
        raise ValueError('Choose distinct tests from 1 to 7 and a positive worker count')
    if render:
        visual_dependencies()
    started = perf_counter()
    observations = read_ffi_report(report, tests)
    selected = []
    for cfg in json.loads(_resource('ffi_cases.json')):
        test = cfg['test']
        if test in observations:
            window, sensors = observations[test]
            if window[1] - window[0] != cfg['window_s']:
                raise ValueError(f'FFI Test {test}: report window does not match the benchmark')
            selected.append(dict(cfg, observed_window_s=window, sensors=sensors))
    output.mkdir(parents=True, exist_ok=True)
    workers = min(workers, len(selected))
    run_case = partial(_run_case, output=output, render=render)
    if workers == 1:
        results = list(map(run_case, selected))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_case, selected))
    rows = [row for case_rows, _ in results for row in case_rows]
    summary = dict(version=__version__, source=REPORT, source_url=REPORT_URL,
                   seed=SEED, workers=workers, visualization=render, step_s=DT,
                   particles_per_s=PARTICLES_PER_S, cases=[result for _, result in results],
                   overall=_metrics(rows), wall_s=perf_counter()-started)
    _write_sensors(output / 'sensor_comparison.csv', rows)
    _write_json(output / 'summary.json', summary)
    _write_index(output, summary, render)
    return summary
