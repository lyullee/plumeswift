"""Run the FFI open-field LH₂ concentration benchmark without external inputs."""

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from functools import partial
from importlib.resources import files
from importlib.util import find_spec
import json
import math
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from . import __version__
from .boundaries import AxisBoundary, DomainBoundaries
from .ffi_report import REPORT, REPORT_URL, read_ffi_report
from .ffi_weather import read_weather
from .interpolation import FieldSampler
from .io import write_arrays
from .lh2 import LH2Momentum
from .lh2_case import prepare_cryogenic
from .lh2_visual import GridAverage, visual_dependencies, write_figure
from .particles import ParticleIntegrator, ParticleSimulation
from .sources import ContinuousRelease, Point, Source
from .statistics import SensorBoxes, TimeAveragedSensors
from .thermo import LegacyEquilibrium
from .walls import VoxelWalls
from .wind_case import read_wind_case


ORIGIN = (120., 120., .5)
DT = .2
SEED = 20261107
PARTICLES_PER_S = 100
NOZZLE_AREA = math.pi * (.0254 / 2)**2
LEVELS = (.5, 2., 4.)
# FFI-RAPPORT 20/03101, Tables 2.2–2.8. Speed is m/s, bearing is degrees.
WIND_SPREAD = ((.8, 14.), (.8, 10.), (1.8, 11.), (1.6, 10.), (1.9, 12.), (.9, 15.), (1.4, 11.))
# DNV 853182 rev. 2 §4.1 gives the dimensions. The center is estimated
# from FFI Figure 2.7 using the pad and 30 m arc as the drawing scale.
ISO_CONTAINER = dict(east_m=3.5, north_m=11., length_m=12.18, width_m=2.43, height_m=5.14)


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
                detected_at_least_4=int(np.count_nonzero(actual4 & (predicted >= 4.))),
                false_positive_at_least_4=int(np.count_nonzero(~actual4 & (predicted >= 4.))))


def _write_sensors(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cryogenic_nozzle(cfg):
    horizontal = cfg['orientation'] == 'horizontal'
    # Giannissi et al. (ICHS 2023), sections 2.1 and 3.1, table 2.
    # The exit quality is a published CFD boundary condition for tests 5/6.
    # For the other cases the archived phase inputs are explicit assumptions.
    documented_quality = cfg['test'] in (5, 6)
    position = (*ORIGIN[:2], .5 if horizontal else .32)
    jet = dict(direction=(1., 0., 0.) if horizontal else (0., 0., -1.),
               nozzle_diameter_m=.0254, handoff_distance_m=.5,
               exit_vapor_mass_fraction=.105 if documented_quality else 1. - cfg['liquid_fraction'])
    basis = 'ICHS 2023 ADREA-HF inlet (Table 2)' if documented_quality else 'archived phase assumption'
    return position, jet, basis


def _wind_frame(case, test, compiled):
    wind = case.run()
    if not wind['solver'].converged:
        raise RuntimeError(f'FFI Test {test}: wind solver did not converge')
    workspace = wind['workspace']
    return (FieldSampler(workspace.coordinates, {k: workspace.arrays[k] for k in 'uvw'},
                         wind['turbulence'], mixlength=workspace.arrays['mixlength'], compiled=compiled),
            np.isin(workspace.arrays['icellflag'], (0, 2)))


def _simulate(cfg, wind_path, centres, *, render, physics='legacy', cryogenic_options=None):
    wind = read_wind_case(wind_path, poisson_backend='cpu', legacy_turbulence_xml=True, native_sensor_math=True)
    frames = cfg.get('measured_wind', [])
    compiled = find_spec('numba') is not None

    def load_frame(frame):
        if frame is None:
            return _wind_frame(wind, cfg['test'], compiled)
        height = cfg['wind_reference_height_m']
        speed, direction = frame[2:4] if height == 10 else frame[4:6]
        profile = replace(wind.profile, speed=speed, direction=direction, reference_height=height)
        station, = wind.stations
        return _wind_frame(replace(wind, profile=profile, stations=(replace(station, profile=profile),)),
                           cfg['test'], compiled)

    sampler, solid = load_frame(frames[0] if frames else None)
    boundaries = DomainBoundaries([AxisBoundary(axis[1], axis[-2])
                                   for axis in (sampler.xf, sampler.yf, sampler.zf)])
    cryogenic = None
    horizontal = cfg['orientation'] == 'horizontal'
    if physics == 'legacy':
        thermo = LegacyEquilibrium(ambient_temperature=cfg['Ta'], liquid_fraction=cfg['liquid_fraction'],
                                   relative_humidity=cfg['RH_assumed'], out_of_range='raise')
        speed = cfg['rate_kg_s'] / (70.8 * NOZZLE_AREA) if horizontal else 0.
        sources = [Source(Point(ORIGIN), ContinuousRelease(cfg['rate_kg_s'], PARTICLES_PER_S),
                          initial_drift=(speed, 0., 0.))]
        momentum = LH2Momentum(thermo)
    else:
        position, jet, basis = _cryogenic_nozzle(cfg)
        options = {'jet': jet}
        if not horizontal and physics in ('pool', 'dense'):
            options['pool'] = dict(radius_m=.75, ground_temperature_k=cfg['Ta'], ground_effusivity=1800.)
        if physics == 'dense':
            options['dense_layer'] = dict(depth_m=1.)
        if cryogenic_options:
            options.update(cryogenic_options)
            if not horizontal:
                options.pop('crossflow_jet', None)
            if 'ground_heat' in options:
                options['ground_heat'] = dict(options['ground_heat'], ground_temperature_k=cfg['Ta'])
        cryogenic = prepare_cryogenic(options, cfg['Ta'], position, cfg['rate_kg_s'], PARTICLES_PER_S,
                                      sampler, solid, relative_humidity_pct=cfg['RH_assumed'])
        thermo, sources, momentum = cryogenic.thermo, cryogenic.sources, cryogenic.momentum
    start, end = cfg['spinup_s'], cfg['sim_s']
    receptors = TimeAveragedSensors(SensorBoxes(centres), start, end, thermo)
    volume = float(sampler.dx * sampler.dy * np.diff(sampler.zf)[0])
    grid = GridAverage(sampler, solid.shape, thermo, volume, start, end) if render else None
    integrator = ParticleIntegrator(sampler, boundaries, VoxelWalls(sampler, solid, compiled=compiled),
                                    np.random.default_rng(SEED), compiled_gle=compiled)
    observers = [receptors, grid] if render else [receptors]
    run = ParticleSimulation(sources, integrator, observers, momentum, volume, track_enthalpy=cryogenic is not None,
                             track_carrier_air=cryogenic is not None and cryogenic.track_carrier_air,
                             density_entrainment=None if cryogenic is None else cryogenic.entrainment)
    if frames:
        for i, frame in enumerate(frames):
            if i:
                next_sampler, _ = load_frame(frame)
                run.integrator.fields = next_sampler
                if cryogenic is not None:
                    cryogenic.refresh_crossflow(next_sampler, solid)
            accounting = run.run(frame[1] - cfg['release_clock_origin_s'], DT)
    else:
        accounting = run.run(end, DT)
    prediction = 100 * receptors.mean()
    if (not np.isfinite(prediction).all()
            or not math.isclose(receptors.sampled_s, cfg['window_s'], abs_tol=1e-7)
            or (cryogenic is None and not math.isclose(accounting['released_mass_kg'], cfg['rate_kg_s'] * end, rel_tol=1e-8))
            or abs(accounting['mass_balance_error_kg']) >= 1e-6):
        raise RuntimeError(f'FFI Test {cfg["test"]}: invalid sensor or mass accounting result')
    diagnostics = None if cryogenic is None else cryogenic.diagnostics(end, accounting)
    if diagnostics is not None:
        diagnostics['nozzle'] = dict(position_m=position, phase_basis=basis, **jet)
    return prediction, accounting, sampler, solid, grid, compiled, diagnostics


def _run_case(cfg, *, output, render, physics='legacy', cryogenic_options=None, wind_roughness=None):
    started = perf_counter()
    test = cfg['test']
    folder = output / f'test{test}'
    folder.mkdir(exist_ok=True)
    _write_json(folder / 'case.json', cfg)
    winds = ET.fromstring(_resource('ffi_winds.xml'))
    series = winds.find('metParams/sensor/timeSeries')
    series.find('speed').text = str(cfg['wind_speed'])
    series.find('direction').text = str(cfg['wind_from'])
    if wind_roughness is not None:
        series.find('siteZ0').text = str(wind_roughness)
    if 'site_container' in cfg:
        geometry = cfg['site_container']
        buildings = winds.find('buildingsParams')
        buildings.find('numBuildings').text = '1'
        rectangle = ET.SubElement(buildings, 'rectangularBuilding')
        fields = dict(xStart=ORIGIN[0] + geometry['east_m'] - geometry['length_m'] / 2,
                      yStart=ORIGIN[1] + geometry['north_m'] - geometry['width_m'] / 2,
                      length=geometry['length_m'], width=geometry['width_m'], height=geometry['height_m'],
                      buildingRotation=0., baseHeight=0.)
        for key, value in fields.items():
            ET.SubElement(rectangle, key).text = str(value)
    wind_path = folder / 'winds.xml'
    ET.ElementTree(winds).write(wind_path, encoding='utf-8')

    # Match the eight significant figures of the archived benchmark receptors.
    centres = []
    for sensor in cfg['sensors']:
        radius, bearing = sensor['radius_m'], math.radians(sensor['bearing_deg'])
        centres.append([float(format(value, '.8g')) for value in (
            ORIGIN[0] + radius * math.sin(bearing), ORIGIN[1] + radius * math.cos(bearing),
            sensor['height_m'])])
    prediction, accounting, sampler, solid, grid, compiled, diagnostics = _simulate(
        cfg, wind_path, centres, render=render, physics=physics, cryogenic_options=cryogenic_options)
    rows = [dict(test=test, sensor=sensor['id'], x_m=x-ORIGIN[0], y_m=y-ORIGIN[1], z_m=z,
                 observed_volpct=obs, predicted_volpct=float(pred), error_pp=float(pred-obs),
                 observed_start_s=cfg['observed_window_s'][0], observed_end_s=cfg['observed_window_s'][1],
                 model_start_s=cfg['spinup_s'], model_end_s=cfg['sim_s'])
            for sensor, (x, y, z), pred in zip(cfg['sensors'], centres, prediction, strict=True)
            for obs in [sensor['observed_volpct']]]
    result = dict(test=test, **_metrics(rows), accounting=accounting, accelerated=compiled,
                  compute_s=perf_counter()-started, rendered_isosurfaces_volpct=[])
    if diagnostics is not None:
        result['cryogenic'] = diagnostics
    _write_sensors(folder / 'sensor_comparison.csv', rows)
    if render:
        concentration = (100 * grid.integral[1:] / grid.sampled_s).astype(np.float32)
        concentration[solid[1:]] = np.nan
        coords = dict(x=sampler.x - ORIGIN[0], y=sampler.y - ORIGIN[1], z=sampler.z[1:])
        write_arrays(folder / 'concentration.npz',
                     {'h2_mean_volpct': concentration, 'solid': solid[1:]}, coords)
        title = f'PlumeSwift · FFI Test {test} · Mean H₂ concentration ({cfg["spinup_s"]:g}–{cfg["sim_s"]:g} s)'
        nozzle_height = ORIGIN[2] if diagnostics is None else diagnostics['nozzle']['position_m'][2]
        result['rendered_isosurfaces_volpct'] = write_figure(
            folder / 'plume.html', concentration, coords, LEVELS, (0., 0., nozzle_height), title,
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


def _weather_members(cfg, *, joint=False):
    """Match the reported means and variances with steady wind states."""
    speed, direction = cfg['wind_speed'], cfg['wind_from']
    speed_sd, direction_sd = WIND_SPREAD[cfg['test'] - 1]
    a, b = math.sqrt(3) * speed_sd, math.sqrt(3) * direction_sd
    axis_weight = 1 / 9 if joint else 1 / 6
    members = [('mean', 4 / 9 if joint else 1 / 3, cfg),
               ('speed_low', axis_weight, dict(cfg, wind_speed=speed - a)),
               ('speed_high', axis_weight, dict(cfg, wind_speed=speed + a)),
               ('bearing_low', axis_weight, dict(cfg, wind_from=(direction - b) % 360)),
               ('bearing_high', axis_weight, dict(cfg, wind_from=(direction + b) % 360))]
    if joint:
        # Product of two three-point rules. No wind speed/bearing covariance is reported.
        members.extend((f'speed_{speed_name}_bearing_{bearing_name}', 1 / 36,
                        dict(cfg, wind_speed=speed + speed_offset, wind_from=(direction + bearing_offset) % 360))
                       for speed_name, speed_offset in (('low', -a), ('high', a))
                       for bearing_name, bearing_offset in (('low', -b), ('high', b)))
    return members


def _run_weather_member(job, *, output, physics, cryogenic_options, wind_roughness):
    name, weight, cfg = job
    folder = output / 'weather_members' / name
    folder.mkdir(parents=True, exist_ok=True)
    rows, result = _run_case(cfg, output=folder, render=False, physics=physics,
                             cryogenic_options=cryogenic_options, wind_roughness=wind_roughness)
    return rows, dict(result, name=name, weight=weight, wind_speed=cfg['wind_speed'], wind_from=cfg['wind_from'])


def _aggregate_weather(cfg, results, output):
    # Classify the weighted mean concentration, which is the observed statistic.
    rows = []
    for aligned in zip(*(member_rows for member_rows, _ in results), strict=True):
        reference = aligned[0]
        if any(row['sensor'] != reference['sensor'] for row in aligned):
            raise RuntimeError('Weather ensemble receptors do not align')
        prediction = math.fsum(row['predicted_volpct'] * member['weight']
                               for row, (_, member) in zip(aligned, results, strict=True))
        rows.append(dict(reference, predicted_volpct=prediction, error_pp=prediction-reference['observed_volpct']))
    result = dict(test=cfg['test'], **_metrics(rows), weather_members=[member for _, member in results])
    folder = output / f'test{cfg["test"]}'
    folder.mkdir(exist_ok=True)
    _write_json(folder / 'case.json', cfg)
    _write_sensors(folder / 'sensor_comparison.csv', rows)
    _write_json(folder / 'summary.json', result)
    return rows, result


def run_ffi(tests, output, workers=1, *, report, render=True, physics='legacy',
            cryogenic_rh=None, ground_heat_transfer=None, density_entrainment=None, weather_ensemble=False,
            wind_roughness=None, joint_weather_ensemble=False, measured_weather=None, wind_interval_s=20.,
            wind_reference_height_m=10, conservative_handoff=False, site_container=False, crossflow_jet=False,
            plume_section_forces=False):
    """Run selected FFI cases against the user's local copy of the official report."""
    if workers < 1 or not tests or len(set(tests)) != len(tests) or not set(tests) <= set(range(1, 8)):
        raise ValueError('Choose distinct tests from 1 to 7 and a positive worker count')
    if physics not in ('legacy', 'jet', 'pool', 'dense'):
        raise ValueError('Choose legacy, jet, pool or dense physics')
    if weather_ensemble and joint_weather_ensemble:
        raise ValueError('Choose one FFI weather ensemble')
    ensemble = weather_ensemble or joint_weather_ensemble
    if ensemble and measured_weather is not None:
        raise ValueError('Choose either a steady weather ensemble or measured wind')
    if wind_reference_height_m not in (5, 10):
        raise ValueError('The FFI wind mast provides measurements at 5 and 10 m')
    if ensemble and render:
        raise ValueError('The FFI weather ensemble requires --no-3d')
    if wind_roughness is not None and not 0 < wind_roughness < 10:
        raise ValueError('Wind roughness must be positive and below the 10 m reference height')
    cryogenic_options = {}
    if conservative_handoff:
        cryogenic_options['conservative_handoff'] = True
    if crossflow_jet:
        cryogenic_options['crossflow_jet'] = True
    if plume_section_forces:
        cryogenic_options['plume_section_forces'] = True
    if cryogenic_rh is not None:
        if not 0 <= cryogenic_rh <= 100:
            raise ValueError('Cryogenic relative humidity must be between 0 and 100 percent')
        cryogenic_options['relative_humidity_pct'] = cryogenic_rh
    if ground_heat_transfer is not None:
        if not math.isfinite(ground_heat_transfer) or ground_heat_transfer <= 0:
            raise ValueError('Ground heat-transfer coefficient must be positive and finite')
        cryogenic_options['ground_heat'] = dict(heat_transfer_w_m2_k=ground_heat_transfer)
    if density_entrainment is not None:
        if not math.isfinite(density_entrainment) or density_entrainment <= 0:
            raise ValueError('Density entrainment coefficient must be positive and finite')
        cryogenic_options['density_entrainment'] = dict(coefficient=density_entrainment, seed=SEED)
    if cryogenic_options and physics == 'legacy':
        raise ValueError('The selected cryogenic options require jet, pool or dense physics')
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
            cfg = dict(cfg, observed_window_s=window, sensors=sensors)
            if site_container:
                cfg['site_container'] = ISO_CONTAINER
            if measured_weather is not None:
                # Test 5 restarted after its first release. The concentration
                # table describes the second release only (FFI Appendix A.5).
                origin = 200. if test == 5 else 0.
                frames = read_weather(measured_weather / f'test{test}.xlsx', origin, window[1], wind_interval_s,
                                      wind_reference_height_m)
                cfg.update(measured_wind=frames, release_clock_origin_s=origin,
                           wind_reference_height_m=wind_reference_height_m,
                           spinup_s=window[0] - origin, sim_s=window[1] - origin)
            selected.append(cfg)
    output.mkdir(parents=True, exist_ok=True)
    if ensemble:
        jobs = [member for cfg in sorted(selected, key=lambda c: c['sim_s'], reverse=True)
                for member in _weather_members(cfg, joint=joint_weather_ensemble)]
        run_case = partial(_run_weather_member, output=output, physics=physics, cryogenic_options=cryogenic_options,
                           wind_roughness=wind_roughness)
    else:
        jobs = selected
        run_case = partial(_run_case, output=output, render=render, physics=physics,
                           cryogenic_options=cryogenic_options, wind_roughness=wind_roughness)
    workers = min(workers, len(jobs))
    if workers == 1:
        results = list(map(run_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_case, jobs))
    if ensemble:
        results = [_aggregate_weather(cfg, [pair for pair in results if pair[1]['test'] == cfg['test']], output)
                   for cfg in selected]
    rows = [row for case_rows, _ in results for row in case_rows]
    summary = dict(version=__version__, source=REPORT, source_url=REPORT_URL,
                   seed=SEED, workers=workers, visualization=render, step_s=DT, physics=physics,
                   weather_ensemble=ensemble,
                   weather_ensemble_rule='joint' if joint_weather_ensemble else 'axis' if ensemble else None,
                   measured_weather=None if measured_weather is None else dict(
                       interval_s=wind_interval_s, reference_height_m=wind_reference_height_m),
                   site_container=ISO_CONTAINER if site_container else None,
                   wind_roughness_m=wind_roughness,
                   cryogenic_options=cryogenic_options,
                   particles_per_s=PARTICLES_PER_S, cases=[result for _, result in results],
                   overall=_metrics(rows), wall_s=perf_counter()-started)
    _write_sensors(output / 'sensor_comparison.csv', rows)
    _write_json(output / 'summary.json', summary)
    _write_index(output, summary, render)
    return summary
