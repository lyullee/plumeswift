"""Run selected SMEDIS field trials with PlumeSwift's 3-D particle transport."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import xlrd
from smedis_flow_inputs import ApproachWindBlend, JetDragLength, WakeWindObservation

from plumeswift.boundaries import AxisBoundary, DomainBoundaries
from plumeswift.building_pipeline import BuildingPipeline, UpwindClosure
from plumeswift.domain import Domain
from plumeswift.fence_wake import FenceShearTurbulence, SolidFenceWake
from plumeswift.geometry import PolygonBuilding
from plumeswift.interpolation import FieldSampler
from plumeswift.lh2 import LH2Momentum
from plumeswift.lh2_dense import DenseLayerExchange, ShallowLayerPressure
from plumeswift.meteorology import SensorProfile
from plumeswift.particles import ParticleIntegrator, ParticleSimulation
from plumeswift.sources import ContinuousRelease, Point, Source
from plumeswift.statistics import SensorBoxes, TimeAveragedSensors
from plumeswift.surface_layer import SurfaceLayerTurbulence
from plumeswift.turbulence import LocalTurbulence
from plumeswift.walls import VoxelWalls
from plumeswift.wind_case import WindCase


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'data' / 'dispersion_benchmarks' / 'batch1_24.zip'
OUTPUT = ROOT / 'results' / 'smedis_benchmark'
UPDATED = ROOT / 'results' / 'smedis_followup'
R = 8.314462618


def load(trial):
    with ZipFile(ARCHIVE) as archive:
        sheet = xlrd.open_workbook(file_contents=archive.read(f'batch1/{trial.lower()}.xls')).sheet_by_index(0)
    data = [sheet.row_values(i) for i in range(sheet.nrows)]
    headers = {row[0].strip(): i for i, row in enumerate(data)
               if isinstance(row[0], str) and row[0].startswith('#') and
               ('Sensor' in row[0] or row[0].startswith('# Arc_Positions'))}
    values = {row[0].strip(): row[1] for row in data[:min(headers.values())]}

    def table(section, columns):
        begin = headers[section] + 2
        end = min((i for i in headers.values() if i >= begin), default=len(data))
        return np.asarray([r[:columns] for r in data[begin:end] if isinstance(r[0], float)], dtype=float)

    return values, table('# Concentration_Sensor', 4), table('# Arc_Positions', 4), data


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    error = predicted - observed
    return dict(n=len(error), rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)), fac2=int(np.count_nonzero((predicted >= .5 * observed)
                                                                   & (predicted <= 2 * observed)
                                                                   & (observed > 0))),
                nonzero=int(np.count_nonzero(observed > 0)))


def run_particles(domain, profile, buildings, source, observers, end, dt, make_momentum=None,
                  turbulence_model=None, diagnostics=None, wind_only=False, building_pipeline=None):
    wind = WindCase(domain, profile, buildings=buildings, wake_mode=2 if buildings else 0,
                    upwind_mode=2 if buildings else 0, rooftop_mode=2 if buildings else 0,
                    building_pipeline=building_pipeline,
                    turbulence=turbulence_model or LocalTurbulence(native_quirks=False,
                                               terrain_roughness=profile.roughness)).run()
    workspace = wind['workspace']
    solid = np.isin(workspace.arrays['icellflag'], (0, 2))
    turbulence = wind['turbulence']
    sampler = FieldSampler(workspace.coordinates, {k: workspace.arrays[k] for k in 'uvw'}, turbulence,
                           mixlength=turbulence['mixingLengths'], compiled=True)
    if diagnostics is not None:
        diagnostics(sampler, turbulence, wind['turbulence_metadata'])
    if wind_only:
        return None, wind['solver']
    walls = VoxelWalls(sampler, solid, compiled=True)
    boundaries = DomainBoundaries([AxisBoundary(axis[1], axis[-2]) for axis in (sampler.xf, sampler.yf, sampler.zf)])
    momentum = None if make_momentum is None else make_momentum(sampler, solid, workspace.building_context)
    pressure = None if momentum is None else momentum.shallow_layer
    integrator = ParticleIntegrator(sampler, boundaries, walls, np.random.default_rng(20261110), compiled_gle=True,
                                    vertical_transport=pressure.vertical_correction if isinstance(pressure, DenseLayerExchange) else None)
    simulation = ParticleSimulation((source,), integrator, observers, momentum, np.prod(domain.spacing),
                                    track_enthalpy=momentum is not None)
    accounting = simulation.run(end, dt)
    assert abs(accounting['mass_balance_error_kg']) < 1e-6
    assert np.isclose(accounting['released_mass_kg'], source.release.mass_rate_kg_s * source.release.end_s)
    return accounting, wind['solver']


def prairie(trial, variant='baseline'):
    started = perf_counter()
    vals, observations, arcs, _ = load(trial)
    observations = observations[observations[:, 3] >= 0]
    sensor_bearing, radius, z, observed = observations.T
    # The SMEDIS PG bearings use the arc coordinate, with 90 degrees on the
    # nominal downwind line. Rotate both reported wind and receptors together.
    wind_axis = float(str(vals[next(k for k in vals if k.startswith('wind direction ('))]).split()[0])
    angle = np.deg2rad(sensor_bearing - wind_axis)
    origin = (70., 360., vals['source elevation (m)'])
    centres = np.c_[origin[0] + radius * np.cos(angle), origin[1] + radius * np.sin(angle), z]
    domain = Domain((132, 96, 80), (7.5, 7.5, 1.))
    profile = SensorProfile(vals['wind speed (m/s)'], 270., vals['measurement height for wind speed (m)'],
                            vals['roughness length z0 (m)'], vals['inverse Monin-Obukhov length (1/m)'])
    model = None
    if variant == 'stability':
        sigma_v = profile.speed * np.deg2rad(vals['domain-avg sigma-theta (deg)'])
        model = SurfaceLayerTurbulence(
            native_quirks=False, terrain_roughness=profile.roughness,
            friction_velocity=vals['friction velocity u-star (m/s)'],
            reciprocal_length=profile.reciprocal_length, crosswind_std=sigma_v)
    output = OUTPUT if variant == 'baseline' else UPDATED / variant
    output.mkdir(parents=True, exist_ok=True)

    def save_turbulence(sampler, fields, metadata):
        rows = []
        for height in (1.5, 5.5, 15.5):
            k, j, i = sampler.cell_indices(np.array([[400., 360., height]]))
            rows.append(dict(height_m=height, ustar_m_s=metadata['u_star'],
                             crosswind_std_m_s=float(np.sqrt(fields['tyy'][k, j, i][0])),
                             vertical_std_m_s=float(np.sqrt(fields['tzz'][k, j, i][0])),
                             C0_eps_m2_s3=float(fields['CoEps'][k, j, i][0])))
        write_csv(output / f'{trial}_turbulence.csv', rows)
    duration = vals['spill duration (s)']
    assert duration == vals['averaging time for mean values (s)']
    source = Source(Point(origin), ContinuousRelease(vals['spill/evaporation rate (kg/s)'], 180., end_s=duration))
    groups = []
    for distance in np.unique(radius):
        mask = radius == distance
        width = min(24., distance * .04)
        groups.append((mask, TimeAveragedSensors(SensorBoxes(centres[mask], (width, width, .5)), 0., duration)))
    accounting, solver = run_particles(domain, profile, (), source, [o for _, o in groups], duration, .5,
                                       turbulence_model=model, diagnostics=save_turbulence,
                                       wind_only=variant == 'baseline-wind')
    if accounting is None:
        return dict(trial=trial, wind_iterations=solver.iterations, wind_converged=solver.converged)
    predicted = np.empty_like(observed)
    density = 101325. * .064 / (R * vals['ambient temperature #1-lower (K)'])
    for mask, observer in groups:
        assert np.isclose(observer.sampled_s, duration)
        predicted[mask] = 1e6 * observer.mean() / density
    rows = [dict(trial=trial, bearing_deg=a, radius_m=b, z_m=c, observed_ppm=o, predicted_ppm=p)
            for a, b, c, o, p in zip(sensor_bearing, radius, z, observed, predicted, strict=True)]
    arc_rows = []
    for distance, height, peak, width in arcs:
        mask = (radius == distance) & (z == height)
        if not np.any(mask):
            continue
        crosswind = distance * np.sin(angle[mask])
        c = predicted[mask]
        spread = np.sqrt(np.average((crosswind - np.average(crosswind, weights=c))**2, weights=c)) if c.sum() else 0.
        arc_rows.append(dict(trial=trial, radius_m=distance, z_m=height, observed_ppm=peak,
                             measured_grid_max_ppm=float(observed[mask].max()), predicted_ppm=float(c.max()),
                             observed_sigma_y_m=width, predicted_sensor_sigma_y_m=float(spread)))
    write_csv(output / f'{trial}_sensors.csv', rows)
    write_csv(output / f'{trial}_arcs.csv', arc_rows)
    return dict(trial=trial, **stats(observed, predicted),
                arc_max=stats([r['observed_ppm'] for r in arc_rows], [r['predicted_ppm'] for r in arc_rows]),
                wall_s=perf_counter()-started, wind_iterations=solver.iterations, accounting=accounting)


@dataclass(frozen=True)
class PropaneVapour:
    """Adiabatic ideal-gas propane/air mixture downstream of the documented handoff."""
    ambient_temperature: float
    pressure: float = 1e5
    propane_mw: float = .0441
    air_mw: float = .02897
    propane_cp: float = 1600.
    air_cp: float = 1005.

    @property
    def ambient_density(self):
        return self.pressure * self.air_mw / (R * self.ambient_temperature)

    def evaluate(self, concentration, energy):
        k = self.pressure * self.air_mw * self.air_cp / R
        a = concentration * (self.propane_cp - self.air_cp * self.air_mw / self.propane_mw)
        b = k - a * self.ambient_temperature - energy
        temperature = 2 * k * self.ambient_temperature / (b + np.sqrt(b*b + 4*a*k*self.ambient_temperature))
        fraction = concentration * R * temperature / (self.pressure * self.propane_mw)
        if np.any(fraction > 1):
            raise ValueError('Propane parcel reconstruction exceeds pure-gas density')
        density = concentration + (1 - fraction) * self.pressure * self.air_mw / (R * temperature)
        return SimpleNamespace(temperature=temperature, density=density, volume_fraction=fraction)


@dataclass(frozen=True)
class SourceWindow:
    centre: np.ndarray
    span: np.ndarray
    height: float

    def sample(self, count, rng):
        offsets = rng.random((count, 2)) - .5
        return self.centre + offsets[:, :1] * self.span + offsets[:, 1:] * np.array([0., 0., self.height])


def wind_records(data, section):
    first = next(i for i, row in enumerate(data) if row[0] == section) + 2
    rows = []
    for row in data[first:]:
        if not isinstance(row[0], float):
            break
        rows.append(row[:5])
    return np.array(rows)


class DenseWakeInventory:
    """Time-weighted marker inventories for matched physical volumes."""

    regions = (('approach', 18., 20.5), ('crest', 20.5, 21.5),
               ('gap', 21.5, 24.5), ('downstream', 25.5, 32.))

    def __init__(self, origin, thermo):
        self.origin = origin
        self.thermo = thermo
        self.totals = np.zeros((len(self.regions), 7))
        self.duration = 0.
        self.sampler = None

    def observe(self, time_start, dt, positions, masses, *, enthalpy):
        self.duration += dt
        local = positions-self.origin
        x, down, z = local[:, 0], -local[:, 1], local[:, 2]
        domain = self.sampler
        for totals, (_, first, last) in zip(self.totals, self.regions, strict=True):
            inside = (x >= -15.8) & (x <= 29.) & (down >= first) & (down < last)
            low = inside & (z < 1.)
            low_mass = masses[low]
            active_pressure_mass = 0.
            if low.any():
                cells = domain.cell_indices(positions[low])
                ids = np.ravel_multi_index(cells, (len(domain.z), len(domain.y), len(domain.x)))
                _, inverse = np.unique(ids, return_inverse=True)
                volume = domain.dx*domain.dy*(domain.zf[cells[0][0]+1]-domain.zf[cells[0][0]])
                cell_mass = np.bincount(inverse, weights=low_mass)
                cell_energy = np.bincount(inverse, weights=low_mass*enthalpy[low])
                state = self.thermo.evaluate(cell_mass/volume, cell_energy/volume)
                active_pressure_mass = cell_mass[state.density > 1.02*self.thermo.ambient_density].sum()
            totals += dt*np.array([masses[inside].sum(), masses[inside & (z < .1)].sum(),
                                   masses[inside & (z < .5)].sum(), low_mass.sum(),
                                   masses[inside & (z >= 2.)].sum(), active_pressure_mass,
                                   np.dot(x[low], low_mass)])

    def rows(self):
        for (region, _, _), total in zip(self.regions, self.totals, strict=True):
            all_mass, surface, below_half, below_one, over_crest, old_pressure, x_moment = total/self.duration
            yield dict(region=region, all_mass_kg=all_mass, z_lt_0p1_kg=surface, z_lt_0p5_kg=below_half,
                       z_lt_1_kg=below_one, z_ge_2_kg=over_crest, z_lt_1_and_rho_gt_1p02_kg=old_pressure,
                       low_centroid_x_m=x_moment/below_one if below_one else 0.)


def propane(trial, variant='baseline'):
    started = perf_counter()
    vals, observations, arcs, data = load(trial)
    observations = observations[observations[:, 3] >= 0]
    raw_xyz, observed = observations[:, :3], observations[:, 3]
    origin = np.array([40.5, 56.5, 0.])
    centres = raw_xyz + origin
    count = int(vals['number of fences'])
    buildings = []
    for i in range(count):
        first = 30 + 7 * i
        x1, y1, x2, y2, height = [data[j][1] for j in range(first, first + 5)]
        assert y1 == y2 and height == 2
        # The archive describes zero-thickness planes; resolve each with one voxel.
        buildings.append(PolygonBuilding.rectangle(min(x1, x2) + origin[0], y1 + origin[1] - .5,
                                                   abs(x1 - x2), 1., height))
    dense_variants = ('dense-reference', 'dense-contact', 'dense-exchange', 'dense-support', 'dense-lee')
    dense_model = variant in dense_variants
    flow_variants = ('flow-source', 'flow-inflow', 'flow-both', 'flow-wake')
    variability_model = variant in ('var-minus', 'var-plus', 'mix-center', 'mix-minus', 'mix-plus')
    mix_wake = variability_model and variant.startswith('mix-')
    direction_node = {'minus': -np.sqrt(3.), 'plus': np.sqrt(3.)}.get(variant.split('-')[-1], 0.) if variability_model else 0.
    flow_model = variant in flow_variants or variability_model
    source_scale = variant in ('flow-source', 'flow-both', 'flow-wake') or variability_model
    approach_wind = variant in ('flow-inflow', 'flow-both', 'flow-wake') or variability_model
    fence_model = variant in ('fence-mean', 'fence-full', 'fence-gap') or dense_model or flow_model
    fence_diagnostics = fence_model or variant == 'fence-reference'
    refine = variant in ('grid', 'both') or fence_diagnostics
    domain = Domain((92, 156 if refine else 78, 72), (1., .5 if refine else 1., .25))
    profile = SensorProfile(vals['site average windspeed at Zref'], vals['site average wind direction'],
                            vals['reference height'], vals['surface roughness  (m)'], -1/90)
    speed_records = wind_records(data, '# Wind_Speed_Sensor')
    direction_records = wind_records(data, '# Wind_Direction_Sensor')
    if variant in ('weather', 'both') or fence_diagnostics:
        # Use only the upwind mast for the inflow. Near-wall observations
        # already contain the wake and serve as independent diagnostics.
        upstream = speed_records[speed_records[:, 1] > 0]
        mast_direction = direction_records[direction_records[:, 1] > 0][0]
        upstream_direction = (mast_direction[3] + direction_node*mast_direction[4]) % 360.
        unit = SensorProfile(1., upstream_direction, profile.reference_height, profile.roughness, profile.reciprocal_length)
        shape = np.hypot(*unit.evaluate(upstream[:, 2]))
        fitted_speed = float(shape @ upstream[:, 3] / (shape @ shape))
        profile = SensorProfile(fitted_speed, unit.direction, unit.reference_height, unit.roughness, unit.reciprocal_length)
    local_profile = profile
    inflow = None
    if approach_wind:
        approach = speed_records[(speed_records[:, 1] >= -20.) & (speed_records[:, 1] < 0.)]
        near_dir = direction_records[(direction_records[:, 1] >= -20.) & (direction_records[:, 1] < 0.)][0]
        approach_direction = (near_dir[3] + direction_node*near_dir[4]) % 360.
        unit = SensorProfile(1., approach_direction, profile.reference_height, profile.roughness, profile.reciprocal_length)
        shape = np.hypot(*unit.evaluate(approach[:, 2]))
        local_profile = SensorProfile(float(shape @ approach[:, 3] / (shape @ shape)), unit.direction,
                                      unit.reference_height, unit.roughness, unit.reciprocal_length)
        inflow = ApproachWindBlend(profile, local_profile, upstream[0, 1]+origin[1], near_dir[1]+origin[1])
    output = OUTPUT if variant == 'baseline' else UPDATED / variant
    output.mkdir(parents=True, exist_ok=True)
    thermo = PropaneVapour(273.15 + vals[next(k for k in vals if k.startswith('ambient temperature'))])
    rate = vals['release rate (kg/s)']
    duration = vals['release duration  (s)']
    assert duration == vals['averaging time for mean values (s)']
    bearing = np.deg2rad(vals['release direction a'])
    axis = np.array([np.sin(bearing), np.cos(bearing), 0.])
    across = np.array([-axis[1], axis[0], 0.])
    # SMEDIS batch1/equivsrc.txt for EEC36, vapour-only plane.
    speed, fraction, temperature, halfwidth = 22.8, .2, 196., .07
    density = fraction * thermo.pressure * thermo.propane_mw / (R * temperature)
    scale = np.sqrt(rate / (density * speed * (2 * halfwidth**2)))
    centre = origin + np.array([0., 0., vals['z']]) + .9 * axis
    ambient = np.array([*profile.evaluate(np.array(centre[2])), 0.])
    if inflow is not None:
        weight = np.clip((inflow.mast_y-centre[1])/(inflow.mast_y-inflow.approach_y), 0., 1.)
        ambient[:2] = (1.-weight)*ambient[:2] + weight*np.asarray(local_profile.evaluate(centre[2]))
    air_ratio = (1 - fraction) / fraction * thermo.air_mw / thermo.propane_mw
    enthalpy = (temperature - thermo.ambient_temperature) * (thermo.propane_cp + air_ratio * thermo.air_cp)
    source = Source(SourceWindow(centre, across * (2 * halfwidth * scale), halfwidth * scale),
                    ContinuousRelease(rate, 600., end_s=duration), initial_drift=tuple(speed * axis - ambient),
                    specific_enthalpy_j_kg=enthalpy)
    observer = TimeAveragedSensors(SensorBoxes(centres, (1., 1., .1)), 0., duration, thermo)
    inventory = DenseWakeInventory(origin, thermo) if dense_model else None

    def momentum(sampler, solid, context):
        if inventory is not None:
            inventory.sampler = sampler
        if variant in ('dense-exchange', 'dense-support'):
            support_depth = .5 if variant == 'dense-support' else 0.
            return LH2Momentum(thermo, shallow_layer=DenseLayerExchange(sampler, solid, 0., 1., support_depth_m=support_depth),
                               ground_contact_z=0.)
        if variant == 'dense-contact':
            return LH2Momentum(thermo, shallow_layer=ShallowLayerPressure(sampler, solid, 0., 1., 0.),
                               ground_contact_z=0.)
        if variant == 'dense-lee' and buildings:
            layer = DenseLayerExchange(sampler, solid, 0., 1., support_depth_m=.5,
                                       wake_influence=context.fence_influence)
            return LH2Momentum(thermo, shallow_layer=layer, ground_contact_z=0.)
        diameter = 4.*halfwidth*scale/3.
        drag = JetDragLength(centre, axis, diameter) if source_scale else None
        return LH2Momentum(thermo, shallow_layer=ShallowLayerPressure(sampler, solid, 0., 1.), drag_length=drag)

    def save_wind(sampler, fields, metadata):
        rows = []
        for kind, records in (('speed', speed_records), ('direction', direction_records)):
            for x, y, z, observed_wind, deviation in records:
                if y > 0:
                    u, v = profile.evaluate(z)
                else:
                    u, v, _ = sampler.sample_wind(np.array([[x, y, z]]) + origin)[0]
                estimate = np.hypot(u, v) if kind == 'speed' else (270 - np.rad2deg(np.arctan2(v, u))) % 360
                rows.append(dict(kind=kind, x_m=x, y_m=y, z_m=z, observed=observed_wind,
                                 predicted=float(estimate), observed_std=deviation))
        write_csv(output / f'{trial}_wind.csv', rows)
        if fence_diagnostics:
            probes = np.array([[x, y, z] for x in (-2.5, 2.5)
                               for y in (-19., -22., -23., -24., -26., -28., -31.)
                               for z in (.05, .25, .5, 1., 1.5, 2., 2.5, 3., 4.)])
            flow = sampler.sample_wind(probes + origin)
            stress, _, eps, _, _ = sampler.sample_turbulence_and_distance(probes + origin)
            write_csv(output / f'{trial}_wake_profiles.csv',
                      [dict(x_m=x, y_m=y, z_m=z, u_m_s=u, v_m_s=v, w_m_s=w,
                            tzz_m2_s2=tau[2, 2], C0_eps_m2_s3=e)
                       for (x, y, z), (u, v, w), tau, e in zip(probes, flow, stress, eps, strict=True)])

    stages = [] if inflow is None else [inflow]
    if fence_model and buildings:
        stages.extend((UpwindClosure(2), SolidFenceWake(
            local_profile, confined_gap=variant == 'fence-gap' or dense_model or flow_model)))
        if variant == 'flow-wake':
            behind = direction_records[direction_records[:, 1] < -21.][0]
            local_speed = speed_records[(speed_records[:, 1] < -21.) & (speed_records[:, 2] == behind[2])][0, 3]
            theta = np.deg2rad(270.-behind[3])
            components = tuple(local_speed*np.array([np.cos(theta), np.sin(theta)]))
            height = buildings[0].height
            stages.append(WakeWindObservation(tuple(behind[:3]+origin), components, (4.*height, 2.*height, .75*height)))
    pipeline = BuildingPipeline(tuple(stages)) if stages else None
    turbulence = None
    if buildings and (variant in ('fence-full', 'fence-gap') or dense_model or flow_model):
        vertical_std = 0.
        if mix_wake:
            wake_speeds = speed_records[(speed_records[:, 1] < -21.) & (speed_records[:, 1] >= -25.)]
            low_speed = wake_speeds[wake_speeds[:, 2] == 1.5, 3].item()
            crest_speed = wake_speeds[wake_speeds[:, 2] == 3., 3].item()
            vertical_std = .25*abs(crest_speed-low_speed)
        turbulence = FenceShearTurbulence(terrain_roughness=profile.roughness, vertical_shear_std=vertical_std,
                                          shear_height=buildings[0].height)

    observers = [observer] if inventory is None else [observer, inventory]
    accounting, solver = run_particles(domain, profile, tuple(buildings), source, observers, duration, .1, momentum,
                                       diagnostics=save_wind, wind_only=variant in ('baseline-wind', 'fence-reference'),
                                       turbulence_model=turbulence, building_pipeline=pipeline)
    if accounting is None:
        return dict(trial=trial, wind_iterations=solver.iterations, wind_converged=solver.converged)
    if inventory is not None:
        write_csv(output / f'{trial}_inventory.csv', list(inventory.rows()))
    assert np.isclose(observer.sampled_s, duration)
    predicted = 100 * observer.mean()
    rows = [dict(trial=trial, fences=count, x_m=x, y_m=y, z_m=z, observed_volpct=o, predicted_volpct=p)
            for (x, y, z), o, p in zip(raw_xyz, observed, predicted, strict=True)]
    write_csv(output / f'{trial}_sensors.csv', rows)
    return dict(trial=trial, fences=count, **stats(observed, predicted), wall_s=perf_counter()-started,
                wind_iterations=solver.iterations, wind_speed_m_s=profile.speed, wind_from_deg=profile.direction,
                wind_converged=solver.converged, accounting=accounting)


def run_trial(task):
    trial, _, variant = task.partition(':')
    variant = variant or 'baseline'
    result = prairie(trial, variant) if trial.startswith('PG') else propane(trial, variant)
    output = OUTPUT if variant == 'baseline' else UPDATED / variant
    result['variant'] = variant
    (output / f'{trial}_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trials', nargs='+')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.trials))) as pool:
        for result in pool.map(run_trial, args.trials):
            print(result, flush=True)
