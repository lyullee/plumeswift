"""Run the three FLADIS vapor-handoff benchmarks with PlumeSwift."""
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import xlrd
from scipy.integrate import trapezoid

from smedis_benchmark import (
    R, ContinuousRelease, Domain, JetDragLength, LH2Momentum, SensorBoxes,
    SensorProfile, ShallowLayerPressure, Source, SurfaceLayerTurbulence,
    TimeAveragedSensors, run_particles, stats, write_csv,
)
from plumeswift.lh2_jet import Disk


ROOT = Path(__file__).resolve().parent
SURVEY_X_ORIGIN_M = 20.
SURVEY_Y_ORIGIN_M = 104.
INPUT = Path('tmp/ammonia_benchmarks')
# JRIII input specification tables 1, 4 and 9 and site geometry table 2.
TRIALS = {
    9: dict(rate=.40, window=600, duration=900, speed=6.1, direction=296, length=348,
            ustar=.44, sigma_v_ratio=3.14, air_k=288.65, pressure=102000, x0=0.,
            handoff=4.2, jet_speed=4.75, jet_k=203.7, diameter=.88, far=238),
    16: dict(rate=.27, window=600, duration=1200, speed=4.4, direction=278, length=138,
             ustar=.41, sigma_v_ratio=1.92, air_k=289.65, pressure=102000, x0=-2.,
             handoff=3.1, jet_speed=5.12, jet_k=203.9, diameter=.73, far=240),
    24: dict(rate=.46, window=400, duration=600, speed=4.9, direction=281, length=-77,
             ustar=.405, sigma_v_ratio=2.06, air_k=290.65, pressure=101300, x0=0.,
             handoff=4.4, jet_speed=4.22, jet_k=204.0, diameter=1.069, far=238),
}
ARCS = ((20, .1, (2., 2., .2), 14.), (70, .5, (4., 4., .4), 36.),
        (240, 1.5, (8., 8., .6), 88.))
MW = .01703052
AIR_MW = .02897
CP = 2100.
AIR_CP = 1005.


class AmmoniaVapour:
    """Ideal-gas NH3 and air with sensible enthalpy relative to ambient."""

    def __init__(self, temperature, pressure):
        self.ambient_temperature = temperature
        self.pressure = pressure
        self.ambient_density = pressure * AIR_MW / (R * temperature)

    def evaluate(self, concentration, energy):
        k = self.pressure * AIR_MW * AIR_CP / R
        a = concentration * (CP - AIR_CP * AIR_MW / MW)
        b = k - a * self.ambient_temperature - energy
        temp = 2 * k * self.ambient_temperature / (
            b + np.sqrt(b*b + 4*a*k*self.ambient_temperature))
        fraction = concentration * R * temp / (self.pressure * MW)
        if not np.isfinite(temp).all() or np.any((fraction < 0.) | (fraction > 1.)):
            raise ValueError('Invalid ammonia gas state')
        density = concentration + (1. - fraction)*self.pressure*AIR_MW/(R*temp)
        return SimpleNamespace(concentration=concentration, temperature=temp, density=density, volume_fraction=fraction)


def observations(trial):
    sheet = xlrd.open_workbook(INPUT/f'fladis{trial}.xls').sheet_by_index(0)
    data = [sheet.row_values(i) for i in range(sheet.nrows)]
    sensors = next(i for i, r in enumerate(data) if r[0] == '# Concentration_Sensor')
    arc_table = next(i for i, r in enumerate(data) if str(r[0]).startswith('# distance (m)'))
    readings = np.array([r[:4] for r in data[sensors+2:]
                         if all(isinstance(value, float) for value in r[:4]) and r[3] >= 0])
    arc_summary = np.array([r[:4] for r in data[arc_table+1:sensors]
                            if all(isinstance(value, float) for value in r[:4])])
    # The 7/10 m, 40 m and 110 m auxiliary points are outside the three arcs.
    return [readings[mask] for mask in (
        (readings[:,0] >= 16) & (readings[:,0] <= 21),
        (readings[:,0] >= 64) & (readings[:,0] <= 71),
        readings[:,0] > 200)], arc_summary


def width(y, concentration):
    i = np.argsort(y)
    y, c = y[i], concentration[i]
    integral = trapezoid(c, y)
    if integral <= 0.:
        return None
    centre = trapezoid(c*y, y)/integral
    return float(np.sqrt(trapezoid(c*(y-centre)**2, y)/integral))


def run(trial, *, output=ROOT, particles_per_s=120., thermo_type=AmmoniaVapour,
        turbulence_type=SurfaceLayerTurbulence, simulate=run_particles, source_energy=None):
    started = perf_counter()
    cfg = TRIALS[trial]
    measured, published = observations(trial)
    origin = np.array([SURVEY_X_ORIGIN_M, SURVEY_Y_ORIGIN_M, 1.5])
    centre = origin + [cfg['x0'] + cfg['handoff'], 0, 0]
    direction = (cfg['direction'] - 110. + 90.) % 360.
    profile = SensorProfile(cfg['speed'], direction, 10., .04, 1./cfg['length'])
    thermo = thermo_type(cfg['air_k'], cfg['pressure'])
    fraction = .12
    mass_flux = fraction * cfg['pressure'] * MW/(R * cfg['jet_k']) * cfg['jet_speed']
    diameter = 2.*np.sqrt(cfg['rate']/(np.pi*mass_flux))
    air_ratio = (1. - fraction)/fraction * AIR_MW/MW
    source_enthalpy = (cfg['jet_k']-cfg['air_k'])*(CP + air_ratio*AIR_CP)
    if source_energy is not None:
        source_enthalpy = source_energy(thermo, cfg['jet_k'], fraction)
    source = Source(Disk(tuple(centre), (1., 0., 0.), diameter/2.),
                    ContinuousRelease(cfg['rate'], particles_per_s, end_s=cfg['window']),
                    initial_drift=tuple(np.array([cfg['jet_speed'], 0., 0.]) -
                                        np.r_[profile.evaluate(centre[2]), 0.]),
                    specific_enthalpy_j_kg=source_enthalpy)
    domain = Domain((145, 104, 68), (2., 2., .25))
    observed_groups = []
    arc_grids = []
    for (nominal, height, box, span), points in zip(ARCS, measured, strict=True):
        observed_groups.append(TimeAveragedSensors(
            SensorBoxes(points[:,:3] + [*origin[:2], 0.], box), 0., cfg['window'], thermo))
        radius = cfg['far'] if nominal == 240 else nominal
        # Match the fixed-frame survey x/y origin. The far group uses the
        # actual test radius, including the trial-16 source offset.
        across = np.arange(-span, span + .1, box[1]/2)
        xyz = np.c_[cfg['x0'] + np.sqrt(radius**2-across**2), across,
                    np.full_like(across, height)] + [*origin[:2], 0.]
        arc_grids.append((across, TimeAveragedSensors(SensorBoxes(xyz, box), 0., cfg['window'], thermo)))

    def momentum(sampler, solid, _):
        return LH2Momentum(thermo, shallow_layer=ShallowLayerPressure(sampler, solid, 0., 1.),
                           drag_length=JetDragLength(centre, np.array([1., 0., 0.]), diameter))

    turbulence = turbulence_type(terrain_roughness=.04, friction_velocity=cfg['ustar'],
                                        crosswind_std=cfg['ustar']*cfg['sigma_v_ratio'],
                                        reciprocal_length=1./cfg['length'], flow_from_deg=direction)
    accounting, wind = simulate(domain, profile, (), source,
                                      observed_groups + [ob for _, ob in arc_grids],
                                      cfg['window'], .25, momentum, turbulence_model=turbulence)
    if not wind.converged:
        raise RuntimeError(f'FLADIS {trial} wind did not converge')
    sensor_rows, arc_rows = [], []
    for (arc, h, _, _), points, measured_observer, (across, arc_observer) in zip(
            ARCS, measured, observed_groups, arc_grids, strict=True):
        prediction = 1e6*measured_observer.mean()
        measured_ppm = 1e4*points[:,3]
        arc_pred = 1e6*arc_observer.mean()
        at_height = np.isclose(points[:,2], h)
        archive = published[np.isclose(published[:,1], h) &
                            (np.abs(published[:,0]-arc) < 10)]
        x_pub, _, max_pub, width_pub = archive[0]
        for (x, y, z, _), a, b in zip(points, measured_ppm, prediction, strict=True):
            sensor_rows.append(dict(trial=trial, nominal_arc_m=arc, x_m=x, y_m=y, z_m=z,
                                    observed_ppmv=a, model_ppmv=b))
        arc_rows.append(dict(trial=trial, nominal_arc_m=arc, smedis_arc_m=x_pub,
                             protocol_arc_m=cfg['far'] if arc==240 else arc,
                             height_m=h, averaging_s=cfg['window'],
                             published_arcmax_ppmv=max_pub*1e4,
                             observed_sensor_peak_at_height_ppmv=float(np.max(measured_ppm[at_height])),
                             modeled_sensor_peak_at_height_ppmv=float(np.max(prediction[at_height])),
                             modeled_arcmax_ppmv=float(np.max(arc_pred)),
                             published_width_sigma_y_m=width_pub,
                             observed_matched_width_m=width(points[at_height,1], measured_ppm[at_height]),
                             modeled_matched_width_m=width(points[at_height,1], prediction[at_height]),
                             modeled_arc_width_m=width(across, arc_pred),
                             **stats(measured_ppm, prediction)))
    write_csv(output/f'fladis{trial}_sensors.csv', sensor_rows)
    write_csv(output/f'fladis{trial}_arcs.csv', arc_rows)
    result = dict(trial=trial, **cfg, tabulated_source_diameter_m=cfg['diameter'],
                  source_diameter_mass_consistent_m=diameter, source_enthalpy_j_kg=source_enthalpy,
                  dt_s=.25, particles_per_s=particles_per_s, wind_iterations=wind.iterations,
                  accounting=accounting, wall_s=perf_counter()-started)
    (output/f'fladis{trial}_run.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run, TRIALS):
            print(json.dumps(result), flush=True)
