"""Controlled FLADIS tests for surface heating, wind direction and vertical turbulence."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

import run_fladis as baseline
from plumeswift.boundaries import AxisBoundary, DomainBoundaries
from plumeswift.interpolation import FieldSampler
from plumeswift.lh2_ground_heat import CloudGroundHeat
from plumeswift.particles import ParticleIntegrator, ParticleSimulation
from plumeswift.walls import VoxelWalls
from plumeswift.wind_case import WindCase
from plumeswift.thermodynamics.ammonia_moist import MoistAmmoniaVapour
from plumeswift.thermodynamics.particle_density import LeaveOneOutDensity


OUTPUT = baseline.ROOT/'sensitivity'
MET_MODES = ("mast", "lateral", "midpoint", "moist", "moist_midpoint", "source_moist_midpoint", "loo_moist_midpoint")
MODES = ('control', 'heat', 'wind', 'vertical', 'combined')
DIR_STD = {9: 14.5, 16: 11.6, 24: 10.44}
VERTICAL_RATIO = {9: 1.57, 16: 1.22, 24: 1.26}


class HeatedAmmonia(baseline.AmmoniaVapour):
    def heating_limit(self, concentration, target):
        fraction = concentration * baseline.R * target/(self.pressure*baseline.MW)
        air_density = (1. - fraction)*self.pressure*baseline.AIR_MW/(baseline.R*target)
        return (concentration*baseline.CP + air_density*baseline.AIR_CP)*(
            target - self.ambient_temperature)


@dataclass(frozen=True, kw_only=True)
class VerticalTurbulence(baseline.SurfaceLayerTurbulence):
    measured_ratio: float

    def profiles(self, height):
        variance, dissipation = super().profiles(height)
        reference, _ = super().profiles(10.)
        scale = (self.measured_ratio*self.friction_velocity)**2/reference
        return variance*scale, dissipation


class TurningSampler(FieldSampler):
    def turn(self, angle_deg):
        self.angle = np.deg2rad(-angle_deg)
        self.rotation = np.array([[np.cos(self.angle), -np.sin(self.angle)],
                                  [np.sin(self.angle), np.cos(self.angle)]])

    def sample_wind(self, positions):
        wind = super().sample_wind(positions)
        if hasattr(self, 'mast_x'):
            bearing = np.interp(positions[:, 0], self.mast_x, self.mast_bearing)
            wind[:, 1] = wind[:, 0] * np.tan(np.deg2rad(self.axis_bearing + 180. - bearing))
        wind[:,:2] = wind[:,:2] @ self.rotation.T
        return wind


class LateralMastSampler(TurningSampler):
    def configure_lateral(self, stations):
        far = np.array([row for row in stations if row[0] == max(r[0] for r in stations)])
        far = far[np.argsort(far[:, 1])]
        if far.shape != (2, 5) or far[1, 1] <= far[0, 1]:
            raise ValueError('Require two distinct far-mast lateral positions')
        slopes = np.tan(np.deg2rad(self.axis_bearing + 180. - far[:, 3]))
        self.lateral_slope_gradient = float(np.diff(slopes)[0]/np.diff(far[:, 1])[0])
        self.lateral_midpoint = float(np.mean(far[:, 1]))
        self.lateral_slope_offset = float(np.mean(slopes) - np.tan(np.deg2rad(
            self.axis_bearing + 180. - self.mast_bearing[-1])))
        self.survey_y_origin = baseline.SURVEY_Y_ORIGIN_M

    def sample_wind(self, positions):
        if not hasattr(self, 'lateral_slope_gradient'):
            return super().sample_wind(positions)
        wind = FieldSampler.sample_wind(self, positions)
        bearing = np.interp(positions[:, 0], self.mast_x, self.mast_bearing)
        upstream, downstream = self.mast_x[-2:]
        ramp_distance = np.maximum(positions[:, 0]-upstream, 0.)
        span = downstream-upstream
        weight = np.minimum(ramp_distance/span, 1.)
        # For u=U*exp(-integral(B dx)) and v=u*(A+B*y),
        # du/dx + dv/dy vanishes in the unrotated mean field.
        integral = (np.minimum(ramp_distance, span)**2/(2*span)
                    + np.maximum(ramp_distance-span, 0.))
        wind[:, 0] *= np.exp(-self.lateral_slope_gradient*integral)
        lateral = positions[:, 1]-self.survey_y_origin-self.lateral_midpoint
        slope = np.tan(np.deg2rad(self.axis_bearing+180.-bearing))
        slope += weight*(self.lateral_slope_offset+self.lateral_slope_gradient*lateral)
        wind[:, 1] = wind[:, 0]*slope
        wind[:, :2] = wind[:, :2] @ self.rotation.T
        return wind


def direction_sequence(trial, duration):
    cfg = baseline.TRIALS[trial]
    lateral_sigma = cfg['ustar']*cfg['sigma_v_ratio']
    resolved = np.rad2deg(np.arctan(lateral_sigma/cfg['speed']))
    residual = np.sqrt(max(0., DIR_STD[trial]**2-resolved**2))
    t = np.arange(0., duration, 5.)+2.5
    # The reported clockwise turn motivates the slow ramp for Trial 9.
    raw = t if trial == 9 else np.sin(2*np.pi*t/120.)
    return (raw-np.mean(raw))*(residual/np.std(raw)), residual


def mast_direction(trial):
    sheet = baseline.xlrd.open_workbook(baseline.INPUT/f'fladis{trial}.xls').sheet_by_index(0)
    rows = [sheet.row_values(i) for i in range(sheet.nrows)]
    labels = {str(row[0]).strip(): row[1] for row in rows}
    axis = next(float(value) for key, value in labels.items() if key.startswith('x-axis orientation'))
    ideal = next(float(value) for key, value in labels.items() if key.startswith('ideal wind direction'))
    if not np.isclose((axis+180.) % 360., ideal) or not np.isclose(axis, 110.):
        raise ValueError('Unexpected survey axis')
    start = next(i for i,row in enumerate(rows) if row[0] == '# Wind_Direction_Sensor') + 2
    stations = []
    for row in rows[start:]:
        if not isinstance(row[0], float):
            break
        if row[2] == 10.:
            stations.append(tuple(map(float, row[:5])))
    groups = {}
    for x, _, _, bearing, _ in stations:
        groups.setdefault(x, []).append(bearing)
    x = np.array(sorted(groups))
    bearing = [float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(groups[key]))))) % 360.)
               for key in x]
    if len(stations) != 4 or len(x) != 3:
        raise ValueError(f'Unexpected mast data for FLADIS {trial}')
    return axis, x, np.array(bearing), stations


def run_job(job, *, output=None, vertical_buoyancy=True, coarsen_buoyancy=False,
            configure_momentum=None, track_mixture_inertia=False):
    trial, mode, rate = job
    corrected = mode in ('midpoint', 'moist', 'moist_midpoint', 'source_moist_midpoint', 'loo_moist_midpoint')
    lateral = mode == 'lateral' or corrected
    moist = mode in ('moist', 'moist_midpoint', 'source_moist_midpoint', 'loo_moist_midpoint')
    midpoint = mode in ('midpoint', 'moist_midpoint', 'source_moist_midpoint', 'loo_moist_midpoint')
    mast = mode in MET_MODES
    heat = mode in ('heat', 'combined') or mast
    turning = mode in ('wind', 'combined') or mast
    vertical = mode in ('vertical', 'combined') or mast
    folder = 'lateral_wind' if mode == 'lateral' else 'mast_wind'
    path = (baseline.ROOT/folder/f'pps{rate}' if mast else OUTPUT/f'pps{rate}'/mode)
    if corrected:
        path = baseline.ROOT/'mixing_thermo'/f'pps{rate}'/mode
    if output is not None:
        path = output
    path.mkdir(parents=True, exist_ok=True)

    def make_turbulence(**kwargs):
        if vertical:
            return VerticalTurbulence(measured_ratio=VERTICAL_RATIO[trial], **kwargs)
        return baseline.SurfaceLayerTurbulence(**kwargs)

    def simulate(domain, profile, buildings, source, observers, end, dt, make_momentum,
                 *, turbulence_model):
        result = WindCase(domain, profile, turbulence=turbulence_model).run()
        workspace = result['workspace']
        solid = np.isin(workspace.arrays['icellflag'], (0, 2))
        turbulence = result['turbulence']
        sampler_type = LateralMastSampler if lateral else TurningSampler if turning else FieldSampler
        sampler = sampler_type(workspace.coordinates, {k: workspace.arrays[k] for k in 'uvw'},
                               turbulence, mixlength=turbulence['mixingLengths'], compiled=True)
        boundary = DomainBoundaries([AxisBoundary(axis[1], axis[-2])
                                     for axis in (sampler.xf, sampler.yf, sampler.zf)])
        integrator = ParticleIntegrator(sampler, boundary, VoxelWalls(sampler, solid, compiled=True),
                                        np.random.default_rng(20261110), compiled_gle=True,
                                        scheme='midpoint' if midpoint else 'backward-euler')
        momentum = make_momentum(sampler, solid, workspace.building_context)
        if mode == 'loo_moist_midpoint':
            momentum.force_density_reconstruction = LeaveOneOutDensity()
        momentum.vertical_buoyancy = vertical_buoyancy
        if coarsen_buoyancy:
            from fladis_force_study import PairedVerticalCells
            momentum.vertical_force_density = PairedVerticalCells(solid.shape, len(sampler.zf)-2)
        if heat:
            momentum.ground_heat = CloudGroundHeat(solid.shape, 1, sampler.dx*sampler.dy,
                                                   momentum.thermo.ambient_temperature, 20.)
        if configure_momentum is not None:
            configure_momentum(momentum, sampler)
        simulation = ParticleSimulation((source,), integrator, observers, momentum,
                                        np.prod(domain.spacing), track_enthalpy=True,
                                        track_mixture_inertia=track_mixture_inertia)
        if turning:
            angles, residual = direction_sequence(trial, end)
            source_location = np.asarray(source.geometry.centre)[None, :]
            sampler.turn(0.)
            absolute_jet = np.asarray(source.initial_drift) + sampler.sample_wind(source_location)[0]
            if mast:
                axis, x, bearing, stations = mast_direction(trial)
                sampler.mast_x = x + baseline.SURVEY_X_ORIGIN_M
                sampler.mast_bearing = bearing
                sampler.axis_bearing = axis
                if lateral:
                    sampler.configure_lateral(stations)
            for index, angle in enumerate(angles):
                sampler.turn(angle)
                source.initial_drift = tuple(absolute_jet - sampler.sample_wind(source_location)[0])
                accounting = simulation.run(min((index+1)*5., end), dt)
                if corrected and (index+1) % 20 == 0:
                    print(f'progress trial={trial} mode={mode} t={min((index+1)*5., end):.0f}/{end}', flush=True)
        else:
            residual = 0.
            accounting = simulation.run(end, dt)
        assert abs(accounting['mass_balance_error_kg']) < 1e-6
        assert np.isclose(accounting['released_mass_kg'], source.release.mass_rate_kg_s*end)
        if mast:
            if lateral:
                accounting['lateral_slope_gradient_per_m'] = sampler.lateral_slope_gradient
                accounting['lateral_slope_offset'] = sampler.lateral_slope_offset
                accounting['lateral_mast_midpoint_m'] = sampler.lateral_midpoint
            accounting['mast_station_xy_z_fromdeg_sigma'] = stations
            accounting['mast_station_x_m'] = x.tolist()
            accounting['mast_mean_direction_deg'] = bearing.tolist()
        accounting['heat_j'] = momentum.ground_heat_j
        accounting['gle_scheme'] = integrator.scheme
        accounting['track_mixture_inertia'] = track_mixture_inertia
        accounting['vertical_buoyancy'] = vertical_buoyancy
        accounting['vertical_force_cell_height_m'] = sampler.zf[2]-sampler.zf[1]
        if coarsen_buoyancy:
            accounting['vertical_force_cell_height_m'] *= 2
        accounting['force_density_reconstruction'] = 'leave-one-out' if mode == 'loo_moist_midpoint' else 'cell-total'
        accounting['relative_humidity_pct'] = getattr(momentum.thermo, 'relative_humidity_pct', 0.)
        accounting['water_loading_kg_per_kg_dry_air'] = getattr(momentum.thermo, 'water_loading', 0.)
        accounting['direction_residual_std_deg'] = residual
        accounting['vertical_sigma_reference_m_s'] = float(np.sqrt(
            turbulence_model.profiles(10.)[0]))
        return accounting, result['solver']

    def make_thermo(temperature, pressure):
        if not moist:
            return HeatedAmmonia(temperature, pressure)
        sheet = baseline.xlrd.open_workbook(baseline.INPUT/f'fladis{trial}.xls').sheet_by_index(0)
        key, value = next(sheet.row_values(i)[:2] for i in range(sheet.nrows)
                          if 'relative humidity' in str(sheet.cell_value(i, 0)).lower())
        humidity = float(value) * (1. if '%' in key else 100.)
        return MoistAmmoniaVapour(temperature, pressure, humidity)

    def source_energy(thermo, temperature, fraction):
        concentration = fraction*baseline.MW*thermo.pressure/(baseline.R*temperature)
        return float(thermo.heating_limit(concentration, temperature)/concentration)

    return baseline.run(trial, output=path, particles_per_s=rate, thermo_type=make_thermo,
                        turbulence_type=make_turbulence, simulate=simulate,
                        source_energy=source_energy if mode in ('source_moist_midpoint', 'loo_moist_midpoint') else None) | {'mode': mode}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, default=40)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--modes', nargs='+', choices=MODES+MET_MODES, default=MODES)
    parser.add_argument('--trials', nargs='+', type=int, choices=baseline.TRIALS, default=baseline.TRIALS)
    args = parser.parse_args()
    jobs = [(n, mode, args.particles) for mode in args.modes for n in args.trials]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            print(result['trial'], result['mode'], round(result['wall_s'], 1), result['accounting'],
                  flush=True)
