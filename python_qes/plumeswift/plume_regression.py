"""Analytical QES plume regressions (GPL-3.0-only)."""
import numpy as np

from .cpu_plume_random import LinuxQESRandom
from .domain import Domain
from .plume_config import PlumeConfig
from .plume_fields import _standalone_stress_divergence
from .plume_sources import SourceSpec
from .plume_transport import PlumeModel
from .sources import Point, Source, StepRelease
from .timing import milliseconds
from .wind_workspace_output import native_workspace_coordinates
from .io import WindWorkspace


SINUSOIDAL_LIMITS = {
    'entropy_min': -.04,
    'mean_w_rmse': .03,
    'variance_w_rmse': .03,
    'mean_delta_w_rmse': .6,
    'variance_delta_w_rmse': .3,
}
UNIFORM_LIMITS = {'max_relative_error': .1, 'relative_rmse': .02, 'r2_min': .99}


def _fixture_workspace(domain, mean_wind):
    zero = np.zeros(domain.face_shape, dtype=np.float32)
    flags = np.ones(domain.cell_shape, dtype=np.int32)
    flags[0] = 2
    return WindWorkspace(dict(u=np.full_like(zero, mean_wind), v=zero, w=zero, icellflag=flags),
                         native_workspace_coordinates(domain))


def sinusoidal_diagnostics(z, w, delta_w, step_s):
    """Evaluate the original 25-bin entropy and 20-bin fluctuation RMSEs."""
    z, w, delta_w = (np.asarray(values, dtype=np.float32) for values in (z, w, delta_w))
    if (z.ndim != 1 or z.shape != w.shape or z.shape != delta_w.shape or not len(z)
            or not np.isfinite([z, w, delta_w]).all() or np.any((z < 0) | (z > 1))
            or not np.isfinite(step_s) or step_s <= 0):
        raise ValueError('Sinusoidal diagnostics require finite particle samples within z = 0 to 1 and a positive step')
    entropy_bins = (z.astype(float)/(1/25+1e-9)).astype(int)
    counts = np.bincount(entropy_bins, minlength=25)
    relative = counts/(len(z)/25)
    positive = relative > 0
    entropy = -float(np.sum(relative[positive]*np.log(relative[positive])))

    bins = (z.astype(float)/(1/20+1e-9)).astype(int)
    counts = np.bincount(bins, minlength=20)
    if np.any(counts == 0):
        return dict(entropy=entropy, mean_w_rmse=None, variance_w_rmse=None,
                    mean_delta_w_rmse=None, variance_delta_w_rmse=None)

    def moments(values):
        mean = np.bincount(bins, weights=values, minlength=20)/counts
        variance = np.bincount(bins, weights=(values-mean[bins])**2, minlength=20)/counts
        return mean, variance

    def rmse(actual, expected):
        return float(np.sqrt(np.mean((actual-expected)**2)))

    phase = 2*np.pi*(np.arange(20)+.5)/20
    expected_variance = 1.1+np.sin(phase)
    mean_w, variance_w = moments(w)
    mean_delta, variance_delta = moments(delta_w)
    return dict(entropy=entropy, mean_w_rmse=rmse(mean_w, 0.),
                variance_w_rmse=rmse(variance_w, expected_variance),
                mean_delta_w_rmse=rmse(mean_delta/step_s, 2*np.pi*np.cos(phase)),
                variance_delta_w_rmse=rmse(variance_delta/step_s, 4*expected_variance**1.5))


def run_sinusoidal_regression(*, particles=200_000, duration=12., seed=0, native_rng=False):
    """Run the original sinusoidal fixture with the vectorized Python transport."""
    step = .01
    if particles < 1 or not np.isfinite(duration) or duration < step:
        raise ValueError('Expected positive particles and a duration of at least 0.01 seconds')
    domain = Domain((20, 20, 50), (1/20, 1/20, 1/49))
    workspace = _fixture_workspace(domain, 0.)
    z = workspace.coordinates['z']
    variance = (1.1+np.sin(2*np.pi*z.astype(float))).astype(np.float32)
    stress = np.broadcast_to(variance[:, None, None], domain.cell_shape).copy()
    zero = np.zeros_like(stress)
    turbulence = dict(txx=stress, tyy=stress, tzz=stress, txy=zero, txz=zero, tyz=zero, mixingLengths=zero,
                      CoEps=np.broadcast_to((4*variance.astype(float)**1.5)[:, None, None],
                                            domain.cell_shape).astype(np.float32))
    spacing = np.asarray(domain.spacing, dtype=np.float32)
    _standalone_stress_divergence(turbulence, spacing)
    config = PlumeConfig(duration, step, 0., 1e-10, 'triLinear', ('periodic',)*3, 'doNothing',
                         0, 4000, (0.,)*3, (1.,)*3, (5, 5, 20), seed)
    rng = LinuxQESRandom(seed) if native_rng else None
    model = PlumeModel(config, rng=rng, track_particles=True, native_arithmetic=True)
    model.set_fields(workspace, turbulence, spacing)
    simulation = model.simulation
    source = SourceSpec(0, None, StepRelease(particles, 0., 0., 0.), 0., 0.)
    simulation.sources = (source.create(simulation.integrator.boundaries, source_seed=seed if native_rng else None),)
    for _ in model.advance(milliseconds(duration)):
        pass
    batch = simulation.batch
    accounting = simulation.accounting()
    metrics = (sinusoidal_diagnostics(batch.positions[:, 2], batch.fluctuations[:, 2],
                                      batch.tracking['delta_fluctuation'][:, 2], step)
               if accounting['active_particles'] else dict.fromkeys(
                   ('entropy', 'mean_w_rmse', 'variance_w_rmse', 'mean_delta_w_rmse', 'variance_delta_w_rmse')))
    errors = ('mean_w_rmse', 'variance_w_rmse', 'mean_delta_w_rmse', 'variance_delta_w_rmse')
    passed = (accounting['active_particles'] == particles and accounting['rogue_particles'] == 0
              and metrics['entropy'] is not None and metrics['entropy'] > SINUSOIDAL_LIMITS['entropy_min']
              and all(metrics[key] is not None and metrics[key] < SINUSOIDAL_LIMITS[key] for key in errors))
    return dict(case='sinusoidal', seed=seed, arithmetic='native-cpu-float32',
                rng='linux-qes' if native_rng else 'numpy', duration_s=model.time_ms/1000, particles=particles,
                active_particles=accounting['active_particles'], rogue_particles=accounting['rogue_particles'],
                metrics=metrics, limits=SINUSOIDAL_LIMITS, passed=bool(passed))


def uniform_diagnostics(counts, coordinates, *, box_volume, step_s, averaging_s, release_rate):
    """Compare the original normalized concentration profiles with the Gaussian plume."""
    u_mean, u_star, source_height, boundary_layer = map(np.float32, (2., .174, 70., 1000.))
    x = np.asarray(coordinates['x'], dtype=np.float32)[7:]-20
    y = np.asarray(coordinates['y'], dtype=np.float32)-50
    z = np.asarray(coordinates['z'], dtype=np.float32)-source_height
    counts = np.asarray(counts, dtype=np.float32)
    if counts.shape != (len(z), len(y), len(x)+7) or not len(x) or np.any(x <= 0):
        raise ValueError('Expected a concentration grid with at least one downstream profile')
    if not np.isfinite([box_volume, step_s, averaging_s, release_rate]).all() or min(
            box_volume, step_s, averaging_s, release_rate) <= 0:
        raise ValueError('Gaussian diagnostics require positive volume, step, averaging time and release rate')
    travel = x/u_mean
    integral_time = np.float32(1)/(np.float32(2.5)*u_star/boundary_layer)
    spread = np.float32(1)/(1+np.sqrt(travel/integral_time))
    sigma_y = np.float32(1.78)*u_star*travel*spread
    kappa = np.float32(.4)
    sigma_w = (np.float32(1)/(np.float32(3)*kappa*kappa))**(np.float32(1)/np.float32(3))*u_star
    sigma_z = sigma_w*travel*spread
    rate = np.float32(release_rate)
    norm = u_mean*source_height*source_height/rate
    amplitude = rate/(2*np.pi*float(u_mean)*sigma_y.astype(float)*sigma_z.astype(float))
    model = (amplitude*np.exp(-.5*y[None, :, None]**2/sigma_y**2)
             *np.exp(-.5*z[:, None, None]**2/sigma_z**2)*norm).astype(np.float32)
    conversion = np.float32(step_s)/np.float32(averaging_s)/np.float32(box_volume)
    simulation = counts[:, :, 7:]*conversion*norm
    residual = simulation-model
    peak = model.max(axis=(0, 1))
    # The original accumulates each plane's mean, then the two sums in
    # x-major order with a float store after every addition.
    planes = simulation.transpose(2, 0, 1).reshape(len(x), -1)
    mean = np.add.accumulate(planes, axis=1)[:, -1]/np.float32(planes.shape[1])
    spread_total = np.add.accumulate(((planes-mean[:, None])**2).ravel())[-1]
    residual_total = np.add.accumulate((residual.transpose(2, 0, 1)**2).ravel())[-1]
    r2 = float(np.float32(1)-residual_total/spread_total) if spread_total > 0 else None
    return dict(x_over_h=(x/source_height).tolist(),
                max_relative_error=(np.max(abs(residual), axis=(0, 1))/peak).tolist(),
                relative_rmse=(np.sqrt(np.mean(residual**2, axis=(0, 1)))/peak).tolist(), r2=r2)


def run_uniform_regression(*, particles_per_step=400, duration=2100., averaging_start=300., seed=0):
    """Run the original uniform-flow Gaussian plume fixture."""
    step = 1.
    if (particles_per_step < 1 or not np.isfinite([duration, averaging_start]).all()
            or not 0 <= averaging_start < duration or duration < step):
        raise ValueError('Expected positive release count and a duration after the averaging start')
    domain = Domain((102, 102, 141), (1., 1., 1.))
    workspace = _fixture_workspace(domain, 2.)
    z = workspace.coordinates['z'].astype(float)
    u_star = float(np.float32(.174))
    profile = np.where(z > 0, (1-z/1000)**1.5, 0.)
    shape = domain.cell_shape
    turbulence = {key: np.broadcast_to((factor*profile).astype(np.float32)[:, None, None], shape)
                  for key, factor in zip(('txx', 'tyy', 'tzz', 'txz'),
                                         ((2.50*u_star)**2, (1.78*u_star)**2, (1.27*u_star)**2, -u_star**2),
                                         strict=True)}
    zero = np.zeros(shape, dtype=np.float32)
    coeps = np.zeros_like(z)
    coeps[1:] = float(np.float32(5.7))*u_star**3/(.4*z[1:])*(1-.85*z[1:]/1000)**1.5
    turbulence.update(txy=zero, tyz=zero, mixingLengths=zero,
                      CoEps=np.broadcast_to(coeps.astype(np.float32)[:, None, None], shape))
    spacing = np.asarray(domain.spacing, dtype=np.float32)
    _standalone_stress_divergence(turbulence, spacing)
    config = PlumeConfig(duration, step, 0., 1e-10, 'triLinear', ('exiting', 'exiting', 'reflection'),
                         'doNothing', milliseconds(averaging_start), 1_900_000,
                         (0., 1., 1.), (100., 99., 139.), (20, 49, 69), seed)
    model = PlumeModel(config, native_collection_math=True)
    model.set_fields(workspace, turbulence, spacing)
    simulation = model.simulation
    simulation.sources = (Source(Point((20., 50., 70.)), StepRelease(particles_per_step, 0., 0., duration)),)
    for _ in model.advance(milliseconds(duration)):
        pass
    account = simulation.accounting()
    collection = model.collection
    metrics = (uniform_diagnostics(collection.p_count, config.coordinates, box_volume=collection.volume,
                                   step_s=step, averaging_s=collection.averaging_s, release_rate=particles_per_step)
               if collection.averaging_s > 0 else None)
    passed = (account['rogue_particles'] == 0 and metrics is not None and metrics['r2'] is not None
              and metrics['r2'] > UNIFORM_LIMITS['r2_min']
              and max(metrics['max_relative_error']) < UNIFORM_LIMITS['max_relative_error']
              and max(metrics['relative_rmse']) < UNIFORM_LIMITS['relative_rmse'])
    return dict(case='uniform', seed=seed, duration_s=model.time_ms/1000, released_particles=account['released_particles'],
                active_particles=account['active_particles'], rogue_particles=account['rogue_particles'],
                averaging_s=float(collection.averaging_s), metrics=metrics, limits=UNIFORM_LIMITS, passed=bool(passed))
