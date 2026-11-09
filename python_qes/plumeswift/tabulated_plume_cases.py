"""QES Channel3D and BaileyLES input preprocessing and timestep series."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytical_cases import _write_fields
from .prescribed_diagnostics import write_case_diagnostics
from .prescribed_figures import write_profile_figures
from .prescribed_plume import run_instantaneous_plume, write_instantaneous_config
from .timing import milliseconds


@dataclass(frozen=True)
class Case:
    stem: str
    duration: float
    period: float
    steps: tuple[float, ...]
    collection_upper: tuple[float, float, float]
    boxes: int


TABULATED_CASES = {
    'channel': Case('Channel3D', 18.3, 1.83, (1.83, .183, .00183), (1, 1, 1), 50),
    'bailey-les': Case('BaileyLES', 222., 22.2, (22.2, 2.22, .222, .0222), (6393.2, 6393.2, 1000), 32),
}


def _load_vector(path):
    values = np.loadtxt(path)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f'Expected a finite one-dimensional profile in {path}')
    return values


def _channel_fields(folder):
    x = (np.arange(50)+.5)/50
    z = (np.arange(51)-.5)/49
    inputs = [_load_vector(folder/f'channel_{name}.txt') for name in ('u', 'sigma2', 'epps')]
    if len({len(values) for values in inputs}) != 1:
        raise ValueError('Channel profiles must share a vertical grid')
    levels = np.linspace(0, 1, len(inputs[0]))
    speed, stress, eps = (np.interp(z, levels, values) for values in inputs)
    speed[0], speed[-1] = -speed[1], speed[-2]
    stress[0], stress[-1] = stress[1], stress[-2]
    eps[:2], eps[-1] = inputs[2][1], eps[-2]
    tke = np.real((z*eps).astype(complex)**(2/3))
    velocity = np.zeros((52, 51, 51))
    velocity[:-1, :-1] = speed[:, None, None]
    zero = np.zeros_like(velocity)
    wind = dict(u=velocity, v=zero, w=zero)
    profiles = dict(txx=stress, tyy=stress, tzz=stress, txy=0., txz=0., tyz=0., CoEps=4*eps, tke=tke)
    fields = {key: np.broadcast_to(np.broadcast_to(values, z.shape)[:, None, None], (51, 50, 50))
              for key, values in profiles.items()}
    return dict(x=x, y=x, z=z), (np.arange(52)-1)/49, wind, fields


def _les_field(folder, key, *, velocity, staggered_axis=None):
    values = _load_vector(folder/f'LES_{key}.txt')
    if len(values) != 32**3:
        raise ValueError(f'LES_{key}.txt must contain exactly 32³ values')
    # MATLAB reshape fills x first. Python's file writer takes (z,y,x).
    interior = values.reshape((32, 32, 32), order='F').transpose(2, 1, 0)
    field = np.pad(interior, ((1, 1), (1, 1), (1, 1)), mode='wrap')
    field[-1] = field[-2]
    field[0] = -field[2] if velocity else field[2]
    if not velocity:
        field[1] = field[2]
    # interp3 targets either the original face axis or its cell midpoints.
    for axis in range(3):
        if axis != staggered_axis:
            lower = [slice(None)]*3
            upper = lower.copy()
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            field = .5*(field[tuple(lower)]+field[tuple(upper)])
    if staggered_axis is None:
        field[0], field[-1] = field[1], field[-2]
    else:
        pad = [(0, int(axis != staggered_axis)) for axis in range(3)]
        field = np.pad(field, pad)
        if staggered_axis == 0:
            field[[0, -1]] = 0
        else:
            field[0], field[-2] = -field[1], field[-3]
    return field


def _bailey_fields(folder):
    spacing = np.array([2*np.pi*1000, 2*np.pi*1000, 1000])/31
    coords = {axis: (np.arange(33)-.5)*delta for axis, delta in zip('xyz', spacing, strict=True)}
    wind = {key: _les_field(folder, key, velocity=True, staggered_axis=axis)
            for key, axis in zip('uvw', (2, 1, 0), strict=True)}
    fields = {key: _les_field(folder, key, velocity=False)
              for key in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz')}
    eps = _les_field(folder, 'epps', velocity=False)
    fields['CoEps'] = 4*eps
    tke = np.zeros(33)
    # The original uses MATLAB linear indexing epps_out(k) along x here.
    tke[1:] = (coords['z'][1:]*eps[0, 0, 1:])**(2/3)
    fields['tke'] = np.broadcast_to(tke[:, None, None], eps.shape)
    return coords, (np.arange(34)-1)*spacing[2], wind, fields


def _source_bounds(coords, z_faces):
    # The native Domain derives spacing by subtracting the stored floats.
    axes = [coords[key].astype('f4') for key in 'xy']
    half = .5*float(axes[0][1]-axes[0][0])
    horizontal = [(float(np.float32(values[1]-half)), float(np.float32(values[-2]+half))) for values in axes]
    return (*horizontal, (0., float(np.float32(z_faces[-2]))))


def prepare_tabulated_case(name, profiles, output, *, steps=None, particles=100_000, duration=None):
    case = TABULATED_CASES[name]
    steps = case.steps if steps is None else tuple(dict.fromkeys(steps))
    duration = case.duration if duration is None else duration
    if (particles < 1 or not steps or not set(steps).issubset(case.steps)
            or not np.isfinite(duration) or not max(steps) <= duration <= case.duration):
        raise ValueError('Expected a positive particle count, supported timesteps and valid duration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prepare_fields = _channel_fields if name == 'channel' else _bailey_fields
    coords, z_faces, wind, fields = prepare_fields(Path(profiles))
    wind['icellflag'] = np.ones(tuple(len(coords[key]) for key in 'zyx'), dtype=np.int32)
    wind_path = output/f'{case.stem}_windsWk.nc'
    turbulence = output/f'{case.stem}_turbOut.nc'
    _write_fields(wind_path, dict(coords, z_face=z_faces), wind, faces=('z_face', 'y_face', 'x_face'))
    _write_fields(turbulence, coords, fields)
    configs = {}
    for step in steps:
        config = output/f'{case.stem}_{step:g}_{duration:g}.xml'
        write_instantaneous_config(config, step, particles, _source_bounds(coords, z_faces),
                                   duration=duration, output_period=min(case.period, duration),
                                   collection_upper=case.collection_upper, boxes=(case.boxes,)*3,
                                   vertical_bc='reflection')
        configs[step] = config
    return wind_path, turbulence, configs


def run_tabulated_case(name, profiles, output, *, steps=None, particles=100_000, duration=None,
                       prepare_only=False, seed=0, native_serial=False, plot=False):
    if plot and prepare_only:
        raise ValueError('Plotting requires particle transport')
    wind, turbulence, configs = prepare_tabulated_case(name, profiles, output, steps=steps,
                                                       particles=particles, duration=duration)
    case = TABULATED_CASES[name]
    duration = case.duration if duration is None else duration
    report = dict(wind=str(wind), turbulence=str(turbulence), cases={})
    for step, config in configs.items():
        item = dict(config=str(config), transport_step_s=step, clock_step_ms=milliseconds(step))
        if not prepare_only:
            item['plume'] = run_instantaneous_plume(wind, turbulence, config, output, seed=seed,
                                                   native_serial=native_serial)
            item['diagnostics'] = write_case_diagnostics(config, turbulence, step, duration,
                                                         period=min(case.period, duration),
                                                         height=case.collection_upper[2], les=name == 'bailey-les')
        report['cases'][f'{step:g}'] = item
    if plot:
        report['plots'] = write_profile_figures(wind.parent, case.stem, report['cases'],
                                                 ghosts=True, les=name == 'bailey-les')
    return report
