"""File-based Sinusoidal3D test case from QES testCases (GPL-3.0)."""
from pathlib import Path

import numpy as np

from .analytical_cases import _write_fields
from .prescribed_diagnostics import write_case_diagnostics
from .prescribed_figures import write_profile_figures
from .prescribed_plume import run_instantaneous_plume, write_instantaneous_config


TIME_STEPS = (4., .1, .05, .01)


def prepare_sinusoidal_case(output, *, steps=TIME_STEPS, particles=100_000):
    """Write one stationary field pair and one XML file for each selected timestep."""
    steps = tuple(dict.fromkeys(steps))
    if particles < 1 or not steps or not set(steps).issubset(TIME_STEPS):
        raise ValueError('Expected a positive particle count and supported timesteps')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dz = 1/49
    # QES 2.3.1 treats the outer centers of the old 20-cell layout as
    # horizontal halos. Keep its resulting 0.05–0.95 m transport domain.
    xy = (np.arange(20)+.5)/20
    z = (np.arange(51)-.5)*dz
    coords = dict(x=xy, y=xy, z=z)
    cell_shape = len(z), len(xy), len(xy)
    face_shape = tuple(length+1 for length in cell_shape)
    zero = np.zeros(face_shape, dtype=np.float32)
    wind = output/'Sinusoidal3D_windsWk.nc'
    turbulence = output/'Sinusoidal3D_turbOut.nc'
    _write_fields(wind, dict(coords, z_face=(np.arange(52)-1)*dz),
                  dict(u=zero, v=zero, w=zero, icellflag=np.ones(cell_shape, dtype=np.int32)),
                  faces=('z_face', 'y_face', 'x_face'))
    variance = 1.1+np.sin(2*np.pi*z)
    dissipation = variance**1.5
    # The fractional power below ground is complex in MATLAB. Use its
    # real part in this real-valued file. Offline transport does not use TKE.
    tke = np.real((z*dissipation).astype(complex)**(2/3))
    profiles = dict(txx=variance, tyy=variance, tzz=variance, txy=0., txz=0., tyz=0.,
                    CoEps=4*dissipation, tke=tke)
    _write_fields(turbulence, coords, {
        key: np.broadcast_to(np.broadcast_to(profile, z.shape)[:, None, None], cell_shape)
        for key, profile in profiles.items()})
    configs = {}
    for step in steps:
        xml = output/f'Sinusoidal3D_{step:g}_12.xml'
        write_instantaneous_config(xml, step, particles, ((.05, .95), (.05, .95), (0., 1.)),
                                   duration=12, output_period=4, collection_upper=(1, 1, 1),
                                   boxes=(5, 5, 20), vertical_bc='periodic')
        configs[step] = xml
    return wind, turbulence, configs


def run_sinusoidal_case(output, *, steps=TIME_STEPS, particles=100_000,
                        prepare_only=False, seed=0, native_serial=False, plot=False):
    if plot and prepare_only:
        raise ValueError('Plotting requires particle transport')
    wind, turbulence, configs = prepare_sinusoidal_case(output, steps=steps, particles=particles)
    report = dict(wind=str(wind), turbulence=str(turbulence), cases={})
    for step, xml in configs.items():
        case = dict(config=str(xml))
        if not prepare_only:
            case['plume'] = run_instantaneous_plume(wind, turbulence, xml, output, seed=seed,
                                                   native_serial=native_serial)
            case['diagnostics'] = write_case_diagnostics(xml, turbulence, step, 12)
        report['cases'][f'{step:g}'] = case
    if plot:
        report['plots'] = write_profile_figures(wind.parent, 'Sinusoidal3D', report['cases'])
    return report
