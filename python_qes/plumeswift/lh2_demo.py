"""Run and visualize a self-contained, synthetic LH₂ release case."""

from importlib.resources import files
import json
from math import prod

import numpy as np

from .boundaries import AxisBoundary, DomainBoundaries
from .domain import Domain
from .geometry import PolygonBuilding
from .interpolation import FieldSampler
from .io import write_arrays
from .lh2 import LH2Momentum
from .lh2_case import prepare_cryogenic
from .lh2_visual import GridAverage, visual_dependencies, write_figure
from .meteorology import SensorProfile
from .particles import ParticleIntegrator, ParticleSimulation
from .sources import ContinuousRelease, Point, Source
from .thermo import LegacyEquilibrium
from .turbulence import LocalTurbulence
from .walls import VoxelWalls
from .wind_case import WindCase


def example_input():
    """Read the default input included in the installed distribution."""
    return files('plumeswift').joinpath('example_data', 'lh2.json').read_text(encoding='utf-8')


def run_demo(config, output):
    """Simulate the bundled input schema and write editable input and 3D results."""
    visual_dependencies()

    start, end, dt = (config[key] for key in ('average_from_s', 'duration_s', 'step_s'))
    if not np.isfinite([start, end, dt]).all() or not (0 <= start < end and dt > 0):
        raise ValueError('Require 0 <= average_from_s < duration_s and a positive step_s')
    levels = sorted(config['isosurfaces_volpct'])
    if len(levels) != 3 or not np.isfinite(levels).all() or not 0 < levels[0] < levels[1] < levels[2] <= 100:
        raise ValueError('Provide three distinct isosurfaces_volpct in (0, 100]')
    domain = Domain(tuple(config['grid_cells']), tuple(config['cell_size_m']))
    if any(n < minimum for n, minimum in zip(domain.cells, (3, 3, 2), strict=True)):
        raise ValueError('grid_cells must be at least (3, 3, 2) for this example')
    profile = SensorProfile(config['wind_speed_m_s'], config['wind_from_deg'],
                            config['wind_reference_height_m'], config['roughness_length_m'])
    b = config['building']
    building = PolygonBuilding.rectangle(*(b[key] for key in ('x_m', 'y_m', 'length_m', 'width_m', 'height_m')))
    wind = WindCase(domain, profile, buildings=(building,), wake_mode=2,
                    turbulence=LocalTurbulence(native_quirks=False)).run()
    if not wind['solver'].converged:
        raise ValueError('The example wind solver did not converge; check grid, wind and obstacle inputs')
    workspace = wind['workspace']
    turbulence = wind['turbulence']
    sampler = FieldSampler(workspace.coordinates, {key: workspace.arrays[key] for key in 'uvw'},
                           turbulence, mixlength=turbulence['mixingLengths'])
    solid = np.isin(workspace.arrays['icellflag'], (0, 2))
    axes = (sampler.xf, sampler.yf, sampler.zf)
    boundaries = DomainBoundaries([AxisBoundary(axis[1], axis[-2]) for axis in axes])
    position = config['source_position_m']
    if (len(position) != 3 or not all(axis[1] < value < axis[-2] for value, axis in zip(position, axes))
            or solid[sampler.cell_indices(np.array([position]))][0]):
        raise ValueError('source_position_m must lie in a fluid cell inside the simulation boundaries')

    cryogenic = None
    if 'cryogenic' in config:
        cryogenic = prepare_cryogenic(config['cryogenic'], config['ambient_temperature_k'], position,
                                      config['release_rate_kg_s'], config['particles_per_s'], sampler, solid,
                                      relative_humidity_pct=config.get('relative_humidity_pct', 0.))
        thermo, sources, momentum = cryogenic.thermo, cryogenic.sources, cryogenic.momentum
    else:
        thermo = LegacyEquilibrium(ambient_temperature=config['ambient_temperature_k'],
                                   liquid_fraction=config['liquid_mass_fraction'],
                                   relative_humidity=config['relative_humidity_pct'], out_of_range='raise')
        sources = [Source(Point(position), ContinuousRelease(config['release_rate_kg_s'], config['particles_per_s']))]
        momentum = LH2Momentum(thermo)
    volume = prod(domain.spacing)
    observer = GridAverage(sampler, solid.shape, thermo, volume, start, end)
    integrator = ParticleIntegrator(sampler, boundaries, VoxelWalls(sampler, solid),
                                    np.random.default_rng(config['seed']))
    simulation = ParticleSimulation(sources, integrator, [observer], momentum, volume,
                                    track_enthalpy=cryogenic is not None,
                                    track_carrier_air=cryogenic is not None and cryogenic.track_carrier_air,
                                    density_entrainment=None if cryogenic is None else cryogenic.entrainment)
    accounting = simulation.run(end, dt)
    if not observer.sampled_s:
        raise ValueError('No steps overlap the requested averaging window')

    # Drop the below-ground ghost layer and mask the obstacle in the contour data.
    concentration = (100 * observer.integral[1:] / observer.sampled_s).astype(np.float32)
    concentration[solid[1:]] = np.nan
    coordinates = dict(x=sampler.x, y=sampler.y, z=sampler.z[1:])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'case.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    write_arrays(output / 'concentration.npz', {'h2_mean_volpct': concentration, 'solid': solid[1:]}, coordinates)
    shown = write_figure(output / 'plume.html', concentration, coordinates, levels, position,
                         f'PlumeSwift · Synthetic LH₂ release · Mean H₂ concentration ({start:g}–{end:g} s)',
                         building=building)
    result = dict(averaged_s=observer.sampled_s, peak_mean_volpct=float(np.nanmax(concentration)),
                  rendered_isosurfaces_volpct=shown, accounting=accounting, html=str(output / 'plume.html'))
    if cryogenic is not None:
        result['cryogenic'] = cryogenic.diagnostics(end, accounting)
        (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
