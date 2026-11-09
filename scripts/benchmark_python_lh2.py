"""Run independent Python QES and public SLABx-LH2 on the same FFI inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
from dataclasses import replace
from functools import partial
import json
import math
import os
from pathlib import Path
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from benchmark_lh2 import CASES, rows
from plumeswift.assembly import assemble, AssemblyConfig, ComponentChoice
from plumeswift.boundaries import AxisBoundary, DomainBoundaries
from plumeswift.contracts import ModelDomainError
from plumeswift.geometry import PolygonBuilding
from plumeswift.interpolation import FieldSampler
from plumeswift.lh2 import LH2Momentum
from plumeswift.particles import ParticleIntegrator, ParticleSimulation
from plumeswift.particle_sampling_pool import ParticleSamplingPool
from plumeswift.sources import Source, Point, ContinuousRelease
from plumeswift.statistics import SensorBoxes, TimeAveragedSensors
from plumeswift.walls import VoxelWalls
from plumeswift.wind_case import read_wind_case


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/python_lh2_validation'
NOZZLE_AREA = math.pi*(.0254/2)**2
ORIGIN = (120., 120., .5)
PARTICLES_PER_S = 100
DT = .2
CFD_CONTAINER = dict(x=120., y=130., length=13., width=2.5, height=5.2)


def python_qes(folder, cfg, boxes, seed, thermo):
    started = perf_counter()
    case = read_wind_case(folder/'winds.xml', poisson_backend='cpu',
                          legacy_turbulence_xml=True, native_sensor_math=True)
    if cfg.get('cfd_layout'):
        case = replace(case, buildings=(PolygonBuilding.rectangle(**CFD_CONTAINER),), mesh_type=1,
                       native_staircase_metrics=True)
    wind = case.run()
    assert wind['solver'].converged, wind['solver']
    workspace = wind['workspace']
    compiled = cfg.get('compiled_transport', False)
    sampler = FieldSampler(workspace.coordinates, {key: workspace.arrays[key] for key in 'uvw'},
                           wind['turbulence'], mixlength=workspace.arrays['mixlength'], compiled=compiled)
    boundaries = DomainBoundaries([AxisBoundary(axis[1], axis[-2])
                                   for axis in (sampler.xf, sampler.yf, sampler.zf)])
    composition = assemble(AssemblyConfig(ComponentChoice('thermo.'+thermo, {
        'ambient_temperature': cfg['Ta'], 'liquid_fraction': cfg['liquid_fraction'],
        'relative_humidity': cfg['RH_assumed'],
        'out_of_range': 'raise',
    }), ComponentChoice('poisson.krylov'), seed=seed))
    speed = cfg['rate_kg_s']/(70.8*NOZZLE_AREA) if cfg['orientation'] == 'horizontal' else 0.
    source = Source(Point(ORIGIN), ContinuousRelease(cfg['rate_kg_s'], PARTICLES_PER_S),
                    initial_drift=(speed, 0., 0.))
    observer = TimeAveragedSensors(boxes, cfg['spinup_s'], cfg['sim_s'], composition.thermo)
    walls = VoxelWalls(sampler, np.isin(workspace.arrays['icellflag'], (0, 2)), compiled=compiled)
    layer_volumes = np.diff(sampler.zf)*sampler.dx*sampler.dy
    volumes = (float(layer_volumes[0]) if np.all(layer_volumes == layer_volumes[0])
               else np.broadcast_to(layer_volumes[:, None, None],
                                    (len(sampler.z), len(sampler.y), len(sampler.x))).ravel())
    locate = None
    if cfg.get('momentum_cells') == 'native':
        locate = partial(sampler.native_candidate_cell_indices,
                         spacing=(sampler.dx, sampler.dy, float(np.min(np.diff(sampler.zf)))))
    def progress(t, active):
        print(f'QES test {cfg["test"]}: {t:g}/{cfg["sim_s"]:g} physical s, '
              f'{active} active, {perf_counter()-started:.1f} wall s', flush=True)

    workers = cfg.get('particle_workers', 1)
    context = ParticleSamplingPool(sampler, workers) if workers > 1 else nullcontext()
    with context as sampling_pool:
        integrator = ParticleIntegrator(sampler, boundaries, walls, np.random.default_rng(seed),
                                        composition.config.numeric, sampling_pool=sampling_pool,
                                        compiled_gle=cfg.get('compiled_gle', False))
        simulation = ParticleSimulation([source], integrator, [observer],
                                        LH2Momentum(composition.thermo, composition.config.physics), volumes,
                                        momentum_cell_indices=locate)
        setup_s = perf_counter()-started
        accounting = simulation.run(cfg['sim_s'], DT, progress)
    prediction = 100*observer.mean()
    wall_s = perf_counter()-started
    assert np.all(np.isfinite(prediction))
    assert math.isclose(observer.sampled_s, cfg['window_s'], abs_tol=1e-7)
    assert math.isclose(accounting['released_mass_kg'], cfg['rate_kg_s']*cfg['sim_s'], rel_tol=1e-8)
    assert abs(accounting['mass_balance_error_kg']) < 1e-6
    return prediction, dict(wall_s=wall_s, setup_s=setup_s, integration_s=wall_s-setup_s,
                            accounting=accounting, seed=seed, thermo=thermo,
                            parallel_sampling_batches=sampling_pool.parallel_batches if sampling_pool else 0)


def slab(cfg, boxes):
    from slabx.core.plume import run_dispersion
    from slabx.core.source import HorizontalJet
    from slabx.post.concentration import concentration_field
    from slabx.submodels.atmosphere import Atmosphere
    from slabx.thermo.coolprop import CoolPropThermo, coolprop_water
    from slabx_lh2.plume_width import plume_width_coupling
    from slabx_lh2.water_ice import with_sublimation
    from review_slabx_lh2 import H2

    started = perf_counter()
    atm = Atmosphere(u_ref=cfg['wind_speed'], z_ref=10, T=cfg['Ta'], rh=cfg['RH_assumed'], z0=.01, stability='D')
    source = HorizontalJet(substance=H2, rate=cfg['rate_kg_s'], area=NOZZLE_AREA, duration=cfg['sim_s'],
                           liquid_fraction=cfg['liquid_fraction'], height=ORIGIN[2], T_source=20.37)
    with plume_width_coupling():
        trajectory, _ = run_dispersion(source, atm, CoolPropThermo(H2, fluid='Hydrogen'),
                                       with_sublimation(coolprop_water()), x_max=160, n_puff_steps=40)
    direction = math.radians(cfg['wind_from']+180)
    ux, uy = math.sin(direction), math.cos(direction)
    offsets = np.array((-1/3, 0., 1/3))
    fields = {}
    prediction = []
    for east, north, z in boxes.centres-np.array((ORIGIN[0], ORIGIN[1], 0.)):
        values = []
        for height in z+offsets*boxes.widths[2]:
            if height not in fields:
                fields[height] = concentration_field(trajectory, atm, z=height, t_avg=cfg['window_s'],
                                                     t_release=cfg['sim_s'])
            field = fields[height]
            for x in east+offsets*boxes.widths[0]:
                for y in north+offsets*boxes.widths[1]:
                    values.append(np.interp(x*ux+y*uy, field.x, field.at_y(x*uy-y*ux), left=0., right=0.))
        prediction.append(100*float(np.mean(values)))
    return np.asarray(prediction), perf_counter()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', nargs='+', type=int, default=list(range(1, 8)))
    parser.add_argument('--thermo', choices=('coolprop', 'legacy'), default='legacy')
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--model', choices=('qes', 'slab', 'both'), default='both')
    parser.add_argument('--slab-repeats', type=int, default=3)
    parser.add_argument('--momentum-cells', choices=('physical', 'native'), default='physical',
                        help='Use physical cell faces or replay native QES addressing for LH2 forces')
    parser.add_argument('--case-workers', type=int, default=1,
                        help='Run independent FFI cases in separate processes')
    parser.add_argument('--particle-workers', type=int, default=1,
                        help='Sample a single stationary wind field in worker processes')
    parser.add_argument('--compiled-transport', action='store_true',
                        help='Use optional Numba acceleration for interpolation and voxel-wall traversal')
    parser.add_argument('--compiled-gle', action='store_true',
                        help='Use the optional specialized float64 GLE linear solver')
    parser.add_argument('--tag', default='', help='Append a distinct result label')
    parser.add_argument('--cfd-layout', action='store_true',
                        help='Dry-air sensitivity for FFI 1/3. QES includes the schematic container.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cfd_layout and set(args.tests)-{1, 3}:
        parser.error('--cfd-layout only applies to FFI Tests 1 and 3')
    if args.slab_repeats < 1:
        parser.error('--slab-repeats must be positive')
    if args.case_workers < 1 or len(set(args.tests)) != len(args.tests):
        parser.error('Use a positive worker count and distinct tests')
    if args.particle_workers < 1 or args.particle_workers > 1 and args.case_workers > 1:
        parser.error('Use a positive particle worker count and only one multiprocessing level')
    if args.tag and not args.tag.replace('_', '').isalnum():
        parser.error('--tag must contain letters, digits or underscores')
    started = perf_counter()
    if args.case_workers == 1:
        for test in args.tests:
            run_case(test, args)
    else:
        with ProcessPoolExecutor(max_workers=min(args.case_workers, len(args.tests))) as pool:
            list(pool.map(partial(run_case, args=args), args.tests))
    print(f'Suite elapsed: {perf_counter()-started:.3f} s', flush=True)


def run_case(test, args):
    observations = rows()
    suffix = '_cfd_layout' if args.cfd_layout else ''
    if args.momentum_cells == 'native':
        suffix += '_native_cells'
    if args.particle_workers > 1:
        suffix += f'_p{args.particle_workers}'
    if args.tag:
        suffix += '_'+args.tag
    else:
        if args.compiled_transport:
            suffix += '_compiled'
        if args.compiled_gle:
            suffix += '_gle'
    folder = CASES/f'test{test}'
    cfg = json.loads((folder/'config.json').read_text())
    if args.cfd_layout:
        cfg.update(RH_assumed=0., cfd_layout=True)
    records = [record for record in observations if int(record['test']) == test]
    boxes = SensorBoxes(np.loadtxt(folder/'receptors.txt'))
    assert len(records) == len(boxes.centres)
    if args.model in ('slab', 'both'):
        predictions, times = zip(*(slab(cfg, boxes) for _ in range(args.slab_repeats)), strict=True)
        np.testing.assert_array_equal(predictions, np.broadcast_to(predictions[0], np.shape(predictions)))
        result = {'wall_s': float(np.median(times)), 'times_s': times}
        save(test, 'slab'+suffix, cfg | {'container': None}, records, predictions[0], result)
    if args.model in ('qes', 'both'):
        name = f'qes_{args.thermo}_{args.seed}'+suffix
        configuration = cfg | {'container': CFD_CONTAINER if args.cfd_layout else None}
        if args.momentum_cells == 'native':
            configuration['momentum_cells'] = 'native'
        if args.particle_workers > 1:
            configuration['particle_workers'] = args.particle_workers
        if args.compiled_transport:
            configuration['compiled_transport'] = True
        if args.compiled_gle:
            configuration['compiled_gle'] = True
        try:
            prediction, result = python_qes(folder, configuration, boxes, args.seed, args.thermo)
        except ModelDomainError as error:
            failure = dict(test=test, model=name, configuration=configuration,
                           error_type=type(error).__name__, message=str(error))
            (OUT/f'test{test}_{name}_failure.json').write_text(
                json.dumps(failure, indent=2)+'\n', encoding='utf-8')
            raise
        save(test, name, configuration, records, prediction, result)


def save(test, model, cfg, records, prediction, result):
    path = OUT/f'test{test}_{model}'
    payload = dict(test=test, model=model, configuration=cfg, **result)
    path.with_suffix('.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with path.with_suffix('.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('test', 'sensor', 'observed_volpct', 'predicted_volpct'))
        writer.writerows((test, row['sensor'], row['average'], value) for row, value in zip(records, prediction, strict=True))
    print(model, test, result['wall_s'], flush=True)


if __name__ == '__main__':
    main()
