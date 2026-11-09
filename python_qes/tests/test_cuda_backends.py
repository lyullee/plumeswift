from dataclasses import replace
from datetime import datetime, timedelta
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.contracts import NumericPolicy, PhysicalCoefficients
from plumeswift.cuda_particle_buffer import ACTIVE, INACTIVE, ROGUE, CudaParticleBuffer
from plumeswift.cuda_partition import partition_cuda
from plumeswift.cuda_plume_advection import advect_particles
from plumeswift.cuda_plume_boundaries import apply_historical_exiting
from plumeswift.cuda_plume_collection import CudaPlumeCollection
from plumeswift.cuda_plume_interpolation import CudaPlumeInterpolation
from plumeswift.cuda_plume_run import CudaPlumeRun, _wind_schedule
from plumeswift.cuda_plume_random import CudaPlumeRandom
from plumeswift.cuda_ray_mixing import cuda_ray_distances
from plumeswift.cuda_fire_potential import CudaHeatPotential, _kernel as _fire_potential_kernel
from plumeswift.cuda_wind_fields import CudaTurbulenceFields, CudaWindFields
from plumeswift.domain import Domain
from plumeswift.fire_front import cpu_front, cuda_front
from plumeswift.numerics.cuda_poisson import CudaRedBlackSOR
from plumeswift.numerics.poisson import Stencil, make_poisson
from plumeswift.turbulence import LocalTurbulence
from plumeswift.triangle_mesh import TriangleMesh
from plumeswift.wind_case import read_wind_case
from plumeswift.wind_projection import corrected_velocity, project, rhs_from_workspace


def test_cuda_plume_wind_schedule_preserves_time_at_duplicates_and_run_end():
    start = datetime(2020, 1, 1)
    dates = tuple(start+timedelta(seconds=second) for second in (0, 2, 2, 4, 10))
    assert list(_wind_schedule(dates, 0, 6)) == [(0, 2), (2, 4), (3, 6)]
    assert list(_wind_schedule(dates, 1, 3)) == [(2, 2), (3, 3)]
    assert list(_wind_schedule(dates, 4, 8)) == [(4, 8)]
    with pytest.raises(ValueError, match='nondecreasing'):
        list(_wind_schedule(dates[::-1], 0, 6))
    with pytest.raises(ValueError, match='whole-second'):
        list(_wind_schedule((start, start+timedelta(seconds=.5)), 0, 6))


if find_spec('optix') is not None:
    from plumeswift.optix_ray_mixing import load_optix

    load_optix()


@pytest.fixture
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip('CUDA hardware is required for this test')
    return cp


def test_cuda_plume_partition_and_collection(cuda):
    source = CudaParticleBuffer(6)
    target = CudaParticleBuffer(6)
    source.state[:] = cuda.asarray([ACTIVE, INACTIVE, ACTIVE, ACTIVE, ROGUE, INACTIVE])
    source.ids[:] = cuda.arange(6)
    source.positions[:, 0] = cuda.asarray([0., 1., 2., float('nan'), 0., 0.])
    source.previous_stress[:] = 3
    source.stress[:] = 7
    target.stress[:] = -1
    assert source.slot_counts() == (3, 3)
    collector = CudaPlumeCollection((0., 0., 0.), (1., 1., 1.), (2, 1, 1))
    collector.collect(source.positions, source.state, .1)
    active = source.compact_to(target)
    assert active == 3
    np.testing.assert_array_equal(target.ids[:active].get(), [0, 2, 3])
    np.testing.assert_array_equal(target.previous_stress[:active].get(), 3)
    np.testing.assert_array_equal(target.stress[:active].get(), -1)
    np.testing.assert_array_equal(target.state[active:].get(), INACTIVE)
    collector.collect(target.positions, target.state, .2)
    np.testing.assert_array_equal(collector.copyback(), [[[2, 2]]])
    assert collector.averaging_s == np.float32(.3)
    assert target.insert_from(source, active, 6) == 3
    np.testing.assert_array_equal(target.ids[active:].get(), [0, 1, 2])
    np.testing.assert_array_equal(target.stress[active:].get(), 7)
    collector.reset()
    np.testing.assert_array_equal(collector.copyback(), 0)
    assert collector.averaging_s == 0
    empty = CudaParticleBuffer(0)
    assert empty.compact_to(CudaParticleBuffer(0)) == 0
    collector.collect(empty.positions, empty.state, .25)
    assert collector.averaging_s == .25


def test_cuda_historical_combined_partition_clears_source_slots(cuda):
    source, target = CudaParticleBuffer(4), CudaParticleBuffer(4)
    source.state[:] = cuda.asarray([ROGUE, ACTIVE, INACTIVE, ACTIVE])
    source.ids[:] = cuda.arange(100, 104)
    assert source.compact_to(target, clear_source=True) == 2
    np.testing.assert_array_equal(source.state.get(), INACTIVE)
    np.testing.assert_array_equal(source.ids.get(), [100, 101, 102, 103])
    np.testing.assert_array_equal(target.state.get(), [ACTIVE, ACTIVE, INACTIVE, INACTIVE])
    np.testing.assert_array_equal(target.ids[:2].get(), [101, 103])


def test_cuda_historical_exiting_marks_nonactive_slots_and_preserves_nan(cuda):
    model = CudaParticleBuffer(4)
    points = np.array([[0., .5, .5], [np.nan, .5, .5], [.5, .5, 1.], [1.5, .5, .5]], dtype=np.float32)
    model.positions.set(points)
    model.state.set(np.array([ROGUE, ACTIVE, ROGUE, ROGUE], dtype=np.int32))
    apply_historical_exiting(model, ((0, 1),)*3, count=3)
    np.testing.assert_array_equal(model.state.get(), [INACTIVE, ACTIVE, INACTIVE, ROGUE])
    np.testing.assert_array_equal(model.positions.get(), points)
    advect_particles(model, np.zeros((3, 4)), ((0, 1),)*3, historical_exiting=True)
    assert model.state[1].item() == ROGUE


def test_cuda_plume_axis_boundaries_with_large_wrap_and_reflection_limit(cuda):
    model = CudaParticleBuffer(4)
    model.state.fill(ACTIVE)
    position = cuda.asarray([[.5, 128.25, 1.25], [.5, -128.25, 21.], [.5, 2., 0.], [0., 1., 1.]])
    model.fluctuations[:, 2] = .125
    model.positions[:] = position-model.fluctuations
    advect_particles(model, np.zeros((3, 4)), ((0, 1),)*3,
                     boundaries=('exiting', 'periodic', 'reflection'))
    np.testing.assert_array_equal(model.positions.get(), [[.5, .25, .75], [.5, .75, 1.], [.5, 1., 0.], [0., 1., 1.]])
    np.testing.assert_array_equal(model.fluctuations.get()[:, 2], [-.125, .125, .125, .125])
    np.testing.assert_array_equal(model.state.get(), [ROGUE, INACTIVE, ROGUE, INACTIVE])


def test_cuda_plume_random_pads_odd_draws_and_preserves_buffer(cuda):
    odd, even = CudaPlumeRandom(seed=42), CudaPlumeRandom(seed=42)
    target = odd.create('release', 9)
    even.create('release', 10)
    assert odd.generate('release') is target
    np.testing.assert_array_equal(target.get(), even.generate('release').get()[:9])
    for rng in (odd, even):
        rng.create('advection', 12)
    np.testing.assert_array_equal(odd.generate('advection').get(), even.generate('advection').get())
    odd.destroy('release')
    with pytest.raises(KeyError):
        odd.get('release')


def test_cuda_random_diagnostics_match_signed_transform_and_replay(cuda):
    unsigned, signed = CudaPlumeRandom(seed=431), CudaPlumeRandom(seed=431)
    uniform = unsigned.uniform(65536).get()
    centered = signed.uniform(65536, signed=True).get()
    np.testing.assert_array_equal(centered, 2*uniform-1)
    assert np.all((uniform > 0) & (uniform <= 1))
    counts = np.bincount(np.minimum((uniform*10).astype(int), 9), minlength=10)
    assert np.max(abs(counts-6553)) <= 65536*.005
    words = unsigned.uint32(65536).get()
    np.testing.assert_array_equal(words, signed.uint32(65536).get())
    assert .45 < np.unpackbits(words.view(np.uint8)).mean() < .55
    assert signed.uniform(0).size == signed.uint32(0).size == 0


def test_cuda_partition_preserves_bits_and_stable_groups(cuda):
    source = np.array([2., np.nan, -0., 1., -2., 0., np.inf, -np.inf], dtype=np.float32)
    result, split = partition_cuda(cuda.asarray(source), 0)
    assert split == 4
    np.testing.assert_array_equal(result.get().view(np.uint32), source[[2, 4, 5, 7, 0, 1, 3, 6]].view(np.uint32))
    result, split = partition_cuda([], 0)
    assert result.size == split == 0
    with pytest.raises(ValueError, match='one-dimensional'):
        partition_cuda([[1, 2]], 0)


def test_cuda_plume_interpolation_face_offsets_and_release(cuda):
    k, j, i = np.indices((6, 7, 8), dtype=np.float32)
    linear = i+2*j+4*k
    wind = {name: linear for name in 'uvw'}
    cell = linear[:-1, :-1, :-1]
    turbulence = {name: cell for name in ('txy', 'txz', 'tyz', 'div_tau_x', 'div_tau_y', 'div_tau_z', 'nuT')}
    turbulence.update(txx=cell*0, tyy=cell*0-4, tzz=cell*0+9, CoEps=cell*0-1)
    interpolation = CudaPlumeInterpolation(wind, turbulence, (2., 1., .5))
    particles = CudaParticleBuffer(6)
    particles.positions[:] = cuda.asarray([(4., 2., .5), (4., 2., .5), (0., 2., .5),
                                          (float('nan'), 2., .5), (0., 0., 0.), (4., 2., .5)])
    particles.state[:] = cuda.asarray([ACTIVE, ROGUE, ACTIVE, ACTIVE, ACTIVE, ACTIVE])
    interpolation.sample(particles, 5)
    np.testing.assert_array_equal(particles.state.get(), [ACTIVE, ROGUE, ACTIVE, INACTIVE, INACTIVE, ACTIVE])
    # The x=0 stencil crosses a row boundary within the original flat allocation.
    np.testing.assert_array_equal(particles.mean_velocity.get()[[0, 2]], [[11., 11.5, 12.5], [9., 12.5, 13.5]])
    np.testing.assert_array_equal(particles.mean_velocity.get()[[1, 3, 4, 5]], 0.)
    np.testing.assert_array_equal(particles.stress.get()[[0, 2]],
                                  [[0., 10.5, 10.5, -4., 10.5, 9.], [0., 11., 11., -4., 11., 9.]])
    np.testing.assert_array_equal(particles.dissipation.get(), np.array([1e-6, 0, 1e-6, 0, 0, 0], dtype=np.float32))
    interpolation.sample(particles, 5, release=True)
    np.testing.assert_array_equal(particles.previous_fluctuations.get()[:3],
                                  np.array([[1e-8, 2., 3.]]*3, dtype=np.float32))
    np.testing.assert_array_equal(particles.previous_fluctuations.get()[3:], 0.)


def test_cuda_wind_fields_round_trip_preserves_face_order_and_allocation(cuda):
    shape = (3, 4, 5)
    wind = CudaWindFields(shape)
    allocation = wind.data.data.ptr
    bits = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) + np.uint32(0x3f800000)
    bits[0, 0, :3] = [0x80000000, 0x7fc01234, 0x7f800000]
    sources = {name: np.roll(bits, i, axis=1).view(np.float32) for i, name in enumerate('uvw')}
    wind.upload(sources)
    with pytest.raises(ValueError, match='face grid'):
        wind.upload(dict(sources, w=np.zeros((2, 4, 5))))
    outputs = {name: np.empty((3, 4, 10), dtype=np.float32)[..., ::2] for name in 'uvw'}
    wind.download(outputs)
    for name in 'uvw':
        np.testing.assert_array_equal(outputs[name].view(np.uint32), sources[name].view(np.uint32))
    wind.upload({name: sources[name][..., ::-1] for name in 'uvw'})
    assert wind.data.data.ptr == allocation
    wind.download(outputs)
    for name in 'uvw':
        np.testing.assert_array_equal(outputs[name].view(np.uint32), sources[name][..., ::-1].view(np.uint32))
    wind.close()
    with pytest.raises(RuntimeError, match='closed'):
        wind.download(outputs)


def test_cuda_turbulence_fields_transfer_and_reload(cuda):
    shape = (3, 4, 5)
    field = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    values = {name: (field+i)[..., ::-1] for i, name in enumerate(CudaTurbulenceFields.names)}
    turbulence = CudaTurbulenceFields(shape)
    turbulence.upload(values)
    outputs = {name: np.empty(shape, dtype=np.float32) for name in values}
    turbulence.download(outputs)
    for name in values:
        np.testing.assert_array_equal(outputs[name], values[name])

    wind = {name: np.zeros(tuple(n+1 for n in shape), dtype=np.float32) for name in 'uvw'}
    interpolation = CudaPlumeInterpolation(wind, values, (1, 1, 1))
    pointer = interpolation.tke.data.ptr
    np.testing.assert_array_equal(interpolation.tke.get(), values['tke'])
    with pytest.raises(ValueError, match='turbulence cell grid'):
        interpolation.load_fields(wind, dict(values, tke=np.zeros((2, 2, 2))))
    np.testing.assert_array_equal(interpolation.tke.get(), values['tke'])
    interpolation.load_fields(wind, {name: values[name] for name in values if name != 'tke'})
    assert interpolation.tke.data.ptr == pointer
    np.testing.assert_array_equal(interpolation.tke.get(), 0.)
    interpolation.load_fields(wind, dict(values, tke=cuda.asarray(field)))
    assert interpolation.tke.data.ptr == pointer
    np.testing.assert_array_equal(interpolation.tke.get(), field)


def test_cuda_plume_runner_advects_and_bounds_capacity(cuda):
    wind = {name: np.full((8, 8, 8), 2. if name == 'u' else 0., dtype=np.float32) for name in 'uvw'}
    levels = {'txx': 1., 'tyy': 1., 'tzz': 1., 'CoEps': .25}
    turbulence = {name: np.full((7, 7, 7), levels.get(name, 0.), dtype=np.float32)
                  for name in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'div_tau_x', 'div_tau_y',
                               'div_tau_z', 'CoEps', 'nuT')}
    interpolation = CudaPlumeInterpolation(wind, turbulence, (12., 15., 20.))
    collection = CudaPlumeCollection((0., 0., 0.), (10., 15., 20.), (8, 6, 6))
    model = CudaPlumeRun(interpolation, collection, ((6., 78.), (7.5, 97.5), (0., 110.)), 12)
    for step in range(5):
        model.step(4, collect=step >= 1)
    assert (model.steps, model.requested, model.inserted) == (5, 20, 12)
    np.testing.assert_array_equal(model.particles.ids.get(), np.arange(12))
    np.testing.assert_array_equal(model.particles.state.get(), ACTIVE)
    positions = model.particles.positions.get()
    assert positions[:, 0].mean() > 20.
    assert np.all((positions > model.bounds[:, 0]) & (positions < model.bounds[:, 1]))
    assert collection.averaging_s == 4.
    assert collection.copyback().sum() == 44

    rollover = CudaPlumeRun(interpolation, collection, model.bounds, 4)
    rollover.requested = (1 << 32)-2
    rollover.step(4, collect=False)
    np.testing.assert_array_equal(rollover.particles.ids.get(), [4294967294, 4294967295, 0, 1])


@pytest.mark.parametrize(('shape', 'sites'), [
    ((5, 11), [(0, 0), (4, 8)]),
    ((53, 19), [(0, 18), (21, 13), (49, 2)]),
    ((9, 7), []),
    ((3, 1030), [(1, 0)]),
    ((5, 4101), [(1, 1024), (4, 2047)]),
])
def test_cuda_initial_fire_front(cuda, shape, sites):
    mask = np.zeros(shape, dtype=bool)
    for site in sites:
        mask[site] = True
    actual = cuda_front(mask)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, cpu_front(mask).astype(np.float32))


def test_cuda_pressure_on_variable_stencil_with_mirrored_bottom(cuda):
    rng = np.random.default_rng(714)
    shape = (10, 12, 13)
    fields = [rng.uniform(.4, 1.4, shape) for _ in range(6)]
    fluid = np.ones(shape, dtype=bool)
    fluid[3:6, 5:8, 4:7] = False
    stencil = Stencil(*fields, fluid)
    reference = np.zeros(shape)
    reference[stencil.active] = rng.normal(size=len(stencil.indices))
    stencil.boundaries(reference)
    rhs = -stencil.apply(reference)
    p, report = CudaRedBlackSOR(NumericPolicy(relative_tolerance=1e-10)).solve(stencil, rhs)
    assert report.converged and report.iterations > 0
    np.testing.assert_allclose(p, reference, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(p[0], p[1])
    p_initial, report = CudaRedBlackSOR().solve(stencil, rhs, initial=p)
    assert report.iterations == 0
    np.testing.assert_array_equal(p, p_initial)


def test_cuda_does_not_report_unconverged_iterate_as_solution(cuda):
    shape = (8, 8, 8)
    stencil = Stencil(*[np.ones(shape)]*6, np.ones(shape, dtype=bool))
    solver = CudaRedBlackSOR(NumericPolicy(max_iterations=1))
    with pytest.raises(RuntimeError, match='Poisson convergence failed'):
        solver.solve(stencil, np.ones(shape))


@pytest.mark.parametrize('backend', ['cuda-global-memory', 'cuda-shared-memory'])
@pytest.mark.parametrize('width, copied', [(24, 625), (32, 1024)])
def test_legacy_cuda_ground_copy_uses_native_launch_span(backend, width, copied):
    root = Path(__file__).resolve().parents[2]
    xml = root/'data/cases/empty_west/winds.xml'
    case = read_wind_case(xml, compute_turbulence=False, poisson_backend=backend, legacy_turbulence_xml=True)
    source = np.arange(4*width*width, dtype=np.float32)
    pressure = source.copy().reshape(4, width, width)
    case.poisson._apply_ground_boundary(pressure)
    plane = width*width
    expected = source.copy()
    expected[:copied] = source[plane:plane+copied]
    np.testing.assert_array_equal(pressure.ravel(), expected)


@pytest.mark.parametrize('backend', ['cuda-dynamic-legacy', 'cuda-global-memory', 'cuda-shared-memory',
                                     'cuda-device-parallel'])
def test_cuda_legacy_mirrors_bottom_after_both_colors(cuda, backend):
    if backend == 'cuda-device-parallel' and not (
        not cuda.cuda.runtime.is_hip and 35 <= int(cuda.cuda.Device().compute_capability) < 90
        and cuda.cuda.nvrtc.getVersion()[0] in (11, 12)
    ):
        pytest.skip('CUDA CDP1 device and toolkit are required')
    shape = (4, 5, 5)
    stencil = Stencil(*[np.ones(shape)]*6, np.ones(shape, dtype=bool))
    rhs = np.zeros(shape)
    rhs[1, 1, 2] = -6.
    solver = make_poisson(backend, NumericPolicy(max_iterations=1))
    pressure, report = solver.solve(stencil, rhs)
    # Red (1,1,2) is followed by black (1,1,1) using the red value.
    np.testing.assert_allclose(pressure[1, 1, [1, 2]], [1.78**2/6, 1.78], rtol=1e-6)
    np.testing.assert_array_equal(pressure[0], pressure[1])
    assert report.converged is False
    assert report.iterations == 1
    assert report.update_linf == float(pressure[1, 1, 2])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_projection_matches_host_stages_on_stretched_rectangular_grid(cuda, dtype):
    rng = np.random.default_rng(98)
    domain = Domain((12, 11, 9), (1.3, .8, .6), tuple(np.linspace(.6, 1.4, 9)))
    workspace = domain.workspace()
    for name in 'efghmn':
        workspace.arrays[name] = rng.uniform(.3, 1.2, domain.cell_shape).astype(dtype)
    for name in ('u0', 'v0', 'w0'):
        workspace.arrays[name] = rng.normal(size=domain.face_shape).astype(dtype)
    workspace.arrays['icellflag'][2:5, 4:7, 3:6] = 0
    workspace.arrays['icellflag'][1, 0, -1] = 2
    physics = PhysicalCoefficients(alpha_horizontal=1.2, alpha_vertical=.7)
    solver = CudaRedBlackSOR(NumericPolicy(relative_tolerance=1e-10))
    expected_rhs = rhs_from_workspace(workspace, physics)
    expected_pressure, _ = solver.solve(workspace.stencil(), expected_rhs)
    expected_velocity = corrected_velocity(workspace, expected_pressure, physics)

    device_workspace = replace(workspace, arrays={name: cuda.asarray(value) for name, value in workspace.arrays.items()})
    rhs = rhs_from_workspace(device_workspace, physics, array_module=cuda).get()
    np.testing.assert_allclose(rhs, expected_rhs, rtol=0, atol=1e-13)
    actual, pressure, report = project(workspace, solver, physics)
    np.testing.assert_allclose(pressure, expected_pressure, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(pressure[0], pressure[1])
    assert workspace.stencil().residual(pressure, expected_rhs) <= 1.01*report.residual_linf+1e-13
    for name in 'uvw':
        np.testing.assert_allclose(actual[name], expected_velocity[name], rtol=0, atol=1e-11)


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_cuda_turbulence_gradient_preserves_wall_modes(cuda, mode):
    rng = np.random.default_rng(192)
    domain = Domain((14, 13, 24), (1.5, .8, .7), tuple(np.linspace(.7, 1.4, 24)))
    workspace = domain.workspace()
    workspace.arrays['icellflag'][3:6, 3:5, 4:8] = 0
    workspace.arrays['icellflag'][1:3, 7:9, 7:9] = 2
    velocity = {name: rng.normal(size=domain.face_shape)+(1.5 if name == 'u' else 0.) for name in 'uvw'}
    cpu = LocalTurbulence(building_wall=mode, terrain_wall=mode)
    gpu = replace(cpu, gradient_backend='cuda')
    np.testing.assert_allclose(cpu.velocity_gradient(workspace, velocity), gpu.velocity_gradient(workspace, velocity),
                               atol=1e-13, rtol=0)
    expected, _ = cpu.build(workspace, velocity)
    actual, metadata = gpu.build(workspace, velocity)
    assert metadata['gradient_backend'] == 'cuda'
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=0, err_msg=name)


def test_cuda_turbulence_rejects_incompatible_face_shape(cuda):
    domain = Domain((6, 6, 6), (1., 1., 1.))
    velocity = {name: np.zeros(domain.face_shape) for name in 'uvw'}
    velocity['u'] = velocity['u'][:-1]
    with pytest.raises(ValueError, match='matching staggered face and cell grids'):
        LocalTurbulence(gradient_backend='cuda', building_wall=0, terrain_wall=0).velocity_gradient(domain.workspace(), velocity)


def test_cuda_rays_traverse_multiple_leaves_and_keep_edge_hits(cuda):
    quad = [[[0, 0, 0], [2, 0, 0], [2, 2, 0]], [[0, 0, 0], [2, 2, 0], [0, 2, 0]]]
    triangles = np.asarray(quad+[[[1, 1, 1]]*3], dtype=float)
    mesh = TriangleMesh(triangles, leaf_size=1)
    points = np.array([[1., 1., 2.], [1., 1., 0.], [4., 4., 2.], [0., 0., 2.], [1., 1., -2.]])
    rays = np.array([[0., 0., -1.], [0., 0., 1.], [1., 0., 0.]])
    np.testing.assert_array_equal(cuda_ray_distances(mesh, points, rays), [2., 0., np.inf, 2., 2.])


def test_optix_ray_budget_preserves_all_point_results(cuda, monkeypatch):
    pytest.importorskip('optix')
    from plumeswift import optix_ray_mixing

    mesh = TriangleMesh(np.array([[(0., 0., 0.), (4., 0., 0.), (0., 4., 0.)]]))
    points = np.column_stack((np.ones(5), np.ones(5), np.arange(1., 6.)))
    rays = np.array([(0., 0., -1.), (0., 0., 1.), (1., 0., 0.)])
    reference = optix_ray_mixing.optix_ray_distances(mesh, points, rays)
    monkeypatch.setattr(optix_ray_mixing, '_MAX_RAYS_PER_LAUNCH', 6)
    tiled = optix_ray_mixing.optix_ray_distances(mesh, points, rays)
    np.testing.assert_array_equal(reference, np.arange(1., 6.))
    np.testing.assert_array_equal(tiled, reference)


@pytest.mark.parametrize('backend', ['cuda', 'optix'])
def test_cuda_method_three_xml_reuses_method_four_file(cuda, tmp_path, backend):
    if backend == 'optix':
        pytest.importorskip('optix')
    root = Path(__file__).resolve().parents[2]
    tree = ET.parse(root/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '10 10 19'
    section = tree.find('turbParams')
    section.find('method').text = '3'
    section.find('samples').text = '17'
    section.find('save').text = 'true'
    ET.SubElement(section, 'rayBackend').text = backend
    ET.SubElement(section, 'LMfile').text = 'distances.nc'
    path = tmp_path/'wind.xml'
    tree.write(path)
    case = read_wind_case(path)
    assert case.mixing_ray_backend == backend
    result = case.run()
    section.remove(section.find('rayBackend'))
    section.find('method').text = '4'
    section.find('save').text = 'false'
    tree.write(path)
    reloaded = read_wind_case(path).run()
    for key in ('mixingLengths', 'Lm', 'txx'):
        np.testing.assert_array_equal(result['turbulence'][key], reloaded['turbulence'][key])


def test_cuda_fire_potential_near_axis_uses_zero_radial_index(cuda):
    radial, vertical = (cuda.asarray(values, dtype=cuda.float64)
                        for values in ([[4., 5.], [6., 7.]], [[1., 2.], [3., 4.]]))
    g = cuda.zeros(2, dtype=cuda.float64)
    velocity = cuda.zeros((3, 2, 1, 3), dtype=cuda.float64)
    _fire_potential_kernel()[(1,), (4,)](
        velocity, radial, vertical, g, g, *map(np.int64, (3, 1, 1, 2, 0, 0)),
        *map(np.float64, (4e-6, 1., 1., 1e-6, 1., 0., 2., 1.)))
    np.testing.assert_array_equal(velocity[:, 1, 0].get(), [[0.]*3, [0.]*3, [6.]*3])


@pytest.mark.parametrize('heat', [100000., .002])
def test_cuda_fire_potential_preserves_layers_and_finite_far_axis(cuda, heat):
    domain = Domain((18, 17, 9), (1., 1., 1.))
    workspace = domain.workspace()
    ground = workspace.arrays['terrain']
    fire = SimpleNamespace(terrain=ground, burn_flag=np.zeros(ground.shape, dtype=int), heat=np.zeros_like(ground),
                           dx=1., dy=1., dz=1.)
    fire.burn_flag[3, 4], fire.heat[3, 4] = 1, heat
    table = CudaHeatPotential(np.full((61, 31), .2), np.full((61, 31), .6), np.array([.1, .2, .3]),
                             np.array([.2, .1, .3]), np.arange(31.), np.arange(61.))
    blocks, expected_mix = table._merged_sources(fire, workspace, cuda=True)
    potential, mix = table.apply(fire, workspace)
    expected = np.zeros_like(potential)
    y, x = np.indices(ground.shape)
    for first, last, cx, cy, origin, speed, length in blocks:
        for layer in range(first, last):
            z = max(0., (layer-origin)/length)
            expected[:, layer] += table._sample((x-cx)/length, (y-cy)/length, z, speed)
    np.testing.assert_array_equal(mix, expected_mix)
    np.testing.assert_allclose(potential, expected, rtol=1e-12, atol=1e-12)
    assert not np.any(potential[:, :2])
    assert potential[2, 2, 3, 4] > 0.
    assert not np.any(potential[:2, :, 3, 4])
