import numpy as np
import pytest
from plumeswift.sources import Source,Point,Cube,Line,SphereShell,ContinuousRelease,InstantaneousRelease
from plumeswift.boundaries import AxisBoundary,DomainBoundaries
from plumeswift.interpolation import Trilinear
from plumeswift.statistics import SensorBoxes,TimeAveragedSensors
from plumeswift.lh2 import LH2Momentum
from plumeswift.thermo import LegacyEquilibrium

def test_trilinear_exact_for_affine_field_on_stretched_grid():
    x=np.arange(6.);y=np.arange(5.)*2;z=np.array([0.,.5,2.,4.])
    zz,yy,xx=np.meshgrid(z,y,x,indexing='ij');field=2*xx+3*yy-4*zz+7
    p=np.array([[1.1,3.2,.8],[4.8,7.7,3.7]])
    assert np.max(abs(Trilinear(x,y,z)(field,p)-(2*p[:,0]+3*p[:,1]-4*p[:,2]+7)))<1e-12

def test_sources_geometry_and_exact_mass_under_step_refinement():
    rng=np.random.default_rng(1)
    sphere=SphereShell((1,2,3),4).sample(1000,rng)
    assert np.max(abs(np.linalg.norm(sphere-(1,2,3),axis=1)-4))<1e-12
    cube=Cube((0,0,0),(1,2,3)).sample(1000,rng)
    assert np.all(cube>=0)and np.all(cube<=(1,2,3))
    for dt in [.1,.2]:
        source=Source(Point((0,0,1)),ContinuousRelease(.5,10,start_s=.25,end_s=1.85))
        mass=0
        for t in np.arange(0,2,dt):mass+=source.emit(t,dt,rng)[1].sum()
        assert mass==pytest.approx(.8)

def test_reflection_matches_iterative_original_for_large_overshoots():
    p=np.array([-3.,-2.,-1.,-.1,.1,1.,1.1,2.,3.,4.,8.2]);v=np.ones(len(p))
    expected=p.copy();ev=v.copy()
    for n in range(len(p)):
        while expected[n]<0 or expected[n]>1:
            expected[n]=(2-expected[n])if expected[n]>1 else -expected[n];ev[n]*=-1
    actual,av,ok=AxisBoundary(0,1,'reflection').apply(p,v)
    assert np.allclose(actual,expected)and np.array_equal(av,ev)and ok.all()

def test_boundary_exiting_is_not_reactivated_by_another_axis():
    boundary=DomainBoundaries([AxisBoundary(0,1),AxisBoundary(0,1,'periodic'),AxisBoundary(0,1,'reflection')])
    _,_,ok=boundary.apply(np.array([[2.,2.,2.]]),np.ones((1,3)))
    assert not ok[0]

def test_lh2_marker_subdivision_has_same_force():
    coupling=LH2Momentum(LegacyEquilibrium());results=[]
    for n in [1,100]:
        position=np.broadcast_to([0.,0.,1.],(n,3));mass=np.full(n,.2/n);drift=np.zeros((n,3))
        velocity,diagnostic=coupling.update(position,mass,drift,np.zeros(n,dtype=int),np.array([1.]),.1)
        results.append(velocity[0]);assert diagnostic['active_mass_kg']==pytest.approx(.2)
    assert np.allclose(results[0],results[1])

def test_sensor_mass_count_and_partial_time_window():
    boxes=SensorBoxes(np.array([[0.,0.,1.]]),(2,2,2));observer=TimeAveragedSensors(boxes,.25,.75)
    for t in [0.,.5]:observer.observe(t,.5,np.array([[0.,0.,1.],[4.,4.,4.]]),np.array([8.,80.]))
    assert observer.sampled_s==pytest.approx(.5)
    assert observer.mean()[0]==pytest.approx(1.)

def test_sensor_height_filter_keeps_float64_boundary_precision():
    boxes = SensorBoxes(np.array([[0., 0., 1.], [0., 0., 1.+2e-8]]), (2., 2., 2e-8))
    positions = np.array([[0., 0., 1.], [1., 0., 1.]], dtype=np.float32)
    concentrations = boxes.concentrations(positions, np.array([8., 80.]))
    np.testing.assert_array_equal(concentrations, [1e8, 0.])


def uniform_fields():
    from plumeswift.interpolation import FieldSampler
    coordinates={'x':np.arange(10)+.5,'y':np.arange(10)+.5,'z':np.arange(11)-.5,'z_face':np.arange(12)-1.}
    wind={key:np.full((12,11,11),1. if key=='u'else 0.)for key in 'uvw'}
    turbulence={key:np.full((11,10,10),.01 if key in ['txx','tyy','tzz']else .02 if key=='CoEps'else 0.)
                for key in ['txx','txy','txz','tyy','tyz','tzz','CoEps','nuT','div_tau_x','div_tau_y','div_tau_z']}
    return FieldSampler(coordinates,wind,turbulence,mixlength=np.full((11,10,10),10.))

def test_python_particle_pipeline_mass_and_mean_advection():
    from plumeswift.particles import ParticleIntegrator,ParticleSimulation
    from plumeswift.walls import NoWalls
    fields=uniform_fields();rng=np.random.default_rng(123)
    boundary=DomainBoundaries([AxisBoundary(0,10),AxisBoundary(0,10),AxisBoundary(0,10)])
    integrator=ParticleIntegrator(fields,boundary,NoWalls(),rng,scheme='midpoint')
    source=Source(Point((2,5,5)),InstantaneousRelease(1.,2000))
    simulation=ParticleSimulation([source],integrator)
    result=simulation.run(2.,.1)
    assert result['mass_balance_error_kg']==pytest.approx(0,abs=1e-12)
    assert result['active_particles']==2000
    assert np.max(abs(simulation.batch.positions.mean(axis=0)-[4,5,5]))<.02

def test_stationary_field_processes_preserve_particle_trajectories():
    from plumeswift.particle_sampling_pool import ParticleSamplingPool
    from plumeswift.particles import ParticleIntegrator, ParticleSimulation
    from plumeswift.walls import VoxelWalls

    fields = uniform_fields()
    boundaries = DomainBoundaries([AxisBoundary(0, 10) for _ in range(3)])
    boxes = SensorBoxes(np.array([[2., 5., 5.]]), (2, 2, 2))
    solid = np.zeros(fields.mixlength.shape, dtype=bool)
    solid[:, :, 3] = True

    def run(pool=None):
        integrator = ParticleIntegrator(fields, boundaries, VoxelWalls(fields, solid),
                                        np.random.default_rng(123), scheme='midpoint', sampling_pool=pool)
        source = Source(Point((2.7, 5, 5)), InstantaneousRelease(1., 251))
        observer = TimeAveragedSensors(boxes, 0., .4)
        simulation = ParticleSimulation([source], integrator, [observer])
        accounting = simulation.run(.4, .1)
        return simulation.batch, accounting, observer.mean()

    serial, serial_accounting, serial_mean = run()
    with ParticleSamplingPool(fields, 2, min_particles=1) as pool:
        parallel, parallel_accounting, parallel_mean = run(pool)
        assert pool.parallel_batches > 0
    assert np.all(serial.positions[:, 0] < 3.)
    for name in ('positions', 'fluctuations', 'mass', 'previous_stress'):
        np.testing.assert_array_equal(getattr(serial, name), getattr(parallel, name))
    np.testing.assert_array_equal(serial_mean, parallel_mean)
    assert serial_accounting == parallel_accounting


def test_voxel_wall_traversal_cannot_tunnel_through_thin_obstacle():
    from plumeswift.walls import VoxelWalls
    fields=uniform_fields();solid=np.zeros((11,10,10),dtype=bool);solid[:,:,3]=True
    walls=VoxelWalls(fields,solid)
    position,fluctuation=walls.reflect(np.array([[2.5,5.,5.]]),np.array([[3.,0.,0.]]),np.array([[1.,0.,0.]]))
    assert position[0,0]==pytest.approx(.5,abs=1e-8)
    assert fluctuation[0,0]==-1


def test_compiled_transport_matches_numpy_on_stretched_grid_and_walls():
    pytest.importorskip('numba')
    from plumeswift.interpolation import CompiledTrilinear
    from plumeswift.walls import VoxelWalls

    rng = np.random.default_rng(72)
    axes = np.arange(6.), np.arange(5.)*2., np.array([0., .5, 2., 4.], dtype=np.float32)
    points = rng.uniform([0., 0., 0.], [5., 8., 4.], (200, 3))
    face = np.array([2., 4., .5])
    points = np.vstack((points, face, np.nextafter(face, -np.inf), np.nextafter(face, np.inf)))
    values = rng.random((4, 5, 6, 7)).astype(np.float32)
    np.testing.assert_array_equal(Trilinear(*axes)(values, points),
                                  CompiledTrilinear(*axes)(values, points))

    fields = uniform_fields()
    solid = np.zeros(fields.mixlength.shape, dtype=bool)
    solid[:, :, 3] = True
    solid[0] = True
    positions = rng.uniform([.1, .1, .1], [9.9, 9.9, 8.9], (300, 3))
    displacement = rng.normal(0., 2., positions.shape)
    fluctuations = rng.normal(size=positions.shape)
    expected = VoxelWalls(fields, solid).reflect(positions, displacement, fluctuations)
    actual = VoxelWalls(fields, solid, compiled=True).reflect(positions, displacement, fluctuations)
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(reference, result)

def test_long_simulation_clock_has_exact_release_count():
    release=ContinuousRelease(.8283,100)
    total=sum(release.count_and_mass(i*.2,.2)[0]for i in range(1875))
    assert total==37500

def test_packed_interpolation_is_only_a_numerical_optimization():
    fields=uniform_fields();rng=np.random.default_rng(71)
    for field in fields.turb.values():field[:]=rng.random(field.shape)
    # Rebuild the immutable packed representation after making this synthetic fixture.
    from plumeswift.interpolation import FieldSampler
    fields=FieldSampler(fields.c,fields.wind,fields.turb,mixlength=fields.mixlength)
    p=rng.uniform([1,1,1],[8,8,8],(200,3))
    actual=fields.sample_cells(fields._packed,p)
    expected=np.column_stack([fields.sample_cells(fields.mixlength if key=='mixlength'else fields.turb[key],p)
                              for key in fields._packed_names])
    assert np.max(abs(actual-expected))<1e-12

def test_release_shorter_than_particle_interval_does_not_drop_mass():
    source=Source(Point((0,0,1)),ContinuousRelease(.5,1,end_s=.01))
    positions,masses=source.emit(0,.2,np.random.default_rng(0))
    assert len(positions)>0
    assert masses.sum()==pytest.approx(.005)
    assert source._pending_mass==0

def test_zero_mass_markers_do_not_make_lh2_force_nonfinite():
    coupling=LH2Momentum(LegacyEquilibrium());drift=np.array([[1.,2.,3.]])
    updated,diagnostic=coupling.update(np.array([[0.,0.,1.]]),np.array([0.]),drift,np.array([0]),np.array([1.]),.1)
    assert np.array_equal(updated,drift)
    assert diagnostic['active_mass_kg']==0
