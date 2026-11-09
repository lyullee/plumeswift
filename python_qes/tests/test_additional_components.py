import numpy as np
import pytest
from plumeswift.assembly import default_registry, ComponentChoice
from plumeswift.deposition import vegetation_survival, ground_survival, apply_survival
from plumeswift.numerics.vector3 import dot, length, reflect


def test_upstream_cpu_vector_class_arithmetic():
    a = np.array([1., 2., 3.], dtype=np.float32)
    b = a.copy()
    np.testing.assert_array_equal(a, b)
    c = a+b
    np.testing.assert_array_equal(c, [2., 4., 6.])
    c += a
    np.testing.assert_array_equal(c, [3., 6., 9.])
    c /= np.float32(3)
    np.testing.assert_array_equal(c, a)
    c *= np.float32(3)
    np.testing.assert_array_equal(c, np.float32(3)*a)
    np.testing.assert_array_equal(c, a*np.float32(3))
    np.testing.assert_array_equal(a-b, np.zeros(3))
    assert dot(a, b) == (a*b).sum() == 14
    assert abs(length(np.ones(3, dtype=np.float32))-np.sqrt(3.)) < 1e-6
    x = np.array([2., 2., -2.], dtype=np.float32)
    normal = np.array([0., 0., 1.], dtype=np.float32)
    np.testing.assert_array_equal(reflect(x, normal), [2., 2., 2.])


def test_upstream_cpu_vector_speed_case_final_state():
    # The benchmark resets the vector every iteration, so one execution
    # reproduces the same final-state assertion as twenty million repeats.
    vector = np.array([1., 2., 3.], dtype=np.float32)
    normalized = vector/length(vector)
    x, y, z = vector
    norm = np.sqrt(x*x+y*y+z*z)
    np.testing.assert_array_equal(normalized, [x/norm, y/norm, z/norm])


def test_deposition_survival_units_and_mass_bookkeeping():
    displacements = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    velocity = np.tile([2., 0, 0], (3, 1))
    tau = np.tile(np.eye(3), (3, 1, 1))
    survival = vegetation_survival(displacements, velocity, 1000., 10e-6, tau, np.ones(3))
    assert survival[0] == 1.
    assert 0 < survival[2] < survival[1] < 1
    # Additive path exposure must compose multiplicatively.
    assert survival[2] == pytest.approx(survival[1]**2)
    mass = np.array([1., 1e-14, 2.])
    result = apply_survival(mass, survival, 1000., 10e-6)
    np.testing.assert_allclose(result.remaining_kg+result.deposited_kg+result.cutoff_discard_kg, mass)
    assert result.inactive[1]


def test_ground_zero_shear_limit_and_zero_motion():
    tau = np.zeros((2, 3, 3))
    survival = ground_survival([[0., 0, 0], [1., 0, 0]], [[0., 0, 0], [2., 0, 0]], tau, .01, .1, 1.)
    np.testing.assert_allclose(survival, [1., np.exp(-.05)])


def test_nested_building_pipeline_can_be_assembled_and_replaced():
    registry = default_registry()
    pipeline = registry.create('building.pipeline', stages=(
        ComponentChoice('building.upwind', {'mode': 1}),
        ComponentChoice('building.wake', {'mode': 2, 'rooftop_mode': 0}),
    ))
    assert len(pipeline.stages) == 2
    assert pipeline.stages[0].mode == 1
    # Empty ordered geometry is a valid no-op; a custom stage is independent.
    context = pipeline.apply(None, ())
    assert context.wakes == {}


def test_powerlaw_against_compiled_actual_upstream_methods():
    from pathlib import Path
    from plumeswift.powerlaw import PowerLawSampler
    path = Path(__file__).resolve().parents[2]/'results/python_port/powerlaw/native.csv'
    if not path.exists():
        pytest.skip('Run scripts/check_python_powerlaw.py for original-method reference')
    reference = np.loadtxt(path, delimiter=',')
    positions = np.zeros((len(reference), 3)); positions[:, 2] = reference[:, 0]
    sampler = PowerLawSampler.__new__(PowerLawSampler); sampler.mixlength = None
    tau, flux, ce, viscosity, _ = sampler.sample_turbulence_and_distance(positions)
    got = np.c_[positions[:, 2], sampler.sample_wind(positions), tau[:, 0, 0], tau[:, 0, 1], tau[:, 0, 2], tau[:, 1, 1], tau[:, 1, 2], tau[:, 2, 2], flux, ce]
    np.testing.assert_allclose(got, reference[:, :14], rtol=3e-7, atol=1e-7)
    assert viscosity is None
    assert np.all(reference[:, 14] == 123)
    assert np.linalg.eigvalsh(tau).min() < 0  # preserve native aggregate behavior
