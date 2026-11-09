import numpy as np
import pytest
import json
from dataclasses import asdict
from plumeswift.numerics.poisson import Stencil,RedBlackSOR,KrylovPoisson
from plumeswift.numerics.langevin import implicit_gle,realizable,invariants
from plumeswift.numerics.matrix3 import inverse, native_inverse, to_column_major
from plumeswift.contracts import NumericPolicy


def test_native_matrix_inverse_replays_nonsymmetric_fallback():
    matrix = np.array([[2., 1., 4.], [0., 3., 5.], [1., 0., 2.]])
    native_cofactors = [6., 5., -3., -0., 0., -0., -12., -10., 6.]
    np.testing.assert_array_equal(to_column_major(native_inverse(matrix)), np.array(native_cofactors)/1e11)
    np.testing.assert_allclose(matrix@inverse(matrix), np.eye(3), atol=1e-14)

def test_poisson_manufactured_and_interchangeability():
    shape=(10,12,14);coefficients=[np.ones(shape)for _ in range(6)]
    s=Stencil(*coefficients,np.ones(shape,dtype=bool));k,j,i=np.indices(shape)
    exact=np.sin(np.pi*i/(shape[2]-1))*np.sin(np.pi*j/(shape[1]-1))*np.cos(np.pi*(k-.5)/(2*(shape[0]-1.5)))
    s.boundaries(exact);rhs=-s.apply(exact)
    policy=NumericPolicy(relative_tolerance=1e-10,absolute_tolerance=1e-11)
    answers=[]
    for solver in [RedBlackSOR(policy),KrylovPoisson(policy)]:
        p,report=solver.solve(s,rhs);assert report.converged
        assert report.residual_linf<=policy.absolute_tolerance+policy.relative_tolerance*np.max(abs(rhs[s.active]))
        assert np.max(abs(p-exact))<1e-8
        answers.append(p)
        assert json.loads(json.dumps(asdict(report),allow_nan=False))['converged'] is True
    assert np.max(abs(answers[0]-answers[1]))<1e-8

def test_gle_upstream_algebra_with_common_random_numbers():
    rng=np.random.default_rng(123);n=100
    m=rng.normal(size=(n,3,3));tau=m@np.swapaxes(m,-1,-2)+np.eye(3)
    old=tau*.97;v=rng.normal(size=(n,3));flux=rng.normal(size=(n,3));noise=rng.normal(size=(n,3))
    ce=np.full(n,.13);dt=np.full(n,.02)
    inv=np.linalg.inv(tau);d=(tau-old)/dt[:,None,None]
    # Independent expression transcribed from original A and b definitions.
    a=-np.eye(3)+.5*(-ce[:,None,None]*inv+d@inv)*dt[:,None,None]
    b=-v-.5*flux*dt[:,None]-np.sqrt(ce*dt)[:,None]*noise
    expected=np.einsum('nij,nj->ni',np.linalg.inv(a),b)
    actual,_=implicit_gle(v,old,tau,flux,ce,dt,noise)
    assert np.max(abs(expected-actual))<1e-12


def test_compiled_gle_preserves_stress_repair_and_reference_fallback():
    pytest.importorskip('numba')
    from plumeswift.numerics.compiled_gle import solve_three_by_three
    from plumeswift.numerics.langevin import _solve_gle

    rng = np.random.default_rng(58)
    samples = rng.normal(size=(100, 3, 3))
    stress = samples @ samples.transpose(0, 2, 1) + np.eye(3)
    stress[:3] = (np.eye(3), np.eye(3)*.0215, np.zeros((3, 3)))
    np.testing.assert_array_equal(realizable(stress, compiled=True), realizable(stress))
    velocity, flux, noise = rng.normal(size=(3, len(stress), 3))
    args = (velocity, stress*.97, stress, flux, .13, .02, noise)
    expected, expected_stress = implicit_gle(*args)
    actual, actual_stress = implicit_gle(*args, compiled=True)
    np.testing.assert_array_equal(actual_stress, expected_stress)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)

    weak = np.diag([1., 2., 1e-12])[None]
    rhs = np.array([[1., 2., 3.]])
    assert solve_three_by_three(weak, rhs)[1][0]
    np.testing.assert_array_equal(_solve_gle(weak, rhs, True), _solve_gle(weak, rhs, False))

def test_midpoint_preserves_stationary_covariance_analytically():
    # Constant isotropic stress; this is a numerical property of the unchanged SDE.
    variance=.09;ce=.18;dt=.5;a=.5*ce/variance
    tau=np.eye(3)[None,:,:]*variance;zero=np.zeros((1,3))
    v,_=implicit_gle(np.ones((1,3)),tau,tau,zero,ce,dt,zero,scheme='midpoint')
    eta,_=implicit_gle(zero,tau,tau,zero,ce,dt,np.ones((1,3)),scheme='midpoint')
    stationary=float(eta[0,0]**2/(1-v[0,0]**2))
    assert abs(stationary-variance)<1e-12
    backward=ce*dt/((1+a*dt)**2-1)
    assert backward<variance

def test_stress_repair_preserves_valid_stress_and_repairs_zero():
    valid=np.diag([1.,2.,3.])[None,:,:]
    assert np.array_equal(realizable(valid),valid)
    repaired=realizable(np.zeros((2,3,3)))
    assert np.all(invariants(repaired)>=1e-5)

def test_nonfinite_stress_fails_explicitly():
    with pytest.raises(ValueError):realizable(np.eye(3)[None,:,:]*np.nan)
