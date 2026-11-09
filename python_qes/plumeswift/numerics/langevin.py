"""Batched generalized Langevin integration, same QES GLE equations.

References: QES-Public GLE_Solver.cpp, VectorMath.h, GPL-3.0; University
of Utah/University of Minnesota Duluth and the contributors listed upstream.
The equations and stress-invariant repair are preserved. Linear systems are
solved directly rather than explicitly inverting A. No native QES is called.
"""
import numpy as np

def invariants(tau):
    a=np.asarray(tau)
    tr=np.trace(a,axis1=-2,axis2=-1)
    second=(tr*tr-np.einsum('...ij,...ji->...',a,a))/2
    return np.stack([tr,second,np.linalg.det(a)],axis=-1)

def realizable(tau,tolerance=1e-5, *, compiled=False):
    a=np.asarray(tau,dtype=float).copy()
    if a.shape[-2:]!=(3,3)or not np.isfinite(a).all():raise ValueError("Stress tensor must be finite 3x3")
    transpose = np.swapaxes(a, -1, -2)
    symmetric = compiled and np.array_equal(a, transpose)
    if not symmetric and not np.allclose(a, transpose, atol=1e-12):
        raise ValueError("Stress must be symmetric")
    flat = a.reshape(-1, 3, 3)
    candidates = None
    if compiled and symmetric and 0 < tolerance < np.inf:
        from .compiled_gle import stress_needs_invariants

        needs_check = stress_needs_invariants(flat, tolerance)
        if not np.any(needs_check):
            return a
        if not np.all(needs_check):
            candidates = np.flatnonzero(needs_check)
    inv = invariants(flat if candidates is None else flat[candidates])
    bad = ~np.all(inv > tolerance, axis=1)
    if not bad.any():return a
    indices = np.flatnonzero(bad) if candidates is None else candidates[bad]
    original=flat[indices];iv=inv[bad];b=4/3*iv[:,0];c=iv[:,1]
    discriminant=b*b-16/3*c
    ks=1.01*(-b+np.sqrt(np.maximum(discriminant,0)))/(8/3)
    ks=np.where((ks<tolerance)|(discriminant<0),.5*abs(iv[:,0]),ks)
    # Upstream ks=0 remains zero forever for a zero tensor. A numerical floor
    # repairs that degeneracy, without adding an alternative turbulence model.
    ks=np.maximum(ks,tolerance)
    shifted=original+np.eye(3)[None,:,:]*(2/3*ks)[:,None,None]
    for _ in range(1000):
        failing=np.any(invariants(shifted)<tolerance,axis=1)
        if not failing.any():break
        ks[failing]*=1.05
        shifted[failing]=original[failing]+np.eye(3)[None,:,:]*(2/3*ks[failing])[:,None,None]
    else:raise FloatingPointError("Stress invariant repair did not converge")
    flat[indices]=shifted
    return a

def _solve_gle(matrix, rhs, compiled):
    if compiled:
        from .compiled_gle import solve_three_by_three

        result, fallback = solve_three_by_three(matrix, rhs)
        if np.any(fallback):
            result[fallback] = np.linalg.solve(matrix[fallback], rhs[fallback, :, None])[..., 0]
        return result
    return np.linalg.solve(matrix, rhs[..., None])[..., 0]


def implicit_gle(velocity_old,tau_old,tau_current,flux_divergence,coeps,dt,normal_draws,tolerance=1e-5,
                 scheme="backward-euler", *, native_determinant_gate=False, compiled=False):
    """Return updated velocity and repaired stress; draws supplied by caller.

    Deterministic common random numbers enable regression against upstream.
    dt may be per particle. No random generator or global state is hidden here.
    """
    v=np.asarray(velocity_old,dtype=float);n=v.shape[0]
    tau=realizable(tau_current,tolerance,compiled=compiled)
    delta=np.asarray(dt,dtype=float)
    if delta.ndim==0:delta=np.full(n,float(delta))
    ce=np.broadcast_to(coeps,(n,)).astype(float)
    if np.any(delta<=0)or np.any(ce<0):raise ValueError("dt>0 and CoEps>=0 required")
    det_tau = abs(np.linalg.det(tau)) if native_determinant_gate else None
    if callable(normal_draws):
        # QES rejects a singular stress before consuming any random numbers.
        sample = normal_draws
        valid_stress = np.isfinite(det_tau) & (det_tau >= 1e-10) if det_tau is not None else np.ones(n, dtype=bool)
        normal_draws = np.zeros_like(v)
        normal_draws[valid_stress] = sample(size=(int(valid_stress.sum()), 3))
    noise=np.sqrt(ce*delta)[:,None]*normal_draws
    forcing=.5*np.asarray(flux_divergence)*delta[:,None]
    old=np.asarray(tau_old)
    isotropic_noise=np.eye(3)[None,:,:]*(ce*delta)[:,None,None]
    if scheme=="backward-euler":
        # Algebraically identical to the original A^-1 b: substitute v=tau*y.
        # This removes BOTH stress inversion and dTau/dt. The transformed
        # matrix is symmetric and is positive definite for valid old/current
        # stresses and nonnegative dissipation.
        A=.5*(tau+old+isotropic_noise)
        b=v+forcing+noise
    elif scheme=="midpoint":
        # Same SDE. For constant stresses this time integrator preserves the
        # stationary covariance exactly; backward Euler damps that covariance.
        correction=.25*(tau-old-isotropic_noise)
        A=tau-correction
        previous_normalized=np.linalg.solve(tau,v[...,None])[...,0]
        b=v+np.einsum('nij,nj->ni',correction,previous_normalized)+forcing+noise
    else:raise ValueError(f"Unsupported GLE numerical scheme: {scheme}")
    if native_determinant_gate:
        if scheme != 'backward-euler':
            raise ValueError('Native determinant gate requires backward Euler')
        # Native checks det(tau) and det(-A @ inv(tau)) before either inversion.
        det_a = abs(np.linalg.det(A))
        valid = np.isfinite(det_tau) & np.isfinite(det_a) & (det_tau >= 1e-10) & (det_a >= 1e-10*det_tau)
        normalized = np.full_like(v, np.nan)
        try:
            normalized[valid] = _solve_gle(A[valid], b[valid], compiled)
        except np.linalg.LinAlgError:
            for index in np.flatnonzero(valid):
                try:
                    normalized[index] = np.linalg.solve(A[index], b[index])
                except np.linalg.LinAlgError:
                    pass
    else:
        normalized=_solve_gle(A,b,compiled)
    updated=np.einsum('nij,nj->ni',tau,normalized)
    if not native_determinant_gate and not np.isfinite(updated).all():raise FloatingPointError("Non-finite GLE solution")
    return updated,tau
