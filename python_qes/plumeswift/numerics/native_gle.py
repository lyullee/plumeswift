"""Single-precision CPU QES GLE for replay, from GLE_Solver.cpp and VectorMath.h."""
import numpy as np

from .matrix3 import symmetric_invariants


def native_realizable(tau, tolerance):
    """Apply the native diagonal repair, including its bounded zero-stress case."""
    result = np.array(tau, dtype=np.float32, copy=True)
    tolerance = np.float32(tolerance)
    invariants = symmetric_invariants(result)
    repair = ~np.all(invariants > tolerance, axis=1)
    if not repair.any():
        return result

    original = result[repair]
    trace, minor_sum, _ = invariants[repair].T
    b = (np.float32(4)/np.float32(3))*trace
    with np.errstate(invalid='ignore'):
        ks = np.float32(1.01)*(-b+np.sqrt(b*b-(np.float32(16)/np.float32(3))*minor_sum))
    ks /= np.float32(8)/np.float32(3)
    bad_guess = (ks < tolerance) | np.isnan(ks)
    ks[bad_guess] = np.float32(.5)*np.abs(trace[bad_guess])
    indices = np.arange(3)
    shifted = original.copy()
    factor = np.float32(2)/np.float32(3)
    shifted[:, indices, indices] += (factor*ks)[:, None]
    for _ in range(1000):
        failing = (ks != 0) & np.any(symmetric_invariants(shifted) < tolerance, axis=1)
        if not failing.any():
            break
        # The native iteration multiplies by a double literal before storage.
        ks[failing] = ks[failing].astype(np.float64)*1.050
        shifted[:, indices, indices] = original[:, indices, indices]+(factor*ks)[:, None]
    result[repair] = shifted
    return result


def _invert(matrix):
    a, b, c, d, e, f, g, h, i = matrix.reshape(-1, 9).T
    m11, m12, m13 = e*i-f*h, d*i-f*g, d*h-e*g
    det = a*m11-b*m12+c*m13
    # The reference rejects only values strictly below the cutoff. NaN
    # determinants reach the fluctuation check after consuming random draws.
    valid = ~(abs(det.astype(np.float64)) < 1e-10)
    adjugate = np.column_stack((
        m11, -(b*i-c*h), b*f-e*c,
        -m12, a*i-c*g, -(a*f-c*d),
        m13, -(a*h-b*g), a*e-b*d,
    ))
    with np.errstate(invalid='ignore'):
        inverse = np.divide(adjugate, det[:, None], out=np.zeros_like(adjugate), where=valid[:, None])
    return inverse.reshape(matrix.shape), valid


def native_implicit_gle(velocity, previous, stress, flux, coeps, dt, normal_draws, tolerance):
    """Return native velocity and stored stress; use NaN velocity to signal rejection."""
    velocity = np.asarray(velocity, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    stress = native_realizable(stress, tolerance)
    inverse, valid = _invert(stress)
    updated = np.full_like(velocity, np.nan)
    stored = previous.copy()
    if not valid.any():
        return updated, stored
    draws = normal_draws(size=(int(valid.sum()), 3)) if callable(normal_draws) else normal_draws[valid]
    draws = np.asarray(draws, dtype=np.float32)
    steps = np.broadcast_to(np.asarray(dt, dtype=np.float32), (len(velocity),))[valid, None]
    eps = np.broadcast_to(np.asarray(coeps, dtype=np.float32), (len(velocity),))[valid, None]
    flux = np.asarray(flux, dtype=np.float32)[valid]
    inverse = inverse[valid]
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = (stress[valid]-previous[valid])/steps[:, :, None]
        matrix = -eps[:, :, None]*inverse
        for axis in range(3):
            matrix += delta[:, :, axis, None]*inverse[:, None, axis, :]
        matrix = -np.eye(3, dtype=np.float32)+np.float32(.5)*matrix*steps[:, :, None]
    force = -velocity[valid]-np.float32(.5)*flux*steps-np.sqrt(eps*steps)*draws
    inverse, solved = _invert(matrix)
    slots = np.flatnonzero(valid)[solved]
    products = inverse[solved]*force[solved, None, :]
    updated[slots] = products[:, :, 0]+products[:, :, 1]+products[:, :, 2]
    stored[slots] = stress[slots]
    return updated, stored
