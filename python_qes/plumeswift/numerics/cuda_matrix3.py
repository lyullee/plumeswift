"""Device-callable float32 QES matrix operations (GPL-3)."""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=3)
def device_matrix3(profile='modern'):
    """Return device operations for production or a historical diagnostic layout."""
    if profile not in ('modern', 'historical-struct', 'historical-scalar'):
        raise ValueError(f'Unknown CUDA matrix profile {profile!r}')
    import cupy as cp
    from cupyx import jit

    historical_struct = profile == 'historical-struct'
    historical_scalar = profile == 'historical-scalar'
    inverse_threshold = 1e-10 if historical_scalar else 1e-6

    @jit.rawkernel(device=True)
    def determinant(a, b, c, d, e, f):
        if historical_struct:
            # Both historical struct helpers used yz instead of xy in this term.
            return a*(d*f-e*e)-b*(e*f-e*c)+c*(b*e-d*c)
        return a*(d*f-e*e)-b*(b*f-e*c)+c*(b*e-d*c)

    @jit.rawkernel(device=True)
    def invariants(a, b, c, d, e, f):
        if historical_scalar:
            return a+d+f, a*d+a*f+d*f-b*b-c*c-e*e, determinant(a, b, c, d, e, f)
        return a+d+f, a*d+a*f+d*f-(b*b+c*c+e*e), determinant(a, b, c, d, e, f)

    @jit.rawkernel(device=True)
    def repair(stress, p, tol):
        a, b, c = stress[p, 0], stress[p, 1], stress[p, 2]
        d, e, f = stress[p, 3], stress[p, 4], stress[p, 5]
        first, second, third = invariants(a, b, c, d, e, f)
        if not (first > tol and second > tol and third > tol):
            beta = np.float32((4./3.)*first)
            discriminant = np.float64(beta*beta)-(16./3.)*second
            ks = np.float32(1.01*(-beta+cp.sqrt(discriminant))/(8./3.))
            if ks < tol or cp.isnan(ks):
                ks = np.float32(.5)*cp.abs(first)
            for iteration in range(1001):
                shift = (2./3.)*np.float64(ks)
                aa = np.float32(np.float64(a)+shift)
                dd = np.float32(np.float64(d)+shift)
                ff = np.float32(np.float64(f)+shift)
                first, second, third = invariants(aa, b, c, dd, e, ff)
                if ks == 0 or not (first < tol or second < tol or third < tol) or iteration == 1000:
                    stress[p, 0] = aa
                    stress[p, 3] = dd
                    stress[p, 5] = ff
                    return
                ks = np.float32(np.float64(ks)*1.05)

    @jit.rawkernel(device=True)
    def invert(a, b, c, d, e, f, g, h, i):
        det = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        if cp.abs(det) < inverse_threshold:
            zero = np.float32(0.)
            return False, zero, zero, zero, zero, zero, zero, zero, zero, zero
        return (True, (e*i-f*h)/det, -(b*i-c*h)/det, (b*f-e*c)/det,
                -(d*i-f*g)/det, (a*i-c*g)/det, -(a*f-c*d)/det,
                (d*h-g*e)/det, -(a*h-b*g)/det, (a*e-b*d)/det)

    @jit.rawkernel(device=True)
    def multiply(a, b, c, d, e, f, g, h, i, x, y, z):
        return x*a+y*b+z*c, x*d+y*e+z*f, x*g+y*h+z*i

    return invariants, repair, invert, multiply
