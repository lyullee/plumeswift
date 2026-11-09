"""Experimental QES CUDA plume Langevin solve and advection (GPL-3)."""
from functools import lru_cache
from operator import index

import numpy as np

from .cuda_particle_buffer import ACTIVE, INACTIVE, ROGUE
from .cuda_plume_boundaries import boundary_ids, checked_bounds, device_boundary, device_historical_exit
from .numerics.cuda_matrix3 import device_matrix3


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    from cupyx import jit

    boundary = device_boundary()
    historical_exit = device_historical_exit()

    _, repair, invert, multiply = device_matrix3()

    @jit.rawkernel(device=True)
    def system_entry(eps, entry, a, b, c, x, y, z, diagonal):
        return diagonal+np.float32(.5)*(-eps*entry+a*x+b*y+c*z)

    @jit.rawkernel(device=True)
    def solve(p, states, stress, previous_stress, fluct, old, flux, eps, draws):
        repair(stress, p, np.float32(1e-7))
        a, b, c = stress[p, 0], stress[p, 1], stress[p, 2]
        d, e, f = stress[p, 3], stress[p, 4], stress[p, 5]
        valid, l11, l12, l13, l21, l22, l23, l31, l32, l33 = invert(a, b, c, b, d, e, c, e, f)
        if not valid:
            states[p] = ROGUE
            return
        da, db, dc = a-previous_stress[p, 0], b-previous_stress[p, 1], c-previous_stress[p, 2]
        dd, de, df = d-previous_stress[p, 3], e-previous_stress[p, 4], f-previous_stress[p, 5]
        dissipation = eps[p]
        diagonal, off_diagonal = np.float32(-1.), np.float32(-0.)
        # The native launch uses one second and builds the full float32 system
        # with the symmetric half of the inverse stress matrix.
        a11 = system_entry(dissipation, l11, l11, l12, l13, da, db, dc, diagonal)
        a12 = system_entry(dissipation, l12, l12, l22, l23, da, db, dc, off_diagonal)
        a13 = system_entry(dissipation, l13, l13, l23, l33, da, db, dc, off_diagonal)
        a21 = system_entry(dissipation, l12, l11, l12, l13, db, dd, de, off_diagonal)
        a22 = system_entry(dissipation, l22, l12, l22, l23, db, dd, de, diagonal)
        a23 = system_entry(dissipation, l23, l13, l23, l33, db, dd, de, off_diagonal)
        a31 = system_entry(dissipation, l13, l11, l12, l13, dc, de, df, off_diagonal)
        a32 = system_entry(dissipation, l23, l12, l22, l23, dc, de, df, off_diagonal)
        a33 = system_entry(dissipation, l33, l13, l23, l33, dc, de, df, diagonal)
        valid, a11, a12, a13, a21, a22, a23, a31, a32, a33 = invert(a11, a12, a13, a21, a22, a23, a31, a32, a33)
        if not valid:
            states[p] = ROGUE
            return
        scale = cp.sqrt(dissipation)
        half = np.float32(.5)
        x = -old[p, 0]-half*flux[p, 0]-scale*draws[0, p]
        y = -old[p, 1]-half*flux[p, 1]-scale*draws[1, p]
        z = -old[p, 2]-half*flux[p, 2]-scale*draws[2, p]
        fluct[p, 0], fluct[p, 1], fluct[p, 2] = multiply(a11, a12, a13, a21, a22, a23, a31, a32, a33, x, y, z)
        for axis in range(3):
            value = fluct[p, axis]
            if cp.abs(value) >= 10 or cp.isnan(value):
                fluct[p, axis] = np.float32(0.)
                states[p] = ROGUE
        for component in range(6):
            previous_stress[p, component] = stress[p, component]

    @jit.rawkernel()
    def advance(states, positions, mean, stress, previous_stress, fluct, old, delta, flux, eps, draws, bounds, modes,
                apply_boundaries, historical_exiting):
        p = jit.blockDim.x*jit.blockIdx.x+jit.threadIdx.x
        if p < draws.shape[1] and states[p] == ACTIVE:
            solve(p, states, stress, previous_stress, fluct, old, flux, eps, draws)
            for axis in range(3):
                position = positions[p, axis]+(mean[p, axis]+fluct[p, axis])
                if not apply_boundaries:
                    location, velocity, alive = position, fluct[p, axis], True
                elif historical_exiting:
                    location, velocity = position, fluct[p, axis]
                    alive = not historical_exit(position, bounds[axis, 0], bounds[axis, 1])
                else:
                    location, velocity, alive = boundary(position, fluct[p, axis], bounds[axis, 0], bounds[axis, 1], modes[axis])
                positions[p, axis] = location
                fluct[p, axis] = velocity
                if not alive:
                    states[p] = INACTIVE
                delta[p, axis] = fluct[p, axis]-old[p, axis]
                old[p, axis] = fluct[p, axis]

    return advance


def advect_particles(particles, draws, bounds, count=None, *, boundaries=('exiting',)*3, historical_exiting=False):
    """Run a native 1 s step, with bounds=None for the unbounded test kernel."""
    modes = boundary_ids(boundaries)
    apply_boundaries = bounds is not None
    if apply_boundaries:
        count, bounds = checked_bounds(particles, bounds, count)
    else:
        count = particles.capacity if count is None else index(count)
        if not 0 <= count <= particles.capacity or historical_exiting:
            raise ValueError('Invalid CUDA plume count or historical exiting without bounds')
        bounds = np.zeros((3, 2), dtype=np.float32)
    if historical_exiting and np.any(modes != 0):
        raise ValueError('Historical CUDA advection requires exiting boundaries on every axis')
    cp = particles.cp
    with cp.cuda.Device(particles.device):
        draws = cp.asarray(draws, dtype=cp.float32, order='C')
        if draws.shape != (3, count):
            raise ValueError('Expected three axis-major normal draws per particle')
        if count:
            block = 128
            _kernel()[((count+block-1)//block,), (block,)](
                particles.state, particles.positions, particles.mean_velocity, particles.stress,
                particles.previous_stress, particles.fluctuations, particles.previous_fluctuations,
                particles.delta_fluctuations, particles.flux_divergence, particles.dissipation, draws, cp.asarray(bounds),
                cp.asarray(modes), np.bool_(apply_boundaries), np.bool_(historical_exiting))
