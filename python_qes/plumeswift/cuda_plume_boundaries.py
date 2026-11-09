"""Axis policies for the experimental CUDA QES plume (GPL-3)."""
from functools import lru_cache
from operator import index

import numpy as np

from .cuda_particle_buffer import INACTIVE


BOUNDARY_MODES = ('exiting', 'periodic', 'reflection')


def boundary_ids(modes):
    modes = tuple(modes)
    if len(modes) != 3 or any(mode not in BOUNDARY_MODES for mode in modes):
        raise ValueError('Expected three CUDA plume boundary modes: exiting, periodic or reflection')
    return np.array([BOUNDARY_MODES.index(mode) for mode in modes], dtype=np.int32)


def checked_bounds(particles, bounds, count):
    count = particles.capacity if count is None else index(count)
    bounds = np.asarray(bounds, dtype=np.float32)
    if (not 0 <= count <= particles.capacity or bounds.shape != (3, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(bounds[:, 1].astype(float)-bounds[:, 0] > np.finfo(np.float32).max)):
        raise ValueError('Invalid CUDA plume count or XYZ domain bounds')
    return count, bounds


@lru_cache(maxsize=1)
def device_historical_exit():
    from cupyx import jit

    @jit.rawkernel(device=True)
    def exits(position, lower, upper):
        # Ordered comparisons leave an unordered NaN coordinate unchanged.
        return position <= lower or position >= upper

    return exits


@lru_cache(maxsize=1)
def _historical_exiting_kernel():
    from cupyx import jit

    exits = device_historical_exit()

    @jit.rawkernel()
    def apply(count, states, positions, bounds):
        particle = jit.blockDim.x*jit.blockIdx.x+jit.threadIdx.x
        if particle < count:
            for axis in range(3):
                if exits(positions[particle, axis], bounds[axis, 0], bounds[axis, 1]):
                    states[particle] = INACTIVE

    return apply


def apply_historical_exiting(particles, bounds, count=None):
    """Apply the historical standalone three-axis exiting kernel to every slot."""
    count, bounds = checked_bounds(particles, bounds, count)
    if count:
        cp = particles.cp
        block = 128
        with cp.cuda.Device(particles.device):
            _historical_exiting_kernel()[((count+block-1)//block,), (block,)](
                np.int64(count), particles.state, particles.positions, cp.asarray(bounds))


@lru_cache(maxsize=1)
def device_boundary():
    import cupy as cp
    from cupyx import jit

    @jit.rawkernel(device=True)
    def apply(position, fluctuation, lower, upper, mode):
        if not cp.isfinite(position):
            return position, fluctuation, False
        if mode == 0:
            return position, fluctuation, position > lower and position < upper
        if mode == 2:
            for _ in range(20):
                if position > upper:
                    position = upper-(position-upper)
                elif position < lower:
                    position = lower-(position-lower)
                else:
                    return position, fluctuation, True
                fluctuation = -fluctuation
            return position, fluctuation, False

        # Keep the native float32 wrapping for ordinary displacements.
        width = upper-lower
        for _ in range(64):
            if position < lower:
                position += width
            elif position > upper:
                position -= width
            else:
                return position, fluctuation, True
        # Bound large excursions and cases where adding the width cannot advance a float32 value.
        span = np.float64(upper)-np.float64(lower)
        if position < lower:
            remainder = cp.fmod(np.float64(position)-np.float64(lower), span)
            position = lower if remainder == 0. else np.float32(np.float64(upper)+remainder)
        elif position > upper:
            remainder = cp.fmod(np.float64(position)-np.float64(lower), span)
            position = upper if remainder == 0. else np.float32(np.float64(lower)+remainder)
        return position, fluctuation, True

    return apply
