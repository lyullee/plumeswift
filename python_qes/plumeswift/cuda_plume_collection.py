"""QES experimental CUDA plume collection in Cartesian boxes (GPL-3)."""
from functools import lru_cache
from operator import index

import numpy as np

from .cuda_particle_buffer import ACTIVE
from .numerics.cuda_runtime import load_cupy


@lru_cache(maxsize=1)
def _kernel():
    from cupyx import jit

    @jit.rawkernel()
    def collect(positions, states, counts, origin, widths):
        particle = jit.blockDim.x*jit.blockIdx.x+jit.threadIdx.x
        if particle < states.size and states[particle] == ACTIVE:
            x = np.float64(positions[particle, 0]-origin[0])/(np.float64(widths[0])+1e-9)
            y = np.float64(positions[particle, 1]-origin[1])/(np.float64(widths[1])+1e-9)
            z = np.float64(positions[particle, 2]-origin[2])/(np.float64(widths[2])+1e-9)
            nz, ny, nx = counts.shape
            # Bounds before conversion also exclude invalid and nonfinite positions.
            if x >= 0. and x < nx and y >= 0. and y < ny and z >= 0. and z < nz:
                jit.atomic_add(counts, (int(z), int(y), int(x)), np.uint64(1))

    return collect


class CudaPlumeCollection:
    """Accumulate active particle counts and QES averaging time on CUDA."""

    def __init__(self, lower, widths, boxes, *, device=0):
        lower, widths = (np.asarray(values, dtype=np.float32) for values in (lower, widths))
        boxes = tuple(map(index, boxes))
        if (lower.shape != (3,) or widths.shape != (3,) or len(boxes) != 3 or
                not np.isfinite([lower, widths]).all() or np.any(widths <= 0) or min(boxes) <= 0):
            raise ValueError('Expected finite XYZ origin, positive widths and positive box counts')
        self.device = device
        self.cp = load_cupy(device)
        self.volume = float(widths[0]*widths[1]*widths[2])
        self.averaging_s = 0.
        with self.cp.cuda.Device(device):
            self.lower = self.cp.asarray(lower)
            self.widths = self.cp.asarray(widths)
            self.counts = self.cp.zeros(boxes[::-1], dtype=self.cp.uint64)

    def collect(self, positions, states, dt):
        dt = np.float32(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('Collection time step must be finite and nonnegative')
        cp = self.cp
        with cp.cuda.Device(self.device):
            positions = cp.asarray(positions, dtype=cp.float32, order='C')
            states = cp.asarray(states, order='C')
            if states.ndim != 1 or states.dtype.kind not in 'iu' or positions.shape != (states.size, 3):
                raise ValueError('Expected N by 3 positions and N integer particle states')
            if states.size:
                block = 256
                _kernel()[((states.size+block-1)//block,), (block,)](
                    positions, states, self.counts, self.lower, self.widths)
                cp.cuda.get_current_stream().synchronize()
        self.averaging_s = float(np.float32(self.averaging_s)+dt)

    def copyback(self):
        with self.cp.cuda.Device(self.device):
            return self.cp.asnumpy(self.counts)

    def reset(self):
        with self.cp.cuda.Device(self.device):
            self.counts.fill(0)
        self.averaging_s = 0.
