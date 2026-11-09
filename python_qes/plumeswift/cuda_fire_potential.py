"""CUDA implementation of the QES-Fire induced wind kernel (GPL-3)."""
from functools import lru_cache

import numpy as np

from .fire_potential import HeatPotential
from .numerics.cuda_runtime import load_cupy


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    from cupyx import jit

    @jit.rawkernel()
    def potential(out, table_r, table_z, g, gp, nx, ny, first, last, cx, cy, dx, dy, dz, dr, dh, origin, speed, length):
        index = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        plane = nx*ny
        k = first+index//plane
        if k < last:
            j, i = (index//nx)%ny, index%nx
            x, y = (i-cx)*dx/length, (j-cy)*dy/length
            z = (k*dz-origin)/length
            if z < 0.:
                z = np.float64(0.)
            radius = cp.sqrt(x*x+y*y)
            radial, vertical = np.float64(0.), np.float64(0.)
            if radius < 30. and z < 60.:
                ri, zi = 0 if radius < 1e-5 else int(radius/dr), int(z/dh)
                if ri >= table_r.shape[1] or zi >= table_r.shape[0]:
                    vertical = np.float64(np.nan)
                else:
                    if radius >= 1e-5:
                        radial = table_r[zi, ri]
                    vertical = table_z[zi, ri]
            else:
                distance = cp.sqrt(radius*radius+z*z)
                angle = .5*(1.+z/distance)
                coordinate = 2*g.size*(angle-.5)
                low = cp.minimum(np.int64(cp.floor(coordinate)), g.size-1)
                high = cp.minimum(np.int64(cp.ceil(coordinate)), g.size-1)
                value = .5*(g[low]+g[high])
                derivative = .5*(gp[low]+gp[high])
                root = cp.sqrt(cp.maximum(np.float64(0.), angle*(1.-angle)))
                power = distance**np.float64(-1/3)
                scale = np.float64(2*np.pi)*distance*distance*distance
                if root > 0.:
                    radial = radius/scale+power*(np.float64(5/6)*(1.-2.*angle)*value/root-root*derivative)
                vertical = z/scale+power*(np.float64(5/3)*value+.5*(1.-2.*angle)*derivative)
            if radius > 0.:
                out[0, k, j, i] += speed*radial*x/radius
                out[1, k, j, i] += speed*radial*y/radius
            out[2, k, j, i] += speed*vertical

    return potential


class CudaHeatPotential(HeatPotential):
    """Native GPU merge semantics and a Python CUDA potential kernel."""

    def apply(self, fire, workspace):
        cp = load_cupy(0)
        blocks, mix = self._merged_sources(fire, workspace, cuda=True)
        ny, nx = fire.terrain.shape
        layers = workspace.arrays['u0'].shape[0]-2
        with cp.cuda.Device(0):
            result = cp.zeros((3, layers, ny, nx), dtype=cp.float64)
            if blocks:
                tables = tuple(cp.asarray(array, dtype=cp.float64, order='C')
                               for array in (self.radial, self.vertical, self.g, self.g_prime))
                spacing = tuple(map(np.float64, (fire.dx, fire.dy, fire.dz,
                    self.radius[1]-self.radius[0], self.height[1]-self.height[0])))
                block = 128
                for first, last, cx, cy, origin, speed, length in blocks:
                    grid = (((last-first)*ny*nx+block-1)//block,)
                    _kernel()[grid, (block,)](result, *tables, *map(np.int64, (nx, ny, first, last, cx, cy)),
                                              *spacing, *map(np.float64, (origin, speed, length)))
            potential = cp.asnumpy(result)
        if not np.isfinite(potential).all():
            raise ValueError('Fire lookup table is shorter than the native near-field range or produced a non-finite potential')
        self._superimpose(workspace, potential)
        return potential, mix
