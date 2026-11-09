"""Fused Python CUDA kernel for the nine staggered velocity derivatives."""
from functools import lru_cache

import numpy as np

from .numerics.cuda_runtime import load_cupy


@lru_cache(maxsize=1)
def _kernel():
    from cupyx import jit

    @jit.rawkernel(device=True)
    def centred(field, face, offset):
        return .5*(field[face]+field[face+offset])

    @jit.rawkernel(device=True)
    def vertical(field, face, offset, plane, below, above):
        middle = centred(field, face, offset)
        upper = centred(field, face+plane, offset)
        lower = centred(field, face-plane, offset)
        return ((below/above)*(upper-middle)+(above/below)*(middle-lower))/(below+above)

    @jit.rawkernel()
    def derivatives(cells, u, v, w, z, dz, nx, ny, dx, dy, gradient):
        thread = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        if thread < cells.size:
            cell = cells[thread]
            k = cell//(nx*ny)
            j = (cell//nx)%ny
            i = cell%nx
            row = nx+1
            plane = row*(ny+1)
            face = (k*(ny+1)+j)*row+i
            below = z[k]-z[k-1]
            above = z[k+1]-z[k]
            gradient[cell, 0] = (u[face+1]-u[face])/dx
            gradient[cell, 1] = (centred(u, face+row, 1)-centred(u, face-row, 1))/(2*dy)
            gradient[cell, 2] = vertical(u, face, 1, plane, below, above)
            gradient[cell, 3] = (centred(v, face+1, row)-centred(v, face-1, row))/(2*dx)
            gradient[cell, 4] = (v[face+row]-v[face])/dy
            gradient[cell, 5] = vertical(v, face, row, plane, below, above)
            gradient[cell, 6] = (centred(w, face+1, plane)-centred(w, face-1, plane))/(2*dx)
            gradient[cell, 7] = (centred(w, face+row, plane)-centred(w, face-row, plane))/(2*dy)
            gradient[cell, 8] = (w[face+plane]-w[face])/dz[k]

    return derivatives


def cuda_velocity_gradient(workspace, velocity, fluid, *, device=0):
    z, dz = workspace.coordinates['z'], workspace.arrays['dz_array']
    x, y = (workspace.coordinates[name] for name in ('x', 'y'))
    nz, ny, nx = fluid.shape
    face_shape = (nz+1, ny+1, nx+1)
    if any(velocity[name].shape != face_shape for name in 'uvw') or any(
            values.shape != (size,) for values, size in ((x, nx), (y, ny), (z, nz), (dz, nz))):
        raise ValueError('CUDA velocity gradient requires matching staggered face and cell grids')
    if fluid[[0, -1]].any() or fluid[:, [0, -1]].any() or fluid[:, :, [0, -1]].any():
        raise ValueError('CUDA velocity gradient requires inactive boundary cells')
    cells = np.flatnonzero(fluid)
    cp = load_cupy(device)
    with cp.cuda.Device(device):
        result = cp.zeros((fluid.size, 9), dtype=cp.float64)
        if cells.size:
            block = 128
            faces = (cp.asarray(velocity[name], order='C').ravel() for name in 'uvw')
            _kernel()[((cells.size+block-1)//block,), (block,)](
                cp.asarray(cells, dtype=cp.int64), *faces, cp.asarray(z), cp.asarray(dz),
                np.int64(nx), np.int64(ny), np.float64(x[1]-x[0]), np.float64(y[1]-y[0]), result)
        return cp.asnumpy(result).reshape(fluid.shape+(3, 3))
