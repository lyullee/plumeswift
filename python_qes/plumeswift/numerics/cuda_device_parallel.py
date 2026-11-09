"""Python JIT port of the legacy QES device-launched pressure solver (GPL-3)."""

from functools import lru_cache
import time

import numpy as np

from ..contracts import PhysicalCoefficients
from .cuda_dynamic_legacy import CudaDynamicLegacy, _sweep_kernel
from .cuda_runtime import load_cupy


_THREADS = 1024
_ITERATIONS_PER_LAUNCH = 128


@lru_cache(maxsize=1)
def _controller():
    import cupy as cp
    from cupyx import jit
    from cupyx.jit import _compile, _cuda_types, _internal_types

    class Launch(_internal_types.BuiltinFunc):
        """Emit a CUDA child launch for a Python JIT kernel in this module."""

        def call(self, env, kernel, blocks, *args):
            values = [_internal_types.Data.init(arg, env) for arg in args]
            types = tuple(value.ctype for value in values)
            name, _ = _compile._transpile_func_obj(kernel.obj._func, ['__global__'], env.mode,
                                                   types, _cuda_types.void, env.generated)
            grid = _internal_types.Data.init(blocks, env).code
            params = ', '.join(value.code for value in values)
            if '--device-c' not in env.generated.options:
                env.generated.options += ('--device-c', '-DCUDA_FORCE_CDP1_IF_SUPPORTED')
            return _internal_types.Data(f'{name}<<<{grid}, {_THREADS}>>>({params})', _cuda_types.void)

    class DeviceStatus(_internal_types.BuiltinFunc):
        def __init__(self, expression):
            super().__init__()
            self.expression = expression

        def call(self, env):
            return _internal_types.Data(self.expression, _cuda_types.Scalar(np.int32))

    launch = Launch()
    launch_status = DeviceStatus('cudaGetLastError()')
    wait_for_children = DeviceStatus('cudaDeviceSynchronize()')
    sweep = _sweep_kernel()

    @jit.rawkernel(device=True)
    def child_status():
        status = launch_status()
        if status == 0:
            status = wait_for_children()
        return status

    @jit.rawkernel()
    def save(pressure, previous):
        index = np.int64(jit.grid(1))
        if index < pressure.size:
            previous[index] = pressure[index]

    @jit.rawkernel()
    def mirror(pressure, plane):
        index = np.int64(jit.grid(1))
        if index < plane:
            pressure[index] = pressure[index+plane]

    @jit.rawkernel()
    def changes(pressure, previous, maxima):
        lane = np.int32(jit.threadIdx.x)
        index = np.int64(jit.grid(1))
        block_values = jit.shared_memory(np.float32, _THREADS)
        delta = np.float32(0)
        if index < pressure.size:
            delta = cp.absolute(pressure[index]-previous[index])
            if not cp.isfinite(delta):
                delta = np.float32(np.inf)
        block_values[lane] = delta
        stride = _THREADS//2
        while stride > 0:
            jit.syncthreads()
            if lane < stride:
                block_values[lane] = cp.maximum(block_values[lane], block_values[lane+stride])
            stride = stride//2
        if lane == 0:
            maxima[jit.blockIdx.x] = block_values[0]

    @jit.rawkernel(device=True)
    def cell_coordinates(index, nx, ny):
        k = index//(nx*ny)
        j = (index//nx) % ny
        return k, j, index % nx

    @jit.rawkernel()
    def divergence(rhs, coeff, u, v, w, dz, nx, ny, dx, dy, scale):
        index = np.int64(jit.grid(1))
        if index < rhs.size:
            k, j, i = cell_coordinates(index, nx, ny)
            if k > 0 and k < dz.size-1:
                face_row, face_plane = nx+1, (nx+1)*(ny+1)
                face = k*face_plane+j*face_row+i
                horizontal = ((coeff[0, index]*u[face+1]-coeff[1, index]*u[face])*dx
                              +(coeff[2, index]*v[face+face_row]-coeff[3, index]*v[face])*dy)
                # The native 0.5 literal promotes the vertical fluxes to double.
                half = np.float64(0.5)
                up = (coeff[4, index]*dz[k])*half*(dz[k]+dz[k+1])*w[face+face_plane]
                down = (coeff[5, index]*w[face]*dz[k])*half*(dz[k]+dz[k-1])
                rhs[index] = scale*(horizontal+(up-down))

    @jit.rawkernel()
    def correct(pressure, coeff, u, v, w, dz, nx, ny, dx, dy, scale_h, scale_v):
        index = np.int64(jit.grid(1))
        if index < pressure.size:
            k, j, i = cell_coordinates(index, nx, ny)
            if k > 0 and k < dz.size-1 and j > 0 and i > 0:
                face_row, face_plane = nx+1, (nx+1)*(ny+1)
                face = k*face_plane+j*face_row+i
                u[face] += scale_h*coeff[1, index]*dx*(pressure[index]-pressure[index-1])
                v[face] += scale_h*coeff[3, index]*dy*(pressure[index]-pressure[index-nx])
                w[face] += scale_v*coeff[5, index]*dz[k]*(pressure[index]-pressure[index-nx*ny])

    @jit.rawkernel()
    def clear_solid_faces(u, v, w, flags, nx, ny):
        index = np.int64(jit.grid(1))
        if nx*ny <= index and index < flags.size:
            k, j, i = cell_coordinates(index, nx, ny)
            face_row, face_plane = nx+1, (nx+1)*(ny+1)
            face = k*face_plane+j*face_row+i
            # Each thread owns the low faces. At the high side, clear the
            # exterior face as well. No two threads write the same face.
            solid = flags[index] == 0 or flags[index] == 2
            if solid or (i > 0 and (flags[index-1] == 0 or flags[index-1] == 2)):
                u[face] = 0
            if solid or (j > 0 and (flags[index-nx] == 0 or flags[index-nx] == 2)):
                v[face] = 0
            if solid or (k > 1 and (flags[index-nx*ny] == 0 or flags[index-nx*ny] == 2)):
                w[face] = 0
            if solid:
                if i == nx-1:
                    u[face+1] = 0
                if j == ny-1:
                    v[face+face_row] = 0
                if index+nx*ny >= flags.size:
                    w[face+face_plane] = 0

    @jit.rawkernel()
    def parent(pressure, previous, coeff, rhs, red, black, maxima, status, error,
               u, v, w, dz, flags, nx, ny, dx, dy, rhs_scale, scale_h, scale_v,
               omega, tolerance, max_iterations, batch, projection):
        iterations = status[0]
        cuda_error = np.int32(0)
        if projection and iterations == 0:
            launch(divergence, maxima.size, rhs, coeff, u, v, w, dz, nx, ny, dx, dy, rhs_scale)
        stop = min(max_iterations, iterations+batch)
        change = error[0]
        while cuda_error == 0 and iterations < stop and change > tolerance and cp.isfinite(change):
            launch(save, maxima.size, pressure, previous)
            plane = nx*ny
            if red.size > 0:
                launch(sweep, (red.size+_THREADS-1)//_THREADS, red, pressure, coeff, rhs, nx, plane, omega)
            if black.size > 0:
                launch(sweep, (black.size+_THREADS-1)//_THREADS, black, pressure, coeff, rhs, nx, plane, omega)
            launch(mirror, (plane+_THREADS-1)//_THREADS, pressure, plane)
            launch(changes, maxima.size, pressure, previous, maxima)
            cuda_error = child_status()
            if cuda_error == 0:
                change = np.float32(0)
                for index in range(maxima.size):
                    change = cp.maximum(change, maxima[index])
                iterations += 1
        if projection and cuda_error == 0 and cp.isfinite(change):
            if iterations == max_iterations or change <= tolerance:
                launch(correct, maxima.size, pressure, coeff, u, v, w, dz, nx, ny, dx, dy, scale_h, scale_v)
                launch(clear_solid_faces, maxima.size, u, v, w, flags, nx, ny)
                cuda_error = child_status()
        status[0], status[1] = iterations, cuda_error
        error[0] = change

    return parent


class CudaDeviceParallel(CudaDynamicLegacy):
    """Device-launched child grids and convergence checks using CUDA CDP1."""

    method = 'cupy.cuda-device-parallel-SOR'

    def _run(self, cp, stencil, coefficients, rhs, pressure, projection=None):
        if cp.cuda.runtime.is_hip or not 35 <= int(cp.cuda.Device().compute_capability) < 90:
            raise RuntimeError('The device-parallel solver requires an NVIDIA GPU supporting CUDA CDP1 (SM 3.5–8.x)')
        if cp.cuda.nvrtc.getVersion()[0] not in (11, 12):
            raise RuntimeError('The device-parallel solver requires CUDA 11 or 12 with the CDP1 device runtime')
        _, ny, nx = stencil.shape
        nodes = stencil.indices
        parity = (nodes//(nx*ny)+(nodes//nx)%ny+nodes%nx)%2
        red, black = (cp.asarray(nodes[parity == color]) for color in (0, 1))
        previous = cp.empty_like(pressure).ravel()
        maxima = cp.empty((pressure.size+_THREADS-1)//_THREADS, dtype=cp.float32)
        status = cp.zeros(2, dtype=cp.int64)
        error = cp.ones(1, dtype=cp.float32)
        if projection is None:
            empty = cp.empty(0, dtype=cp.float32)
            projection_args = (empty, empty, empty, empty, cp.empty(0, dtype=cp.int32),
                               np.float32(0), np.float32(0), np.float64(0), np.float64(0), np.float64(0))
        else:
            projection_args = projection
        u, v, w, dz, flags, dx, dy, rhs_scale, scale_h, scale_v = projection_args
        tolerance = np.float32(self.policy.relative_tolerance)
        max_iterations = np.int64(self.policy.max_iterations)
        controller = _controller()
        while True:
            controller[(1,), (1,)](
                pressure.ravel(), previous, coefficients.reshape(6, -1), rhs.ravel(), red, black, maxima, status, error,
                u, v, w, dz, flags, np.int64(nx), np.int64(ny), dx, dy, rhs_scale, scale_h, scale_v,
                np.float32(self.omega), tolerance, max_iterations, np.int64(_ITERATIONS_PER_LAUNCH), projection is not None)
            iterations, cuda_error = map(int, status.get())
            if cuda_error:
                raise RuntimeError(f'CUDA device-parallel child kernel failed with CUDA error {cuda_error}')
            change = float(error.item())
            if iterations >= max_iterations or change <= tolerance or not np.isfinite(change):
                return iterations, change

    def _iterate(self, cp, stencil, coefficients, rhs, pressure):
        return self._run(cp, stencil, coefficients, rhs, pressure)

    def project(self, workspace, physics=None):
        start = time.perf_counter()
        cp = load_cupy(self.device)
        p = physics or PhysicalCoefficients()
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = (np.float32(np.median(np.diff(c[axis]))) for axis in 'xy')
        stencil = workspace.stencil()
        with cp.cuda.Device(self.device):
            fields = {key: cp.array(a[key+'0'], dtype=cp.float32, copy=True, order='C') for key in 'uvw'}
            dz = cp.asarray(a['dz_array'], dtype=cp.float32, order='C')
            flags = cp.asarray(a['icellflag'], dtype=cp.int32, order='C').ravel()
            coefficients = cp.asarray(np.stack([a[key] for key in 'efghmn']), dtype=cp.float32, order='C')
            pressure = cp.zeros(stencil.shape, dtype=cp.float32)
            rhs = cp.zeros_like(pressure)
            projection = (*(field.ravel() for field in fields.values()), dz, flags, dx, dy,
                          np.float64(-2*p.alpha_horizontal**2), np.float64(1/(2*p.alpha_horizontal**2)),
                          np.float64(1/(2*p.alpha_vertical**2)))
            iterations, change = self._run(cp, stencil, coefficients, rhs, pressure, projection)
            if not bool(cp.isfinite(rhs).all().item()):
                raise ValueError('Non-finite device-parallel CUDA SOR RHS')
            host_pressure, report = self._report(cp, stencil, pressure, rhs, iterations, change, start)
            return {key: cp.asnumpy(value) for key, value in fields.items()}, host_pressure, report
