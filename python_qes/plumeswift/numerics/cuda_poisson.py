"""GPU red-black SOR with Python kernels and the QES pressure stencil."""
from dataclasses import replace
from functools import lru_cache
import time

import numpy as np

from ..contracts import NumericPolicy, SolverReport
from .cuda_runtime import load_cupy


_COEFFICIENTS = ('east', 'west', 'north', 'south', 'up', 'down')


@lru_cache(maxsize=1)
def _kernels():
    from cupyx import jit

    @jit.rawkernel(device=True)
    def neighbours(node, p, coefficients, row, plane):
        value = (coefficients[0, node]*p[node+1]+coefficients[1, node]*p[node-1]
                 +coefficients[2, node]*p[node+row]+coefficients[3, node]*p[node-row]
                 +coefficients[4, node]*p[node+plane])
        if node >= 2*plane:
            value += coefficients[5, node]*p[node-plane]
        return value

    @jit.rawkernel()
    def sweep(nodes, p, coefficients, diagonal, rhs, row, plane, omega):
        thread = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        if thread < nodes.size:
            node = nodes[thread]
            target = (neighbours(node, p, coefficients, row, plane)-rhs[node])/diagonal[node]
            p[node] += omega*(target-p[node])

    @jit.rawkernel()
    def measure(nodes, p, coefficients, diagonal, rhs, row, plane, residual):
        thread = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        if thread < nodes.size:
            node = nodes[thread]
            error = diagonal[node]*p[node]-neighbours(node, p, coefficients, row, plane)+rhs[node]
            residual[thread] = -error if error < 0 else error

    return sweep, measure


class CudaRedBlackSOR:
    """Solve in float64 on an NVIDIA GPU without a native QES runtime."""

    def __init__(self, policy=None, omega=1.78, device=0, check_every=8):
        self.policy = policy or NumericPolicy()
        self.omega = float(omega)
        self.device = int(device)
        self.check_every = int(check_every)
        if not 0 < self.omega < 2 or self.device < 0 or self.check_every < 1:
            raise ValueError('CUDA SOR requires 0 < omega < 2, a nonnegative device and a positive check interval')

    @staticmethod
    def _coefficients(cp, stencil):
        return cp.asarray(np.stack([getattr(stencil, key) for key in _COEFFICIENTS]))

    def _report(self, iterations, residual, limit, start):
        report = SolverReport(bool(np.isfinite(residual) and residual <= limit*1.01), iterations,
                              residual, None, time.perf_counter()-start, 'cupy.cuda-red-black-SOR')
        if not report.converged:
            raise RuntimeError(f'Poisson convergence failed: {report}')
        return report

    def solve(self, stencil, rhs, initial=None):
        cp = load_cupy(self.device)

        start = time.perf_counter()
        rhs = np.asarray(rhs, dtype=np.float64)
        pressure = np.zeros(stencil.shape) if initial is None else np.array(initial, dtype=np.float64, copy=True, order='C')
        if rhs.shape != stencil.shape or pressure.shape != stencil.shape or not np.isfinite(rhs).all() or not np.isfinite(pressure).all():
            raise ValueError('CUDA SOR requires a finite pressure and RHS matching the stencil shape')
        stencil.boundaries(pressure)
        nodes = stencil.indices
        limit = self.policy.absolute_tolerance+self.policy.relative_tolerance*np.max(abs(rhs.ravel()[nodes]), initial=0.)
        iterations = 0
        if stencil.residual(pressure, rhs) > limit:
            with cp.cuda.Device(self.device):
                values, iterations, _ = self._iterate(cp, stencil, cp.asarray(rhs), cp.asarray(pressure),
                                                       self._coefficients(cp, stencil), limit)
                pressure = cp.asnumpy(values)
            stencil.boundaries(pressure)
        residual = stencil.residual(pressure, rhs) if np.isfinite(pressure).all() else float('inf')
        return pressure, self._report(iterations, residual, limit, start)

    def project(self, workspace, physics=None):
        """Keep divergence, multiplier and velocity correction on one device."""
        from ..wind_projection import corrected_velocity, rhs_from_workspace

        cp = load_cupy(self.device)
        stencil = workspace.stencil()
        with cp.cuda.Device(self.device):
            coefficients = cp.asarray(np.stack([workspace.arrays[key] for key in 'efghmn']))
            arrays = {key: cp.asarray(workspace.arrays[key]) for key in ('u0', 'v0', 'w0', 'dz_array', 'icellflag')}
            arrays.update({key: coefficient.astype(workspace.arrays[key].dtype, copy=False)
                           for key, coefficient in zip('efghmn', coefficients, strict=True)})
            device_workspace = replace(workspace, arrays=arrays)
            rhs = rhs_from_workspace(device_workspace, physics, array_module=cp)
            if not bool(cp.isfinite(rhs).all().item()):
                raise ValueError('Non-finite Poisson RHS')
            start = time.perf_counter()
            active_rhs = rhs.ravel()[cp.asarray(stencil.indices)]
            residual = float(cp.max(cp.abs(active_rhs)).item()) if active_rhs.size else 0.
            limit = self.policy.absolute_tolerance+self.policy.relative_tolerance*residual
            pressure = cp.zeros(stencil.shape, dtype=cp.float64)
            iterations = 0
            if residual > limit:
                pressure, iterations, residual = self._iterate(cp, stencil, rhs, pressure, coefficients, limit)
                stencil.boundaries(pressure)
            report = self._report(iterations, residual, limit, start)
            velocity = corrected_velocity(device_workspace, pressure, physics, array_module=cp)
            return {key: cp.asnumpy(value) for key, value in velocity.items()}, cp.asnumpy(pressure), report

    def _iterate(self, cp, stencil, rhs, pressure, coefficients, limit):
        sweep, measure = _kernels()
        _, ny, nx = stencil.shape
        row, plane = np.int64(nx), np.int64(nx*ny)
        nodes = stencil.indices
        parity = (nodes//plane+(nodes//row)%ny+nodes%row)%2
        colors = tuple(cp.asarray(nodes[parity == color], dtype=cp.int64) for color in (0, 1))
        active = cp.asarray(nodes, dtype=cp.int64)
        # Eliminate p[0] = p[1] from the first fluid layer's diagonal. A
        # GPU sweep never writes into the fluid layer to mirror the ghost.
        diagonal = stencil.diagonal.astype(np.float64, copy=True)
        diagonal[1] -= stencil.down[1]
        if np.any(diagonal.ravel()[nodes] <= 0):
            raise ValueError('Singular or disconnected pressure cells')
        values = pressure.ravel()
        residual = cp.empty(len(nodes), dtype=cp.float64)
        args = (values, coefficients.reshape(6, -1), cp.asarray(diagonal.ravel()), rhs.ravel(), row, plane)
        block = 128
        launches = [(indices, ((indices.size+block-1)//block,)) for indices in colors if indices.size]
        blocks = ((len(nodes)+block-1)//block,)
        iterations = 0
        while iterations < self.policy.max_iterations:
            count = min(self.check_every, self.policy.max_iterations-iterations)
            for _ in range(count):
                for indices, grid in launches:
                    sweep[grid, (block,)](indices, *args, np.float64(self.omega))
            iterations += count
            measure[blocks, (block,)](active, *args, residual)
            error = float(cp.max(residual).item())
            if not np.isfinite(error) or error <= limit:
                break
        return pressure, iterations, error
