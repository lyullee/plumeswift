"""Deterministic port of the legacy QES dynamic GPU pressure stages (GPL-3)."""
from dataclasses import replace
from functools import lru_cache
import time

import numpy as np

from ..contracts import NumericPolicy, SolverReport
from ..wind_projection import corrected_velocity, rhs_from_workspace
from .cuda_poisson import _COEFFICIENTS
from .cuda_runtime import load_cupy


@lru_cache(maxsize=1)
def _sweep_kernel():
    from cupyx import jit

    @jit.rawkernel()
    def sweep(nodes, pressure, coefficients, rhs, row, plane, omega):
        thread = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        if thread < nodes.size:
            i = nodes[thread]
            e, f = coefficients[0, i], coefficients[1, i]
            g, h = coefficients[2, i], coefficients[3, i]
            m, n = coefficients[4, i], coefficients[5, i]
            neighbours = (e*pressure[i+1]+f*pressure[i-1]+g*pressure[i+row]
                          +h*pressure[i-row]+m*pressure[i+plane]+n*pressure[i-plane])
            relaxed = (omega/(e+f+g+h+m+n))*(neighbours-rhs[i])
            pressure[i] = relaxed+(1.0-omega)*pressure[i]

    return sweep


class CudaDynamicLegacy:
    """Run the legacy float32 sweeps and update stopping rule with ordered launches."""

    method = 'cupy.cuda-legacy-dynamic-SOR'

    def __init__(self, policy=None, omega=1.78, device=0):
        self.policy = policy or NumericPolicy()
        self.omega = float(omega)
        self.device = int(device)
        if not 0 < np.float32(self.omega) < 2 or self.device < 0:
            raise ValueError('Legacy CUDA SOR requires 0 < omega < 2 and a nonnegative device')

    def _iterate(self, cp, stencil, coefficients, rhs, pressure):
        tolerance = np.float32(self.policy.relative_tolerance)
        if self.policy.max_iterations == 0 or tolerance >= 1:
            return 0, 1.
        sweep = _sweep_kernel()
        _, ny, nx = stencil.shape
        row, plane = np.int64(nx), np.int64(nx*ny)
        # WindWorkspace includes every interior cell with a usable diagonal.
        nodes = stencil.indices
        parity = (nodes//plane+(nodes//row)%ny+nodes%row)%2
        block = 128
        colors = [cp.asarray(nodes[parity == color]) for color in (0, 1)]
        args = (pressure.ravel(), coefficients.reshape(6, -1), rhs.ravel(), row, plane, np.float32(self.omega))
        previous = cp.empty_like(pressure)
        for iteration in range(1, self.policy.max_iterations+1):
            previous[...] = pressure
            for indices in colors:
                if indices.size:
                    sweep[((indices.size+block-1)//block,), (block,)](indices, *args)
            self._apply_ground_boundary(pressure)
            change = self._max_change(cp, pressure, previous)
            if change <= tolerance or not np.isfinite(change):
                break
        return iteration, change

    @staticmethod
    def _apply_ground_boundary(pressure):
        pressure[0] = pressure[1]

    @staticmethod
    def _max_change(cp, pressure, previous):
        return float(cp.max(cp.abs(pressure-previous)).item())

    def _report(self, cp, stencil, pressure, rhs, iterations, change, start):
        host_pressure = cp.asnumpy(pressure)
        if not np.isfinite(host_pressure).all():
            raise FloatingPointError('Non-finite legacy CUDA SOR iterate')
        residual = stencil.residual(host_pressure, cp.asnumpy(rhs))
        return host_pressure, SolverReport(change <= float(np.float32(self.policy.relative_tolerance)),
                                           iterations, residual, change, time.perf_counter()-start,
                                           self.method)

    @staticmethod
    def _rhs(cp, workspace, physics):
        return rhs_from_workspace(workspace, physics, array_module=cp).astype(cp.float32)

    @staticmethod
    def _velocity(cp, workspace, pressure, physics):
        return corrected_velocity(workspace, pressure, physics, array_module=cp, dtype=cp.float32)

    def solve(self, stencil, rhs, initial=None):
        start = time.perf_counter()
        rhs = np.asarray(rhs, dtype=np.float32)
        pressure = np.zeros(stencil.shape, dtype=np.float32) if initial is None else np.array(
            initial, dtype=np.float32, copy=True, order='C')
        if rhs.shape != stencil.shape or pressure.shape != stencil.shape or not np.isfinite(rhs).all() or not np.isfinite(pressure).all():
            raise ValueError('Legacy CUDA SOR requires finite pressure and RHS matching the stencil shape')
        stencil.boundaries(pressure)
        cp = load_cupy(self.device)
        with cp.cuda.Device(self.device):
            coefficients = cp.asarray(np.stack([getattr(stencil, name) for name in _COEFFICIENTS]), dtype=cp.float32)
            rhs, pressure = cp.asarray(rhs, order='C'), cp.asarray(pressure)
            iterations, change = self._iterate(cp, stencil, coefficients, rhs, pressure)
            return self._report(cp, stencil, pressure, rhs, iterations, change, start)

    def project(self, workspace, physics=None):
        cp = load_cupy(self.device)
        with cp.cuda.Device(self.device):
            arrays = {key: cp.asarray(workspace.arrays[key], dtype=cp.float32)
                      for key in ('u0', 'v0', 'w0', 'dz_array', *'efghmn')}
            arrays['icellflag'] = cp.asarray(workspace.arrays['icellflag'])
            device_workspace = replace(workspace, arrays=arrays)
            rhs = self._rhs(cp, device_workspace, physics)
            if not bool(cp.isfinite(rhs).all().item()):
                raise ValueError('Non-finite legacy CUDA SOR RHS')
            stencil = workspace.stencil()
            coefficients = cp.stack([arrays[key] for key in 'efghmn'])
            pressure = cp.zeros(stencil.shape, dtype=cp.float32)
            start = time.perf_counter()
            iterations, change = self._iterate(cp, stencil, coefficients, rhs, pressure)
            host_pressure, report = self._report(cp, stencil, pressure, rhs, iterations, change, start)
            velocity = self._velocity(cp, device_workspace, pressure, physics)
            return {key: cp.asnumpy(value) for key, value in velocity.items()}, host_pressure, report
