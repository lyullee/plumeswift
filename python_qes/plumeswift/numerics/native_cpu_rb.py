"""Float32 CPU projection from QES Solver_CPU_RB.cpp and Solver_CPU.cpp (GPL-3.0)."""
import time

import numpy as np

from ..contracts import NumericPolicy, PhysicalCoefficients, SolverReport
from ..wind_projection import zero_solid_faces


class NativeCPUSOR:
    """Shared projection and stopping rule for the two original CPU solvers."""

    def __init__(self, policy=None):
        self.policy = policy or NumericPolicy()

    @staticmethod
    def _relax(previous, weighted, omega):
        return weighted+(np.float32(1)-omega)*previous

    def solve(self, stencil, rhs, initial=None):
        start = time.perf_counter()
        rhs = np.asarray(rhs, dtype=np.float32)
        pressure = (np.zeros(stencil.shape, dtype=np.float32) if initial is None
                    else np.array(initial, dtype=np.float32, copy=True, order='C'))
        if (rhs.shape != stencil.shape or pressure.shape != stencil.shape
                or not np.isfinite(rhs).all() or not np.isfinite(pressure).all()):
            raise ValueError('Native CPU SOR needs finite pressure and RHS matching the stencil')
        nz, ny, nx = stencil.shape
        k, j, i = np.indices((nz-2, ny-2, nx-2))
        k, j, i = k+1, j+1, i+1
        nodes = ((k*ny+j)*nx+i).ravel()
        coefficients = [np.asarray(getattr(stencil, name), dtype=np.float32).ravel()
                        for name in ('east', 'west', 'north', 'south', 'up', 'down')]
        omega = np.float32(1.78)
        groups = []
        for ids in self._groups(nodes, (k+j+i).ravel()):
            e, f, g, h, m, n = (coefficient[ids] for coefficient in coefficients)
            diagonal = e+f+g+h+m+n
            if np.any(diagonal <= 0):
                raise ValueError('Native CPU SOR encountered a zero interior diagonal')
            groups.append((ids, e, f, g, h, m, n, omega/diagonal, rhs.ravel()[ids]))

        flat = pressure.ravel()
        plane = nx*ny
        tolerance = float(np.float32(self.policy.relative_tolerance))
        change = 1.
        iterations = 0
        while iterations < self.policy.max_iterations and change > tolerance:
            previous = pressure[1:].copy()
            for ids, e, f, g, h, m, n, relaxation, source in groups:
                neighbours = (e*flat[ids+1]+f*flat[ids-1]+g*flat[ids+nx]
                              +h*flat[ids-nx]+m*flat[ids+plane]+n*flat[ids-plane])
                flat[ids] = self._relax(flat[ids], relaxation*(neighbours-source), omega)
            pressure[0] = pressure[1]
            change = float(np.max(np.abs(pressure[1:]-previous)))
            iterations += 1
        if not np.isfinite(pressure).all():
            raise FloatingPointError('Non-finite native CPU SOR iterate')
        residual = stencil.residual(pressure, rhs)
        return pressure, SolverReport(change <= tolerance, iterations, residual, change,
                                      time.perf_counter()-start, self.method)

    @staticmethod
    def _downward_flux(coefficient, dz, adjacent_dz, velocity):
        return coefficient*dz*np.float32(.5)*(dz+adjacent_dz)*velocity

    @staticmethod
    def _correct(field, coefficient, width, difference, alpha):
        factor = np.float32(1)/(np.float32(2)*np.float32(alpha**2))
        return field+factor*coefficient*width*difference

    def initial_divergence(self, workspace, physics=None):
        """Compute the original weighted divergence over all horizontal cells."""
        p = physics or PhysicalCoefficients()
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = (np.float32(np.median(np.diff(c[axis]))) for axis in 'xy')
        e, f, g, h, m, n = (np.asarray(a[key], dtype=np.float32) for key in 'efghmn')
        u, v, w = (np.asarray(a[key], dtype=np.float32) for key in ('u0', 'v0', 'w0'))
        dz = np.asarray(a['dz_array'], dtype=np.float32)[:, None, None]
        mid = slice(1, -1)
        faces = slice(1, -2)
        horizontal = ((e[mid]*u[faces, :-1, 1:]-f[mid]*u[faces, :-1, :-1])*dx
                      +(g[mid]*v[faces, 1:, :-1]-h[mid]*v[faces, :-1, :-1])*dy)
        # Only the first vertical half literal promotes the C++ product.
        upward = (m[mid]*dz[mid]).astype(np.float64)*.5*(dz[mid]+dz[2:])*w[2:-1, :-1, :-1]
        downward = self._downward_flux(n[mid], dz[mid], dz[:-2], w[faces, :-1, :-1])
        rhs = np.zeros_like(e)
        rhs[mid] = (-2*p.alpha_horizontal**2)*(horizontal+(upward-downward))
        return rhs

    def project(self, workspace, physics=None, *, initial=None):
        p = physics or PhysicalCoefficients()
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = (np.float32(np.median(np.diff(c[axis]))) for axis in 'xy')
        f, h, n = (np.asarray(a[key], dtype=np.float32) for key in 'fhn')
        u, v, w = (np.array(a[key], dtype=np.float32, copy=True) for key in ('u0', 'v0', 'w0'))
        dz = np.asarray(a['dz_array'], dtype=np.float32)[:, None, None]
        mid = slice(1, -1)
        faces = slice(1, -2)
        rhs = self.initial_divergence(workspace, p)
        pressure, report = self.solve(workspace.stencil(), rhs, initial)

        center = pressure[mid, 1:, 1:]
        cell = (mid, slice(1, None), slice(1, None))
        area = (faces, mid, mid)
        for field, coefficient, width, difference, alpha in (
            (u, f, dx, center-pressure[mid, 1:, :-1], p.alpha_horizontal),
            (v, h, dy, center-pressure[mid, :-1, 1:], p.alpha_horizontal),
            (w, n, dz[mid], center-pressure[:-2, 1:, 1:], p.alpha_vertical),
        ):
            field[area] = self._correct(field[area], coefficient[cell], width, difference, alpha)
        zero_solid_faces(u, v, w, a['icellflag'])
        return dict(u=u, v=v, w=w), pressure, report


class NativeRedBlackSOR(NativeCPUSOR):
    """Sweep even cells and then odd cells."""

    method = 'numpy.native-cpu-red-black-SOR'

    @staticmethod
    def _groups(nodes, levels):
        return [nodes[levels % 2 == color] for color in (0, 1)]


class NativeSerialSOR(NativeCPUSOR):
    """Sweep the native serial dependency order using independent wavefronts."""

    method = 'numpy.native-cpu-serial-SOR'

    @staticmethod
    def _groups(nodes, levels):
        # In the original k/j/i traversal, west, south and down have already
        # been updated. A level of k+j+i preserves exactly those dependencies.
        order = np.argsort(levels, kind='stable')
        boundaries = np.flatnonzero(np.diff(levels[order]))+1
        return np.split(nodes[order], boundaries)


class HistoricalCPUSOR(NativeSerialSOR):
    """Historical CPUSolver.cpp, including its double literal promotions.

    Use initial_divergence alone for the historical solveWind=true branch,
    which populates the RHS and leaves the existing wind and multiplier alone.
    """

    method = 'numpy.historical-cpu-SOR'

    @staticmethod
    def _relax(previous, weighted, omega):
        return weighted+(1.-float(omega))*previous.astype(np.float64)

    @staticmethod
    def _downward_flux(coefficient, dz, adjacent_dz, velocity):
        return (coefficient*dz).astype(np.float64)*.5*(dz+adjacent_dz)*velocity

    @staticmethod
    def _correct(field, coefficient, width, difference, alpha):
        return field+(1./(2*alpha**2))*coefficient.astype(np.float64)*width*difference
