"""Ordered ports of the original QES CUDA global and shared memory solvers."""

import numpy as np

from ..contracts import PhysicalCoefficients
from ..wind_projection import zero_solid_faces
from .cuda_dynamic_legacy import CudaDynamicLegacy


class CudaGlobalMemory(CudaDynamicLegacy):
    """Original float32 red-black updates and a global maximum-change reduction."""

    method = 'cupy.cuda-global-memory-SOR'

    def __init__(self, policy=None, omega=1.78, device=0, *, native_boundary=False):
        super().__init__(policy, omega, device)
        self.native_boundary = native_boundary

    def _apply_ground_boundary(self, pressure):
        if not self.native_boundary:
            return super()._apply_ground_boundary(pressure)
        _, ny, nx = pressure.shape
        plane = ny*nx
        block_size = 1024
        threads = ((plane+block_size-1)//block_size)*block_size
        # QES launches using the cell plane but checks against the face plane.
        # Snapshot the overlapping source span to make the ground copy ordered.
        stop = min(threads, (nx+1)*(ny+1), pressure.size-plane)
        flat = pressure.ravel()
        flat[:stop] = flat[plane:plane+stop].copy()

    @staticmethod
    def _rhs(cp, workspace, physics):
        p = physics or PhysicalCoefficients()
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = (np.float32(np.median(np.diff(c[axis]))) for axis in 'xy')
        e, f, g, h, m, n = (a[key][1:-1] for key in 'efghmn')
        u, v, w = (a[key] for key in ('u0', 'v0', 'w0'))
        dz = a['dz_array'][:, None, None]
        horizontal = ((e*u[1:-2, :-1, 1:]-f*u[1:-2, :-1, :-1])*dx
                      +(g*v[1:-2, 1:, :-1]-h*v[1:-2, :-1, :-1])*dy)
        # The original vertical terms promote to double at the 0.5 literal.
        up = (m*dz[1:-1]).astype(cp.float64)*.5*(dz[1:-1]+dz[2:])*w[2:-1, :-1, :-1]
        down = (n*w[1:-2, :-1, :-1]*dz[1:-1]).astype(cp.float64)*.5*(dz[1:-1]+dz[:-2])
        rhs = cp.zeros_like(a['e'])
        rhs[1:-1] = (-2*p.alpha_horizontal**2)*(horizontal+(up-down))
        return rhs

    @staticmethod
    def _velocity(cp, workspace, pressure, physics):
        p = physics or PhysicalCoefficients()
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = (np.float32(np.median(np.diff(c[axis]))) for axis in 'xy')
        u, v, w = (a[key].copy() for key in ('u0', 'v0', 'w0'))
        center = pressure[1:-1, 1:, 1:]
        area = (slice(1, -2), slice(1, -1), slice(1, -1))
        cell = (slice(1, -1), slice(1, None), slice(1, None))
        for field, coefficient, spacing, difference, alpha in (
            (u, 'f', dx, center-pressure[1:-1, 1:, :-1], p.alpha_horizontal),
            (v, 'h', dy, center-pressure[1:-1, :-1, 1:], p.alpha_horizontal),
            (w, 'n', a['dz_array'][1:-1, None, None], center-pressure[:-2, 1:, 1:], p.alpha_vertical),
        ):
            factor = a[coefficient][cell].astype(cp.float64)/(2*alpha**2)
            field[area] = field[area]+factor*spacing*difference
        zero_solid_faces(u, v, w, a['icellflag'], array_module=cp)
        return dict(u=u, v=v, w=w)


class CudaSharedMemory(CudaGlobalMemory):
    """Original block maxima with a separate, ordered final reduction."""

    method = 'cupy.cuda-shared-memory-SOR'

    @staticmethod
    def _max_change(cp, pressure, previous):
        block = 1024
        delta = cp.abs(pressure-previous).ravel()
        padded = cp.pad(delta, (0, (-delta.size) % block))
        block_maxima = padded.reshape(-1, block).max(axis=1)
        return float(block_maxima.max().item())
