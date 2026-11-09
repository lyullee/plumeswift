"""Staggered wind and cell-centred turbulence interpolation for CUDA QES-Plume."""
from functools import lru_cache
from operator import index

import numpy as np

from .cuda_particle_buffer import ACTIVE, INACTIVE
from .cuda_wind_fields import CudaTurbulenceFields, CudaWindFields


_TURBULENCE = CudaTurbulenceFields.names[:-1]


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    from cupyx import jit

    @jit.rawkernel(device=True)
    def weights(x, y, z, dx, dy, dz, weight_epsilon):
        # The native index denominator uses a double literal even for float32 input.
        xi = cp.floor(np.float64(x)/(np.float64(dx)+1e-7))
        yi = cp.floor(np.float64(y)/(np.float64(dy)+1e-7))
        zi = cp.floor(np.float64(z)/(np.float64(dz)+1e-7))
        wx = np.float32(x/dx-cp.floor(np.float64(x)/(np.float64(dx)+weight_epsilon)))
        wy = np.float32(y/dy-cp.floor(np.float64(y)/(np.float64(dy)+weight_epsilon)))
        wz = np.float32(z/dz-cp.floor(np.float64(z)/(np.float64(dz)+weight_epsilon)))
        return xi, yi, zi, wx, wy, wz

    @jit.rawkernel(device=True)
    def stencil_base(field, x, y, z, nx, ny):
        if not (x >= -1 and x < nx and y >= -1 and y < ny
                and z >= 0 and z < field.shape[1]//(nx*ny)-1):
            return np.int64(-1)
        # Native face and cell arrays use flat corner addresses. A horizontal
        # edge stencil may cross a row or plane while remaining in the array.
        base = np.int64((z*ny+y)*nx+x)
        return base if 0 <= base and base+nx*ny+nx+1 < field.shape[1] else np.int64(-1)

    @jit.rawkernel(device=True)
    def sample(field, component, base, nx, ny, a, b, c):
        aa, bb = np.float32(1.)-a, np.float32(1.)-b
        top = base+nx*ny
        low = (aa*bb*field[component, base] + a*bb*field[component, base+1]
               + a*b*field[component, base+nx+1] + aa*b*field[component, base+nx])
        high = (aa*bb*field[component, top] + a*bb*field[component, top+1]
                + a*b*field[component, top+nx+1] + aa*b*field[component, top+nx])
        return (high-low)*c+low

    @jit.rawkernel(device=True)
    def sigma_of(stress):
        deviation = cp.sqrt(cp.abs(stress))
        return np.float32(1e-8) if deviation == 0 else deviation

    @jit.rawkernel()
    def interpolate(n, release, spacing, positions, states, wind, turbulence, nx, ny,
                    mean, stress, sigma, flux, eps, viscosity):
        p = jit.blockDim.x*jit.blockIdx.x+jit.threadIdx.x
        if p < n and (release or states[p] == ACTIVE):
            dx, dy, dz = spacing[0], spacing[1], spacing[2]
            x, y, z = positions[p, 0], positions[p, 1], positions[p, 2]
            half = np.float32(.5)
            xc, yc, zc = x-half*dx, y-half*dy, z+half*dz
            ci, cj, ck, ca, cb, cc = weights(xc, yc, zc, dx, dy, dz, 1e-7)
            tcx, tcy = nx-1, ny-1
            cell = stencil_base(turbulence, ci, cj, ck, tcx, tcy)
            if cell < 0:
                states[p] = INACTIVE
                return
            if not release:
                ui, uj, uk, ua, ub, uc = weights(x, yc, zc, dx, dy, dz, 1e-7)
                vi, vj, vk, va, vb, vc = weights(xc, y, zc, dx, dy, dz, 1e-4)
                wi, wj, wk, wa, wb, wc = weights(xc, yc, z+dz, dx, dy, dz, 1e-7)
                u = stencil_base(wind, ui, uj, uk, nx, ny)
                v = stencil_base(wind, vi, vj, vk, nx, ny)
                w = stencil_base(wind, wi, wj, wk, nx, ny)
                if u < 0 or v < 0 or w < 0:
                    states[p] = INACTIVE
                    return
                mean[p, 0] = sample(wind, 0, u, nx, ny, ua, ub, uc)
                mean[p, 1] = sample(wind, 1, v, nx, ny, va, vb, vc)
                mean[p, 2] = sample(wind, 2, w, nx, ny, wa, wb, wc)
            for i in range(6):
                stress[p, i] = sample(turbulence, i, cell, tcx, tcy, ca, cb, cc)
            if release:
                sigma[p, 0] = sigma_of(stress[p, 0])
                sigma[p, 1] = sigma_of(stress[p, 3])
                sigma[p, 2] = sigma_of(stress[p, 5])
            else:
                for i in range(3):
                    flux[p, i] = sample(turbulence, 6+i, cell, tcx, tcy, ca, cb, cc)
                eps[p] = cp.maximum(sample(turbulence, 9, cell, tcx, tcy, ca, cb, cc), np.float32(1e-6))
                viscosity[p] = sample(turbulence, 10, cell, tcx, tcy, ca, cb, cc)

    return interpolate


class CudaPlumeInterpolation:
    """Keep the current wind and turbulence record on-device for particle sampling."""

    def __init__(self, velocity, turbulence, spacing, *, device=0):
        spacing = np.asarray(spacing, dtype=np.float32)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError('CUDA plume interpolation requires positive finite XYZ spacing')
        cell_shape = np.shape(turbulence['txx'])
        if velocity['u'].shape != tuple(n+1 for n in cell_shape):
            raise ValueError('CUDA plume interpolation requires matching face and cell grids')
        self.device = device
        self._wind = CudaWindFields(velocity['u'].shape, device=device)
        self._turbulence = CudaTurbulenceFields(cell_shape, device=device)
        self._wind.upload(velocity)
        self._turbulence.upload(self._with_tke(turbulence))
        self.cp = self._wind.cp
        with self.cp.cuda.Device(device):
            self.spacing = self.cp.asarray(spacing)

    @property
    def wind(self):
        return self._wind.data

    @property
    def turbulence(self):
        return self._turbulence.data[:-1]

    @property
    def tke(self):
        """Device TKE field from the current record, or zero for older inputs."""
        return self._turbulence.data[-1]

    @classmethod
    def uniform_fixture(cls, cell_shape, spacing, *, device=0):
        """Build the original QES uniform GPU test fields on a ZYX cell grid."""
        cell_shape = tuple(map(index, cell_shape))
        if len(cell_shape) != 3 or min(cell_shape) < 2:
            raise ValueError('Uniform GPU plume fields require three cell axes of size at least two')
        face_shape = tuple(n+1 for n in cell_shape)
        wind = {name: np.broadcast_to(np.float32(name == 'u'), face_shape) for name in 'uvw'}
        values = dict(txx=1., tyy=1., tzz=1., CoEps=.1, nuT=.1)
        turbulence = {name: np.broadcast_to(np.float32(values.get(name, 0.)), cell_shape) for name in _TURBULENCE}
        return cls(wind, turbulence, spacing, device=device)

    def _with_tke(self, fields):
        if 'tke' in fields:
            return fields
        return dict(fields, tke=np.broadcast_to(np.float32(0), self._turbulence.shape))

    def load_fields(self, velocity, turbulence):
        """Replace fields on the same grid without replacing the particle pool."""
        turbulence = self._with_tke(turbulence)
        self._turbulence._arrays(turbulence)
        self._wind.upload(velocity)
        self._turbulence.upload(turbulence)

    def sample(self, particles, count=None, *, release=False):
        """Write stage fields in place and deactivate particles lacking a valid stencil."""
        count = particles.capacity if count is None else index(count)
        if particles.device != self.device or not 0 <= count <= particles.capacity:
            raise ValueError('CUDA plume particle buffer has the wrong device or count')
        if count:
            block = 128
            with self.cp.cuda.Device(self.device):
                wind, turbulence = self.wind, self.turbulence
                ny, nx = wind.shape[-2:]
                _kernel()[((count+block-1)//block,), (block,)](
                    np.int64(count), np.bool_(release), self.spacing, particles.positions, particles.state,
                    wind.reshape(wind.shape[0], -1), turbulence.reshape(turbulence.shape[0], -1),
                    np.int32(nx), np.int32(ny), particles.mean_velocity, particles.stress,
                    particles.previous_fluctuations, particles.flux_divergence, particles.dissipation,
                    particles.eddy_viscosity)
