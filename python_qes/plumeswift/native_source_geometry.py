"""Seedable Linux QES source geometry, independent of the transport RNG (GPL-3)."""
from math import log, sqrt

import numpy as np

from .cpu_plume_random import _float32
from .sources import Cube, Line, SphereShell


_UINT32_RANGE = 1 << 32
_UNIFORM_MAX = np.float32(1.-2.**-24)


class NativeSourceGeometry:
    """Match each native geometry's scalar-seeded MT19937 and float operations."""

    def __init__(self, geometry, seed):
        self.geometry = geometry
        # The scalar-seeded NumPy legacy engine and std::mt19937 use the
        # same state initialization. Generator(MT19937(seed)) uses SeedSequence.
        self._engine = np.random.RandomState(seed % _UINT32_RANGE)
        self._saved_normal = None

    def _uniform(self, shape=None):
        raw = self._engine.randint(0, _UINT32_RANGE, size=shape, dtype=np.uint32)
        return np.minimum(raw.astype(np.float32)/np.float32(_UINT32_RANGE), _UNIFORM_MAX)

    def _normal(self):
        if self._saved_normal is not None:
            value, self._saved_normal = self._saved_normal, None
            return value
        while True:
            u, v = self._uniform((2,))
            x = _float32(2*float(u)-1)
            y = _float32(2*float(v)-1)
            radius_sq = _float32(_float32(x*x)+_float32(y*y))
            if 0. < radius_sq <= 1.:
                break
        log_radius = _float32(log(radius_sq))
        scale = _float32(sqrt(_float32(-2*log_radius/radius_sq)))
        self._saved_normal = _float32(x*scale)
        return _float32(y*scale)

    def sample(self, count, rng):
        geometry = self.geometry
        if isinstance(geometry, Cube):
            low, high = np.asarray((geometry.minimum, geometry.maximum), dtype=np.float32)
            return self._uniform((count, 3))*(high-low)+low
        if isinstance(geometry, Line):
            low, high = np.asarray((geometry.start, geometry.end), dtype=np.float32)
            return self._uniform((count, 1))*(high-low)+low
        if isinstance(geometry, SphereShell):
            xyz = np.fromiter((self._normal() for _ in range(3*count)), dtype=np.float32, count=3*count)
            xyz = xyz.reshape(count, 3)
            squares = xyz*xyz
            norm_sq = (squares[:, 0]+squares[:, 1])+squares[:, 2]
            inverse_norm = np.float32(1.)/np.sqrt(norm_sq)
            centre = np.asarray(geometry.centre, dtype=np.float32)
            return centre+np.float32(geometry.radius)*xyz*inverse_norm[:, None]
        raise TypeError(f'Unsupported QES random geometry {type(geometry).__name__}')
