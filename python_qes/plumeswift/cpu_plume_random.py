"""Linux libstdc++ streams for QES Random and RandomSingleton (GPL-3)."""
from functools import lru_cache
from math import log, prod, sqrt
from operator import index
from struct import Struct
from time import time_ns

import numpy as np


_FLOAT32 = Struct('<f')
_MODULUS = 2147483647
_LARGEST_UNIFORM = 1.-2.**-24
_MULTIPLIER = 16807
_BATCH_SIZE = 65536


def _float32(value):
    return _FLOAT32.unpack(_FLOAT32.pack(value))[0]


def resolve_cpu_seed(seed):
    """Use the original whole-second wall clock when no explicit seed is given."""
    return time_ns()//1_000_000_000 if seed is None else index(seed)


@lru_cache(maxsize=1)
def _powers():
    power = 1
    factors = np.empty(_BATCH_SIZE, dtype=np.int64)
    for i in range(_BATCH_SIZE):
        power = (_MULTIPLIER*power) % _MODULUS
        factors[i] = power
    factors.setflags(write=False)
    return factors


class LinuxQESRandom:
    """libstdc++ minstd_rand0 and QES's cached polar normal transform."""

    def __init__(self, seed=None):
        # Linux's engine seeds from the unsigned 64-bit conversion of long.
        self._state = (resolve_cpu_seed(seed) % (1 << 64)) % _MODULUS or 1
        self._saved_normal = None

    def _uniform(self):
        self._state = (_MULTIPLIER*self._state) % _MODULUS
        value = _float32(self._state-1)/2147483648.
        return min(value, _LARGEST_UNIFORM)

    def _normal(self):
        if self._saved_normal is not None:
            sample, self._saved_normal = self._saved_normal, None
            return sample
        while True:
            x = _float32(2*self._uniform()-1)
            y = _float32(2*self._uniform()-1)
            radius_sq = _float32(_float32(x*x)+_float32(y*y))
            if 0. < radius_sq < 1.:
                break
        scale = _float32(sqrt(-2*log(radius_sq)/radius_sq))
        self._saved_normal = _float32(y*scale)
        return _float32(x*scale)

    @staticmethod
    def _draw(sample, size):
        if size is None:
            return sample()
        shape = size if isinstance(size, tuple) else (index(size),)
        if any(dimension < 0 for dimension in shape):
            raise ValueError('Negative random array dimension')
        return np.fromiter((sample() for _ in range(prod(shape))), dtype=np.float32,
                           count=prod(shape)).reshape(shape)

    def random(self, size=None):
        if size is None:
            return self._uniform()
        shape = tuple(map(index, size)) if isinstance(size, tuple) else (index(size),)
        if any(dimension < 0 for dimension in shape):
            raise ValueError('Negative random array dimension')
        count = prod(shape)
        if count < 256:
            return self._draw(self._uniform, shape)
        result = np.empty(count, dtype=np.float32)
        for start in range(0, count, _BATCH_SIZE):
            block = result[start:start+_BATCH_SIZE]
            states = self._states(len(block))
            block[:] = self._uniforms(states)
            self._state = int(states[-1])
        return result.reshape(shape)

    def _states(self, count):
        # Both factors are below the 31-bit modulus, so int64 preserves
        # the product exactly before reducing it.
        return (self._state*_powers()[:count]) % _MODULUS

    @staticmethod
    def _uniforms(states):
        return np.minimum((states-1).astype(np.float32)*np.float32(2.**-31), _LARGEST_UNIFORM)

    def normal(self, size=None):
        if size is None:
            return self._normal()
        shape = tuple(map(index, size)) if isinstance(size, tuple) else (index(size),)
        if any(dimension < 0 for dimension in shape):
            raise ValueError('Negative random array dimension')
        count = prod(shape)
        if count < 256:
            return self._draw(self._normal, shape)
        result = np.empty(count, dtype=np.float32)
        cursor = 0
        if self._saved_normal is not None:
            result[0], self._saved_normal = self._saved_normal, None
            cursor = 1
        while cursor < count:
            remaining = count-cursor
            pairs_needed = (remaining+1)//2
            pairs = min(_BATCH_SIZE//2, pairs_needed+pairs_needed//3)
            states = self._states(2*pairs)
            uniforms = self._uniforms(states)
            xy = (2*uniforms-1).reshape(-1, 2)
            radii = xy[:, 0]**2+xy[:, 1]**2
            accepted = np.flatnonzero((radii > 0) & (radii < 1))[:pairs_needed]
            radius = radii[accepted].astype(np.float64)
            scale = np.sqrt(-2*np.log(radius)/radius).astype(np.float32)
            normals = (xy[accepted]*scale[:, None]).ravel()
            used = min(remaining, len(normals))
            result[cursor:cursor+used] = normals[:used]
            cursor += used
            if cursor == count:
                # Discard speculative draws after the last accepted pair.
                self._state = int(states[2*accepted[-1]+1])
                if len(normals) > used:
                    self._saved_normal = float(normals[-1])
            else:
                self._state = int(states[-1])
        return result.reshape(shape)


class NativeCPUStreams:
    """Replay GNU OpenMP static streams or the identical non-OpenMP single stream."""

    def __init__(self, seed, workers):
        workers = index(workers)
        if workers < 1:
            raise ValueError('Native CPU replay needs at least one worker')
        seed = resolve_cpu_seed(seed)
        self.streams = tuple(LinuxQESRandom(seed ^ worker) for worker in range(workers))

    def owners(self, indices, count):
        workers = len(self.streams)
        width, remainder = divmod(count, workers)
        ends = np.arange(1, workers+1, dtype=np.int64)
        return np.searchsorted(width*ends+np.minimum(ends, remainder), indices, side='right')

    def initial_draws(self, ordinals, count):
        owners = self.owners(ordinals, count)
        draws = np.empty((len(ordinals), 3), dtype=np.float32)
        for owner in np.unique(owners):
            mask = owners == owner
            draws[mask] = self.streams[owner].normal(size=(int(mask.sum()), 3))
        return draws
