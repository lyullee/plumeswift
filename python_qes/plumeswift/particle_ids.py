"""Run-scoped particle IDs shared by all plume models in a calculation."""
from operator import index

import numpy as np


class ParticleIDs:
    """Allocate consecutive IDs, with uint32 rollover in native replay."""

    def __init__(self, *, native=False, start=0):
        self.native = native
        self.next_id = index(start)
        if self.native:
            self.next_id %= 1 << 32

    def take(self, count):
        count = index(count)
        if count < 0:
            raise ValueError('Particle count must be nonnegative')
        stop = self.next_id+count
        ids = np.arange(self.next_id, stop, dtype=np.int64)
        if self.native:
            ids &= (1 << 32)-1
            stop %= 1 << 32
        self.next_id = stop
        return ids
