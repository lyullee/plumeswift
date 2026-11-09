"""Process-based field sampling for a stationary QES plume workspace."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _initialize_worker(fields):
    global _worker_fields
    _worker_fields = fields


def _sample(fields, positions):
    return (fields.sample_wind(positions), *fields.sample_turbulence_and_distance(positions))


def _sample_worker(positions):
    return _sample(_worker_fields, positions)


class ParticleSamplingPool:
    """Sample a read-only field snapshot in workers and preserve particle order."""

    def __init__(self, fields, workers, *, min_particles=2048):
        if workers < 2 or min_particles < 1:
            raise ValueError('Use at least two workers and a positive particle threshold')
        self.fields = fields
        self.workers = workers
        self.min_particles = min_particles
        self._pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker, initargs=(fields,))
        self.parallel_batches = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._pool.shutdown(cancel_futures=True)

    def sample(self, positions):
        if len(positions) < self.min_particles:
            return _sample(self.fields, positions)
        blocks = np.array_split(positions, min(self.workers, len(positions)))
        parts = list(self._pool.map(_sample_worker, blocks))
        self.parallel_batches += 1
        return tuple(None if arrays[0] is None else np.concatenate(arrays)
                     for arrays in zip(*parts, strict=True))
