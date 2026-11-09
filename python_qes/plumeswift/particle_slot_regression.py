"""Deterministic counterpart of the original ManagedContainer example."""
import numpy as np

from .cpu_plume_random import LinuxQESRandom
from .particle_ids import ParticleIDs
from .particle_slots import reserve_particle_slots


def run_managed_container(*, steps=10_000, release_count=1000, seed=0, on_step=None):
    if steps < 1 or release_count < 1:
        raise ValueError('Managed container regression needs positive steps and release count')
    active = np.empty(0, dtype=bool)
    ids = np.empty(0, dtype=np.uint32)
    id_sequence = ParticleIDs(native=True)
    inserted = 0
    for _ in range(steps):
        slots, capacity = reserve_particle_slots(np.flatnonzero(active), release_count, len(active))
        growth = capacity-len(active)
        if growth:
            active = np.pad(active, (0, growth))
            ids = np.pad(ids, (0, growth))
        ids[slots] = id_sequence.take(release_count)
        inserted += release_count
        active[slots] = True
        # The C++ example reconstructs the generator every iteration. Use
        # one fixed seed in place of its wall-clock seed.
        draws = LinuxQESRandom(seed).random(capacity)
        active[draws.astype(np.float64) > .8] = False
        if on_step is not None:
            on_step(capacity, int(active.sum()), inserted, int(ids[slots[-1]]), len(ids), slots)
    last_id = int(ids[slots[-1]])
    invariants_passed = len(ids) == len(active) and last_id == (inserted-1) % (1 << 32)
    return dict(steps=steps, release_count=release_count, seed=seed, capacity=len(active),
                active_particles=int(active.sum()), inserted=inserted, last_particle_id=last_id,
                companion_length=len(ids), invariants_passed=invariants_passed)
