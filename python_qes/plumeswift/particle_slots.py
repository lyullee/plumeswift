"""CPU slot allocation compatible with QES ManagedContainer (GPL-3)."""
import numpy as np


def first_free_slots(occupied, count):
    """Return the first free slots in the original increasing order."""
    if not count:
        return np.empty(0, dtype=np.int64)
    # Among this many slots, at least count are free even if some occupied
    # slots have larger indices.
    limit = len(occupied)+count
    free = np.ones(limit, dtype=bool)
    free[occupied[occupied < limit]] = False
    return np.flatnonzero(free)[:count]


def reserve_particle_slots(occupied, count, capacity):
    """Grow by the entire release count when the current allocation is full."""
    if len(occupied)+count > capacity:
        capacity += count
    return first_free_slots(occupied, count), capacity
