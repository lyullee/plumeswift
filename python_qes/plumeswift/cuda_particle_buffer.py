"""Device particle storage, compaction and insertion for QES-Plume (GPL-3)."""
from operator import index

from .numerics.cuda_runtime import load_cupy

ACTIVE, INACTIVE, ROGUE = range(3)

# The native partition only carries fields needed before interpolation.
_CARRIED = ('state', 'ids', 'positions', 'previous_fluctuations', 'delta_fluctuations', 'previous_stress')
_INSERTED = ('state', 'ids', 'positions', 'previous_fluctuations', 'stress')


class CudaParticleBuffer:
    """Structure of arrays compatible with the experimental CUDA plume stages."""

    def __init__(self, capacity, *, device=0):
        self.capacity = index(capacity)
        if self.capacity < 0:
            raise ValueError('Particle capacity must be nonnegative')
        self.device = device
        self.cp = load_cupy(device)
        cp = self.cp
        with cp.cuda.Device(device):
            self.state = cp.full(capacity, INACTIVE, dtype=cp.int32)
            self.ids = cp.zeros(capacity, dtype=cp.uint32)
            self.dissipation = cp.zeros(capacity, dtype=cp.float32)
            self.eddy_viscosity = cp.zeros(capacity, dtype=cp.float32)
            self.positions = cp.zeros((capacity, 3), dtype=cp.float32)
            self.mean_velocity = cp.zeros_like(self.positions)
            self.fluctuations = cp.zeros_like(self.positions)
            self.previous_fluctuations = cp.zeros_like(self.positions)
            self.delta_fluctuations = cp.zeros_like(self.positions)
            self.flux_divergence = cp.zeros_like(self.positions)
            self.stress = cp.zeros((capacity, 6), dtype=cp.float32)
            self.previous_stress = cp.zeros_like(self.stress)

    def _check_peer(self, other):
        if self is other or self.device != other.device:
            raise ValueError('Expected distinct particle buffers on the same device')

    def slot_counts(self):
        """Return active and reusable slot counts, including rogue slots as reusable."""
        with self.cp.cuda.Device(self.device):
            active = int(self.cp.count_nonzero(self.state == ACTIVE).item())
        return active, self.capacity-active

    def compact_to(self, destination, *, clear_source=False):
        """Copy active particles in order, optionally marking all source slots reusable."""
        self._check_peer(destination)
        if self.capacity != destination.capacity:
            raise ValueError('Partition buffers must have equal capacities')
        cp = self.cp
        with cp.cuda.Device(self.device):
            selected = cp.flatnonzero(self.state == ACTIVE)
            active = selected.size
            for field in _CARRIED:
                getattr(destination, field)[:active] = getattr(self, field)[selected]
            destination.state[active:] = INACTIVE
            if clear_source:
                self.state.fill(INACTIVE)
        return active

    def insert_from(self, incoming, offset, count):
        """Copy releases into free slots and return the inserted prefix length."""
        self._check_peer(incoming)
        offset, count = index(offset), index(count)
        if not 0 <= offset <= self.capacity or not 0 <= count <= incoming.capacity:
            raise ValueError('Insertion range is outside the particle buffers')
        inserted = min(count, self.capacity-offset)
        with self.cp.cuda.Device(self.device):
            for field in _INSERTED:
                getattr(self, field)[offset:offset+inserted] = getattr(incoming, field)[:inserted]
        return inserted
