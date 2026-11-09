"""The original CPU release loop's slot-indexed state and ordinal lookups."""
import numpy as np


class NativeReleaseSlots:
    """Retain state at inactive slots for reuse by the original release loop."""

    def __init__(self):
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.fluctuations = self.positions.copy()
        self.previous_fluctuations = self.positions.copy()
        self.stress = np.zeros((0, 3, 3), dtype=np.float32)
        self.delta = self.positions.copy()

    def prepare(self, capacity, reserved, emissions, initial_active, track):
        growth = capacity-len(self.positions)
        if growth:
            self.positions = np.pad(self.positions, ((0, growth), (0, 0)))
            self.fluctuations = np.pad(self.fluctuations, ((0, growth), (0, 0)))
            self.previous_fluctuations = np.pad(self.previous_fluctuations, ((0, growth), (0, 0)))
            self.stress = np.pad(self.stress, ((0, growth), (0, 0), (0, 0)))
            self.delta = np.pad(self.delta, ((0, growth), (0, 0)))
        metadata = []
        offset = 0
        for _, positions, _ in emissions:
            count = len(positions)
            self.positions[reserved[offset:offset+count]] = positions
            # The native metadata loop reads the first count slots after each source.
            metadata.append(self.positions[:count].copy() if track else None)
            offset += count
        # The solidity loop tests the ordinal as a slot, while initializing the
        # newly allocated slot at that ordinal when the test succeeds.
        return initial_active(self.positions[:len(reserved)]), metadata

    def capture(self, batch):
        slots = batch.slots
        self.positions[slots] = batch.positions
        self.fluctuations[slots] = batch.fluctuations
        self.previous_fluctuations[slots] = batch.previous_fluctuations
        self.stress[slots] = batch.previous_stress
        if batch.tracking is not None:
            self.delta[slots] = batch.tracking['delta_fluctuation']
