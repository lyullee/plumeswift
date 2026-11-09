"""Device storage for staggered QES wind and cell-centred turbulence fields."""
from operator import index

import numpy as np

from .numerics.cuda_runtime import load_cupy


class _CudaGridFields:
    def __init__(self, shape, *, device):
        self.shape = tuple(map(index, shape))
        if len(self.shape) != 3 or min(self.shape) < 2:
            raise ValueError(f'Expected three {self.grid} axes of size at least two')
        self.device = device
        self.cp = load_cupy(device)
        with self.cp.cuda.Device(device):
            self.data = self.cp.empty((len(self.names), *self.shape), dtype=self.cp.float32)

    def _arrays(self, fields):
        if self.data is None:
            raise RuntimeError(f'CUDA {self.grid} fields are closed')
        arrays = tuple(fields[name] for name in self.names)
        if any(np.shape(field) != self.shape for field in arrays):
            raise ValueError(f'Fields must match the allocated CUDA {self.grid} grid')
        return arrays

    def upload(self, fields):
        """Replace the device fields without replacing their allocation."""
        arrays = self._arrays(fields)
        with self.cp.cuda.Device(self.device):
            for component, field in zip(self.data, arrays):
                if isinstance(field, np.ndarray):
                    component.set(np.ascontiguousarray(field, dtype=np.float32))
                else:
                    component[...] = self.cp.asarray(field, dtype=self.cp.float32)

    def download(self, fields):
        """Copy the device fields into the caller's writable host arrays."""
        arrays = self._arrays(fields)
        if any(not isinstance(field, np.ndarray) or not field.flags.writeable
               or not np.can_cast(np.float32, field.dtype, casting='same_kind') for field in arrays):
            raise ValueError('CUDA field download requires writable host arrays with a compatible dtype')
        with self.cp.cuda.Device(self.device):
            for component, field in zip(self.data, arrays):
                np.copyto(field, component.get())

    def close(self):
        """Release ownership of the CuPy allocation."""
        self.data = None


class CudaWindFields(_CudaGridFields):
    """Allocate and transfer float32 U/V/W fields in native ZYX face order."""

    names = ('u', 'v', 'w')
    grid = 'wind face'

    def __init__(self, face_shape, *, device=0):
        super().__init__(face_shape, device=device)
        self.face_shape = self.shape

    def upload(self, velocity):
        super().upload(velocity)

    def download(self, velocity):
        super().download(velocity)


class CudaTurbulenceFields(_CudaGridFields):
    """Allocate and transfer the twelve QES turbulence fields in ZYX cell order."""

    names = ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'div_tau_x', 'div_tau_y', 'div_tau_z', 'CoEps', 'nuT', 'tke')
    grid = 'turbulence cell'

    def __init__(self, cell_shape, *, device=0):
        super().__init__(cell_shape, device=device)
