"""MTGP32 draws and named normal buffers for the CUDA plume (GPL-3)."""
from operator import index
from weakref import finalize

import numpy as np

from .numerics.cuda_runtime import load_cupy


class CudaPlumeRandom:
    """Use one device RNG for particle releases and advection draws."""

    def __init__(self, *, seed=1234, device=0):
        seed = index(seed)
        if not 0 <= seed < 2**64:
            raise ValueError('Seed must be an integer between 0 and 2**64 - 1')
        self.device = device
        self.cp = load_cupy(device)
        from cupy_backends.cuda.libs import curand

        self._curand = curand
        with self.cp.cuda.Device(device):
            self._generator = curand.createGenerator(curand.CURAND_RNG_PSEUDO_MTGP32)
            finalize(self, curand.destroyGenerator, self._generator)
            curand.setPseudoRandomGeneratorSeed(self._generator, seed)
        self._buffers = {}

    def create(self, key, length):
        length = index(length)
        if length < 0:
            raise ValueError('Random buffer length must be nonnegative')
        with self.cp.cuda.Device(self.device):
            self._buffers[key] = self.cp.zeros(length, dtype=self.cp.float32)
        return self._buffers[key]

    def generate(self, key, mean=0., stddev=1.):
        mean, stddev = np.float32(mean), np.float32(stddev)
        if not np.isfinite([mean, stddev]).all() or stddev < 0:
            raise ValueError('Expected finite normal distribution parameters and nonnegative deviation')
        buffer = self._buffers[key]
        if buffer.size:
            with self.cp.cuda.Device(self.device):
                self._set_stream()
                # cuRAND generates normals in pairs. Discard only the spare draw.
                output = self.cp.empty(buffer.size+1, dtype=self.cp.float32) if buffer.size % 2 else buffer
                self._curand.generateNormal(self._generator, output.data.ptr, output.size, mean, stddev)
                if output is not buffer:
                    buffer[:] = output[:-1]
        return buffer

    def _set_stream(self):
        self._curand.setStream(self._generator, self.cp.cuda.get_current_stream().ptr)

    def _draw(self, size, dtype, generate):
        size = index(size)
        if size < 0:
            raise ValueError('Random sample size must be nonnegative')
        output = self.cp.empty(size, dtype=dtype)
        if size:
            self._set_stream()
            generate(self._generator, output.data.ptr, size)
        return output

    def uniform(self, size, *, signed=False):
        """Draw float32 uniform samples on (0, 1] or the native signed interval."""
        with self.cp.cuda.Device(self.device):
            output = self._draw(size, self.cp.float32, self._curand.generateUniform)
            if signed:
                output *= np.float32(2)
                output -= np.float32(1)
            return output

    def uint32(self, size):
        """Draw the full 32-bit words used by the original GPU bit-frequency test."""
        with self.cp.cuda.Device(self.device):
            return self._draw(size, self.cp.uint32, self._curand.generate)

    def get(self, key):
        return self._buffers[key]

    def destroy(self, key):
        self._buffers.pop(key, None)
