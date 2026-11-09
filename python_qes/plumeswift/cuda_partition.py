"""Float32 pivot partition for the original CUDA diagnostic (GPL-3)."""
import numpy as np

from .numerics.cuda_runtime import load_cupy


def partition_cuda(values, pivot, *, device=0):
    """Return a device array split at ``<= pivot`` and its lower-group size.

    The original GPU kernel leaves group order unspecified. This port keeps
    input order within both groups. NaNs enter the upper group.
    """
    threshold = np.asarray(pivot, dtype=np.float32)
    if threshold.ndim != 0:
        raise ValueError('CUDA partition pivot must be a scalar')
    cp = load_cupy(device)
    with cp.cuda.Device(device):
        data = cp.asarray(values, dtype=cp.float32)
        if data.ndim != 1:
            raise ValueError('CUDA pivot partition requires a one-dimensional array')
        lower = data <= threshold[()]
        left, right = data[lower], data[~lower]
        return cp.concatenate((left, right)), left.size
