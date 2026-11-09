"""Exact initial QES fire-front distance in grid cells (GPL-3)."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from .numerics.cuda_runtime import load_cupy

_DISTANCE_LIMIT = 1000


def cpu_front(ignited):
    if not ignited.any():
        return np.full(ignited.shape, float(_DISTANCE_LIMIT))
    return np.minimum(distance_transform_edt(~ignited), _DISTANCE_LIMIT)


def _tiles(size):
    # CuPy pads the distance-transform input to a square. Bound that allocation
    # with overlapping windows on large domains. Ignition outside the halo can
    # only contribute a distance that is already capped.
    tile_size = 1024 if size > 3072 else size
    for start in range(0, size, tile_size):
        end = min(start+tile_size, size)
        lower = max(start-_DISTANCE_LIMIT, 0)
        upper = min(end+_DISTANCE_LIMIT, size)
        yield slice(start, end), slice(lower, upper), slice(start-lower, end-lower)


def cuda_front(ignited):
    """Evaluate the native float32 distance with a device distance transform."""
    cp = load_cupy(0)
    from cupyx.scipy.ndimage import distance_transform_edt as cuda_edt

    front = np.full(ignited.shape, float(_DISTANCE_LIMIT))
    with cp.cuda.Device(0):
        for rows, rows_halo, rows_crop in _tiles(ignited.shape[0]):
            for cols, cols_halo, cols_crop in _tiles(ignited.shape[1]):
                window = ignited[rows_halo, cols_halo]
                if window.any():
                    distances = cuda_edt(cp.asarray(~window), float64_distances=False)
                    core = distances[rows_crop, cols_crop]
                    cp.minimum(core, _DISTANCE_LIMIT, out=core)
                    front[rows, cols] = cp.asnumpy(core)
    return front
