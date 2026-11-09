"""Load the optional CUDA array backend on request."""


def load_cupy(device):
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('The CUDA backend requires CuPy built for the installed CUDA toolkit') from exc
    if not 0 <= device < cp.cuda.runtime.getDeviceCount():
        raise RuntimeError(f'CUDA device {device} is unavailable')
    return cp
