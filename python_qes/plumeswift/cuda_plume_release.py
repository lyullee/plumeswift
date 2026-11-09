"""Initialize and insert new experimental CUDA plume particles (GPL-3)."""
from operator import index

from .cuda_particle_buffer import ACTIVE, CudaParticleBuffer


def release_particles(destination, offset, count, interpolation, random, allocate_ids, *, position=(20., 50., 70.)):
    """Apply release stress and Gaussian fluctuations before inserting a prefix."""
    offset, count = index(offset), index(count)
    if not 0 <= offset <= destination.capacity or count < 0:
        raise ValueError('Release offset or count is outside the particle buffer')
    if destination.device != interpolation.device or destination.device != random.device:
        raise ValueError('Release stages must use the same CUDA device')
    if not count:
        return 0
    cp = destination.cp
    with cp.cuda.Device(destination.device):
        location = cp.asarray(position, dtype=cp.float32)
        if location.shape not in ((3,), (count, 3)):
            raise ValueError('Release position must be one XYZ point or one per particle')
        incoming = CudaParticleBuffer(count, device=destination.device)
        incoming.state.fill(ACTIVE)
        incoming.positions[:] = location
        ids = cp.asarray(allocate_ids(count))
        if ids.shape != (count,) or ids.dtype.kind not in 'ui':
            raise ValueError('Expected one integer ID per released particle')
        incoming.ids[:] = ids
        interpolation.sample(incoming, release=True)
        key = object()
        random.create(key, 3*count)
        try:
            draws = random.generate(key).reshape(3, count).T
            incoming.previous_fluctuations *= draws
            return destination.insert_from(incoming, offset, count)
        finally:
            random.destroy(key)
