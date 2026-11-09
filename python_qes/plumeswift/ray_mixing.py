"""Python implementation of the QES OptiX method-3 ray sampling equations."""
from dataclasses import dataclass

import numpy as np

from .triangle_mesh import TriangleMesh


def _xorwow_uniform(count):
    mask = (1 << 32)-1
    salt0 = (1099087573*(129 ^ 0xaad26b49)) & mask
    salt1 = (2591861531*0xf7dcefdd) & mask
    weyl = (6615241+salt0+salt1) & mask
    a, b, c, d, e = ((123456789+salt0) & mask, 362436069 ^ salt0, (521288629+salt1) & mask,
                     88675123 ^ salt1, (5783321+salt0) & mask)
    bits = np.empty(count, dtype=np.uint32)
    for index in range(count):
        t = a ^ (a >> 2)
        a, b, c, d, e = b, c, d, e, (e ^ (e << 4) ^ t ^ (t << 1)) & mask
        weyl = (weyl+362437) & mask
        bits[index] = (e+weyl) & mask
    return bits.astype(np.float32)*np.float32(2**-32)+np.float32(2**-33)


def ray_directions(samples):
    """Native cardinal rays followed by the seeded CUDA XORWOW sequence."""
    if int(samples) != samples or samples < 1:
        raise ValueError('Method 3 requires a positive integer ray count')
    cardinal = np.array(((0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)), dtype=np.float32)
    if samples <= len(cardinal):
        return cardinal[:samples]
    uniform = _xorwow_uniform(2*(samples-len(cardinal))).reshape(-1, 2)
    theta = (uniform[:, 0].astype(float)*np.pi).astype(np.float32)
    phi = ((uniform[:, 1]*np.float32(2)).astype(float)*np.pi).astype(np.float32)
    sin_theta = np.sin(theta)
    # CUDA source uses sin(theta)*sin(phi) for y.
    vectors = np.column_stack((np.cos(phi)*sin_theta, sin_theta*np.sin(phi), np.cos(theta)))
    norm = np.sqrt(np.sum(vectors*vectors, axis=1))
    return np.concatenate((cardinal, vectors/norm[:, None]))


def _courtyard_roof(rings, top):
    """Triangulate the material intervals between horizontal vertex levels."""
    edges = np.concatenate([np.stack((ring[:-1], ring[1:]), axis=1) for ring in rings])
    levels = np.unique(edges[..., 1])
    for lower, upper in zip(levels[:-1], levels[1:]):
        mid = .5*(lower+upper)
        active = edges[(edges[:, 0, 1] > mid) != (edges[:, 1, 1] > mid)]
        if not len(active):
            continue
        a, b = active[:, 0], active[:, 1]
        slope = (b[:, 0]-a[:, 0])/(b[:, 1]-a[:, 1])
        order = np.argsort(a[:, 0]+(mid-a[:, 1])*slope)
        lower_x = (a[:, 0]+(lower-a[:, 1])*slope)[order]
        upper_x = (a[:, 0]+(upper-a[:, 1])*slope)[order]
        for left, right in zip(range(0, len(order), 2), range(1, len(order), 2)):
            a = (lower_x[left], lower, top)
            b = (lower_x[right], lower, top)
            c = (upper_x[right], upper, top)
            d = (upper_x[left], upper, top)
            yield (a, b, c)
            yield (a, c, d)


def mixing_scene(workspace, buildings=(), terrain_mesh=None):
    """Build the original method-3 geometry plus the supported courtyard rings."""
    a, c = workspace.arrays, workspace.coordinates
    triangles = []
    if terrain_mesh is None:
        _, ny, nx = a['u0'].shape
        dx = np.float32(c['x'][1]-c['x'][0])
        east, north = float(np.float32(nx)*dx), float(np.float32(ny)*dx)
        triangles.extend((((0., 0., 0.), (east, 0., 0.), (east, north, 0.)),
                          ((0., 0., 0.), (east, north, 0.), (0., north, 0.))))
    for building in buildings:
        rings = tuple(np.asarray(ring, dtype=np.float32) for ring in (building.vertices, *building.holes))
        base = np.float32(building.base_height)
        top = base+np.float32(building.height)
        for ring in rings:
            for first, second in zip(ring[:-1], ring[1:]):
                a, b = (*first, base), (*second, base)
                c, d = (*first, top), (*second, top)
                triangles.append((a, b, d))
                # Original side triangles share the vertical edge. For
                # courtyards use a complete quad on each new ring wall.
                triangles.append((a, d, c) if building.holes else (c, d, b))
        if building.holes:
            triangles.extend(_courtyard_roof(rings, top))
        else:
            roof = np.column_stack((rings[0][:-1], np.full(len(rings[0])-1, top)))
            triangles.extend((roof[0], first, second) for first, second in zip(roof[1:-1], roof[2:]))
    vertices = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    if terrain_mesh is not None:
        vertices = np.concatenate((terrain_mesh.triangles, vertices))
    return TriangleMesh(vertices)


@dataclass(frozen=True)
class RayMixing:
    samples: int = 2000
    backend: str = 'cpu'

    def __post_init__(self):
        if self.backend not in ('cpu', 'cuda', 'optix'):
            raise ValueError('Mixing ray backend must be cpu, cuda or optix')

    def __call__(self, workspace, mesh):
        a, c = workspace.arrays, workspace.coordinates
        flags = a['icellflag']
        k, j, i = np.nonzero((flags != 0) & (flags != 2))
        dx = np.float32(c['x'][1]-c['x'][0])
        dz = np.float32(a['dz_array'][0])
        points = ((np.column_stack((i, j, k))+.5)*[dx, dx, dz]).astype(np.float32)
        directions = ray_directions(self.samples)
        if self.backend == 'cuda':
            from .cuda_ray_mixing import cuda_ray_distances
            distances = cuda_ray_distances(mesh, points, directions)
        elif self.backend == 'optix':
            from .optix_ray_mixing import optix_ray_distances
            distances = optix_ray_distances(mesh, points, directions)
        else:
            distances = self._cpu_distances(mesh, points, directions)
        # Native leaves FLT_MAX for rays missing every triangle. A terrain
        # halo can produce misses; use the default local height for those cells.
        missed = ~np.isfinite(distances)
        distances[missed] = np.maximum(0., c['z'][k[missed]]-a['terrain'][j[missed], i[missed]])
        result = np.zeros(flags.shape)
        result[k, j, i] = distances.astype(np.float32)
        return result

    @staticmethod
    def _cpu_distances(mesh, points, directions):
        distances = np.full(len(points), np.inf)
        for sample in range(0, len(directions), 32):
            rays = directions[sample:sample+32]
            count = len(rays)
            tile = max(1, 16384//count)
            for start in range(0, len(points), tile):
                stop = min(start+tile, len(points))
                origins = np.repeat(points[start:stop], count, axis=0)
                current = distances[start:stop]
                maximum = np.repeat(np.minimum(current, 1e16), count)
                batch = mesh.ray_parameters(origins, np.tile(rays, (stop-start, 1)), maximum=maximum)
                np.minimum(current, batch.reshape(-1, count).min(axis=1), out=current)
        return distances
