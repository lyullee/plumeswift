"""QES triangle height queries with a Python BVH (Triangle.cpp/BVH.cpp, GPL-3).

The mesh retains individual triangle vertices, including discontinuous duplicate
vertices in native DEM construction. Height is the maximum nonnegative hit.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass(frozen=True)
class RayHit:
    triangle: int
    parameter: float
    distance: float
    point: np.ndarray
    normal: np.ndarray


class TriangleMesh:
    def __init__(self, triangles, leaf_size=8):
        self.triangles = np.asarray(triangles, dtype=float)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not len(self.triangles) or not np.isfinite(self.triangles).all():
            raise ValueError('Expected finite N x 3 x 3 triangle vertices')
        if leaf_size < 1:
            raise ValueError('BVH leaf size must be positive')
        self.low = self.triangles.min(axis=1); self.high = self.triangles.max(axis=1)
        self.nodes = []
        def build(indices, depth=0):
            node = len(self.nodes); self.nodes.append(None)
            lo, hi = self.low[indices].min(axis=0), self.high[indices].max(axis=0)
            if len(indices) <= leaf_size:
                self.nodes[node] = (lo, hi, indices, None)
            else:
                axis = depth % 2
                order = np.argsort(self.low[indices, axis]+self.high[indices, axis], kind='stable')
                middle = len(indices)//2
                left = build(indices[order[:middle]], depth+1)
                right = build(indices[order[middle:]], depth+1)
                self.nodes[node] = (lo, hi, left, right)
            return node
        build(np.arange(len(self.triangles)))
        # Native ray ties select the right subtree. Preserve its tree ordering
        # independently of the configurable, larger height-query leaves.
        def ray_order(ids, depth=0):
            if len(ids) <= 2:
                return list(ids)
            axis = depth % 2
            midpoint = self.low[ids, axis]+self.high[ids, axis]
            # Native mergeSort chooses the right item on equal midpoints.
            order = np.argsort(midpoint[::-1], kind='stable')
            ids = ids[::-1][order]
            middle = len(ids)//2
            return ray_order(ids[:middle], depth+1)+ray_order(ids[middle:], depth+1)
        self._ray_rank = np.empty(len(self.triangles), dtype=int)
        self._ray_rank[ray_order(np.arange(len(self.triangles)))] = np.arange(len(self.triangles))
        normals = np.cross(self.triangles[:, 1]-self.triangles[:, 0],
                           self.triangles[:, 2]-self.triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        self.normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)

    @cached_property
    def _native_geometry(self):
        triangles = self.triangles.astype(np.float32)
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.sqrt((normals*normals).sum(axis=1))
        normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)
        return triangles, normals

    def ray_hit(self, origin, direction, *, max_parameter=np.inf, native_math=False):
        """Nearest forward hit; direction is not normalized (as in QES Ray).

        Zero-direction slab components are handled explicitly to avoid native
        0*infinity NaNs. Parallel/degenerate triangles cannot produce a hit.
        QES ignores its triangle t0/t1 arguments; t=0 is therefore accepted.
        max_parameter is an explicit caller filter, defaulting to the full ray.
        """
        dtype = np.float32 if native_math else float
        origin, direction = np.asarray(origin, dtype=dtype), np.asarray(direction, dtype=dtype)
        if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([origin, direction]).all():
            raise ValueError('A ray requires finite three-component origin and direction')
        if np.isnan(max_parameter) or max_parameter < 0:
            raise ValueError('Invalid maximum ray parameter')
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        best = None
        triangles, normals = self._native_geometry if native_math else (self.triangles, self.normals)
        stack = [0]
        while stack:
            lo, hi, first, second = self.nodes[stack.pop()]
            near, far = 0., max_parameter if best is None or native_math else best.parameter
            for axis in range(3):
                if direction[axis] == 0:
                    if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                        far = -1.; break
                else:
                    t0, t1 = sorted(((lo[axis]-origin[axis])/direction[axis],
                                     (hi[axis]-origin[axis])/direction[axis]))
                    near, far = max(near, t0), min(far, t1)
            if near > far:
                continue
            if second is not None:
                stack.extend((first, second)); continue
            for index in first:
                a, b, c = triangles[index]
                aa, bb, cc = a-b; dd, ee, ff = a-c
                gg, hh, ii = direction; jj, kk, ll = a-origin
                eihf, gfdi, dheg = ee*ii-hh*ff, gg*ff-dd*ii, dd*hh-ee*gg
                determinant = aa*eihf+bb*gfdi+cc*dheg
                if determinant == 0:
                    continue
                akjb, jcal, blkc = aa*kk-jj*bb, jj*cc-aa*ll, bb*ll-kk*cc
                beta = (jj*eihf+kk*gfdi+ll*dheg)/determinant
                gamma = (ii*akjb+hh*jcal+gg*blkc)/determinant
                t = -(ff*akjb+ee*jcal+dd*blkc)/determinant
                if gamma < 0 or gamma > 1 or beta < 0 or beta > 1-gamma or t < 0 or t > max_parameter:
                    continue
                point = origin+t*direction
                distance = (float(np.float32(np.linalg.norm((point-origin).astype(float))))
                            if native_math else float(t*norm))
                key = distance if native_math else t
                previous = None if best is None else (best.distance if native_math else best.parameter)
                if previous is None or key < previous or (key == previous and self._ray_rank[index] > self._ray_rank[best.triangle]):
                    best = RayHit(int(index), float(t), distance, point, normals[index].copy())
        return best

    def ray_parameters(self, origins, directions, *, maximum=1e16):
        """Find first-hit parameters for a batch of rays against the BVH."""
        origins, directions = (np.asarray(array, dtype=float) for array in (origins, directions))
        if (origins.ndim != 2 or origins.shape[1:] != (3,) or directions.shape != origins.shape
                or not np.isfinite(origins).all() or not np.isfinite(directions).all()):
            raise ValueError('Expected paired finite N x 3 ray origins and directions')
        limits = np.broadcast_to(np.asarray(maximum, dtype=float), (len(origins),))
        if np.isnan(limits).any() or np.any(limits < 0):
            raise ValueError('Ray parameter limits must be nonnegative')
        hits = np.full(len(origins), np.inf)
        stationary = directions == 0
        reciprocal = np.divide(1., directions, out=np.zeros_like(directions), where=~stationary)
        stack = [(0, np.arange(len(origins)))]
        while stack:
            node, ids = stack.pop()
            lo, hi, first, second = self.nodes[node]
            p = origins[ids]
            fixed = stationary[ids]
            inv = reciprocal[ids]
            low, high = (lo-p)*inv, (hi-p)*inv
            entry = np.where(fixed, -np.inf, np.minimum(low, high)).max(axis=1)
            exit = np.where(fixed, np.inf, np.maximum(low, high)).min(axis=1)
            accepted = ((np.maximum(entry, 0.) <= np.minimum(exit, np.minimum(limits[ids], hits[ids])))
                        & ~np.any(fixed & ((p < lo) | (p > hi)), axis=1))
            ids = ids[accepted]
            if not len(ids):
                continue
            if second is not None:
                stack.extend(((first, ids), (second, ids)))
                continue
            origin, direction = origins[ids], directions[ids]
            for index in first:
                a, b, c = self.triangles[index]
                first_edge, second_edge = b-a, c-a
                p = np.cross(direction, second_edge)
                determinant = np.einsum('ni,i->n', p, first_edge)
                invert = np.divide(1., determinant, out=np.zeros_like(determinant), where=determinant != 0)
                delta = origin-a
                u = np.einsum('ni,ni->n', delta, p)*invert
                q = np.cross(delta, first_edge)
                v = np.einsum('ni,ni->n', direction, q)*invert
                t = np.einsum('ni,i->n', q, second_edge)*invert
                valid = ((determinant != 0) & (u >= 0.) & (v >= 0.) & (u+v <= 1.) & (t >= 0.)
                         & (t <= limits[ids]) & (t < hits[ids]))
                hits[ids[valid]] = t[valid]
        return hits

    def height(self, x, y, *, native_math=False):
        """Return the highest upward hit, or -1 outside the terrain."""
        dtype = np.float32 if native_math else float
        xx, yy = np.broadcast_arrays(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError('Non-finite mesh query')
        triangles = self._native_geometry[0] if native_math else self.triangles
        shape = xx.shape; xx, yy = xx.ravel(), yy.ravel()
        result = np.full(len(xx), -1., dtype=dtype)
        stack = [(0, np.arange(len(xx)))]
        while stack:
            node_id, ids = stack.pop()
            lo, hi, first, second = self.nodes[node_id]
            if native_math:
                lo, hi = lo.astype(dtype), hi.astype(dtype)
            inside = (xx[ids] >= lo[0]) & (xx[ids] <= hi[0]) & (yy[ids] >= lo[1]) & (yy[ids] <= hi[1])
            ids = ids[inside]
            if not len(ids):
                continue
            if second is not None:
                stack.extend([(first, ids), (second, ids)])
                continue
            for triangle_id in first:
                a, b, c = triangles[triangle_id]
                # Same determinant and barycentric equations as getHeightTo.
                aa, bb, cc = a-b; dd, ee, ff = a-c
                jj, kk, ll = a[0]-xx[ids], a[1]-yy[ids], a[2]
                determinant = aa*ee-bb*dd
                if determinant == 0:
                    continue  # zero projected area cannot define a height
                akjb = aa*kk-jj*bb
                height = -(ff*akjb+ee*(jj*cc-aa*ll)+dd*(bb*ll-kk*cc))/determinant
                gamma = akjb/determinant
                beta = (jj*ee-kk*dd)/determinant
                hit = (height >= 0) & (gamma >= 0) & (gamma <= 1) & (beta >= 0) & (beta <= 1-gamma)
                selected = ids[hit]
                result[selected] = np.maximum(result[selected], height[hit])
        return result.reshape(shape)

    @classmethod
    def read_obj(cls, path):
        """Read triangular OBJ faces, including native indexed terrain exports."""
        from pathlib import Path
        points, faces = [], []
        with Path(path).open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                fields = line.partition('#')[0].split()
                if not fields:
                    continue
                if fields[0] == 'v':
                    if len(fields) < 4:
                        raise ValueError(f'Incomplete OBJ vertex at line {number}')
                    points.append(tuple(map(float, fields[1:4])))
                elif fields[0] == 'f':
                    if len(fields) != 4:
                        raise ValueError(f'OBJ requires triangular faces at line {number}')
                    indices = [int(value.split('/', 1)[0]) for value in fields[1:]]
                    face = [index-1 if index > 0 else len(points)+index for index in indices]
                    if 0 in indices or min(face) < 0:
                        raise ValueError(f'Invalid OBJ vertex index at line {number}')
                    faces.append(face)
        if not points or not faces or max(map(max, faces)) >= len(points):
            raise ValueError('OBJ has no triangles or references unavailable vertices')
        return cls(np.asarray(points)[faces])

    def write_obj(self, path, *, indexed=False):
        """Write native-indexed or unshared OBJ vertices without geometric rounding."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = self.triangles.reshape(-1, 3)
        faces = np.arange(1, len(points)+1).reshape(-1, 3)
        if indexed:
            vertices = {}
            indices = [vertices.setdefault(tuple(point), len(vertices)+1) for point in points]
            points = tuple(vertices)
            faces = np.asarray(indices).reshape(-1, 3)
        with path.open('w', encoding='ascii') as stream:
            for point in points:
                stream.write('v '+' '.join(format(float(v), '.17g') for v in point)+'\n')
            for a, b, c in faces:
                stream.write(f'f {a} {b} {c}\n')
