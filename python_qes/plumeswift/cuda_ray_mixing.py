"""Python CUDA BVH traversal for the mixing-length ray distance queries."""
from functools import lru_cache

import numpy as np

from .numerics.cuda_runtime import load_cupy


def _threaded_tree(mesh):
    bounds = np.empty((len(mesh.nodes), 2, 3))
    ranges = np.empty((len(mesh.nodes), 3), dtype=np.int32)
    order = []

    def visit(node, escape):
        low, high, left, right = mesh.nodes[node]
        bounds[node, 0], bounds[node, 1] = low, high
        if right is None:
            start = len(order)
            order.extend(left)
            ranges[node] = (start, len(order), escape)
        else:
            ranges[node] = (-1, left, escape)
            visit(left, right)
            visit(right, escape)

    visit(0, len(mesh.nodes))
    return bounds, ranges, mesh.triangles[order]


@lru_cache(maxsize=1)
def _kernel():
    from cupyx import jit

    @jit.rawkernel(device=True)
    def enters(bounds, node, points, point, directions, ray, limit):
        near = np.float64(0.)
        far = limit
        for axis in range(3):
            origin = points[point, axis]
            direction = directions[ray, axis]
            low = bounds[node, 0, axis]
            high = bounds[node, 1, axis]
            if direction == 0.:
                if origin < low or origin > high:
                    return False
            else:
                reciprocal = 1./direction
                entry = (low-origin)*reciprocal
                exit = (high-origin)*reciprocal
                if entry > exit:
                    entry, exit = exit, entry
                near = entry if entry > near else near
                far = exit if exit < far else far
                if near > far:
                    return False
        return True

    @jit.rawkernel(device=True)
    def intersect(edges, triangle, points, point, directions, ray, limit):
        ex, ey, ez = edges[triangle, 1, 0], edges[triangle, 1, 1], edges[triangle, 1, 2]
        fx, fy, fz = edges[triangle, 2, 0], edges[triangle, 2, 1], edges[triangle, 2, 2]
        dx, dy, dz = directions[ray, 0], directions[ray, 1], directions[ray, 2]
        px, py, pz = dy*fz-dz*fy, dz*fx-dx*fz, dx*fy-dy*fx
        determinant = px*ex+py*ey+pz*ez
        if determinant == 0.:
            return limit
        inverse = 1./determinant
        tx = points[point, 0]-edges[triangle, 0, 0]
        ty = points[point, 1]-edges[triangle, 0, 1]
        tz = points[point, 2]-edges[triangle, 0, 2]
        u = (tx*px+ty*py+tz*pz)*inverse
        if u < 0. or u > 1.:
            return limit
        qx, qy, qz = ty*ez-tz*ey, tz*ex-tx*ez, tx*ey-ty*ex
        v = (dx*qx+dy*qy+dz*qz)*inverse
        if v < 0. or u+v > 1.:
            return limit
        t = (qx*fx+qy*fy+qz*fz)*inverse
        return t if t >= 0. and t < limit else limit

    @jit.rawkernel()
    def trace(points, directions, bounds, ranges, edges, current, result):
        thread = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        rays = directions.shape[0]
        point = thread//rays
        if point < points.shape[0]:
            ray = thread%rays
            best = current[point]
            node = 0
            while node < ranges.shape[0]:
                escape = ranges[node, 2]
                if enters(bounds, node, points, point, directions, ray, best):
                    start = ranges[node, 0]
                    end = ranges[node, 1]
                    if start < 0:
                        escape = end
                    else:
                        for triangle in range(start, end):
                            best = intersect(edges, triangle, points, point, directions, ray, best)
                node = escape
            result[point, ray] = best

    return trace


def cuda_ray_distances(mesh, points, directions, *, device=0):
    """Minimum hit per point, in bounded batches with one upload per scene."""
    cp = load_cupy(device)
    bounds, ranges, edges = _threaded_tree(mesh)
    edges[:, 1:] -= edges[:, :1]
    # One representable step above the inclusive CPU limit also serves as
    # the miss sentinel. The traversal always compares against this bound.
    limit = np.nextafter(1e16, np.inf)
    with cp.cuda.Device(device):
        geometry = tuple(cp.asarray(value) for value in (bounds, ranges, edges))
        origins = cp.asarray(points, dtype=cp.float64)
        rays = cp.asarray(directions, dtype=cp.float64)
        best = cp.full(len(points), limit, dtype=cp.float64)
        block = 128
        split = min(5, len(directions))
        batches = [(0, split), *((start, min(start+256, len(directions)))
                                 for start in range(split, len(directions), 256))]
        for first, last in batches:
            count = last-first
            tile = (1 << 20)//count
            for start in range(0, len(points), tile):
                stop = min(start+tile, len(points))
                output = cp.empty((stop-start, count), dtype=cp.float64)
                grid = ((output.size+block-1)//block,)
                _kernel()[grid, (block,)](origins[start:stop], rays[first:last], *geometry, best[start:stop], output)
                best[start:stop] = output.min(axis=1)
        result = cp.asnumpy(best)
    result[result == limit] = np.inf
    return result
