"""Terrain cut cells from native DEM corner heights (DTEHeightField, GPL-3)."""
from itertools import combinations
from math import sqrt

import numpy as np

from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f


_SIDES = ((0, 1), (1, 2), (2, 3), (0, 3))
_DIAGONAL = (0, 2)
_INTERSECTIONS = ((0, 1), _DIAGONAL, (0, 3), (1, 2), (2, 3))
_CORNERS = ((1, (0, 1), (1, 2)), (3, (0, 3), (2, 3)))


def _terrain_graph(corners, bottom, top):
    height = corners[:, 2]
    state = np.where(height < bottom, -1, np.where(height > top, 1, 0))
    points = list(corners.copy())
    for point in points:
        point[2] = np.clip(point[2], bottom, top)
    edges = set()
    crossings = {side: [-1, -1] for side in _INTERSECTIONS}

    def connect(a, b):
        if a >= 0 and b >= 0:
            edges.add((min(a, b), max(a, b)))

    for side, indices in crossings.items():
        a, b = side
        if state[a] == state[b]:
            if side != _DIAGONAL:
                connect(a, b)
            continue
        low, high = sorted((state[a], state[b]))
        for level, z, outside in ((1, top, 1), (0, bottom, -1)):
            if outside not in (low, high):
                continue
            point = corners[a].copy()
            for axis in (0, 1):
                delta = corners[b, axis]-point[axis]
                if delta:
                    slope = (height[b]-height[a])/delta
                    offset = abs((height[a]-z)/slope)
                    point[axis] += np.copysign(offset, delta)
            point[2] = z
            index = len(points)
            points.append(point)
            indices[level] = index
            for corner in side:
                if state[corner] in (0, outside):
                    connect(corner, index)
        connect(*indices)

    for corner, left, right in _CORNERS:
        if state[corner] != 0:
            level = int(state[corner] > 0)
            connect(crossings[left][level], crossings[right][level])

    mid_bottom, mid_top = crossings[_DIAGONAL]
    if state[0] == state[2] == 0:
        middle = len(points)
        points.append((corners[0]+corners[2])*.5)
        connect(0, middle)
        connect(2, middle)
        for corner, left, right in _CORNERS:
            if state[corner] == 0:
                connect(corner, middle)
            else:
                for index in (*crossings[left], *crossings[right]):
                    connect(index, middle)
    elif (mid_bottom >= 0) != (mid_top >= 0):
        middle = max(mid_bottom, mid_top)
        for corner, left, right in _CORNERS:
            level = int(state[corner] > 0)
            if state[corner] == 0 or min(crossings[left][level], crossings[right][level]) < 0:
                connect(corner, middle)
        for side in _SIDES:
            for index in crossings[side]:
                connect(index, middle)
    elif min(mid_bottom, mid_top) >= 0:
        for corner, _, _ in _CORNERS:
            connect(corner, mid_bottom if state[corner] < 0 else mid_top)
        for side in _SIDES:
            low, high = crossings[side]
            connect(low, mid_bottom)
            connect(low, mid_top)
            connect(high, mid_top)
    else:
        level = int(state[0] > 0)
        a, b, c, d = (crossings[side][level] for side in _SIDES)
        if state[1] != state[0] and state[3] != state[0]:
            for first, second in ((d, a), (d, b), (d, c), (a, b), (c, b)):
                connect(first, second)
            for corner, left, right in _CORNERS:
                if state[corner] == -state[0]:
                    connect(crossings[left][1], crossings[right][0])
        elif state[1] == state[0]:
            connect(1, d)
            connect(1, c)
        else:
            connect(3, a)
            connect(3, b)
    return np.asarray(points), edges, crossings


def _horizontal_opening(points, edges, plane, area):
    face = np.flatnonzero(points[:, 2] >= plane)
    local = points[:, :2]-points[0, :2]
    blocked = np.float32(0.)
    for i, j, k in combinations(face, 3):
        if (i, j) in edges and (i, k) in edges and (j, k) in edges:
            (ax, ay), (bx, by), (cx, cy) = local[[i, j, k]]
            triangle = (ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))*np.float32(.5)
            blocked += abs(triangle)
    return float(np.float32(1.-float(blocked/area)))


def _vertical_opening(points, side, axis, intersections, top, vertical_area, horizontal_area):
    first, second = points[list(side)]
    polygon = []
    if first[2] < top:
        polygon.extend((first, (first[0], first[1], top)))
    elif intersections[1] == -1:
        polygon.append((first[0], first[1], top))
    if second[2] < top:
        polygon.extend((second, (second[0], second[1], top)))
    polygon.extend(points[index] for index in intersections if index != -1)
    if len(polygon) <= 2:
        return 0.
    local = np.asarray(polygon, dtype=np.float32)-points[0]
    centered = local-np.add.accumulate(local, axis=0)[-1]/np.float32(len(local))
    angles = atan2f(centered[:, 2], centered[:, axis])*np.float32(180/np.float32(np.pi))
    # Native merge sort takes points from the right half first when angles tie.
    ordered = local[::-1][np.argsort(angles[::-1], kind='stable')]
    next_point = np.roll(ordered, -1, axis=0)
    sums = (next_point+ordered).astype(float)
    deltas = next_point-ordered
    # Keep the cancelling projections of the native expression. The running
    # float32 sum can round differently if the projections are dropped.
    terms = ((.5*sums[:, 1]*deltas[:, 2])/float(vertical_area)
             +(.5*sums[:, 0]*deltas[:, 2])/float(vertical_area)
             +(.5*sums[:, 0]*deltas[:, 1])/float(horizontal_area))
    opening = np.float32(0.)
    for term in terms:
        opening = np.float32(float(opening)+term)
    return min(1., float(opening))


def _cut_geometry(heights, bottom, top, spacing, width, i, j):
    # Native intersections round in global grid coordinates before the area calculation recenters them.
    x0, x1, y0, y1 = np.asarray([i, i+1, j, j+1], dtype=np.float32)*spacing
    corners = np.array(((x0, y0, heights[0]), (x0, y1, heights[1]),
                        (x1, y1, heights[2]), (x1, y0, heights[3])), dtype=np.float32)
    points, edges, crossings = _terrain_graph(corners, bottom, top)
    horizontal_area = spacing*spacing
    vertical_area = spacing*width
    n, m = (_horizontal_opening(points, edges, z, horizontal_area) for z in (bottom, top))
    e, f, g, h = (_vertical_opening(points, side, axis, crossings[side], top, vertical_area, horizontal_area)
                  for side, axis in (((2, 3), 1), ((0, 1), 1), ((1, 2), 0), ((0, 3), 0)))
    openings = np.array([e, f, g, h, m, n])
    openings[openings <= .05] = 0.
    # Native normals use the blocked area on each face. Its volume accumulator
    # is reset for each face, so only the last face and cut-surface term survive.
    blocked = ((1.-openings)*np.array([vertical_area]*4+[horizontal_area]*2)).astype(np.float32)
    normal_area = blocked[[1, 3, 5]]-blocked[[0, 2, 4]]
    length = np.float32(sqrt(sum(float(value)**2 for value in normal_area)))
    normal = normal_area/length if length else np.zeros(3, dtype=np.float32)
    divisor = np.float32(3)*spacing*spacing*width
    solid_volume = width*blocked[4]/divisor+(points[0, 2]-bottom)*normal[2]*length/divisor
    return openings, normal, max(0., float(np.float32(1)-solid_volume)), points[0]


def apply_cutcell_terrain(workspace, corners):
    a, c = workspace.arrays, workspace.coordinates
    flags = a['icellflag']
    dx, dy = (np.float32(c[axis][1]-c[axis][0]) for axis in 'xy')
    widths = np.asarray(a['dz_array'], dtype=np.float32)
    centers, faces = native_vertical_coordinates(widths)
    lower, upper = faces[1:-1], faces[2:]
    lowest, highest = corners.min(axis=-1), corners.max(axis=-1)
    solid = upper[:, None, None] <= lowest[None, :, :]
    flags[1:, :-1, :-1][solid] = 2
    cut = ~solid & (lower[:, None, None] < highest[None, :, :])
    coefficients = tuple(a[name] for name in 'efghmn')
    for layer, j, i in np.argwhere(cut):
        k = layer+1
        idx = (k, j, i)
        opening, normal, fraction, point = _cut_geometry(corners[j, i], lower[layer], upper[layer], dx, widths[k], i, j)
        a['terrain_volume_frac'][idx] = fraction
        for name, component in zip(('ni', 'nj', 'nk'), normal):
            a[name][idx] = component
        center = np.array([(np.float32(i)+np.float32(.5))*dx, (np.float32(j)+np.float32(.5))*dy, centers[k]])
        offsets = normal*(point-center)
        distance = np.float32(sqrt(sum(float(value)**2 for value in offsets)))
        a['wall_distance'][idx] = distance if a['center_id'][idx] == 1 else -distance
        if fraction <= .1 or not opening.any():
            flags[idx] = 2
        else:
            flags[idx] = 8
            for values, aperture in zip(coefficients, opening):
                values[idx] = aperture
