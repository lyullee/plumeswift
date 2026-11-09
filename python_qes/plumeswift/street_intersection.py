"""Standalone legacy street intersection routines from StreetIntersection.cpp (GPL-3)."""
import numpy as np

from .geometry import zero_solid_faces


def _scan(flags, axis, direction, intersection=None):
    lines = np.moveaxis(flags, axis, -1)
    matched = np.zeros(lines.shape, dtype=bool)
    width = lines.shape[-1]
    positions = range(1, width) if direction == 1 else range(width-2, -1, -1)
    joined = None if intersection is None else np.moveaxis(intersection, axis, -1)
    for index in np.ndindex(lines.shape[:-1]):
        if joined is not None and index[-1] == 0:
            continue
        line, output = lines[index], matched[index]
        active = False
        start = 0
        for position in positions:
            flag = line[position]
            canyon_behind = line[position-direction] == 6
            if joined is None:
                if canyon_behind and flag not in (0, 2, 6):
                    active = True
                    start = position
                if flag in (0, 1, 2, 6):
                    active = False
            else:
                active = joined[index][position] and (active or canyon_behind)
            output[position] = active
        if joined is None and active:
            # The legacy reverse loop also clears the high-index tail.
            output[start:] = False
    return np.moveaxis(matched, -1, axis)


def mark_street_intersections(workspace):
    """Apply the legacy directional scans and mark intersecting cells as 12."""
    zero_solid_faces(workspace)
    flags = workspace.arrays['icellflag']
    x = _scan(flags, 2, 1) | _scan(flags, 2, -1)
    y = _scan(flags, 1, 1) | _scan(flags, 1, -1)
    joined = x & y
    forward = _scan(flags, 2, 1, joined) | _scan(flags, 1, 1, joined)
    backward = _scan(flags, 2, -1, joined) | _scan(flags, 1, -1, joined)
    mask = forward & backward
    flags[mask] = 12
    return mask


def smooth_street_intersections(workspace, iterations=10):
    """Run the legacy in-place six-face relaxation on flag-12 interfaces."""
    if int(iterations) != iterations or iterations < 0:
        raise ValueError('Street intersection iterations must be a nonnegative integer')
    if iterations == 0:
        return
    arrays = workspace.arrays
    intersection = arrays['icellflag'] == 12
    updates = []
    for k, j, i in np.argwhere(intersection[1:, 1:, 1:])+1:
        fields = tuple(arrays[name] for name, neighbour in (
            ('u0', intersection[k, j, i-1]), ('v0', intersection[k, j-1, i]),
            ('w0', intersection[k-1, j, i])) if neighbour)
        if not fields:
            continue
        weights = tuple(float(arrays[name][k, j, i]) for name in 'efghmn')
        total = sum(weights)
        if not np.isfinite(total) or total <= 0.:
            raise ValueError('Street intersection face needs positive finite total aperture')
        adjacent = ((k, j, i+1), (k, j, i-1), (k, j+1, i), (k, j-1, i), (k+1, j, i), (k-1, j, i))
        updates.append(((k, j, i), fields, adjacent, weights, total))
    for _ in range(int(iterations)):
        for cell, fields, adjacent, weights, total in updates:
            for field in fields:
                field[cell] = sum(weight*field[index] for weight, index in zip(weights, adjacent))/total
