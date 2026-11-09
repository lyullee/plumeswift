"""Optional compiled loops for physical-coordinate transport."""

import numpy as np
from numba import literal_unroll, njit

from ..contracts import ModelDomainError


@njit(cache=True, nogil=True)
def _lower_index(axis, point, spacing, left):
    if spacing > 0 and axis[0] <= point <= axis[-1]:
        guess = min(int((point-axis[0])/spacing), len(axis)-1)
        if left:
            guess -= point <= axis[guess]
            if guess >= 0 and guess+1 < len(axis) and axis[guess] < point <= axis[guess+1]:
                return guess
        elif guess+1 < len(axis) and axis[guess] <= point < axis[guess+1]:
            return guess
    if left:
        return np.searchsorted(axis, point, 'left') - 1
    return np.searchsorted(axis, point, 'right') - 1


@njit(cache=True, nogil=True)
def _weight(axis, point, spacing):
    index = min(max(_lower_index(axis, point, spacing, True), 0), len(axis)-2)
    return index, (point - axis[index]) / (axis[index + 1] - axis[index])


@njit(cache=True, nogil=True)
def trilinear(values, positions, x, y, z, spacing):
    result = np.zeros((len(positions), values.shape[3]))
    for row in range(len(positions)):
        i, wx = _weight(x, positions[row, 0], spacing[0])
        j, wy = _weight(y, positions[row, 1], spacing[1])
        k, wz = _weight(z, positions[row, 2], spacing[2])
        for weight in (wx, wy, wz):
            if weight < -.500001 or weight > 1.500001:
                raise ModelDomainError("Interpolation outside grid support")
        xw, yw, zw = (1-wx, wx), (1-wy, wy), (1-wz, wz)
        for dk in range(2):
            for dj in range(2):
                for di in range(2):
                    weight = (xw[di]*yw[dj])*zw[dk]
                    for channel in range(values.shape[3]):
                        result[row, channel] += values[k+dk, j+dj, i+di, channel]*weight
    return result


@njit(cache=True, nogil=True)
def reflect_voxels(positions, displacement, fluctuations, axes, spacing, solid, epsilon, max_crossings):
    p, v = positions.copy(), fluctuations.copy()
    shape = solid.shape[::-1]
    indices = np.empty(3, dtype=np.intp)
    d = np.empty(3)
    for row in range(len(p)):
        for axis in range(3):
            d[axis] = displacement[row, axis]
        for _ in range(max_crossings):
            fraction = np.inf
            normal = 0
            for axis in literal_unroll((0, 1, 2)):
                faces = axes[axis]
                index = _lower_index(faces, p[row, axis], spacing[axis], False)
                if index < 0 or index >= shape[axis]:
                    raise ValueError('Wall reflection starts outside the geometric grid')
                indices[axis] = index
                if d[axis] != 0:
                    boundary = faces[index + (d[axis] > 0)]
                    time = (boundary - p[row, axis])/d[axis]
                    if time < fraction:
                        fraction, normal = time, axis
            if fraction >= 1-1e-13:
                for axis in range(3):
                    p[row, axis] += d[axis]
                break
            step = 1 if d[normal] > 0 else -1
            indices[normal] += step
            if indices[normal] < 0 or indices[normal] >= shape[normal]:
                for axis in range(3):
                    p[row, axis] += d[axis]
                break
            hit = solid[indices[2], indices[1], indices[0]]
            fraction = max(fraction, 0.)
            for axis in range(3):
                p[row, axis] += fraction*d[axis]
                d[axis] *= 1-fraction
            if hit:
                d[normal] *= -1
                v[row, normal] *= -1
                step = -step
            p[row, normal] += epsilon*step
        else:
            raise RuntimeError('Voxel-wall traversal exceeded numerical crossing limit')
    return p, v


def uniform_spacing(axes):
    """Provide an index hint. The kernel still checks its geometric bracket."""
    widths = [np.diff(axis) for axis in axes]
    return tuple(float(width[0]) if len(width) and np.all(width == width[0]) else 0.
                 for width in widths)
