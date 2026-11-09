"""Post-projection logarithmic wall correction from QES Wall.cpp (GPL-3)."""
from dataclasses import dataclass
import math

import numpy as np

from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f, sincosf


_F = np.float32
_VK = _F(.4)


def _speed(speed, near, far, roughness):
    if near <= 0 or far <= 0 or near == roughness:
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        distance_ratio = float(far/near)
        roughness_ratio = float(near/roughness)
        if not (0 < distance_ratio < math.inf and 0 < roughness_ratio < math.inf) or roughness_ratio == 1.:
            return None
        step, denominator = _F(math.log(distance_ratio)), _F(math.log(roughness_ratio))
        friction = _F(.1)
        for _ in range(20):
            corrected = speed-(friction/_VK)*step
            friction = (_VK*corrected)/denominator
    return corrected if np.isfinite(corrected) else None


def _round(value):
    return math.floor(value+.5) if value >= 0 else math.ceil(value-.5)


def _cut_pair(cell, normal, distance, in_air, coordinates, spacing, roughness):
    x, y, z, z_face = coordinates
    k, j, i = cell
    dx, dy, dz = spacing
    origin = np.array((x[i], y[j], z[k]), dtype=np.float32)
    widths = np.array(spacing, dtype=np.float32)
    stride = normal*widths
    limit = _F(math.sqrt(float(dx)**2+float(dy)**2+float(dz)**2))

    def sample(step, vertical):
        delta = (normal*_F(step))*widths
        ii = _round((float(origin[0])-.001+float(delta[0]))/float(dx))-int(normal[0] < 0)
        jj = _round((float(origin[1])-.001+float(delta[1]))/float(dy))-int(normal[1] < 0)
        height = _F(origin[2]+delta[2])
        if vertical == 'centre-up':
            kk = int(np.searchsorted(z[1:], height))+1
        elif vertical == 'centre-down':
            kk = int(np.searchsorted(z, height))-1
        else:
            kk = min(len(z)-1, int(np.searchsorted(z_face[2:], height))+1)
        return kk, jj, ii

    def valid(node):
        kk, jj, ii = node
        return 1 <= kk < len(z)-1 and 0 <= jj < len(y) and 0 <= ii < len(x)

    def wall_distance(node):
        kk, jj, ii = node
        offset = np.array((x[ii], y[jj], z[kk]), dtype=np.float32)-origin
        return _F(math.sqrt(sum(float(value)**2 for value in offset))+float(distance))

    if (in_air and distance > roughness
            and _F(math.log(float(_F(1)+limit/distance))) <= _F(math.log(float(distance/roughness)))):
        second = sample(1, 'face')
        return (cell, second, distance, wall_distance(second)) if valid(second) else None

    # A zero normal cannot advance the search. For a nonzero normal,
    # bound the loop by the first nonzero axis crossing the full domain.
    span = (len(x)*float(dx), len(y)*float(dy), float(z_face[-1]-z_face[0]))
    bounds = [math.ceil(width/abs(float(delta)))+3 for width, delta in zip(span, stride) if delta]
    if not bounds:
        return None
    for step in range(1, min(bounds)):
        first = sample(step, 'centre-up' if normal[2] >= 0 else 'face')
        second = sample(step+1, 'face' if normal[2] >= 0 else 'centre-down')
        if not valid(first) or not valid(second):
            break
        near, far = wall_distance(first), wall_distance(second)
        if (near > roughness and far > 0
                and _F(math.log(float(far/near))) <= _F(math.log(float(near/roughness)))):
            return first, second, near, far
    return None


def _normal(arrays, cell, spacing):
    normal = np.array([arrays[name][cell] for name in ('ni', 'nj', 'nk')], dtype=np.float32)
    if not normal.any():
        dx, dy, dz = spacing
        k = cell[0]
        widths = arrays['dz_array']
        e, f, g, h, m, n = (_F(arrays[name][cell]) for name in 'efghmn')
        front, behind = (opening*(dy*dz)*(dx*dx) for opening in (e, f))
        left, right = (opening*(dx*dz)*(dy*dy) for opening in (g, h))
        below = _F(float(n*(dx*dy))*(float(dz)*.5*float(dz+_F(widths[k+1]))))
        above = _F(float(m*(dx*dy))*(float(dz)*.5*float(dz+_F(widths[k-1]))))
        area = np.array((front-behind, left-right, above-below), dtype=np.float32)
        length = _F(math.sqrt(sum(float(value)**2 for value in area)))
        if length:
            normal = area/length
    normal[np.abs(normal) < _F(.05)] = 0
    return normal


@dataclass(frozen=True)
class WallLogCorrection:
    roughness: float

    def __post_init__(self):
        limits = np.finfo(np.float32)
        if not limits.tiny <= self.roughness <= limits.max:
            raise ValueError('Log-law wall roughness must be positive and representable in float32')

    def __call__(self, workspace, velocity):
        """Modify the projected face arrays in native traversal order."""
        arrays, axes = workspace.arrays, workspace.coordinates
        flags = arrays['icellflag_initial']
        nz, ny, nx = flags.shape
        u, v, w = (velocity[name] for name in 'uvw')
        dx = _F(axes['x'][1]-axes['x'][0])
        # The native wall routine reads dy from the aliased domain accessor.
        spacing = (dx, dx, _F(arrays['dz_array'].min()))
        roughness = _F(self.roughness)
        coordinates = (axes['x'].astype(np.float32), axes['y'].astype(np.float32),
                       *native_vertical_coordinates(arrays['dz_array']))
        cut = np.isin(flags, (7, 8))
        cut[0] = cut[-1] = False
        for k, i, j in np.argwhere(cut.transpose(0, 2, 1)):
            cell = (k, j, i)
            local_spacing = (*spacing[:2], _F(arrays['dz_array'][k]))
            normal = _normal(arrays, cell, local_spacing)
            distance = _F(arrays['wall_distance'][cell])
            pair = _cut_pair(cell, normal, distance, arrays['center_id'][cell] == 1,
                             coordinates, local_spacing, roughness)
            if pair is None:
                continue
            first, second, near, far = pair
            current = np.array((u[first], v[first], w[first]), dtype=np.float32)
            projection = current*normal
            tangent = current-(projection[0]+projection[1]+projection[2])*normal
            length = _F(math.sqrt(sum(float(value)**2 for value in tangent)))
            if length == 0:
                continue
            tangent /= length
            following = np.array((u[second], v[second], w[second]), dtype=np.float32)
            projection = following*tangent
            speed = abs(projection[0]+projection[1]+projection[2])
            corrected = _speed(speed, near, far, roughness) if speed else None
            if corrected is not None:
                u[first], v[first], w[first] = corrected*tangent

        solid = np.isin(flags, (0, 2))
        eligible = ~solid & (flags != 8)
        eligible[0] = eligible[-1] = False
        for axis, normal, parallel1, parallel2, width in ((0, w, u, v, spacing[2]), (2, u, v, w, dx), (1, v, u, w, dx)):
            for direction in (1, -1):
                touches = np.roll(solid, direction, axis=axis)
                if axis != 0:
                    edge = [slice(None)]*3
                    edge[axis] = 0 if direction == 1 else -1
                    touches[tuple(edge)] = False
                for k, i, j in np.argwhere((eligible & touches).transpose(0, 2, 1)):
                    cell = (k, j, i)
                    sample = list(cell)
                    sample[axis] += direction
                    sample = tuple(sample)
                    first, second = _F(parallel1[sample]), _F(parallel2[sample])
                    speed = _F(math.hypot(float(first), float(second)))
                    corrected = _speed(speed, _F(.5)*width, _F(1.5)*width, roughness)
                    if corrected is not None:
                        sine, cosine = sincosf(atan2f(second, first))
                        normal[cell] = 0.
                        parallel1[cell] = corrected*cosine
                        parallel2[cell] = corrected*sine
        return velocity
