"""CPU QES stair-step trajectory splitting and reflection (GPL-3.0)."""
from math import floor, trunc

import numpy as np

from .numerics.vector3 import length32, reflect32


def _surface_order(fractions):
    # Three entries use libstdc++ insertion order. A stationary axis has a
    # NaN fraction, which compares false in both directions in the source.
    order = [0, 1, 2]
    for right in (1, 2):
        if fractions[order[right]] < fractions[order[0]]:
            order.insert(0, order.pop(right))
            continue
        left = right
        while left and fractions[order[left]] < fractions[order[left-1]]:
            order[left], order[left-1] = order[left-1], order[left]
            left -= 1
    return order


class NativeStaircase:
    """Endpoint-triggered wall policy for the native CPU replay path."""

    def __init__(self, sampler, solid):
        dx, _, dz = sampler.native_spacing
        self.dx, self.dz = map(np.float32, (dx, dz))
        self.x, self.y, self.zf = (np.asarray(axis, dtype=np.float32)
                                   for axis in (sampler.x, sampler.y, sampler.zf))
        self.solid = np.asarray(solid, dtype=bool).ravel()
        _, ny, nx = solid.shape
        self.strides = (1, nx, nx*ny)

    def _cell(self, position):
        x, y, z = map(float, position)
        dx, dz = float(self.dx), float(self.dz)
        return (floor(x/(dx+1e-9)) + self.strides[1]*floor(y/(dx+1e-9))
                + self.strides[2]*floor((z+dz)/(dz+1e-9)))

    def _indices(self, cell):
        _, nx, plane = self.strides
        k = trunc(cell/plane)
        j = trunc((cell-k*plane)/nx)
        return cell-k*plane-j*nx, j, k

    def _solid(self, cell):
        if not 0 <= cell < len(self.solid):
            raise IndexError('Native wall cell outside grid')
        return self.solid[cell]

    def _surface(self, start, movement):
        cell = self._cell(start)
        i, j, k = self._indices(cell)
        if not (0 <= i < len(self.x) and 0 <= j < len(self.y) and 0 <= k < len(self.zf)-1):
            raise IndexError('Native wall surface outside grid')
        sign = np.sign(movement)
        # The native Domain::getDomainSize returns dx for both horizontal widths.
        half = np.float32(.5)*self.dx
        faces = np.array((self.x[i]+sign[0]*half, self.y[j]+sign[1]*half,
                          self.zf[k+int(sign[2] > 0)]), dtype=np.float32)
        fractions = np.divide(faces-start, movement, out=np.full(3, np.nan, dtype=np.float32),
                              where=movement != 0)
        eps = np.float32(.01)
        acceptable = (fractions >= -eps) & (fractions.astype(float) <= 1.+float(eps))
        near_end = acceptable & (fractions.astype(float) > 1.-float(eps))
        fractions[near_end] -= np.float32(2)*eps
        valid = np.flatnonzero(acceptable)
        if len(valid) == 1:
            axis = valid[0]
        elif len(valid) > 1:
            for axis in _surface_order(fractions):
                if sign[axis] == 0:
                    # The Linux reference reaches an out-of-range lookup
                    # when a stationary axis precedes the blocking surface.
                    raise IndexError('Undefined stationary wall candidate')
                cell += int(sign[axis])*self.strides[axis]
                if self._solid(cell):
                    break
            else:
                raise IndexError('No native wall surface')
        else:
            raise IndexError('No native wall surface')
        normal = np.zeros(3, dtype=np.float32)
        normal[axis] = -sign[axis]
        return fractions[axis], normal

    def _reflect_segment(self, start, direction, length, fluctuation):
        movement = length*direction
        end = start+movement
        for _ in range(10):
            if not self._solid(self._cell(end)):
                norm = length32(movement)
                return end, movement/norm if norm else direction, fluctuation
            fraction, normal = self._surface(start, movement)
            to_wall = fraction*movement
            start = start+to_wall
            remainder = movement-to_wall
            norm = length32(remainder)
            if not norm:
                raise IndexError('Undefined native reflection direction')
            end = start+norm*reflect32(remainder/norm, normal)
            fluctuation = reflect32(fluctuation, normal)
            movement = end-start
        raise IndexError('Native wall reflection limit')

    def _collision_candidates(self, end, active):
        if len(end) < 32:
            return range(len(end))
        dx, dz = float(self.dx), float(self.dz)
        indices = np.floor((end.astype(float)+(0, 0, dz))/(np.array([dx, dx, dz])+1e-9))
        finite = np.isfinite(indices).all(axis=1)
        active[~finite] = False
        # Fall back to scalar integers for large indices, which could lose
        # precision in a flattened address even if their sum is in range.
        _, row, plane = self.strides
        exact = (np.abs(indices) <= 2**53//(3*plane)).all(axis=1)
        fast = np.flatnonzero(exact)
        i, j, k = indices[fast].T
        cells = (i+row*j+plane*k).astype(np.int64)
        inside = (cells >= 0) & (cells < len(self.solid))
        candidates = finite & ~exact
        # The original retries an out-of-range endpoint through trajectory
        # splitting, since a shorter segment may reflect before the exit.
        candidates[fast[~inside]] = True
        candidates[fast[inside]] = self.solid[cells[inside]]
        return np.flatnonzero(candidates)

    def reflect_native(self, positions, displacement, fluctuations):
        delta = np.asarray(displacement, dtype=np.float32)
        end = np.asarray(positions, dtype=np.float32)+delta
        velocity = np.array(fluctuations, dtype=np.float32, copy=True)
        if len(end) != len(delta) or len(end) != len(velocity):
            raise ValueError('Wall reflection requires one displacement and fluctuation per endpoint')
        active = np.ones(len(end), dtype=bool)
        for index in self._collision_candidates(end, active):
            endpoint, movement, fluctuation = end[index], delta[index], velocity[index]
            try:
                cell = self._cell(endpoint)
                if 0 <= cell < len(self.solid):
                    if not self.solid[cell]:
                        continue
                origin = endpoint-movement
                length = length32(movement)
                travelled, segment = np.float32(0), length
                if length:
                    direction = movement/length
                    while abs(float(travelled-length)) >= 1e-3:
                        old = self._indices(self._cell(origin))
                        new = self._indices(self._cell(origin+segment*direction))
                        if any(abs(a-b) > 1 for a, b in zip(old, new, strict=True)):
                            segment *= np.float32(.5)
                            if float(segment) < .125*float(length):
                                raise IndexError('Native wall segment too short')
                        else:
                            travelled += segment
                            origin, direction, fluctuation = self._reflect_segment(
                                origin, direction, segment, fluctuation)
                end[index], velocity[index] = origin, fluctuation
            except (IndexError, ValueError, OverflowError):
                active[index] = False
        return end, velocity, active
