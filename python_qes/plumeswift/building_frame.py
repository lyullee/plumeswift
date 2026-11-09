"""Shared geometric state for the original building parameterizations."""
from dataclasses import dataclass
from math import atan2, cos, pi, sin
import numpy as np
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f, sincosf
from .upwind import _round


def face_angles(local, *, native_math=False, wrap_slack=0.):
    dx, dy = np.diff(local, axis=0).T
    if native_math:
        angles = (atan2f(dy, dx).astype(float)+.5*pi).astype(np.float32)
        wrapped = angles.astype(float) > pi+wrap_slack
        angles[wrapped] = angles[wrapped].astype(float)-2*pi
    else:
        angles = np.arctan2(dy, dx)+.5*pi
        angles[angles > pi+wrap_slack] -= 2*pi
    return angles


@dataclass
class BuildingFrame:
    vertices: np.ndarray
    center: np.ndarray
    local: np.ndarray
    direction: float
    reference: tuple[float, float]
    k_start: int
    k_end: int
    horizontal_bounds: tuple[int, int, int, int]
    width: float
    length: float
    rectangular: bool

    def rooftop_faces(self, *, wrap_slack=0., native_math=False):
        """Preserve the native float angle decision at the 30-degree boundary."""
        if native_math:
            angles = face_angles(self.local, native_math=True, wrap_slack=wrap_slack).astype(float)
            return np.flatnonzero(abs(angles) >= pi-float(np.float32(pi/6)))
        u, v = map(np.float32, self.reference)
        direction = np.float32(atan2(v, u))
        cs, sn = np.float32(cos(direction)), np.float32(sin(direction))
        x, y = (self.vertices.astype(np.float32)-self.center.astype(np.float32)).T
        dx = np.diff(x*cs+y*sn)
        dy = np.diff(-x*sn+y*cs)
        angles = np.array([np.float32(atan2(ey, ex)) for ex, ey in zip(dx, dy)], dtype=float)
        angles = np.float32(angles+.5*pi).astype(float)
        wrapped = angles > pi+wrap_slack
        angles[wrapped] = np.float32(angles[wrapped]-2*pi)
        return np.flatnonzero(abs(angles) >= pi-float(np.float32(pi/6)))

    @classmethod
    def create(cls, workspace, building, *, native_math=False, native_dy_alias=False):
        a, c = workspace.arrays, workspace.coordinates
        dx, dy = c['x'][1]-c['x'][0], c['y'][1]-c['y'][0]
        scalar = np.float32 if native_math else float
        spacing = np.array([dx, dy], dtype=scalar)
        dx, dy = spacing
        vertices = np.asarray(building.vertices, dtype=scalar)
        if native_math:
            center = np.add.accumulate(vertices[:-1], axis=0)[-1]/scalar(len(vertices)-1)
            z, z_face = native_vertical_coordinates(a['dz_array'])
        else:
            center = vertices[:-1].mean(axis=0)
            z, z_face = c['z'], c['z_face']
        # Native rooftop traverses the bounds stored by setCellFlags. The
        # centroid wind sample still uses the actual per-axis cell spacing.
        bounds_spacing = spacing[0] if native_dy_alias else spacing
        istart, jstart = np.trunc(vertices.min(axis=0)/bounds_spacing).astype(int)
        iend, jend = np.trunc(vertices.max(axis=0)/bounds_spacing+1).astype(int)
        base, height = scalar(building.base_height), scalar(building.height)
        kstart = 1+np.flatnonzero(base <= z_face[2:])[0]
        kend = 1+np.flatnonzero(base+height < z[1:])[0]
        ic, jc = _round(center[0]/dx)-1, _round(center[1]/dy)-1
        _, ny, nx = a['u0'].shape
        ic, jc = min(max(ic, 0), nx-1), min(max(jc, 0), ny-1)
        uh, vh = a['u0'][kend, jc, ic], a['v0'][kend, jc, ic]
        area_terms = vertices[:-1, 0]*vertices[1:, 1]-vertices[:-1, 1]*vertices[1:, 0]
        if native_math:
            direction = scalar(atan2(vh, uh))
            sn, cs = map(scalar, sincosf(direction))
            x, y = (vertices-center).T
            local = np.column_stack((x*cs+y*sn, -x*sn+y*cs))
            area = abs(np.add.accumulate(scalar(.5)*area_terms)[-1])
            lower, upper = np.minimum(local.min(axis=0), 0), np.maximum(local.max(axis=0), 0)
            width, length = area/(upper-lower)
        else:
            direction = np.arctan2(vh, uh)
            cs, sn = np.cos(direction), np.sin(direction)
            local = (vertices-center)@np.array([[cs, -sn], [sn, cs]])
            area = abs(.5*np.sum(area_terms))
            width, length = area/np.ptp(local[:, 0]), area/np.ptp(local[:, 1])
        rectangular = len(vertices) == 5
        if rectangular:
            edges = np.diff(vertices, axis=0)
            normals = np.c_[edges[:, 1], -edges[:, 0]]
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            rectangular = bool(np.all(abs(np.sum(normals[:-1]*normals[1:], axis=1)) < abs(np.cos(np.pi/2-np.deg2rad(1)))))
        return cls(vertices, center, local, float(direction), (float(uh), float(vh)), int(kstart), int(kend),
                   (int(istart), int(iend), int(jstart), int(jend)), float(width), float(length), rectangular)
