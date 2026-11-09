"""Original QES sidewall recirculation and internal boundary layer.

Sidewall.cpp, GPL-3.0-only. The native last-v-face shell-width reuse in
cell-flag assignment is retained explicitly for reproducibility.
"""
from math import sin, cos, sqrt, log, ceil, atan2, pi
import numpy as np
from .building_frame import BuildingFrame, face_angles
from .numerics.linux_float_angles import atan2f, sincosf


def sidewall(workspace, building, wake_state, *, roughness=.01, native_dy_alias=True, native_math=False):
    a, c = workspace.arrays, workspace.coordinates
    scalar = np.float32 if native_math else float
    frame = BuildingFrame.create(workspace, building, native_math=native_math)
    if wake_state is None:
        # Native width_eff is assigned by polygonWake. An unset width
        # supplies no defined sidewall extent in a legacy wake-off run.
        return frame
    dx = scalar(c['x'][1]-c['x'][0]); actual_dy = scalar(c['y'][1]-c['y'][0])
    dy = dx if native_dy_alias else actual_dy; dxy = min(dx, actual_dy)
    nz, ny, nx = a['u0'].shape
    vertices = frame.vertices; nfaces = len(vertices)-1
    relative = face_angles(frame.local, native_math=native_math).astype(float, copy=False)
    tol = float(scalar(np.deg2rad(10)))
    eligible = (abs(relative) >= .5*pi-tol) & (abs(relative) <= .5*pi+tol)
    if not eligible.any():
        return frame
    width = wake_state.frame.width
    height, roughness = float(scalar(building.height)), scalar(roughness)
    scale = scalar(min(width, height)**(2/3)*max(width, height)**(1/3))
    rcx, vd = scalar(.9*float(scale)), scalar(.5*.22*float(scale))
    half_rcx, half_cell = .5*float(rcx), .5*float(dxy)
    ypref = scalar(float(vd)/sqrt(half_rcx))
    flags = a['icellflag']
    for face in np.flatnonzero(eligible):
        left = relative[face] > 0
        adjacent = (face+nfaces-1) % nfaces if left else (face+1) % nfaces
        if abs(relative[adjacent]) < pi-tol:
            continue
        start, end = (vertices[face], vertices[face+1]) if left else (vertices[face+1], vertices[face])
        face_length = scalar(np.linalg.norm((end-start).astype(float, copy=False)))
        if native_math:
            sn, cs = map(scalar, sincosf(atan2f(end[1]-start[1], end[0]-start[0])))
        else:
            angle = atan2(end[1]-start[1], end[0]-start[0]); cs, sn = cos(angle), sin(angle)
        istart, jstart = ceil(start[0]/dx)-1, ceil(start[1]/dy)-1
        active = True; uref = vref = scalar(0.)
        for k in range(frame.k_end-1, frame.k_start-1, -1):
            if not active:
                continue
            for j in range(max(0, jstart-1), min(ny-2, jstart+1)+1):
                for i in range(max(0, istart-1), min(nx-2, istart+1)+1):
                    flag = flags[k, j, i]
                    if flag in (0, 7):
                        uref, vref = scalar(a['u0'][k, j, i]), scalar(a['v0'][k, j, i])
                    elif flag not in (1, 3, 11):
                        active = False
            if not active:
                continue
            for xid in range(1, ceil(float(max(face_length, rcx))/half_cell)+1):
                xp = scalar(xid*half_cell)
                shell_width = ypref*scalar(sqrt(xp))
                for yid in range(1, ceil(float(shell_width)/half_cell)+3):
                    yp = scalar(yid*half_cell if left else -yid*half_cell)
                    # The native ray coordinates are float variables. Preserve
                    # their precision before ceil: a double-only ray just above
                    # an integer grid line can incorrectly enter the next cell.
                    xx, yy = scalar(np.float32(start[0]+xp*cs-yp*sn)), scalar(np.float32(start[1]+xp*sn+yp*cs))
                    i, j = ceil(xx/dx)-1, ceil(yy/dy)-1
                    cell = i+j*(nx-1)+k*(nx-1)*(ny-1)
                    velocity = i+j*nx+k*nx*ny
                    # Native cell and face strides differ. A ray across an
                    # axis boundary can still address both allocated buffers.
                    if native_math:
                        if not (0 <= cell < flags.size and 0 <= velocity < a['u0'].size):
                            continue
                    elif not (0 <= i < nx-1 and 0 <= j < ny-1):
                        continue
                    if flags.flat[cell] in (0, 2):
                        continue
                    for field, ox, oy, ref in [('u0', 0., .5, uref), ('v0', .5, 0., vref)]:
                        xoffset = scalar((i+ox)*float(dx))-start[0]
                        yoffset = scalar((j+oy)*float(dy))-start[1]
                        xu, yu = xoffset*cs+yoffset*sn, -xoffset*sn+yoffset*cs
                        part = scalar(1-((half_rcx-float(xu))/half_rcx)**2)
                        shell_width = vd*scalar(sqrt(part)) if part > 0 else scalar(0.)
                        internal = ypref*scalar(sqrt(xu)) if xu >= 0 else float('nan')
                        if abs(yu) <= shell_width:
                            a[field].flat[velocity] = -ref*abs((shell_width-abs(yu))/vd)
                        elif abs(yu) <= internal:
                            near = scalar(log((abs(yu)+roughness)/roughness))
                            far = scalar(log((internal+roughness)/roughness))
                            a[field].flat[velocity] = ref*near/far
                    xx = scalar((i+.5)*float(dx))-start[0]
                    yy = scalar((j+.5)*float(dy))-start[1]
                    xc, yc = xx*cs+yy*sn, -xx*sn+yy*cs
                    internal = ypref*scalar(sqrt(xc)) if xc >= 0 else float('nan')
                    # Upstream computes center shell_width_calc but does not
                    # update shell_width: it still contains the v-face value.
                    if (abs(yc) <= shell_width or abs(yc) <= internal) and flags.flat[cell] not in (7, 8):
                        flags.flat[cell] = 9
    return frame
