"""QES building wake turbulence (NonLocalMixing.cpp, GPL-3)."""
from math import ceil, cos, floor, sin, sqrt

import numpy as np

from .building_frame import face_angles
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import sincosf
from .upwind import _round
from .wake import face_wake_lengths


def apply_nonlocal_mixing(workspace, cell_velocity, stress, mixing, energy, dissipation, turbulence_flags,
                          *, native_math=False):
    context = workspace.building_context
    if context is None:
        return
    for number, building in context.ordered_buildings:
        wake = context.wakes.get(number)
        if wake is not None:
            _building_mixing(workspace, cell_velocity, number, building, wake,
                             stress, mixing, energy, dissipation, turbulence_flags, native_math=native_math)


def _building_mixing(workspace, velocity, number, building, wake, tau, lm, tke, eps, turbulence_flags,
                     *, native_math):
    a, c = workspace.arrays, workspace.coordinates
    flags, owners = a['icellflag'], a['ibuilding_flag']
    nz, ny, nx = a['u0'].shape
    scalar = np.float32 if native_math else float
    dx = scalar(c['x'][1]-c['x'][0])
    dy_actual = scalar(c['y'][1]-c['y'][0])
    dxy = min(dx, dy_actual)
    dy = dx  # Domain::getDomainSize aliases dy to dx.
    z = native_vertical_coordinates(a['dz_array'])[0] if native_math else c['z']
    # Later closures overwrite the native building's rotation and vertices.
    # Effective width and lateral bounds still belong to the wake stage.
    frame = wake.mixing_frame if native_math and wake.mixing_frame is not None else wake.frame
    if not (1 <= frame.k_start <= nz-2 and 1 <= frame.k_end <= nz-2):
        return
    local_vertices = frame.local.astype(scalar, copy=False)
    xi, yi = local_vertices.T
    if native_math:
        sn, cs = map(scalar, sincosf(frame.direction))
    else:
        cs, sn = cos(frame.direction), sin(frame.direction)
    cx, cy = map(scalar, frame.center)
    building_height, base = map(scalar, (building.height, building.base_height))
    height = building_height+base
    width = scalar(wake.frame.width)
    w_over_h, l_over_h = (float(scalar(ratio)) for ratio in (wake.width_over_height, wake.length_over_height))
    length = scalar(1.8*float(height)*w_over_h/(l_over_h**.3*(1+.24*w_over_h)))
    if native_math and wake.mixing_angles is not None:
        relative = wake.mixing_angles.astype(scalar, copy=False)
    else:
        relative = face_angles(wake.frame.local.astype(scalar, copy=False), native_math=native_math, wrap_slack=.0001)
    nodes, length, stop, eligible = face_wake_lengths(frame, relative, length, native_math=native_math)

    kbottom = next((k for k in range(1, frame.k_start+1) if base <= z[k]), frame.k_start)
    ktop = next((k for k in range(frame.k_start, nz-2) if height < z[k+1]), nz-3)
    kk = next((k for k in range(frame.k_start, frame.k_end) if .75*float(building_height)+float(base) <= float(z[k])),
              frame.k_end-1)

    def grid(x, y):
        return (x*cs-y*sn+cx)/dx, (x*sn+y*cs+cy)/dy

    def projected(k, j, i):
        u, v = map(scalar, velocity[k, j, i, :2])
        return u*cs+v*sn

    # The last allocated reference vertex is zero in the native routine.
    lateral_points = local_vertices.copy()
    lateral_points[-1] = 0.
    refs = []
    for point, outward in ((np.argmin(lateral_points[:, 1]), -1.), (np.argmax(lateral_points[:, 1]), 1.)):
        x, y = lateral_points[point]
        xp = scalar(float(x)+(.5 if x < 0 else -.5)*float(dxy))
        if native_math:
            y = scalar(float(y)-outward*.5*float(dxy))
            yp = scalar(float(y)+outward*3.*float(dxy))
        else:
            yp = y+outward*2.5*dxy
        gx, gy = grid(xp, yp)
        refs.append((floor(gy), floor(gx)))
    # The native guard is a contradictory conjunction. Preserve the local
    # closure when a reference would require an out-of-grid read.
    if any(not (0 <= i < nx-1 and 0 <= j < ny-1) for j, i in refs):
        return
    ic, jc = _round(cx/dx)-1, _round(cy/(dy_actual if native_math else dy))-1
    if not (0 <= ic < nx-1 and 0 <= jc < ny-1):
        return
    von_karman, farwake_factor = scalar(.4), scalar(3.)
    vertical_friction = von_karman*projected(ktop+1, jc, ic)
    wake_y = wake.frame.local[:-1, 1].astype(scalar, copy=False)
    low, high = min(scalar(0.), wake_y.min()), max(scalar(0.), wake_y.max())
    for k in range(ktop, kbottom-1, -1):
        zbuild = z[k]-base
        right, left = (projected(k, j, i) for j, i in refs)
        vertical_factor = max(0., 1-float(zbuild/height)**2)
        for face in range(stop+1):
            if not eligible[face]:
                continue
            dy_face = yi[face+1]-yi[face]
            if dy_face == 0:
                continue
            perpendicular = abs(relative[face]) < scalar(np.deg2rad(.01))
            node_delta = nodes[face+1]-nodes[face]
            length_slope = node_delta/dy_face
            wall_slope = (xi[face+1]-xi[face])/dy_face

            def local(y):
                offset = y-yi[face]
                delta = offset*node_delta/dy_face if native_math else offset*length_slope
                wall = xi[face] if perpendicular else wall_slope*offset+xi[face]
                return nodes[face]+delta, wall

            for yid in range(2*ceil(abs(dy_face)/dxy)+1):
                yc = scalar(float(yi[face])-.5*yid*float(dxy))
                lr, wall = local(yc)
                ynorm = high if yc >= 0 else low
                canyon, first_fluid = scalar(1.), False
                for xid in range(1, ceil(lr/dxy)+1):
                    xc = scalar(xid)*dxy
                    gx, gy = grid(xc+wall, yc)
                    i, j = int(gx), int(gy)
                    if not (0 <= i < nx-1 and 0 <= j < ny-1):
                        break
                    solid = flags[kk, j, i] in (0, 2)
                    if solid and first_fluid:
                        canyon = xc/length
                        break
                    first_fluid |= not solid
                first_fluid = False
                for xid in range(1, 2*ceil(farwake_factor*lr/dxy)+1):
                    xc = scalar(.5*xid*float(dxy))
                    gx, gy = grid(xc+wall, yc)
                    i, j = int(gx), int(gy)
                    if not (0 < i < nx-2 and 0 < j < ny-2):
                        break
                    if flags[k, j, i] in (0, 2):
                        if first_fluid:
                            if owners[k, j, i] == number:
                                first_fluid = False
                            elif canyon < 1 or flags[kk, j, i] in (0, 2):
                                break
                        continue
                    first_fluid = True
                    wake_flags = [True, True, True]
                    for component, (ii, jj) in enumerate(((_round(gx), j), (i, _round(gy)),
                                                         (ceil(gx)-1, ceil(gy)-1))):
                        shell = 2 if component == 2 else 1
                        if not (0 < ii < nx-shell and 0 < jj < ny-shell):
                            continue
                        xp = scalar(ii)*dx-cx if component == 0 else scalar((ii+.5)*float(dx)-float(cx))
                        yp = scalar(jj)*dy-cy if component == 1 else scalar((jj+.5)*float(dy)-float(cy))
                        xr, yr = xp*cs+yp*sn, -xp*sn+yp*cs
                        local_length, local_wall = local(yr)
                        xr -= local_wall
                        dn = scalar(0.)
                        if abs(yr) < abs(ynorm) and abs(ynorm) > scalar(1e-9) and zbuild < height and height > scalar(1e-9):
                            shape = (1-float(yr/ynorm)**2)*vertical_factor
                            span = float(canyon*local_length)
                            dn = scalar(sqrt(shape*span**2)) if native_math else abs(span)*sqrt(shape)
                        wake_flags[component] = xr <= farwake_factor*dn
                        if component != 2:
                            continue
                        index = (k, jj, ii)
                        if (dn > 0 and wake_flags[2] and yi[face+1] <= yr <= yi[face]
                                and flags[index] not in (0, 2)):
                            if xr <= dn or canyon == 1:
                                ci, cj = ceil((cs*xr+cx)/dx)-1, ceil((sn*xr+cy)/dy)-1
                                center = projected(k, cj, ci) if 0 < ci < nx-2 and 0 < cj < ny-2 else scalar(.9*float(left))
                                horizontal_defect = max(abs(left-center), abs(right-center))
                                horizontal = (xr > dn or float(horizontal_defect)/(.5*float(width))
                                              >= float(projected(*index))/(.75*float(building_height)))
                                friction = von_karman*horizontal_defect if horizontal else vertical_friction
                                _write_stress(index, friction, yr, cs, sn, width, building_height, horizontal,
                                              tau, lm, tke, eps, turbulence_flags, native_math=native_math)
                        if not any(wake_flags):
                            break
                    else:
                        continue
                    break


def _write_stress(index, friction, y, cs, sn, width, height, horizontal, tau, lm, tke, eps, flags,
                  *, native_math):
    u2 = friction*friction
    xy = abs(float(y))/(.5*float(width))*float(u2) if horizontal else 0.
    xz, yz = (0., 0.) if horizontal else (-u2*cs, -u2*sn)
    uu = np.float32(4)*u2 if native_math else 4*u2
    ww = np.float32(1.3)*np.float32(1.3)*u2 if native_math else 1.69*u2
    tau[index] = ((uu, xy, xz), (xy, uu, yz), (xz, yz, ww))
    lm[index] = width if horizontal else .75*float(height)
    flags[index] = 11 if horizontal else 12
    tke[index] = .5*float(uu+uu+ww) if native_math else .5*(4+4+1.69)*u2
    eps[index] = 5.7*float(friction)**3/float(lm[index])
