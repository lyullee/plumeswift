"""Original QES legacy street-canyon closure (StreetCanyon.cpp, GPL-3.0).

Native undefined out-of-domain reads are guarded. Empirical coefficients,
velocity projections, reference selection and sequential overwrites are retained.
Both native closures share the geometric search. Their constitutive branches
remain explicit: legacy mode 1 and wedge-mixing modified mode 2.
"""
from functools import lru_cache
from math import atan2, sin, cos, hypot, ceil, log, pi
import numpy as np
from .building_frame import BuildingFrame, face_angles
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f, sincosf
from .upwind import _round


def street_canyon(workspace, building, wake_state, buildings_by_id, max_velocity, *, roughness=.01, native_dy_alias=True, mode=1,
                  native_math=False):
    if mode not in (1, 2):
        raise ValueError('Street canyon mode must be 1 or 2')
    a, c = workspace.arrays, workspace.coordinates
    scalar = np.float32 if native_math else float
    one, two = scalar(1.), scalar(2.)
    f = BuildingFrame.create(workspace, building, native_math=native_math)
    dx = scalar(c['x'][1]-c['x'][0]); actual_dy = scalar(c['y'][1]-c['y'][0])
    dy = dx if native_dy_alias else actual_dy; dxy = min(dx, actual_dy)
    nz, ny, nx = a['u0'].shape
    z, z_face = native_vertical_coordinates(a['dz_array']) if native_math else (c['z'], c['z_face'])
    flags, ibuild = a['icellflag'], a['ibuilding_flag']
    roughness = scalar(roughness)
    dz = float(np.min(a['dz_array']))
    half_cell = .5*float(dxy)
    tolerance = float(scalar(np.deg2rad(.01)))
    angle = atan2f if native_math else atan2

    @lru_cache(maxsize=128)
    def components(direction):
        if native_math:
            sine, cosine = sincosf(direction)
            return scalar(cosine), scalar(sine)
        return cos(direction), sin(direction)

    xi, yi = f.local.T; cs, sn = components(f.direction); cx, cy = f.center
    relative = face_angles(f.local, native_math=native_math).astype(float, copy=False)
    if wake_state is None:
        # The original search uses Lr from the wake stage. With no defined
        # search length, keep the local geometry and leave the winds intact.
        return f, relative
    angle_limit = components(scalar(.75*pi))[0]
    cross_direction = along_direction = normal_direction = scalar(0.)
    search_shell = 0 if native_math and mode == 1 else 1
    cell_row, cell_plane = nx-1, (nx-1)*(ny-1)

    def coordinates(x, y, subtract=None):
        gx, gy = x*cs-y*sn+cx, x*sn+y*cs+cy
        if subtract is None:
            gx, gy = gx/dx, gy/dy
        else:
            gx, gy = (float(gx)-subtract)/float(dx), (float(gy)-subtract)/float(dy)
        return ceil(gx)-1, ceil(gy)-1
    def valid(i, j, k, shell=0):
        return shell <= i < nx-1-shell and shell <= j < ny-1-shell and 0 <= k < nz-1
    for face in range(len(relative)):
        if abs(relative[face]) >= .5*pi-.0001:
            continue
        perpendicular = abs(relative[face]) > pi-tolerance or abs(relative[face]) < tolerance
        first, second = f.vertices[face:face+2]
        normal = scalar(float(angle(second[1]-first[1], second[0]-first[0]))+.5*pi)
        if float(normal) > pi:
            normal = scalar(float(normal)-2*pi)
        def wall_at(y):
            return xi[face] if perpendicular else (xi[face+1]-xi[face])/(yi[face+1]-yi[face])*(y-yi[face])+xi[face]
        for yid in range(2*ceil(abs(yi[face]-yi[face+1])/dxy)+1):
            yc = scalar(float(min(yi[face], yi[face+1]))+yid*half_cell)
            wall = wall_at(yc); top = False
            kref = 0; magnitude = canyon_direction = scalar(0.)
            for k in range(f.k_end-1, f.k_start-1, -1):
                i1, j1 = coordinates(wall-dxy, yc)
                if not valid(i1, j1, k) or flags[k, j1, i1] != 0:
                    continue
                canyon = False; gap = 0.; reverse = False; first_fluid = -1; last_fluid = -1
                for xid in range(1, 2*ceil(scalar(wake_state.length)/dxy)+1):
                    xc = scalar(xid*half_cell)
                    i, j = coordinates(xc+wall, yc)
                    if not valid(i, j, k, shell=search_shell):
                        break
                    solid = flags[k, j, i] in (0, 2)
                    if not solid and first_fluid < 0:
                        first_fluid = xid
                    if not solid or first_fluid < 0:
                        continue
                    canyon = True; last_fluid = xid-1; gap = scalar((last_fluid-first_fluid)*half_cell)
                    if not top:
                        kref = k+1
                        ic, jc = coordinates(np.float64(last_fluid*half_cell+float(wall)), yc, .001)
                        if not valid(ic, jc, kref, shell=search_shell) or flags[kref, jc, ic] in (0, 2):
                            canyon = top = False; gap = 0.; break
                        center = ic+jc*cell_row+kref*cell_plane
                        uvalues, vvalues = [], []
                        for delta in [-1, 1]:
                            # The legacy reference tests neighbor cells in the
                            # packed cell buffer, including at a lateral edge.
                            x_neighbor = center+delta
                            y_neighbor = center+delta*cell_row
                            if 0 <= x_neighbor < flags.size and flags.flat[x_neighbor] not in (0, 2):
                                uvalues.append(scalar(a['u0'][kref, jc, ic+(delta > 0)]))
                            if 0 <= y_neighbor < flags.size and flags.flat[y_neighbor] not in (0, 2):
                                vvalues.append(scalar(a['v0'][kref, jc+(delta > 0), ic]))
                        if not uvalues and not vvalues:
                            canyon = top = False; gap = 0.; break
                        umean = sum(uvalues, scalar(0.))/scalar(len(uvalues)) if uvalues else scalar(0.)
                        vmean = sum(vvalues, scalar(0.))/scalar(len(vvalues)) if vvalues else scalar(0.)
                        magnitude = scalar(hypot(float(umean), float(vmean)))
                        if uvalues and vvalues:
                            canyon_direction = scalar(angle(vmean, umean))
                        elif uvalues:
                            canyon_direction = scalar(0. if umean > 0 else pi)
                        else:
                            canyon_direction = scalar(.5*pi if vmean > 0 else -.5*pi)
                        top = True
                        downstream_id = ibuild[k, j, i]
                        if (abs(gap) > 0 and downstream_id >= 0 and
                            scalar(buildings_by_id[downstream_id].height)+scalar(buildings_by_id[downstream_id].base_height) < scalar(building.height)+scalar(building.base_height) and
                            float(z_face[k+1]/gap) < .65):
                            canyon = top = False; gap = 0.; break
                        if magnitude > max_velocity:
                            canyon = top = False; gap = 0.; break
                    downstream_id = ibuild[k, j, i]
                    if downstream_id < 0:
                        continue
                    downstream = buildings_by_id[downstream_id]
                    ic, jc = coordinates(np.float64(float(xc)-half_cell+float(wall)), yc, .001)
                    if not valid(ic, jc, k):
                        break
                    if mode == 1 and ibuild[k, jc, ic] >= 0:
                        continue
                    vertices = np.asarray(downstream.vertices, dtype=scalar)
                    for first, second in zip(vertices[:-1], vertices[1:]):
                        normal_direction = scalar(float(angle(second[1]-first[1], second[0]-first[0]))+.5*pi)
                        if float(normal_direction) > pi+.001:
                            normal_direction = scalar(float(normal_direction)-2*pi)
                        if float(normal_direction) <= -pi:
                            normal_direction = scalar(float(normal_direction)+2*pi)
                        midpoint = .5*(first+second)
                        xx, yy = (ic+.5)*float(dx)-float(midpoint[0]), (jc+.5)*float(dy)-float(midpoint[1])
                        ncos, nsin = map(float, components(normal_direction))
                        xd = scalar(xx*ncos+yy*nsin)
                        yd = scalar(-xx*nsin+yy*ncos)
                        segment = scalar(np.linalg.norm((second-first).astype(float, copy=False)))
                        if abs(xd) >= dxy or float(abs(yd)) > .5*float(segment):
                            continue
                        down_relative = canyon_direction-normal_direction
                        if float(down_relative) > pi+.001:
                            down_relative = scalar(float(down_relative)-2*pi)
                            if float(abs(down_relative)) < .001:
                                down_relative = scalar(0.)
                        if float(down_relative) <= -pi:
                            down_relative = scalar(float(down_relative)+2*pi)
                            if float(abs(down_relative)) < .001:
                                down_relative = scalar(0.)
                        reverse = float(abs(down_relative)) < .5*pi
                        cross_direction = scalar(float(normal_direction)+pi) if reverse else normal_direction
                        along_direction = scalar(float(normal_direction)-.5*pi if down_relative >= 0 else float(normal_direction)+.5*pi)
                        if components(normal_direction-normal)[0] <= angle_limit:
                            break
                    if components(normal_direction-normal)[0] > angle_limit:
                        canyon = top = False; gap = 0.
                    break
                if float(cross_direction) > pi+.001:
                    cross_direction = scalar(float(cross_direction)-2*pi)
                if float(cross_direction) <= -pi:
                    cross_direction = scalar(float(cross_direction)+2*pi)
                if float(along_direction) > pi+.001:
                    along_direction = scalar(float(along_direction)-2*pi)
                if float(along_direction) <= -pi:
                    along_direction = scalar(float(along_direction)+2*pi)
                if not canyon or float(gap) <= .9*float(dxy):
                    continue
                log_here, log_ref = (scalar(log(z[level]/roughness)) for level in (k, kref))
                along = abs(magnitude*components(canyon_direction-along_direction)[0])*log_here/log_ref
                cross = abs(magnitude*components(canyon_direction-cross_direction)[0])*scalar(.3 if mode == 2 else 1.)
                along_components, cross_components = components(along_direction), components(cross_direction)
                zpos = z_face[kref]-z[k]
                width = abs(yi[face]-yi[face+1])
                for xid in range(first_fluid, last_fluid+(1 if mode == 2 else 3)):
                    xc = scalar(xid*half_cell)
                    i, j = coordinates(xc+wall, yc, .001)
                    if not valid(i, j, k):
                        raise ValueError('Street-canyon write outside domain')
                    if mode == 2 and flags[k, j, i] in (0, 2):
                        continue
                    gx = ((xc+wall)*cs-yc*sn+cx)/dx
                    gy = ((xc+wall)*sn+yc*cs+cy)/dy
                    iu, jv = _round(gx), _round(gy)
                    for component, ii, jj, ox, oy in [(0, iu, j, 0., .5), (1, i, jv, .5, 0.), (2, i, j, .5, .5)]:
                        xx = scalar((ii+.5)*float(dx)-float(cx)) if ox else scalar(ii)*dx-cx
                        yy = scalar((jj+.5)*float(dy)-float(cy)) if oy else scalar(jj)*dy-cy
                        xr, yr = xx*cs+yy*sn, -xx*sn+yy*cs
                        xpos = xr-wall_at(yr)
                        if not -half_cell < float(xpos) <= float(gap)+(.001 if mode == 2 else half_cell):
                            continue
                        if component < 2:
                            if not (0 <= ii < nx and 0 <= jj < ny):
                                raise ValueError('Street-canyon velocity write outside domain')
                            along_projected = along*along_components[component]
                            cross_projected = cross_components[component]
                            left, right = two*xpos/gap, one-xpos/gap
                            if mode == 1:
                                value = along_projected+cross*left*two*right*cross_projected
                            else:
                                # Original ordering matters: a non-real fSC can
                                # be overwritten by either wedge layer. Preserve
                                # those assignments before checking final fields.
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    half_width, wedge = .5*float(width), .2*float(xpos)
                                    fsc = scalar(np.power(1-np.divide(float(abs(yr)), half_width-wedge), .25))
                                    value = along_projected+cross*fsc*left*two*right*cross_projected
                                    if half_width-wedge <= float(abs(yr)+dxy):
                                        scale = float(cross*cross_projected*log_here/log_ref)
                                        value = float(along_projected)-scale*np.tanh(np.divide(float(abs(yr))-half_width+wedge, wedge))/np.tanh(1.)
                                    if float(zpos) <= wedge+dz:
                                        value = float(along_projected)-float(cross*cross_projected)*np.tanh(np.divide(wedge-float(zpos), wedge))/np.tanh(1.)
                            a['u0' if component == 0 else 'v0'][k, jj, ii] = scalar(value)
                        else:
                            if flags[k, jj, ii] not in (0, 2) and flags[k-1, jj, ii] not in (0, 2):
                                vertical = abs(.5*float(cross)*float(one-two*xpos/gap))*float(one-two*(gap-xpos)/gap)
                                a['w0'][k, jj, ii] = scalar(vertical if reverse else -vertical)
                                if mode == 2:
                                    flags[k, jj, ii] = 6
                            if mode == 1 and flags[k, jj, ii] not in (7, 8):
                                flags[k, jj, ii] = 6
    return f, relative
