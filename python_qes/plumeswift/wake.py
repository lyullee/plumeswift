"""Literal QES polygon cavity/far-wake model (PolygonWake.cpp, GPL-3.0).

Sequential ray traversal and deferred far-wake writes preserve the upstream
overlap semantics. High-rise and rooftop-dependent wake-height equations are
included; street-canyon and rooftop velocity models remain separate components.
"""
from dataclasses import dataclass
from math import sin, cos, sqrt, ceil, pi
import numpy as np
from .building_frame import BuildingFrame, face_angles
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import sincosf
from .numerics.linux_float_power import powf_three_halves
from .upwind import _round


@dataclass
class WakeState:
    frame: BuildingFrame
    length: float
    height: float
    rooftop: bool
    tall: bool
    length_over_height: float
    width_over_height: float
    mixing_frame: BuildingFrame | None = None
    mixing_angles: np.ndarray | None = None


def face_wake_lengths(frame, relative, length, *, native_math=False):
    """Project each wake length onto its face, then interpolate to vertices."""
    count = len(frame.vertices)-1
    scalar = np.float32 if native_math else float
    yi = frame.local[:, 1]
    if native_math:
        yi = yi.astype(scalar, copy=False)
    face_lengths = np.full(count, -1., dtype=scalar)
    eligible = abs(relative.astype(float, copy=False)) < .5*pi
    cosine = sincosf(relative[eligible])[1] if native_math else np.cos(relative[eligible])
    face_lengths[eligible] = length*cosine
    node_lengths = np.zeros(count+1, dtype=scalar)
    average = total = scalar(0.); stop = count-1
    for face in range(count):
        if face_lengths[face] > 0:
            previous, next_ = (face+count-1) % count, (face+1) % count
            lp, ln = face_lengths[previous], face_lengths[next_]
            lf = face_lengths[face]
            node_lengths[face] = lf if lp < 0 else ((yi[face]-yi[next_])*lf+(yi[previous]-yi[face])*lp)/(yi[previous]-yi[next_])
            node_lengths[face+1] = lf if ln < 0 else ((yi[next_]-yi[next_+1])*ln+(yi[face]-yi[next_])*lf)/(yi[face]-yi[next_+1])
            average += lf*(yi[face]-yi[next_]); total += yi[face]-yi[next_]
        if np.all(abs(frame.vertices[face+1]-frame.vertices[0]) < .1):
            stop = face; break
    if total == 0:
        if native_math and not eligible[:stop+1].any():
            # A repeated GIS start vertex can close the native face scan
            # before any downwind face. The native wake traversal is empty.
            return node_lengths, scalar(0.), stop, eligible
        raise ValueError('No eligible downwind polygon face')
    return node_lengths, average/total, stop, eligible


def polygon_wake(workspace, building, building_id=0, *, mode=2, rooftop_mode=2, high_rise=False, native_dy_alias=True, native_math=False):
    if mode < 1:
        raise ValueError('Wake mode must be positive')
    a, c = workspace.arrays, workspace.coordinates
    frame = BuildingFrame.create(workspace, building, native_math=native_math)
    scalar = np.float32 if native_math else float
    dx = scalar(c['x'][1]-c['x'][0]); dy_actual = scalar(c['y'][1]-c['y'][0])
    dy = dx if native_dy_alias else dy_actual; dxy = min(dx, dy_actual)
    nz, ny, nx = a['u0'].shape
    z = native_vertical_coordinates(a['dz_array'])[0] if native_math else c['z']
    if native_math:
        sn, cs = map(scalar, sincosf(frame.direction))
    else:
        cs, sn = cos(frame.direction), sin(frame.direction)
    cx, cy = frame.center; xi, yi = frame.local.T
    uh, vh = frame.reference
    relative = face_angles(frame.local, native_math=native_math, wrap_slack=.0001)
    perpendicular = abs(relative) < scalar(np.deg2rad(.01))
    building_height, base = scalar(building.height), scalar(building.base_height)
    effective_height = building_height+base
    height = effective_height
    tall = False
    l_over_h = w_over_h = 0.

    def wake_length(wake_height):
        nonlocal tall, l_over_h, w_over_h
        l_over_h = min(max(scalar(frame.length)/wake_height, scalar(.3)), scalar(3.))
        w_over_h = min(scalar(frame.width)/wake_height, scalar(10.))
        if wake_height/scalar(frame.length) >= 2 and high_rise:
            tall = True
        h, l, w = map(float, (wake_height, l_over_h, w_over_h))
        if tall:
            return scalar(2.34*h*w/(l**-1.37*(-.27+1.86*w)))
        return scalar(1.8*h*w/(l**.3*(1+.24*w)))

    length = wake_length(height)
    roof = False
    flags, ibuild = a['icellflag'], a['ibuilding_flag']
    if frame.rectangular and rooftop_mode == 2:
        eligible = frame.rooftop_faces(wrap_slack=.0001, native_math=native_math)
        if len(eligible):
            face = eligible[-1]; upstream = 0
            for yid in range(2*ceil(abs(yi[face]-yi[face+1])/dxy)+1):
                yc = scalar(float(min(yi[face], yi[face+1]))+.5*yid*float(dxy))
                wall = xi[face] if perpendicular[face] else (xi[face+1]-xi[face])/(yi[face+1]-yi[face])*(yc-yi[face])+xi[face]
                for xid in range(ceil(length/dxy)+1, 0, -1):
                    xc = -xid*dxy
                    # ceil before division is deliberate in this branch.
                    i = int(ceil((xc+wall)*cs-yc*sn+cx)/dx-1)
                    j = int(ceil((xc+wall)*sn+yc*cs+cy)/dy-1)
                    if 0 < i < nx-2 and 0 < j < ny-2 and flags[frame.k_end-1, j, i] == 0:
                        upstream += 1; break
            roof = upstream < ceil(abs(yi[face]-yi[face+1])/dxy)
            if roof:
                scale = scalar(min(frame.width, float(effective_height))**(2/3)*max(frame.width, float(effective_height))**(1/3))
                rcx, vd = scalar(.9*float(scale)), scalar(.5*.22*float(scale))
                hd = min(frame.width, frame.length)
                if hd < rcx:
                    half = .5*float(rcx)
                    shell = scalar(float(vd)*sqrt(1-((half-hd)/half)**2))
                    if shell > 0:
                        height += shell
                length = wake_length(height)
    zs = xsep = 0.; ks = 0
    if tall:
        h, l, w = map(float, (height, l_over_h, w_over_h))
        zs = scalar((-1.47*l-.69*w+1.25)*h)
        xsep = scalar(4.34*h*w/(l**-.16*(2.99-.81*w)))
        height = scalar((1.4077*l-1.5705*w+.9297)*h)
        ks = min(int(np.searchsorted(z, zs)), nz-2)
        if native_math:
            ks = max(ks, frame.k_start)
    node_lengths, length, stop, eligible = face_wake_lengths(frame, relative, length, native_math=native_math)
    kbottom = next((k for k in range(1, frame.k_start+1) if base <= z[k]), frame.k_start)
    ktop = next((k for k in range(frame.k_start, nz-2) if height < z[k+1]), nz-3)
    kk = next((k for k in range(frame.k_start, frame.k_end) if .75*float(building_height)+float(base) <= float(z[k])), frame.k_end-1)
    cavity_factor, wake_factor = map(scalar, (1.1, .1) if mode > 1 else (1., 0.))
    delayed = {'u0': [], 'v0': []}
    for k in range(ktop, kbottom-1, -1):
        zbuild = z[k]-base
        for face in range(stop+1):
            if not eligible[face]:
                continue
            denom = yi[face+1]-yi[face]
            def local_at(y):
                lr = node_lengths[face]+(y-yi[face])*(node_lengths[face+1]-node_lengths[face])/denom
                wall = xi[face] if perpendicular[face] else (xi[face+1]-xi[face])/denom*(y-yi[face])+xi[face]
                return lr, wall
            for yid in range(2*ceil(abs(denom)/dxy)+1):
                yc = scalar(float(yi[face])-.5*yid*float(dxy))
                lr, wall = local_at(yc)
                ynorm = max(0., yi.max()) if yc >= 0 else min(0., yi.min())
                canyon = scalar(1.); first_fluid = -1
                for xid in range(1, ceil(lr/dxy)+1):
                    xc = xid*dxy
                    i, j = int(((xc+wall)*cs-yc*sn+cx)/dx), int(((xc+wall)*sn+yc*cs+cy)/dy)
                    if not (0 < i < nx-2 and 0 < j < ny-2):
                        break
                    solid = flags[kk, j, i] in (0, 2)
                    if not solid and first_fluid < 0:
                        first_fluid = xid
                    if solid and first_fluid > 0:
                        canyon = xc/length; break
                first_fluid = -1
                for xid in range(1, 2*ceil(3*lr/dxy)+1):
                    xc = scalar(.5*xid*float(dxy))
                    gx = ((xc+wall)*cs-yc*sn+cx)/dx; gy = ((xc+wall)*sn+yc*cs+cy)/dy
                    i, j = int(gx), int(gy)
                    if not (0 < i < nx-2 and 0 < j < ny-2):
                        break
                    solid = flags[k, j, i] in (0, 2)
                    if not solid and first_fluid < 0:
                        first_fluid = xid
                    if solid:
                        if first_fluid >= 0:
                            if ibuild[k, j, i] == building_id:
                                first_fluid = -1
                            elif canyon < 1 or flags[kk, j, i] in (0, 2):
                                break
                        continue
                    wake_flags = [True]*3
                    for component, field, ii, jj in [(0, 'u0', _round(gx), int(gy)), (1, 'v0', int(gx), _round(gy)), (2, 'w0', ceil(gx)-1, ceil(gy)-1)]:
                        limitx, limity = (nx-2, ny-2) if component == 2 else (nx-1, ny-1)
                        if not (0 < ii < limitx and 0 < jj < limity):
                            continue
                        xp = ii*dx-cx if component == 0 else scalar((ii+.5)*float(dx)-float(cx))
                        yp = jj*dy-cy if component == 1 else scalar((jj+.5)*float(dy)-float(cy))
                        xr, yr = xp*cs+yp*sn, -xp*sn+yp*cs
                        local_lr, local_wall = local_at(yr); xr -= local_wall
                        dn = scalar(0.)
                        if abs(yr) < abs(ynorm) and abs(ynorm) > 1e-9 and zbuild < height and height > 1e-9:
                            lateral = 1-float(yr/ynorm)**2
                            if not tall:
                                dn = scalar(sqrt(lateral*(1-float(zbuild/height)**2)*float(canyon*local_lr)**2))
                            else:
                                upper = (zbuild > zs if component == 0 else zbuild >= zs) if component != 2 else k >= ks
                                if upper:
                                    dn = scalar(sqrt(lateral*(1-float((zbuild-zs)/(height-zs))**2)*float(canyon*local_lr)**2))
                                elif component == 2:
                                    dn = scalar(sqrt(lateral*(1-float((zbuild-zs)/zs)**2)*float(canyon*(local_lr-xsep))**2)+float(xsep))
                                else:
                                    dn = scalar(sqrt(lateral)*(sqrt(1-float((zbuild-zs)/zs)**2)*float(canyon*(local_lr-xsep))+float(xsep)))
                        if xr > 3*dn:
                            wake_flags[component] = False
                        index = (k, jj, ii)
                        if dn <= 0 or not wake_flags[component] or not yi[face+1] <= yr <= yi[face] or flags[index] in (0, 2):
                            if component == 2 and not any(wake_flags):
                                break
                            continue
                        if component == 2:
                            if flags[index] not in (7, 8):
                                if xr <= dn:
                                    flags[index] = 4
                                elif canyon == 1:
                                    flags[index] = 5
                        elif xr > dn:
                            if canyon == 1:
                                ratio = dn/(xr+wake_factor*dn)
                                power = powf_three_halves(ratio) if native_math else float(ratio)**1.5
                                value = scalar(float(scalar(a[field][index]))*(1-float(power)))
                                delayed[field].append((index, value)); a['w0'][index] = 0.
                        else:
                            ref = uh if component == 0 else vh
                            a[field][index] = scalar(-ref*min((1-float(xr/(cavity_factor*dn)))**2, 1)*min(sqrt(1-abs(float(yr/ynorm))), 1))
                            if not tall:
                                a['w0'][index] = 0.
                            else:
                                factor = (-1 if xr <= dn/2 else 1) if zbuild >= zs else -2
                                value = scalar(factor*float(abs(zbuild-zs)/height)*a[field][index])
                                if component == 0:
                                    a['w0'][index] = value
                                else:
                                    a['w0'][index] = min(value, a['w0'][index]) if factor == 1 else max(value, a['w0'][index])
                    else:
                        continue
                    break
    for field, updates in delayed.items():
        for index, value in updates:
            a[field][index] = value
    return WakeState(frame, float(length), float(height), bool(roof), bool(tall), l_over_h, w_over_h)
