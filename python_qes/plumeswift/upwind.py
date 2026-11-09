"""Original QES Rockle/MVP/HMVP upwind cavity (UpwindCavity.cpp, GPL-3.0).

This component updates staggered initial winds in-place. It is independent of
the wake, sidewall, rooftop and street-canyon components, in native call order.
"""
import numpy as np
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f, sincosf


def _round(value):
    value = float(value)
    return int(np.copysign(np.floor(abs(value)+.5), value))


def upwind_cavity(workspace, building, mode=2, *, native_dy_alias=True, native_math=False):
    if mode == 0:
        return
    if mode not in (1, 2, 3):
        raise ValueError("Upwind flag must be 0, 1, 2 or 3")
    a, c = workspace.arrays, workspace.coordinates
    scalar = np.float32 if native_math else float
    vertices = np.asarray(building.vertices, dtype=scalar)
    dx = scalar(c['x'][1]-c['x'][0]); actual_dy = scalar(c['y'][1]-c['y'][0])
    dy = dx if native_dy_alias else actual_dy; dxy = min(dx, actual_dy)
    nz, ny, nx = a['u0'].shape
    centroid = np.add.accumulate(vertices[:-1], axis=0)[-1]/scalar(len(vertices)-1) if native_math else vertices[:-1].mean(axis=0)
    ic = _round(centroid[0]/dx)-1; jc = _round(centroid[1]/actual_dy)-1
    ic, jc = min(max(ic, 0), nx-1), min(max(jc, 0), ny-1)
    z_center, z_face = native_vertical_coordinates(a['dz_array']) if native_math else (c['z'], c['z_face'])
    building_height, base = map(scalar, (building.height, building.base_height))
    starts = np.flatnonzero(base <= z_face[2:])
    ends = np.flatnonzero(base+building_height < z_center[1:])
    if not len(starts) or not len(ends):
        raise ValueError("Building outside vertical domain")
    k_start, k_end = 1+starts[0], 1+ends[0]
    uh, vh = a['u0'][k_end, jc, ic], a['v0'][k_end, jc, ic]
    direction = np.arctan2(vh, uh)
    if native_math:
        sn, cs = map(scalar, sincosf(direction))
    else:
        cs, sn = np.cos(direction), np.sin(direction)
    height_factor, length_factor = scalar(.6), scalar(.4)
    tol = float(scalar(np.deg2rad(10)))
    margin = .1*float(dxy)
    for first, second in zip(vertices[:-1], vertices[1:]):
        half = .5*(first-second).astype(float, copy=False)
        xf1 = scalar(half[0]*float(cs)+half[1]*float(sn)); yf1 = scalar(-half[0]*float(sn)+half[1]*float(cs))
        xf2, yf2 = -xf1, -yf1
        angle = atan2f if native_math else np.arctan2
        relative = float(scalar(float(angle(yf2-yf1, xf2-xf1))+.5*np.pi))
        if relative > np.pi+.0001:
            relative = float(scalar(relative-2*np.pi))
        if abs(relative) <= np.pi-tol:
            continue
        perpendicular = float(scalar(float(angle(second[1]-first[1], second[0]-first[0]))+.5*np.pi))
        if perpendicular <= -np.pi:
            perpendicular = float(scalar(perpendicular+2*np.pi))
        gamma = 0.
        if perpendicular >= .75*np.pi:
            gamma = perpendicular-np.pi
        elif perpendicular >= .25*np.pi:
            gamma = perpendicular-.5*np.pi
        elif perpendicular < -.75*np.pi:
            gamma = perpendicular+np.pi
        elif perpendicular < -.25*np.pi:
            gamma = perpendicular+.5*np.pi
        if native_math:
            sg, cg = map(scalar, sincosf(scalar(gamma)))
            cos_rel = scalar(sincosf(relative)[1])
        else:
            cg, sg, cos_rel = np.cos(gamma), np.sin(gamma), np.cos(relative)
        uhr = scalar(uh*float(cg)+vh*float(sg))
        vhr = scalar(-uh*float(sg)+vh*float(cg))
        face_length = scalar(np.hypot(float(xf2-xf1), float(yf2-yf1)))
        numerator = scalar(2. if mode == 1 else 1.5)*face_length*cos_rel
        lf = scalar(abs(float(numerator)/(1+.8*float(face_length)/float(building_height))))
        if lf == 0 or yf2 == 0:
            raise ValueError("Degenerate upwind face")
        vortex_height = min(face_length, building_height) if mode == 3 else building_height
        vortex_top, retard_top = height_factor*vortex_height, height_factor*building_height
        top_candidates = np.flatnonzero(retard_top+base <= z_center[k_start:])
        k_top = k_start+top_candidates[0]+1 if len(top_candidates) else len(z_center)
        x_margin, y_margin = (_round(1.5*float(lf)/float(spacing)) for spacing in (dx, dy))
        imin = max(_round(min(first[0], second[0])/dx)-x_margin-1, 1)
        imax = min(_round(max(first[0], second[0])/dx)+x_margin, nx-2)
        jmin = max(_round(min(first[1], second[1])/dy)-y_margin-1, 1)
        jmax = min(_round(max(first[1], second[1])/dy)+y_margin, ny-2)
        if imin >= imax or jmin >= jmax:
            continue
        k, j, i = np.meshgrid(np.arange(k_start, k_top), np.arange(jmin, jmax), np.arange(imin, imax), indexing='ij')
        index = (k.ravel(), j.ravel(), i.ravel())
        k, j, i = index
        height = z_center[k]-base
        average = .5*(first+second)
        fluid = ~np.isin(a['icellflag'][index], [0, 2])
        for component, xoffset, yoffset in [('u0', 0., .5), ('v0', .5, 0.), ('w0', .5, .5)]:
            xx = (i+.5)*float(dx)-float(average[0]) if xoffset else i.astype(scalar)*dx-average[0]
            yy = (j+.5)*float(dy)-float(average[1]) if yoffset else j.astype(scalar)*dy-average[1]
            xr = (xx*cs+yy*sn).astype(scalar); yr = (-xx*sn+yy*cs).astype(scalar)
            active = (abs(yr) <= abs(yf2)) & (vortex_top > height) & fluid
            if not active.any():
                continue
            sel = tuple(v[active] for v in index)
            y, z, x = yr[active], height[active], xr[active]
            intersect = (xf2-xf1)/(yf2-yf1)*(y-yf1)+xf1
            lateral = 1-(y/abs(yf2)).astype(float)**2
            ellipse = (-float(lf)*np.sqrt(np.maximum(lateral*(1-(z/vortex_top).astype(float)**2), 0))).astype(scalar)
            distance = x-intersect
            in_front = distance.astype(float, copy=False) <= margin
            values = a[component][sel].astype(np.float32 if native_math else a[component].dtype, copy=True)
            touched = np.zeros(values.shape, dtype=bool)
            if mode == 1:
                touched = (distance >= ellipse) & in_front
                values[touched] = 0.
            elif component == 'w0':
                outer = (distance >= ellipse) & (distance < length_factor*ellipse)
                inner = (distance >= length_factor*ellipse) & in_front
                values[outer] *= length_factor
                along = np.pi*abs(distance[inner]).astype(float)/float(length_factor*lf)
                values[inner] = -np.hypot(uh, vh)*(.1*np.cos(along)-.05)
                touched = outer | inner
            else:
                xrz = (-float(lf)*np.sqrt(np.maximum(lateral*(1-(z/retard_top).astype(float)**2), 0))).astype(scalar)
                rz_end = length_factor*ellipse
                outer = (distance >= xrz) & (distance < rz_end)
                if mode == 3:
                    values[outer] = values[outer]*((distance[outer]-xrz[outer]).astype(float)*(float(length_factor)-1)
                                                   /(rz_end[outer]-xrz[outer])+1)
                else:
                    values[outer] *= length_factor
                inner = (distance >= rz_end) & in_front
                if component == 'u0':
                    ur, vr = values[inner]*cg, values[inner]*sg
                else:
                    ur, vr = values[inner]*sg, values[inner]*cg
                vertical = -float(height_factor)*np.cos(np.pi*z[inner].astype(float)/(.5*float(vortex_height)))+.05
                along = -float(height_factor)*np.sin(np.pi*abs(distance[inner]).astype(float)/float(length_factor*lf))
                if .25*np.pi <= abs(perpendicular) <= .75*np.pi:
                    vr = (-float(vhr)*vertical*along).astype(scalar)
                else:
                    ur = (-float(uhr)*vertical*along).astype(scalar)
                values[inner] = ur*cg-vr*sg if component == 'u0' else ur*sg+vr*cg
            a[component][sel] = values
            if component == 'w0':
                flag = a['icellflag'][sel]
                flag[touched & ~np.isin(flag, [7, 8]) & (sel[0] < nz-2)] = 3
                a['icellflag'][sel] = flag
