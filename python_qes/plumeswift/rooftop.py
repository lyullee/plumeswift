"""Original QES logarithmic/vortex rooftop (Rooftop.cpp, GPL-3.0)."""
from math import sin, cos, sqrt, log
import numpy as np
from .building_frame import BuildingFrame, face_angles
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import sincosf


def rooftop(workspace, building, wake_state, *, mode=2, roughness=.01, native_dy_alias=True, native_math=False):
    if mode == 0:
        return
    if mode not in (1, 2) or roughness <= 0:
        raise ValueError('Invalid rooftop mode or roughness')
    a, c = workspace.arrays, workspace.coordinates
    frame = BuildingFrame.create(workspace, building, native_math=native_math,
                                 native_dy_alias=native_math and native_dy_alias)
    if wake_state is None:
        # Without a wake width, the native log shell has no defined extent.
        return frame, None
    method = 2 if mode == 2 and frame.rectangular and wake_state.rooftop else 1
    scalar = np.float32 if native_math else float
    dx = scalar(c['x'][1]-c['x'][0]); dy = dx if native_dy_alias else scalar(c['y'][1]-c['y'][0])
    nz = a['u0'].shape[0]
    if native_math:
        z, zf = native_vertical_coordinates(a['dz_array'])
        sn, cs = map(scalar, sincosf(frame.direction))
    else:
        z, zf = c['z'], c['z_face']
        cs, sn = cos(frame.direction), sin(frame.direction)
    height, base, roughness = map(scalar, (building.height, building.base_height, roughness))
    effective = height+base
    cx, cy = frame.center; xi, yi = frame.local.T
    xfront, yfront = min(0., xi.min()), min(0., yi.min())
    istart, iend, jstart, jend = frame.horizontal_bounds
    kref = next((k for k in range(frame.k_end, nz-1) if 1.5*float(effective) < zf[k+1]), nz-2)
    width = wake_state.frame.width
    scale = scalar(min(width, float(height))**(2/3)*max(width, float(height))**(1/3))
    rcx, vd = scalar(.99*float(scale)), scalar(.22*float(scale))
    half_rcx = .5*float(rcx)
    zref = scalar(float(vd)/sqrt(half_rcx))
    kendv = next((k for k in range(frame.k_end, kref+1) if effective+vd < z[k]), kref)
    urooftop = vrooftop = 0.
    flags = a['icellflag']
    _, ny, nx = flags.shape
    faces = [None]
    if method == 2:
        faces = frame.rooftop_faces(native_math=native_math)
    for _ in faces:
        for j in range(max(jstart, 0), min(jend-1, ny)):
            for i in range(max(istart, 0), min(iend-1, nx)):
                below = flags[frame.k_end-1, j, i]
                applicable = [False, False]
                solid_flags = (0, 7) if method == 1 else (0,)
                whole = below in solid_flags
                if whole:
                    applicable = [True, True]
                else:
                    applicable = [i > 0 and flags[frame.k_end-1, j, i-1] in solid_flags,
                                  j > 0 and flags[frame.k_end-1, j-1, i] in solid_flags]
                if flags[frame.k_end, j, i] in (0, 2, 4, 6) or not any(applicable):
                    continue
                hd, zrefs, shells, denoms, reference = [], [], [], [], []
                for component, field in enumerate(('u0', 'v0')):
                    if component == 0:
                        xx, yy = i*dx-cx, (j+.5)*float(dy)-float(cy)
                        xr, yr = scalar(float(xx*cs)+yy*float(sn)), scalar(float(-xx*sn)+yy*float(cs))
                    else:
                        xx, yy = (i+.5)*float(dx)-float(cx), j*dy-cy
                        xr, yr = scalar(xx*float(cs)+float(yy*sn)), scalar(-xx*float(sn)+float(yy*cs))
                    distance = min(abs(xr-xfront), abs(yr-yfront)) if method == 1 else abs(xr-xfront)
                    hd.append(distance); zrefs.append(zref*scalar(sqrt(distance)))
                    shell_top = zrefs[-1]+height+base if native_math else zrefs[-1]+effective
                    ks = next((k for k in range(frame.k_end-1, nz-2) if shell_top < z[k+1]), 0)
                    if ks <= frame.k_end-1:
                        applicable[component] = False; denominator = scalar(1.)
                    else:
                        denominator = scalar(1/float(scalar(log((z[ks]-effective)/roughness))))
                    denoms.append(denominator); reference.append(scalar(a[field][ks, j, i]))
                    part = scalar(1-((half_rcx-float(distance))/half_rcx)**2)
                    shells.append(vd*scalar(sqrt(part)) if part > 0 else scalar(0.))
                for k in range(frame.k_end, (kref if method == 1 else kendv)+1):
                    if flags[k, j, i] == 0:
                        break
                    zroof = z[k]-effective; changed = [False, False]
                    for component, field in [(0, 'u0'), (1, 'v0')]:
                        if not applicable[component]:
                            continue
                        if zroof <= zrefs[component]:
                            if component == 0:
                                urooftop = scalar(a[field][k, j, i])
                            else:
                                vrooftop = scalar(a[field][k, j, i])
                            a[field][k, j, i] = reference[component]*scalar(log(zroof/roughness))*denoms[component]
                            changed[component] = True
                            if whole and flags[k, j, i] not in (7, 8):
                                flags[k, j, i] = 10
                        if method == 2 and hd[component] < rcx and zroof <= shells[component]:
                            original = urooftop if component == 0 else vrooftop
                            a[field][k, j, i] = -original*abs((shells[component]-zroof)/vd)
                            changed[component] = True
                            if whole and flags[k, j, i] not in (7, 8):
                                flags[k, j, i] = 10
                    if not any(changed):
                        break
    angles = face_angles(frame.local, native_math=native_math) if method == 2 else None
    return frame, angles
