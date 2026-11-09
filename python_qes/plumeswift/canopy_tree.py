"""QES isolated tree attenuation and elliptical wake (GPL-3)."""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, exp, hypot, sqrt

import numpy as np

from .geometry import PolygonBuilding
from .numerics.linux_float_angles import atan2f, sincosf
from .numerics.linux_float_transcendentals import expf, logf
from .upwind import _round
from .vegetation import CanopyPrism


@lru_cache(maxsize=16384)
def _wake_ellipse(along, side, vertical, height, *, native_math):
    if native_math:
        ratio = float(np.float32(along)/np.float32(height))
        side, vertical = float(np.float32(side)), float(np.float32(vertical))
        radius = float(np.float32(sqrt(vertical**2+side**2)))
        _, cosine = sincosf(atan2f(vertical, side))
        cos_sq = float(cosine)**2
    else:
        ratio = along/height
        radius = hypot(vertical, side)
        cos_sq = (side/radius)**2 if radius else 1.
    width = .05*ratio+2.12 if ratio < 7.77 else 1.70*ratio**.19
    if native_math:
        width = float(np.float32(width))
    scale = sqrt(1-(1-((width-1.15)/(width+1.15))**2)*cos_sq)
    if native_math:
        # The native membership test uses the stored float radius and diameter.
        diameter = float(np.float32((width-1.15)/scale*height))
    else:
        diameter = (width-1.15)*height/scale
    return ratio, radius, diameter


@dataclass(frozen=True)
class IsolatedTree(CanopyPrism):
    vertices: tuple[tuple[float, float], ...]
    height: float
    leaf_area_index: float
    peak_fraction: float
    base_height: float = 0.
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self):
        PolygonBuilding(self.vertices, self.height, self.base_height, self.holes)
        if not np.isfinite([self.leaf_area_index, self.peak_fraction]).all() or self.leaf_area_index < 0:
            raise ValueError('Invalid isolated tree leaf area index or peak height')

    @classmethod
    def circle(cls, x, y, width, height, attenuation, peak_fraction, base_height=0.):
        if not np.isfinite(width) or width <= 0:
            raise ValueError('Tree width must be positive')
        angles = np.arange(36)*2*3.14159265359/36
        vertices = np.float32(np.column_stack((x+.5*width*np.cos(angles), y+.5*width*np.sin(angles))))
        vertices = np.vstack((vertices, vertices[0])).astype(float)
        return cls(tuple(map(tuple, vertices)), height, 2*attenuation, peak_fraction, base_height)

    def flag(self, workspace, state, number):
        columns = super().flag(workspace, state, number, cell_flag=22, attenuation=.5*self.leaf_area_index)
        top = max((state.top_index[j, i] for j, i in columns), default=0)
        if ceil(1.5*top) > len(workspace.coordinates['z']):
            raise ValueError('Domain too short for native isolated tree')
        return columns

    def apply(self, workspace, state, columns):
        arrays, coords = workspace.arrays, workspace.coordinates
        nz, ny, nx = arrays['icellflag'].shape
        scalar = np.float32 if state.native_geometry else float
        one = scalar(1.)
        logarithm, exponential = (logf, expf) if state.native_geometry else (np.log, np.exp)
        heights = state.vertical_coordinates[0] if state.native_geometry else coords['z']
        for j, i in columns:
            top, base = scalar(state.top[j, i]), scalar(state.base[j, i])
            if top <= 0:
                continue
            height, z0 = scalar(state.height[j, i]), scalar(state.roughness[j, i])
            displacement = scalar(state.column_displacement(j, i, sentinel_only=True,
                                                            native_math=state.native_geometry))
            attenuation_index = state.top_index[j, i]-1
            for k in range(1, nz):
                z = heights[k]
                if z < base or z >= 2*top:
                    continue
                relative = z-base
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    if z < top:
                        if state.attenuation[attenuation_index, j, i] <= 0:
                            continue
                        attenuation_index = k
                        attenuation = scalar(state.attenuation[k, j, i])
                        factor = logarithm((height-displacement)/z0)*exponential(attenuation*(relative/height-one))/logarithm(relative/z0)
                    else:
                        weight = (relative-height)/height
                        if state.native_geometry:
                            factor = logarithm((relative-displacement)/z0)/logarithm(relative/z0)
                            factor = (one-weight)*factor+weight
                        else:
                            factor = (1-weight)*logarithm((relative-displacement)/z0)/logarithm(relative/z0)+weight
                if factor > 1 or factor < 0:
                    factor = scalar(1.)
                arrays['u0'][k, j, i] = scalar(arrays['u0'][k, j, i])*factor
                arrays['v0'][k, j, i] = scalar(arrays['v0'][k, j, i])*factor
                if i < nx-1 and state.top[j, i+1] == 0:
                    arrays['u0'][k, j, i+1] = scalar(arrays['u0'][k, j, i+1])*factor
                if j < ny-1 and state.top[j+1, i] == 0:
                    arrays['v0'][k, j+1, i] = scalar(arrays['v0'][k, j+1, i])*factor

    def center(self, dx):
        points = np.asarray(self.vertices[:-1], dtype=np.float32)
        center = np.add.accumulate(points, axis=0)[-1]/np.float32(len(points))
        cx, cy = map(float, center)
        return cx, cy, _round(cx/dx)-1, _round(cy/dx)-1

    def _reference_column(self, dx, columns):
        cx, cy, i, j = self.center(dx)
        if self.holes and columns and (j, i) not in columns:
            j, i = min(columns, key=lambda pair: ((pair[1]+.5)*dx-cx)**2+((pair[0]+.5)*dx-cy)**2)
        return j, i

    def wake_reference(self, workspace, top, columns=()):
        if top == 0:
            return 0., 0.
        x = workspace.coordinates['x']
        j, i = self._reference_column(x[1]-x[0], columns)
        return workspace.arrays['u0'][top+1, j, i], workspace.arrays['v0'][top+1, j, i]

    def apply_wake(self, workspace, state, number, defects):
        arrays, coords = workspace.arrays, workspace.coordinates
        flags = arrays['icellflag']
        nz, ny, nx = flags.shape
        start, end = state.bottom_at_setup[number], state.top_at_setup[number]
        if end == 0:
            return
        dx = coords['x'][1]-coords['x'][0]
        dxy = min(dx, coords['y'][1]-coords['y'][0])
        z = state.vertical_coordinates[0]
        height, base = self.height, self.base_height
        peak = self.peak_fraction*height
        if state.native_geometry:
            height, base = float(np.float32(height)), float(np.float32(base))
            peak = float(np.float32(self.peak_fraction)*np.float32(height))
        cx, cy, _, _ = self.center(dx)
        jc, ic = self._reference_column(dx, state.columns[number])
        bottom = next((k for k in range(1, start+1) if base <= z[k]), start)
        top = next((k+1 for k in range(start, nz-1) if 1.5*(height+base) < z[k+1]), nz-1)
        core = next((k for k in range(start, end) if .75*height+base <= z[k]), end-1 if end > start else 0)
        if flags[core, jc, ic] in (3, 4, 6):
            return
        u, v = state.wake_references[number]
        # Native wake rays round their float32 coordinates to staggered faces.
        scalar = np.float32 if state.native_geometry else float
        if state.native_geometry:
            sn, cs = map(scalar, sincosf(atan2f(v, u)))
        else:
            theta = np.arctan2(v, u)
            cs, sn = np.cos(theta), np.sin(theta)
        dx, cx, cy = map(scalar, (dx, cx, cy))
        ref = ceil(1.5*end)
        if ref >= len(z):
            raise ValueError('Isolated tree reference height is outside the center grid')
        with np.errstate(divide='ignore', invalid='ignore'):
            u_ref, v_ref = (scalar(arrays[name][ref, jc, ic]) for name in ('u0', 'v0'))
            if state.native_geometry:
                speed = np.sqrt(u_ref*u_ref+v_ref*v_ref)
                roughness = scalar(state.roughness[jc, ic])
                gap = scalar(z[ref])-scalar(base)+scalar(state.displacement[jc, ic])
                friction = (speed*scalar(.4)/logf(gap/roughness))/speed
            else:
                speed = hypot(u_ref, v_ref)
                friction = .4/np.log((z[ref]-base+state.displacement[jc, ic])/state.roughness[jc, ic]) if speed else np.nan
        span = 2*height
        length = 11*height
        for k in range(top, bottom-1, -1):
            vertical = scalar(z[k])-scalar(base)-scalar(peak)
            for row in range(1, 2*ceil(2*span/dxy)):
                lateral = scalar(.5*row*dxy-span)
                if abs(lateral) > span:
                    continue
                first_fluid = False
                for col in range(2*ceil(length/dxy)+1):
                    downstream = scalar(.5*col*dxy)
                    x = (downstream*cs-lateral*sn+cx)/dx
                    y = (downstream*sn+lateral*cs+cy)/dx
                    i, j = ceil(x)-1, ceil(y)-1
                    if not (0 < i < nx-1 and 0 < j < ny-1):
                        break
                    flag = flags[k, j, i]
                    first_fluid |= flag not in (0, 2)
                    if flag in (0, 2, 22):
                        if first_fluid and state.owner[k, j, i] == number:
                            first_fluid = False
                        elif flag in (0, 2) or (first_fluid and flags[core, j, i] in (0, 2)):
                            break
                        continue
                    past_end = 0
                    for component, fi, fj, ox, oy in (
                        (0, _round(x), int(y), 0., .5),
                        (1, int(x), _round(y), .5, 0.),
                        (2, i, j, .5, .5),
                    ):
                        if not (0 < fi < nx and 0 < fj < ny):
                            continue
                        xp = scalar((fi+ox)*float(dx)-float(cx))
                        yp = scalar((fj+oy)*float(dx)-float(cy))
                        side = -xp*sn+yp*cs
                        if abs(side) > span:
                            break
                        along = xp*cs+yp*sn
                        depth = length if abs(side) < span and span > 1e-9 and height > 1e-9 else 0.
                        if along > depth:
                            past_end += 1
                            continue
                        if depth == 0 or flags[k, fj, fi] in (0, 2):
                            continue
                        ratio, radius, diameter = _wake_ellipse(along, side, vertical, height,
                                                               native_math=state.native_geometry)
                        if radius >= .5*diameter:
                            continue
                        if component == 2:
                            flags[k, fj, fi] = 23
                        else:
                            center_defect = float(friction)*(-.63*ratio+9.33) if ratio < 7.77 else float(friction)*90.68*ratio**-1.48
                            if state.native_geometry:
                                radius, diameter = scalar(radius), scalar(diameter)
                                exponent = -(radius*radius)/(scalar(.083)*diameter*diameter)
                                defect = scalar(center_defect)*expf(exponent)
                            else:
                                defect = center_defect*exp(-radius**2/(.083*diameter**2))
                            if defect > defects[component, k, fj, fi]:
                                defects[component, k, fj, fi] = defect
                    else:
                        if past_end != 3:
                            continue
                    break
