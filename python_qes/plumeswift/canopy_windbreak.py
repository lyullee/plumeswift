"""QES Guan porous windbreak and wake, CanopyWindbreak.cpp (GPL-3)."""
from dataclasses import dataclass
from math import ceil, hypot, pi, tanh
import numpy as np
from .geometry import PolygonBuilding
from .upwind import _round
from .vegetation import CanopyPrism


def _native_wake_defect(component, porosity, height, spread, shear, z, zwmo, xl, yl, y1, y2):
    # QES stores the shear arguments in float before evaluating the double tanh.
    f32 = np.float32
    if component == 0:
        exponent = 1.5*float(f32(z-zwmo))/float(shear)
    else:
        scale = 1.5/float(shear)
        exponent = scale*float(z)-scale*float(zwmo)
    aero = f32(.5*(1.-porosity)*tanh(exponent)+.5*(1.+porosity))
    recovery = f32(.5-.5*tanh((float(f32(xl-f32(7.5)*height))-2.*height)/height))
    defect = f32((1.-float(aero))*float(recovery))
    ds = f32(.5*float(spread)*float(xl))
    left = 1.5*float(f32(yl-f32(y2-ds)))
    right = 1.5*float(f32(f32(y1+ds)-yl))
    if ds == 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            left, right = np.divide((left, right), ds)
    else:
        left, right = left/float(ds), right/float(ds)
    defect = f32(float(defect)*(.5-.5*tanh(left)))
    return f32(float(defect)*(.5-.5*tanh(right)))


@dataclass(frozen=True)
class Windbreak(CanopyPrism):
    vertices: tuple[tuple[float, float], ...]
    height: float
    porosity: float
    understory_height: float
    base_height: float = 0.
    model: int = 2
    fetch: float = 7.  # Parsed by QES; not used in either active closure.

    def __post_init__(self):
        PolygonBuilding(self.vertices, self.height, self.base_height)
        if not np.isfinite([self.porosity, self.understory_height, self.fetch]).all() or not 0 <= self.porosity <= 1:
            raise ValueError('Invalid windbreak porosity or geometry parameters')

    @classmethod
    def rectangle(cls, x, y, length, width, height, porosity, understory_height, rotation, base_height=0., model=2, fetch=7.):
        # QES windbreak length is along the transverse edge (unlike the
        # rectangular building/homogeneous-canopy input convention).
        b = PolygonBuilding.rectangle(x, y, width, length, height, rotation, base_height)
        return cls(b.vertices, height, porosity, understory_height, base_height, model, fetch)

    @property
    def aerodynamic_porosity(self):
        return self.porosity if self.model == 1 else self.porosity**.4

    def _porosity(self, native_math):
        if not native_math:
            return self.aerodynamic_porosity
        porosity = np.float32(self.porosity)
        return porosity if self.model == 1 else np.float32(float(porosity)**.4)

    def flag(self, workspace, state, number):
        columns = super().flag(workspace, state, number, cell_flag=24)
        top = max((state.top_index[j, i] for j, i in columns), default=0)
        if ceil(1.5*top) > len(workspace.coordinates['z']):
            raise ValueError('Domain too short for native windbreak')
        return columns

    def apply(self, workspace, state, columns):
        a, c = workspace.arrays, workspace.coordinates
        terrain_id = a.get('terrain_id', np.ones(a['terrain'].shape, dtype=int))
        faces_u, faces_v = set(), set()
        for j, i in columns:
            relative = c['z_face']-c['z'][terrain_id[j, i]]
            for k in range(state.bot_index[j, i], state.top_index[j, i]):
                if self.understory_height <= relative[k] <= self.height:
                    faces_u.update(((k, j, i), (k, j, i+1)))
                    faces_v.update(((k, j, i), (k, j+1, i)))
        porosity = self._porosity(state.native_geometry)
        for key, indices in [('u0', faces_u), ('v0', faces_v)]:
            if indices:
                index = tuple(np.array(sorted(indices)).T)
                velocity = a[key][index]
                if state.native_geometry:
                    velocity = velocity.astype(np.float32)
                a[key][index] = velocity*porosity

    def apply_wake(self, workspace, state, number, defects):
        a, c = workspace.arrays, workspace.coordinates
        flags = a['icellflag']; nzc, nyc, nxc = flags.shape
        dx = c['x'][1]-c['x'][0]; dy = dx
        dxy = min(dx, c['y'][1]-c['y'][0])
        dz = float(np.min(a['dz_array']))
        terrain_id = a.get('terrain_id', np.ones((nyc, nxc), dtype=int))
        top = state.top_at_setup[number]
        if top == 0:
            return
        vertices = np.asarray(self.vertices, dtype=float)
        center = vertices[:-1].mean(axis=0)
        cx, cy = center; ic, jc = _round(cx/dx)-1, _round(cy/dy)-1
        theta = np.arctan2(a['v0'][top, jc, ic], a['u0'][top, jc, ic]); cs, sn = np.cos(theta), np.sin(theta)
        relative = (vertices-center)@np.array([[cs, -sn], [sn, cs]])
        closed = np.flatnonzero(np.all(np.abs(vertices[1:]-vertices[0]) < .1, axis=1))
        stop = int(closed[0])+1
        y1, y2 = min(0., relative[:stop, 1].min()), max(0., relative[:stop, 1].max())
        native = state.native_geometry
        porosity = self._porosity(native)
        h = self.height; tol = .01*pi/180
        if native:
            f32 = np.float32
            udelt, uave = f32(1)-porosity, f32(.5)*(f32(1)+porosity)
            spread_classic = f32(.14*float(udelt)/float(uave))
            height32 = f32(h)
        else:
            spread_classic = .14*(1-porosity)/(.5*(1+porosity))
        z_coords, z_faces = state.vertical_coordinates
        for edge in range(0, stop, 2):
            x0, y0 = relative[edge]; x1, ye = relative[edge+1]
            angle = np.arctan2(ye-y0, x1-x0)+.5*pi
            if angle > pi+.0001: angle -= 2*pi
            if abs(angle) >= .5*pi-tol:
                continue
            slope = 0. if abs(angle) < tol else (x1-x0)/(ye-y0)
            for y_id in range(2*ceil(abs(y0-ye)/dxy)+1):
                yc = y0-.5*y_id*dxy; xwall = x0+slope*(yc-y0)
                ynorm = y2 if yc >= 0 else y1
                for z_step in range(ceil(5*h/dz)):
                    z_id = 5*h/dz-z_step
                    x_id_min = -1
                    for x_id in range(1, 2*ceil(11.5*h/dxy)+1):
                        xc = .5*x_id*dxy
                        x = ((xc+xwall)*cs-yc*sn+cx)/dx
                        y = ((xc+xwall)*sn+yc*cs+cy)/dy
                        i, j = ceil(x)-1, ceil(y)-1
                        # Original && predicate cannot catch exits and may
                        # dereference an invalid terrain id before checking.
                        if not (0 <= i < nxc and 0 <= j < nyc):
                            continue
                        terrain = terrain_id[j, i]
                        k = int(z_id+terrain-1); ktop = int(h/dz+terrain)
                        if not (0 <= k < nzc and 0 <= ktop <= nzc):
                            continue
                        flag = flags[k, j, i]
                        if flag not in (0, 2) and x_id_min < 0:
                            x_id_min = x_id
                        if flag in (0, 2, 24) and x_id_min >= 0:
                            if a['ibuilding_flag'][k, j, i] == number:
                                x_id_min = -1
                            elif flag in (0, 2):
                                break
                        if flag in (0, 2, 24):
                            continue
                        if native:
                            u, v, w = (f32(a[key][ktop, j, i]) for key in ('u0', 'v0', 'w0'))
                            top_speed = np.sqrt(u*u+v*v)
                            zwmo = height32+w/top_speed*f32(dx) if top_speed else height32
                            upstream = f32(2.*float(f32(.19))/float(top_speed)) if top_speed else f32(np.inf)
                            spread = np.sqrt(spread_classic*spread_classic+upstream*upstream)
                            shear = spread*f32(xc)
                        else:
                            top_speed = hypot(a['u0'][ktop, j, i], a['v0'][ktop, j, i])
                            zwmo = h+a['w0'][ktop, j, i]/top_speed*dx if top_speed else h
                            spread = hypot(spread_classic, .38/top_speed) if top_speed else np.inf
                            shear = spread*xc
                        z = z_coords[k]-z_faces[terrain]
                        end_flags = [False, False, False]
                        for component, fi, fj, ox, oy in [(0, _round(x), int(y), 0., .5),
                                                         (1, int(x), _round(y), .5, 0.),
                                                         (2, i, j, .5, .5)]:
                            if not (0 < fi < nxc and 0 < fj < nyc):
                                continue
                            xp, yp = (fi+ox)*dx-cx, (fj+oy)*dy-cy
                            yl = -xp*sn+yp*cs
                            xl = xp*cs+yp*sn-(x0+slope*(yl-y0))
                            depth = h if abs(yl) < abs(ynorm) and abs(ynorm) > 1e-9 and h > 1e-9 else 0.
                            end_flags[component] = xl > 11.5*depth
                            if depth == 0 or end_flags[component] or flags[k, fj, fi] in (0, 2):
                                continue
                            if component == 2:
                                flags[k, fj, fi] = 25
                                continue
                            if native:
                                defects[component, k, fj, fi] = _native_wake_defect(
                                    component, float(porosity), height32, spread_classic, shear, z, zwmo,
                                    f32(xl), f32(yl), f32(y1), f32(y2))
                                continue
                            aeropor = .5*(1-porosity)*tanh(1.5*(z-zwmo)/shear)+.5*(1+porosity)
                            recovery = .5-.5*tanh((xl-9.5*h)/h)
                            ds = .5*spread_classic*xl
                            with np.errstate(divide='ignore', invalid='ignore'):
                                left = .5-.5*np.tanh(np.divide(1.5*(yl-(y2-ds)), ds))
                                right = .5-.5*np.tanh(np.divide(1.5*((y1+ds)-yl), ds))
                            defects[component, k, fj, fi] = (1-aeropor)*recovery*left*right
                        if all(end_flags):
                            break
