"""QES canopy state, homogeneous vegetation and regression (GPL-3).

Definitions are immutable. A fresh CanopyState owns each run's grid arrays,
so model components can be reused and replaced without leaking prior state.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import PolygonBuilding, ring_contains
from .meteorology import canopy_displacement
from .numerics.linux_float_transcendentals import expf, logf


class CanopyPrism:
    """Shared native column assignment for prism canopies."""
    def flag(self, workspace, state, number, *, cell_flag=20, attenuation=None, base_face_offset=0):
        a, c = workspace.arrays, workspace.coordinates
        z, faces = state.vertical_coordinates
        scalar = np.float32 if state.native_geometry else float
        base, height = scalar(self.base_height), scalar(self.height)
        vertices = np.asarray(self.vertices, dtype=np.float32)
        dx = c['x'][1]-c['x'][0]
        # QES Domain::getDomainSize aliases dy to dx in canopy code.
        dy = dx
        nx, ny = len(c['x']), len(c['y'])
        start = np.trunc(vertices.min(axis=0)/[dx, dy]).astype(int)
        end = np.trunc(vertices.max(axis=0)/[dx, dy]+1).astype(int)
        if min(start) < 0 or end[0] >= nx or end[1] >= ny:
            raise ValueError('Canopy polygon is outside the native canopy bounds')
        x = ((np.arange(start[0], end[0])+.5)*dx).astype(np.float32)
        y = ((np.arange(start[1], end[1])+.5)*dy).astype(np.float32)[:, None]
        inside = ring_contains(vertices, x, y)
        for ring in getattr(self, 'holes', ()):
            inside &= ~ring_contains(np.asarray(ring, dtype=np.float32), x, y)
        columns = []
        for j in range(start[1], end[1]):
            for i in range(start[0], end[0]):
                if not inside[j-start[1], i-start[0]] or a['icellflag_footprint'][j, i] == 0:
                    continue
                columns.append((j, i))
                if cell_flag is not None:
                    a['icellflag_footprint'][j, i] = cell_flag
                terrain = scalar(a['terrain'][j, i])
                bottom_height, top_height = terrain+base, terrain+height
                bottoms = np.flatnonzero(bottom_height <= z[1:])
                tops = np.flatnonzero(top_height < z[1:])
                if not len(bottoms) or not len(tops):
                    raise ValueError('Canopy top/base must be inside the vertical domain')
                bot, top = int(bottoms[0]+1), int(tops[0]+1)
                if bot > top:
                    raise ValueError('Canopy bottom index is above its top index')
                state.bot_index[j, i], state.top_index[j, i] = bot, top
                state.bot[j, i], state.top[j, i] = bottom_height, top_height
                state.base[j, i] = faces[bot+base_face_offset]
                state.height[j, i] = top_height-bottom_height
                for k in range(bot, top):
                    if a['icellflag'][k, j, i] not in (0, 2):
                        if cell_flag is not None:
                            a['icellflag'][k, j, i] = cell_flag
                        if attenuation is not None:
                            state.attenuation[k, j, i] = attenuation
                        state.owner[k, j, i] = number
        return tuple(columns)


@dataclass(frozen=True)
class HomogeneousCanopy(CanopyPrism):
    vertices: tuple[tuple[float, float], ...]
    height: float
    attenuation: float
    base_height: float = 0.
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self):
        PolygonBuilding(self.vertices, self.height, self.base_height, self.holes)
        if not np.isfinite(self.attenuation) or self.attenuation < 0:
            raise ValueError('Canopy attenuation must be finite and nonnegative')

    @classmethod
    def rectangle(cls, x, y, length, width, height, attenuation, rotation=0., base_height=0.):
        geometry = PolygonBuilding.rectangle(x, y, length, width, height, rotation, base_height)
        return cls(geometry.vertices, height, attenuation, base_height)

    def flag(self, workspace, state, number):
        return super().flag(workspace, state, number, attenuation=self.attenuation)

    def apply(self, workspace, state, columns):
        a, c = workspace.arrays, workspace.coordinates
        nzc, nyc, nxc = a['icellflag'].shape
        terrain_id = a.get('terrain_id', np.ones((nyc, nxc), dtype=int))
        scalar = np.float32 if state.native_geometry else float
        one = scalar(1.)
        logarithm, exponential = (logf, expf) if state.native_geometry else (np.log, np.exp)
        z = state.vertical_coordinates[0] if state.native_geometry else c['z']
        for j, i in columns:
            height, z0 = scalar(state.height[j, i]), scalar(state.roughness[j, i])
            if state.top[j, i] <= 0:
                continue
            displacement = scalar(state.column_displacement(j, i, native_math=state.native_geometry))
            for k in range(1, nzc):
                relative = z[k]-z[terrain_id[j, i]]
                if relative < 0:
                    continue
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    if relative < height:
                        attenuation = scalar(state.attenuation[k, j, i])
                        above = scalar(state.attenuation[k+1, j, i]) if k+1 < nzc else scalar(0.)
                        below = scalar(state.attenuation[k-1, j, i])
                        if above != attenuation or below != attenuation:
                            count = scalar(1+int(above > 0)+int(below > 0))
                            attenuation = (attenuation+max(above, scalar(0))+max(below, scalar(0)))/count
                        factor = logarithm((height-displacement)/z0)*exponential(attenuation*(relative/height-one))/logarithm(relative/z0)
                    else:
                        factor = logarithm((relative-displacement)/z0)/logarithm(relative/z0)
                if factor > 1 or factor < 0 or np.isnan(factor):
                    factor = scalar(1.)
                a['u0'][k, j, i] = scalar(a['u0'][k, j, i])*factor
                a['v0'][k, j, i] = scalar(a['v0'][k, j, i])*factor
                if i < nxc-1 and state.top[j, i+1] == 0:
                    a['u0'][k, j, i+1] = scalar(a['u0'][k, j, i+1])*factor
                if j < nyc-1 and state.top[j+1, i] == 0:
                    # Native above-canopy branch uses cellAdd on a FACE id:
                    # its +y offset is nxc, rather than face stride nxc+1.
                    flat = np.ravel_multi_index((k, j, i), a['v0'].shape)
                    offset = nxc+1 if relative < height else nxc
                    values = a['v0'].ravel()
                    values[flat+offset] = scalar(values[flat+offset])*factor


def slope_match(z0, height, attenuation):
    tolerance = z0/100; lo = z0 if z0 <= height else .1; hi = height
    d, residual = .5*(lo+hi), tolerance*10
    if attenuation <= 0:
        return 10000.
    for _ in range(200):
        if abs(residual) <= tolerance or not z0 < d < height:
            break
        d = .5*(lo+hi)
        residual = np.log((height-d)/z0)-height/(attenuation*(height-d))
        if residual > 0: lo = d
        elif residual < 0: hi = d
    return .7*height if d > height else d


class CanopyState:
    def __init__(self, workspace, *, native_geometry=False):
        shape = workspace.arrays['icellflag'].shape; horizontal = shape[1:]
        self.native_geometry = native_geometry
        if native_geometry:
            from .domain import native_vertical_coordinates
            self.vertical_coordinates = native_vertical_coordinates(workspace.arrays['dz_array'])
        else:
            self.vertical_coordinates = (workspace.coordinates['z'], workspace.coordinates['z_face'])
        for name in ['bot', 'top', 'base', 'height', 'roughness', 'ustar', 'displacement']:
            setattr(self, name, np.zeros(horizontal))
        self.bot_index = np.zeros(horizontal, dtype=int)
        self.top_index = np.zeros(horizontal, dtype=int)
        self.attenuation = np.zeros(shape)
        self.owner = np.zeros(shape, dtype=int)
        self.columns = []
        self.top_at_setup = []
        self.bottom_at_setup = []
        self.wake_references = {}
        self.row_flows = {}

    def column_displacement(self, j, i, *, sentinel_only=False, native_math=False):
        height, z0 = self.height[j, i], self.roughness[j, i]
        attenuation = self.attenuation[self.top_index[j, i]-1, j, i]
        # Native bisection can exit at its initial midpoint.
        if attenuation > 0 and z0 >= height:
            displacement = height if z0 == height else 10000.
        elif attenuation > 0 and z0 == 0:
            displacement = .5*height
        else:
            displacement = canopy_displacement(self.ustar[j, i], z0, height, attenuation, native_math=native_math)
        use_slope_match = displacement == 10000. if sentinel_only else displacement > height
        if use_slope_match:
            displacement = slope_match(z0, height, attenuation)
        self.displacement[j, i] = displacement
        return displacement

    def regression(self, workspace):
        a = workspace.arrays
        z = self.vertical_coordinates[0]
        nz = len(z)+1; k_top = 0
        for j, i in zip(*np.nonzero(self.top_index > 0)):
            top = self.top_index[j, i]
            extent = self.top[j, i]+self.height[j, i]
            if self.native_geometry:
                extent = np.float32(extent)
            for k in range(top, nz-2):
                k_top = k
                if extent < z[k+1]: break
            if k_top == top: k_top = top+1
            k_top = min(k_top, nz-1)
            if k_top >= len(z):
                raise ValueError('Native canopy regression would access a height beyond its allocation')
            u, v = (a[key][top:k_top+1, j, i] for key in ('u0', 'v0'))
            terrain = np.float32(a['terrain'][j, i]) if self.native_geometry else a['terrain'][j, i]
            relative = z[top:k_top+1]-terrain
            if len(u) < 2 or np.any(relative <= 0):
                raise ValueError('Canopy regression requires at least two above-ground samples')
            if self.native_geometry:
                self.ustar[j, i], self.roughness[j, i] = _native_canopy_regression(u, v, relative)
                continue
            speed = np.hypot(u, v)
            log_height = np.log(relative)
            centered_speed, centered_log = speed-speed.mean(), log_height-log_height.mean()
            # Same inverse regression as Canopy.cpp, with centered double
            # sums to avoid float cancellation in n*sum(x*x)-sum(x)**2.
            covariance = np.dot(centered_speed, centered_log)
            if covariance == 0:
                raise FloatingPointError('Singular native canopy regression; no substitute roughness applied')
            self.ustar[j, i] = .4*np.dot(centered_speed, centered_speed)/covariance
            self.roughness[j, i] = np.exp(log_height.mean()-.4/self.ustar[j, i]*speed.mean())
            if not np.isfinite([self.ustar[j, i], self.roughness[j, i]]).all():
                raise FloatingPointError('Non-finite canopy regression')

    def export(self, workspace):
        for name in ['bot', 'top', 'base', 'height', 'roughness', 'ustar', 'displacement', 'bot_index', 'top_index', 'attenuation', 'owner']:
            workspace.arrays['canopy_'+name] = getattr(self, name)


def _native_canopy_regression(u, v, relative):
    from math import exp, log, sqrt

    # Canopy.cpp uses float running sums and promotes each pow(..., 2.0).
    f32 = np.float32
    sum_x = sum_y = sum_xy = sum_x_sq = f32(0)
    for uu, vv, zz in zip(u, v, relative, strict=True):
        speed = f32(sqrt(float(f32(uu))**2+float(f32(vv))**2))
        y = f32(log(float(zz)))
        sum_x += speed
        sum_y += y
        sum_xy += speed*y
        sum_x_sq = f32(float(sum_x_sq)+float(speed)**2)
    count = f32(len(u))
    covariance = count*sum_xy-sum_x*sum_y
    if covariance == 0:
        raise FloatingPointError('Singular native canopy regression; no substitute roughness applied')
    vk = f32(.4)
    ustar = f32(float(vk)*(float(count*sum_x_sq)-float(sum_x)**2)/float(covariance))
    roughness = f32(exp(float(sum_y/count-(vk/ustar)*(sum_x/count))))
    if not np.isfinite([ustar, roughness]).all():
        raise FloatingPointError('Non-finite canopy regression')
    return ustar, roughness


@dataclass(frozen=True)
class Vegetation:
    canopies: tuple = ()
    wake: bool = True

    def setup(self, workspace, *, native_geometry=False):
        state = CanopyState(workspace, native_geometry=native_geometry)
        for number, canopy in enumerate(self.canopies):
            columns = canopy.flag(workspace, state, number)
            state.columns.append(columns)
            state.top_at_setup.append(max((state.top_index[j, i] for j, i in columns), default=0))
            state.bottom_at_setup.append(min((state.bot_index[j, i] for j, i in columns), default=0))
        state.export(workspace)
        return state

    def apply(self, workspace, state):
        state.regression(workspace)
        for number in self.order():
            canopy = self.canopies[number]
            if hasattr(canopy, 'apply_rows'):
                state.row_flows[number] = canopy.apply_rows(workspace, state)
            else:
                canopy.apply(workspace, state, state.columns[number])
            if self.wake and hasattr(canopy, 'wake_reference'):
                footprint = (state.columns[number],) if getattr(canopy, 'holes', ()) else ()
                state.wake_references[number] = canopy.wake_reference(workspace, state.top_at_setup[number], *footprint)
        state.export(workspace)

    def order(self):
        # Native ordering is descending, with the right half first on ties.
        # XML homogeneous height_eff is uninitialized upstream: use the
        # defined declared height+base, not arbitrary heap contents.
        return sorted(range(len(self.canopies)), key=lambda n: (-(self.canopies[n].height+self.canopies[n].base_height), -n))

    def apply_wake(self, workspace, state):
        if not self.wake:
            return
        closures = [(n, self.canopies[n].apply_wake) for n in self.order() if hasattr(self.canopies[n], 'apply_wake')]
        if not closures:
            return
        a = workspace.arrays
        defects = np.zeros((2, *a['u0'].shape), dtype=np.float32 if state.native_geometry else float)
        for number, closure in closures:
            closure(workspace, state, number, defects)
        for component, defect in zip(('u0', 'v0'), defects, strict=True):
            if state.native_geometry:
                stored = a[component].astype(np.float32)
                a[component][:] = (stored*(1.-defect.astype(float))).astype(np.float32)
            else:
                a[component] *= 1.-defect
