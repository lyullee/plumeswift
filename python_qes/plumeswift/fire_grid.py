"""CPU QES fire front, forcing and burn-state updates (GPL-3)."""
from dataclasses import dataclass, fields

import numpy as np

from .fire_front import cpu_front
from .fire_fuels import Fuel, fuel_properties
from .fire_spread import FireProperties, balbi
from .fire_spread_startup import calm_startup


def _round_index(value):
    return np.copysign(np.floor(abs(value)+.5), value).astype(int)


def _float_norm(x, y):
    return np.sqrt(x*x+y*y)


@dataclass(frozen=True)
class Ignition:
    x: float
    y: float
    length: float
    width: float
    height: float = 0.
    base_height: float = 0.


@dataclass(frozen=True)
class TimedIgnition:
    time: float
    x: float
    y: float


def read_timed_ignitions(path, *, native_input=False):
    from .io import _open_netcdf

    with _open_netcdf(path, raw=native_input) as nc:
        time = nc['time'][:]
        # NcVar::getVar reads the stored values up to the length of time.
        stop = len(time) if native_input and time.ndim == 1 else None
        fields = time, nc['y1'][:stop], nc['x1'][:stop]
    if any(np.ma.is_masked(field) for field in fields):
        raise ValueError('Missing data in timed ignition file')
    time, x, y = (np.asarray(field, dtype=np.float32) for field in fields)
    if time.ndim != 1 or x.shape != time.shape or y.shape != time.shape or not np.isfinite([time, x, y]).all():
        raise ValueError('Expected matching finite one-dimensional ignition coordinates and times')
    # Native swaps the two file axes and applies a 750 m y-origin.
    return tuple(TimedIgnition(float(t), float(i), float(j)) for t, i, j in zip(time, x, 750-y))


class FireGrid:
    """A reusable front on the horizontal QES wind cell grid."""

    def __init__(self, terrain, spacing, fuel_codes, ignitions=(), *, moisture, cure, courant,
                 raster_fuel=False, timed_ignitions=(), spread=balbi, initialize_front=cpu_front,
                 native_flame_sampling=False, native_fire_clock=False, native_ignition_input=False):
        self.terrain = np.array(terrain, dtype=float, copy=True)
        if self.terrain.ndim != 2 or min(self.terrain.shape) < 3 or not np.isfinite(self.terrain).all():
            raise ValueError('Fire terrain requires a finite horizontal grid of at least 3 by 3 cells')
        if not np.isfinite([*spacing, moisture, cure, courant]).all() or min(spacing) <= 0 or moisture < 0 or courant <= 0:
            raise ValueError('Invalid fire spacing, moisture, cure or Courant value')
        self.dx, self.dy, self.dz = spacing
        self.moisture, self.cure, self.courant = moisture, cure, courant
        self.spread = spread
        self.native_flame_sampling = native_flame_sampling
        self.native_fire_clock = native_fire_clock
        self.native_front = native_flame_sampling and native_fire_clock
        self.fuel_map = np.broadcast_to(fuel_codes, self.terrain.shape).copy()
        self.fuel = fuel_properties(self.fuel_map, raster=raster_fuel)
        self.interior = (slice(1, -1), slice(1, -1))
        shape = self.terrain.shape
        cells = np.asarray(shape[::-1])
        coordinate_type = np.float32 if native_ignition_input else float
        grid_spacing = np.array((self.dx, self.dy), dtype=coordinate_type)
        upper_face = np.multiply(cells, grid_spacing, dtype=coordinate_type)
        self.timed_ignitions = []
        for event in timed_ignitions:
            if not np.isfinite([event.time, event.x, event.y]).all():
                raise ValueError('Invalid timed ignition')
            coordinates = np.array((event.x, event.y), dtype=coordinate_type)
            if not np.isfinite(coordinates).all():
                raise ValueError('Timed ignition is outside the fire grid')
            rounded = _round_index((coordinates/grid_spacing).astype(float, copy=False))
            if native_ignition_input:
                # Move.cpp clips the upper face indices, then flattens them.
                i, j = map(int, np.minimum(rounded, cells))
                index = i+j*shape[1]
                if 0 <= index < self.terrain.size:
                    j, i = divmod(index, shape[1])
                    self.timed_ignitions.append((event.time, j, i))
                    continue
            if np.any(coordinates > upper_face) or np.any(rounded < 0):
                raise ValueError('Timed ignition is outside the fire grid')
            # Keep nominal faces that lack storage on the nearest allocated cell.
            i, j = np.minimum(rounded, cells-1)
            self.timed_ignitions.append((event.time, j, i))
        ignited = np.zeros(shape, dtype=bool)
        for ignition in ignitions:
            origin = np.array((ignition.x, ignition.y), dtype=coordinate_type)
            size = np.array((ignition.length, ignition.width), dtype=coordinate_type)
            corners = np.stack((origin, origin+size))/grid_spacing
            if not np.isfinite(corners).all():
                raise ValueError('Invalid ignition rectangle')
            # Native corner arithmetic is float32. Widen only for the
            # half-away-from-zero rounding step to avoid rounding twice.
            (a, c), (b, d) = _round_index(corners.astype(float, copy=False)).tolist()
            if native_ignition_input and (a >= b or c >= d):
                # An empty native loop does not address any storage.
                continue
            if 0 <= a <= b <= shape[1] and 0 <= c <= d <= shape[0]:
                ignited[c:d, a:b] = True
                continue
            # Fire.cpp uses flat row offsets even for rectangles crossing an edge.
            # The first and last offsets bound all writes in between.
            if (native_ignition_input and 0 <= c*shape[1]+a
                    and (d-1)*shape[1]+b <= ignited.size):
                flat = ignited.ravel()
                for row in range(c, d):
                    offset = row*shape[1]
                    flat[offset+a:offset+b] = True
                continue
            raise ValueError('Ignition rectangle is outside the fire grid')
        self.front = initialize_front(ignited)
        self.burn = ignited.astype(float)
        self.burn_flag = self.burn.copy()
        self.burn_out = np.zeros(shape)
        residence_dtype = np.float32 if native_fire_clock else float
        self.burn_time = np.zeros(shape, dtype=residence_dtype)
        self.smoke = ignited.astype(np.int32)
        self.force = np.zeros(shape)
        self.heat = np.zeros(shape)
        self.del_plus = np.zeros(shape)
        self.del_min = np.zeros(shape)
        self.normal_x = np.zeros(shape)
        self.normal_y = np.zeros(shape)
        self.slope_x = np.zeros(shape)
        self.slope_y = np.zeros(shape)
        sx = self.terrain[1:-1, 2:]-self.terrain[1:-1, :-2]
        sy = self.terrain[2:, 1:-1]-self.terrain[:-2, 1:-1]
        self.slope_x[self.interior] = sx/(2*self.dx*np.hypot(sx, 2*self.dx))
        self.slope_y[self.interior] = sy/(2*self.dy*np.hypot(sy, 2*self.dy))
        self.properties = FireProperties(*(
            np.zeros(shape, dtype=residence_dtype if field.name == 'residence_time' else float)
            for field in fields(FireProperties)
        ))
        self.time = 0.

    def timestep(self):
        rates = self.properties.spread_rate[self.burn == 1]
        if self.native_fire_clock:
            maximum = np.fmax.reduce(rates, initial=.05)
            return float(np.float32(self.courant)*np.float32(self.dx)/np.float32(maximum))
        return self.courant*self.dx/max(.05, np.max(rates, initial=0.))

    def _gradients(self, j, i):
        dtype = np.float32 if self.native_front else float
        center = self.front[j, i].astype(dtype, copy=False)
        # Both native level-set implementations interchange the x/y divisors.
        bx = (center-self.front[j, i-1].astype(dtype, copy=False))/dtype(self.dy)
        fx = (self.front[j, i+1].astype(dtype, copy=False)-center)/dtype(self.dy)
        by = (center-self.front[j-1, i].astype(dtype, copy=False))/dtype(self.dx)
        fy = (self.front[j+1, i].astype(dtype, copy=False)-center)/dtype(self.dx)
        # Mixed float/int fmin and fmax widen the native Godunov norm before
        # storing it as a float. The front normals instead stay in float32.
        bx_wide, fx_wide, by_wide, fy_wide = (value.astype(float, copy=False) for value in (bx, fx, by, fy))
        self.del_plus[j, i] = np.sqrt(np.maximum(bx_wide, 0)**2+np.minimum(fx_wide, 0)**2+
                                      np.maximum(by_wide, 0)**2+np.minimum(fy_wide, 0)**2).astype(dtype, copy=False)
        self.del_min[j, i] = np.sqrt(np.maximum(fx_wide, 0)**2+np.minimum(bx_wide, 0)**2+
                                     np.maximum(fy_wide, 0)**2+np.minimum(by_wide, 0)**2).astype(dtype, copy=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm2 = _float_norm if self.native_front else np.hypot
            pp, mp, pm, mm = norm2(fx, fy), norm2(bx, fy), norm2(fx, by), norm2(bx, by)
            nx = fx/pp+bx/mp+fx/pm+bx/mm
            ny = fy/pp+fy/mp+by/pm+by/mm
            norm = norm2(nx, ny)
            self.normal_x[j, i] = np.where(nx == 0, 0, nx/norm)
            self.normal_y[j, i] = np.where(ny == 0, 0, ny/norm)

    def _fuel(self, j, i):
        return Fuel(*(getattr(self.fuel, field.name)[j, i] for field in fields(Fuel)))

    def _wind_force(self, workspace, j, i, k, fuel):
        a = workspace.arrays
        if np.any(k < 0) or np.any(k >= a['u'].shape[0]):
            raise ValueError('Flame wind sampling level is outside the face grid')
        u = .5*(a['u'][k, j, i]+a['u'][k, j, i+1])
        v = .5*(a['v'][k, j, i]+a['v'][k, j+1, i])
        nx, ny = self.normal_x[j, i], self.normal_y[j, i]
        sx, sy = self.slope_x[j, i], self.slope_y[j, i]
        options = dict(cure=self.cure, dx=self.dx, dy=self.dy)
        startup = self.native_front and self.time == 0 and self.spread is balbi and np.ndim(fuel.one_hour) == 0
        calm = (u == 0) & (v == 0) if startup else None
        if calm is not None and calm.all():
            spread = calm_startup(fuel, nx, ny, sx, sy, self.moisture, **options)
        else:
            spread = self.spread(fuel, u, v, nx, ny, sx, sy, self.moisture, **options)
            if calm is not None and calm.any():
                exact = calm_startup(fuel, nx[calm], ny[calm], sx[calm], sy[calm], self.moisture, **options)
                for field in fields(FireProperties):
                    getattr(spread, field.name)[calm] = getattr(exact, field.name)
        for field in fields(FireProperties):
            getattr(self.properties, field.name)[j, i] = getattr(spread, field.name)
        self.force[j, i] = spread.spread_rate
        return spread

    def _flame_levels(self, j, i):
        terrain = self.terrain[j, i]
        height, residence = self.properties.flame_height[j, i], self.properties.residence_time[j, i]
        if self.native_flame_sampling:
            terrain, height, residence, elapsed, dz = map(
                np.float32, (terrain, height, residence, self.burn_time[j, i], self.dz))
            fraction = elapsed/residence if residence else np.float32(0.)
            decay_height = height*(np.float32(1.)-fraction)
            fuel_depth = np.float32(float(np.float32(self.fuel.depth[j, i]))*.3048)
            # LevelSetNB stores the heights as floats before dividing by dz.
            midpoint = np.float32(float(decay_height)/2+float(terrain)+float(fuel_depth))
            maximum = height+terrain+fuel_depth
            positions = np.array((terrain, midpoint if decay_height > 0 else terrain,
                                  maximum if height != 0 else terrain), dtype=np.float32)
            return tuple(map(int, _round_index((positions/dz).astype(float))))
        fraction = self.burn_time[j, i]/residence if residence else 0.
        decay_height = height*(1-fraction)
        fuel_depth = self.fuel.depth[j, i]*.3048
        bottom = int(_round_index(np.asarray(terrain/self.dz)))
        sample = int(_round_index(np.asarray((terrain+fuel_depth+.5*decay_height)/self.dz))) if decay_height > 0 else bottom
        top = int(_round_index(np.asarray((terrain+fuel_depth+height)/self.dz))) if height != 0 else bottom
        return bottom, sample, top

    def _mark_flame(self, workspace, j, i, bottom, top, flag):
        array = workspace.arrays['icellflag']
        if not 0 <= bottom <= top < array.shape[0]:
            raise ValueError('Flame extent is outside the cell grid')
        array[bottom:top+1, j, i] = flag

    def forcing(self, workspace, *, narrow=True):
        """Calculate forcing using the native full-grid or narrow-band update order."""
        if workspace.arrays['terrain'].shape != self.terrain.shape:
            raise ValueError('Fire and wind horizontal grids differ')
        self.force.fill(0)
        if narrow:
            self.heat.fill(0)
        else:
            j, i = np.indices(tuple(size-2 for size in self.terrain.shape))+1
            self._gradients(j, i)
            height = self.properties.flame_height[j, i]
            position = self.terrain[j, i]+np.where(height == 0, 0, height+self.fuel.depth[j, i]*.3048)
            self._wind_force(workspace, j, i, _round_index(position/self.dz), self._fuel(j, i))
        for j, i in np.argwhere(self.burn == 1):
            bottom, sample, top = self._flame_levels(j, i)
            if narrow:
                # The reference Linux narrow-band driver clamps the initial
                # NaN-derived flame index to face level zero. Do so explicitly
                # before the first residence time is available.
                if self.native_flame_sampling and self.properties.residence_time[j, i] == 0:
                    sample = 0
                y = np.arange(max(1, j-2), min(self.terrain.shape[0]-1, j+3))
                x = np.arange(max(1, i-2), min(self.terrain.shape[1]-1, i+3))
                yy, xx = np.meshgrid(y, x, indexing='ij')
                self._gradients(yy, xx)
                result = self._wind_force(workspace, yy, xx, max(0, sample), self._fuel(j, i))
                self.heat[yy, xx] = result.heat_release*np.clip(1-self.front[yy, xx], 0, 1)
            else:
                self._wind_force(workspace, j, i, sample, self._fuel(j, i))
            self._mark_flame(workspace, j, i, bottom, top, 12)

    def advance(self, workspace):
        """Advance the native level-set transport and residence-time states."""
        dt = self.timestep()
        region = self.interior
        previous = self.burn_flag
        if self.timed_ignitions:
            start = np.float32(self.time)
            end = start+np.float32(dt)
            for time, j, i in self.timed_ignitions:
                if start <= time <= end and previous[j, i] < 2:
                    self.front[j, i] = 0
                    self.burn[j, i] = 1
        self.burn_time[region] += dt*(previous[region] == 1)
        unburned = previous[region] < 1
        finished = (self.burn[region] == 1) & (self.burn_time[region] >= self.properties.residence_time[region])
        for j, i in np.argwhere(finished)+1:
            bottom, _, top = self._flame_levels(j, i)
            self._mark_flame(workspace, j, i, bottom, top, 1)
        self.burn[region][finished] = 2
        self.force[region][finished] = 0
        self.heat[region][finished] = 0
        force = self.force[region]
        if self.native_front:
            force = force.astype(np.float32).astype(float)
            updated = self.front[region].astype(np.float32).astype(float)-dt*(
                np.fmax(force, 0)*self.del_plus[region]+np.fmin(force, 0)*self.del_min[region])
            self.front[region] = updated.astype(np.float32)
        else:
            self.front[region] -= dt*(np.maximum(force, 0)*self.del_plus[region]+np.minimum(force, 0)*self.del_min[region])
        preheat = unburned & (self.front[region] <= 1)
        ignite = unburned & (self.front[region] <= .1)
        self.burn[region][preheat] = .5
        self.burn[region][ignite] = 1
        self.smoke[region][ignite] = 1
        self.burn_flag[region] = self.burn[region]
        self.burn_out[region] = self.burn[region]
        self.time += dt
        return dt

    def step(self, workspace, *, narrow=True):
        self.forcing(workspace, narrow=narrow)
        return self.advance(workspace)
