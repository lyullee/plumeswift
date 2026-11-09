"""HRRR NetCDF fields, stability, and QES profile 5 (GPL-3.0-only)."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from operator import index
from pathlib import Path

import numpy as np

from .io import _open_netcdf
from .meteorology import Station, stability_correction
from .native_sensor import _log, _stability
from .qes_xml import read_qes_xml
from .utm import from_utm, to_utm


_SURFACE = ('UGRD_10maboveground', 'VGRD_10maboveground', 'SFCR_surface', 'TCDC_entireatmosphere',
            'DSWRF_surface', 'SHTFL_surface', 'FRICV_surface', 'TMP_surface')
_ALOFT = ('HPBL_surface', *(f'{component}GRD_{level}mb' for level in (925, 850, 700) for component in 'UV'))
_SMOKE_DENSITY = 'MASSDEN_8maboveground'


def _variable(nc, name, dims, *, positional):
    if name not in nc.variables:
        raise ValueError(f'HRRR field {name} must use dimensions {dims}')
    variable = nc[name]
    if positional:
        expected = tuple(len(nc.dimensions[dim]) for dim in dims)
        if len(dims) == 1 and variable.ndim == 2:
            expected += (1,)
        if variable.ndim != len(expected) or any(size < need for size, need in zip(variable.shape, expected)):
            raise ValueError(f'HRRR field {name} does not cover dimensions {dims}')
    elif variable.dimensions != dims:
        raise ValueError(f'HRRR field {name} must use dimensions {dims}')
    return variable


def _data(nc, name, dims, *, positional=False):
    variable = _variable(nc, name, dims, positional=positional)
    if positional:
        # Native reads use the named dimension lengths but address variables by position.
        selection = tuple(slice(len(nc.dimensions[dim])) for dim in dims)
        if len(dims) == 1 and variable.ndim == 2:
            selection += (0,)
    else:
        selection = slice(None)
    values = variable[selection]
    if np.ma.is_masked(values) or not np.isfinite(values).all():
        raise ValueError(f'HRRR field {name} contains missing or nonfinite data')
    return np.asarray(values, dtype=float)


def _wind(u, v, longitude, *, native_math=False):
    if native_math:
        rotation = (float(np.float32(.622515))*(longitude+97.5)*.017453).astype(np.float32).astype(float)
        cosine, sine = (trig(rotation).astype(np.float32) for trig in (np.cos, np.sin))
        u, v = u.astype(np.float32), v.astype(np.float32)
        x = (cosine*u+sine*v).astype(float)
        y = (-sine*u+cosine*v).astype(float)
        return np.sqrt(x*x+y*y), np.fmod(270-np.arctan2(y, x)*180/np.pi, 360.)
    angle = .622515*(longitude+97.5)*.017453
    x = np.cos(angle)*u+np.sin(angle)*v
    y = -np.sin(angle)*u+np.cos(angle)*v
    return np.hypot(x, y), np.mod(270-np.rad2deg(np.arctan2(y, x)), 360.)


def _locations(nc, origin, zone, size, margin, *, positional=False):
    for axis in ('x', 'y'):
        _data(nc, axis, (axis,), positional=positional)
    lat = _data(nc, 'latitude', ('y', 'x'), positional=positional)
    longitude = _data(nc, 'longitude', ('y', 'x'), positional=positional)
    lon = np.where(longitude > 180., longitude-360., longitude)
    ox, oy = origin
    sw = from_utm(ox-margin, oy-margin, zone)
    ne = from_utm(ox+size[0]+margin, oy+size[1]+margin, zone)
    i, j = np.nonzero(((lon >= sw[0]) & (lon <= ne[0]) & (lat >= sw[1]) & (lat <= ne[1])).T)
    longitude = lon[j, i]
    x, y, zones = to_utm(longitude, lat[j, i], native_zone=positional)
    return j, i, longitude, x, y, zones


def _fields(nc, names, record, j, i, *, positional=False):
    y0, x0 = j.min(), i.min()
    selection = (record, slice(y0, j.max()+1), slice(x0, i.max()+1))
    for name in names:
        variable = _variable(nc, name, ('time', 'y', 'x'), positional=positional)
        selected = variable[selection][j-y0, i-x0]
        if np.ma.is_masked(selected) or not np.isfinite(selected).all():
            raise ValueError(f'HRRR field {name} contains missing or nonfinite values at selected sites')
        yield np.asarray(selected, dtype=float)


def _reciprocal(mode, speed, cloud, radiation, flux, ustar, temperature, *, native_math=False):
    if mode == 0:
        return np.zeros_like(speed)
    if mode == 2:
        heat_capacity, gravity = (float(np.float32(1006)*np.float32(1.2)), float(np.float32(.4)*np.float32(9.81))) if native_math else (1006*1.2, .4*9.81)
        denominator = heat_capacity*temperature*ustar**3
        if np.any((denominator == 0) & (flux != 0)):
            raise ValueError('Nonzero HRRR surface heat flux requires nonzero friction velocity and temperature')
        return np.divide(-gravity*flux, denominator, out=np.zeros_like(speed), where=denominator != 0)
    strong = np.select((speed < 3, speed < 5), (-.4, -.22), default=-.074)
    moderate = np.select((speed < 2, speed < 5, speed < 6), (-.4, -.22, -.074))
    slight = np.select((speed < 2, speed < 5), (-.22, -.074))
    night = np.where(cloud > 50, np.where(speed < 3, .018, 0.), np.select((speed < 3, speed < 5), (.047, .018)))
    return np.select((radiation > 700, radiation >= 350, radiation > 0), (strong, moderate, slight), default=night)


@dataclass(frozen=True)
class HRRRProfile:
    speed: float
    direction: float
    roughness: float
    reciprocal_length: float
    boundary_height: float
    pressure_wind: tuple
    kind = 5

    def evaluate(self, heights, altitude=0., *, native_math=False):
        if native_math:
            return self._evaluate_cpu(heights, altitude)
        array = np.asarray(heights)
        z = array.ravel()
        z0 = self.roughness
        if z0 <= 0:
            raise ValueError('HRRR profile requires positive surface roughness')
        reciprocal = self.reciprocal_length
        psi0 = stability_correction(z0*reciprocal)
        theta = np.deg2rad(270-self.direction)
        direction = np.array((np.cos(theta), np.sin(theta)))
        gain = 1/(np.log1p(10/z0)+stability_correction(10*reciprocal)+psi0)
        if abs(.4*gain) >= 2:
            return np.broadcast_to((self.speed*direction).reshape((2,)+(1,)*array.ndim), (2, *array.shape))
        profile = (self.speed*gain*(np.log1p(np.maximum(0., z)/z0)
                                    +stability_correction(np.maximum(0., z)*reciprocal)+psi0))[:, None]*direction
        if reciprocal > 0:
            top = self.boundary_height
            surface = max(10., .05*top)
            surface_values = profile[(z >= 0) & (z <= surface)]
            match = surface_values[-1] if .05*top >= 10 and len(surface_values) else self.speed*direction
            speed, direction_from = self.pressure_wind[np.searchsorted((762., 1450.), top+altitude, side='right')]
            # The native implementation converts pressure direction twice.
            angle = np.deg2rad(np.deg2rad(270-direction_from))
            upper = speed*np.array((np.cos(angle), np.sin(angle)))
            above = z > surface
            fraction = np.clip((z[above]-surface)/(top-surface), 0., 1.)
            profile[above] = match+(upper-match)*fraction[:, None]
        profile[z < 0] = 0.
        return profile.T.reshape((2, *array.shape))

    def _evaluate_cpu(self, heights, altitude):
        array = np.asarray(heights, dtype=np.float32)
        z = array.ravel()
        speed, direction, roughness, reciprocal, top = map(np.float32, (
            self.speed, self.direction, self.roughness, self.reciprocal_length, self.boundary_height))
        if roughness <= 0:
            raise ValueError('HRRR profile requires positive surface roughness')
        theta = float(np.float32((270.-float(direction))*np.pi/180.))
        components = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        vk, reference = np.float32(.4), np.float32(10)
        psi0 = _stability(roughness, reciprocal)
        ustar = speed*vk/(_log((reference+roughness)/roughness)+_stability(reference, reciprocal)+psi0)
        measured = components*speed
        valid = z >= 0
        result = np.zeros((len(z), 2), dtype=np.float32)
        if abs(ustar) >= np.float32(2)*speed:
            result[valid] = measured
        else:
            surface_layer = np.float32(.05)*top
            surface = max(reference, surface_layer)
            log_levels = valid & ((z <= surface) | (reciprocal <= 0))
            level = z[log_levels]
            stability = _stability(level, reciprocal, stable=level*reciprocal >= 0)
            log_profile = _log((level+roughness)/roughness)+stability+psi0
            result[log_levels] = log_profile[:, None]*(components*ustar/vk)
            if reciprocal > 0:
                pressure_index = np.searchsorted((762., 1450.), np.float32(altitude)+top, side='right')
                upper_speed, upper_direction = self.pressure_wind[pressure_index]
                # The two angle conversions and their double intermediates are in QES.
                angle = ((270.-upper_direction)*np.pi/180.)*np.pi/180.
                upper = upper_speed*np.array((np.cos(angle), np.sin(angle)))
                anchor = result[log_levels][-1] if surface_layer >= reference and np.any(log_levels) else measured
                transition = (z > surface) & (z <= top)
                result[transition] = (z[transition]-surface)[:, None]*(upper-anchor)/(top-surface)+anchor
                result[z > max(top, surface)] = upper
        return result.T.reshape((2, *array.shape))


@dataclass(frozen=True)
class HRRRStation(Station):
    map_position: tuple[float, float] | None = None
    map_index: int | None = None


@dataclass(frozen=True)
class HRRRInput:
    path: str
    scheme: int
    stability: int
    origin: tuple
    zone: int

    def __post_init__(self):
        if self.scheme not in (0, 1, 2) or self.stability not in (0, 1, 2):
            raise ValueError('HRRR interpolationScheme and stabilityClasses require 0, 1, or 2')
        if not (1 <= self.zone <= 60 and len(self.origin) == 2 and np.isfinite(self.origin).all()):
            raise ValueError('HRRR requires a finite UTM origin and zone')

    @classmethod
    def from_smoke_xml(cls, path, *, origin, zone, hrrr_path=None):
        """Read the historical HRRR smoke density configuration used by Yakima."""
        path = Path(path).resolve()
        root = read_qes_xml(path)
        source = root.find('sourceParameters')
        if root.tag != 'QESPlumeParameters' or source is None:
            raise ValueError('Expected sourceParameters in a QES plume XML file')
        if any(child.tag not in ('HRRRFile', 'inputFields') for child in source):
            raise ValueError('Expected only HRRRFile and inputFields in HRRR smoke sourceParameters')
        fields = [field for node in source.findall('inputFields') for field in (node.text or '').split()]
        if fields and fields != [_SMOKE_DENSITY]:
            raise ValueError(f'HRRR smoke inputFields must specify {_SMOKE_DENSITY}')
        filename = source.findtext('HRRRFile', '').strip()
        if not filename:
            raise ValueError('HRRRFile is required in HRRR smoke sourceParameters')
        data_path = path.parent/filename if hrrr_path is None else Path(hrrr_path).resolve()
        return cls(str(data_path), 0, 0, tuple(origin), zone)

    def stations(self, domain, halo, step_limit, *, native_math=False, stored=False):
        nx, ny, _ = domain.cells
        dx = domain.spacing[0]
        ox, oy = self.origin
        nearest = self.scheme == 1
        with _open_netcdf(self.path) as nc:
            if stored:
                nc.set_auto_scale(False)
            j, i, lon, x, y, zones = _locations(nc, self.origin, self.zone, (nx*dx, ny*dx), 6000., positional=stored)
            if native_math:
                x, y = x.astype(float), y.astype(float)
            local_x = np.where(zones == self.zone, x-ox, x+(zones-self.zone-1)*458830.+458830.-ox)
            local_y = y-oy
            local = np.column_stack((local_x, local_y))
            if self.scheme != 2:
                if stored or nearest:
                    # QES compares double station UTM coordinates against
                    # the upper limits formed by float domain arithmetic.
                    east, north = np.asarray(self.origin, dtype=np.float32)+np.asarray((nx, ny), dtype=np.float32)*np.float32(dx)
                else:
                    east, north = ox+nx*dx, oy+ny*dx
                inside = (x >= ox) & (x <= east) & (y >= oy) & (y <= north)
            else:
                inside = np.ones(len(x), dtype=bool)
            grid_ids = tuple(zip(j[inside].tolist(), i[inside].tolist()))
            if not grid_ids:
                raise ValueError('No usable HRRR stations within the QES domain')
            times = _data(nc, 'time', ('time',), positional=stored)
            if not times.size:
                raise ValueError('HRRR file contains no time records')
            # QES converts HRRR times to time_t before matching sensors.
            # Keep repeated output times and use the last matching record.
            seconds = tuple(map(int, times))
            latest = {second: record for record, second in enumerate(seconds)}
            schedule = sorted(seconds)[:step_limit or None]
            selected = {latest[second]: second for second in schedule}
            first = None
            profiles_by_time = {}
            for record in dict.fromkeys((0, *selected)):
                fields = _fields(nc, (*_SURFACE, *_ALOFT), record, j, i, positional=stored)
                u, v, z0, cloud, radiation, flux, ustar, temperature, pbl, *aloft = fields
                speed, direction = _wind(u, v, lon, native_math=native_math)
                if native_math:
                    speed = speed.astype(np.float32)
                reciprocal = _reciprocal(self.stability, speed, cloud, radiation, flux, ustar, temperature, native_math=native_math)
                pbl = np.maximum(pbl, np.select((reciprocal > 0, reciprocal < 0), (200., 1000.), default=100.))
                pressure = [_wind(a, b, lon, native_math=native_math) for a, b in zip(aloft[::2], aloft[1::2])]
                profiles = tuple(HRRRStation(*halo.translate(point), HRRRProfile(
                    float(speed[k]), float(direction[k]), float(z0[k]), float(reciprocal[k]), float(pbl[k]),
                    tuple((float(s[k]), float(d[k])) for s, d in pressure)),
                    tuple(map(float, point)) if self.scheme in (1, 2) else None,
                    k if nearest else None)
                    for k, point in enumerate(local))
                if first is None:
                    first = profiles
                if record in selected:
                    second = selected[record]
                    timestamp = (datetime(1970, 1, 1)+timedelta(seconds=second)).isoformat(timespec='seconds')
                    profiles_by_time[second] = (timestamp, tuple(p for p, keep in zip(profiles, inside) if keep))
        return tuple(profiles_by_time[second] for second in schedule), first, grid_ids

    def smoke_sources(self, domain, record, *, stored=False):
        """Return in-domain smoke densities. Stored mode uses native NetCDF reads."""
        nx, ny, _ = domain.cells
        dx = domain.spacing[0]
        with _open_netcdf(self.path) as nc:
            if stored:
                nc.set_auto_scale(False)
            record = index(record)
            if 'time' not in nc.dimensions or not 0 <= record < len(nc.dimensions['time']):
                raise ValueError('HRRR smoke record is outside the time axis')
            j, i, _, x, y, zones = _locations(nc, self.origin, self.zone, (nx*dx, ny*dx), 0., positional=stored)
            density, = (_fields(nc, (_SMOKE_DENSITY,), record, j, i, positional=stored) if len(j)
                        else (np.empty(0),))
            if stored:
                density = density.astype(np.float32)
        return dict(y_index=j, x_index=i, utm_x=x, utm_y=y, utm_zone=zones, mass_density=density)
