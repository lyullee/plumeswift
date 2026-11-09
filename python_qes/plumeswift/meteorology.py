"""QES sensor profiles. FROM directions are degrees clockwise from north.

Ported equations: WindProfilerSensorType.cpp / WINDSGeneralData.cpp,
GPL-3.0-only. No generic wake or CFD model is substituted.
"""
from dataclasses import dataclass
import numpy as np


def stability_correction(z_over_l):
    t = np.asarray(z_over_l, dtype=float)
    x = np.power(1-15*np.minimum(t, 0), .25)
    return np.where(t >= 0, 4.7*t, -2*np.log((1+x)/2)-np.log((1+x*x)/2)+2*np.arctan(x)-np.pi/2)


def canopy_displacement(ustar, z0, canopy_top, attenuation, vk=.4, psi=0., *, native_math=False):
    if attenuation <= 0:
        return np.float32(.99*float(canopy_top)) if native_math else .99*canopy_top
    if ustar == 0 or not 0 < z0 < canopy_top:
        raise ValueError("Native canopy bisection requires nonzero ustar and 0 < z0 < canopy height")
    if native_math:
        ustar, z0, canopy_top, attenuation, vk, psi = map(np.float32, (ustar, z0, canopy_top, attenuation, vk, psi))
    scalar = np.float32 if native_math else float

    def residual(displacement):
        gap = canopy_top-displacement
        logarithm = scalar(np.log(float(gap/z0)))
        uhc = ustar/vk*(logarithm+psi)
        return attenuation*uhc*vk/ustar-canopy_top/gap

    tolerance = z0/100
    d1, d2 = z0, canopy_top
    d = .5*(d1+d2)
    fi = residual(d1)
    fnew = tolerance*10
    for _ in range(200):
        if abs(fnew) <= tolerance or not z0 < d < canopy_top:
            break
        d = .5*(d1+d2)
        if native_math and d == canopy_top:
            break  # The source midpoint has rounded onto the singular endpoint.
        fnew = residual(d)
        if fnew*fi > 0:
            d1 = d
        elif fnew*fi < 0:
            d2 = d
    return 10000. if d > canopy_top else d


@dataclass(frozen=True)
class SensorProfile:
    speed: float
    direction: float
    reference_height: float
    roughness: float = .02
    reciprocal_length: float = 0.
    kind: int = 1
    canopy_height: float = 0.
    canopy_attenuation: float = 0.

    def evaluate(self, heights):
        z = np.asarray(heights, dtype=float)
        if not np.isfinite([self.speed, self.direction, self.reference_height, self.roughness, self.reciprocal_length]).all() or self.speed < 0:
            raise ValueError("Invalid sensor measurement")
        if np.any(z < 0):
            raise ValueError("Profile heights must be above terrain")
        if self.kind == 0:
            magnitude = np.full(z.shape, self.speed)
        elif self.kind == 1:
            if self.roughness <= 0 or self.reference_height <= 0:
                raise ValueError("Log profile requires positive z0 and reference height")
            psi_ref = stability_correction(self.reference_height*self.reciprocal_length)
            ustar = self.speed*.4/(np.log((self.reference_height+self.roughness)/self.roughness)+psi_ref)
            magnitude = ustar/.4*(np.log((z+self.roughness)/self.roughness)+stability_correction(z*self.reciprocal_length))
            if abs(ustar) >= 2*self.speed:
                magnitude = np.full(z.shape, self.speed)
        elif self.kind == 2:
            if self.reference_height <= 0:
                raise ValueError("Power profile requires a positive reference height")
            magnitude = self.speed*(z/self.reference_height)**self.roughness
        elif self.kind == 3:
            h, atten, z0 = self.canopy_height, self.canopy_attenuation, self.roughness
            if not 0 < z0 < h or self.reference_height <= 0 or self.speed == 0:
                raise ValueError('Canopy profile requires positive speed, reference height, and 0 < z0 < canopy height')
            psi_ref = float(stability_correction(self.reference_height*self.reciprocal_length))
            ustar = self.speed*.4/(np.log(self.reference_height/z0)+psi_ref)
            displacement = canopy_displacement(ustar, z0, h, atten, psi=psi_ref)
            psi_h = stability_correction((h-displacement)*self.reciprocal_length)
            uh = ustar/.4*(np.log((h-displacement)/z0)+psi_h)
            if self.reference_height < h:
                # Native placement of -1 is OUTSIDE the attenuation product.
                scale = self.speed/(uh*np.exp(atten*(self.reference_height/h)-1))
            else:
                psi_ref = stability_correction((self.reference_height-displacement)*self.reciprocal_length)
                scale = self.speed/(ustar/.4*(np.log((self.reference_height-displacement)/z0)+psi_ref))
            ustar *= scale; uh *= scale
            magnitude = np.zeros(z.shape)
            below, above = z < h, z > h
            magnitude[below] = uh*np.exp(atten*(z[below]/h-1))
            magnitude[above] = ustar/.4*(np.log((z[above]-displacement)/z0)+stability_correction((z[above]-displacement)*self.reciprocal_length))
        else:
            raise NotImplementedError(f"Sensor profile flag {self.kind} is not implemented by this component")
        theta = np.deg2rad(270-self.direction)
        return magnitude*np.cos(theta), magnitude*np.sin(theta)


def single_sensor_flat(workspace, profile):
    """Single-station branch on flat terrain, including native top/ghost values."""
    if np.any(workspace.arrays['terrain'] != 0):
        raise NotImplementedError("Terrain-dependent single-station shifting is separate")
    a, c = workspace.arrays, workspace.coordinates
    a['u0'].fill(0); a['v0'].fill(0); a['w0'].fill(0)
    u, v = profile.evaluate(c['z'][1:])
    a['u0'][1:-1] = u[:, None, None]
    a['v0'][1:-1] = v[:, None, None]


@dataclass(frozen=True)
class Station:
    x: float
    y: float
    profile: SensorProfile


def barnes_flat(workspace, stations, *, native_dy_alias=True):
    """Original two-pass Barnes implementation, restricted to flat terrain.

    Upstream computes narrower gamma=.2 weights but uses the FIRST weights
    for its residual pass. This routine preserves that behavior; silently
    switching to textbook Barnes weights would change the physical input.
    """
    stations = tuple(stations)
    if len(stations) == 1:
        return single_sensor_flat(workspace, stations[0].profile)
    if len(stations) < 2:
        raise ValueError('Barnes interpolation needs stations')
    a, c = workspace.arrays, workspace.coordinates
    if np.any(a['terrain'] != 0):
        raise NotImplementedError('Terrain-dependent Barnes profile shifting is separate')
    nz, ny, nx = a['u0'].shape
    dx = c['x'][1]-c['x'][0]; dy = dx if native_dy_alias else c['y'][1]-c['y'][0]
    positions = np.array([[s.x, s.y] for s in stations])
    if not np.isfinite(positions).all() or np.any(positions <= 0) or np.any(positions >= [(nx-1.5)*dx, (ny-1.5)*dy]):
        raise ValueError('This Barnes component requires stations inside the interpolable grid')
    pair_distances = np.linalg.norm(positions[:, None]-positions[None], axis=-1)
    np.fill_diagonal(pair_distances, np.inf)
    if np.any(pair_distances == 0):
        raise ValueError('Coincident stations make native Barnes length scale singular')
    nearest = np.minimum(pair_distances.min(axis=1), 1e6)
    length_scale = 5.052*(2*nearest.mean()/np.pi)**2
    x, y = (np.arange(nx)-.5)*dx, (np.arange(ny)-.5)*dy
    xx, yy = np.meshgrid(x, y)
    squared_distance = (positions[:, 0, None, None]-xx)**2+(positions[:, 1, None, None]-yy)**2
    weights = np.exp(-squared_distance/length_scale)
    zero = weights.sum(axis=0) == 0
    weights[:, zero] = 1e-20
    weights /= weights.sum(axis=0)
    profiles = np.zeros((len(stations), nz, 2))
    for index, station in enumerate(stations):
        profiles[index, 1:-1] = np.stack(station.profile.evaluate(c['z'][1:]), axis=-1)
        if station.profile.kind == 0:
            profiles[index, -1] = profiles[index, -2]
    target = np.minimum(np.arange(1, nz), nz-2)
    first = np.einsum('sji,skv->kjiv', weights, profiles[:, target], optimize=True)
    interpolated = np.zeros_like(profiles)
    for index, (sx, sy) in enumerate(positions):
        i = np.searchsorted(x, sx, side='left')-1
        j = np.searchsorted(y, sy, side='left')-1
        wx, wy = (sx-x[i])/dx, (sy-y[j])/dy
        interpolated[index, 1:] = ((1-wy)*((1-wx)*first[:, j, i]+wx*first[:, j, i+1])+
                                  wy*((1-wx)*first[:, j+1, i]+wx*first[:, j+1, i+1]))
    residual = profiles[:, target]-interpolated[:, target]
    corrected = first+np.einsum('sji,skv->kjiv', weights, residual, optimize=True)
    a['u0'].fill(0); a['v0'].fill(0); a['w0'].fill(0)
    a['u0'][1:] = corrected[..., 0]; a['v0'][1:] = corrected[..., 1]
