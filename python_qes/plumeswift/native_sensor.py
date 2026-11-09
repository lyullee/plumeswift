"""Mixed-precision QES CPU profiles and Barnes weights (GPL-3.0)."""
from dataclasses import replace

import numpy as np

from .contracts import UnportedFeature
from .domain import native_vertical_coordinates
from .meteorology import canopy_displacement
from .numerics.linux_float_angles import atanf, sincosf


def _stability(height, reciprocal, *, stable=None):
    height64 = np.asarray(height, dtype=np.float64)
    if stable is None:
        stable = reciprocal >= 0
    if np.all(stable):
        return (4.7*height64*float(reciprocal)).astype(np.float32)
    unstable_height = np.where(stable, 0., height64)
    x32 = np.power(1.-15.*unstable_height*float(reciprocal), .25).astype(np.float32)
    x = x32.astype(float)
    atan = atanf(x32).astype(float)
    unstable = -2.*np.log(.5*(1.+x))-np.log(.5*(1.+x*x))+2.*atan-.5*np.pi
    return np.where(stable, 4.7*height64*float(reciprocal), unstable).astype(np.float32)


def _log(value):
    return np.log(np.asarray(value, dtype=np.float64)).astype(np.float32)


def _canopy_parameters(profile, speed, ref, roughness, reciprocal):
    top, attenuation = map(np.float32, (profile.canopy_height, profile.canopy_attenuation))
    if not np.isfinite([top, attenuation]).all() or not 0 < roughness < top or speed <= 0:
        raise ValueError('Native canopy profile requires positive speed and 0 < roughness < canopy height')
    vk = np.float32(.4)
    stable_ref = ref*reciprocal > 0
    psi = _stability(ref, reciprocal, stable=stable_ref)
    ustar = speed*vk/(_log(ref/roughness)+psi)
    if not np.isfinite(ustar) or ustar == 0:
        raise ValueError('Native canopy friction velocity is singular')
    displacement = canopy_displacement(ustar, roughness, top, attenuation, vk, psi, native_math=True)
    gap = top-displacement
    if gap <= 0:
        raise ValueError('Native canopy displacement reaches the canopy top')
    canopy_wind = (ustar/vk)*(_log(gap/roughness)+_stability(gap, reciprocal, stable=top*reciprocal > 0))
    if ref < top:
        scale = np.float32(float(speed)/(float(canopy_wind)*np.exp(float(attenuation*(ref/top))-1.)))
    else:
        gap = ref-displacement
        scale = speed/((ustar/vk)*(_log(gap/roughness)+_stability(gap, reciprocal, stable=stable_ref)))
    if not np.isfinite(scale):
        raise ValueError('Native canopy reference wind is singular')
    return top, attenuation, displacement, ustar*scale, canopy_wind*scale, scale


def advance_canopy_station(station):
    """Replay the source's in-place reference speed update for a repeated record."""
    profile = station.profile
    if profile.kind != 3:
        return station
    inputs = map(np.float32, (profile.speed, profile.reference_height, profile.roughness, profile.reciprocal_length))
    *_, normalized_speed = _canopy_parameters(profile, *inputs)
    return replace(station, profile=replace(profile, speed=float(normalized_speed)))


def _canopy_profile(profile, z, components, speed, ref, roughness, reciprocal):
    top, attenuation, displacement, ustar, canopy_wind, _ = _canopy_parameters(profile, speed, ref, roughness, reciprocal)
    below, above = z < top, z > top
    result = np.zeros(z.shape+(2,), dtype=np.float32)
    exponent = float(attenuation)*((z[below]/top).astype(float)-1.)
    result[below] = np.exp(exponent)[..., None]*(components*canopy_wind)
    gap = z[above]-displacement
    stability = _stability(gap, reciprocal, stable=z[above]*reciprocal > 0)
    result[above] = (_log(gap/roughness)+stability)[..., None]*(components*ustar/np.float32(.4))
    return result


def evaluate_sensor(profile, heights):
    """Evaluate an upstream station profile at float32 heights above ground."""
    if profile.kind == 4:
        return np.column_stack(profile.evaluate(heights, native_math=True))
    if profile.kind not in (0, 1, 2, 3):
        raise UnportedFeature('Native sensor arithmetic supports station profile flags 0 through 4')
    speed, direction, ref, roughness, reciprocal = map(np.float32, (
        profile.speed, profile.direction, profile.reference_height, profile.roughness, profile.reciprocal_length))
    z = np.asarray(heights, dtype=np.float32)
    if not np.isfinite([speed, direction, ref, roughness, reciprocal]).all() or speed < 0 or np.any(z < 0):
        raise ValueError('Invalid sensor measurement')
    angle = np.float32((270.-float(direction))*np.pi/180.)
    if abs(angle) < 120:
        sine, cosine = sincosf(angle)
    else:
        # Preserve the existing large-angle path outside the native reducer's range.
        sine, cosine = np.sin(float(angle)), np.cos(float(angle))
    components = np.array([cosine, sine], dtype=np.float32)
    uniform = components*speed
    if profile.kind == 0:
        return np.broadcast_to(uniform, z.shape+(2,))
    if profile.kind == 1 and ref == 0 and roughness > 0 and speed > 0:
        # The source's zero log denominator gives infinite friction velocity,
        # and its free-convection limit then uses the uniform reference wind.
        return np.broadcast_to(uniform, z.shape+(2,))
    if ref <= 0:
        raise ValueError('Native profile requires a positive reference height')
    if profile.kind == 2:
        scale = np.power((z/ref).astype(float), float(roughness)).astype(np.float32)
        return scale[..., None]*uniform
    if profile.kind == 3:
        return _canopy_profile(profile, z, components, speed, ref, roughness, reciprocal)
    if roughness <= 0:
        raise ValueError('Native log profile requires positive roughness')
    vk = np.float32(.4)
    reference_log = np.float32(np.log(float((ref+roughness)/roughness)))
    ustar = speed*vk/(reference_log+_stability(ref, reciprocal))
    if abs(ustar) >= np.float32(2)*speed:
        return np.broadcast_to(uniform, z.shape+(2,))
    logarithm = np.log(((z+roughness)/roughness).astype(float)).astype(np.float32)
    return (logarithm+_stability(z, reciprocal))[..., None]*(components*ustar/vk)


def barnes_scale(positions):
    """Original float32 station spacing and Barnes length scale."""
    delta = positions[:, None]-positions[None, :]
    distance = np.sqrt((delta.astype(float)**2).sum(axis=2)).astype(np.float32)
    np.fill_diagonal(distance, np.inf)
    nearest = np.minimum(distance.min(axis=1), np.float32(1e6))
    mean = np.add.accumulate(nearest)[-1]/np.float32(len(positions))
    if mean == 0:
        raise ValueError('Zero mean nearest-station distance makes the native Barnes length scale singular')
    return np.float32(5.052*(2.*float(mean)/np.pi)**2)


def barnes_weights(positions, x, y):
    """Calculate the shared float32 CPU Barnes weights in station order."""
    scale = barnes_scale(positions)
    xdiff = (positions[:, 0, None, None]-x).astype(float)
    ydiff = (positions[:, 1, None, None]-y[:, None]).astype(float)
    reciprocal = float(np.float32(1)/scale)
    weights = np.exp(-reciprocal*xdiff**2-reciprocal*ydiff**2).astype(np.float32)
    zero = ~np.any(weights, axis=0)
    weights[:, zero] = np.float32(1e-20)
    return weights


def _weighted_sum(weights, profiles):
    total = np.zeros((len(profiles[0]), *weights.shape[1:], 2), dtype=np.float32)
    denominator = np.zeros(weights.shape[1:], dtype=np.float32)
    for weight, profile in zip(weights, profiles, strict=True):
        total += weight[None, :, :, None]*profile[:, None, None, :]
        denominator += weight
    return total/denominator[None, :, :, None]


def initialize_flat(workspace, stations, profile):
    """Initialize flat single-station or two-pass Barnes winds in native order."""
    a, c = workspace.arrays, workspace.coordinates
    if np.any(a['terrain'] != 0):
        raise UnportedFeature('Native float32 sensor initialization requires flat terrain')
    for key in ('u0', 'v0', 'w0'):
        a[key].fill(0)
    z, _ = native_vertical_coordinates(a['dz_array'])
    heights = z[1:]
    if len(stations) < 2:
        chosen = stations[0].profile if stations else profile
        samples = evaluate_sensor(chosen, heights)
        a['u0'][1:-1], a['v0'][1:-1] = (component[:, None, None] for component in samples.T)
        return
    _, ny, nx = a['u0'].shape
    dx = np.float32(c['x'][1]-c['x'][0])
    x, y = (((np.arange(length)-.5)*float(dx)).astype(np.float32) for length in (nx, ny))
    positions = np.asarray([(station.x, station.y) for station in stations], dtype=np.float32)
    limits = np.array([nx-1, ny-1], dtype=np.float32)*dx
    site_cells = np.trunc(positions/dx)
    if not np.isfinite(site_cells).all() or np.any(site_cells < 0) or np.any(site_cells >= [nx, ny]):
        raise ValueError('Native float32 Barnes stations must reference allocated grid cells')
    interior = np.all((positions > 0) & (positions < limits), axis=1)
    profiles = np.stack([evaluate_sensor(station.profile, heights) for station in stations])
    # A flat grid uses the final physical profile level for the extra top face.
    profiles = np.concatenate((profiles, profiles[:, -1:]), axis=1)
    weights = barnes_weights(positions, x, y)
    first = _weighted_sum(weights, profiles)
    packed = first.reshape(len(first), ny*nx, 2)
    # For edge stations the native CPU substitutes the station profile for
    # the interpolated field, so the flat-grid residual is zero.
    residual = np.zeros_like(profiles)
    for index in np.flatnonzero(interior):
        sx, sy = positions[index]
        i, j = np.searchsorted(x, sx)-1, np.searchsorted(y, sy)-1
        wx, wy = (sx-x[i])/dx, (sy-y[j])/dx
        # Native east-edge samples can cross into the next packed row.
        column = j*nx+i
        upper = (1-wx)*packed[:, column+nx]+wx*packed[:, column+nx+1]
        lower = (1-wx)*packed[:, column]+wx*packed[:, column+1]
        residual[index] = profiles[index]-(wy*upper+(1-wy)*lower)
        residual[index, -1] = residual[index, -2]
    corrected = first+_weighted_sum(weights, residual)
    a['u0'][1:], a['v0'][1:] = corrected[..., 0], corrected[..., 1]
