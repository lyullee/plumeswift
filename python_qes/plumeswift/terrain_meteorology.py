"""Native terrain-following station shifts and two-pass Barnes interpolation.

Port of WindProfilerSensorType/BarnCPU; model equations are unchanged. Undefined
native top-center access in Barnes is handled by extending the coordinate axis,
as for staggered interpolation, and recorded as a numerical boundary correction.
"""
import numpy as np

from .domain import native_vertical_coordinates
from .native_sensor import barnes_weights, evaluate_sensor


def _vertical_coordinates(workspace, native_math):
    c = workspace.coordinates
    if not native_math:
        return c['z'], c['z_face']
    return native_vertical_coordinates(workspace.arrays['dz_array'])


def station_profiles(workspace, stations, *, native_math=False):
    a, c = workspace.arrays, workspace.coordinates
    nz, ny, nx = a['u0'].shape; dx = c['x'][1]-c['x'][0]; dy = c['y'][1]-c['y'][0]
    scalar = np.float32 if native_math else float
    if native_math:
        dx = dy = np.float32(dx)
    z, z_face = _vertical_coordinates(workspace, native_math)
    ground = a.get('terrain_face_id')
    profiles = np.zeros((len(stations), nz, 2), dtype=scalar); starts = []; abl = []
    for number, station in enumerate(stations):
        i, j = int(scalar(station.x)/dx), int(scalar(station.y)/dy)
        if not 0 <= i < nx or not 0 <= j < ny:
            raise ValueError('Terrain sensor lies outside the allocated domain')
        start = 1 if ground is None else int(ground[j, i])
        starts.append(start)
        heights = z[start:]-z_face[start]
        if native_math:
            if station.profile.kind == 0:
                profiles[number, start:] = evaluate_sensor(station.profile, [0])[0]
            else:
                profiles[number, start:-1] = evaluate_sensor(station.profile, heights)
        else:
            profiles[number, start:-1] = np.stack(station.profile.evaluate(heights), axis=-1)
            if station.profile.kind == 0:
                theta = np.deg2rad(270-station.profile.direction)
                profiles[number, -1] = station.profile.speed*np.array([np.cos(theta), np.sin(theta)])
        reciprocal = scalar(station.profile.reciprocal_length)
        abl.append(200. if reciprocal > 0 else 1000. if reciprocal < 0 else 100.)
    return profiles, np.array(starts), np.array(abl, dtype=scalar)


def single_sensor_terrain(workspace, station, *, sampled=None, native_math=False):
    a = workspace.arrays
    profile, starts, abl = station_profiles(workspace, (station,), native_math=native_math) if sampled is None else sampled
    nz = a['u0'].shape[0]; start = starts[0]; target_start = a['terrain_face_id']
    z, _ = _vertical_coordinates(workspace, native_math)
    difference = z[target_start]-z[start]
    surface_height = .05*np.where(abs(difference) > abl[0], abl[0], 2*abl[0]-difference)
    a['u0'].fill(0); a['v0'].fill(0); a['w0'].fill(0)
    jj, ii = np.indices(target_start.shape)
    for k in range(1, nz-1):
        target = k+target_start-1; index = k+start-1
        valid = target < nz
        first = z[k] <= surface_height
        chosen = np.where(first, index, min(index, nz-2))
        if np.any(valid & (chosen >= nz)):
            raise FloatingPointError('Native single-sensor profile index exceeds its allocation')
        selected = (target[valid], jj[valid], ii[valid])
        a['u0'][selected] = profile[0, chosen[valid], 0]
        a['v0'][selected] = profile[0, chosen[valid], 1]


def _weighted_levels(weights, values, denominator, native_math):
    products = weights[..., None]*values
    if native_math:
        return np.add.accumulate(products, axis=0)[-1]/denominator[..., None]
    return products.sum(axis=0)/denominator[..., None]


def barnes_terrain(workspace, stations, *, variant='native-cpu', sampled=None, native_math=False):
    if variant not in ('native-cpu', 'native-gpu'):
        raise ValueError('Unknown native Barnes equation variant')
    stations = tuple(stations)
    if len(stations) == 1:
        return single_sensor_terrain(workspace, stations[0], sampled=sampled, native_math=native_math)
    if not stations:
        raise ValueError('Barnes interpolation needs stations')
    a, c = workspace.arrays, workspace.coordinates
    nz, ny, nx = a['u0'].shape
    dx = c['x'][1]-c['x'][0]; dy = dx  # pinned getDomainSize dy alias
    scalar = np.float32 if native_math else float
    dx = dy = scalar(dx)
    x, y = (((np.arange(length)-.5)*float(dx)).astype(scalar) for length in (nx, ny))
    positions = np.array([[s.x, s.y] for s in stations], dtype=scalar)
    if not np.isfinite(positions).all() or (not native_math and sampled is None and
            (np.any(positions <= 0) or np.any(positions >= [x[-1], y[-1]]))):
        raise ValueError('Barnes stations must be inside the interpolable grid')
    if native_math:
        weights = barnes_weights(positions, x, y)
        denominator = np.add.accumulate(weights, axis=0)[-1]
    else:
        distances = np.linalg.norm(positions[:, None]-positions[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances == 0): raise ValueError('Coincident Barnes stations')
        dn = np.minimum(distances.min(axis=1), 1e6).mean()
        length = 5.052*(2*dn/np.pi)**2
        xx, yy = np.meshgrid(x, y)
        weights = np.exp(-((positions[:, 0, None, None]-xx)**2+(positions[:, 1, None, None]-yy)**2)/length)
        weights[:, weights.sum(axis=0) == 0] = 1e-20
        denominator = weights.sum(axis=0)
    profiles, starts, abl = station_profiles(workspace, stations, native_math=native_math) if sampled is None else sampled
    target_start = a['terrain_face_id']; jj, ii = np.indices(target_start.shape)
    # Numerical coordinate extension for the undefined native top-center read.
    z, z_face = _vertical_coordinates(workspace, native_math)
    z_extended = np.r_[z, z[-1]+scalar(a['dz_array'][-1])]
    a['u0'].fill(0); a['v0'].fill(0); a['w0'].fill(0)
    mappings = []
    for k in range(nz-1):
        target = k+target_start; valid = target < nz
        indices = []
        for start, height in zip(starts, abl):
            difference = z[target_start]-z[start]
            correction = difference if variant == 'native-gpu' else abs(difference)
            surface_height = .05*np.where(abs(difference) > height, height, 2*height-correction)
            if k+start > nz-2:
                chosen = np.full(target.shape, nz-2, dtype=int)
            else:
                level = z[k] if variant == 'native-gpu' else z_extended[np.minimum(target, nz-1)]
                above = level-z_face[target_start] > surface_height
                chosen = np.where(above & (target > k+start), target, k+start)
            indices.append(np.minimum(chosen, nz-1))
        indices = np.array(indices); mappings.append((target, valid, indices))
        values = profiles[np.arange(len(stations))[:, None, None], indices]
        field = _weighted_levels(weights, values, denominator, native_math)
        selected = (target[valid], jj[valid], ii[valid])
        a['u0'][selected] = field[..., 0][valid]; a['v0'][selected] = field[..., 1][valid]
    interpolated = np.zeros_like(profiles)
    for number, (sx, sy) in enumerate(positions):
        inside = 0 < sx < scalar(nx-1)*dx and 0 < sy < scalar(ny-1)*dy
        if native_math and not inside:
            # Original outlying-station branch writes the top fallback at k,
            # while the ordinary levels are stored at k + terrain_face_id.
            start = starts[number]
            for k in range(nz):
                level = k+start
                if level > nz-2:
                    interpolated[number, k] = profiles[number, -2]
                else:
                    interpolated[number, level] = profiles[number, level]
            continue
        if sampled is not None and not inside:
            interpolated[number] = profiles[number]
            interpolated[number, -1] = profiles[number, -2]
            continue
        j = np.clip(np.searchsorted(y, sy, side='left')-1, 0, ny-2)
        # Keep the native packed east neighbor where all four samples stay
        # in the face plane. The final northeast footprint has no such neighbor.
        last_i = nx-1 if native_math and j < ny-2 else nx-2
        i = np.clip(np.searchsorted(x, sx, side='left')-1, 0, last_i)
        wx, wy = (sx-x[i])/dx, (sy-y[j])/dy
        start = target_start[j, i]
        column = j*nx+i
        for component, name in enumerate(['u0', 'v0']):
            neighbors = [column, column+1, column+nx, column+nx+1]
            values = a[name].reshape(nz, ny*nx)[start:, neighbors].astype(scalar, copy=False)
            upper = (1-wx)*values[:, 2]+wx*values[:, 3]
            lower = (1-wx)*values[:, 0]+wx*values[:, 1]
            interpolated[number, start:, component] = wy*upper+(1-wy)*lower
    residual = profiles-interpolated
    for target, valid, indices in mappings:
        values = residual[np.arange(len(stations))[:, None, None], indices]
        correction = _weighted_levels(weights, values, denominator, native_math)
        selected = (target[valid], jj[valid], ii[valid])
        for component, name in enumerate(('u0', 'v0')):
            a[name][selected] = a[name][selected].astype(scalar, copy=False)+correction[..., component][valid]
