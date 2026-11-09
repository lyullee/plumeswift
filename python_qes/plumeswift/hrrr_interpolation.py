"""Terrain-following HRRR interpolation on the staggered QES grid."""
import numpy as np

from .domain import native_vertical_coordinates
from .terrain_meteorology import barnes_terrain


def initialize_hrrr(workspace, stations, grid_ids, scheme, *, variant='native-cpu', datum=0., native_math=False,
                    barnes_backend='cpu', nearest_map_positions=(), legacy_terrain_stride=False):
    gpu_barnes = scheme == 0 and barnes_backend == 'cuda'
    native_math = native_math or gpu_barnes
    a, c = workspace.arrays, workspace.coordinates
    nz, ny, nx = a['u0'].shape
    scalar = np.float32 if native_math else float
    dx, dy = scalar(c['x'][1]-c['x'][0]), scalar(c['y'][1]-c['y'][0])
    positions = np.array([(station.x, station.y) for station in stations], dtype=scalar)
    z, z_face = native_vertical_coordinates(a['dz_array']) if native_math else (c['z'], c['z_face'])
    terrain_id = a.get('terrain_face_id', np.ones((ny, nx), dtype=int))
    a['terrain_face_id'] = terrain_id
    indices = np.trunc(positions/dx).astype(int)
    ix, iy = np.clip(indices, 0, (nx-2, ny-2)).T
    starts = terrain_id[iy, ix]
    terrain = a['terrain']
    elevations = terrain[iy, ix]
    if legacy_terrain_stride:
        # The source uses its face index on cell terrain for pressure-level
        # selection. Preserve defined reads and guard indices beyond allocation.
        packed = iy*nx+ix
        valid = packed < terrain.size
        elevations[valid] = terrain.ravel()[packed[valid]]
    profiles = np.zeros((len(stations), nz, 2), dtype=scalar)
    heights = np.array([s.profile.boundary_height for s in stations], dtype=scalar)
    for n, station in enumerate(stations):
        start = starts[n]
        profiles[n, start:-1] = np.stack(
            station.profile.evaluate(z[start:]-z_face[start], scalar(datum)+scalar(elevations[n]), native_math=native_math), axis=-1)
    if scheme == 0:
        sampled = profiles, starts, heights
        if gpu_barnes and len(stations) > 1:
            from .cuda_barnes import barnes_cuda
            return barnes_cuda(workspace, stations, sampled=sampled)
        return barnes_terrain(workspace, stations, variant=variant, sampled=sampled, native_math=native_math)

    jj, ii = np.indices((ny, nx))
    # QES builds both site maps before applying the halo to stations. The
    # vertical profiles and bilinear weights use the translated positions.
    map_positions = np.array([getattr(station, 'map_position', None) or (station.x, station.y)
                              for station in stations], dtype=scalar)
    if scheme == 1:
        available = None
        if nearest_map_positions:
            map_positions = np.asarray(nearest_map_positions, dtype=np.float32)
            available = np.zeros(len(map_positions), dtype=int)
            available[[station.map_index for station in stations]] = np.arange(len(stations))
        ids = _nearest_sites(map_positions, ii, jj, dx, dy, native_math=native_math, available=available)[None]
        covered = np.ones((ny, nx), dtype=bool)
    else:
        ids, weights, covered = _quadrilaterals(grid_ids, positions, ii, jj, dx, dy, native_math=native_math,
                                                map_positions=map_positions)
    source_start = starts[ids]
    abl = heights[ids]
    target_center = np.r_[z, z[-1]+scalar(a['dz_array'][-1])]
    source_z = z_face if scheme == 2 else target_center
    difference = abs(target_center[terrain_id]-source_z[source_start])
    surface_height = .05*np.where(difference > abl, abl, 2*abl-difference)
    for name in ('u0', 'v0', 'w0'):
        a[name].fill(0.)
    for k in range(nz-1):
        target = k+terrain_id
        valid = (target < nz) & covered
        level = target_center[np.minimum(target, nz-1)]-z_face[terrain_id]
        reference = k+source_start
        chosen = np.where((level > surface_height) & (target > reference), target, reference)
        # The highest stored profile sample is at nz-2. An elevated
        # target column can reach its top face before the source does.
        chosen = np.minimum(chosen, nz-2)
        values = profiles[ids, chosen]
        if scheme == 1:
            field = values[0]
        elif native_math:
            field = np.zeros((ny, nx, 2), dtype=np.float32)
            field[covered] = _bilinear_cpu(values[:, covered], weights[:, covered])
        else:
            field = np.sum(weights[..., None]*values, axis=0)
        selection = target[valid], jj[valid], ii[valid]
        a['u0'][selection] = field[valid, 0]
        a['v0'][selection] = field[valid, 1]
    if scheme == 2:
        for name in ('u0', 'v0'):
            layer = a[name][1:-1]
            layer[:, :-1, 0] = layer[:, :-1, 1]
            layer[:, :-1, -1] = layer[:, :-1, -2]
            layer[:, 0, 1:-1] = layer[:, 1, 1:-1]
            layer[:, -1, 1:-1] = layer[:, -2, 1:-1]


def _bilinear_cpu(values, distances):
    x21, x34, west_low, east_low, west_high, east_high, span, south, north = distances[..., None]
    low = values[0]*east_low/x21+values[1]*west_low/x21
    high = values[3]*east_high/x34+values[2]*west_high/x34
    return north*low/span+south*high/span


def _nearest_sites(positions, ii, jj, dx, dy, *, native_math=False, available=None):
    ny, nx = ii.shape
    scalar = positions.dtype.type
    dx, dy = scalar(dx), scalar(dy)
    original_indices = np.trunc(positions/np.array((dx, dy), dtype=scalar)).astype(int)
    eligible = np.flatnonzero(np.all((original_indices >= 0) & (original_indices < (nx-1, ny-1)), axis=1))
    if not len(eligible):
        if available is None:
            raise ValueError('No HRRR station is inside the nearest-site grid')
        return np.full((ny, nx), available[0], dtype=int)
    east = ii.astype(scalar)*dx-positions[eligible, 0, None, None]
    north = jj.astype(scalar)*dy-positions[eligible, 1, None, None]
    distance = east.astype(float)**2+north.astype(float)**2
    if native_math or available is not None:
        distance = np.minimum(np.sqrt(distance).astype(np.float32), np.float32(100000.))
    nearest = eligible[np.argmin(distance, axis=0)]
    if available is not None:
        # Unmatched map entries use the first active station. Native's
        # startup map also defaults to source zero beyond the search radius.
        nearest = np.where(distance.min(axis=0) < 100000., nearest, 0)
        nearest = available[nearest]
    return nearest


def _quadrilaterals(grid_ids, positions, ii, jj, dx, dy, *, native_math=False, map_positions=None):
    if map_positions is None:
        map_positions = positions
    shape = ii.shape
    interior = (ii > 0) & (jj > 0) & (ii < shape[1]-1) & (jj < shape[0]-1)
    covered = np.zeros(shape, dtype=bool)
    ids = np.zeros((4, *shape), dtype=int)
    scalar = np.float32 if native_math else float
    weights = np.zeros((9 if native_math else 4, *shape), dtype=scalar)
    lookup = {grid_id: n for n, grid_id in enumerate(grid_ids)}
    x, y = ii.astype(scalar)*dx, jj.astype(scalar)*dy
    for n, (j, i) in enumerate(grid_ids):
        neighbors = [lookup.get((j, i+1)), lookup.get((j+1, i+1)), lookup.get((j+1, i))]
        if None in neighbors:
            continue
        corners = [n, *neighbors]
        (_, mx2, _, mx4), (my1, _, my3, _) = map_positions[corners].T
        mask = interior & ~covered & (x >= mx4) & (x <= mx2) & (y >= my1) & (y <= my3)
        if not mask.any():
            continue
        (x1, x2, x3, x4), (y1, y2, y3, y4) = positions[corners].T
        if x1 == x2 or x3 == x4:
            raise ValueError('Degenerate HRRR cell in bilinear interpolation')
        west_low, west_high = x[mask]-x1, x[mask]-x4
        if native_math:
            south = west_low*(y2-y1)/(x2-x1)+y1
            north = west_high*(y3-y4)/(x3-x4)+y4
        else:
            west_east_low, west_east_high = west_low/(x2-x1), west_high/(x3-x4)
            south = y1+west_east_low*(y2-y1)
            north = y4+west_east_high*(y3-y4)
        if np.any(north == south):
            raise ValueError('Degenerate HRRR cell in bilinear interpolation')
        yj = jj[mask].astype(scalar)*dx  # The native interpolator aliases dy to dx.
        if native_math:
            weights[:, mask] = np.broadcast_arrays(x2-x1, x3-x4, west_low, x2-x[mask], west_high,
                                                    x3-x[mask], north-south, yj-south, north-yj)
        else:
            fraction = (yj-south)/(north-south)
            weights[:, mask] = ((1-fraction)*(1-west_east_low), (1-fraction)*west_east_low,
                                fraction*west_east_high, fraction*(1-west_east_high))
        ids[:, mask] = np.asarray(corners)[:, None]
        covered[mask] = True
    return ids, weights, covered
