"""Python CUDA implementation of the QES two-pass Barnes wind profiler."""
from functools import lru_cache

import numpy as np

from .domain import native_vertical_coordinates
from .native_sensor import barnes_scale
from .numerics.cuda_runtime import load_cupy
from .terrain_meteorology import station_profiles


@lru_cache(maxsize=1)
def _kernels():
    import cupy as cp
    from cupyx import jit

    @jit.rawkernel()
    def horizontal_weights(positions, x, y, inverse, weights, totals):
        column = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        nx = x.size
        if column < totals.size:
            j, i = column//nx, column%nx
            total = np.float32(0)
            for site in range(positions.shape[0]):
                dx = np.float64(positions[site, 0]-x[i])
                dy = np.float64(positions[site, 1]-y[j])
                weight = np.float32(cp.exp(-inverse*(dx*dx)-inverse*(dy*dy)))
                weights[column, site] = weight
                total += weight
            if total == 0:
                total = np.float32(0)
                for site in range(positions.shape[0]):
                    weights[column, site] = np.float32(1e-20)
                    total += np.float32(1e-20)
            totals[column] = total

    @jit.rawkernel()
    def project_levels(values, weights, totals, ground, starts, abl, z, faces, wind, correct):
        tid = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        plane, nz = ground.size, wind.shape[0]
        k, column = tid//plane, tid%plane
        if k < nz-1:
            base = ground[column]
            level = k+base
            if level < nz:
                for component in range(2):
                    total = np.float32(0)
                    for site in range(starts.size):
                        start = starts[site]
                        difference = z[base]-z[start]
                        surface = np.float32(.05)*abl[site]
                        if cp.abs(difference) <= abl[site]:
                            surface = np.float32(.05)*(np.float32(2)*abl[site]-difference)
                        index = k+start
                        if index > nz-2:
                            index = nz-2
                        elif z[k]-faces[base] > surface and level > index:
                            index = level
                        total += weights[column, site]*values[site, index, component]
                    result = total/totals[column]
                    if correct:
                        result += wind[level, column, component]
                    wind[level, column, component] = result

    @jit.rawkernel()
    def station_residual(profiles, first, ground, starts, cells, fractions, nx, residual):
        tid = jit.blockIdx.x*jit.blockDim.x+jit.threadIdx.x
        nz = profiles.shape[1]
        site, level = tid//nz, tid%nz
        if site < profiles.shape[0]:
            column = cells[site]
            for component in range(2):
                estimate = np.float32(0)
                if column >= 0:
                    if level >= ground[column]:
                        wx, wy = fractions[site, 0], fractions[site, 1]
                        left, lower_weight = np.float32(1)-wx, np.float32(1)-wy
                        upper = left*first[level, column+nx, component]+wx*first[level, column+nx+1, component]
                        lower = left*first[level, column, component]+wx*first[level, column+1, component]
                        estimate = wy*upper+lower_weight*lower
                elif starts[site] <= level and level <= nz-2:
                    estimate = profiles[site, level, component]
                residual[site, level, component] = profiles[site, level, component]-estimate

    return horizontal_weights, project_levels, station_residual


def barnes_cuda(workspace, stations, *, sampled=None, device=0):
    """Run both Barnes passes on CUDA using host-sampled station profiles."""
    arrays, coords = workspace.arrays, workspace.coordinates
    nz, ny, nx = arrays['u0'].shape
    dx = np.float32(coords['x'][1]-coords['x'][0])
    positions = np.asarray([(station.x, station.y) for station in stations], dtype=np.float32)
    if len(stations) < 2 or not np.isfinite(positions).all():
        raise ValueError('CUDA Barnes needs at least two finite station positions')
    x, y = (((np.arange(size)-.5)*float(dx)).astype(np.float32) for size in (nx, ny))
    inside = np.all((positions > 0) & (positions < np.array([nx-1, ny-1], np.float32)*dx), axis=1)
    sx, sy = positions[inside].T
    j = np.clip(np.searchsorted(y, sy)-1, 0, ny-2)
    # Read native packed east neighbors while the footprint stays in-plane.
    i = np.minimum(np.searchsorted(x, sx)-1, nx-1-(j == ny-2))
    cells = np.full(len(stations), -1, dtype=np.int32)
    cells[inside] = j*nx+i
    fractions = np.zeros_like(positions)
    fractions[inside] = np.column_stack(((sx-x[i])/dx, (sy-y[j])/dx))
    profiles, starts, abl = station_profiles(workspace, stations, native_math=True) if sampled is None else sampled
    ground = arrays.get('terrain_face_id')
    if ground is None:
        ground = np.ones((ny, nx), dtype=np.int32)
    if ground.shape != (ny, nx) or np.any((ground < 0) | (ground >= nz-1)):
        raise ValueError('CUDA Barnes needs valid ground indices on the wind grid')
    z, faces = native_vertical_coordinates(arrays['dz_array'])
    inverse = np.float64(np.float32(1)/barnes_scale(positions))
    cp = load_cupy(device)
    with cp.cuda.Device(device):
        weight_kernel, project, residual_kernel = _kernels()
        block = (128,)

        def grid(size):
            return ((size+block[0]-1)//block[0],)

        profiles, positions, x, y, abl, z, faces, cells, fractions = map(
            cp.asarray, (profiles, positions, x, y, abl, z, faces, cells, fractions))
        ground, starts = (cp.asarray(a, dtype=cp.int32).ravel() for a in (ground, starts))
        weights = cp.empty((nx*ny, len(stations)), dtype=cp.float32)
        totals = cp.empty(nx*ny, dtype=cp.float32)
        wind = cp.zeros((nz, nx*ny, 2), dtype=cp.float32)
        residual = cp.empty_like(profiles)
        weight_kernel[grid(nx*ny), block](positions, x, y, inverse, weights, totals)
        args = (weights, totals, ground, starts, abl, z, faces, wind)
        levels = grid((nz-1)*nx*ny)
        project[levels, block](profiles, *args, np.bool_(False))
        residual_kernel[grid(nz*len(stations)), block](profiles, wind, ground, starts, cells, fractions,
                                                       np.int32(nx), residual)
        project[levels, block](residual, *args, np.bool_(True))
        velocity = cp.asnumpy(wind).reshape(nz, ny, nx, 2)
    for component, name in enumerate(('u0', 'v0')):
        arrays[name][:] = velocity[..., component]
    arrays['w0'].fill(0)
