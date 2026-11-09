"""Standalone 2018 QES single-building CPU and CUDA benchmarks (GPL-3.0)."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from .domain import Domain
from .geometry import PolygonBuilding, block_solid_neighbours
from .numerics.native_cpu_rb import NativeRedBlackSOR, NativeSerialSOR
from .wind_projection import zero_solid_faces


def _scratch_sor(coefficients, rhs, max_iterations, tolerance, *, xp, grouping):
    """Retain the scratch solver's double pressure and mean-update stopping rule."""
    nz, ny, nx = rhs.shape
    k, j, i = np.indices((nz-2, ny-2, nx-2))+1
    nodes = ((k*ny+j)*nx+i).ravel()
    omega = np.float32(1.78)
    stay = np.float32(1)-omega
    groups = []
    for ids in grouping(nodes, (k+j+i).ravel()):
        e, f, g, h, m, n = (field.ravel()[ids] for field in coefficients)
        groups.append(tuple(map(xp.asarray, (ids, e, f, g, h, m, n, omega/(e+f+g+h+m+n), rhs.ravel()[ids]))))
    pressure = xp.zeros(rhs.shape)
    flat = pressure.ravel()
    plane = nx*ny
    error, reduced_error, iterations = 1., 0., 0
    while iterations < max_iterations and error > tolerance:
        previous = pressure.copy()
        # Each CPU wavefront or CUDA checkerboard contains independent cells.
        for ids, e, f, g, h, m, n, weight, source in groups:
            neighbours = (e*flat[ids+1]+f*flat[ids-1]+g*flat[ids+nx]
                          +h*flat[ids-nx]+m*flat[ids+plane]+n*flat[ids-plane])
            flat[ids] = weight*(neighbours-source)+stay*flat[ids]
        pressure[0] = pressure[1]
        error = float(xp.mean(xp.abs(pressure-previous)))
        if iterations == 0:
            reduced_error = error*1e-3
        iterations += 1
    return pressure, dict(iterations=iterations, mean_update=error, reduced_error=reduced_error)


def _write_velocity(path, coordinates, velocity):
    z, y, x = np.meshgrid(*(coordinates[axis] for axis in 'zyx'), indexing='ij')
    k, j, i = np.indices(z.shape)
    data = np.column_stack([field.ravel() for field in (i, j, k, x, y, z, *velocity)])
    np.savetxt(path, data, fmt=('%d',)*3+('%.6g',)*6, delimiter='\t')


def run_single_building(output, *, backend='cpu'):
    """Solve and write both of the original diagnostic velocity tables."""
    if backend == 'cpu':
        xp, grouping, context = np, NativeSerialSOR._groups, nullcontext()
    elif backend == 'cuda':
        from .numerics.cuda_runtime import load_cupy

        xp = load_cupy(0)
        grouping, context = NativeRedBlackSOR._groups, xp.cuda.Device(0)
    else:
        raise ValueError(f'Unknown single-building solver backend {backend}')
    started = time.perf_counter()
    dx = np.float32(5)
    domain = Domain((40, 40, 40), (float(dx),)*3)
    workspace = domain.workspace()
    arrays, coordinates = workspace.arrays, workspace.coordinates
    PolygonBuilding.rectangle(90., 90., 20., 20., 20.).flag(workspace)
    u, v, w = (arrays[name] for name in ('u0', 'v0', 'w0'))
    roughness, reference, speed = map(np.float32, (.1, 10., 5.))
    heights = coordinates['z'][1:].astype(np.float32)
    log_ratio = np.log(((heights+roughness)/roughness).astype(float)).astype(np.float32)
    log_reference = np.float32(np.log(float((reference+roughness)/roughness)))
    u[1:-1] = (speed*(log_ratio/log_reference))[:, None, None]
    zero_solid_faces(u, v, w, arrays['icellflag'])

    rhs = -2*(np.diff(u[:-1, :-1], axis=2)+np.diff(v[:-1, :, :-1], axis=1)
              +np.diff(w[:, :-1, :-1], axis=0))/dx
    block_solid_neighbours(arrays, native_outer_wrap=False)
    coefficients = [arrays[key].astype(np.float32)/(dx*dx) for key in 'efghmn']
    with context:
        pressure, report = _scratch_sor(coefficients, rhs, 500, float(np.float32(1e-9)), xp=xp, grouping=grouping)
        if backend == 'cuda':
            pressure = xp.asnumpy(pressure)

    target = np.s_[1:-1, 1:-1, 1:-1]
    current = pressure[1:, 1:, 1:]
    for field, neighbour in zip((u, v, w), (pressure[1:, 1:, :-1], pressure[1:, :-1, 1:], pressure[:-1, 1:, 1:]), strict=True):
        field[target] += (current-neighbour)/(2*float(dx))
    zero_solid_faces(u, v, w, arrays['icellflag'])
    report['solve_seconds'] = time.perf_counter()-started

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    face_path = output/'Final velocity.dat'
    cell_path = output/'Final velocity, cell-centered.dat'
    # The final scratch coordinate in each allocation was unwritten. Give
    # these unused terminal entries the zeros present in the shipped tables.
    face_coordinates = {axis: np.r_[coordinates[axis], 0.] for axis in 'xyz'}
    _write_velocity(face_path, face_coordinates, (u, v, w))
    centered = (.5*(u[:-1, :-1, :-1]+u[:-1, :-1, 1:]),
                .5*(v[:-1, :-1, :-1]+v[:-1, 1:, :-1]),
                .5*(w[:-1, :-1, :-1]+w[1:, :-1, :-1]))
    _write_velocity(cell_path, coordinates, centered)
    return report | dict(backend=backend, face_table=str(face_path), cell_table=str(cell_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.'))
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    print(json.dumps(run_single_building(args.output, backend=args.backend), indent=2))


if __name__ == '__main__':
    main()
