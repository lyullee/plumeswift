"""Original serial local-mixing algorithm (LocalMixingSerial.cpp, GPL-3.0).

This is the native bounded point-propagation algorithm, not an EDT or nearest
triangle replacement. Wall ordering, front/left index offsets, absent above-wall
propagation and the top transition are retained for native parity.
"""
from dataclasses import dataclass
import numpy as np


def save_mixing_length(path, variable, values):
    """Write the static float32 field used by upstream method 4."""
    from .netcdf_legacy import open_legacy_output

    with open_legacy_output(path) as output:
        for axis, size in zip('xyz', reversed(values.shape), strict=True):
            output.define_dimension(axis, size)
        output.define_field(variable, ('z', 'y', 'x'), 'f4', units='m', long_name='distance to nearest object')
        output.write(variable, values)


@dataclass(frozen=True)
class NetCDFMixing:
    path: str
    variable: str = 'mixlength'
    native_hyperslab: bool = False

    def __call__(self, workspace):
        from .netcdf_legacy import open_legacy_input

        shape = workspace.arrays['icellflag'].shape
        with open_legacy_input(self.path, raw=self.native_hyperslab) as nc:
            if tuple(nc.dimension_size(axis) for axis in 'zyx') != shape:
                raise ValueError('Mixing-length NetCDF dimensions do not match the domain')
            try:
                variable = nc.field(self.variable)
            except KeyError:
                raise ValueError(f'Missing mixing-length variable {self.variable}') from None
            if self.native_hyperslab:
                # LocalMixingNetCDF supplies positional start/count and reads
                # stored numeric values through the NetCDF C++ API.
                data = nc.read(self.variable, dtype='f8', start=(0, 0, 0), count=shape)
            elif variable.dimensions != ('z', 'y', 'x'):
                raise ValueError('Mixing-length input must use static (z,y,x) dimensions')
            else:
                data = nc.read(self.variable)
            if np.ma.isMaskedArray(data) and np.ma.getmaskarray(data).any():
                raise ValueError('Missing/masked mixing-length values')
            values = np.array(data, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Mixing length must be finite and nonnegative')
        return values


@dataclass(frozen=True)
class SerialMixing:
    native_dy_alias: bool = True
    native_math: bool = False

    def __call__(self, workspace):
        a, c = workspace.arrays, workspace.coordinates
        flags = a['icellflag']; nz, ny, nx = flags.shape
        x, y, z = c['x'], c['y'], c['z']
        dx = x[1]-x[0]; dy = dx if self.native_dy_alias else y[1]-y[0]
        zf = c['z_face']
        terrain = a['terrain']
        half = .5
        if self.native_math:
            from .domain import native_vertical_coordinates

            dx, dy = map(np.float32, (dx, dy))
            half = np.float32(.5)
            x = (np.arange(nx, dtype=np.float32)+half)*dx
            y = (np.arange(ny, dtype=np.float32)+half)*np.float32(y[1]-y[0])
            z, zf = native_vertical_coordinates(a['dz_array'])
            terrain = terrain.astype(np.float32, copy=False)
        xf = np.r_[half*(x[0]+x[1])-dx, half*(x[:-1]+x[1:]), half*(x[-2]+x[-1])+dx]
        yf = np.r_[half*(y[0]+y[1])-dy, half*(y[:-1]+y[1:]), half*(y[-2]+y[-1])+dy]
        solid = np.isin(flags, [0, 2]); interior = np.zeros(flags.shape, dtype=bool)
        interior[:-1, 1:-1, 1:-1] = True
        max_z = float(np.max(np.where((solid & interior), z[:, None, None], 0.)))
        candidates = np.flatnonzero(z.astype(float, copy=False) > 3*max_z)
        max_height = int(candidates[0]) if len(candidates) else nz-1
        result = np.zeros(flags.shape)
        seed = np.where(np.arange(nz)[:, None, None] < max_height, z[:, None, None]-terrain[None], z[:, None, None])
        fluid_seed = interior & ~solid
        result[fluid_seed] = np.maximum(np.broadcast_to(seed, result.shape)[fluid_seed], 0.)
        wall_domain = interior.copy(); wall_domain[0] = False
        wall_domain &= ~solid
        def walls(axis, offset):
            mask = wall_domain & np.roll(solid, -offset, axis=axis)
            # Native loops i,j,k, which matters when later walls reset values.
            return np.argwhere(mask.transpose(2, 1, 0))
        def reduce_plane(k, j1, j2, i1, i2, point):
            xx, yy, zz = point
            x_offset = (x[None, i1:i2+1]-xx).astype(float, copy=False)
            y_offset = (y[j1:j2+1, None]-yy).astype(float, copy=False)
            z_offset = float(z[k]-zz)
            distance = np.sqrt(x_offset**2+y_offset**2+z_offset**2)
            if self.native_math:
                distance = distance.astype(np.float32)
            view = result[k, j1:j2+1, i1:i2+1]
            np.minimum(view, distance, out=view)
        for i, j, k in walls(0, -1):
            maximum = max_height-k
            result[k, j, i] = z[k]-zf[k]
            neighbors = [flags[k-1, j, i+1], flags[k-1, j, i-1], flags[k-1, j+1, i], flags[k-1, j-1, i]]
            if all(value == 2 for value in neighbors):
                continue
            if all(value == 0 for value in neighbors):
                for kk in range(maximum+1):
                    result[k+kk, j, i] = min(abs(z[k+kk]-zf[k]), result[k+kk, j, i])
            else:
                for kk in range(maximum+1):
                    i1, i2 = max(i-maximum-kk, 1), min(i+maximum+kk, nx-1)
                    j1, j2 = max(j-maximum-kk, 1), min(j+maximum+kk, ny-1)
                    reduce_plane(k+kk, j1, j2, i1, i2, (x[i], y[j], zf[k]))
        # Native collects above-wall indices but does not propagate them.
        for i, j, k in walls(2, -1):
            # The inclusive native loop can read x_cc[nx]. Its flattened
            # destination is on the next row's untouched outer x boundary.
            maximum = min(k, nx-1-i)
            result[k, j, i] = x[i]-xf[i]
            for ii in range(maximum+1):
                reduce_plane(k, max(j-ii, 0), min(j+ii+1, ny-1), i+ii, i+ii, (xf[i], y[j], z[k]))
        for i, j, k in walls(2, 1):
            maximum = k if i-k > 0 else i
            result[k, j, i] = xf[i+1]-x[i]
            for ii in range(0, -maximum-1, -1):
                if i+1+ii >= nx:
                    raise ValueError('Native front-wall propagation would leave the grid')
                reduce_plane(k, max(j+ii, 0), min(j-ii+1, ny-1), i+1+ii, i+1+ii, (xf[i+1], y[j], z[k]))
        for i, j, k in walls(1, -1):
            # Likewise, stop before y_cc[ny], which would wrap to the next plane.
            maximum = min(k, ny-1-j)
            result[k, j, i] = y[j]-yf[j]
            for jj in range(maximum+1):
                reduce_plane(k, j+jj, j+jj, max(i-jj, 0), min(i+jj+1, nx-1), (x[i], yf[j], z[k]))
        for i, j, k in walls(1, 1):
            maximum = k if j-k > 0 else j
            result[k, j, i] = yf[j+1]-y[j]
            for jj in range(0, -maximum-1, -1):
                if j+1+jj >= ny:
                    raise ValueError('Native left-wall propagation would leave the grid')
                reduce_plane(k, j+1+jj, j+1+jj, max(i+jj, 0), min(i-jj+1, nx-1), (x[i], yf[j+1], z[k]))
        k1 = min(max_height-1, nz-1); k2 = min(k1+k1//5, nz-1)
        if k2 > k1:
            base = result[k1, 1:-1, 1:-1].copy()
            slope = (result[k2, 1:-1, 1:-1]-base)/(z[k2]-z[k1])
            if self.native_math:
                slope = slope.astype(np.float32)
            for k in range(k1, k2):
                result[k, 1:-1, 1:-1] = base+(z[k]-z[k1])*slope
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite serial mixing length')
        return result
