"""Serial QES fire-induced velocity potential (GPL-3)."""
from dataclasses import dataclass
from math import pi, sqrt

import numpy as np


@dataclass(frozen=True)
class HeatPotential:
    radial: np.ndarray
    vertical: np.ndarray
    g: np.ndarray
    g_prime: np.ndarray
    radius: np.ndarray
    height: np.ndarray
    native_math: bool = False

    @classmethod
    def read(cls, path, *, native_input=False):
        from .io import _open_netcdf

        with _open_netcdf(path, raw=native_input) as nc:
            radial, g, radius, height = (nc[name][:] for name in ('u_r', 'G', 'rStar', 'zStar'))
            if native_input and radial.ndim == 2 and g.ndim == 1:
                # The native counts for these paired variables come from u_r and G.
                nz, nr = radial.shape
                vertical, g_prime = nc['u_z'][:nz, :nr], nc['Gprime'][:len(g)]
            else:
                vertical, g_prime = nc['u_z'][:], nc['Gprime'][:]
            values = radial, vertical, g, g_prime, radius, height
        if any(np.ma.is_masked(value) for value in values):
            raise ValueError('Missing fire potential lookup data')
        dtype = np.float32 if native_input else float
        table = cls(*(np.asarray(value, dtype=dtype) for value in values), native_math=native_input)
        table._validate(native_input=native_input)
        return table

    def _validate(self, *, native_input=False):
        if (self.radius.ndim != 1 or self.height.ndim != 1 or self.g.ndim != 1 or
                min(len(self.radius), len(self.height), len(self.g)) < 2 or
                self.radial.ndim != 2 or min(self.radial.shape) < 1 or
                self.radial.shape != self.vertical.shape or self.g_prime.shape != self.g.shape or
                (not native_input and self.radial.shape != (len(self.height), len(self.radius)))):
            raise ValueError('Invalid fire potential lookup dimensions')
        if any(not np.isfinite(value).all() for value in vars(self).values()):
            raise ValueError('Non-finite fire potential lookup values')
        for coordinates in (self.radius, self.height):
            if native_input:
                if coordinates[1] <= coordinates[0]:
                    raise ValueError('Fire potential lookup needs a positive first coordinate interval')
            elif coordinates[0] != 0 or np.any(np.diff(coordinates) <= 0):
                raise ValueError('Fire potential lookup coordinates must increase from zero')

    def _sample(self, x, y, z, speed):
        if self.native_math:
            x32, y32 = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
            r = np.sqrt(x32*x32+y32*y32).astype(float)
        else:
            r = np.hypot(x, y)
        radial = np.empty_like(r)
        vertical = np.empty_like(r)
        near = (r < 30) & (z < 60)
        if np.any(near):
            radius = r[near]
            on_axis = radius < 1e-5
            ri = np.floor(np.where(on_axis, 0., radius)/(self.radius[1]-self.radius[0])).astype(int)
            zi = int(np.floor(z/(self.height[1]-self.height[0])))
            if zi >= self.radial.shape[0] or np.any(ri >= self.radial.shape[1]):
                raise ValueError('Fire lookup table is shorter than the native near-field range')
            radial[near] = np.where(on_axis, 0, self.radial[zi, ri])
            vertical[near] = self.vertical[zi, ri]
        far = ~near
        if np.any(far):
            radius = r[far]
            distance = np.hypot(radius, z)
            if self.native_math:
                # The serial reference rounds both trig calls to float before
                # evaluating the far-field radial term close to the axis.
                with np.errstate(divide='ignore'):
                    angle = np.arctan(radius.astype(np.float32)/np.float32(z))
                angle_coordinate = ((np.float32(1)+np.cos(angle))*np.float32(.5)).astype(float)
            else:
                angle_coordinate = .5*(1+z/distance)
            coordinate = 2*len(self.g)*(angle_coordinate-.5)
            # Native far-axis x1=1 selects an index one past the table.
            # Use its endpoint and evaluate radial velocity only off axis.
            low = np.clip(np.floor(coordinate).astype(int), 0, len(self.g)-1)
            high = np.clip(np.ceil(coordinate).astype(int), 0, len(self.g)-1)
            g = .5*(self.g[low]+self.g[high])
            gp = .5*(self.g_prime[low]+self.g_prime[high])
            root = np.sqrt(angle_coordinate*(1-angle_coordinate))
            ratio = np.divide(1-2*angle_coordinate, root, out=np.zeros_like(root), where=root > 0)
            radial[far] = radius/(2*pi*distance**3)+distance**(-1/3)*((5/6)*ratio*g-root*gp)
            vertical[far] = z/(2*pi*distance**3)+distance**(-1/3)*((5/3)*g+.5*(1-2*angle_coordinate)*gp)
        scale = np.divide(speed*radial, r, out=np.zeros_like(r), where=r > 0)
        return scale*x, scale*y, speed*vertical

    def apply(self, fire, workspace):
        """Superimpose native block-merged potential onto the initial wind."""
        blocks, mix = self._merged_sources(fire, workspace)
        ny, nx = fire.terrain.shape
        potential = np.zeros((3, workspace.arrays['u0'].shape[0]-2, ny, nx))
        yy, xx = np.indices((ny, nx))
        scalar = np.float32 if self.native_math else float
        dx, dy, dz = map(scalar, (fire.dx, fire.dy, fire.dz))
        for first, last, cx, cy, origin, velocity, length in blocks:
            # PotentialSerial forms dimensionless coordinates as floats before
            # sampling. Widen after that step for the potential evaluation.
            length, origin = scalar(length), scalar(origin)
            x = ((xx-cx).astype(scalar)*dx/length).astype(float, copy=False)
            y = ((yy-cy).astype(scalar)*dy/length).astype(float, copy=False)
            for k in range(first, last):
                z = max(0., float((scalar(k)*dz-origin)/length))
                for component, values in enumerate(self._sample(x, y, z, velocity)):
                    potential[component, k] += values
        self._superimpose(workspace, potential)
        return potential, mix

    def _merged_sources(self, fire, workspace, *, cuda=False):
        a = workspace.arrays
        ny, nx = fire.terrain.shape
        nz = a['u0'].shape[0]
        if a['u0'].shape[1:] != (ny+1, nx+1):
            raise ValueError('Fire potential and staggered wind grids differ')
        dx, dy, dz = fire.dx, fire.dy, fire.dz
        mix = np.array(a.get('terrain_id', np.ones((ny, nx))), dtype=float, copy=True)
        blocks = []
        burning = fire.burn_flag == 1
        yy, xx = np.indices((ny, nx))
        lambda_mix = sqrt(25/132)/.09
        filt = 1
        while filt < nx:
            old_mix = mix.copy()
            for x0 in range(0, nx-filt, filt):
                for y0 in range(0, ny-filt, filt):
                    window = (slice(y0, y0+filt), slice(x0, x0+filt))
                    selected = burning[window]
                    heat = fire.heat[window][selected].sum()
                    if heat == 0:
                        continue
                    cx = int(xx[window][selected].mean())
                    cy = int(yy[window][selected].mean())
                    terrain_sum = fire.terrain[window][selected].sum()
                    virtual_origin = -.2869*dx*filt*lambda_mix+int(terrain_sum/selected.sum())
                    first = int(np.floor(old_mix[window][selected].mean()))
                    limit = lambda_mix*dx*filt/dz+terrain_sum
                    if cuda:
                        # PotentialGlobal stores the ceiling at each merge and
                        # includes its upper layer, excluding the lower one.
                        first += 1
                        limit = np.ceil(limit)
                    mix[window] = limit
                    last = int(np.ceil(min(nz-3, limit)))
                    if cuda:
                        last += 1
                    velocity = (9.81**2*heat/(1.125*293.15*1150))**.2
                    length = (heat/(1.125*1150*293.15*sqrt(9.81)))**.4
                    if max(0, first) < last:
                        blocks.append((max(0, first), last, cx, cy, virtual_origin, velocity, length))
            filt *= 2
        return blocks, mix

    @staticmethod
    def _superimpose(workspace, potential):
        a = workspace.arrays
        _, layers, ny, nx = potential.shape
        k, j, i = np.indices((layers-1, ny-1, nx-1))
        j += 1
        i += 1
        fluid = np.isin(a['icellflag'][k, j, i], (1, 12))
        ids = k[fluid], j[fluid], i[fluid]
        flat = np.ravel_multi_index(ids, potential.shape[1:])
        # Native flat offsets cross the last row for the edge faces.
        for name, values, offset in zip(('u0', 'v0', 'w0'), potential, (1, nx, nx*ny)):
            values = values.ravel()
            a[name][ids] += .5*(values[flat]+values[flat+offset])
