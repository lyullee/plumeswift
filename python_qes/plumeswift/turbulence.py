"""QES local turbulence construction, separated from non-local closures.

Equations and wall ordering from TURBGeneralData.cpp/TURBWall.cpp,
GPL-3.0-only. No LES/RANS substitute is used. Components accept explicit
mixing distances, so local distance algorithms can be replaced independently.
"""
from dataclasses import dataclass
import numpy as np

from .canopy_roc_turbulence import apply_stress as apply_row_stress, apply_viscosity as apply_row_viscosity
from .domain import native_vertical_coordinates
from .numerics.linux_float_angles import atan2f, sincosf


def cell_velocity(velocity, *, tangential_shell=False, native_math=False):
    u, v, w = (velocity[k] for k in ['u', 'v', 'w'])
    if tangential_shell:
        # Native wall routines can average tangential velocities one cell
        # beyond the cell-center shell using the allocated staggered faces.
        # The unavailable outward NORMAL average remains explicitly undefined.
        result = np.full(u.shape+(3,), np.nan, dtype=np.float32 if native_math else float)
        result[:, :, :-1, 0] = .5*(u[:, :, :-1]+u[:, :, 1:])
        result[:, :-1, :, 1] = .5*(v[:, :-1, :]+v[:, 1:, :])
        result[:-1, :, :, 2] = .5*(w[:-1, :, :]+w[1:, :, :])
        return result
    return np.stack([.5*(u[:-1, :-1, :-1]+u[:-1, :-1, 1:]),
                     .5*(v[:-1, :-1, :-1]+v[:-1, 1:, :-1]),
                     .5*(w[:-1, :-1, :-1]+w[1:, :-1, :-1])], axis=-1)


def default_mixing_distance(workspace, *, native_math=False):
    a, c = workspace.arrays, workspace.coordinates
    z, terrain = c['z'], a['terrain']
    if native_math:
        z, _ = native_vertical_coordinates(a['dz_array'])
        terrain = terrain.astype(np.float32, copy=False)
    result = np.maximum(z[:, None, None]-terrain[None], 0)
    result[np.isin(a.get('icellflag_initial', a['icellflag']), [0, 2])] = 0.
    result[0] = result[-1] = 0.
    return result


def _fluid(workspace):
    a = workspace.arrays
    fluid = ~np.isin(a.get('icellflag_initial', a['icellflag']), [0, 2])
    fluid[[0, -1]] = False; fluid[:, [0, -1]] = False; fluid[:, :, [0, -1]] = False
    return fluid


def _wall_indices(flags, flag):
    fluid = ~np.isin(flags, [0, 2])
    valid = fluid.copy()
    valid[[0, -1]] = False; valid[:, 0] = False; valid[:, :, 0] = False
    result = {}
    for name, axis, offset in [('below', 0, -1), ('above', 0, 1), ('back', 2, -1), ('front', 2, 1), ('right', 1, -1), ('left', 1, 1)]:
        stride = [flags.shape[1]*flags.shape[2], flags.shape[2], 1][axis]
        match = valid & (np.roll(flags.ravel(), -offset*stride).reshape(flags.shape) == flag)
        result[name] = np.where(match)
    return result


def _turbulence_flags(workspace, fluid):
    a = workspace.arrays
    cells = a.get('icellflag_initial', a['icellflag'])
    flags = fluid.astype(np.int32)
    cut_region = np.s_[:-1, 1:-1, 1:-1]
    for solid, cut, code in ((0, 7, 4), (2, 8, 2)):
        region = flags[cut_region]
        region[cells[cut_region] == cut] = code+1
        for index in _wall_indices(cells, solid).values():
            flags[index] = code
    return flags


def _shift(index, axis, offset):
    result = list(index); result[axis] = result[axis]+offset
    return tuple(result)


def _native_cell_neighbour(values, index, axis, offset):
    """Native flattened pointer offset, including horizontal shell row wraps."""
    nz, ny, nx = values.shape
    k, j, i = index
    flat = (k*ny+j)*nx+i+offset*[ny*nx, nx, 1][axis]
    if np.any(flat < 0) or np.any(flat >= values.size):
        raise IndexError('Native stress wall pointer would leave its allocation')
    return values.ravel()[flat]


def _vertical_derivative(field, index, z, *, uniform_width=None, native_math=False):
    k = index[0]
    lo, hi = _shift(index, 0, -1), _shift(index, 0, 1)
    if uniform_width is not None:
        return (field[hi]-field[lo])/(2*uniform_width)
    a, b = z[k]-z[k-1], z[k+1]-z[k]
    expand = (slice(None),)+(None,)*(field.ndim-3)
    span = z[k+1]-z[k-1] if native_math else a+b
    return ((a/b)[expand]*(field[hi]-field[index])+(b/a)[expand]*(field[index]-field[lo]))/span[expand]


def _interior_gradient(velocity, cv, fluid, dx, dy, z, dz, *, native_math=False, force_nonuniform_z=False):
    ids = np.where(fluid)
    gradient = np.zeros(fluid.shape+(3, 3), dtype=np.float32 if native_math else float)
    for coordinate, axis, spacing in [(0, 2, dx), (1, 1, dy)]:
        derivative = (cv[_shift(ids, axis, 1)]-cv[_shift(ids, axis, -1)])/(2*spacing)
        gradient[ids+(slice(None), coordinate)] = derivative
    uniform_width = dz[0] if native_math and not force_nonuniform_z and np.all(dz == dz[0]) else None
    gradient[ids+(slice(None), 2)] = _vertical_derivative(cv, ids, z, uniform_width=uniform_width, native_math=native_math)
    k, j, i = ids
    for component, (field, plus, spacing) in enumerate([
        ('u', (k, j, i+1), dx), ('v', (k, j+1, i), dy), ('w', (k+1, j, i), dz[k])]):
        gradient[ids+(component, component)] = (velocity[field][plus]-velocity[field][ids])/spacing
    return gradient


def _matrix_product(a, b):
    return (a[..., :, 0, None]*b[..., None, 0, :]+a[..., :, 1, None]*b[..., None, 1, :]
            +a[..., :, 2, None]*b[..., None, 2, :])


def _rotate_stress_cpu(stress, angles, sigma, *, explicit_sigma):
    """Use the source's double matrix products and float stress stores."""
    sn, cs = (value.astype(float) for value in sincosf(angles))
    rotation = np.zeros((len(angles), 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = cs
    rotation[:, 0, 1], rotation[:, 1, 0] = -sn, sn
    rotation[:, 2, 2] = 1.
    inverse = rotation.swapaxes(-1, -2).copy()
    inverse[:, :2, :2] /= (cs*cs+sn*sn)[:, None, None]
    aligned = _matrix_product(rotation, _matrix_product(stress.astype(float), inverse))
    # The C++ multiply writes the lower triangle into the shared off-diagonal slots last.
    for i, j in ((1, 0), (2, 0), (2, 1)):
        aligned[:, j, i] = aligned[:, i, j]
    pope = np.float32(.55)
    sigma = np.asarray(sigma, dtype=np.float32)
    if explicit_sigma:
        gains = np.float32(1.5)*sigma*sigma*pope*pope
    else:
        sigma64, pope64 = sigma.astype(float), float(pope)
        gains = (1.5*sigma64*sigma64*pope64*pope64).astype(np.float32)
    for component, gain in enumerate(gains):
        aligned[:, component, component] = np.abs(float(gain)*aligned[:, component, component])
    result = _matrix_product(rotation.swapaxes(-1, -2), _matrix_product(aligned, inverse.swapaxes(-1, -2)))
    for i, j in ((1, 0), (2, 0), (2, 1)):
        result[:, j, i] = result[:, i, j]
    return result.astype(np.float32)


@dataclass(frozen=True)
class LocalTurbulence:
    sigma: tuple[float, float, float] = (2.5, 2., 1.3)
    upper_bound: float = 15.
    background: float = 0.
    building_wall: int = 2
    terrain_wall: int = 2
    wall_roughness: float = .01
    terrain_roughness: float = .02
    native_quirks: bool = True
    nonlocal_mixing: bool = False
    gradient_backend: str = 'cpu'
    native_math: bool = False
    explicit_sigma: bool = True
    native_zero_mixing_nan: bool = False
    force_nonuniform_z: bool = False

    def __post_init__(self):
        if len(self.sigma) != 3 or not np.isfinite(self.sigma).all() or min(self.sigma) < 0:
            raise ValueError('Turbulence sigma requires three finite nonnegative values')
        if not np.isfinite([self.upper_bound, self.background, self.wall_roughness, self.terrain_roughness]).all():
            raise ValueError('Non-finite turbulence parameter')
        if self.upper_bound < 0 or self.background < 0 or min(self.wall_roughness, self.terrain_roughness) <= 0:
            raise ValueError('Invalid turbulence bound/background/roughness')
        if self.building_wall not in (0, 1, 2) or self.terrain_wall not in (0, 1, 2):
            raise ValueError('Wall mode must be 0, 1 or 2')
        if self.gradient_backend not in ('cpu', 'cuda'):
            raise ValueError('Turbulence gradient backend must be cpu or cuda')

    def velocity_gradient(self, workspace, velocity):
        a, c = workspace.arrays, workspace.coordinates
        dx = c['x'][1]-c['x'][0]; dy = c['y'][1]-c['y'][0]
        z = c['z']; dz = a['dz_array']; flags = a.get('icellflag_initial', a['icellflag'])
        if self.native_math:
            dx, dy = map(np.float32, (dx, dy))
            z, _ = native_vertical_coordinates(dz)
            dz = dz.astype(np.float32)
            velocity = {key: np.asarray(velocity[key], dtype=np.float32) for key in 'uvw'}
        fluid = _fluid(workspace)
        walls_enabled = self.building_wall != 0 or self.terrain_wall != 0
        cv = cell_velocity(velocity, tangential_shell=True, native_math=self.native_math) if self.gradient_backend == 'cpu' or walls_enabled else None
        if self.gradient_backend == 'cuda':
            from .cuda_gradient import cuda_velocity_gradient
            gradient = cuda_velocity_gradient(workspace, velocity, fluid)
        else:
            gradient = _interior_gradient(velocity, cv, fluid, dx, dy, z, dz, native_math=self.native_math,
                                          force_nonuniform_z=self.force_nonuniform_z)
        if not walls_enabled:
            return gradient
        wall_dy = dx if self.native_quirks else dy
        z0_u, z0_v = a.get('z0_u'), a.get('z0_v')
        face_shape = velocity['u'].shape[1:]
        has_face_roughness = np.shape(z0_u) == np.shape(z0_v) == face_shape
        if has_face_roughness and self.native_math:
            z0_u, z0_v = (np.asarray(roughness, dtype=np.float32) for roughness in (z0_u, z0_v))
        for flag, mode, roughness in [(0, self.building_wall, self.wall_roughness), (2, self.terrain_wall, self.terrain_roughness)]:
            if mode == 0:
                continue
            walls = _wall_indices(flags, flag)
            for name, axis, offset, components, coordinate, spacing in [
                ('below', 0, 1, (0, 1), 2, None), ('above', 0, -1, (0, 1), 2, None),
                ('back', 2, 1, (1, 2), 0, dx), ('front', 2, -1, (1, 2), 0, dx),
                ('right', 1, 1, (0, 2), 1, wall_dy), ('left', 1, -1, (0, 2), 1, wall_dy)]:
                index = walls['below'] if mode == 1 and name == 'left' and self.native_quirks else walls[name]
                if len(index[0]) == 0:
                    continue
                if mode == 1:
                    distance = .5*dz[index[0]] if spacing is None else .5*spacing
                    wall_z0 = roughness
                    if flag == 2 and has_face_roughness:
                        j, i = index[1:]
                        wall_z0 = .25*(z0_u[j, i]+z0_u[j, i+1]+z0_v[j, i]+z0_v[j+1, i])
                    if np.any(wall_z0 <= 0):
                        raise ValueError("Invalid logarithmic wall roughness")
                    if self.native_math:
                        # Native face sums and roughness are float32, while
                        # the unsuffixed 0.5 promotes the wall law to double.
                        distance = np.asarray(distance, dtype=float)
                        wall_z0 = np.asarray(wall_z0, dtype=np.float32).astype(float)
                    denominator = distance*np.log(distance/wall_z0)
                    if np.any(denominator == 0):
                        raise ValueError("Singular native logarithmic wall law")
                    for component in components:
                        gradient[index+(component, coordinate)] = cv[index+(component,)]/denominator
                elif mode == 2:
                    neighbour = _shift(index, axis, offset)
                    denominator = offset*(z[neighbour[0]]-z[index[0]]) if spacing is None else spacing
                    for component in components:
                        adjacent = cv[neighbour+(component,)]
                        if self.native_math:
                            adjacent = adjacent.astype(float)
                        gradient[index+(component, coordinate)] = offset*(adjacent-cv[index+(component,)])/denominator
                else:
                    raise ValueError("Wall mode must be 0, 1 or 2")
        return gradient

    def stress_divergence(self, workspace, tau):
        a, c = workspace.arrays, workspace.coordinates
        dx = c['x'][1]-c['x'][0]; dy = c['y'][1]-c['y'][0]; z = c['z']
        dtype = np.float32 if self.native_math else float
        uniform_width = None
        if self.native_math:
            dx, dy = map(np.float32, (dx, dy))
            dz = a['dz_array'].astype(np.float32)
            z, _ = native_vertical_coordinates(dz)
            if not self.force_nonuniform_z and np.all(dz == dz[0]):
                uniform_width = dz[0]
            tau = np.asarray(tau, dtype=np.float32)
        wall_dy = dx if self.native_quirks else dy
        ids = np.where(_fluid(workspace))
        result = np.zeros(tau.shape[:-1], dtype=dtype)
        for component in range(3):
            tox, toy, toz = [tau[..., component, d] for d in range(3)]
            derivatives = np.zeros(a['icellflag'].shape+(3,), dtype=dtype)
            derivatives[ids+(0,)] = (tox[_shift(ids, 2, 1)]-tox[_shift(ids, 2, -1)])/(2*dx)
            derivatives[ids+(1,)] = (toy[_shift(ids, 1, 1)]-toy[_shift(ids, 1, -1)])/(2*dy)
            derivatives[ids+(2,)] = _vertical_derivative(toz, ids, z, uniform_width=uniform_width, native_math=self.native_math)
            for flag in [0, 2]:
                walls = _wall_indices(a.get('icellflag_initial', a['icellflag']), flag)
                for name, axis, offset, coordinate, field, spacing in [
                    ('below', 0, 1, 2, toz, None), ('above', 0, -1, 2, toz, None),
                    ('back', 2, 1, 0, tox, dx), ('front', 2, -1, 0, tox, dx),
                    ('right', 1, 1, 1, toy, wall_dy), ('left', 1, -1, 1, toy, wall_dy)]:
                    index = walls[name]; neighbour = _shift(index, axis, offset)
                    denominator = spacing
                    if spacing is None:
                        k = index[0]
                        denominator = z[k+1]-z[k] if offset == 1 or self.native_quirks else z[k]-z[k-1]
                    current = toy[index] if name == 'front' and self.native_quirks else field[index]
                    derivatives[index+(coordinate,)] = offset*(_native_cell_neighbour(field, index, axis, offset)-current)/denominator
            result[ids+(component,)] = derivatives[ids].sum(axis=-1)
        return result

    def build(self, workspace, velocity, *, sensor_roughness=.02, building_heights=(), mixing_distance=None,
              include_gradients=False, previous_turbulence=None):
        a, c = workspace.arrays, workspace.coordinates
        fluid = _fluid(workspace)
        distance = (default_mixing_distance(workspace, native_math=self.native_math)
                    if mixing_distance is None else np.asarray(mixing_distance))
        if distance.shape != fluid.shape or not np.isfinite(distance).all():
            raise ValueError("Mixing distance must be finite and match the cell grid")
        if np.any(distance[fluid] < 0):
            raise ValueError("Mixing distance must be nonnegative in active cells")
        dtype = np.float32 if self.native_math else float
        z_face = c['z_face']
        if self.native_math:
            velocity = {key: np.asarray(velocity[key], dtype=np.float32) for key in 'uvw'}
            z_face = native_vertical_coordinates(a['dz_array'])[1]
        lm = np.zeros(fluid.shape, dtype=dtype)
        # QES multiplies its float von Kármán constant by a double distance.
        mixing_distance_active = distance[fluid]
        if self.native_math:
            mixing_distance_active = mixing_distance_active.astype(float, copy=False)
        if previous_turbulence is None:
            lm[fluid] = float(dtype(.4))*mixing_distance_active
        else:
            # The original TURBGeneralData initializes Lm only at startup.
            # Non-local wake changes feed the next record's local closure.
            lm[:] = previous_turbulence['Lm']
        if len(building_heights):
            mean_height = float(np.mean(building_heights)); z0 = .1*mean_height; d0 = .7*mean_height; zref = 3*mean_height
            if self.native_math:
                heights = np.asarray(building_heights, dtype=dtype)
                mean_height = float(np.add.accumulate(heights)[-1]/dtype(len(heights)))
                z0, d0, zref = map(dtype, (.1*mean_height, .7*mean_height, 3.*mean_height))
        else:
            z0 = sensor_roughness; d0 = 0.; zref = 100*z0
            if self.native_math:
                z0, d0 = dtype(z0), dtype(0.)
                zref = dtype(100)*z0
        kref = int(np.searchsorted(z_face, zref))
        if kref >= len(c['z']) or zref <= d0 or z0 <= 0:
            raise ValueError("Turbulence reference plane is outside the domain or invalid")
        cv = cell_velocity(velocity) if self.nonlocal_mixing else None
        reference_velocity = (cv[kref] if cv is not None else
                              cell_velocity({key: value[kref:kref+2] for key, value in velocity.items()})[0])
        reference_fluid = ~np.isin(a['icellflag'][kref], [0, 2])
        if not reference_fluid.any():
            raise ValueError("No fluid cells at turbulence reference plane")
        if self.native_math:
            speeds = np.linalg.norm(reference_velocity[reference_fluid].astype(float), axis=-1)
            total = dtype(0.)
            for speed in speeds:
                total = dtype(float(total)+speed)
            uref = total/dtype(len(speeds))
            logarithm = dtype(np.log(float((zref-d0)/z0)))
            ustar = dtype(.4*float(uref)/float(logarithm))
        else:
            uref = np.linalg.norm(reference_velocity, axis=-1)[reference_fluid].mean()
            ustar = .4*uref/np.log((zref-d0)/z0)
        bound = dtype(self.upper_bound)*ustar*ustar
        gradient = self.velocity_gradient(workspace, velocity)
        strain = .5*(gradient+gradient.swapaxes(-1, -2))
        viscosity = np.zeros(fluid.shape, dtype=dtype); tke = np.zeros_like(viscosity); eps = np.zeros_like(viscosity)
        mixing = lm[fluid]
        zero_mixing = np.nan if self.native_zero_mixing_nan else 0.
        pope = dtype(.55)
        if self.native_math:
            sxx, syy, szz = (strain[..., i, i][fluid] for i in range(3))
            sxy, sxz, syz = (strain[..., i, j][fluid] for i, j in ((0, 1), (0, 2), (1, 2)))
            squared = ((sxx*sxx+syy*syy+szz*szz).astype(float)+2.*(sxy*sxy+sxz*sxz+syz*syz)).astype(dtype)
            viscosity[fluid] = (mixing*mixing).astype(float)*np.sqrt(2.*squared.astype(float))
            ratio = np.divide(viscosity[fluid], pope*mixing, out=np.zeros_like(mixing), where=mixing > 0)
            tke[fluid] = np.minimum(ratio*ratio, bound)
        else:
            strain_rate = np.sqrt(2*np.sum(strain[fluid]**2, axis=(-1, -2)))
            viscosity[fluid] = mixing**2*strain_rate
            tke[fluid] = np.minimum((mixing*strain_rate/pope)**2, bound)
        if self.native_zero_mixing_nan:
            tke[fluid & (lm == 0)] = np.nan
        for flow in workspace.row_canopy_flows:
            apply_row_viscosity(workspace, flow, lm, tke, viscosity)
        mixing = lm[fluid]
        energy = np.minimum(tke[fluid], bound)
        dissipation_scale = np.sqrt(energy)*pope
        eps[fluid] = np.divide(5.7*dissipation_scale.astype(float)**3, mixing,
                               out=np.full_like(mixing, zero_mixing, dtype=float), where=mixing > 0)
        tau = np.zeros_like(strain)
        # Build the tensor a plane at a time to bound the float64
        # intermediates without changing the order of operations per cell.
        identity = np.eye(3)
        for plane, active in enumerate(fluid):
            diffusive_stress = viscosity[plane][active, None, None]*strain[plane][active]
            if self.native_math:
                diffusive_stress = diffusive_stress.astype(float)
            local_energy = np.minimum(tke[plane][active], bound).astype(float)
            tau[plane][active] = (2/3)*local_energy[:, None, None]*identity-2*diffusive_stress
        kdir = np.searchsorted(z_face, dtype(15)+a['terrain'].astype(dtype, copy=False))
        # Native stress rotation only visits columns containing an active cell.
        kdir[~fluid.any(axis=0)] = 0
        if np.any(kdir >= velocity['u'].shape[0]):
            raise ValueError("Native stress rotation requires a plane 15 m above terrain")
        jj, ii = np.indices(kdir.shape)
        wind_y, wind_x = (velocity[key][kdir, jj, ii] for key in ('v', 'u'))
        if self.native_math:
            theta = atan2f(wind_y, wind_x)
            for plane, active in enumerate(fluid):
                tau[plane][active] = _rotate_stress_cpu(tau[plane][active], theta[active], self.sigma,
                                                       explicit_sigma=self.explicit_sigma)
        else:
            ids = np.where(fluid)
            angles = np.arctan2(wind_y, wind_x)[ids[1], ids[2]]
            rotation = np.zeros((len(angles), 3, 3)); cs, sn = np.cos(angles), np.sin(angles)
            rotation[:, 0, 0] = rotation[:, 1, 1] = cs
            rotation[:, 0, 1] = -sn; rotation[:, 1, 0] = sn; rotation[:, 2, 2] = 1
            aligned = rotation@tau[fluid]@rotation.swapaxes(-1, -2)
            for component, sigma in enumerate(self.sigma):
                aligned[:, component, component] = np.abs(1.5*sigma*sigma*.55*.55*aligned[:, component, component])
            tau[fluid] = rotation.swapaxes(-1, -2)@aligned@rotation
        for flow in workspace.row_canopy_flows:
            apply_row_stress(workspace, flow, velocity, tke, viscosity, tau)
        turbulence_flags = (_turbulence_flags(workspace, fluid) if previous_turbulence is None
                            else previous_turbulence['iturbflag'].copy())
        if self.nonlocal_mixing:
            from .nonlocal_mixing import apply_nonlocal_mixing
            apply_nonlocal_mixing(workspace, cv, tau, lm, tke, eps, turbulence_flags, native_math=self.native_math)
        for plane, active in enumerate(fluid):
            tau[plane][active] = np.clip(tau[plane][active], -bound, bound)
        if self.background > 0:
            for component in range(3):
                tau[..., component, component][fluid] += dtype(self.background)*dtype(self.background)
        divergence = self.stress_divergence(workspace, tau)
        fields = {key: tau[..., i, j] for key, i, j in [('txx', 0, 0), ('txy', 0, 1), ('txz', 0, 2), ('tyy', 1, 1), ('tyz', 1, 2), ('tzz', 2, 2)]}
        fields.update({key: divergence[..., i] for i, key in enumerate(['div_tau_x', 'div_tau_y', 'div_tau_z'])})
        fields.update(CoEps=eps, nuT=viscosity, tke=tke, mixingLengths=distance, Lm=lm, iturbflag=turbulence_flags)
        if include_gradients:
            fields.update({f'G{axis}{direction}': gradient[..., i, j] for i, axis in enumerate('xyz')
                           for j, direction in enumerate('xyz')})
        for key, field in fields.items():
            invalid = np.isinf(field) if self.native_zero_mixing_nan else ~np.isfinite(field)
            if invalid.any():
                raise FloatingPointError(f"Non-finite turbulence {key}")
        return fields, dict(u_reference=float(uref), u_star=float(ustar), z_reference=float(zref),
                            native_quirks=self.native_quirks, nonlocal_mixing=self.nonlocal_mixing,
                            gradient_backend=self.gradient_backend, native_math=self.native_math)
