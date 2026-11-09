"""WRF/SFIRE grid, wind snapshots, and atmospheric stations."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .domain import Domain, native_vertical_coordinates
from .io import _open_netcdf
from .measured_profile import MeasuredProfile
from .meteorology import Station
from .terrain import GridTerrain
from .utm import from_utm, from_wrf_utm, to_utm


_ROUGHNESS = (.1, .5, .5, .5, .5, .5, .1, .1, .15, .15, .075, .3, .075, .5, .065, .01,
              .065, .0001, .15, .1, .06, .0001)
_FIRE_MESH_VARIABLES = ('ZSF', 'HGT', 'FXLONG')
_FIRE_WIND_VARIABLES = ('HT_FMW', 'U0_FMW', 'V0_FMW', 'W0_FMW', 'CHSUM0_FMW')


class _WRFFieldMissingData(ValueError):
    """NetCDF masked a field value that a live peer may still be writing."""


def _require_fire_mesh_variables(nc, names, mode):
    missing = [name for name in names if name not in nc.variables]
    if missing:
        message = f'QES {mode} requires a QES-compatible WRF-SFIRE file. Missing variables: {", ".join(missing)}.'
        if {'U', 'V', 'PH', 'PHB', 'HGT'} <= nc.variables.keys():
            message += ' Atmospheric WRF output can be read separately through the WRF + DEM input mode.'
        raise ValueError(message)


def _field(nc, name, rank, key, *, stored=False):
    if name not in nc.variables or nc[name].ndim != rank:
        raise ValueError(f'WRF requires rank-{rank} variable {name}')
    variable = nc[name]
    if stored:
        variable.set_auto_scale(False)
    values = variable[key]
    if np.ma.is_masked(values):
        raise _WRFFieldMissingData(f'WRF variable {name} contains missing data')
    if not np.isfinite(values).all():
        raise ValueError(f'WRF variable {name} contains nonfinite data')
    return np.asarray(values)


def frame_checksum(*fields):
    """Bitwise XOR of the float32 field values, as in WRFInput.cpp."""
    checksum = 0
    for field in fields:
        checksum ^= int(np.bitwise_xor.reduce(np.asarray(field, dtype=np.float32).ravel().view(np.int32), initial=0))
    return checksum


def _fire_wind_fields(nc, record, nx, ny):
    # The native reader uses the U0_FMW level count for all four hyperslabs.
    wind = nc['U0_FMW']
    if wind.ndim != 4 or not wind.shape[1]:
        raise ValueError('WRF fire mesh requires a nonempty rank-4 U0_FMW')
    nz = wind.shape[1]
    heights = _field(nc, 'HT_FMW', 2, (record, slice(nz)), stored=True)
    selection = (record, slice(nz), slice(ny), slice(nx))
    winds = tuple(_field(nc, name, 4, selection, stored=True) for name in ('U0_FMW', 'V0_FMW', 'W0_FMW'))
    if heights.shape != (nz,) or any(field.shape != (nz, ny, nx) for field in winds):
        raise ValueError('WRF fire mesh coordinates and wind shapes disagree')
    return heights, winds


def _native_fire_face_wind(first, second, heights, vertical, lower, upper, j, i):
    """Evaluate the original float32 interpolation on both sides of a face."""
    a, b = first[lower, j, i], second[lower, j, i]
    width = heights[upper]-heights[lower]
    offset = vertical-heights[lower]
    within = (vertical > heights[0]) & (vertical < heights[-1])
    for values, nodes in ((a, first), (b, second)):
        values[within] += (offset[within]*(nodes[upper, j, i][within]-values[within])
                           / width[within])
    below = (vertical >= 0) & (vertical <= heights[0])
    a[below] = vertical[below]*first[0, j, i][below]/heights[0]
    b[below] = vertical[below]*second[0, j, i][below]/heights[0]
    wind = np.float32(.5)*(a+b)
    wind[vertical < 0] = 0
    return wind


@dataclass(frozen=True)
class FireMeshGrid:
    path: str
    cells: tuple
    spacing: tuple
    terrain: np.ndarray

    @classmethod
    def read(cls, path):
        with _open_netcdf(path) as nc:
            return cls._read_grid(nc, path)

    @staticmethod
    def _read_grid(nc, path):
        _require_fire_mesh_variables(nc, _FIRE_MESH_VARIABLES, 'fire-mesh input')
        terrain_shape = nc['ZSF'].shape[1:]
        atm_y, atm_x = nc['HGT'].shape[1:]
        sy, sx = terrain_shape[0]//(atm_y+1), terrain_shape[1]//(atm_x+1)
        if not sx or not sy:
            raise ValueError('WRF fire mesh must be a refinement of the atmospheric mesh')
        ny, nx = nc['FXLONG'].shape[1:]
        ny, nx = ny-sy, nx-sx
        if min(nx, ny) < 3:
            raise ValueError('WRF fire mesh needs at least three cells along each horizontal axis')
        # WRFInput stores the divided double attributes in float fire-mesh metrics.
        spacing = float(np.float32(float(nc.DX)/sx)), float(np.float32(float(nc.DY)/sy)), 12.
        terrain = _field(nc, 'ZSF', 3, (0, slice(ny), slice(nx)), stored=True)
        if terrain.shape != (ny, nx):
            raise ValueError('WRF fire mesh coordinates and terrain shapes disagree')
        # The native maximum accumulator starts at the smallest positive
        # double, so a wholly below-sea-level mesh uses zero as its maximum.
        height_range = np.float32(max(float(terrain.max()), 0.)-float(terrain.min()))
        nz = int(np.floor(1.1*float(height_range)/spacing[2]))+1
        terrain.setflags(write=False)
        return FireMeshGrid(str(path), (nx, ny, nz), spacing, terrain)

    def describe(self):
        return dict(type='wrf-fire-mesh', path=self.path, cells=self.cells, spacing=self.spacing)

    def domain(self, halo, widths=None):
        hx, hy = halo.cells(self.spacing)
        nx, ny, nz = self.cells
        return Domain((nx+2*hx, ny+2*hy, nz), self.spacing, widths)

    def mesh(self, domain):
        return GridTerrain(self.terrain).mesh(Domain(self.cells, self.spacing))

    def cut_geometry(self, domain, halo):
        """Supply fire-mesh node heights to the shared QES cut-cell pass."""
        return GridTerrain(self.terrain).cut_geometry(domain, halo)


class WRFChecksumError(ValueError):
    """A frame header and the corresponding wind data disagree."""


@dataclass(frozen=True)
class FireMeshInput(FireMeshGrid):
    heights: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    record: int

    @classmethod
    def read(cls, path):
        with _open_netcdf(path) as nc:
            _require_fire_mesh_variables(nc, _FIRE_MESH_VARIABLES+_FIRE_WIND_VARIABLES, 'fire-mesh input')
            grid = cls._read_grid(nc, path)
            return cls.read_record(nc, grid, nc['U0_FMW'].shape[0]-1)

    @classmethod
    def read_record(cls, nc, grid, record):
        if not 0 <= record < nc['U0_FMW'].shape[0]:
            raise ValueError(f'WRF fire mesh has no wind record {record}')
        nx, ny, _ = grid.cells
        # The handshake checksum is computed from the stored float32 values.
        heights, winds = _fire_wind_fields(nc, record, nx, ny)
        stored = int(_field(nc, 'CHSUM0_FMW', 1, record, stored=True))
        if heights[0] <= 0 or np.any(np.diff(heights) <= 0):
            raise ValueError('WRF fire mesh heights must be positive and increasing')
        if frame_checksum(*winds) != stored:
            raise WRFChecksumError(f'WRF wind frame checksum mismatch at record {record}')
        for array in (heights, *winds):
            array.setflags(write=False)
        return cls(grid.path, grid.cells, grid.spacing, grid.terrain, heights, *winds, record)

    def describe(self):
        return dict(super().describe(), record=self.record, checksum=frame_checksum(self.u, self.v, self.w))

    def initialize(self, workspace, halo, *, native_math=False):
        arrays, coord = workspace.arrays, workspace.coordinates
        nz, ny, nx = arrays['u0'].shape
        sx, sy, _ = self.cells
        hx, hy = halo.cells(self.spacing)
        z, z_face = (native_vertical_coordinates(arrays['dz_array']) if native_math
                     else (coord['z'], coord['z_face']))
        ground = z_face[arrays['terrain_face_id'][hy+1:hy+sy, hx+1:hx+sx]]
        j, i = np.indices(ground.shape)
        height = self.heights.astype(np.float32, copy=False) if native_math else self.heights
        ii = np.clip(np.arange(nx)-hx-1, 0, sx-2)
        jj = np.clip(np.arange(ny)-hy-1, 0, sy-2)
        fields = (arrays['u0'], arrays['v0'])
        if native_math:
            u, v = (component.astype(np.float32, copy=False) for component in (self.u, self.v))
            pairs = ((u[:, 1:, 1:], u[:, 1:, :-1]), (v[:, 1:, 1:], v[:, :-1, 1:]))
        else:
            u = .5*(self.u[:, 1:, 1:].astype(float)+self.u[:, 1:, :-1])
            v = .5*(self.v[:, 1:, 1:].astype(float)+self.v[:, :-1, 1:])
        for field in (*fields, arrays['w0']):
            field.fill(0.)
        for k in range(1, nz-1):
            vertical = z[k]-ground
            upper = np.minimum(np.searchsorted(height, vertical, side='right').clip(1), len(height)-1)
            lower = np.maximum(0, upper-1)
            if native_math:
                lower[vertical >= height[-1]] = len(height)-1
                for field, (first, second) in zip(fields, pairs):
                    interior = _native_fire_face_wind(first, second, height, vertical, lower, upper, j, i)
                    field[k] = interior[jj[:, None], ii]
                continue
            width = height[upper]-height[lower]
            ratio = np.divide(vertical-height[lower], width, out=np.zeros_like(vertical), where=width != 0)
            ratio = np.clip(ratio, 0., 1.)
            ramp = np.clip(vertical/height[0], 0., 1.)
            for field, values in zip(fields, (u, v)):
                interior = ramp*((1-ratio)*values[lower, j, i]+ratio*values[upper, j, i])
                field[k] = interior[jj[:, None], ii]
        for field in fields:
            field[-1] = field[-2]


def _covers_shape(shape, expected):
    return len(shape) == len(expected) and all(size >= minimum for size, minimum in zip(shape, expected))


def _atmospheric_wind_window(nc, shape, rows, columns, *, record=0, require_land=True, stored=False):
    ny, nx = shape
    names = ('HGT', 'LU_INDEX', 'PH', 'PHB', 'U', 'V')
    dimensions = {name: nc[name].shape[1:] for name in names if name in nc.variables}
    levels = dimensions.get('U', ())
    if len(levels) != 3 or levels[0] == 0:
        raise ValueError('WRF atmospheric input requires a nonempty staggered U field')
    nz = int(getattr(nc, 'BOTTOM-TOP_GRID_DIMENSION', levels[0]+1))-1 if stored else levels[0]
    expected = dict(HGT=(ny, nx), LU_INDEX=(ny, nx), PH=(nz+1, ny, nx), PHB=(nz+1, ny, nx),
                    U=(nz, ny, nx+1), V=(nz, ny+1, nx))
    if not require_land and 'LU_INDEX' not in dimensions:
        del expected['LU_INDEX']
    matches = (all(_covers_shape(dimensions.get(name, ()), size) for name, size in expected.items())
               if stored else dimensions == expected)
    if nz < 1 or not matches:
        raise ValueError('WRF atmosphere and staggered grid sizes disagree')
    j, i = slice(int(rows.min()), int(rows.max())+1), slice(int(columns.min()), int(columns.max())+1)
    ground = _field(nc, 'HGT', 3, (0, j, i))
    land = _field(nc, 'LU_INDEX', 3, (0, j, i)) if 'LU_INDEX' in dimensions else np.zeros_like(ground)
    ph, phb = (_field(nc, name, 4, (record, slice(nz+1), j, i)) for name in ('PH', 'PHB'))
    u = _field(nc, 'U', 4, (record, slice(nz), j, slice(i.start, i.stop+1)))
    v = _field(nc, 'V', 4, (record, slice(nz), slice(j.start, j.stop+1), i))
    geopotential = (ph.astype(float)+phb)/9.81
    # De-stagger each component along its face axis before forming profiles.
    return (ground, land, .5*(geopotential[:-1]+geopotential[1:]),
            .5*(u[..., 1:].astype(float)+u[..., :-1]),
            .5*(v[:, 1:, :].astype(float)+v[:, :-1, :]), j.start, i.start)


def atmospheric_stations(path, domain, halo, origin, zone, sample, *, coordinate_mode='geographic', stored=False):
    """Convert the first WRF record to flag-4 profiles using native bounds for stored reads."""
    if coordinate_mode not in ('geographic', 'ideal-local'):
        raise ValueError('WRF coordinates must be geographic or ideal-local')
    if sample < 1 or (coordinate_mode == 'geographic' and not 1 <= zone <= 60):
        raise ValueError('WRF atmospheric input requires a positive sampling stride and a valid UTM zone for geographic coordinates')
    nx, ny, _ = domain.cells
    dx, dy, _ = domain.spacing
    ideal = coordinate_mode == 'ideal-local'
    sw, ne = origin, (origin[0]+nx*dx, origin[1]+ny*dy)
    if not ideal:
        convert = from_wrf_utm if stored else from_utm
        sw, ne = convert(*sw, zone), convert(*ne, zone)
    with _open_netcdf(Path(path)) as nc:
        coordinate_record = 0
        if stored:
            nc.set_auto_scale(False)
            lon_variable = nc.variables.get('XLONG')
            if lon_variable is None or lon_variable.ndim != 3:
                raise ValueError('WRF requires rank-3 variable XLONG')
            height, width = lon_variable.shape[1:]
            shape = (int(getattr(nc, 'SOUTH-NORTH_GRID_DIMENSION', height+1))-1,
                     int(getattr(nc, 'WEST-EAST_GRID_DIMENSION', width+1))-1)
            if min(shape) < 1 or any(not _covers_shape(nc[name].shape[1:], shape)
                                     for name in ('XLONG', 'XLAT') if name in nc.variables):
                raise ValueError('WRF atmosphere coordinates disagree with the declared grid')
            coordinate_record = (0, slice(shape[0]), slice(shape[1]))
        lon, lat = (_field(nc, name, 3, coordinate_record) for name in ('XLONG', 'XLAT'))
        if lon.shape != lat.shape:
            raise ValueError('WRF atmosphere coordinates disagree')
        x_sample, y_sample = lon[::sample, ::sample], lat[::sample, ::sample]
        rows, columns = np.nonzero((x_sample > sw[0]) & (x_sample < ne[0]) &
                                   (y_sample > sw[1]) & (y_sample < ne[1]))
        if not rows.size:
            raise ValueError('WRF has no atmospheric stations in this domain')
        rows *= sample
        columns *= sample
        ground, land, z, u, v, row_start, column_start = _atmospheric_wind_window(
            nc, lon.shape, rows, columns, stored=stored)
    stations = []
    for row, column in zip(rows, columns, strict=True):
        x, y = lon[row, column], lat[row, column]
        if not ideal:
            x, y, _ = to_utm(x, y, wrf_legacy=True)
        j, i = row-row_start, column-column_start
        absolute = z[:, j, i]
        relative = absolute-ground[j, i]
        keep = (absolute < 270.) & (relative > 0.)
        if not keep.any():
            continue
        if stored:
            # Native WRF retains levels below its absolute altitude cap,
            # even when the WRF terrain datum places them below the ground.
            keep = absolute < 270.
        ux, vy = u[keep, j, i], v[keep, j, i]
        speed = np.hypot(ux, vy)
        if stored:
            # WRFInput uses atan(V/U) with the nonpositive-U quadrant even
            # when U is +0. Keep a finite placeholder for calm measurements.
            with np.errstate(divide='ignore', invalid='ignore'):
                direction = np.where(ux > 0, 270., 90.)-np.rad2deg(np.arctan(vy/ux))
            direction = np.where(speed == 0, 270., direction)
        else:
            direction = np.mod(270-np.rad2deg(np.arctan2(vy, ux)), 360.)
        land_use = int(land[j, i])
        roughness = _ROUGHNESS[land_use] if 0 < land_use < len(_ROUGHNESS) else .1
        profile = MeasuredProfile(tuple(relative[keep]), tuple(speed), tuple(direction), roughness,
                                  allow_negative_heights=stored)
        stations.append(Station(*halo.translate((float(x)-origin[0], float(y)-origin[1])), profile))
    if not stations:
        raise ValueError('WRF has no atmospheric stations with usable heights in this domain')
    return tuple(stations)
