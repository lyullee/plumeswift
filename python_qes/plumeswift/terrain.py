"""Native DEM mesh construction and stairstep terrain (DTEHeightField, GPL-3).

RasterIO crop offsets, running minimum, zero-valued out-of-raster vertices and
setDomain's nonnegative-vertex origin normalization follow the pinned source.
Cut-cell terrain is a separate component and is not replaced by stairsteps.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import re
import numpy as np
from .triangle_mesh import TriangleMesh


def _normalize_native_vertices(triangles):
    triangles = np.asarray(triangles, dtype=np.float32).copy()
    for coordinate in range(3):
        values = triangles[..., coordinate]
        nonnegative = values[values >= 0]
        minimum = float(nonnegative.min()) if len(nonnegative) else 99999999.
        triangles[..., coordinate] -= minimum
    return triangles


@dataclass(frozen=True)
class GridTerrain:
    """Triangle mesh and cut-cell corners from elevations at horizontal nodes."""
    heights: object
    halo_x: float = 0.
    halo_y: float = 0.

    def describe(self):
        values = np.asarray(self.heights, dtype='<f8')
        return dict(type='native-height-array', shape=list(values.shape), halo_x=self.halo_x, halo_y=self.halo_y,
                    heights_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def mesh(self, domain):
        nx, ny, _ = domain.cells; dx, dy, _ = domain.spacing
        heights = np.asarray(self.heights, dtype=float)
        if heights.shape != (ny, nx) or not np.isfinite(heights).all():
            raise ValueError('Native height array requires one finite height per input horizontal node')
        yy, xx = np.meshgrid(np.arange(ny)*dy+self.halo_y, np.arange(nx)*dx+self.halo_x, indexing='ij')
        points = np.stack([xx, yy, heights], axis=-1).astype(np.float32)
        a, b = points[:-1, :-1], points[:-1, 1:]
        c, d = points[1:, :-1], points[1:, 1:]
        triangles = np.stack([np.stack([a, b, c], axis=-2), np.stack([c, b, d], axis=-2)], axis=-3).reshape(-1, 3, 3)
        return TriangleMesh(_normalize_native_vertices(triangles))

    def cut_geometry(self, domain, halo):
        """Return the core triangle mesh and its four corner heights on the halo grid."""
        from .domain import Domain

        height = np.asarray(self.heights, dtype=np.float32)
        hx, hy = halo.cells(domain.spacing)
        nx, ny, nz = domain.cells
        core_nx, core_ny = nx-2*hx, ny-2*hy
        if height.shape != (core_ny, core_nx):
            raise ValueError('Height array must match the horizontal domain before halo padding')
        mesh = self.mesh(Domain((core_nx, core_ny, nz), domain.spacing, domain.vertical_widths))
        minimum = np.min(height[height >= 0], initial=np.float32(99999999.))
        height = np.maximum(height-minimum, 0.)
        core = np.stack((height[:-1, :-1], height[1:, :-1], height[1:, 1:], height[:-1, 1:]), axis=-1)
        corners = np.zeros((ny-1, nx-1, 4), dtype=np.float32)
        corners[hy:hy+core_ny-1, hx:hx+core_nx-1] = core
        return mesh, corners


@dataclass(frozen=True)
class DEMTerrain:
    path: str
    origin_flag: int = 0
    utm_x: float = 0.
    utm_y: float = 0.
    distance_x: float = 0.
    distance_y: float = 0.
    target_epsg: int | None = None
    archive_member: str | None = None
    gzip_encoded: bool = False

    @classmethod
    def from_input(cls, paths, filename, **options):
        """Resolve a raster, gzip stream, or member of a local ZIP dataset."""
        filename = str(filename)
        if filename.startswith('/vsigzip/'):
            source = filename.removeprefix('/vsigzip/')
            if not source:
                raise ValueError('DEM /vsigzip/ input requires a local compressed raster')
            return cls(str(Path(source).resolve(strict=True)), gzip_encoded=True, **options)
        vsi_zip = filename.startswith('/vsizip/')
        virtual = filename.replace('\\', '/').removeprefix('/vsizip/')
        archive = member = ''
        if vsi_zip and virtual.startswith('{'):
            archive, _, member = virtual[1:].partition('}/')
        else:
            suffix = re.search(r'\.zip(?=/)', virtual, re.IGNORECASE)
            if suffix is not None:
                archive, member = virtual[:suffix.end()], virtual[suffix.end()+1:]
        if archive and member:
            # QES passes absolute VSI paths to GDAL unchanged. A relative
            # archive inside a VSI path is relative to the working directory.
            archive_path = Path(archive).resolve(strict=True) if vsi_zip else paths.resolve(archive)
            if not vsi_zip and archive_path.is_dir():
                return cls(str(paths.resolve(filename)), **options)
            member = PurePosixPath(member)
            if not member.parts or member.is_absolute() or '..' in member.parts:
                raise ValueError('DEM ZIP member must identify a raster inside the archive')
            return cls(str(archive_path), archive_member=member.as_posix(), **options)
        if vsi_zip:
            raise ValueError('DEM ZIP input requires an explicit raster member')
        return cls(str(paths.resolve(filename)), **options)

    def __post_init__(self):
        if self.target_epsg is not None and self.origin_flag != 1:
            raise ValueError('DEM reprojection requires originFlag=1')

    def describe(self):
        from dataclasses import asdict
        result = asdict(self)
        if self.archive_member is None:
            result.pop('archive_member')
        if not self.gzip_encoded:
            result.pop('gzip_encoded')
        with Path(self.path).open('rb') as source:
            result['sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        return result

    def mesh(self, domain):
        return self._geometry(domain)[0]

    def cut_geometry(self, domain, halo):
        mesh, corners, _ = self._geometry(domain, halo)
        return mesh, corners

    def geometry_with_datum(self, domain, halo=None):
        return self._geometry(domain, halo)

    def _geometry(self, domain, halo=None):
        import rasterio
        nx, ny, _ = domain.cells; dx, dy, _ = domain.spacing
        source = self.path
        if self.archive_member is not None:
            source = f'/vsizip/{{{Path(self.path).as_posix()}}}/{self.archive_member}'
        elif self.gzip_encoded:
            source = f'/vsigzip/{Path(self.path).as_posix()}'
        with rasterio.open(source) as raster:
            if self.target_epsg is None:
                raw, px, py = self._read_native_grid(raster, domain)
                scale, offset = raster.scales[0], raster.offsets[0]
                nodata = raster.nodata
            else:
                from .dem_projection import project_dem
                raw = project_dem(raster, domain, self.target_epsg, (self.utm_x, self.utm_y))
                px, py = dx, dy
                scale, offset, nodata = 1., 0., None
        cropped_h, cropped_w = raw.shape
        step_x, step_y = np.float32(dx)/np.float32(px), np.float32(dy)/np.float32(py)
        def indices(limit, step):
            current = np.float32(0)
            while current < limit:
                yield current
                following = np.float32(current+step)
                if following <= current:
                    raise FloatingPointError('DEM stepping no longer advances in native float precision')
                current = following
        running_min = 100000.
        def query(j, k):
            # queryHeight stores the relative height in float before applying
            # the GDAL band scale and offset in double precision.
            value = (0. if j >= cropped_w or k >= cropped_h
                     else float(np.float32(float(raw[cropped_h-k-1, j])-running_min)))
            if value < 0 or np.isnan(abs(value)):
                value = 0.
            if nodata is not None and abs(value-nodata) < 1e-6:
                return 0.
            return float(np.float32(value*scale+offset))
        triangles = []
        for row in indices(cropped_h, step_y):
            for col in indices(cropped_w, step_x):
                r, q = int(row), int(col)
                running_min = min(running_min, float(raw[cropped_h-r-1, q]))
                # Coordinate assignments and height queries retain float32.
                x0, x1 = float(np.float32(col*px)), float(np.float32(np.float32(col+step_x)*px))
                y0, y1 = float(np.float32(row*py)), float(np.float32(np.float32(row+step_y)*py))
                a = (x0, y0, query(q, r)); b = (x1, y0, query(int(np.float32(col+step_x)), r))
                c = (x0, y1, query(q, int(np.float32(row+step_y))))
                d = (x1, y1, query(int(np.float32(col+step_x)), int(np.float32(row+step_y))))
                triangles.extend([(a, b, c), (c, b, d)])
        triangles = np.asarray(triangles, dtype=np.float32)
        mesh = TriangleMesh(_normalize_native_vertices(triangles))
        if halo is None:
            return mesh, None, running_min
        # setCells uses the final running raster minimum and the pre-shift
        # extrema of the mesh. The native domain accessor aliases dy to dx.
        valid = triangles[..., 2]
        z_min = np.min(valid[valid >= 0], initial=np.float32(99999999.))
        z_max = np.max(valid[valid < np.float32(99999999.)], initial=np.float32(0.))
        cell_dx = np.float32(dx)
        hx, hy = halo.cells((dx, dx))
        x = np.arange(nx-1)-hx
        y = np.arange(ny-1)-hy
        xx, yy = np.meshgrid(x, y)
        active = ((xx >= 0) & (yy >= 0) &
                  (xx <= int(np.float32(cropped_w)*np.float32(px)/cell_dx)) &
                  (yy <= int(np.float32(cropped_h)*np.float32(py)/cell_dx)))
        corners = np.zeros((*xx.shape, 4), dtype=np.float32)
        for index, (di, dj) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
            cols = ((xx[active]+di).astype(np.float32)*cell_dx/np.float32(px)).astype(int)
            rows = ((yy[active]+dj).astype(np.float32)*cell_dx/np.float32(py)).astype(int)
            heights = np.array([query(col, row) for col, row in zip(cols, rows)], dtype=np.float32)
            corners[..., index][active] = np.clip(heights-z_min, 0., z_max)
        return mesh, corners, running_min

    def _read_native_grid(self, raster, domain):
        from rasterio.windows import Window

        nx, ny, _ = domain.cells
        dx, dy = (np.float32(value) for value in domain.spacing[:2])
        transform = raster.transform
        px, py = (np.float32(abs(value)) for value in (transform.a, transform.e))
        if min(px, py) <= 0:
            raise ValueError('DEM pixel spacing must be positive')
        width, height = raster.width, raster.height
        raster_x, raster_y = np.float32(width)*px, np.float32(height)*py
        ox, oy = np.float32(transform.c), np.float32(transform.f-float(raster_y))
        dem_x, dem_y = ox+raster_x, oy+raster_y
        span_x, span_y = np.float32(nx)*dx, np.float32(ny)*dy
        shift_x = shift_y = end_x = end_y = 0
        if self.origin_flag == 0:
            distance_x, distance_y = map(np.float32, (self.distance_x, self.distance_y))
            domain_x, domain_y = ox+distance_x+span_x, oy+distance_y+span_y
            if (distance_x > 0 or distance_y > 0) and distance_x < raster_x and distance_x < raster_y:
                shift_x, shift_y = int(distance_x/px), int(distance_y/py)
            if domain_x < dem_x or domain_y < dem_y:
                end_x, end_y = int((dem_x-domain_x)/px), int((dem_y-domain_y)/py)
        elif self.origin_flag == 1:
            utm_x, utm_y = map(np.float32, (self.utm_x, self.utm_y))
            domain_x, domain_y = utm_x+span_x, utm_y+span_y
            if (utm_x > ox or utm_y > oy) and utm_x < dem_x and utm_y < dem_y:
                shift_x, shift_y = int((utm_x-ox)/px), int((utm_y-oy)/py)
            if (domain_x < dem_x or domain_y < dem_y) and utm_x >= ox and utm_y >= oy:
                end_x, end_y = int((dem_x-domain_x)/px), int((dem_y-domain_y)/py)
        else:
            raise ValueError('DEM originFlag must be 0 or 1')
        cropped_w, cropped_h = width-shift_x-end_x, height-shift_y-end_y
        if min(shift_x, shift_y, end_x, end_y) < 0 or min(cropped_w, cropped_h) < 1:
            raise ValueError('Native DEM crop would be outside the raster')
        raw = raster.read(1, window=Window(shift_x, end_y, cropped_w, cropped_h), out_dtype='float32')
        return raw, float(px), float(py)


def apply_stairstep_terrain(workspace, mesh, *, mark_solid=True, halo_cells=(0, 0), native_sensor_math=False):
    """Sample the native terrain core and copy its cell/face halo borders."""
    a, c = workspace.arrays, workspace.coordinates
    hx, hy = halo_cells
    nx, ny = len(c['x']), len(c['y'])
    if min(hx, hy) < 0 or nx <= 2*hx or ny <= 2*hy:
        raise ValueError('Terrain halo leaves no cells in the DEM interior')
    x, y = c['x'][:nx-2*hx], c['y'][:ny-2*hy]
    if native_sensor_math:
        dx, dy = (np.float32(c[axis][1]-c[axis][0]) for axis in 'xy')
        # WINDSGeneralData samples i*dx + dx/2, with two float32 operations.
        x = np.arange(len(x), dtype=np.float32)*dx+dx*np.float32(.5)
        y = np.arange(len(y), dtype=np.float32)*dy+dy*np.float32(.5)
    xx, yy = np.meshgrid(x, y)
    core = np.maximum(mesh.height(xx, yy, native_math=native_sensor_math), 0.)
    z, z_face = c['z'], c['z_face']
    terrain = np.pad(core, ((hy, hy), (hx, hx)), mode='edge') if hx or hy else core
    a['terrain'] = terrain
    if native_sensor_math:
        from .domain import native_vertical_coordinates
        z, z_face = native_vertical_coordinates(a['dz_array'])
    # Source checks z[k+1] and marks cell k+1; ghost ground is already solid.
    solid = terrain[None] >= z[1:, None, None]
    if mark_solid:
        a['icellflag'][1:][solid] = 2
    a['center_id'] = np.ones(a['icellflag'].shape, dtype=np.int32)
    a['center_id'][1:][solid] = 0
    maximum = len(z)-2
    core_face = np.minimum(np.searchsorted(z_face[1:-1], core, side='left'), maximum).astype(np.int32)
    center_id = np.minimum(np.searchsorted(z[:-1], terrain, side='right'), maximum).astype(np.int32)
    ids = np.ones(a['u0'].shape[1:], dtype=np.int32)
    # Native right/top face halos copy their first unset face, initialized
    # to one. The top-right loop also reads an unset cell ID (or overflows
    # on some tall domains). Keep the defined initializer for that corner.
    ids[:ny-hy, :nx-hx] = np.pad(core_face, ((hy, 0), (hx, 0)), mode='edge') if hx or hy else core_face
    a['terrain_face_id'] = ids; a['terrain_id'] = center_id
    return mesh
