"""QES polygon stair-step flags and wall stencil; cut cells are separate.

Adapted from PolyBuilding.cpp and Wall.cpp (GPL-3.0-only).
"""
from dataclasses import dataclass, field, replace
import numpy as np


Ring = tuple[tuple[float, float], ...]


def ring_contains(vertices, x, y, *, native_ring_breaks=False):
    """Half-open ray crossing used by the native polygon column flags."""
    inside = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=bool)
    start, skip = 0, False
    for index, (first, second) in enumerate(zip(vertices[:-1], vertices[1:])):
        if skip:
            start, skip = index+1, False
            continue
        if native_ring_breaks:
            skip = np.array_equal(second, vertices[start])
        if first[1] == second[1]:
            continue
        crossing = (first[1] <= y) != (second[1] <= y)
        ray = first[0]+(y-first[1])/(second[1]-first[1])*(second[0]-first[0])
        inside ^= crossing & (x < ray)
    return inside


def _validate_ring(ring):
    vertices = np.asarray(ring, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Building requires a finite, closed polygon")
    if not np.array_equal(vertices[0], vertices[-1]):
        raise ValueError("Polygon must repeat the first vertex at the end")


def _ring_covers(vertices, x, y):
    inside = ring_contains(vertices, x, y)
    for first, second in zip(vertices[:-1], vertices[1:]):
        dx, dy = second-first
        if dx == 0 and dy == 0:
            continue
        u, v = x-first[0], y-first[1]
        cross = u*dy-v*dx
        along = u*dx+v*dy
        limit = dx*dx+dy*dy
        tolerance = np.finfo(float).eps*8*max(1., limit)
        inside |= (abs(cross) <= tolerance) & (along >= -tolerance) & (along <= limit+tolerance)
    return inside.all()


def polygon_parts(rings):
    """Group closed rings by containment independently of their winding."""
    rings = tuple(np.asarray(ring, dtype=float) for ring in rings)
    areas = []
    for ring in rings:
        _validate_ring(ring)
        x, y = (ring-ring[0]).T
        area = .5*abs(np.dot(x[:-1], y[1:])-np.dot(y[:-1], x[1:]))
        if area == 0:
            raise ValueError('Polygon ring has zero area')
        areas.append(area)
    parents, depths = {}, {}
    for index in sorted(range(len(rings)), key=areas.__getitem__, reverse=True):
        x, y = rings[index][:-1].T
        candidates = (parent for parent in parents if areas[parent] > areas[index]
                      and _ring_covers(rings[parent], x, y))
        parent = min(candidates, key=areas.__getitem__, default=None)
        parents[index] = parent
        depths[index] = 0 if parent is None else depths[parent]+1
    # Even-depth rings are exteriors. An island inside a hole is
    # emitted as another exterior with its own immediate hole rings.
    return tuple((ring, tuple(rings[child] for child, parent in parents.items() if parent == index))
                 for index, ring in enumerate(rings) if depths[index] % 2 == 0)


def vertex_parts(vertices):
    """Split native XML vertex lists at explicit ring closures."""
    rings = []
    start = 0
    for index in range(1, len(vertices)):
        if index-start >= 3 and np.array_equal(vertices[index], vertices[start]):
            rings.append(vertices[start:index+1])
            start = index+1
    if start < len(vertices):
        tail = vertices[start:]
        rings.append(np.vstack((tail, tail[0])))
    def tuples(ring):
        return tuple(map(tuple, ring))
    return tuple((tuples(exterior), tuple(map(tuples, holes))) for exterior, holes in polygon_parts(rings))


def translate_footprint(model, halo):
    rings = {'vertices': halo.translate(model.vertices)}
    if getattr(model, 'holes', ()):
        rings['holes'] = tuple(halo.translate(ring) for ring in model.holes)
    native_closure_vertices = getattr(model, 'native_closure_vertices', None)
    if native_closure_vertices is not None:
        rings['native_closure_vertices'] = halo.translate(native_closure_vertices)
    if getattr(model, 'detached_parts', ()):
        rings['detached_parts'] = tuple(
            (halo.translate(exterior), tuple(halo.translate(hole) for hole in holes))
            for exterior, holes in model.detached_parts)
    return replace(model, **rings)


@dataclass(frozen=True)
class PolygonBuilding:
    vertices: tuple[tuple[float, float], ...]
    height: float
    base_height: float = 0.
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    terrain_sample_vertices: tuple[tuple[float, float], ...] | None = None
    native_closure_vertices: tuple[tuple[float, float], ...] | None = field(default=None, kw_only=True)
    detached_parts: tuple[tuple[Ring, tuple[Ring, ...]], ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        for ring in (self.vertices, *self.holes):
            _validate_ring(ring)
        if self.native_closure_vertices is not None:
            _validate_ring(self.native_closure_vertices)
        for exterior, holes in self.detached_parts:
            for ring in (exterior, *holes):
                _validate_ring(ring)
        if not np.isfinite([self.height, self.base_height]).all() or self.height <= 0:
            raise ValueError("Invalid building height")

    @classmethod
    def rectangle(cls, x, y, length, width, height, rotation=0., base_height=0., *, native_math=False):
        if length <= 0 or width <= 0:
            raise ValueError("Rectangle dimensions must be positive")
        scalar = np.float32 if native_math else float
        theta = np.deg2rad(rotation)
        if native_math:
            theta = float(scalar(float(scalar(rotation))*(np.pi/180.)))
        origin = np.array([x, y], dtype=scalar)
        transverse = scalar(width)*np.array([-np.sin(theta), np.cos(theta)], dtype=scalar)
        along = scalar(length)*np.array([np.cos(theta), np.sin(theta)], dtype=scalar)
        # Native polyVert stores float coordinates; retain that geometry before
        # integer ray bounds are formed by the empirical parameterizations.
        vertices = np.asarray([origin, origin+transverse, origin+transverse+along, origin+along, origin], dtype=np.float32)
        return cls(tuple(map(tuple, vertices.astype(float))), height, base_height)

    def physical_parts(self):
        """Return the separate solid footprints of this logical building."""
        if not self.detached_parts:
            return (self,)
        return tuple(replace(self, vertices=exterior, holes=holes, detached_parts=(), native_closure_vertices=None)
                     for exterior, holes in ((self.vertices, self.holes), *self.detached_parts))

    def intersects_horizontal_domain(self, coordinates):
        """Test footprint bounds before building geometry or wind sampling."""
        if self.detached_parts:
            return any(part.intersects_horizontal_domain(coordinates) for part in self.physical_parts())
        vertices = np.asarray(self.vertices)
        limit = (coordinates['x_face'][-1], coordinates['y_face'][-1])
        return bool(np.all(vertices.max(axis=0) > 0) and np.all(vertices.min(axis=0) < limit))

    def flag(self, workspace, number=0, *, mark_solid=True, native_dy_alias=False, native_math=False,
             native_wrapped_footprint=False):
        if self.detached_parts:
            for part in self.physical_parts():
                part.flag(workspace, number, mark_solid=mark_solid, native_dy_alias=native_dy_alias,
                          native_math=native_math, native_wrapped_footprint=native_wrapped_footprint)
            return
        a, c = workspace.arrays, workspace.coordinates
        vertices = np.asarray(self.vertices, dtype=np.float32 if native_math else float)
        # Keep the original polygon edges for cut-cell and wind parameterizations.
        dx = c['x'][1]-c['x'][0]; dy = dx if native_dy_alias else c['y'][1]-c['y'][0]
        base, height = self.base_height, self.height
        z, faces = c['z'], c['z_face']
        if native_math:
            from .domain import native_vertical_coordinates
            base, height = np.float32(base), np.float32(height)
            z, faces = native_vertical_coordinates(a['dz_array'])
        effective = base+height
        starts = np.flatnonzero(base <= faces[2:])
        ends = np.flatnonzero(effective < z[1:])
        if not len(starts) or not len(ends):
            raise ValueError("Building top/base must be inside the vertical domain")
        k_start, k_end = 1+starts[0], 1+ends[0]
        x = c['x']
        y = (np.arange(len(c['y']))+.5)*dy if native_dy_alias else c['y']
        if native_math:
            x = ((np.arange(len(x))+.5)*float(np.float32(dx))).astype(np.float32)
            y = ((np.arange(len(y))+.5)*float(np.float32(dy))).astype(np.float32)
        xx, yy = np.meshgrid(x, y)
        early_close = native_math and np.any(np.all(vertices[1:-1] == vertices[0], axis=1))
        inside = ring_contains(vertices, xx, yy, native_ring_breaks=early_close)
        if early_close:
            # A rounded GIS vertex can close the first sub-ring early. The
            # native ray scan then skips an edge and is bounded by its index box.
            spacing = np.asarray([dx, dy], dtype=np.float32)
            lower = np.trunc(vertices.min(axis=0)/spacing).astype(int)
            upper = np.trunc(vertices.max(axis=0)/spacing+1).astype(int)
            i, j = np.arange(len(x))[None, :], np.arange(len(y))[:, None]
            inside &= (lower[0] <= i) & (i <= upper[0]) & (lower[1] <= j) & (j <= upper[1])
        holes = tuple(np.asarray(ring, dtype=vertices.dtype) for ring in self.holes)
        for ring in holes:
            inside &= ~ring_contains(ring, xx, yy)
        if native_wrapped_footprint:
            # The staircase source tests the original x-coordinate, then
            # addresses the cell and footprint by a packed horizontal index.
            nx = len(x)
            dx = np.float32(dx)
            first = int(vertices[:, 0].min()/dx)
            last = int(vertices[:, 0].max()/dx)+1
            if first < 0 or last >= nx:
                low, high = vertices[:, 1].min(), vertices[:, 1].max()
                for j in np.flatnonzero((y >= low) & (y < high)):
                    offset = int(j)*nx
                    columns = np.r_[np.arange(max(first, -offset), 0),
                                    np.arange(nx, min(last, inside.size-1-offset)+1)]
                    centers = ((columns+.5)*float(dx)).astype(np.float32)
                    covered = ring_contains(vertices, centers, y[j], native_ring_breaks=early_close)
                    for ring in holes:
                        covered &= ~ring_contains(ring, centers, y[j])
                    inside.flat[offset+columns[covered]] = True
        selected = np.zeros(a['icellflag'].shape, dtype=bool)
        selected[k_start:k_end] = inside
        if mark_solid:
            a['icellflag'][selected & (a['icellflag'] != 7)] = 0
        a['ibuilding_flag'][selected & (a['ibuilding_flag'] == -1)] = number
        a['icellflag_footprint'][inside] = 0


def wall_stencil(workspace, *, native_dy_alias=False, native_outer_wrap=True, native_math=False):
    """Define stairstep faces then dimensionalize the six native coefficients.

    Domain.h getDomainSize() returns dx twice in the pinned upstream version.
    native_dy_alias reproduces that arithmetic where required for comparisons.
    The default uses the actual y spacing. No cut-cell approximation is made.
    """
    a, c = workspace.arrays, workspace.coordinates
    if np.isin(a['icellflag'], [7, 8]).any():
        raise NotImplementedError("Cut-cell wall matching is not part of this stair-step component")
    for key in 'efghmn':
        a[key] = np.ones(a['icellflag'].shape)
    block_solid_neighbours(a, native_outer_wrap)
    scale_wall_coefficients(a, c, native_dy_alias, native_math=native_math)


def block_solid_neighbours(a, native_outer_wrap=True):
    solid = np.isin(a['icellflag'], [0, 2]); fluid = ~solid
    # Native flattened horizontal indexing crosses a row/plane at positive
    # outer edges. Retain that observable coefficient layout by default.
    for key, axis, offset in [('e', 2, 1), ('f', 2, -1), ('g', 1, 1), ('h', 1, -1), ('m', 0, 1), ('n', 0, -1)]:
        flat_offset = offset*([solid.shape[1]*solid.shape[2], solid.shape[2], 1][axis])
        neighbour = np.roll(solid.ravel(), -flat_offset).reshape(solid.shape)
        valid = np.ones(solid.shape, dtype=bool)
        edge = [slice(None)]*3; edge[axis] = -1 if offset == 1 else 0
        if axis == 0 or offset < 0 or not native_outer_wrap:
            valid[tuple(edge)] = False
        valid[0] = valid[-1] = False
        a[key][fluid & neighbour & valid] = 0.


def scale_wall_coefficients(a, c, native_dy_alias, *, native_math=False):
    dx = c['x'][1]-c['x'][0]; dy = dx if native_dy_alias else c['y'][1]-c['y'][0]
    # Wall::solverCoefficients performs both the denominator products and
    # division in float. The solver may otherwise see a different coefficient.
    dtype = np.float32 if native_math else float
    dx, dy = dtype(dx), dtype(dy)
    dz = np.asarray(a['dz_array'], dtype=dtype)
    lower = (dz[1:-1]*dtype(.5)*(dz[1:-1]+dz[:-2]))[:, None, None]
    upper = (dz[1:-1]*dtype(.5)*(dz[1:-1]+dz[2:]))[:, None, None]
    for key, denominator in zip('efghmn', (dx*dx, dx*dx, dy*dy, dy*dy, upper, lower), strict=True):
        values = a[key][1:-1]
        np.divide(values, denominator, out=values, dtype=dtype)


def zero_solid_faces(workspace):
    a = workspace.arrays
    solid = np.isin(a['icellflag'], [0, 2])
    solid[:, 0] = False; solid[:, :, 0] = False
    k, j, i = np.where(solid)
    a['u0'][k, j, i] = a['u0'][k, j, i+1] = 0.
    a['v0'][k, j, i] = a['v0'][k, j+1, i] = 0.
    a['w0'][k, j, i] = a['w0'][k+1, j, i] = 0.
