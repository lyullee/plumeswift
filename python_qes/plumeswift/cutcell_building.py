"""QES cut-cell openings for vertically extruded building polygons (GPL-3)."""
import numpy as np

from .domain import native_vertical_coordinates
from .geometry import ring_contains


def _clip_axis(polygon, axis, limit, keep_above, origin, dx, *, native_edges):
    if not len(polygon):
        return polygon
    clipped = []
    previous = polygon[-1]
    inside = (previous[axis] >= limit) if keep_above else (previous[axis] <= limit)
    for point in polygon:
        next_inside = (point[axis] >= limit) if keep_above else (point[axis] <= limit)
        if next_inside != inside:
            fraction = (limit-previous[axis])/(point[axis]-previous[axis])
            cross = previous+fraction*(point-previous)
            cross[axis] = limit
            if native_edges and axis == 1 and previous[0] == point[0] and 0 < point[0] < dx:
                # Native vertical segments use a 1e-6 x denominator.
                # Retain its float32 side intersection near the 0.05 cutoff.
                slope = np.float32((point[1]-previous[1])/1e-6)
                edge_y = np.float32(limit+origin[1])-np.float32(previous[1]+origin[1])
                cross[0] = np.float32(edge_y/slope+np.float32(previous[0]+origin[0]))-origin[0]
            clipped.append(cross)
        if next_inside:
            clipped.append(point)
        previous, inside = point, next_inside
    return np.asarray(clipped)


def _clip_footprint(vertices, origin, dx, *, native_edges):
    polygon = vertices-origin
    for axis in (0, 1):
        polygon = _clip_axis(polygon, axis, 0., True, origin, dx, native_edges=native_edges)
        polygon = _clip_axis(polygon, axis, dx, False, origin, dx, native_edges=native_edges)
    return polygon


def _cell_footprint(vertices, origin, dx, *, native_edges=True):
    polygon = _clip_footprint(vertices, origin, dx, native_edges=native_edges)
    if len(polygon) < 3:
        return 0., np.zeros(4)
    next_point = np.roll(polygon, -1, axis=0)
    area = .5*abs(np.sum(polygon[:, 0]*next_point[:, 1]-polygon[:, 1]*next_point[:, 0]))
    lengths = []
    for axis, boundary in ((0, dx), (0, 0.), (1, dx), (1, 0.)):
        face = (polygon[:, axis] == boundary) & (next_point[:, axis] == boundary)
        lengths.append(abs(np.sum(next_point[face, 1-axis]-polygon[face, 1-axis])))
    return area, np.asarray(lengths)


def _perforated_footprint(vertices, holes, origin, dx):
    area, lengths = _cell_footprint(vertices, origin, dx, native_edges=False)
    for hole in holes:
        opening, face_lengths = _cell_footprint(hole, origin, dx, native_edges=False)
        if opening > 0:
            area -= opening
            lengths -= face_lengths
    return np.clip(area, 0., dx*dx), np.clip(lengths, 0., dx)


def _distance_to_ring(vertices, center):
    segments = np.roll(vertices, -1, axis=0)-vertices
    offsets = center-vertices
    length_squared = np.sum(segments**2, axis=1)
    t = np.divide(np.sum(offsets*segments, axis=1), length_squared,
                  out=np.zeros_like(length_squared), where=length_squared > 0)
    return np.linalg.norm(offsets-np.clip(t, 0., 1.)[:, None]*segments, axis=1).min()


def _first_intersection(vertices, origin, dx):
    """Reference point used by the native volume and wall-distance formula."""
    local = vertices-origin
    for first, second in zip(local, np.roll(local, -1, axis=0)):
        if np.all((second >= 0) & (second < dx)):
            return second
        if np.all((first >= 0) & (first < dx)):
            continue
        delta = second-first
        for axis, boundary in ((0, 0.), (0, dx), (1, 0.), (1, dx)):
            if delta[axis] == 0:
                continue
            t = (boundary-first[axis])/delta[axis]
            tangent = first[1-axis]+t*delta[1-axis]
            if 0 <= t <= 1 and 0 < tangent < dx:
                return np.array((boundary, tangent)) if axis == 0 else np.array((tangent, boundary))
    return np.zeros(2)


def _line_face_coordinates(first, second, west, south, east, north):
    xmin, ymin = np.minimum(first, second)
    xmax, ymax = np.maximum(first, second)
    delta_x, delta_y = second-first
    slope = delta_y/delta_x if delta_x else np.float32(delta_y/1e-6)
    yw, ye = (slope*(x-first[0])+first[1] if xmin <= x <= xmax else np.float32(0)
              for x in (west, east))
    xs, xn = ((y-first[1])/slope+first[0] if ymin <= y <= ymax and slope else np.float32(0)
              for y in (south, north))
    return yw, ye, xs, xn


def _endpoint_crossings(first, second, bounds, *, incoming):
    """Native vertex pass keeps the last attempted crossing on each axis."""
    west, south, east, north = bounds
    dx, dy = second-first
    slope = dy/dx if dx else np.float32(float(dy)/1e-6)
    crossings = [(0., 0.), (0., 0.)]
    for axis, (low, high) in enumerate(((west, east), (south, north))):
        minimum, maximum = sorted((first[axis], second[axis]))
        tangent_low, tangent_high = (south, north) if axis == 0 else (west, east)
        for boundary in (low, high):
            if not (minimum <= boundary <= maximum and boundary != first[axis]
                    and (not incoming or boundary != second[axis])):
                continue
            tangent = (slope*(boundary-first[0])+first[1] if axis == 0
                       else (boundary-first[1])/slope+first[0])
            if tangent < tangent_low or tangent > tangent_high:
                tangent = np.float32(0.)
            crossings[axis] = (boundary, tangent) if axis == 0 else (tangent, boundary)
    return crossings


def _native_corner_endpoint(first, second, following, bounds):
    """Check native vertex-pass sentinels and same-face corner rejections."""
    west, south, east, north = bounds
    hits = (_endpoint_crossings(first, second, bounds, incoming=True)
            + _endpoint_crossings(second, following, bounds, incoming=False))
    (_, west_east_in), (south_north_in, _), (_, west_east_out), (south_north_out, _) = hits
    if south_north_in == south_north_out == 0:
        if max(west_east_in, west_east_out) < south or min(west_east_in, west_east_out) > north:
            return False
    if west_east_in == west_east_out == 0:
        if max(south_north_in, south_north_out) < west or min(south_north_in, south_north_out) > east:
            return False
    adjacent = ((east if second[0] == west else west, second[1]),
                (second[0], north if second[1] == south else south))
    return any(hit != (0., 0.) for hit in hits) and not any(hit in adjacent for hit in hits)


def _edge_intersections(vertices, origin, dx, *, ring_joins=(), ring_endpoints=False):
    """Apply the native face-selection cases for segments through a cell."""
    west, south = origin.astype(np.float32)
    east, north = np.asarray(origin+dx, dtype=np.float32)
    samples = []
    reference = None
    missed_corner = False
    bounds = west, south, east, north
    successors = np.roll(vertices, -1, axis=0)
    for edge, (first, second) in enumerate(zip(vertices, successors)):
        if ring_joins:
            following = successors[(edge+1) % len(vertices)]
            vertical_contact = (ring_endpoints and second[0] == west and second[1] in (south, north)
                                and (first[0] == west or following[0] == west))
            if edge not in ring_joins and (edge+1) % len(vertices) not in ring_joins and not vertical_contact:
                continue
        xmin, ymin = np.minimum(first, second)
        xmax, ymax = np.maximum(first, second)
        if east < xmin or west > xmax or north < ymin or south > ymax:
            continue
        yw, ye, xs, xn = _line_face_coordinates(first, second, west, south, east, north)
        w, e, s, n = (west, yw), (east, ye), (xs, south), (xn, north)
        missed_corner |= ((yw in (south, north) or ye in (south, north))
                          and (west <= xs <= east or west <= xn <= east))
        start = len(samples)
        if ring_joins and second[0] in (west, east) and second[1] in (south, north):
            if (west <= first[0] < east and south <= first[1] < north
                    and not (west <= second[0] < east and south <= second[1] < north)):
                continue
            if _native_corner_endpoint(first, second, following, bounds):
                samples.append(second)
                if reference is None:
                    reference = second
            continue
        if ring_joins and edge not in ring_joins:
            continue
        if south < yw < north:
            samples.extend((w, e if south <= ye <= north else s if ye < south else n))
        if south < ye < north and (yw <= south or yw >= north):
            samples.extend((e, s if yw <= south else n))
        if ((yw, ye, xs, xn) == (south, north, west, east)
                or (yw, ye, xs, xn) == (north, south, east, west)):
            # A corner-to-corner cut records a volume reference point but
            # contributes no intersections to any of the cell faces.
            if reference is None:
                reference = w
        if (yw > north or yw < south) and (ye > north or ye < south):
            if west < xs < east and west < xn < east:
                samples.extend((s, n))
        if reference is None and len(samples) > start:
            reference = samples[start]
    if reference is None:
        if missed_corner:
            return 0., np.zeros(4), None, False
        return None
    offset = np.array([west, south])
    world_samples = np.asarray(samples).reshape(-1, 2)
    intersections = (world_samples-offset).astype(float)
    # Native face points keep their face membership even when the difference
    # between two rounded world boundaries is not the nominal cell width.
    intersections[world_samples[:, 0] == east, 0] = dx
    intersections[world_samples[:, 1] == north, 1] = dx
    area, lengths, first = _point_footprint(vertices, origin, dx, intersections,
                                            reference=np.asarray(reference)-offset,
                                            ring_breaks=bool(ring_joins))
    return area, lengths, first, not samples


def _point_footprint(vertices, origin, dx, intersections, *, reference=None, face_masks=None, ring_breaks=False):
    corners = np.array([[0., 0.], [0., dx], [dx, 0.], [dx, dx]])+origin
    if ring_breaks:
        closed = np.vstack((vertices, vertices[0]))
        inside = ring_contains(closed, *corners.astype(np.float32).T, native_ring_breaks=True)
    else:
        inside = np.zeros(len(corners), dtype=bool)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            x, y = corners.T
            if first[1] != second[1]:
                crossing = (first[1] > y) != (second[1] > y)
                intercept = first[0]+(y-first[1])/(second[1]-first[1])*(second[0]-first[0])
                inside ^= crossing & (x < intercept)
    solid_corners = corners[inside]-origin
    points = np.vstack((intersections, solid_corners))
    if reference is None:
        reference = intersections[0]
    if not len(points):
        return 0., np.zeros(4), reference
    shifted = points-points.mean(axis=0)
    polygon = points[np.argsort(np.arctan2(shifted[:, 1], shifted[:, 0]))]
    other = np.roll(polygon, -1, axis=0)
    area = .5*abs(np.sum(polygon[:, 0]*other[:, 1]-polygon[:, 1]*other[:, 0]))
    lengths = np.zeros(4)
    for index, (axis, boundary) in enumerate(((0, dx), (0, 0.), (1, dx), (1, 0.))):
        if face_masks is None:
            values = points[points[:, axis] == boundary, 1-axis]
        else:
            corners_on_face = solid_corners[solid_corners[:, axis] == boundary, 1-axis]
            values = np.r_[corners_on_face, intersections[(face_masks & (1 << index)) != 0, 1-axis]]
        if len(values):
            lengths[index] = np.ptp(values)
    return area, lengths, reference


def _corner_vertex_footprint(vertices, origin, dx):
    """Keep native face ownership for oblique segments meeting at a cell boundary."""
    lower, upper = origin.astype(np.float32), (origin+dx).astype(np.float32)
    indices = np.flatnonzero(np.all((vertices >= lower) & (vertices <= upper), axis=1))
    if len(indices) != 1:
        return None
    index = indices[0]
    vertex = vertices[index]
    incoming, outgoing = vertices[index-1], vertices[(index+1) % len(vertices)]
    if np.any(incoming == vertex) or np.any(outgoing == vertex):
        return None
    owns_vertex = np.all(vertex < upper)
    owned_west_vertex = owns_vertex and vertex[0] == lower[0]
    horizontal_face_vertex = lower[0] < vertex[0] < upper[0] and vertex[1] in (lower[1], upper[1])
    southeast_vertex = (vertex[0] == upper[0] and vertex[1] == lower[1]
                        and incoming[0] > upper[0] and outgoing[0] < upper[0])
    vertex_point = tuple(vertex)
    crossings = []
    y_crossings = []
    for first, second in ((incoming, vertex), (vertex, outgoing)):
        dx_line, dy_line = second-first
        slope = dy_line/dx_line if dx_line else np.float32(dy_line/1e-6)
        xmin, xmax = sorted((first[0], second[0]))
        for side, x in ((1, lower[0]), (0, upper[0])):
            if not xmin <= x <= xmax:
                continue
            y = slope*(x-first[0])+first[1]
            if lower[1] <= y <= upper[1] and (x, y) != vertex_point:
                crossings.append(((x, y), side, y in (lower[1], upper[1])))
        if slope:
            ymin, ymax = sorted((first[1], second[1]))
            for side, y in ((3, lower[1]), (2, upper[1])):
                if not ymin <= y <= ymax:
                    continue
                x = (y-first[1])/slope+first[0]
                if lower[0] <= x <= upper[0] and (x, y) != vertex_point:
                    y_crossings.append(((x, y), side, x in (lower[0], upper[0])))
    # Without an x crossing, a single y crossing cannot pass the native
    # two-intersection test. The vertex itself still belongs to its faces.
    vertex_only = ((owned_west_vertex or horizontal_face_vertex or southeast_vertex)
                   and not crossings and len(y_crossings) == 1)
    mixed_crossings = (horizontal_face_vertex and len(crossings) == len(y_crossings) == 1
                       and not (crossings[0][2] or y_crossings[0][2]))
    upper_vertex_corner = (vertex[1] == upper[1] and not crossings and len(y_crossings) == 1
                           and y_crossings[0][2])
    northwest_vertex = (vertex[0] == lower[0] and vertex[1] == upper[1]
                        and not crossings and len(y_crossings) == 1 and not y_crossings[0][2])
    if not (vertex_only or mixed_crossings or upper_vertex_corner or northwest_vertex
            or any(corner for _, _, corner in crossings)):
        return None
    successors = np.roll(vertices, -1, axis=0)
    overlaps_cell = np.all((np.minimum(vertices, successors) <= upper)
                           & (np.maximum(vertices, successors) >= lower), axis=1)
    overlaps_cell[[index-1, index]] = False
    if overlaps_cell.any():
        return None
    if upper_vertex_corner:
        # Rounding can leave a corner on the y face only. The footprint has
        # the same areas, but native volume and distance start at the vertex.
        area, lengths = _cell_footprint(vertices, origin, dx)
        return area, lengths, vertex-lower, False
    vertex_faces = sum(1 << side for side, coordinate in enumerate((upper[0], lower[0], upper[1], lower[1]))
                       if vertex[side//2] == coordinate)
    points, faces = [vertex], [vertex_faces]
    # Two intersections on the same y face are handled before the x cases.
    if len(y_crossings) == 2 and y_crossings[0][1] == y_crossings[1][1]:
        crossings = y_crossings
    elif mixed_crossings:
        crossings.extend(y_crossings)
    # The x-face cases require a strict y crossing on at least one side.
    # A lone corner falls through; only the interior vertex is retained.
    if len(crossings) == 2:
        (first, first_face, first_corner), (second, second_face, second_corner) = crossings
        if (first_face == second_face and first != second) or not (first_corner and second_corner):
            points.extend((first, second))
            faces.extend((1 << first_face, 1 << second_face))
    reference = None
    if not owns_vertex and index == 0:
        # The outgoing diagonal can be visited in this neighboring cell before
        # the later edge records the vertex itself.
        edge = _edge_intersections(vertices, origin, dx)
        if edge is not None and edge[-1]:
            reference = edge[2]
    footprint = _point_footprint(vertices, origin, dx, np.asarray(points)-lower,
                                 reference=reference, face_masks=np.asarray(faces))
    # A vertex-only boundary entry keeps the upper corner flags. A diagonal
    # visited first in the cell west of an east-face vertex clears them.
    retain_roof_corners = len(points) == 1 and bool(vertex_faces) and (reference is None or vertex[0] != upper[0])
    return (*footprint, retain_roof_corners)


def _vertex_sentinel_footprint(vertices, origin, dx):
    """Retain the global zero used by the native lower-face intersection case."""
    lower, upper = origin.astype(np.float32), (origin+dx).astype(np.float32)
    inside = np.all((vertices >= lower) & (vertices <= upper), axis=1)
    skipped = set()
    count = len(vertices)
    for edge in range(count):
        index = (edge+1) % count
        if edge in skipped or not inside[index]:
            continue
        vertex = vertices[index]
        last = index
        while (last+1) % count != index and inside[(last+1) % count]:
            skipped.add(last)
            last = (last+1) % count
        previous, endpoint, next_vertex = vertices[edge], vertices[last], vertices[(last+1) % count]
        x_crossings = []
        y_crossings = []
        y_face_hits = []
        outgoing_south = None
        for outgoing, (first, second, inner) in enumerate(((previous, vertex, vertex),
                                                          (endpoint, next_vertex, endpoint))):
            delta_x, delta_y = second-first
            slope = delta_y/delta_x if delta_x else np.float32(delta_y/1e-6)
            xmin, xmax = sorted((first[0], second[0]))
            for x in (lower[0], upper[0]):
                if xmin <= x <= xmax and x != inner[0]:
                    y = slope*(x-first[0])+first[1]
                    if lower[1] <= y <= upper[1]:
                        x_crossings.append((x, y))
            ymin, ymax = sorted((first[1], second[1]))
            hit = None
            if slope:
                for y in (lower[1], upper[1]):
                    if ymin <= y <= ymax and y != inner[1]:
                        x = (y-first[1])/slope+first[0]
                        valid = lower[0] <= x <= upper[0]
                        hit = (x if valid else np.float32(0), y, valid)
                        if valid:
                            y_crossings.append(x)
                        if outgoing and y == lower[1] and endpoint[1] == upper[1]:
                            outgoing_south = x
            if hit is not None:
                y_face_hits.append(hit)
        if lower[0] == 0 and len(y_face_hits) == 2 and len(x_crossings) <= 1:
            (x1, y1, valid1), (x2, y2, valid2) = y_face_hits
            if y1 == y2 and valid1 != valid2 and x1 != x2:
                # At x=0 the rejected crossing's sentinel is itself in
                # range, so QES accepts the pair on the same y face.
                points = np.asarray((vertex, (x1, y1), (x2, y2)))-lower
                return _point_footprint(vertices, origin, dx, points, reference=vertex-lower)
        if len(x_crossings) != 1 or y_crossings:
            continue
        x, y = x_crossings[0]
        if not lower[1] < y < upper[1] or (lower[1] == 0. and x == lower[0]):
            continue
        # A single interior x-face intersection selects the lower y face;
        # the missing intersection on that face is still the global zero.
        polygon = _clip_footprint(vertices, origin, dx, native_edges=True).reshape(-1, 2)
        extra = [(-lower[0], 0.)]
        if outgoing_south is not None:
            # The next edge is revisited below an upper-face vertex, with
            # the unfiltered crossing from the edge-only branch.
            extra.append((outgoing_south-lower[0], 0.))
        # The outgoing segment is visited before the closing segment when
        # the first vertex lies at the southeast corner of this cell.
        outgoing_first = (index == 0 and (vertex == (upper[0], lower[1])).all() and x == lower[0]
                          and previous[0] > upper[0] and next_vertex[0] < lower[0])
        reference = np.array([x, y]) if outgoing_first else vertex
        return _point_footprint(vertices, origin, dx, np.vstack((polygon, extra)), reference=reference-lower)
    return None


def _boundary_exit_footprint(vertices, origin, dx):
    """Replay the unclipped outgoing edge from an adjoining cell's vertex."""
    west, south = origin.astype(np.float32)
    east, north = (origin+dx).astype(np.float32)
    in_cell = (vertices[:, 0] >= west) & (vertices[:, 0] <= east) & (vertices[:, 1] >= south) & (vertices[:, 1] <= north)
    candidates = np.flatnonzero(in_cell)
    if len(candidates) != 1:
        return None
    index = candidates[0]
    vertex = vertices[index]
    following = vertices[(index+1) % len(vertices)]
    east_vertex = vertex[0] == east and south < vertex[1] < north
    east_exit = east_vertex and following[0] > east
    north_vertex = vertex[1] == north and west < vertex[0] < east
    if not (east_vertex or north_vertex):
        return None
    previous = vertices[index-1]
    if east_exit and previous[0] > east and np.all(vertices[:, 0] >= east):
        if min(previous[1], following[1]) <= south or max(previous[1], following[1]) >= north:
            # The vertex path can mark a touching cell without creating an
            # open-face change or a nonzero wall normal.
            return 0., np.zeros(4), vertex-[west, south]
        return None
    yw, ye, xs, _ = _line_face_coordinates(vertex, following, west, south, east, north)
    # A westward segment can reach the east-face vertex without reaching the
    # west or south face. The edge-only pass then uses its global-zero sentinel.
    south_sentinel = east_vertex and west > 0 and west < following[0] < east and following[1] > north and xs == 0
    unclipped_exit = (east_exit or north_vertex) and xs > east
    if not (south < ye < north and yw <= south and (unclipped_exit or south_sentinel)):
        return None
    # The first endpoint belongs to the next cell. The native edge-only pass
    # still appends its second intersection to this cell without clamping it.
    lower = np.array([west, south])
    reference = vertex-lower
    points = np.array([reference]) if east_vertex else _clip_footprint(vertices, origin, dx, native_edges=True).reshape(-1, 2)
    return _point_footprint(vertices, origin, dx, np.vstack((points, [xs-west, 0.])), reference=reference)


def _aligned_endpoint(vertices, origin, dx, *, native_hole=False):
    """Handle native coincident-face endpoint cases."""
    west, south = origin.astype(np.float32)
    east, north = np.asarray(origin+dx, dtype=np.float32)
    successors = np.roll(vertices, -1, axis=0)
    for index, (first, second) in enumerate(zip(vertices, successors)):
        if first[0] != second[0] or second[0] not in (west, east) or not south <= second[1] <= north:
            continue
        slope = np.float32((second[1]-first[1])/1e-6)
        if slope == 0:
            continue
        if native_hole and second[1] in (south, north):
            following = successors[(index+1) % len(vertices)]
            if (following[1] == second[1]
                    and (second[0] == west and following[0] >= east or second[0] == east and following[0] <= west)):
                # The native vertex pass rejects the other corner of the
                # same face when the outgoing hole edge follows that face.
                continue
        lo, hi = sorted((first[1], second[1]))
        for y in (south, north):
            if not lo < y < hi:
                continue
            crossing = (y-first[1])/slope+first[0]
            if west < crossing < east:
                return 0., np.zeros(4), second-origin
            if second[0] == east and south < second[1] < north and y == south and crossing > east:
                # The rejected x intersection is stored as global zero.
                # The east-face case still includes that south-face point.
                points = np.array([second-origin, (-west, 0.)])
                return _point_footprint(vertices, origin, dx, points)
    return None


def apply_cutcell_building(workspace, building, number, *, native_math=False, native_ring_joins=False):
    """Apply the polygon prism to its already flagged building columns."""
    parts = building.physical_parts()
    native_outline = building.native_closure_vertices if native_ring_joins else None
    if len(parts) > 1 and native_outline is None:
        for part in parts:
            apply_cutcell_building(workspace, part, number, native_math=native_math)
        return
    native_hole = native_outline is not None and any(part.holes for part in parts)
    physical_polygons = tuple((np.asarray(part.vertices)[:-1], tuple(np.asarray(hole)[:-1] for hole in part.holes))
                              for part in parts) if native_hole else ()
    a, c = workspace.arrays, workspace.coordinates
    flags = a['icellflag']
    owners = a['ibuilding_flag']
    coefficients = tuple(a[key] for key in 'efghmn')
    dx = float(np.float32(c['x'][1]-c['x'][0]))
    exterior = np.asarray(native_outline or building.vertices)
    vertices = exterior[:-1]
    hole_rings = tuple(np.asarray(ring) for ring in building.holes)
    holes = tuple(ring[:-1] for ring in hole_rings)
    native_vertices = vertices.astype(np.float32)
    ring_joins = set()
    if native_outline is not None:
        # Edge traversal follows the whole input list. Corner inclusion in
        # CutBuilding.cpp restarts the ray scan at each closed ring instead.
        start = 0
        for index, vertex in enumerate(native_vertices):
            if index-start >= 3 and np.array_equal(vertex, native_vertices[start]):
                ring_joins.add(index)
                start = index+1
    (imin, jmin), (imax, jmax) = np.floor(np.stack((vertices.min(axis=0), vertices.max(axis=0)))/dx).astype(int)
    dz_array = a['dz_array']
    z, z_face = c['z'], c['z_face']
    base = building.base_height
    if native_math:
        dz_array = dz_array.astype(np.float32, copy=False)
        z, z_face = native_vertical_coordinates(dz_array)
        base = np.float32(base)
    top = np.float32(building.height)+np.float32(base)
    start = max(1, int(np.searchsorted(z_face, base)-1))
    end = int(np.searchsorted(z_face, top)-1)
    if end < start or end >= flags.shape[0]:
        return
    if native_math and (not holes or native_outline is not None):
        from .cutcell_building_walk import apply_walk
        apply_walk(workspace, exterior, number, dx, start, end, top, z, z_face, dz_array,
                   ring_breaks=native_outline is not None)
        return
    for j in range(max(0, jmin-1), min(flags.shape[1]-1, jmax)+1):
        for i in range(max(0, imin-1), min(flags.shape[2]-1, imax)+1):
            origin = np.array([i*dx, j*dx])
            if holes and not native_hole:
                area, lengths = _perforated_footprint(vertices, holes, origin, dx)
                if area <= 1e-12*dx*dx:
                    continue
                center = np.array([c['x'][i], c['y'][j]])
                side_distance = min(_distance_to_ring(ring, center) for ring in (vertices, *holes))
                inside_footprint = ring_contains(exterior, *center) and not any(
                    ring_contains(ring, *center) for ring in hole_rings)
                _apply_perforated_column(workspace, building, number, (j, i), range(start, end+1), dx,
                                         (area, lengths), side_distance, inside_footprint)
                continue
            if native_hole:
                footprints = [_perforated_footprint(polygon, openings, origin, dx) for polygon, openings in physical_polygons]
                area = sum(item[0] for item in footprints)
                lengths = np.sum([item[1] for item in footprints], axis=0)
            else:
                area, lengths = _cell_footprint(vertices, origin, dx)
            interior = area >= (1.-1e-10)*dx**2
            if (imin == i == imax and jmin == j == jmax
                    and np.all((native_vertices > origin.astype(np.float32))
                               & (native_vertices < (origin+dx).astype(np.float32)))):
                # The native edge loop skips polygons without a face crossing.
                # Its separate roof pass uses the whole cell over solid below.
                interior = True
                area, lengths = dx**2, np.full(4, dx)
            first = None if interior or area <= 1e-12*dx**2 else _first_intersection(vertices, origin, dx)
            reference_only = False
            retain_roof_corners = False
            has_vertex = np.any(np.all((vertices >= origin) & (vertices <= origin+dx), axis=1))
            join = (_edge_intersections(native_vertices, origin, dx, ring_joins=ring_joins, ring_endpoints=native_hole)
                    if ring_joins else None)
            if join is not None and join[2] is not None:
                area, lengths, first, reference_only = join
                interior = False
            elif not has_vertex:
                footprint = _edge_intersections(native_vertices, origin, dx)
                if footprint is not None and (not interior or footprint[2] is not None):
                    if footprint[2] is None and area > 1e-12*dx**2:
                        # The side traversal skipped the intersections.
                        # The independent roof pass still visits solid columns.
                        interior = True
                        area, lengths = dx**2, np.full(4, dx)
                    else:
                        area, lengths, first, reference_only = footprint
                        interior = False
            elif not interior:
                footprint = _vertex_sentinel_footprint(native_vertices, origin, dx)
                if footprint is None:
                    footprint = _boundary_exit_footprint(native_vertices, origin, dx)
                if footprint is None:
                    corner = _corner_vertex_footprint(native_vertices, origin, dx)
                    if corner is not None:
                        *footprint, retain_roof_corners = corner
                if footprint is not None:
                    area, lengths, first = footprint
            if area <= 1e-12*dx**2 and has_vertex:
                footprint = _aligned_endpoint(native_vertices, origin, dx, native_hole=native_hole)
                if footprint is not None:
                    area, lengths, first = footprint
            if area <= 1e-12*dx**2 and first is None:
                continue
            roof_area = 0.
            if retain_roof_corners:
                # The partial-roof branch omits the intersection vertex above,
                # but the separate corner-parity pass can still add roof points.
                roof_area, _, _ = _point_footprint(native_vertices, origin, dx, np.empty((0, 2)), reference=first)
            for k in range(start, end+1):
                idx = (k, j, i)
                dz = dz_array[k]
                height = min(dz, top-z_face[k])
                if height <= 0:
                    continue
                if interior:
                    if (k != end or height >= dz or flags[k-1, j, i] != 0
                            or owners[k-1, j, i] != number):
                        continue
                    point = np.array([dx, dx, height])
                else:
                    flag = flags[idx]
                    if flag == 2 or (flag not in (1, 7) and owners[idx] != number):
                        continue
                    point = np.array([first[0], first[1], 0.])
                # A diagonal-only reference never sets the solid height for a
                # full layer. At a partial roof it sets the height explicitly.
                side_height = 0. if reference_only and height >= dz else height
                solid = np.r_[lengths*side_height, area if height >= dz else roof_area, area]
                face_area = np.array([dx*dz]*4+[dx*dx]*2)
                openings = np.clip(1.-solid/face_area, 0., 1.)
                openings[openings <= .05] = 0.
                solid = (1.-openings)*face_area
                normal_area = solid[[1, 3, 5]]-solid[[0, 2, 4]]
                norm = np.linalg.norm(normal_area)
                normal = normal_area/norm if norm else np.zeros(3)
                # An isolated diagonal leaves the native upper-face points at
                # z=0 for full layers, even when parity gives a nonzero area.
                upper_height = 0. if reference_only else height
                volume = (dx*(solid[0]+solid[2])+upper_height*solid[4]+point@normal_area)/(3*dx*dx*dz)
                previous_fraction = a['building_volume_frac'][idx]
                fraction = max(0., float(previous_fraction-volume))
                a['building_volume_frac'][idx] = fraction
                for key, component in zip(('ni', 'nj', 'nk'), normal):
                    a[key][idx] = component
                distance = np.linalg.norm((point-np.array([c['x'][i], c['y'][j], z[k]]))*normal)
                a['wall_distance'][idx] = distance if a['center_id'][idx] else -distance
                for field, blocked in zip(coefficients, 1.-openings):
                    field[idx] = max(0., field[idx]-blocked)
                closed = fraction <= .1 or all(field[idx] == 0 for field in coefficients)
                flags[idx] = 0 if closed else 7
                if closed:
                    for field in coefficients:
                        field[idx] = 1.


def _apply_perforated_column(workspace, building, number, column, layers, dx, footprint, side_distance, inside_footprint):
    a, c = workspace.arrays, workspace.coordinates
    flags, owners = a['icellflag'], a['ibuilding_flag']
    coefficients = tuple(a[key] for key in 'efghmn')
    area, lengths = footprint
    base = float(np.float32(building.base_height))
    top = float(np.float32(building.height)+np.float32(building.base_height))
    for k in layers:
        idx = (k, *column)
        dz = a['dz_array'][k]
        floor = c['z_face'][k]
        ceiling = floor+dz
        overlap = min(top, ceiling)-max(base, floor)
        if overlap <= 0 or flags[idx] == 2 or (flags[idx] not in (1, 7) and owners[idx] != number):
            continue
        solid = np.r_[lengths*overlap, area if top >= ceiling else 0., area if base <= floor else 0.]
        face_area = np.array([dx*dz]*4+[dx*dx]*2)
        openings = np.clip(1.-solid/face_area, 0., 1.)
        openings[openings <= .05] = 0.
        normal_area = solid[[1, 3, 5]]-solid[[0, 2, 4]]
        norm = np.linalg.norm(normal_area)
        normal = normal_area/norm if norm else np.zeros(3)
        for name, component in zip(('ni', 'nj', 'nk'), normal):
            a[name][idx] = component
        center = c['z'][k]
        inside = inside_footprint and base <= center < top
        distance = min(side_distance, abs(center-top), abs(center-base)) if inside else np.hypot(
            0. if inside_footprint else side_distance, max(base-center, center-top, 0.))
        a['wall_distance'][idx] = -distance if inside else distance
        a['building_volume_frac'][idx] = max(0., a['building_volume_frac'][idx]-area*overlap/(dx*dx*dz))
        for field, blocked in zip(coefficients, 1.-openings):
            field[idx] = max(0., field[idx]-blocked)
        closed = a['building_volume_frac'][idx] <= .1 or all(field[idx] == 0 for field in coefficients)
        flags[idx] = 0 if closed else 7
        if closed:
            for field in coefficients:
                field[idx] = 1.
