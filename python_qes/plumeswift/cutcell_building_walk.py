"""Legacy QES cut-cell edge traversal, adapted from CutBuilding.cpp (GPL-3.0-only)."""
from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np

from .geometry import ring_contains
from .numerics.linux_float_angles import atan2f


EAST, WEST, NORTH, SOUTH = 1, 2, 4, 8


@dataclass
class CutColumn:
    reference: tuple
    points: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    has_height: bool = False
    clear_roof: bool = False

    def add(self, samples, *, clear_roof=False):
        for point, faces in samples:
            self.points.append(point)
            self.masks.append(faces)
        self.has_height = True
        self.clear_roof |= clear_roof


def _attempt_crossings(start, end, slope, lower, upper, *, exclude_end, slots=None):
    """An endpoint visit retains the last attempted crossing per axis."""
    x_point, y_point = slots if slots is not None else ((0., 0.), (0., 0.))
    for x in (lower[0], upper[0]):
        if min(start[0], end[0]) <= x <= max(start[0], end[0]) and x != start[0] and (not exclude_end or x != end[0]):
            y = slope*(x-start[0])+start[1]
            x_point = (x, y if lower[1] <= y <= upper[1] else 0.)
    for y in (lower[1], upper[1]):
        if min(start[1], end[1]) <= y <= max(start[1], end[1]) and y != start[1] and (not exclude_end or y != end[1]):
            x = (y-start[1])/slope+start[0]
            y_point = (x if lower[0] <= x <= upper[0] else 0., y)
    return x_point, y_point


def _face_mask(point, lower, upper):
    x, y = point
    return ((EAST if x == upper[0] else 0) | (WEST if x == lower[0] else 0)
            | (NORTH if y == upper[1] else 0) | (SOUTH if y == lower[1] else 0))


def _endpoint_slots(incoming, outgoing, lower, upper, dx):
    """Map the two stored crossing pairs to the four general face slots."""
    (ix, iy), (sx, sy) = incoming
    (ox, oy), (tx, ty) = outgoing
    west, south = lower
    east, north = upper
    low, high = south-dx, north+dx
    yw = ye = xs = xn = 0.
    if iy:
        if ix == west:
            yw = iy
            if oy and ox == east:
                ye = oy
            if tx:
                if ty == south:
                    xs, ye = tx, low
                if ty == north:
                    xn, ye = tx, high
        if ix == east:
            ye = iy
            if oy and ox == west:
                yw = oy
            if tx:
                if ty == south:
                    xs, yw = tx, low
                if ty == north:
                    xn, yw = tx, high
        if sx and ix in (west, east):
            if sy == south:
                xs = sx
            if sy == north:
                xn = sx
    elif oy:
        if ox == west:
            yw = oy
            if sx:
                if sy == south:
                    xs, ye = sx, low
                if sy == north:
                    xn, ye = sx, high
        if ox == east:
            ye = oy
            if sx:
                if sy == south:
                    xs, yw = sx, low
                if sy == north:
                    xn, yw = sx, high
    else:
        if sy == south:
            xs, xn, yw, ye = sx, tx, low, high
        if sy == north:
            xs, xn, yw, ye = tx, sx, high, low
    return yw, ye, xs, xn


def _same_face_samples(incoming, outgoing, lower, upper):
    samples = []
    west, south = lower
    east, north = upper
    for axis, faces, low, high in ((0, (WEST, EAST), south, north), (1, (SOUTH, NORTH), west, east)):
        first, second = incoming[axis], outgoing[axis]
        tangential = 1-axis
        if first[axis] != second[axis] or first[tangential] == second[tangential]:
            continue
        if not (low <= first[tangential] <= high and low <= second[tangential] <= high):
            continue
        for bound, face in zip((lower[axis], upper[axis]), faces):
            if first[axis] == bound:
                samples.extend(((first, face), (second, face)))
    return samples


def _general_samples(slots, lower, upper):
    """Return reference and face samples in native case order."""
    yw, ye, xs, xn = slots
    west, south = lower
    east, north = upper
    w, e = ((west, yw), WEST), ((east, ye), EAST)
    s, n = ((xs, south), SOUTH), ((xn, north), NORTH)
    if south < yw < north:
        if south <= ye <= north:
            return w[0], (w, e)
        return w[0], (w, s if ye < south else n)
    if south < ye < north and (yw <= south or yw >= north):
        return e[0], (e, s if yw <= south else n)
    if ((yw, ye, xs, xn) == (south, north, west, east)
            or (yw, ye, xs, xn) == (north, south, east, west)):
        return w[0], ()
    if (yw > north or yw < south) and (ye > north or ye < south) and west < xs < east and west < xn < east:
        return s[0], (s, n)
    return None


def _through_slots(start, slope, minimum, maximum, lower, upper):
    yw, ye = (slope*(x-start[0])+start[1] if minimum[0] <= x <= maximum[0] else 0.
              for x in (lower[0], upper[0]))
    xs, xn = ((y-start[1])/slope+start[0] if minimum[1] <= y <= maximum[1] else 0.
              for y in (lower[1], upper[1]))
    return yw, ye, xs, xn


def _reject_endpoint(point, incoming, outgoing, lower, upper):
    west, south = lower
    east, north = upper
    ix, it = incoming
    ox, ot = outgoing
    if it[0] == ot[0] == 0 and (max(ix[1], ox[1]) < south or min(ix[1], ox[1]) > north):
        return True
    if ix[1] == ox[1] == 0 and (max(it[0], ot[0]) < west or min(it[0], ot[0]) > east):
        return True
    hits = (*incoming, *outgoing)
    if point[0] in (west, east) and point[1] in (south, north):
        # The east corners share the west-corner checks with the y side swapped.
        check_y = point[1] if point[0] == west else (south if point[1] == north else north)
        other_y = north if check_y == south else south
        if (east, check_y) in hits or (west, other_y) in hits:
            return True
    return not any(x or y for x, y in hits)


def walk_columns(vertices, dx, nx, ny):
    """Collect the native side-walk contributions before adding solid corners."""
    # Domain::getDomainSize reports dx for both horizontal widths.
    vertices = np.asarray(vertices, dtype=np.float32)
    dx = np.float32(dx)
    count = len(vertices)-1
    starts, ends = vertices[:-1], vertices[1:]
    minimum, maximum = np.minimum(starts, ends), np.maximum(starts, ends)
    first_cell = (starts/dx).astype(int)
    last_cell = (ends/dx).astype(int)
    begin = np.maximum((minimum/dx-np.float32(1)).astype(int), 0)
    finish = np.minimum(np.floor(maximum/dx).astype(int), (nx-1, ny-1))
    delta = ends-starts
    slopes = np.empty(count, dtype=np.float32)
    vertical = delta[:, 0] == 0
    slopes[vertical] = delta[vertical, 1].astype(float)/1e-6
    slopes[~vertical] = delta[~vertical, 1]/delta[~vertical, 0]
    boundaries_x = np.arange(nx+1, dtype=np.float32)*dx
    boundaries_y = np.arange(ny+1, dtype=np.float32)*dx
    skipped = []
    cursor = 0
    columns = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for edge, (first, second) in enumerate(zip(starts, ends)):
            if cursor < len(skipped) and edge == skipped[cursor]:
                cursor += 1
                continue
            for i in range(begin[edge, 0], finish[edge, 0]+1):
                west, east = boundaries_x[i:i+2]
                for j in range(begin[edge, 1], finish[edge, 1]+1):
                    if (i, j) == tuple(first_cell[edge]) and (i, j) != tuple(last_cell[edge]):
                        continue
                    south, north = boundaries_y[j:j+2]
                    lower, upper = (west, south), (east, north)
                    column = j, i
                    if west <= second[0] <= east and south <= second[1] <= north:
                        incoming = _attempt_crossings(first, second, slopes[edge], lower, upper, exclude_end=True)
                        next_edge = (edge+1) % count
                        outgoing = _attempt_crossings(starts[next_edge], ends[next_edge], slopes[next_edge],
                                                      lower, upper, exclude_end=False)
                        points = [tuple(second)]
                        while (len(points) < count-1 and west <= ends[next_edge, 0] <= east
                               and south <= ends[next_edge, 1] <= north):
                            points.append(tuple(ends[next_edge]))
                            skipped.append(next_edge)
                            next_edge = (next_edge+1) % count
                            outgoing = _attempt_crossings(starts[next_edge], ends[next_edge], slopes[next_edge],
                                                          lower, upper, exclude_end=False, slots=outgoing)
                        if _reject_endpoint(points[0], incoming, outgoing, lower, upper):
                            continue
                        cut = columns.setdefault(column, CutColumn(points[0]))
                        cut.add((point, _face_mask(point, lower, upper)) for point in points)
                        shared_face = _same_face_samples(incoming, outgoing, lower, upper)
                        if shared_face:
                            cut.add(shared_face)
                            continue
                        slots = _endpoint_slots(incoming, outgoing, lower, upper, dx)
                    else:
                        if east < minimum[edge, 0] or north < minimum[edge, 1]:
                            continue
                        slots = _through_slots(first, slopes[edge], minimum[edge], maximum[edge], lower, upper)
                        if not any(slots):
                            continue
                    samples = _general_samples(slots, lower, upper)
                    if samples is None:
                        continue
                    point, faces = samples
                    cut = columns.setdefault(column, CutColumn(point))
                    if faces:
                        cut.add(faces, clear_roof=True)
                    else:
                        cut.clear_roof = True
    return columns


def _face_coefficient(points, axes, dx, dz):
    if not len(points):
        return np.float32(1.)
    points = np.asarray(points, dtype=np.float32)
    center = np.add.accumulate(points, axis=0)[-1]/np.float32(len(points))
    offsets = (points-center)[:, axes]
    angles = np.fromiter((float(atan2f(y, x))*(180/pi) for x, y in offsets),
                         dtype=np.float32, count=len(points))
    # The reference merge takes the right sublist first on an equal angle.
    order = len(points)-1-np.argsort(angles[::-1], kind='stable')
    ordered = points[order]
    following = np.roll(ordered, -1, axis=0)
    x_sum, y_sum, _ = (ordered+following).astype(float).T
    _, dy, dz_step = (following-ordered).astype(float).T
    vertical_area, horizontal_area = float(dx*dz), float(dx*dx)
    terms = (.5*y_sum*dz_step/vertical_area + .5*x_sum*dz_step/vertical_area
             + .5*x_sum*dy/horizontal_area)
    blocked = np.float32(0.)
    for value in terms:
        blocked = np.float32(float(blocked)+value)
    opening = np.float32(1.)-blocked
    return np.float32(0.) if float(opening) <= .05 else min(np.float32(1.), opening)


def _column_faces(cut, corners, lower, dx, dz, height, *, roof):
    side_height = height if cut.has_height or roof else np.float32(0.)
    points = [(np.float32(x-lower[0]), np.float32(y-lower[1])) for x, y in cut.points]
    masks = cut.masks
    relative_corners = [(x, y) for x, y in corners]
    corner_masks = [_face_mask((x, y), (0., 0.), (dx, dx)) for x, y in relative_corners]
    floor = points+relative_corners
    ceiling = [] if roof and cut.clear_roof else relative_corners if roof else floor
    openings = []
    areas = []
    for face, tangent in ((EAST, 1), (WEST, 1), (NORTH, 0), (SOUTH, 0)):
        face_points = [(x, y, elevation) for (x, y), mask in zip(floor, masks+corner_masks, strict=True)
                       if mask & face for elevation in (0., side_height)]
        area = dx*dz
        opening = _face_coefficient(face_points, (tangent, 2), dx, dz)
        openings.append(opening)
        areas.append(np.float32((1.-float(opening))*float(area)))
    for points, elevation in ((ceiling, side_height), (floor, 0.)):
        area = dx*dx
        opening = _face_coefficient([(x, y, elevation) for x, y in points], (0, 1), dx, dz)
        openings.append(opening)
        areas.append(np.float32((1.-float(opening))*float(area)))
    return np.array(openings, dtype=np.float32), np.array(areas, dtype=np.float32), side_height


def apply_walk(workspace, vertices, number, dx, start, end, top, z, z_face, dz_array, *, ring_breaks=False):
    """Apply ordered cut points and the independent top-of-building pass."""
    arrays, coordinates = workspace.arrays, workspace.coordinates
    flags, owners = arrays['icellflag'], arrays['ibuilding_flag']
    fields = tuple(arrays[name] for name in 'efghmn')
    dx = np.float32(dx)
    ny, nx = flags.shape[1:]
    vertices = np.asarray(vertices, dtype=np.float32)
    columns = walk_columns(vertices, dx, nx, ny)
    roof_height = top-z_face[end]
    partial_roof = roof_height < dz_array[end]
    if partial_roof and end > 0:
        imin, jmin = (vertices.min(axis=0)/dx).astype(int)
        imax, jmax = (vertices.max(axis=0)/dx+np.float32(1.)).astype(int)
        lower_solid = (flags[end-1] == 0) & (owners[end-1] == number)
        for j, i in np.argwhere(lower_solid):
            if imin <= i <= imax and jmin <= j <= jmax:
                columns.setdefault((j, i), None)
    local_corners = np.array(((0., 0.), (dx, 0.), (0., dx), (dx, dx)), dtype=np.float32)
    interior_faces = None
    for (j, i), cut in columns.items():
        lower = np.array([i, j], dtype=np.float32)*dx
        upper = np.array([i+1, j+1], dtype=np.float32)*dx
        corners = np.array(((lower[0], lower[1]), (upper[0], lower[1]),
                            (lower[0], upper[1]), (upper[0], upper[1])), dtype=np.float32)
        solid = ring_contains(vertices, *corners.T, native_ring_breaks=ring_breaks)
        solid_corners = local_corners[solid]
        layer_faces = {}
        if cut is not None:
            reference = np.asarray(cut.reference, dtype=np.float32)-lower
            point = np.array((reference[0], reference[1], 0.), dtype=np.float32)
        for k in (range(start, end+1) if cut is not None else (end,)):
            index = k, j, i
            if cut is not None and (flags[index] == 2 or (flags[index] not in (1, 7) and owners[index] != number)):
                continue
            dz = dz_array[k]
            roof = k == end and partial_roof
            height = roof_height if roof else dz
            if height <= 0.:
                continue
            if cut is None:
                point = np.array((dx, dx, height), dtype=np.float32)
                if interior_faces is None:
                    full_roof = CutColumn((0., 0.), clear_roof=True)
                    interior_faces = _column_faces(full_roof, local_corners, (0., 0.), dx, dz, height, roof=True)
                openings, areas, side_height = interior_faces
            else:
                layer = (dz, height, roof)
                if layer not in layer_faces:
                    layer_faces[layer] = _column_faces(cut, solid_corners, lower, dx, dz, height, roof=roof)
                openings, areas, side_height = layer_faces[layer]
            for field, opening in zip(fields, openings, strict=True):
                previous = np.float32(field[index])
                field[index] = opening if previous == 1. else max(0., float(previous)-(1.-float(opening)))
            difference = areas[[1, 3, 5]]-areas[[0, 2, 4]]
            norm = np.float32(sqrt(sum(float(value)**2 for value in difference)))
            normal = difference/norm if norm else difference
            scale = np.float32(3.)*dx*dx*dz
            flux = point*normal*norm
            volume = ((dx*areas[0]/scale+dx*areas[2]/scale)+side_height*areas[4]/scale
                      + (flux[0]+flux[1]+flux[2])/scale)
            fraction = max(0., float(np.float32(arrays['building_volume_frac'][index])-volume))
            arrays['building_volume_frac'][index] = fraction
            for name, value in zip(('ni', 'nj', 'nk'), normal, strict=True):
                arrays[name][index] = value
            center = np.array((coordinates['x'][i], coordinates['y'][j], z[k]), dtype=np.float32)
            distance = np.float32(sqrt(sum(float(value)**2 for value in (point-center)*normal)))
            arrays['wall_distance'][index] = distance if arrays['center_id'][index] else -distance
            closed = fraction <= .1 or all(field[index] == 0. for field in fields)
            flags[index] = 0 if closed else 7
            if closed:
                for field in fields:
                    field[index] = 1.
