"""Post-process terrain/building cut-cell openings as in QES Wall.cpp (GPL-3)."""
import numpy as np

from .geometry import block_solid_neighbours, scale_wall_coefficients


def cutcell_wall_stencil(workspace, *, native_math=False):
    """Close solid interfaces, reconcile neighbour apertures and scale coefficients."""
    a = workspace.arrays
    for name in 'efghmn':
        a[name] = np.ascontiguousarray(a[name], dtype=float)
    define_cutcell_walls(workspace)
    scale_wall_coefficients(a, workspace.coordinates, native_dy_alias=True, native_math=native_math)


def define_cutcell_walls(workspace):
    """Apply the native wall pass in place, before any solver coefficient scaling."""
    a = workspace.arrays
    flags = a['icellflag']
    coefficients = tuple(a[name] for name in 'efghmn')
    nx, ny = flags.shape[2], flags.shape[1]
    inner_z = slice(1, None)
    for values in coefficients:
        cut = values[inner_z]
        np.clip(cut, 0., 1., out=cut)
        cut[cut < .05] = 0.

    def solidify(mask, flag):
        flags[mask] = flag
        for values in coefficients:
            values[mask] = 1.

    columns = np.ones(flags.shape, dtype=bool)
    columns[0] = False
    if 'building_volume_frac' in a:
        collapsed = columns & np.isin(flags, (0, 7)) & (a['building_volume_frac'] <= .1)
        solidify(collapsed, 0)
    closed = np.logical_and.reduce([values == 0 for values in coefficients])
    solidify(columns & closed & (flags == 7), 0)
    block_solid_neighbours(a)

    # The original visits k, then i, then j and updates in place. Vertical
    # pairs write into the neighbour. Horizontal pairs read its opposite face.
    is_cut = np.isin(flags.ravel(), (7, 8))
    offsets = (1, -1, nx, -nx, nx*ny, -nx*ny)
    touches_cut = np.logical_or.reduce([np.roll(is_cut, -offset) for offset in offsets]).reshape(flags.shape)
    eligible = touches_cut & np.isin(flags, (1, 7, 8))
    eligible[0] = eligible[-1] = False
    flat = tuple(values.ravel() for values in coefficients)
    for k, i, j in np.argwhere(eligible.transpose(0, 2, 1)):
        cell = i+j*nx+k*nx*ny
        for face, offset in enumerate(offsets):
            if (face == 1 and i == 0) or (face == 3 and j == 0):
                continue
            neighbour = cell+offset
            if is_cut[neighbour]:
                opposite = face ^ 1
                if face < 4:
                    flat[face][cell] = flat[opposite][neighbour]
                else:
                    flat[opposite][neighbour] = flat[face][cell]

    closed = np.logical_and.reduce([values == 0 for values in coefficients])
    solidify(columns & closed & np.isin(flags, (0, 7)), 0)
    solidify(columns & closed & np.isin(flags, (2, 8)), 2)
    block_solid_neighbours(a)
