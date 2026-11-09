"""Staggered QES domain, including the ground ghost cell (GPL-3.0-only).

Port of src/qes/Domain.cpp. Arrays use (z,y,x); sizes in inputs use (x,y,z).
"""
from dataclasses import dataclass
import numpy as np


def native_vertical_coordinates(widths):
    """Build vertical coordinates with the source Domain's float32 stores."""
    widths = np.asarray(widths, dtype=np.float32)
    faces = np.empty(len(widths)+1, dtype=np.float32)
    faces[0], faces[1] = -widths[0], 0
    np.add.accumulate(widths[1:], out=faces[2:])
    return np.float32(.5)*(faces[:-1]+faces[1:]), faces


@dataclass(frozen=True)
class HorizontalHalo:
    """Native local geometry translation and DEM border widths."""

    x: float = 0.
    y: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.x, self.y]).all() or min(self.x, self.y) < 0:
            raise ValueError('Horizontal halo must be finite and nonnegative')

    def cells(self, spacing):
        dx, dy = spacing[:2]
        return (int(np.float32(self.x)/np.float32(dx)),
                int(np.float32(self.y)/np.float32(dy)))

    def translate(self, coordinates):
        offset = np.asarray((self.x, self.y), dtype=np.float32)
        if not offset.any():
            return coordinates
        moved = np.asarray(coordinates, dtype=np.float32)+offset
        if moved.ndim == 1:
            return tuple(map(float, moved))
        return tuple(map(tuple, moved.astype(float)))


@dataclass(frozen=True)
class Domain:
    cells: tuple[int, int, int]
    spacing: tuple[float, float, float]
    vertical_widths: tuple[float, ...] | None = None

    def __post_init__(self):
        if len(self.cells) != 3 or any(int(n) != n or n < minimum
                                       for n, minimum in zip(self.cells, (2, 2, 1), strict=True)):
            raise ValueError('At least two horizontal cells per direction and one vertical cell are required')
        if len(self.spacing) != 3 or not np.isfinite(self.spacing).all() or min(self.spacing) <= 0:
            raise ValueError("Grid spacing must be finite and positive")
        widths = np.full(self.cells[2], self.spacing[2]) if self.vertical_widths is None else np.asarray(self.vertical_widths, dtype=float)
        if widths.shape != (self.cells[2],) or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("Custom dz requires one positive width per input vertical cell")
        dz = np.r_[widths[0], widths]
        zf = np.r_[-dz[0], 0., np.cumsum(widths)]
        nx, ny, _ = self.cells
        dx, dy, _ = self.spacing
        coordinates = dict(x=(np.arange(nx)+.5)*dx, y=(np.arange(ny)+.5)*dy,
                           z=.5*(zf[:-1]+zf[1:]), x_face=np.arange(nx+1)*dx,
                           y_face=np.arange(ny+1)*dy, z_face=zf)
        for value in [dz, *coordinates.values()]:
            value.setflags(write=False)
        object.__setattr__(self, 'dz', dz)
        object.__setattr__(self, 'coordinates', coordinates)

    @property
    def cell_shape(self):
        nx, ny, nz = self.cells
        return nz+1, ny, nx

    @property
    def face_shape(self):
        return tuple(n+1 for n in self.cell_shape)

    @property
    def interior(self):
        mask = np.zeros(self.cell_shape, dtype=bool)
        mask[1:-1, 1:-1, 1:-1] = True
        return mask

    def workspace(self):
        from .io import WindWorkspace
        arrays = {k: np.ones(self.cell_shape) for k in ['e', 'f', 'g', 'h', 'm', 'n']}
        arrays.update({k: np.zeros(self.face_shape) for k in ['u0', 'v0', 'w0']})
        flags = np.ones(self.cell_shape, dtype=np.int32)
        flags[0] = 2
        arrays.update(icellflag=flags, ibuilding_flag=np.full(self.cell_shape, -1, dtype=np.int32),
                      dz_array=self.dz.copy(), terrain=np.zeros(self.cell_shape[1:]),
                      icellflag_footprint=np.ones(self.cell_shape[1:], dtype=np.int32))
        return WindWorkspace(arrays, dict(self.coordinates))
