"""Legacy Mesh.cpp ray mixing and SphereDirections.h port (GPL-3).

This is the public CPU utility, not the separate OptiX method-3 algorithm.
The original independently sampled phi for x/y/z is intentionally retained.
"""
import numpy as np


def legacy_terrain_mixing(workspace, domain):
    """Connect the legacy terrain rays to the wind model's finite mixing field."""
    from .turbulence import default_mixing_distance

    if workspace.terrain_mesh is None:
        raise ValueError('turbParams/method=2 requires a terrain mesh')
    if np.any(domain.dz != domain.dz[0]):
        raise ValueError('turbParams/method=2 requires uniform vertical spacing')
    initial = default_mixing_distance(workspace)
    flags = workspace.arrays['icellflag_initial']
    distances = LegacyMeshMixing(workspace.terrain_mesh).calculate(
        flags, (*domain.spacing[:2], domain.dz[0]), initial=initial)
    # The utility leaves infinity if its sampled rays all miss the mesh.
    return np.where(np.isfinite(distances), distances, initial)


def sphere_directions(samples=512, theta_bounds=(-1., 1.), phi_bounds=(0., 2*3.14159265359)):
    if int(samples) != samples or samples < 0:
        raise ValueError('Direction count must be a nonnegative integer')
    low, high = np.asarray(theta_bounds, dtype=np.float32)
    plow, phigh = np.asarray(phi_bounds, dtype=np.float32)
    if not np.isfinite([low, high, plow, phigh]).all() or not -1 <= low <= high <= 1 or plow > phigh:
        raise ValueError('Invalid sphere direction bounds')
    # std::mt19937(303), libstdc++ uniform_real_distribution<float> uses
    # one 32-bit result per sample. Keep float32 generator rounding.
    bits = np.random.RandomState(303).randint(0, 2**32, size=(int(samples), 4), dtype=np.uint32)
    unit = np.minimum(bits.astype(np.float32)/np.float32(2**32), np.nextafter(np.float32(1), np.float32(0)))
    theta = np.arcsin(low+(high-low)*unit[:, 0])
    phi = plow+(phigh-plow)*unit[:, 1:]
    direction = np.column_stack([np.cos(theta)*np.cos(phi[:, 0]), np.sin(phi[:, 1]), np.cos(theta)*np.sin(phi[:, 2])])
    direction /= np.sqrt(np.sum(direction*direction, axis=1))[:, None]
    return np.r_[direction, np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)]


class LegacyMeshMixing:
    def __init__(self, mesh, directions=None):
        self.mesh = mesh
        self.directions = sphere_directions() if directions is None else np.asarray(directions, dtype=float)
        if self.directions.ndim != 2 or self.directions.shape[1:] != (3,) or not np.isfinite(self.directions).all() or np.any(np.linalg.norm(self.directions, axis=1) == 0):
            raise ValueError('Mixing rays require finite, nonzero N x 3 directions')

    def distance(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not np.isfinite(positions).all():
            raise ValueError('Mixing origins require finite N x 3 points')
        distances = np.full(len(positions), np.inf)
        for index, origin in enumerate(positions):
            for direction in self.directions:
                hit = self.mesh.ray_hit(origin, direction)
                if hit is not None:
                    distances[index] = min(distances[index], hit.distance)
        return distances

    def calculate(self, flags, spacing, initial=None):
        """Native utility grid origin (k+.5)*dz, including its ghost convention.

        Only exact flag 1 cells are updated; other initial values are preserved.
        No sampled intersection leaves infinity, exactly as in Mesh.cpp.
        """
        flags = np.asarray(flags)
        spacing = np.asarray(spacing, dtype=float)
        if flags.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError('Expected 3D flags and positive xyz spacing')
        result = np.zeros(flags.shape) if initial is None else np.array(initial, dtype=float, copy=True)
        if result.shape != flags.shape:
            raise ValueError('Initial mixing length shape differs from flags')
        k, j, i = np.nonzero(flags == 1)
        points = (np.column_stack([i, j, k])+.5)*spacing
        result[k, j, i] = self.distance(points)
        return result
