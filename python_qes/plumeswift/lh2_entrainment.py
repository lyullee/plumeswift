"""Density-driven exchange between a cloud and neighboring ambient cells."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DensityEntrainment:
    sampler: object
    solid: np.ndarray
    coefficient: float
    gravity: float = 9.80665
    seed: int = 0
    exchanged_volume_m3: float = field(default=0., init=False)
    transferred_h2_kg: float = field(default=0., init=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if not np.isfinite([self.coefficient, self.gravity]).all() or min(self.coefficient, self.gravity) <= 0:
            raise ValueError('Density entrainment requires a positive finite coefficient and gravity')
        self.rng = np.random.default_rng(self.seed)

    def mix(self, batch, thermo, dt):
        """Sample face exchange, transporting whole H₂ mass/enthalpy markers."""
        if not len(batch.mass):
            return
        shape = self.solid.shape
        flat = np.ravel_multi_index(self.sampler.cell_indices(batch.positions), shape)
        cells, inverse = np.unique(flat, return_inverse=True)
        mass = np.bincount(inverse, weights=batch.mass)
        enthalpy = np.bincount(inverse, weights=batch.mass * batch.specific_enthalpy)
        indices = np.stack(np.unravel_index(cells, shape)[::-1], axis=1)
        faces = (self.sampler.xf, self.sampler.yf, self.sampler.zf)
        widths = np.column_stack([np.diff(edges)[indices[:, axis]] for axis, edges in enumerate(faces)])
        volume = widths.prod(axis=1)
        concentration = mass / volume
        density = thermo.evaluate(concentration, enthalpy / volume).density
        contrast = density / thermo.ambient_density - 1.
        # Stable heavy clouds entrain through their gravity-current sides.
        # Light clouds draw ambient air across their rising upper interface.
        scale = self.coefficient * np.sqrt(self.gravity * widths[:, 2])
        side_speed = scale * np.sqrt(np.maximum(contrast, 0.))
        top_speed = scale * np.sqrt(np.maximum(-contrast, 0.))
        directions = ((0, -1), (0, 1), (1, -1), (1, 1), (2, 1))
        rates = np.zeros((len(cells), len(directions)))
        for face, (axis, offset) in enumerate(directions):
            neighbor = indices[:, axis] + offset
            interior = (neighbor >= 1) & (neighbor < shape[2 - axis] - 1)
            target = indices.copy()
            target[:, axis] = np.clip(neighbor, 0, shape[2 - axis] - 1)
            target_tuple = tuple(target.T[::-1])
            neighbor_ids = np.ravel_multi_index(target_tuple, shape)
            loc = np.minimum(np.searchsorted(cells, neighbor_ids), len(cells) - 1)
            neighbor_c = np.where(cells[loc] == neighbor_ids, concentration[loc], 0.)
            gradient = np.divide(neighbor_c, concentration, out=np.ones_like(mass), where=mass > 0)
            speed = top_speed if axis == 2 else side_speed
            rates[:, face] = (speed / widths[:, axis] * np.maximum(0., 1 - gradient)
                              * (interior & ~self.solid[target_tuple]))

        cumulative = np.cumsum(rates, axis=1)
        rate = cumulative[inverse, -1]
        candidates = np.flatnonzero((rate > 0) & (batch.mass > 0))
        moved = candidates[self.rng.random(len(candidates)) < -np.expm1(-rate[candidates] * dt)]
        self.exchanged_volume_m3 += float(np.dot(cumulative[:, -1], volume) * dt)
        if not len(moved):
            return
        draw = self.rng.random(len(moved)) * rate[moved]
        chosen = np.sum(draw[:, None] >= cumulative[inverse[moved]], axis=1)
        for face, (axis, offset) in enumerate(directions):
            markers = moved[chosen == face]
            donor = inverse[markers]
            source = indices[donor, axis]
            neighbor = source + offset
            edges = faces[axis]
            # Enter the neighboring cell through the shared face. Mirror the
            # donor fraction so a marker near the face stays near the face.
            fraction = (batch.positions[markers, axis] - edges[source]) / widths[donor, axis]
            fraction = np.clip(1 - fraction, 1e-7, 1 - 1e-7)
            batch.positions[markers, axis] = edges[neighbor] + fraction * (edges[neighbor + 1] - edges[neighbor])
        self.transferred_h2_kg += float(batch.mass[moved].sum())
