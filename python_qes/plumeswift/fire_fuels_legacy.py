"""Standalone catalogue from the historical QES OldFuelProperties.hpp (GPL-3)."""
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class HistoricalFuel:
    """Properties named and typed as in the original C++ classes."""

    windrf: np.float32
    fgi: np.float32
    fueldepthm: np.float32
    savr: int
    fuelmce: np.float32
    fueldens: int
    st: np.float32
    se: np.float32
    weight: int
    fci_d: np.float32
    fct: int
    ichap: int
    fci: np.float32
    fcbr: np.float32
    hfgl: int
    cmbcnst: int
    fuelheat: np.float32
    fuelmc_g: np.float32
    fuelmc_c: int
    slope: np.float32


def _fuel(*values):
    return HistoricalFuel(*(field.type(value) for field, value in zip(fields(HistoricalFuel), values, strict=True)))


_NONBURNABLE = _fuel(*([0]*20))

# The historical header defines named classes rather than a numeric lookup.
HISTORICAL_FUELS = MappingProxyType({
    'ShortGrass': _fuel(
        0.3600, 0.1660, 0.3050, 3500, 0.1200, 32, 0.0555, 0.0100, 7, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'TimberGrass': _fuel(
        0.3600, 0.8970, 0.3050, 2784, 0.1500, 32, 0.0555, 0.0100, 7, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'TallGrass': _fuel(
        0.4400, 0.6750, 0.7620, 1500, 0.2500, 32, 0.0555, 0.0100, 7, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'Chaparral': _fuel(
        0.5500, 2.4680, 1.8290, 1739, 0.2000, 32, 0.0555, 0.0100, 180, 1.1230,
        60, 1, 2.2460, 0.0187, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'Brush': _fuel(
        0.4200, 0.7850, 0.6100, 1683, 0.2000, 32, 0.0555, 0.0100, 100, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'DormantBrush': _fuel(
        0.4400, 1.3450, 0.7620, 1564, 0.2500, 32, 0.0555, 0.0100, 100, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'SouthernRough': _fuel(
        0.4400, 1.0920, 0.7620, 1562, 0.4000, 32, 0.0555, 0.0100, 100, 0.0000,
        60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'TimberClosedLitter': _fuel(
        0.3600, 1.1210, 0.0610, 1889, 0.3000, 32, 0.0555, 0.0100, 900, 1.1210,
        60, 0, 2.2420, 0.0187, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'HarwoodLitter': _fuel(
        0.3600, 0.7800, 0.0610, 2484, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210,
        120, 0, 2.2420, 0.0093, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'TimberLitter': _fuel(
        0.3600, 2.6940, 0.3050, 1764, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210,
        180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'LoggingSlashLight': _fuel(
        0.3600, 2.5820, 0.3050, 1182, 0.1500, 32, 0.0555, 0.0100, 900, 1.1210,
        180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'LoggingSlashMedium': _fuel(
        0.4300, 7.7490, 0.7010, 1145, 0.2000, 32, 0.0555, 0.0100, 900, 1.1210,
        180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'LoggingSlashHeavy': _fuel(
        0.4600, 13.0240, 0.9140, 1159, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210,
        180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0,
    ),
    'Urban': _NONBURNABLE,
    'Snow': _NONBURNABLE,
    'Agricultural': _NONBURNABLE,
    'Water': _NONBURNABLE,
    'Bare': _NONBURNABLE,
})
