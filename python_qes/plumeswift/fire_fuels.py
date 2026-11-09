"""QES FuelProperties.hpp and FuelMap.cpp fuel catalog (GPL-3)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Fuel:
    # The table retains the original imperial units.
    one_hour: float
    ten_hour: float
    hundred_hour: float
    live_herb: float
    live_woody: float
    sav_one_hour: float
    sav_herb: float
    sav_woody: float
    depth: float
    heat_content: float
    density: float
    extinction_moisture: float


FUELS = {
    1: Fuel(0.7405, 0, 0, 0, 0, 3500, 0, 0, 1, 7496.2, 32.0, 12),  # ShortGrass
    2: Fuel(4.0015, 0, 0, 0, 0, 2784, 0, 0, 1, 7496.2, 32.0, 15),  # TimberGrass
    3: Fuel(3.0111, 0, 0, 0, 0, 1500, 0, 0, 2.5, 7496.2, 32.0, 25),  # TallGrass
    4: Fuel(11.0098, 0, 0, 0, 0, 1739, 0, 0, 6, 7496.2, 32.0, 20),  # Chaparral
    5: Fuel(3.5019, 0, 0, 0, 0, 1683, 0, 0, 2, 7496.2, 32.0, 20),  # Brush
    6: Fuel(6.0000, 0, 0, 0, 0, 1564, 0, 0, 2.5, 7496.2, 32.0, 25),  # DormantBrush
    7: Fuel(4.8714, 0, 0, 0, 0, 1562, 0, 0, 2.5, 7496.2, 32.0, 40),  # SouthernRough
    8: Fuel(5.0008, 0, 0, 0, 0, 1884, 0, 0, 0.2, 7496.2, 32.0, 30),  # TimberClosedLitter
    9: Fuel(4.3718, 0, 0, 0, 0, 2484, 0, 0, 0.2, 7496.2, 32.0, 25),  # HarwoodLitter
    10: Fuel(12.0179, 0, 0, 0, 0, 1764, 0, 0, 1, 7496.2, 32.0, 25),  # TimberLitter
    11: Fuel(11.5183, 0, 0, 0, 0, 1182, 0, 0, 1, 7496.2, 32.0, 15),  # LoggingSlashLight
    12: Fuel(34.5683, 0, 0, 0, 0, 1145, 0, 0, 2.3, 7496.2, 32.0, 20),  # LoggingSlashMedium
    13: Fuel(51.1000, 0, 0, 0, 0, 1159, 0, 0, 4.1, 7496.2, 32.0, 25),  # LoggingSlashHeavy
    91: Fuel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Urban
    92: Fuel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Snow
    93: Fuel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Agricultural
    98: Fuel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Water
    99: Fuel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Bare
    101: Fuel(0.10, 0.00, 0.00, 0.30, 0.00, 2200, 2000, 9999, 0.4, 8000, 32.00, 15),  # GR1
    102: Fuel(0.10, 0.00, 0.00, 1.00, 0.00, 2000, 1800, 9999, 1.0, 8000, 32.00, 15),  # GR2
    103: Fuel(0.10, 0.40, 0.00, 1.50, 0, 1500, 1300, 9999, 2.0, 8000, 32.00, 30),  # GR3
    104: Fuel(0.25, 0.00, 0.00, 1.90, 0, 2000, 1800, 9999, 2.0, 8000, 32.00, 15),  # GR4
    105: Fuel(0.40, 0.00, 0.00, 2.50, 0, 1800, 1600, 9999, 1.5, 8000, 32.00, 40),  # GR5
    106: Fuel(0.10, 0.00, 0.00, 3.40, 0, 2200, 2000, 9999, 1.5, 9000, 32.00, 40),  # GR6
    107: Fuel(1.00, 0.00, 0.00, 5.40, 0, 2000, 1800, 9999, 3.0, 8000, 32.00, 15),  # GR7
    108: Fuel(0.50, 1.00, 0.00, 7.30, 0, 1500, 1300, 9999, 4.0, 8000, 32.00, 30),  # GR8
    109: Fuel(1.00, 1.00, 0.00, 9.00, 0, 1800, 1600, 9999, 5.0, 8000, 32.00, 40),  # GR9
    121: Fuel(0.20, 0.00, 0.00, 0.50, 0.65, 2000, 1800, 1800, 0.9, 8000, 32.00, 15),  # GS1
    122: Fuel(0.50, 0.50, 0.00, 0.60, 1.00, 2000, 1800, 1800, 1.5, 8000, 32.0, 15),  # GS2
    123: Fuel(0.30, 0.25, 0.00, 1.45, 1.25, 1800, 1600, 1600, 1.8, 8000, 32.00, 40),  # GS3
    124: Fuel(1.90, 0.30, 0.10, 3.40, 7.10, 1800, 1600, 1600, 2.1, 8000, 32.00, 40),  # GS4
    141: Fuel(0.25, 0.25, 0.00, 0.15, 1.30, 2000, 1800, 1600, 1.0, 8000, 32.00, 15),  # SH1
    142: Fuel(1.35, 2.40, 0.75, 0.00, 3.85, 2000, 9999, 1600, 1.0, 8000, 32.00, 15),  # SH2
    143: Fuel(0.45, 3.00, 0.00, 0.00, 6.20, 1600, 9999, 1400, 2.4, 8000, 32.00, 40),  # SH3
    144: Fuel(0.85, 1.15, 0.20, 0.00, 2.55, 2000, 1800, 1600, 3.0, 8000, 32.00, 30),  # SH4
    145: Fuel(3.60, 2.10, 0.00, 0.00, 2.90, 750, 9999, 1600, 6.0, 8000, 32.00, 15),  # SH5
    146: Fuel(2.90, 1.45, 0.00, 0.00, 1.40, 750, 9999, 1600, 2.0, 8000, 32.00, 30),  # SH6
    147: Fuel(3.50, 5.30, 2.20, 0.00, 3.40, 750, 9999, 1600, 6.0, 8000, 32.00, 15),  # SH7
    148: Fuel(2.05, 3.40, 0.85, 0.00, 4.35, 750, 9999, 1600, 3.0, 8000, 32.00, 40),  # SH8
    149: Fuel(4.50, 2.45, 0.00, 1.55, 7.00, 750, 1800, 1500, 4.4, 8000, 32.00, 40),  # SH9
    161: Fuel(0.20, 0.90, 1.50, 0.20, 0.90, 2000, 1800, 1600, 0.6, 8000, 32.00, 20),  # TU1
    162: Fuel(0.95, 1.80, 1.25, 0.00, 0.20, 2000, 9999, 1600, 1.0, 8000, 32.00, 30),  # TU2
    163: Fuel(1.10, 0.15, 0.25, 0.65, 1.10, 1800, 1600, 1400, 1.3, 8000, 32.00, 30),  # TU3
    164: Fuel(4.50, 0.00, 0.00, 0.00, 2.00, 2300, 9999, 2000, 0.5, 8000, 32.00, 12),  # TU4
    165: Fuel(4.00, 4.00, 3.00, 0.00, 3.00, 1500, 9999, 750, 1.0, 8000, 32.00, 25),  # TU5
    181: Fuel(1.00, 2.20, 3.60, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00, 30),  # TL1
    182: Fuel(1.40, 2.30, 2.20, 0.00, 0.00, 2000, 9999, 9999, 0.2, 8000, 32.00, 25),  # TL2
    183: Fuel(0.50, 2.20, 2.80, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00, 20),  # TL3
    184: Fuel(0.50, 1.50, 4.20, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00, 25),  # TL4
    185: Fuel(1.15, 2.50, 4.40, 0.00, 0.00, 2000, 9999, 9999, 0.6, 8000, 32.00, 25),  # TL5
    186: Fuel(2.40, 1.20, 1.20, 0.00, 0.00, 2000, 9999, 9999, 0.3, 8000, 32.00, 25),  # TL6
    187: Fuel(0.30, 1.40, 8.10, 0.00, 0.00, 2000, 9999, 9999, 0.4, 8000, 32.00, 25),  # TL7
    188: Fuel(5.80, 1.40, 1.10, 0.00, 0.00, 1800, 9999, 9999, 0.3, 8000, 32.00, 35),  # TL8
    189: Fuel(6.65, 3.30, 4.15, 0.00, 0.00, 1800, 9999, 1600, 0.6, 8000, 32.00, 35),  # TL9
    201: Fuel(1.50, 3.00, 11.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00, 25),  # SB1
    202: Fuel(4.50, 4.25, 4.00, 0.00, 0.00, 2000, 9999, 9999, 1.0, 8000, 32.00, 25),  # SB2
    203: Fuel(5.50, 2.75, 3.00, 0.00, 0.00, 2000, 9999, 9999, 1.2, 8000, 32.00, 25),  # SB3
    204: Fuel(5.25, 3.50, 5.25, 0.00, 0.00, 2000, 9999, 9999, 2.7, 8000, 32.00, 25),  # SB4
}


def fuel_properties(codes, *, raster=False):
    """Resolve the native fuel IDs, with SB4 fallback for raster cells."""
    codes = np.asarray(codes)
    if not np.isfinite(codes).all() or np.any(codes != np.floor(codes)):
        raise ValueError('Expected integral fuel IDs')
    unique, inverse = np.unique(codes, return_inverse=True)
    if not raster and any(code not in FUELS for code in unique):
        raise ValueError('Unknown uniform QES fuel ID')
    columns = np.array([tuple(vars(FUELS.get(code, FUELS[204])).values()) for code in unique]).T
    return Fuel(*(column[inverse].reshape(codes.shape) for column in columns))


def read_fuel_raster(path, shape, dx, dy):
    """Load the native FuelRead pixel window on a horizontal fire grid."""
    import rasterio

    rows, cols = shape
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError('Invalid fire grid spacing')
    with rasterio.open(path) as raster:
        transform = raster.transform
        if transform.a == 0 or transform.e == 0:
            raise ValueError('Fire fuel raster requires nonzero x and y pixel steps')
        # FuelRead indexes the band directly. Only the two diagonal pixel
        # steps determine the strides, even when the affine has off-axis terms.
        sx = np.float32(dx)/np.float32(transform.a)
        sy = np.float32(dy)/np.float32(-transform.e)
        if np.ceil(sx)*np.ceil(sy) < 0:
            raise ValueError('Fire fuel raster would allocate a negative native histogram size')
        if sx < 1:
            if sx < 0 or sy < 0:
                # A negative stride makes one native sampling loop empty.
                return np.zeros(shape, dtype=int)
            xs = np.floor(np.arange(cols, dtype=np.float32)*sx).astype(int)
            ys = np.floor(np.arange(rows, dtype=np.float32)*sy).astype(int)+int(np.ceil(sy))-1
        else:
            xs, ys = np.arange(cols), np.arange(rows)
        if not len(xs) or not len(ys) or xs[-1] >= raster.width or ys[-1] >= raster.height:
            raise ValueError('Fire fuel raster does not cover the native sampling window')
        values = raster.read(1, window=((0, int(ys[-1])+1), (0, int(xs[-1])+1)),
                             out_dtype='float32')[np.ix_(ys, xs)]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite fire fuel codes')
    codes = np.trunc(values).astype(int)
    if sx < 1:
        if np.any((codes < 0) | (codes > 205)):
            raise ValueError('Fuel code is outside the native raster histogram')
        # Native floor(sx) is zero, so each sampled row overwrites slot
        # zero. All remaining histogram samples are zero, winning ties.
        if sy > 1:
            codes.fill(0)
    return codes[::-1].copy()
