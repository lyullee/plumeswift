"""Resample a DEM on the explicit UTM model grid."""
import numpy as np


def project_dem(raster, domain, epsg, origin):
    from rasterio import band
    from rasterio.enums import MaskFlags, Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform_bounds

    if raster.crs is None:
        raise ValueError('DEM reprojection requires source CRS metadata')
    if not np.isfinite(origin).all() or origin[0] <= 0:
        raise ValueError('DEM reprojection requires a finite UTM origin')
    nx, ny, _ = domain.cells
    dx, dy, _ = domain.spacing
    x, y = origin
    grid = from_origin(x, y+ny*dy, dx, dy)
    heights = np.empty((ny, nx), dtype=np.float32)
    target = f'EPSG:{epsg}'
    masked = bool(set(raster.mask_flag_enums[0]) & {MaskFlags.per_dataset, MaskFlags.alpha})
    bounds = (transform_bounds(target, raster.crs, x, y, x+nx*dx, y+ny*dy)
              if masked or raster.crs.is_geographic else None)
    wraps = bounds is not None and bounds[0] > bounds[2]
    if masked or wraps:
        source, transform = _masked_source(raster, bounds, (nx, ny), wraps=wraps)
        reproject(source, heights, src_crs=raster.crs, src_transform=transform, src_nodata=np.nan,
                  dst_crs=target, dst_transform=grid, dst_nodata=np.nan, resampling=Resampling.bilinear)
    else:
        reproject(band(raster, 1), heights, dst_crs=target, dst_transform=grid,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    if not np.isfinite(heights).all():
        raise ValueError('Reprojected DEM contains nodata or does not cover the full model grid')
    heights = (heights.astype(float)*raster.scales[0]+raster.offsets[0]).astype(np.float32)
    if not np.isfinite(heights).all():
        raise ValueError('DEM scale and offset produced non-finite elevations')
    return heights


def _masked_source(raster, bounds, shape, *, wraps):
    """Read the source window with both its explicit mask and nodata applied."""
    from rasterio.windows import Window

    west, south, east, north = bounds
    if wraps:
        east += 360
        center = (raster.bounds.left+raster.bounds.right)/2
        shift = 360*round((center-(west+east)/2)/360)
        west, east = west+shift, east+shift
    inverse = ~raster.transform
    cols = [inverse.a*x+inverse.b*y+inverse.c for x in (west, east) for y in (south, north)]
    rows = [inverse.d*x+inverse.e*y+inverse.f for x in (west, east) for y in (south, north)]
    low, high = np.array([min(cols), min(rows)]), np.array([max(cols), max(rows)])
    padding = np.ceil((high-low)/min(shape)).astype(int)+2
    left, top = np.floor(low).astype(int)-padding
    right, bottom = np.ceil(high).astype(int)+padding
    top, bottom = max(top, 0), min(bottom, raster.height)
    if wraps and (left < 0 or right > raster.width):
        return _wrapped_window(raster, left, top, right, bottom)
    left, right = max(left, 0), min(right, raster.width)
    if left >= right or top >= bottom:
        raise ValueError('Reprojected DEM contains nodata or does not cover the full model grid')
    window = Window(left, top, right-left, bottom-top)
    return _read_masked_window(raster, window), raster.window_transform(window)


def _read_masked_window(raster, window):
    values = raster.read(1, window=window, masked=True, out_dtype='float32').filled(np.nan)
    if raster.nodata is not None:
        values[values == raster.nodata] = np.nan
    return values


def _wrapped_window(raster, left, top, right, bottom):
    """Stitch the two source edges before bilinear resampling across the seam."""
    from rasterio.windows import Window

    transform = raster.transform
    if (not raster.crs.is_geographic or transform.a <= 0 or transform.b != 0 or transform.d != 0):
        raise ValueError('Wrapping a DEM requires a regular longitude raster')
    pixels_per_turn = 360/transform.a
    period = round(pixels_per_turn)
    if period < 1 or not np.isclose(pixels_per_turn, period, rtol=0, atol=1e-6):
        raise ValueError('Wrapping a DEM requires an integer number of columns per longitude turn')
    if top >= bottom:
        raise ValueError('Reprojected DEM contains nodata or does not cover the full model grid')
    window = Window(left, top, right-left, bottom-top)
    values = np.full((bottom-top, right-left), np.nan, dtype=np.float32)
    for shift in (-period, period, 0):
        start, stop = max(left, shift), min(right, shift+raster.width)
        if start < stop:
            source = Window(start-shift, top, stop-start, bottom-top)
            values[:, start-left:stop-left] = _read_masked_window(raster, source)
    return values, raster.window_transform(window)
