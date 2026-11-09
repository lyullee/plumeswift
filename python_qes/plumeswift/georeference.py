"""Projected domain coordinates shared by geographic stations and vector GIS."""
import numpy as np


def utm_epsg(zone, band=''):
    """WGS84 UTM EPSG code using a UTM latitude band, defaulting to north."""
    if not 1 <= zone <= 60:
        raise ValueError('UTMZone must be between 1 and 60 for CRS transformation')
    band = band.strip().upper()
    # Legacy QES templates use a numeric placeholder for the optional band.
    if band and not band.isdecimal() and (len(band) != 1 or band not in 'CDEFGHJKLMNPQRSTUVWX'):
        raise ValueError('UTMZoneLetter must be a UTM latitude band C through X')
    south = bool(band and len(band) == 1 and band in 'CDEFGHJKLM')
    return (32700 if south else 32600)+zone


def transform_points(points, source_crs, target_crs):
    from rasterio.warp import transform

    points = np.asarray(points, dtype=float)
    x, y = transform(source_crs, target_crs, points[:, 0], points[:, 1])
    result = np.column_stack((x, y))
    if not np.isfinite(result).all():
        raise ValueError('CRS transformation produced non-finite coordinates')
    return result


def geographic_station(longitude, latitude, origin, zone, band):
    if not (-180 <= longitude <= 180 and -80 <= latitude <= 84):
        raise ValueError('Geographic station requires longitude in [-180, 180] and UTM latitude in [-80, 84]')
    if not np.isfinite(origin).all() or origin[0] == 0:
        raise ValueError('Geographic station requires a UTM domain origin')
    epsg = utm_epsg(zone, band)
    if (latitude < 0) != (epsg >= 32700) and latitude != 0:
        raise ValueError('Geographic station latitude disagrees with the domain UTM hemisphere')
    position = transform_points(((longitude, latitude),), 'EPSG:4326', f'EPSG:{epsg}')[0]
    return position.astype(np.float32)-np.asarray(origin, dtype=np.float32)
