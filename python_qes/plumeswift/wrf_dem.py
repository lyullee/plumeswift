"""Export WRF or WRF-SFIRE terrain and measured wind profiles to a QES project."""
from datetime import datetime, timedelta
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

from .io import _open_netcdf
from .measured_profile import MeasuredProfile
from .wrf import _atmospheric_wind_window, _field, _ROUGHNESS


def _projection(nc):
    from rasterio.crs import CRS

    if int(getattr(nc, 'MAP_PROJ', 1)) != 1:
        raise ValueError('wrf2dem requires a Lambert conformal WRF grid')
    lat1, lat2, lat0, lon0 = (float(nc.getncattr(name))
                              for name in ('TRUELAT1', 'TRUELAT2', 'MOAD_CEN_LAT', 'STAND_LON'))
    return CRS.from_dict(proj='lcc', lat_1=lat1, lat_2=lat2, lat_0=lat0, lon_0=lon0, a=6370000, b=6370000,
                         units='m')


def _terrain(nc, crs):
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    fire_fields = ('ZSF', 'FXLONG', 'FXLAT')
    fire_fields_present = tuple(name in nc.variables for name in fire_fields)
    if any(fire_fields_present) and not all(fire_fields_present):
        raise ValueError('Incomplete WRF fire mesh requires ZSF, FXLONG and FXLAT')
    has_fire_mesh = all(fire_fields_present)
    fields = fire_fields if has_fire_mesh else ('HGT', 'XLONG', 'XLAT')
    elevations, lon, lat = (_field(nc, name, 3, 0) for name in fields)
    ny, nx = elevations.shape
    sx = sy = 1
    if has_fire_mesh:
        ay, ax = nc['HGT'].shape[1:]
        sx, sy = nx//(ax+1), ny//(ay+1)
    if min(nx, ny) < 2 or min(sx, sy) < 1 or lon.shape != elevations.shape or lat.shape != elevations.shape:
        raise ValueError('WRF terrain coordinates and grid dimensions disagree')
    dx, dy = float(nc.DX)/sx, float(nc.DY)/sy
    if not np.isfinite((dx, dy)).all() or min(dx, dy) <= 0:
        raise ValueError('WRF terrain requires positive cell spacing')
    rows, columns = np.array([0, 0, ny-1, ny-1]), np.array([0, nx-1, 0, nx-1])
    x, y = np.asarray(transform('EPSG:4326', crs, lon[rows, columns], lat[rows, columns]))
    west_nodes, south_nodes = x-columns*dx, y-rows*dy
    if (not np.isfinite((x, y)).all()
            or np.ptp(west_nodes) > max(1., .1*dx) or np.ptp(south_nodes) > max(1., .1*dy)):
        raise ValueError('WRF terrain coordinates do not fit the projected rectangular mesh')
    west, south = float(np.mean(west_nodes)-dx/2), float(np.mean(south_nodes)-dy/2)
    return elevations[::-1].astype(np.float32), from_origin(west, south+ny*dy, dx, dy), fields[0]


def _timestamps(nc, count):
    if 'Times' in nc.variables:
        stamps = nc['Times']
        stamps.set_auto_chartostring(False)
        return tuple(datetime.fromisoformat(np.ma.filled(stamp, b'').tobytes().decode('ascii').strip('\x00 ')
                                            .replace('_', 'T')) for stamp in stamps[:count])
    start = getattr(nc, 'START_DATE', getattr(nc, 'SIMULATION_START_DATE', '2017-11-17_15:00:00'))
    origin = datetime.fromisoformat(start.replace('_', 'T'))
    elapsed = _field(nc, 'XTIME', 1, slice(count)).astype(float)*60 if 'XTIME' in nc.variables else np.arange(count)
    return tuple(origin+timedelta(seconds=round(float(seconds))) for seconds in elapsed)


def _project_xml(nc, crs, terrain, raster_transform, dem_name, sample, nz):
    from rasterio.warp import transform

    height, width = terrain.shape
    dx, dy = raster_transform.a, -raster_transform.e
    west, south = raster_transform.c, raster_transform.f-height*dy
    lon, lat = (_field(nc, name, 3, 0) for name in ('XLONG', 'XLAT'))
    if lon.shape != lat.shape:
        raise ValueError('WRF atmosphere coordinates disagree')
    rows, columns = np.indices(lon.shape)
    x, y = np.asarray(transform('EPSG:4326', crs, lon.ravel(), lat.ravel())).reshape(2, *lon.shape)
    x, y = x-west, y-south
    interior = (x > 0) & (x < (width-.5)*dx) & (y > 0) & (y < (height-.5)*dy)
    sampled = interior & (rows % sample == 0) & (columns % sample == 0)
    if not sampled.any() and interior.any():
        sampled.flat[np.flatnonzero(interior)[0]] = True
    rows, columns = np.nonzero(sampled)
    if not rows.size:
        raise ValueError('WRF atmosphere has no stations inside the exported DEM')
    count = min(2, nc['U'].shape[0])
    timestamps = _timestamps(nc, count)
    profiles = [[] for _ in rows]
    for record, stamp in enumerate(timestamps):
        ground, land, z, u, v, row_start, column_start = _atmospheric_wind_window(
            nc, lon.shape, rows, columns, record=record, require_land=False)
        for station, (row, column) in enumerate(zip(rows, columns, strict=True)):
            j, i = row-row_start, column-column_start
            altitude = z[:, j, i]-ground[j, i]
            keep = (altitude >= 22.) & (altitude <= 330.)
            if not keep.any():
                keep = altitude > 0
            if not keep.any():
                continue
            east, north = u[keep, j, i], v[keep, j, i]
            land_id = int(land[j, i])
            roughness = _ROUGHNESS[land_id] if 0 < land_id < len(_ROUGHNESS) else .1
            profile = MeasuredProfile(tuple(altitude[keep]), tuple(np.hypot(east, north)),
                                      tuple(np.mod(270.-np.rad2deg(np.arctan2(north, east)), 360.)), roughness)
            profiles[station].append((stamp, profile))

    def add(node, fields):
        for key, value in fields.items():
            ET.SubElement(node, key).text = str(value)

    root = ET.Element('QESWindsParameters')
    add(ET.SubElement(root, 'simulationParameters'), dict(
        DEM=dem_name, domain=f'{width} {height} {nz}', cellSize=f'{dx} {dy} 1',
        totalTimeIncrements=count, maxIterations=500, meshTypeFlag=0, originFlag=0))
    met = ET.SubElement(root, 'metParams')
    add(met, {'z0_domain_flag': 0})
    for row, column, timeline in zip(rows, columns, profiles, strict=True):
        if not timeline or len(timeline) != count:
            continue
        sensor = ET.SubElement(met, 'sensor')
        add(sensor, dict(site_coord_flag=1, site_xcoord=x[row, column], site_ycoord=y[row, column]))
        for stamp, profile in timeline:
            series = ET.SubElement(sensor, 'timeSeries')
            add(series, dict(timeStamp=stamp.isoformat(timespec='seconds'), boundaryLayerFlag=4,
                             siteZ0=profile.roughness, reciprocal=0))
            for key, values in (('height', profile.heights), ('speed', profile.speeds), ('direction', profile.directions)):
                for value in values:
                    ET.SubElement(series, key).text = str(value)
    stations = len(met.findall('sensor'))
    if stations == 0:
        raise ValueError('WRF atmosphere has no complete profiles above terrain')
    output = ET.SubElement(root, 'fileOptions')
    add(output, dict(outputFlag=1, massConservedFlag=0, sensorVelocityFlag=0, staggerdVelocityFlag=0))
    for field in ('u', 'v', 'w', 'icell'):
        ET.SubElement(output, 'outputFields').text = field
    ET.indent(root)
    return ET.ElementTree(root), stations, timestamps


def convert_wrf_dem(wrf_path, dem_path, *, sample=12, nz=300):
    """Write a north-up floating-point GeoTIFF and its runnable QES XML."""
    import rasterio

    sample, nz = index(sample), index(nz)
    if sample <= 0 or nz < 3:
        raise ValueError('WRF conversion requires positive station stride and at least three vertical cells')
    wrf_path, dem_path = Path(wrf_path), Path(dem_path)
    xml_path = dem_path.with_name(dem_path.name+'.xml')
    if wrf_path.resolve() in (dem_path.resolve(), xml_path.resolve()):
        raise ValueError('WRF conversion outputs must differ from the input')
    with _open_netcdf(wrf_path) as nc:
        crs = _projection(nc)
        elevations, transform, terrain_source = _terrain(nc, crs)
        xml, stations, timestamps = _project_xml(nc, crs, elevations, transform, dem_path.name, sample, nz)
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dem_path.parent) as temporary:
        temporary = Path(temporary)
        staged_dem, staged_xml = temporary/'dem.tif', temporary/'winds.xml'
        with rasterio.open(staged_dem, 'w', driver='GTiff', width=elevations.shape[1], height=elevations.shape[0],
                           count=1, dtype='float32', crs=crs, transform=transform) as out:
            out.write(elevations, 1)
        xml.write(staged_xml, encoding='utf-8', xml_declaration=True)
        staged_dem.replace(dem_path)
        staged_xml.replace(xml_path)
    return dict(dem=str(dem_path), xml=str(xml_path), stations=stations, terrain_source=terrain_source,
                timestamps=[stamp.isoformat(timespec='seconds') for stamp in timestamps],
                cells=[elevations.shape[1], elevations.shape[0], nz], spacing_m=[transform.a, -transform.e, 1.])
