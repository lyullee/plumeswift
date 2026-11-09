from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.warp import transform
import shapefile

from plumeswift.georeference import geographic_station, utm_epsg
from plumeswift.gis import read_building_shapefile
from plumeswift.wind_case import read_wind_case


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('zone, band, origin, lon, expected_epsg', [
    (31, 'N', (500000., 0.), 3., 32631),
    (31, '17', (500000., 0.), 3., 32631),
    (56, 'H', (500000., 10000000.), 153., 32756),
])
def test_geographic_station_on_utm_central_meridian(zone, band, origin, lon, expected_epsg):
    assert utm_epsg(zone, band) == expected_epsg
    np.testing.assert_array_equal(geographic_station(lon, 0., origin, zone, band), (0., 0.))


def test_geographic_station_rejects_mismatched_hemisphere():
    with pytest.raises(ValueError, match='hemisphere'):
        geographic_station(153., -33., (500000., 6300000.), 56, 'T')


def test_legacy_geographic_station_can_provide_position_in_local_grid(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for key, value in dict(UTMx=500000, UTMy=4500000, UTMZone=17, UTMZoneLetter='T', halo_x=2, halo_y=3).items():
        simulation.find(key).text = str(value)
    sensor = tree.find('metParams/sensor')
    sensor.find('site_coord_flag').text = '3'
    lon, lat = transform('EPSG:32617', 'EPSG:4326', [500008.], [4500007.])
    ET.SubElement(sensor, 'site_lon').text = str(lon[0])
    ET.SubElement(sensor, 'site_lat').text = str(lat[0])
    path = tmp_path/'winds.xml'
    tree.write(path)

    legacy, = read_wind_case(path, legacy_turbulence_xml=True).stations
    modern, = read_wind_case(path).stations
    assert (legacy.x, legacy.y) == (5., 5.)
    assert (modern.x, modern.y) == (10., 10.)

    sensor.remove(sensor.find('site_xcoord'))
    sensor.remove(sensor.find('site_ycoord'))
    tree.write(path)
    geographic_only, = read_wind_case(path, legacy_turbulence_xml=True).stations
    assert (geographic_only.x, geographic_only.y) == (10., 10.)


@pytest.mark.parametrize('input_kind', ('file', 'archive_default', 'archive_layer', 'archive_member'))
def test_geographic_station_and_courtyard_share_halo_frame(tmp_path, input_kind):
    epsg, origin = 32617, (500000., 4500000.)
    outer = [(8, 7), (8, 13), (14, 13), (14, 7)]
    hole = [(9, 8), (13, 8), (13, 12), (9, 12)]

    def degrees(coordinates):
        xy = np.asarray(coordinates)+origin
        return list(zip(*transform(epsg, 4326, xy[:, 0], xy[:, 1])))

    shp = tmp_path/'building.shp'
    with shapefile.Writer(str(shp), shapeType=shapefile.POLYGON) as writer:
        writer.field('H', 'N')
        writer.poly([degrees(outer), degrees(hole)])
        writer.record(5)
    with pytest.raises(ValueError, match='building.prj'):
        read_building_shapefile(shp, origin=origin, target_epsg=epsg)
    shp.with_suffix('.prj').write_text(CRS.from_epsg(4326).to_wkt(), encoding='utf-8')
    filename = shp.name
    if input_kind != 'file':
        archive = tmp_path/'sites.zip'
        member = 'buildings' if input_kind == 'archive_default' else shp.stem
        with ZipFile(archive, 'w') as zipped:
            for suffix in ('.shp', '.shx', '.dbf', '.prj'):
                sidecar = shp.with_suffix(suffix)
                zipped.write(sidecar, f'geometry/{member}{suffix}')
                zipped.write(sidecar, f'other{suffix}')
        with pytest.raises(ValueError, match='Select exactly one'):
            read_building_shapefile(archive, origin=origin, target_epsg=epsg)
        filename = f'{archive.name}/geometry/{shp.name}' if input_kind == 'archive_member' else archive.name

    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    sim = tree.find('simulationParameters')
    for name, value in dict(UTMZone=17, UTMZoneLetter='T', UTMx=origin[0], UTMy=origin[1], halo_x=2, halo_y=3).items():
        sim.find(name).text = str(value)
    ET.SubElement(sim, 'reprojectShapefiles').text = 'true'
    sensor = tree.find('metParams/sensor')
    sensor.find('site_coord_flag').text = '3'
    lon, lat = degrees([outer[0]])[0]
    ET.SubElement(sensor, 'site_lon').text = str(lon)
    ET.SubElement(sensor, 'site_lat').text = str(lat)
    buildings = tree.find('buildingsParams')
    ET.SubElement(buildings, 'SHPFile').text = filename
    if input_kind == 'archive_layer':
        ET.SubElement(buildings, 'SHPBuildingLayer').text = shp.stem
    path = tmp_path/'winds.xml'
    tree.write(path)

    case = read_wind_case(path)
    station, = case.stations
    building, = case.buildings
    assert (station.x, station.y) == (10., 10.)
    assert set(building.vertices) == {(x+2, y+3) for x, y in outer}
    assert len(building.holes) == 1
    assert set(building.holes[0]) == {(x+2, y+3) for x, y in hole}
