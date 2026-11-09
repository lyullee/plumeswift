import gzip
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from plumeswift.dem_projection import project_dem
from plumeswift.domain import Domain
from plumeswift.wind_case import read_wind_case


ROOT = Path(__file__).resolve().parents[2]
ORIGIN = (500000.5, 4500000.25)


@pytest.mark.parametrize('path_style', ['member', 'gdal', 'gdal_braced', 'gdal_relative', 'gzip'])
def test_dem_compression_preserves_terrain_and_wind(tmp_path, monkeypatch, path_style):
    raster_path = tmp_path/'terrain.tif'
    rows, columns = np.indices((10, 10))
    elevations = (25+.75*rows+.5*columns).astype(np.float32)
    with rasterio.open(raster_path, 'w', driver='GTiff', width=10, height=10, count=1, dtype='float32',
                       transform=from_origin(100, 210, 1, 1)) as raster:
        raster.write(elevations, 1)
    archive = tmp_path/'rasters.zip'
    member = 'dataset/terrain.tif'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as container:
        container.write(raster_path, member)
    gzip_path = tmp_path/'terrain.tif.gz'
    gzip_path.write_bytes(gzip.compress(raster_path.read_bytes()))
    monkeypatch.chdir(tmp_path)

    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    sim = tree.find('simulationParameters')
    sim.find('domain').text = '10 10 20'
    dem = ET.SubElement(sim, 'DEM')
    project = tmp_path/'project'
    project.mkdir()
    xml = project/'winds.xml'
    dem.text = str(raster_path)
    tree.write(xml)
    expected = read_wind_case(xml, compute_turbulence=False).initialize().arrays
    dem.text = {'member': f'../{archive.name}/{member}', 'gdal': f'/vsizip/{archive.as_posix()}/{member}',
                'gdal_braced': f'/vsizip/{{{archive.as_posix()}}}/{member}',
                'gdal_relative': f'/vsizip/{archive.name}/{member}', 'gzip': f'/vsigzip/{gzip_path.name}'}[path_style]
    tree.write(xml)
    case = read_wind_case(xml, compute_turbulence=False)
    metadata = case.terrain.describe()
    if path_style == 'gzip':
        assert metadata['gzip_encoded']
        assert Path(metadata['path']) == gzip_path
    else:
        assert metadata['archive_member'] == member
        assert Path(metadata['path']) == archive
    assert len(metadata['sha256']) == 64
    actual = case.initialize().arrays
    assert np.max(actual['terrain']) > 0
    for key in ('terrain', 'icellflag', 'u0', 'v0', 'w0'):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('failure', ['nodata', 'coverage', 'crs', 'mask'])
def test_dem_reprojection_rejects_unavailable_elevations(tmp_path, failure):
    domain = Domain((8, 8, 12), (1., 1., 1.))
    x, y = ORIGIN
    values = np.full((8, 8), 43., dtype=np.float32)
    if failure == 'nodata':
        values[4, 4] = -9999.
    grid = from_origin(x+(4 if failure == 'coverage' else 0), y+8, 1, 1)
    path = tmp_path/'terrain.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=8, height=8, count=1, dtype='float32',
                       transform=grid, crs=None if failure == 'crs' else 32617, nodata=-9999.) as raster:
        raster.write(values, 1)
        if failure == 'mask':
            valid = np.full(values.shape, 255, dtype=np.uint8)
            valid[4, 4] = 0
            raster.write_mask(valid)
    with rasterio.open(path) as raster:
        with pytest.raises(ValueError, match='source CRS' if failure == 'crs' else 'nodata or does not cover'):
            project_dem(raster, domain, 32617, ORIGIN)


def test_dem_reprojection_uses_subpixel_origin_and_physical_elevations(tmp_path):
    x, y = ORIGIN
    rows, cols = np.indices((8, 9))
    values = (400.+2*cols+4*rows).astype(np.float32)
    with rasterio.open(tmp_path/'terrain.tif', 'w', driver='GTiff', width=9, height=8, count=1, dtype='float32',
                       transform=from_origin(x-.5, y+8, 1., 1.), crs=32617) as raster:
        raster.write(values, 1)
        raster.scales = (.5,)
        raster.offsets = (-140.,)
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    for name, setting in dict(domain='8 8 24', originFlag=1, UTMZone=17, UTMZoneLetter='T', UTMx=x, UTMy=y).items():
        simulation.find(name).text = str(setting)
    ET.SubElement(simulation, 'DEM').text = 'terrain.tif'
    ET.SubElement(simulation, 'reprojectDEM').text = 'true'
    path = tmp_path/'wind.xml'
    tree.write(path)
    case = read_wind_case(path)
    assert case.terrain.target_epsg == 32617
    mesh, _, datum = case.terrain.geometry_with_datum(case.domain)
    assert datum == 60.5
    np.testing.assert_array_equal(mesh.height([0., 1., 2.], [0., 0., 0.]), [0., 1., 2.])


@pytest.mark.parametrize('epsg', [32601, 32660])
@pytest.mark.parametrize('masked', [False, True])
def test_dem_interpolates_across_antimeridian(tmp_path, epsg, masked):
    domain = Domain((4, 4, 3), (50_000., 50_000., 1.))
    x, y = transform('EPSG:4326', f'EPSG:{epsg}', [180.], [0.])
    origin = (x[0]-100_000., y[0]-100_000.)
    rows, cols = np.indices((12, 360))
    values = (100 + rows + ((cols+180) % 360 - 180)).astype(np.float32)
    valid = np.full(values.shape, 255, dtype=np.uint8)
    if masked:
        valid[4, 359] = 0
        values[4, 359] = 9999
    seam_columns = (np.arange(12)+354) % 360
    outputs = []
    for filename, west, data, mask in (
        ('global.tif', -180, values, valid),
        ('unwrapped.tif', 174, values[:, seam_columns], valid[:, seam_columns]),
    ):
        path = tmp_path/filename
        with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                           count=1, dtype='float32', crs=4326, transform=from_origin(west, 6, 1, 1)) as raster:
            raster.write(data, 1)
            if masked:
                raster.write_mask(mask)
        with rasterio.open(path) as raster:
            outputs.append(project_dem(raster, domain, epsg, origin))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-5)
    assert outputs[0].max() < 110
