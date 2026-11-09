from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.io import _open_netcdf, read_wind_workspace
from plumeswift.wind_visualization import FIELDS, read_output_fields


ROOT = Path(__file__).resolve().parents[2]


def test_output_fields_validate_native_first_all_and_duplicate_rules():
    def options(*fields):
        node = ET.Element('fileOptions')
        for name in fields:
            ET.SubElement(node, 'outputFields').text = name
        return node

    assert read_output_fields(options()) == FIELDS
    assert read_output_fields(options('all', 'u')) == FIELDS
    assert read_output_fields(options('v', 'icellInitial', 'v')) == ('icellInitial', 'v')
    with pytest.raises(ValueError, match='Unknown visualization output fields'):
        read_output_fields(options('u', 'all'))


def test_legacy_wind_validates_field_names_only_with_visualization(tmp_path):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '14 15 20'
    tree.find('simulationParameters/maxIterations').text = '12'
    tree.find('fileOptions/outputFields').text = 'tke'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    command = [sys.executable, '-m', 'plumeswift', 'qesWinds', '-s', '1', '-w', '-t', '-q', str(xml)]

    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=30)
    assert list(tmp_path.glob('*.nc')) == []
    selected = subprocess.run(command+['-o', 'selected'], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert selected.returncode != 0
    assert "Unknown visualization output fields ['tke']" in selected.stderr


def test_legacy_terrain_output_precedes_coefficient_loading(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    simulation = tree.find('simulationParameters')
    simulation.find('domain').text = '10 10 15'
    simulation.find('readCoefficientsFlag').text = '1'
    ET.SubElement(simulation, 'DEM').text = 'terrain.tif'
    ET.SubElement(simulation, 'COEFF').text = 'missing_coefficients.nc'
    with rasterio.open(tmp_path/'terrain.tif', 'w', driver='GTiff', width=10, height=10,
                       count=1, dtype='float32', transform=rasterio.transform.from_origin(0, 10, 1, 1)) as dem:
        dem.write(np.zeros((10, 10), dtype='float32'), 1)
    xml = tmp_path/'winds.xml'
    tree.write(xml)

    command = [sys.executable, '-m', 'plumeswift', 'qesWinds', '-s', '1', '-m', '-w', '-q', str(xml), '-o', 'output']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)

    assert result.returncode != 0
    assert 'missing_coefficients.nc' in result.stderr
    assert (tmp_path/'output_terrainOut.obj').is_file()
    assert not list(tmp_path.glob('output_*.nc'))


@pytest.mark.parametrize('skip', [False, True])
def test_wind_cli_visualization_is_separate_from_replay_workspace(tmp_path, skip):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/domain').text = '12 13 14'
    options = tree.find('fileOptions')
    options.find('outputFields').text = 'mag'
    ET.SubElement(options, 'outputFields').text = 'icellInitial'
    ET.SubElement(options, 'sensorVelocityFlag').text = 'true'
    xml = tmp_path/'winds.xml'
    tree.write(xml)
    command = [sys.executable, '-m', 'plumeswift', 'wind', str(xml), '--no-turbulence', '--output', str(tmp_path)]
    subprocess.run(command+(['--no-visualization'] if skip else []), check=True, capture_output=True, timeout=90)
    workspace = read_wind_workspace(tmp_path/'winds.nc')
    out = tmp_path/'windsOut.nc'
    if skip:
        assert not out.exists()
        return

    with _open_netcdf(out) as nc:
        assert set(nc.variables) == {'x', 'y', 'z', 't', 'timestamp', 'mag', 'icellInitial'}
        assert nc['mag'].dtype == np.dtype('float32')
        assert nc['mag'].dimensions == ('t', 'z', 'y', 'x')
        np.testing.assert_array_equal(nc['z'][:], workspace.coordinates['z'][1:])
        assert nc['mag'].shape == (1, 14, 13, 12)
        np.testing.assert_array_equal(nc['icellInitial'][0], workspace.arrays['icellflag_initial'][1:])
        u = workspace.arrays['u'].astype(np.float32)
        expected = .5*(u[1:-1, :-1, :-1]+u[1:-1, :-1, 1:])
        np.testing.assert_allclose(nc['mag'][0], expected, atol=1e-7, rtol=0)
