from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.io import netcdf_series, read_turbulence, read_wind_workspace
from plumeswift.wind_case import read_wind_case

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name,count', [('all', 5), ('limited', 2), ('oversized', 5),
                                        ('building', 5), ('external', 5)])
def test_native_multitime_wind(name, count):
    folder = ROOT/'results/python_port/wind_series'/name
    original_wind = folder/'native_windsWk.nc'
    if not original_wind.exists():
        pytest.skip('Run scripts/check_python_wind_series.py --native for native fixture')
    case = read_wind_case(folder/'winds.xml')
    steps = tuple(case.iter_steps())
    assert len(steps) == count
    assert [step.timestamp[-8:] for step in steps] == ['00:00:00', '00:00:30', '00:00:30', '00:01:00', '00:02:10'][:count]
    assert [len(step.stations) for step in steps] == [1, 1, 1, 2, 1][:count]
    assert steps[1].profile.speed == 2.5
    assert all(step.reference_roughness == .025 for step in steps)
    for index, step in enumerate(steps):
        result = step.run()
        reference = read_wind_workspace(original_wind, step=index).arrays
        for key in ('icellflag', 'e', 'f', 'g', 'h', 'm', 'n'):
            np.testing.assert_array_equal(result['workspace'].arrays[key], reference[key])
        for key in ('u0', 'v0', 'w0', 'u', 'v', 'w'):
            np.testing.assert_allclose(result['workspace'].arrays[key], reference[key], atol=2e-6, rtol=0)
        ref_turb = read_turbulence(folder/'native_turbOut.nc', step=index)
        for key in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz', 'CoEps', 'div_tau_x', 'div_tau_y', 'div_tau_z'):
            np.testing.assert_allclose(result['turbulence'][key], ref_turb[key], atol=5e-5, rtol=0)


@pytest.mark.parametrize('limit,count', [(-1, 3), (-(2**32-2), 2)])
def test_legacy_non_wrf_timestep_limit_is_unsigned(tmp_path, limit, count):
    tree = ET.parse(ROOT/'data/cases/empty_west/winds.xml')
    tree.find('simulationParameters/totalTimeIncrements').text = str(limit)
    station = tree.find('metParams/sensor')
    first = station.find('timeSeries')
    for minute in (1, 2):
        record = ET.fromstring(ET.tostring(first))
        record.find('timeStamp').text = f'2020-01-01T01:0{minute}:00'
        station.append(record)
    path = tmp_path/'wind.xml'
    tree.write(path)

    legacy = read_wind_case(path, legacy_turbulence_xml=True)

    assert len(tuple(legacy.iter_steps())) == count
    with pytest.raises(ValueError, match='totalTimeIncrements must be nonnegative'):
        read_wind_case(path)


def test_streaming_series_rejects_bad_record_without_extending_time_axis(tmp_path):
    from netCDF4 import Dataset

    ws = Domain((4, 4, 4), (1., 1., 1.)).workspace()
    path = tmp_path/'winds.nc'
    arrays = {'u0': ws.arrays['u0'], 'terrain': ws.arrays['terrain']}
    with netcdf_series(path, ws.coordinates) as append:
        append(arrays, '2020-01-01T00:00:00', 0)
        with pytest.raises(ValueError, match='Static NetCDF field changed'):
            append({**arrays, 'terrain': arrays['terrain']+1}, '2020-01-01T00:00:01', 1)
        assert not path.exists()
    with Dataset(path) as output:
        assert len(output.dimensions['t']) == 1
        assert output['terrain'].dimensions == ('y', 'x')
