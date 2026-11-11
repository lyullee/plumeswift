from importlib.resources import files
import json
import math

import pytest

from plumeswift.ffi_report import parse_sensor_table


def test_ffi_table_is_parsed_from_local_text():
    text = 'Test Name Test02\nStart 1 sec\nEnd 10 sec\n' + '\n'.join(
        f'OC_{i:02} (R=9,01, Z=0.1, B=45) {i},1 34,2 -1,0 0,2 %vol'
        for i in range(1, 31))
    window, sensors = parse_sensor_table(text, 2)
    assert window == [1, 10]
    assert sensors[1] == dict(id='OC_02', radius_m=9.01, height_m=.1,
                              bearing_deg=45., observed_volpct=2.1)
    with pytest.raises(ValueError, match='30 distinct sensors'):
        parse_sensor_table(text.rsplit('\n', 1)[0], 2)


def test_packaged_ffi_cases_only_contain_model_inputs():
    resources = files('plumeswift').joinpath('example_data')
    assert not resources.joinpath('ffi_lh2.json').is_file()
    cases = json.loads(resources.joinpath('ffi_cases.json').read_text(encoding='utf-8'))
    assert [case['test'] for case in cases] == list(range(1, 8))
    for case in cases:
        assert not {'sensors', 'observed_window_s'} & case.keys()
        assert case['sim_s'] - case['spinup_s'] == case['window_s']


@pytest.mark.parametrize('joint, count', [(False, 5), (True, 9)])
def test_weather_members_preserve_mean_spread_and_fixed_nozzle(joint, count):
    from plumeswift.ffi_lh2 import WIND_SPREAD, _cryogenic_nozzle, _weather_members

    cases = json.loads(files('plumeswift').joinpath('example_data', 'ffi_cases.json').read_text())
    for case, (speed_sd, bearing_sd) in zip(cases, WIND_SPREAD, strict=True):
        members = _weather_members(case, joint=joint)
        assert len(members) == count
        assert all(weight > 0 and state['wind_speed'] > 0 for _, weight, state in members)
        assert math.fsum(weight for _, weight, _ in members) == pytest.approx(1.)
        offsets = [(weight, member['wind_speed'] - case['wind_speed'],
                    (member['wind_from'] - case['wind_from'] + 180) % 360 - 180)
                   for _, weight, member in members]
        assert math.fsum(w * u for w, u, b in offsets) == pytest.approx(0.)
        assert math.fsum(w * b for w, u, b in offsets) == pytest.approx(0.)
        assert math.fsum(w * u**2 for w, u, b in offsets) == pytest.approx(speed_sd**2)
        assert math.fsum(w * b**2 for w, u, b in offsets) == pytest.approx(bearing_sd**2)
        assert math.fsum(w * u * b for w, u, b in offsets) == pytest.approx(0.)
        if joint:
            assert math.fsum(w * u**2 * b**2 for w, u, b in offsets) == pytest.approx(speed_sd**2 * bearing_sd**2)
        assert all(_cryogenic_nozzle(member) == _cryogenic_nozzle(case) for _, _, member in members)


def test_weather_threshold_uses_weighted_sensor_average(tmp_path):
    from plumeswift.ffi_lh2 import _aggregate_weather

    a = [dict(sensor='OC_01', observed_volpct=4.5, predicted_volpct=5.)]
    b = [dict(sensor='OC_01', observed_volpct=4.5, predicted_volpct=1.)]
    rows, summary = _aggregate_weather({'test': 6}, [(a, {'weight': .5}), (b, {'weight': .5})], tmp_path)
    assert rows[0]['predicted_volpct'] == 3.
    assert summary['measured_at_least_4'] == 1
    assert summary['detected_at_least_4'] == 0


def test_measured_weather_averages_across_north_and_flags_missing_secondary(monkeypatch):
    from plumeswift import ffi_weather

    rows = [(i, 2., 359. if i % 2 else 1., -1., 90.) for i in range(5)]
    monkeypatch.setattr(ffi_weather, '_raw_wind', lambda path: iter(rows))
    first, second = ffi_weather.read_weather('unused.xlsx', 0, 4, 2)
    assert first[:2] == [0., 2.] and second[:2] == [2., 4.]
    assert first[2] == pytest.approx(2 * math.cos(math.radians(1)))
    assert (first[3] + 180) % 360 - 180 == pytest.approx(0.)
    assert first[4:] == [None, None]
    with pytest.raises(ValueError, match='insufficient 5 m wind coverage'):
        ffi_weather.read_weather('unused.xlsx', 0, 4, 2, 5)
