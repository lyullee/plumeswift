from importlib.resources import files
import json

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
