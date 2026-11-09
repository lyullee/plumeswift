import pytest

from plumeswift.data_transport import QESDataTransport
from plumeswift.timing import QEStime


def test_qes_clock_arithmetic_and_sensor_intervals():
    start = QEStime('2020-01-01T00:00')
    current = QEStime(start)
    current.increment(10 * 3600 + 12.5)
    current.increment(20 * 3600 + 10 * 60 + 12.5)
    assert current.get_timestamp() == '2020-01-02T06:10:25'
    assert (current + 86400).get_timestamp() == '2020-01-03T06:10:25'

    current = QEStime(start)
    current += 86400
    assert current - start == 86400
    current.set_timestamp('2020-01-04T00:00:00')
    assert current.get_timestamp() == '2020-01-04T00:00:00'

    current = QEStime(start)
    steps = 0
    for interval in range(1, 13):
        next_update = start + interval * 300
        while current < next_update:
            current += 10
            steps += 1
        assert current == next_update
    assert steps == 360
    assert current.get_timestamp() == '2020-01-01T01:00:00'


def test_qes_clock_normalizes_extended_time_fields():
    assert str(QEStime('2020-12-31T24:00:00')) == '2021-01-01T00:00:00'
    assert str(QEStime('2020-01-01T12:60:00')) == '2020-01-01T13:00:00'
    assert str(QEStime('2020-01-01T23:59:60.25')) == '2020-01-02T00:00:00.250000'
    assert str(QEStime('2020-12-31T24:00')) == '2021-01-01T00:00:00'
    assert str(QEStime('2020-1-2T8:30')) == '2020-01-02T08:30:00'
    assert str(QEStime('2020-1-2T8')) == '2020-01-02T08:00:00'


def test_qes_data_transport_values_and_pointer_entries():
    data = QESDataTransport()
    position = [12., 13., 14.]
    velocity = [0.1, 0.2, 0.3]
    integers = list(range(1000))
    data.put('mass', 0.0213)
    data.put('pos', position)
    data.put('vel', velocity, reference=True)
    data.put('lotsOfInts', integers)

    assert data.get('mass', float) == 0.0213
    assert data.get('pos', list) == position
    assert data.get('pos', list) is not position
    assert data.get('vel', list) is velocity
    assert data.get('lotsOfInts', list) == integers
    assert all(data.contains(key) for key in ('mass', 'pos', 'vel', 'lotsOfInts'))
    assert not data.contains('missing')
    with pytest.raises(RuntimeError, match='Key not found'):
        data.get('missing', float)
    with pytest.raises(RuntimeError, match='Type mismatch'):
        data.get('lotsOfInts', float)
