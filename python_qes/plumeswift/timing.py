"""QES UTC timestamp arithmetic and elapsed-time utilities."""
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from numbers import Real
from operator import index
import re
import sys
from time import perf_counter_ns

import numpy as np


_ISO_EXTENDED = re.compile(
    r'(\d{4})-(\d{1,2})-(\d{1,2})T(\d{1,2})(?::(\d{1,2})(?::(\d{1,2})(?:[.,](\d+))?)?)?',
    re.ASCII,
)
_EPOCH = datetime(1970, 1, 1)


def parse_timestamp(value: str) -> datetime:
    """Read a timezone-free QES ISO extended timestamp, including fractions."""
    match = _ISO_EXTENDED.fullmatch(value)
    if match is not None:
        year, month, day, hour = map(int, match.groups()[:4])
        minute, second = (int(part or 0) for part in match.groups()[4:6])
        fraction = match[7] or ''
        # Boost builds a time duration relative to midnight, so a 24-hour
        # timestamp or a 60-second leap marker rolls into the next day.
        return datetime(year, month, day)+timedelta(
            hours=hour, minutes=minute, seconds=second, microseconds=int(fraction[:6].ljust(6, '0')))
    # Preserve ISO formats accepted by the original Python reader.
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is not None:
        raise ValueError('Sensor timestamps require timezone-free UTC')
    return timestamp


def elapsed_milliseconds(current: datetime, start: datetime) -> int:
    """Match Boost total_milliseconds, truncating either sign toward zero."""
    elapsed = current-start
    micros = (elapsed.days*86_400+elapsed.seconds)*1_000_000+elapsed.microseconds
    return abs(micros)//1000 if micros >= 0 else -(abs(micros)//1000)


def milliseconds(seconds):
    scaled = 1000*float(np.float32(seconds))
    return int(np.copysign(np.floor(abs(scaled)+.5), scaled))


def epoch_seconds(timestamp: datetime) -> int:
    """Return native UTC epoch seconds, truncating a fraction toward zero."""
    elapsed = elapsed_milliseconds(timestamp, _EPOCH)
    return elapsed//1000 if elapsed >= 0 else -(-elapsed//1000)


def timestamp_remainder(timestamp: datetime, period_s: float) -> float:
    """Return the native millisecond remainder from the UTC epoch."""
    elapsed = elapsed_milliseconds(timestamp, _EPOCH)
    period = abs(int(period_s*1000))
    remainder = abs(elapsed) % period
    return (remainder if elapsed >= 0 else -remainder)/1000


@total_ordering
class QEStime:
    """Mutable UTC clock with QES float32 steps and millisecond differences."""

    def __init__(self, value=None):
        if value is None:
            self.set_time_to_now()
        else:
            self.set_timestamp(value)

    @property
    def datetime(self):
        return self._time

    def set_time_to_now(self):
        self._time = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    def set_timestamp(self, value):
        if isinstance(value, QEStime):
            value = value.datetime
        elif isinstance(value, str):
            value = parse_timestamp(value)
        elif not isinstance(value, datetime):
            value = _EPOCH+timedelta(seconds=index(value))
        if value.tzinfo is not None:
            raise ValueError('QEStime requires timezone-free UTC')
        self._time = value

    def get_timestamp(self):
        return self._time.isoformat()

    def get_epoch_time(self):
        return epoch_seconds(self._time)

    def increment(self, seconds):
        self._time += timedelta(milliseconds=milliseconds(seconds))

    def __str__(self):
        return self.get_timestamp()

    def __add__(self, seconds):
        if not isinstance(seconds, Real):
            return NotImplemented
        return QEStime(self._time+timedelta(milliseconds=milliseconds(seconds)))

    def __iadd__(self, seconds):
        if not isinstance(seconds, Real):
            return NotImplemented
        self.increment(seconds)
        return self

    def __sub__(self, other):
        if not isinstance(other, QEStime):
            return NotImplemented
        return elapsed_milliseconds(self._time, other.datetime)/1000

    def __eq__(self, other):
        if isinstance(other, str):
            return self._time == parse_timestamp(other)
        if not isinstance(other, QEStime):
            return NotImplemented
        return self._time == other.datetime

    def __lt__(self, other):
        if not isinstance(other, QEStime):
            return NotImplemented
        return self._time < other.datetime

    def __mod__(self, period_s):
        if not isinstance(period_s, Real):
            return NotImplemented
        return timestamp_remainder(self._time, period_s)


class NamedTimers:
    """Independent named wall-clock timers, as in calcTime."""

    def __init__(self, *, clock=perf_counter_ns):
        self._clock = clock
        self._starts = {}

    def reset(self):
        self._starts.clear()

    def start_new_timer(self, name):
        if name in self._starts:
            raise ValueError(f'Timer already exists: {name}')
        self._starts[name] = self._clock()

    def reset_stored_timer(self, name):
        if name not in self._starts:
            raise KeyError(name)
        self._starts[name] = self._clock()

    def elapsed(self, name):
        return (self._clock()-self._starts[name])/1_000_000_000

    def print_stored_time(self, name, *, stream=None):
        total_ms = (self._clock()-self._starts[name])//1_000_000
        total_seconds, ms = divmod(total_ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        parts = [f'{count} {unit}' for count, unit in zip((days, hours, minutes), ('days', 'hours', 'minutes')) if count]
        parts.extend((f'{seconds} seconds', f'{ms} milliseconds'))
        print(f'  Elapsed time for "{name}" timer: '+', '.join(parts), file=sys.stdout if stream is None else stream)


class Timer:
    """Accumulating start/stop timer, as in TimerTool."""

    def __init__(self, name, *, clock=perf_counter_ns):
        self.name = name
        self._clock = clock
        self._elapsed_ns = 0
        self.start()

    def start(self):
        self._started = self._clock()

    def stop(self):
        self._elapsed_ns += self._clock()-self._started

    @property
    def elapsed(self):
        return self._elapsed_ns/1_000_000_000

    def show(self, *, stream=None):
        print(f'Elapsed time for {self.name:<25}{self.elapsed:.6g} s', file=sys.stdout if stream is None else stream)
