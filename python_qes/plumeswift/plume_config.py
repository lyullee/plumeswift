"""Common settings from the QES plume XML contract (GPL-3)."""
from dataclasses import dataclass
import warnings

import numpy as np

from .contracts import UnportedFeature

from .timing import milliseconds


@dataclass(frozen=True)
class PlumeConfig:
    duration_s: float
    step_s: float
    courant: float
    stress_tolerance: float
    interpolation: str
    boundaries: tuple[str, str, str]
    wall: str
    collection_start_ms: int
    collection_period_ms: int
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    boxes: tuple[int, int, int]
    seed: int
    progress_period_ms: int | None = None
    allow_zero_collection_volume: bool = False

    @property
    def coordinates(self):
        return {axis: lower+(np.arange(count)+.5)*((upper-lower)/count)
                for axis, lower, upper, count in zip('xyz', self.lower, self.upper, self.boxes)}


def parse_plume_config(root, *, seed=0, legacy_plume_xml=False):
    if root.tag != 'QESPlumeParameters':
        raise ValueError('Expected QESPlumeParameters root')
    def section(name, names):
        node = root.find(name)
        if node is None and name == 'plumeParameters':
            node = root.find('simulationParameters')
        if node is None:
            raise ValueError(f'Missing {name}')
        if not legacy_plume_xml:
            extra = {child.tag for child in node}-names
            if extra:
                raise UnportedFeature(f'Unsupported {name} fields {sorted(extra)}')
        return node

    def read(node, key, cast=float):
        value = node.findtext(key)
        if value is None:
            raise ValueError(f'Missing plume XML field {key}')
        if legacy_plume_xml and cast is float:
            return float(np.float32(value))
        return cast(value)

    run = section('plumeParameters', {'timeStep', 'simDur', 'CourantNumber', 'invarianceTol', 'interpolationMethod',
                                      'updateFrequency_particleLoop', 'updateFrequency_timeLoop', 'C_0'})
    boundary = section('boundaryConditions', {'xBCtype', 'yBCtype', 'zBCtype', 'wallReflection', 'doDepositions'})
    box_fields = {f'boxBounds{axis}{edge}' for axis in 'XYZ' for edge in (1, 2)}
    collection = section('collectionParameters', box_fields | {'timeAvgStart', 'timeAvgFreq', 'nBoxesX', 'nBoxesY', 'nBoxesZ'})
    # The current QES boundary parser ignores the legacy doDepositions field.
    step, duration, courant, tolerance = (read(run, key) for key in ('timeStep', 'simDur', 'CourantNumber', 'invarianceTol'))
    start, period = (read(collection, key) for key in ('timeAvgStart', 'timeAvgFreq'))
    lower, upper = (tuple(read(collection, f'boxBounds{axis}{edge}') for axis in 'XYZ') for edge in (1, 2))
    boxes = tuple(read(collection, 'nBoxes'+axis, int) for axis in 'XYZ')
    invalid_extent = any(lo < 0 or hi < lo or (hi == lo and not legacy_plume_xml) for lo, hi in zip(lower, upper))
    if (not np.isfinite([step, duration, courant, tolerance, start, period, *lower, *upper]).all() or
            not 0 < step <= duration or not 0 <= courant <= 1 or tolerance <= 0 or start < 0 or period <= 0 or
            milliseconds(step) < 1 or min(boxes) < 1 or invalid_extent):
        raise ValueError('Invalid plume time, turbulence or collection settings')
    period_ms = milliseconds(period)
    if np.float32(period) <= 0 or (period_ms < 1 and not legacy_plume_xml):
        raise ValueError('Invalid plume collection period')
    modes = tuple(read(boundary, axis+'BCtype', str).strip() for axis in 'xyz')
    if any(mode not in ('exiting', 'periodic', 'reflection') for mode in modes):
        raise ValueError('Invalid plume boundary mode')
    wall = boundary.findtext('wallReflection', 'doNothing').strip() or 'doNothing'
    interpolation = run.findtext('interpolationMethod', 'triLinear').strip()
    if wall not in ('doNothing', 'stairstepReflection', 'setInactive', 'meshReflection'):
        raise ValueError('Invalid plume wall policy')
    if interpolation not in ('triLinear', 'nearestCell', 'analyticalPowerLaw'):
        raise ValueError('Invalid plume interpolation method')
    particle_frequency = int(run.findtext('updateFrequency_particleLoop', '1'))
    if particle_frequency < 1:
        raise ValueError('updateFrequency_particleLoop must be at least one')
    progress = run.findtext('updateFrequency_timeLoop')
    progress_ms = None
    if progress is not None:
        seconds = read(run, 'updateFrequency_timeLoop')
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('updateFrequency_timeLoop must be finite and nonnegative')
        steps = np.ceil(np.float32(duration)/np.float32(step))
        if seconds > steps:
            warnings.warn('(SimulationParameters::checkParsedValues): input updateFrequency_timeLoop '
                          'must be smaller than or equal to calculated nTimes! '
                          f'updateFrequency_timeLoop = "{seconds:g}", nTimes = "{steps:g}"',
                          RuntimeWarning, stacklevel=2)
        progress_ms = milliseconds(seconds)
    return PlumeConfig(duration, step, courant, tolerance, interpolation, modes, wall,
                       milliseconds(start), period_ms, lower, upper, boxes, seed, progress_ms,
                       allow_zero_collection_volume=legacy_plume_xml)
