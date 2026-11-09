"""QES particle snapshots with the native clock, columns and physical units."""
import csv
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import tempfile

import numpy as np

from .contracts import UnportedFeature
from .timing import milliseconds


@dataclass(frozen=True)
class ParticleOutputConfig:
    first_ms: int
    period_ms: int


def read_particle_output(root, *, legacy_plume_xml=False):
    node = root.find('particleOutputParameters')
    if node is None:
        raise ValueError('Particle CSV requires particleOutputParameters')
    extra = {child.tag for child in node}-{'outputFrequency', 'outputStartTime', 'outputEndTime', 'outputFields'}
    if extra and not legacy_plume_xml:
        raise UnportedFeature(f'Unsupported particle output fields {sorted(extra)}')
    cast = np.float32 if legacy_plume_xml else float
    raw_period = node.findtext('outputFrequency')
    if raw_period is None:
        raise ValueError('Missing particleOutputParameters/outputFrequency')
    try:
        period = cast(raw_period)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Invalid particleOutputParameters/outputFrequency') from exc
    raw_start = node.findtext('outputStartTime', '-1')
    if legacy_plume_xml:
        # Boost's optional float leaves the native default on conversion failure.
        try:
            start = cast(raw_start)
        except (ValueError, OverflowError):
            start = -1.
        if not np.isfinite(start):
            start = -1.
    else:
        start = cast(raw_start)
    if not np.isfinite([period, start]).all():
        raise ValueError('Particle CSV requires finite output timing')
    period_ms = milliseconds(period)
    if period_ms <= 0 and not legacy_plume_xml:
        raise ValueError('Particle CSV requires a positive millisecond output frequency')
    # A nonpositive native interval leaves the deadline due on every later step.
    # The upstream writer has fixed columns and does not apply outputEndTime.
    return ParticleOutputConfig(period_ms if start < 0 else milliseconds(start), period_ms)


_HEADER = (
    '#particleID,tStrt,sourceIdx,d,m,wdecay,xPos_init,yPos_init,zPos_init,xPos,yPos,zPos,'
    'uFluct,vFluct,wFluct,delta_uFluct,delta_vFluct,delta_wFluct\n'
)


class ParticleOutput:
    def __init__(self, config, output, origin, *, prefix='', native_format=False):
        self.period_ms = config.period_ms
        self.next_ms = config.first_ms
        self.output = Path(output)
        self.origin = origin
        self.prefix = prefix
        self.native_format = native_format
        self.records = 0

    def save(self, time_ms, models):
        if time_ms < self.next_ms:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        current = (self.origin+timedelta(milliseconds=time_ms)).isoformat()
        for tag, model in models.items():
            self._save_model(tag, current, model.simulation.batch)
        self.next_ms += self.period_ms
        self.records += 1

    def _save_model(self, tag, current, batch):
        path = self.output/f"{self.prefix}{tag}_particleData_{current.replace(':', '-')}.csv"
        stream = tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                             dir=path.parent, delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                stream.write(f'# Particle Data File\n# Model particle tag: {tag}\n')
                stream.write(f'# Simulation start time: {self.origin.isoformat()}\n')
                stream.write(f'# Simulation current time: {current}\n')
                stream.write(_HEADER)
                writer = csv.writer(stream, lineterminator='\n')
                for particle_id, mass, position, fluctuation, metadata in zip(
                        batch.identifiers, batch.mass, batch.positions, batch.fluctuations, batch.tracking):
                    release = self.origin+timedelta(seconds=float(metadata['release_s']))
                    identity = (particle_id, release.isoformat(), metadata['source_id'])
                    values = (metadata['diameter_um'], 1000*mass, 1, *metadata['initial_position'],
                              *position, *fluctuation, *metadata['delta_fluctuation'])
                    if self.native_format:
                        stream.write(', '.join((*map(str, identity),
                                                *(format(np.float32(value), '.6g') for value in values)))+'\n')
                    else:
                        writer.writerow((*identity, *values))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
