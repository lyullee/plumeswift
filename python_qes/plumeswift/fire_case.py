"""Read native QES-Fire fuel and ignition configuration (GPL-3)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fire_front import cpu_front
from .fire_fuels import read_fuel_raster
from .fire_grid import FireGrid, Ignition, read_timed_ignitions
from .qes_xml import read_qes_xml


@dataclass(frozen=True)
class FireConfig:
    duration: float
    fuel_type: int
    moisture: float
    cure: float
    courant: float
    ignitions: tuple[Ignition, ...]
    fuel_file: Path | None = None
    ignition_file: Path | None = None
    native_ignition_input: bool = False

    def initialize(self, workspace, spacing, *, initialize_front=cpu_front, native_flame_sampling=False,
                   native_fire_clock=False):
        terrain = workspace.arrays['terrain']
        codes = self.fuel_type
        if self.fuel_file is not None:
            codes = read_fuel_raster(self.fuel_file, terrain.shape, *spacing[:2])
        events = (read_timed_ignitions(self.ignition_file, native_input=self.native_ignition_input)
                  if self.ignition_file is not None else ())
        return FireGrid(terrain, spacing, codes, self.ignitions, moisture=self.moisture,
                        cure=self.cure, courant=self.courant, raster_fuel=self.fuel_file is not None,
                        timed_ignitions=events, initialize_front=initialize_front,
                        native_flame_sampling=native_flame_sampling, native_fire_clock=native_fire_clock,
                        native_ignition_input=self.native_ignition_input)


def read_fire_config(path, *, legacy_fire_xml=False):
    path = Path(path)
    root = read_qes_xml(path)
    fire = root if root.tag == 'fires' else root.find('fires')
    return _parse_fire_config(path, fire, legacy_fire_xml=legacy_fire_xml)


def _parse_fire_config(path, fire, *, legacy_fire_xml=False):
    from .input_paths import QESInputPaths

    paths = QESInputPaths.from_input(path)
    if fire is None:
        raise ValueError('Missing fires XML section')
    # Historical fieldFlag is present in examples but absent from Fires::parseValues.
    allowed = {'fireDur', 'numFires', 'fuelType', 'fmc', 'cure', 'courant', 'ignition', 'fuelMap', 'igTimes', 'fieldFlag'}
    if not legacy_fire_xml and any(node.tag not in allowed for node in fire):
        raise ValueError('Unknown fires XML field')

    def required(element, name, cast=float):
        text = element.findtext(name)
        if text is None:
            raise ValueError(f'Missing fire XML field {name}')
        return cast(text)

    def optional_path(name):
        value = fire.findtext(name, '').strip()
        return paths.resolve(value) if value else None

    ignitions = []
    ignition_fields = ('xStart', 'yStart', 'length', 'width', 'height', 'baseHeight')
    for item in fire.findall('ignition'):
        if not legacy_fire_xml and any(node.tag not in ignition_fields for node in item):
            raise ValueError('Unknown ignition XML field')
        ignitions.append(Ignition(*(required(item, key) for key in ignition_fields)))
    duration = required(fire, 'fireDur')
    if not np.isfinite(duration) or (duration < 0 and not legacy_fire_xml):
        raise ValueError('Invalid fire duration')
    return FireConfig(duration, required(fire, 'fuelType', int), required(fire, 'fmc'),
                      required(fire, 'cure'), required(fire, 'courant'), tuple(ignitions),
                      optional_path('fuelMap'), optional_path('igTimes'))
