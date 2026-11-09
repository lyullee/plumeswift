"""Normalize historical XML names used in the distributed QES wind examples."""
from copy import deepcopy
import xml.etree.ElementTree as ET

from .qes_xml import parse_optional_number


_BUILDING_FLAGS = ('rooftopFlag', 'upwindCavityFlag', 'streetCanyonFlag', 'streetIntersectionFlag',
                   'wakeFlag', 'highRiseFlag', 'sidewallFlag')
_MIXING_FIELDS = ('method', 'samples', 'save', 'LMfile', 'varname')
TIME_SERIES_FIELDS = frozenset(('timeStamp', 'timeEpoch', 'boundaryLayerFlag', 'siteZ0', 'reciprocal',
                                'height', 'speed', 'direction', 'canopyHeight', 'attenuationCoefficient', 'siteP'))


def normalize_zero_wrf_startup(root):
    """Load a startup snapshot for legacy coupled WRF runs that skip the wind loop."""
    sim = root.find('simulationParameters')
    if sim is None or not sim.findtext('WRF', '').strip() or sim.findtext('DEM', '').strip():
        return False
    steps = sim.find('totalTimeIncrements')
    if steps is None or int(steps.text or '') != 0:
        return False
    steps.text = '1'
    return True


def normalize_legacy_turbulence(params, *, cutcell):
    """Apply the native parser and wall-constructor fallback rules."""
    actions = []
    method = int(params.findtext('method', '0'))
    save = params.findtext('save', 'false').strip().lower() in ('true', '1')
    filename = params.findtext('LMfile', '').strip()
    if method == 3 and params.find('rayBackend') is None:
        actions.append('turbParams/method 3 -> 1 (native build without OptiX)')
        method = 1
    if method not in (0, 1, 2, 3, 4):
        actions.append(f'turbParams/method {method} -> 0 (unknown method)')
        method = 0
    if (method == 4 or save) and not filename:
        actions.append('turbParams/method -> 0 (no mixing length file)')
        method = 0
    method_node = params.find('method')
    if method_node is not None:
        method_node.text = str(method)
    if save and method in (0, 4):
        params.find('save').text = 'false'
        actions.append('turbParams/save -> false (inactive for the selected method)')
    if not cutcell:
        for name in ('buildingWallFlag', 'terrainWallFlag'):
            node = params.find(name)
            if node is None:
                continue
            mode = parse_optional_number(params, name, 2, convert=int, legacy=True)
            if mode not in (0, 1, 2):
                actions.append(f'turbParams/{name} {mode} -> 0 (no velocity wall derivative)')
                node.text = '0'
    return tuple(actions)


def normalize_sensor_xml(sensor):
    """Wrap historical sensor-level profile fields in their single time record."""
    profile = [child for child in sensor if child.tag in TIME_SERIES_FIELDS]
    if not profile or sensor.find('timeSeries') is not None:
        return ()
    series = ET.SubElement(sensor, 'timeSeries')
    for field in profile:
        sensor.remove(field)
        series.append(field)
    return ('sensor profile fields -> sensor/timeSeries',)


def normalize_wind_xml(root):
    """Return a canonical tree and an ordered list of compatibility actions."""
    root = deepcopy(root)
    actions = []

    def drop(parent, name):
        for child in parent.findall(name):
            parent.remove(child)
            actions.append(f'ignored {parent.tag}/{name}')

    def move(parent, name, target, replacement=None):
        replacement = replacement or name
        for child in parent.findall(name):
            parent.remove(child)
            source = f'{parent.tag}/{name}'
            destination = f'{target.tag}/{replacement}'
            if target.find(replacement) is None:
                child.tag = replacement
                target.append(child)
                actions.append(f'{source} -> {destination}')
            else:
                actions.append(f'ignored {source} in favor of {destination}')

    old_buildings = root.find('buildings')
    buildings = root.find('buildingsParams')
    if old_buildings is not None and buildings is None:
        old_buildings.tag = 'buildingsParams'
        buildings = old_buildings
        actions.append('buildings -> buildingsParams')

    simulation = root.find('simulationParameters')
    if simulation is not None:
        for old, current in (('UTMX', 'UTMx'), ('UTMY', 'UTMy')):
            move(simulation, old, simulation, current)
        for key in ('UTCConversion', 'Epoch', 'residualReduction', 'useDiffusion', 'heightFactor'):
            drop(simulation, key)
        for key in _BUILDING_FLAGS:
            if buildings is None:
                drop(simulation, key)
            else:
                move(simulation, key, buildings)

    legacy_mixing = root.find('localMixingParam')
    if legacy_mixing is not None:
        turbulence = root.find('turbParams')
        if turbulence is None:
            turbulence = ET.SubElement(root, 'turbParams')
        for key in _MIXING_FIELDS:
            move(legacy_mixing, key, turbulence)
        # Leave unknown children visible to the strict XML audit.
        if not len(legacy_mixing):
            root.remove(legacy_mixing)

    vegetation = root.find('vegetationParams')
    if vegetation is not None:
        drop(vegetation, 'num_canopies')
        for node in vegetation.findall('CanopyHomogeneous'):
            node.tag = 'Homogeneous'
            actions.append('vegetationParams/CanopyHomogeneous -> vegetationParams/Homogeneous')

    options = root.find('fileOptions')
    if options is not None:
        drop(options, 'outputTURBInputFile')
    for sensor in root.findall('metParams/sensor'):
        actions.extend(normalize_sensor_xml(sensor))
    return root, tuple(actions)
