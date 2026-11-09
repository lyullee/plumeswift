"""Native QES XML source builders and fixed-step release semantics (GPL-3)."""
from dataclasses import dataclass
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import UnportedFeature
from .sources import Cube, Geometry, Line, Point, Source, SphereShell, StepRelease
from .timing import milliseconds


def _float(node, key, default=None, *, invalid_default=False):
    raw = node.findtext(key)
    if raw is None and default is None:
        raise ValueError(f'Missing {node.tag}/{key}')
    try:
        if invalid_default and raw is not None and '_' in raw:
            raise ValueError('Digit separator is not accepted by Boost float conversion')
        with np.errstate(over='ignore'):
            value = np.float32(default if raw is None else raw)
        if not np.isfinite(value):
            raise ValueError(f'Non-finite {node.tag}/{key}')
    except (ValueError, OverflowError):
        if invalid_default and default is not None:
            return np.float32(default)
        raise
    return value


def _check_fields(node, fields, *, allow_extra=False):
    if allow_extra:
        return
    extra = {child.tag for child in node}-set(fields)
    if extra:
        raise UnportedFeature(f'Unsupported {node.tag} fields {sorted(extra)}')


def _release(node, step, *, allow_extra=False):
    def optional(key, default=0):
        # Boost get_optional<float> leaves the defaults on failed conversion.
        return _float(node, key, default, invalid_default=allow_extra)

    kind = node.tag.removeprefix('releaseType_')
    if kind == 'instantaneous':
        _check_fields(node, ('releaseTime', 'numPar', 'totalMass'), allow_extra=allow_extra)
        count = int(node.findtext('numPar'))
        if count <= 0:
            raise ValueError('Instantaneous release requires a positive particle count')
        start = optional('releaseTime')
        # Native uses the requested time as both the start offset and
        # the offset from that start. At zero the window is one step.
        end_offset = start
        mass_g = optional('totalMass')/np.float32(count)
    else:
        count = int(node.findtext('particlePerTimestep'))
        if count < 0:
            raise ValueError('Particles per timestep must be nonnegative')
        rate = optional('massPerSec')
        if kind == 'continuous':
            # The pinned release parser leaves these historical nested particle fields unused.
            _check_fields(node, ('releaseStartTime', 'particlePerTimestep', 'massPerSec', 'massPerTimestep',
                                 'particleDensity', 'particleDiameter', 'decayConst'), allow_extra=allow_extra)
            start = optional('releaseStartTime', -1)
            if start == -1:
                start = 0
            end_offset = 1e10
            fixed = optional('massPerTimestep')
            if rate != 0 and fixed != 0:
                raise ValueError('Continuous release accepts only one mass input')
            mass_g = fixed if fixed != 0 else rate*step
        else:
            _check_fields(node, ('releaseStartTime', 'releaseEndTime', 'releaseDuration',
                                 'particlePerTimestep', 'totalMass', 'massPerSec'), allow_extra=allow_extra)
            start = _float(node, 'releaseStartTime')
            end = optional('releaseEndTime', -1)
            duration = optional('releaseDuration')
            if (end == -1) == (duration == 0):
                raise ValueError('Duration release requires releaseEndTime or releaseDuration')
            if end == -1:
                end = start+duration
            else:
                duration = end-start
            if duration <= 0:
                raise ValueError('Release duration must be positive')
            total = optional('totalMass')
            if rate != 0 and total != 0:
                raise ValueError('Duration release accepts only one mass input')
            # In the pinned native builder, the end offset is added to the
            # start and totalMass is divided by particlePerTimestep here.
            end_offset = end
            mass_g = total/np.float32(max(count, 1)) if total != 0 else rate*step
    start_ms = milliseconds(start)
    end_ms = start_ms+milliseconds(end_offset)
    if end_ms < start_ms:
        return StepRelease(0, 0., start_ms/1000, start_ms/1000)
    # SetMass stores only positive masses and divides in float32 grams before
    # writing ParticleCore. Convert the resulting effective step mass to SI.
    mass_kg = float(mass_g/np.float32(count))*count/1000 if count and mass_g > 0 else 0.
    return StepRelease(count, mass_kg, start_ms/1000, end_ms/1000)


def _geometry(node, *, allow_extra=False):
    kind = node.tag.removeprefix('sourceGeometry_')
    if kind == 'FullDomain':
        _check_fields(node, (), allow_extra=allow_extra)
        return None
    prefixes = ('min', 'max') if kind == 'Cube' else ('pos',)
    suffixes = ('_0', '_1') if kind == 'Line' else ('',)
    axes = [tuple(prefix+axis+suffix for axis in 'XYZ') for prefix in prefixes for suffix in suffixes]
    fields = [key for axis in axes for key in axis]
    _check_fields(node, fields+(['radius'] if kind == 'SphereShell' else []), allow_extra=allow_extra)
    vectors = [tuple(float(_float(node, key)) for key in axis) for axis in axes]
    # QES leaves the optional source dimension checks disabled in these XML builders.
    if kind == 'SphereShell':
        return SphereShell(vectors[0], float(_float(node, 'radius')), validate_dimensions=False)
    if kind == 'Cube':
        return Cube(*vectors, validate_dimensions=False)
    return {'Point': Point, 'Line': Line}[kind](*vectors)


@dataclass(frozen=True)
class SourceSpec:
    source_id: int
    geometry: Geometry | None
    release: StepRelease
    diameter_um: float
    density_kg_m3: float
    geometry_seed_index: int | None = None

    def create(self, boundaries, *, source_seed=None, native_full_domain_bounds=False):
        geometry = self.geometry
        if geometry is None:
            x, y, z = boundaries.axes
            if native_full_domain_bounds:
                # PI_SourceGeometry_FullDomain assigns the interleaved
                # getter values to three minima followed by three maxima.
                geometry = Cube((x.lower, x.upper, y.lower), (y.upper, z.lower, z.upper),
                                validate_dimensions=False)
            else:
                # A thin native grid can leave one axis with coincident bounds.
                geometry = Cube((x.lower, y.lower, z.lower), (x.upper, y.upper, z.upper),
                                validate_dimensions=False)
        if source_seed is not None and not isinstance(geometry, Point):
            from .native_source_geometry import NativeSourceGeometry
            seed_index = self.source_id if self.geometry_seed_index is None else self.geometry_seed_index
            geometry = NativeSourceGeometry(geometry, source_seed+seed_index)
        return Source(geometry, self.release, source_id=self.source_id,
                      diameter_um=self.diameter_um, density_kg_m3=self.density_kg_m3)


@dataclass(frozen=True)
class ModelSpec:
    sources: tuple[SourceSpec, ...]
    deposition: bool


def read_models(root, step, *, legacy_plume_xml=False, allow_repeated_tags=False):
    section = root.find('particleParameters')
    # Native reads recognized tags only. The historical adapter has a separate schema.
    allow_extra = legacy_plume_xml and section is not None
    if section is None:
        if legacy_plume_xml:
            return {}
        section = _legacy_base_model(root)
        if section is None:
            return {}
    _check_fields(section, ('baseParticle',), allow_extra=allow_extra)
    models = {}
    source_id = 0
    geometry_seed_index = 0
    release_tags = {'releaseType_'+kind for kind in ('continuous', 'duration', 'instantaneous')}
    geometry_tags = {'sourceGeometry_'+kind for kind in ('Point', 'Line', 'Cube', 'SphereShell', 'FullDomain')}
    for node in section.findall('baseParticle'):
        # The pinned base model stores these controls but does not use them
        # for deposition, decay, or settling in its active advection loop.
        _check_fields(node, ('tag', 'depositionFlag', 'c1', 'c2', 'decayConst', 'source'), allow_extra=allow_extra)
        raw_tag = node.findtext('tag')
        tag = raw_tag if raw_tag is not None else ''
        # Boost keeps literal printable model text, which is also part of
        # the native concentration and particle filenames.
        if not legacy_plume_xml or not tag.isprintable():
            tag = tag.strip()
        duplicate = tag in models and not (legacy_plume_xml and allow_repeated_tags)
        explicit_empty_tag = legacy_plume_xml and raw_tag == ''
        if (not tag and not explicit_empty_tag) or duplicate or any(char in tag for char in '/\\<>:"|?*'):
            raise ValueError(f'Missing, duplicate or unsafe model tag {tag!r}')
        sources = []
        for source in node.findall('source'):
            _check_fields(source, release_tags | geometry_tags | {'particleDiameter', 'particleDensity'},
                          allow_extra=allow_extra)
            releases = [child for child in source if child.tag in release_tags]
            geometries = [child for child in source if child.tag in geometry_tags]
            if len(releases) != 1 or len(geometries) != 1:
                raise ValueError('Each QES source requires exactly one release and one geometry')
            geometry = _geometry(geometries[0], allow_extra=allow_extra)
            release = _release(releases[0], np.float32(step), allow_extra=allow_extra)
            sources.append(SourceSpec(source_id, geometry, release,
                                      float(_float(source, 'particleDiameter', 0)),
                                      float(_float(source, 'particleDensity', 0)),
                                      geometry_seed_index))
            source_id += 1
            # Point has no engine. FullDomain (None) constructs a Cube.
            geometry_seed_index += not isinstance(geometry, Point)
        deposition = node.findtext('depositionFlag', 'false').strip()
        if allow_extra:
            # Boost accepts boolalpha true or a decimal 1 and keeps the
            # original false default when optional conversion fails.
            deposits = deposition == 'true' or deposition.removeprefix('+').lstrip('0') == '1'
        else:
            deposition = deposition.lower()
            if deposition not in ('true', 'false', '0', '1'):
                raise ValueError(f'Invalid depositionFlag for model {tag!r}')
            deposits = deposition in ('true', '1')
        models[tag] = ModelSpec(tuple(sources), deposits)
    return models


def _legacy_base_model(root):
    """Translate the historical standalone tracer source layout into one base model."""
    source_parameters = root.find('sourceParameters')
    old_sources = root.find('sources')
    if source_parameters is not None and old_sources is not None:
        raise ValueError('Select one legacy plume source section')
    if source_parameters is None and old_sources is None:
        return None
    if source_parameters is not None:
        _check_fields(source_parameters, ('source',))
        sources = [deepcopy(source) for source in source_parameters.findall('source')]
    else:
        _check_fields(old_sources, ('numSources', 'SourcePoint'))
        sources = []
        for point in old_sources.findall('SourcePoint'):
            _check_fields(point, ('posX', 'posY', 'posZ', 'ReleaseType_continuous',
                                  'ReleaseType_duration', 'ReleaseType_instantaneous'))
            source = ET.Element('source')
            geometry = ET.SubElement(source, 'sourceGeometry_Point')
            for child in point:
                node = deepcopy(child)
                if node.tag in ('posX', 'posY', 'posZ'):
                    geometry.append(node)
                else:
                    node.tag = 'r'+node.tag[1:]
                    source.append(node)
            sources.append(source)
    if not sources:
        return None
    section = ET.Element('particleParameters')
    model = ET.SubElement(section, 'baseParticle')
    ET.SubElement(model, 'tag').text = 'base'
    for source in sources:
        for tracer in source.findall('particle_Tracer'):
            _check_fields(tracer, ())
            source.remove(tracer)
        for release in source:
            for count in release.findall('parPerTimestep'):
                if release.find('particlePerTimestep') is not None:
                    raise ValueError('Duplicate particles-per-timestep setting')
                count.tag = 'particlePerTimestep'
        model.append(source)
    return section
