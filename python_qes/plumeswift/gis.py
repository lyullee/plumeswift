"""Projected vector polygons, ring nesting, and QES origin handling."""
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np
from .geometry import PolygonBuilding, polygon_parts
from .georeference import transform_points


def _split_archive(path):
    for index, part in enumerate(path.parts):
        if Path(part).suffix.lower() == '.zip':
            return Path(*path.parts[:index+1]), PurePosixPath(*path.parts[index+1:])
    return None


def resolve_vector_path(paths, filename):
    virtual_prefix = '/vsizip/'
    if filename.startswith(virtual_prefix):
        virtual = filename[len(virtual_prefix):]
        if virtual.startswith('{'):
            name, close, remainder = virtual[1:].partition('}')
            if not close or (remainder and not remainder.startswith('/')):
                raise ValueError('Invalid GIS ZIP archive path')
            archive = Path(name)
            member = PurePosixPath(remainder.removeprefix('/'))
        else:
            split = _split_archive(Path(virtual))
            if split is None:
                raise ValueError('GIS /vsizip/ input requires a ZIP archive')
            archive, member = split
        if archive.suffix.lower() != '.zip' or member.is_absolute() or '..' in member.parts:
            raise ValueError('Invalid GIS ZIP archive or member')
        # Native QES passes this absolute VSI path to GDAL as supplied.
        # GDAL resolves the relative archive portion from the working directory.
        return archive.resolve(strict=True).joinpath(*member.parts)
    path = Path(filename)
    archive = _split_archive(path)
    if archive is None:
        return paths.resolve(path)
    filename, member = archive
    return paths.resolve(filename).joinpath(*member.parts)


def _is_shapefile(path):
    return path.suffix.lower() in {'.shp', '.shx', '.dbf'} or _split_archive(path) is not None or (
        not path.exists() and path.with_suffix('.shp').is_file())


def default_building_layer(path):
    archive = _split_archive(path)
    if not _is_shapefile(path) or (archive is not None and not archive[1].parts):
        return 'buildings'
    return None


def _select_shapefile_layer(shapes, layer):
    if layer is None:
        return shapes
    exact = [shape for shape in shapes if shape.stem == layer]
    return exact or [shape for shape in shapes if shape.stem.casefold() == layer.casefold()]


def _polygon_parts(coordinates, source_crs, target_crs):
    rings = []
    for points in coordinates:
        ring = np.asarray(points, dtype=float)
        if ring.ndim != 2 or ring.shape[1] < 2:
            raise ValueError('GIS polygon has no paired horizontal coordinates')
        ring = ring[:, :2]
        if len(ring) < 3 or not np.isfinite(ring).all():
            raise ValueError('GIS rings require at least three finite vertices')
        if source_crs is not None and target_crs is not None and source_crs != target_crs:
            ring = transform_points(ring, source_crs, target_crs)
            if not np.isfinite(ring).all():
                raise ValueError('GIS rings require finite projected vertices')
        if not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack((ring, ring[0]))
        rings.append(ring)
    if not rings:
        raise ValueError('GIS polygon has no rings')
    return polygon_parts(rings)


@contextmanager
def _open_shapefile(path, layer, target_crs):
    import shapefile

    archive_path = _split_archive(path)
    if archive_path is not None:
        filename, member = archive_path
        with ZipFile(filename) as archive, ExitStack() as stack:
            members = {PurePosixPath(name) for name in archive.namelist() if not name.endswith('/')}
            shapes = [name for name in members if name.suffix.lower() == '.shp'
                      and (not member.parts or name.with_suffix('') == member.with_suffix(''))]
            layers = _select_shapefile_layer(shapes, layer)
            if len(layers) != 1:
                raise ValueError(f'Select exactly one shapefile layer in {filename.name}')
            shape = layers[0]
            sidecars = {name.suffix.lower(): name for name in members
                        if name.with_suffix('') == shape.with_suffix('')}
            if '.dbf' not in sidecars:
                raise ValueError(f'Missing {shape.with_suffix(".dbf").name} in {filename.name}')
            source_crs = None
            if target_crs is not None:
                from rasterio.crs import CRS
                if '.prj' not in sidecars:
                    raise ValueError(f'Shapefile CRS transformation requires {shape.with_suffix(".prj").name}')
                source_crs = CRS.from_wkt(archive.read(str(sidecars['.prj'])).decode('utf-8-sig'))
            streams = {ext[1:]: stack.enter_context(archive.open(str(sidecars[ext])))
                       for ext in ('.shp', '.shx', '.dbf') if ext in sidecars}
            with shapefile.Reader(**streams) as reader:
                yield reader, source_crs
        return

    if not _select_shapefile_layer([path], layer):
        raise ValueError('A shapefile contains one layer named after its filename')
    source_crs = None
    if target_crs is not None:
        from rasterio.crs import CRS
        prj = path.with_suffix('.prj')
        if not prj.is_file():
            raise ValueError(f'Shapefile CRS transformation requires {prj.name}')
        source_crs = CRS.from_wkt(prj.read_text(encoding='utf-8-sig'))
    with shapefile.Reader(str(path)) as reader:
        yield reader, source_crs


def _shapefile_records(path, fields, layer, target_crs, *, legacy_ogr=False):
    with _open_shapefile(path, layer, target_crs) as (reader, source_crs):
        field_names = [f[0] for f in reader.fields[1:]]
        missing = set(fields)-set(field_names)
        if missing:
            raise ValueError(f'Missing numeric GIS fields {sorted(missing)}')
        for entry in reader.iterShapeRecords():
            values = tuple(float(entry.record[field]) for field in fields)
            if entry.shape.shapeType not in [5, 15, 25]:
                if legacy_ogr:
                    yield (), values, source_crs
                    continue
                raise ValueError('GIS input must contain polygons')
            points = entry.shape.points
            starts = tuple(entry.shape.parts)+(len(points),)
            if legacy_ogr and len(starts) > 2:
                geometry = entry.shape.__geo_interface__
                exterior = geometry['coordinates'][:1] if geometry['type'] == 'Polygon' else ()
                yield exterior, values, source_crs
                continue
            rings = (points[start:end] for start, end in zip(starts[:-1], starts[1:]))
            yield tuple(rings), values, source_crs


def _vector_records(path, fields, layer, target_crs, *, legacy_ogr=False):
    try:
        import fiona
    except ImportError as exc:
        raise ImportError('Named GIS vector inputs require plumeswift-research[gis]') from exc
    with fiona.open(str(path), layer=layer) as reader:
        numeric_fields = {
            name for name, kind in reader.schema['properties'].items()
            if kind.split(':', 1)[0] in {'int', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'}
        }
        missing = set(fields)-numeric_fields
        if missing:
            raise ValueError(f'Missing numeric GIS fields {sorted(missing)}')
        source_crs = None
        if target_crs is not None:
            from rasterio.crs import CRS
            if not reader.crs_wkt:
                raise ValueError(f'GIS CRS transformation requires a source CRS for layer {reader.name}')
            source_crs = CRS.from_wkt(reader.crs_wkt)
        for entry in reader:
            geometry = entry.geometry
            values = tuple(float(entry.properties[field]) for field in fields)
            if legacy_ogr:
                exterior = geometry.coordinates[:1] if geometry is not None and geometry.type == 'Polygon' else ()
                yield exterior, values, source_crs
                continue
            if geometry is None or geometry.type not in {'Polygon', 'MultiPolygon'}:
                raise ValueError('GIS input must contain polygons')
            parts = (geometry.coordinates,) if geometry.type == 'Polygon' else geometry.coordinates
            for rings in parts:
                yield rings, values, source_crs


def _layer_polygons(path, fields, layer, target_epsg, *, legacy_ogr=False):
    path = Path(path)
    if path.is_dir():
        shapes = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() == '.shp']
        if shapes:
            candidates = _select_shapefile_layer(shapes, layer)
            if len(candidates) != 1:
                raise ValueError(f'Select exactly one shapefile layer in {path.name}')
            path, layer = candidates[0], None
    target_crs = None
    if target_epsg is not None:
        from rasterio.crs import CRS
        target_crs = CRS.from_epsg(target_epsg)
    records = _shapefile_records if _is_shapefile(path) else _vector_records
    legacy_attributes = []
    polygon_number = 0
    for rings, values, source_crs in records(path, fields, layer, target_crs, legacy_ogr=legacy_ogr):
        if legacy_ogr:
            # ESRIShapefile stores attributes from every feature and the
            # wind constructor indexes them by the retained polygon number.
            legacy_attributes.append(values)
            if not rings:
                continue
            values = legacy_attributes[polygon_number]
            polygon_number += 1
        for exterior, holes in _polygon_parts(rings, source_crs, target_crs):
            yield exterior, holes, values


def polygon_layer_extents(path, *, layer=None, target_epsg=None):
    """Return the QES layer minimum, maximum and rounded-up local dimensions."""
    polygons = list(_layer_polygons(path, (), layer, target_epsg))
    if not polygons:
        raise ValueError('GIS layer has no polygons')
    minimum = np.min([exterior.min(axis=0) for exterior, _, _ in polygons], axis=0).astype(np.float32)
    maximum = np.max([exterior.max(axis=0) for exterior, _, _ in polygons], axis=0).astype(np.float32)
    dimensions = np.ceil(maximum-minimum)
    return tuple(map(float, minimum)), tuple(map(float, maximum)), tuple(map(int, dimensions))


def _polygon_features(path, fields, origin, layer, target_epsg, *, legacy_ogr=False):
    polygons = list(_layer_polygons(path, fields, layer, target_epsg, legacy_ogr=legacy_ogr))
    if not polygons:
        return ()
    minimum = np.min([exterior.min(axis=0) for exterior, _, _ in polygons], axis=0)
    use_origin = origin is not None and origin[0] != 0 and (origin[1] != 0 or target_epsg is not None)
    shift = np.asarray(origin if use_origin else minimum,
                       dtype=np.float32)
    # Native reads double OGR positions into float polyVert before applying
    # the float origin offset. Preserve this ordering for baseline parity.
    def translated(ring):
        return tuple(map(tuple, (ring.astype(np.float32)-shift).astype(float)))
    return tuple((translated(exterior), tuple(map(translated, holes)), values)
                 for exterior, holes, values in polygons)


def read_building_shapefile(path, *, height_field='H', height_factor=1., origin=None, layer=None,
                            target_epsg=None, native_math=False, legacy_ogr=False):
    scalar = np.float32 if native_math else float
    height_factor = scalar(height_factor)
    if not np.isfinite(height_factor) or height_factor <= 0:
        raise ValueError('Height factor must be positive')
    # OGR attributes are stored in vector<float>, and the XML factor is float.
    polygons = _polygon_features(path, (height_field,), origin, layer, target_epsg, legacy_ogr=legacy_ogr)
    return tuple(PolygonBuilding(vertices, float(scalar(height)*height_factor), holes=holes)
                 for vertices, holes, (height,) in polygons)


def read_tree_shapefile(path, *, origin=None, layer=None, target_epsg=None, legacy_ogr=False):
    from .canopy_tree import IsolatedTree

    trees = []
    polygons = _polygon_features(path, ('H', 'D', 'LAI'), origin, layer, target_epsg, legacy_ogr=legacy_ogr)
    for vertices, holes, (height, diameter, lai) in polygons:
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('Tree GIS diameter must be positive')
        trees.append(IsolatedTree(vertices, height, lai, .5, holes=holes))
    return tuple(trees)
