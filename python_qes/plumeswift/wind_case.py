"""Composable wind pipeline. Unsupported XML features fail before execution.

Parsing an original file never means silently accepting an unported feature.
This runner currently supports flat domains, rectangular geometry, local
stations with four vertical profile families, original upwind/wake/sidewall/
rooftop and legacy/modified street closures; local mixing methods 0 through 4.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
from .domain import Domain, HorizontalHalo
from .geometry import PolygonBuilding, translate_footprint, vertex_parts, wall_stencil, zero_solid_faces
from .meteorology import SensorProfile, Station, single_sensor_flat, barnes_flat
from .building_pipeline import BuildingPipeline
from .turbulence import LocalTurbulence
from .numerics.cuda_dynamic_legacy import CudaDynamicLegacy
from .numerics.poisson import NATIVE_WIND_SOLVERS, KrylovPoisson, NativeWindPolicy, make_poisson
from .wind_projection import project
from .wind_visualization import FIELDS as VISUALIZATION_FIELDS, read_output_fields
from .wind_wall_log import WallLogCorrection
from .wind_xml import TIME_SERIES_FIELDS, normalize_legacy_turbulence, normalize_sensor_xml, normalize_wind_xml
from .qes_xml import parse_optional_number, parse_vector, read_qes_xml
from .timing import QEStime


class UnsupportedFeatures(ValueError):
    def __init__(self, features):
        self.features = tuple(features)
        super().__init__('Requested upstream features are not ported: '+', '.join(features))


@dataclass(frozen=True)
class MeteorologyStep:
    timestamp: str
    stations: tuple[Station, ...]


@dataclass
class WindCase:
    domain: Domain
    profile: SensorProfile
    buildings: tuple[PolygonBuilding, ...] = ()
    upwind_mode: int = 0
    turbulence: LocalTurbulence | None = None
    poisson: object = field(default_factory=KrylovPoisson)
    wake_mode: int = 0
    rooftop_mode: int = 0
    sidewall_mode: int = 0
    high_rise: bool = False
    wall_roughness: float = .01
    stations: tuple[Station, ...] = ()
    building_pipeline: BuildingPipeline | None = None
    allow_unset_wake: bool = field(default=False, kw_only=True)
    street_canyon_mode: int = 0
    timestamp: str = '2020-01-01T00:00:00'
    mixing_method: int = 0
    mixing_samples: int = 2000
    mixing_ray_backend: str = 'cpu'
    mixing_file: str | None = None
    mixing_variable: str = 'mixlength'
    save_mixing: bool = False
    terrain: object | None = None
    meteorology_variant: str = 'native-cpu'
    vegetation: object | None = None
    schedule: tuple[MeteorologyStep, ...] = ()
    reference_roughness: float | None = None
    coefficient_file: str | None = None
    roughness_split: bool = False
    halo: HorizontalHalo = field(default_factory=HorizontalHalo)
    mesh_type: int = 0
    hrrr: object | None = None
    hrrr_grid_ids: tuple = ()
    hrrr_nearest_map_positions: tuple = field(default=(), kw_only=True)
    wrf: object | None = None
    wrf_atmosphere_path: str | None = None
    wrf_atmosphere_coordinates: str | None = None
    wrf_exchange: object | None = None
    visualization_fields: tuple[str, ...] = VISUALIZATION_FIELDS
    wall_log: WallLogCorrection | None = None
    domain_rotation_deg: float = 0.
    street_intersection_flag: int = 0
    xml_compatibility: tuple[str, ...] = ()
    native_sensor_math: bool = False
    native_wake_math: bool = False
    native_building_math: bool = False
    meteorology_backend: str = 'cpu'
    native_staircase_metrics: bool = False
    native_netcdf_inputs: bool = False
    wrf_zero_steps: bool = False
    coefficient_flag: int = 0

    def iter_steps(self):
        """Yield wind cases in native timestamp order."""
        if self.wrf_exchange is not None:
            raise ValueError('Live WRF requires iter_results() to complete the frame exchange')
        if self.wrf_zero_steps:
            return
        if not self.schedule:
            yield self
        previous_timestamp = None
        previous_stations = ()
        for index, step in enumerate(self.schedule):
            stations = step.stations
            if (self.native_sensor_math or self.meteorology_backend == 'cuda') and step.timestamp == previous_timestamp:
                from .native_sensor import advance_canopy_station
                # The parser selects the last matching station record at each
                # timestamp. Native repeats then reuse that same mutable record.
                stations = tuple(advance_canopy_station(station) for station in previous_stations)
            previous_timestamp, previous_stations = step.timestamp, stations
            yield replace(self, timestamp=step.timestamp, stations=stations,
                          profile=stations[0].profile, schedule=(),
                          save_mixing=self.save_mixing and index == 0)

    def nonlocal_turbulence_state(self, fields):
        """Snapshot the fields changed by the native non-local wake closure."""
        if fields is None or not self.native_netcdf_inputs or self.turbulence is None or not self.turbulence.nonlocal_mixing:
            return None
        return {key: fields[key].copy() for key in ('Lm', 'iturbflag')}

    def iter_results(self, *, initialize_only=False, include_gradients=False):
        exchange = self.wrf_exchange
        previous_turbulence = None

        def calculate(step):
            nonlocal previous_turbulence
            result = step.run(initialize_only=initialize_only, include_gradients=include_gradients,
                              previous_turbulence=previous_turbulence)
            previous_turbulence = step.nonlocal_turbulence_state(result['turbulence'])
            return result

        if self.wrf_zero_steps:
            if exchange is not None:
                exchange.receive(1)
            self.initialize_geometry(prepare_mixing=True)
            if exchange is not None:
                exchange.finish()
            return
        if exchange is None:
            for step in self.iter_steps():
                yield step, calculate(step)
            return
        if initialize_only:
            raise ValueError('Initialization-only output requires offline wind input')
        previous_time = None
        for frame in range(1, exchange.steps+1):
            snapshot, timestamp = exchange.receive(frame)
            if previous_time is not None and timestamp < previous_time:
                raise ValueError('WRF input frame time moved backwards')
            previous_time = timestamp
            step = replace(self, wrf=snapshot, wrf_exchange=None, timestamp=timestamp,
                           save_mixing=self.save_mixing and frame == 1)
            result = calculate(step)
            if not result['solver'].converged and not exchange.legacy_protocol:
                raise RuntimeError(f'QES wind solver did not converge for WRF frame {frame}')
            checksum = exchange.publish(frame, result['workspace'], self.halo)
            result['wrf_frame'] = dict(frame=frame, input_record=snapshot.record, output_checksum=checksum)
            yield step, result
        # WRF must observe the last ordinary acknowledgement before QES
        # publishes the end marker. Native qesWinds validates the final wind
        # payload, but does not read a timestamp for this acknowledgement.
        exchange.receive(exchange.steps+1, record_time=not exchange.legacy_protocol)
        exchange.finish()

    def _pipeline(self):
        if self.building_pipeline is not None:
            return self.building_pipeline
        if not self.buildings:
            return BuildingPipeline(())
        return BuildingPipeline.native_order(self.upwind_mode, self.wake_mode,
            self.sidewall_mode, self.rooftop_mode, self.high_rise, self.wall_roughness, self.street_canyon_mode,
            native_wake_math=self.native_wake_math, native_building_math=self.native_building_math,
            allow_unset_wake=self.allow_unset_wake)

    def manifest(self):
        from dataclasses import asdict
        import hashlib
        pipeline = self._pipeline()
        package = Path(__file__).parent
        return dict(timestamp=self.timestamp, mixing_method=self.mixing_method, mixing_samples=self.mixing_samples,
            domain_rotation_deg=self.domain_rotation_deg,
            street_intersection_flag=self.street_intersection_flag,
            mixing_ray_backend=self.mixing_ray_backend,
            xml_compatibility=self.xml_compatibility,
            mixing_file=self.mixing_file,
            visualization_fields=self.visualization_fields,
            wall_log=asdict(self.wall_log) if self.wall_log is not None else None,
            meteorology_schedule=[asdict(step) for step in self.schedule], reference_roughness=self.reference_roughness,
            coefficient_file=self.coefficient_file, coefficient_flag=self.coefficient_flag,
            roughness_split=self.roughness_split, halo=asdict(self.halo), halo_cells=self.halo.cells(self.domain.spacing),
            mesh_type=self.mesh_type, hrrr=asdict(self.hrrr) if self.hrrr is not None else None,
            wrf=self.wrf.describe() if self.wrf is not None else None,
            wrf_exchange=self.wrf_exchange.describe() if self.wrf_exchange is not None else None,
            wrf_atmosphere_path=self.wrf_atmosphere_path,
            wrf_atmosphere_coordinates=self.wrf_atmosphere_coordinates,
            vegetation=asdict(self.vegetation) if self.vegetation is not None else None,
            terrain=self.terrain.describe() if self.terrain is not None else None, meteorology_variant=self.meteorology_variant,
            meteorology_backend=self.meteorology_backend,
            native_sensor_math=self.native_sensor_math, native_wake_math=self.native_wake_math,
            native_building_math=self.native_building_math,
            native_staircase_metrics=self.native_staircase_metrics,
            mixing_variable=self.mixing_variable, save_mixing=self.save_mixing, domain=asdict(self.domain), profile=asdict(self.profile), stations=[asdict(s) for s in self.stations],
            buildings=[asdict(b) for b in self.buildings],
            building_stages=[dict(implementation=type(s).__module__+'.'+type(s).__qualname__, parameters=asdict(s)) for s in pipeline.stages],
            turbulence=asdict(self.turbulence) if self.turbulence is not None else None,
            poisson=dict(implementation=type(self.poisson).__module__+'.'+type(self.poisson).__qualname__,
                         policy=asdict(self.poisson.policy) if hasattr(self.poisson, 'policy') else None,
                         multiplier_domain='all native internal cells, including solids',
                         bottom_boundary=('native CUDA face-plane span with a source snapshot'
                                          if getattr(self.poisson, 'native_boundary', False)
                                          else 'mirrored ground cell plane')),
            package_sha256={p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))})

    def _prepare_geometry(self):
        workspace = self.domain.workspace()
        mark_solid = self.coefficient_flag == 0 and self.coefficient_file is None
        if self.reference_roughness is not None:
            uniform_z0 = self.reference_roughness
        else:
            uniform_z0 = self.turbulence.terrain_roughness if self.turbulence is not None else self.profile.roughness
        roughness = np.full(self.domain.face_shape[1:], uniform_z0, dtype=float)
        if self.roughness_split:
            split = roughness.shape[1]//2
            roughness[:, :split] = .5
            roughness[:, split:] = .1
        workspace.arrays.update(z0_u=roughness, z0_v=roughness.copy())
        building_ids = tuple(number for number, building in enumerate(self.buildings)
                             if building.intersects_horizontal_domain(workspace.coordinates))
        buildings = tuple(self.buildings[number] for number in building_ids)
        mesh = None
        terrain_datum = 0.
        cutcell = self.mesh_type == 1 and mark_solid
        if cutcell:
            for name in ('wall_distance', 'ni', 'nj', 'nk'):
                workspace.arrays[name] = np.zeros(self.domain.cell_shape)
            for name in ('terrain_volume_frac', 'building_volume_frac', 'center_id'):
                workspace.arrays[name] = np.ones(self.domain.cell_shape)
        if self.terrain is not None:
            from .terrain import apply_stairstep_terrain
            if self.hrrr is not None and hasattr(self.terrain, 'geometry_with_datum'):
                mesh, corners, terrain_datum = self.terrain.geometry_with_datum(self.domain, self.halo if cutcell else None)
            elif cutcell:
                if not hasattr(self.terrain, 'cut_geometry'):
                    raise UnsupportedFeatures(['cut-cell generation for this terrain provider'])
                mesh, corners = self.terrain.cut_geometry(self.domain, self.halo)
            else:
                mesh = self.terrain.mesh(self.domain)
            workspace.terrain_mesh = mesh
            apply_stairstep_terrain(workspace, mesh, mark_solid=mark_solid and not cutcell,
                                    halo_cells=self.halo.cells(self.domain.spacing),
                                    native_sensor_math=self.native_sensor_math or self.native_staircase_metrics)
            if cutcell:
                from .cutcell_terrain import apply_cutcell_terrain
                apply_cutcell_terrain(workspace, corners)
            positioned = []
            for building in buildings:
                vertices = np.asarray(building.terrain_sample_vertices
                                      or (building.native_closure_vertices if building.detached_parts else None)
                                      or building.vertices)
                heights = mesh.height(vertices[:, 0], vertices[:, 1],
                                      native_math=self.native_building_math or self.native_staircase_metrics)
                base = max(0., float(heights[0]))
                for height in heights[1:]:
                    if 0 <= height < base: base = float(height)
                positioned.append(replace(building, base_height=base))
            buildings = tuple(positioned)
        for number, building in zip(building_ids, buildings, strict=True):
            building.flag(workspace, number, mark_solid=mark_solid,
                          native_dy_alias=cutcell or self.native_staircase_metrics,
                          native_math=self.native_building_math or self.native_staircase_metrics,
                          native_wrapped_footprint=self.native_netcdf_inputs and self.mesh_type == 0)
            if cutcell:
                from .cutcell_building import apply_cutcell_building
                apply_cutcell_building(workspace, building, number, native_math=self.native_staircase_metrics,
                                       native_ring_joins=self.native_netcdf_inputs)
        vegetation_state = None
        if self.vegetation is not None:
            from .vegetation import Vegetation
            if self.native_staircase_metrics and type(self.vegetation) is Vegetation:
                vegetation_state = self.vegetation.setup(workspace, native_geometry=True)
            else:
                vegetation_state = self.vegetation.setup(workspace)
        if cutcell:
            from .cutcell_walls import cutcell_wall_stencil
            cutcell_wall_stencil(workspace, native_math=self.native_staircase_metrics)
        else:
            wall_stencil(workspace, native_dy_alias=self.native_staircase_metrics,
                         native_math=self.native_staircase_metrics)
        workspace.arrays['icellflag_initial'] = workspace.arrays['icellflag'].copy()
        if self.coefficient_file is not None:
            from .io import read_solver_coefficients
            # Native skips terrain/building solid flags in reload mode and
            # immediately resets the file's flags to this prepared mask.
            workspace.arrays.update(read_solver_coefficients(
                self.coefficient_file, self.domain.cell_shape, native_hyperslab=self.native_netcdf_inputs,
                prepared_stencil=workspace.arrays))
        return workspace, building_ids, buildings, mesh, terrain_datum, vegetation_state

    def initialize_geometry(self, *, prepare_mixing=False):
        """Construct wind geometry and wall coefficients before loading a wind profile."""
        workspace, _, buildings, mesh, *_ = self._prepare_geometry()
        if prepare_mixing:
            self._prepare_mixing(workspace, buildings, mesh)
        return workspace

    def _prepare_mixing(self, workspace, buildings, mesh):
        if self.turbulence is not None and self.mixing_method == 1:
            from .mixing_length import SerialMixing
            workspace.arrays['mixlength'] = SerialMixing(native_math=self.turbulence.native_math)(workspace)
        elif self.turbulence is not None and self.mixing_method == 2:
            from .mesh_mixing import legacy_terrain_mixing
            workspace.arrays['mixlength'] = legacy_terrain_mixing(workspace, self.domain)
        elif self.turbulence is not None and self.mixing_method == 3:
            from .ray_mixing import RayMixing, mixing_scene
            footprints = (part for building in buildings for part in building.physical_parts())
            workspace.arrays['mixlength'] = RayMixing(self.mixing_samples, self.mixing_ray_backend)(
                workspace, mixing_scene(workspace, footprints, mesh))
        elif self.turbulence is not None and self.mixing_method == 4:
            from .mixing_length import NetCDFMixing
            workspace.arrays['mixlength'] = NetCDFMixing(self.mixing_file, self.mixing_variable,
                                                        native_hyperslab=self.native_netcdf_inputs)(workspace)
        if self.turbulence is not None and self.save_mixing and self.mixing_method in (1, 2, 3):
            from .mixing_length import save_mixing_length
            save_mixing_length(self.mixing_file, self.mixing_variable, workspace.arrays['mixlength'])

    def prepare(self):
        """Create the workspace and a closure for reapplying parameterizations."""
        if self.native_sensor_math and (self.terrain is not None or self.hrrr is not None) and self.meteorology_variant != 'native-cpu':
            raise ValueError('Native float32 sensor initialization requires the CPU meteorology variant')
        if self.wrf_exchange is not None:
            raise ValueError('Live WRF requires iter_results() to obtain the wind frame')
        workspace, building_ids, buildings, mesh, terrain_datum, vegetation_state = self._prepare_geometry()
        self._prepare_mixing(workspace, buildings, mesh)
        if self.wrf is not None:
            self.wrf.initialize(workspace, self.halo, native_math=self.native_sensor_math)
        elif self.meteorology_backend == 'cuda' and self.hrrr is None and len(self.stations) > 1:
            from .cuda_barnes import barnes_cuda
            barnes_cuda(workspace, self.stations)
        elif (self.native_sensor_math or self.meteorology_backend == 'cuda') and self.terrain is None and self.hrrr is None:
            from .native_sensor import initialize_flat
            initialize_flat(workspace, self.stations, self.profile)
        elif self.hrrr is not None:
            from .hrrr_interpolation import initialize_hrrr
            initialize_hrrr(workspace, self.stations, self.hrrr_grid_ids, self.hrrr.scheme,
                            variant=self.meteorology_variant, datum=terrain_datum,
                            native_math=self.native_sensor_math or self.native_netcdf_inputs,
                            barnes_backend=self.meteorology_backend,
                            legacy_terrain_stride=self.native_netcdf_inputs,
                            nearest_map_positions=self.hrrr_nearest_map_positions)
        elif self.terrain is not None:
            from .terrain_meteorology import barnes_terrain
            if not self.stations:
                raise ValueError('Terrain-dependent met input requires explicit Station coordinates')
            barnes_terrain(workspace, self.stations, variant=self.meteorology_variant,
                           native_math=self.native_sensor_math or self.meteorology_backend == 'cuda')
        elif self.stations:
            barnes_flat(workspace, self.stations)
        else:
            single_sensor_flat(workspace, self.profile)
        # All building upwind operations precede wake/sidewall/rooftop in QES.
        # Native mergeSort is ascending effective height and picks the right
        # half first on ties, reversing original order among equal heights.
        height_type = np.float32 if self.native_building_math or self.native_staircase_metrics else float
        ordered = sorted(zip(building_ids, buildings, strict=True),
                         key=lambda item: (height_type(item[1].height)+height_type(item[1].base_height), -item[0]))
        pipeline = self._pipeline()
        if self.building_pipeline is None:
            # Native closures scan the flat XML sequence, including ring
            # joins. Geometry still uses the separate physical footprints.
            ordered = [(number, replace(building, vertices=building.native_closure_vertices)
                        if building.native_closure_vertices is not None else building)
                       for number, building in ordered]

        def apply_parameterizations():
            if self.vegetation is not None:
                self.vegetation.apply(workspace, vegetation_state)
            workspace.building_context = pipeline.apply(workspace, ordered)
            if self.vegetation is not None:
                self.vegetation.apply_wake(workspace, vegetation_state)
                if self.vegetation.wake:
                    workspace.row_canopy_flows = tuple(vegetation_state.row_flows.values())
            zero_solid_faces(workspace)
            for name in ('u0', 'v0', 'w0'):
                if not np.isfinite(workspace.arrays[name]).all():
                    count = int(np.count_nonzero(~np.isfinite(workspace.arrays[name])))
                    raise FloatingPointError(f'Non-finite initial wind {name}: {count} faces after empirical closures')

        return workspace, apply_parameterizations

    def initialize(self):
        workspace, apply_parameterizations = self.prepare()
        apply_parameterizations()
        return workspace

    def run(self, *, initialize_only=False, include_gradients=False, initial_multiplier=None,
            previous_turbulence=None):
        """Run the selected time step. Use iter_steps for a series."""
        if initialize_only:
            workspace = replace(self, turbulence=None).initialize()
            workspace.arrays.update({name: workspace.arrays[name+'0'].copy() for name in 'uvw'})
            return dict(workspace=workspace, multiplier=None, solver=None, turbulence=None, turbulence_metadata=None)
        return self.solve(self.initialize(), include_gradients=include_gradients, initial_multiplier=initial_multiplier,
                          previous_turbulence=previous_turbulence)

    def solve(self, workspace, *, include_gradients=False, initial_multiplier=None, previous_turbulence=None):
        """Project an initialized workspace and calculate turbulence."""
        projection_workspace = workspace
        if self.native_staircase_metrics and isinstance(self.poisson, CudaDynamicLegacy):
            # Native CUDA solvers get (dx, dx, dz) from Domain::getDomainSize.
            # Use the alias for projection only, keeping the physical y axis in the workspace.
            axes = workspace.coordinates
            dx = axes['x'][1]-axes['x'][0]
            legacy_y = dx*(np.arange(len(axes['y']))+.5)
            projection_workspace = replace(workspace, coordinates=axes | {'y': legacy_y})
        velocity, multiplier, report = project(projection_workspace, self.poisson, initial=initial_multiplier)
        if self.wall_log is not None:
            velocity = self.wall_log(workspace, velocity)
            # qesWinds retains the post-wall field in both work and output buffers.
            workspace.arrays.update({name+'0': values.copy() for name, values in velocity.items()})
        workspace.arrays.update(velocity)
        turbulence = metadata = None
        if self.turbulence is not None:
            roughness = self.profile.roughness if self.reference_roughness is None else self.reference_roughness
            diagnostics = {'include_gradients': True} if include_gradients else {}
            if previous_turbulence is not None:
                diagnostics['previous_turbulence'] = previous_turbulence
            buildings = self.buildings
            if self.turbulence.native_math and buildings:
                # Native morphology accumulates heights in the sorted building-id order.
                positioned = dict(getattr(workspace.building_context, 'ordered_buildings', ()))
                indexed = [(number, positioned.get(number, building)) for number, building in enumerate(buildings)]
                indexed.sort(key=lambda item: (np.float32(item[1].height)+np.float32(item[1].base_height), -item[0]))
                buildings = tuple(building for _, building in indexed)
            turbulence, metadata = self.turbulence.build(workspace, velocity,
                sensor_roughness=roughness, building_heights=[b.height for b in buildings],
                mixing_distance=workspace.arrays.get('mixlength'), **diagnostics)
        return dict(workspace=workspace, multiplier=multiplier, solver=report,
                    turbulence=turbulence, turbulence_metadata=metadata)


def read_wind_case(path, *, poisson=None, poisson_backend='cpu', compute_turbulence=True, wrf_timeout=None,
                   turbulence_gradient='cpu', native_sensor_math=False, native_wake_math=False,
                   native_turbulence_math=False, native_building_math=False, wrf_startup_only=False,
                   meteorology_backend='cpu', legacy_turbulence_xml=False, legacy_wrf_protocol=False):
    root = read_qes_xml(path)
    return _read_wind_root(path, root, poisson=poisson, poisson_backend=poisson_backend,
                           compute_turbulence=compute_turbulence, wrf_timeout=wrf_timeout,
                           turbulence_gradient=turbulence_gradient, native_sensor_math=native_sensor_math,
                           native_wake_math=native_wake_math, native_turbulence_math=native_turbulence_math,
                           native_building_math=native_building_math, wrf_startup_only=wrf_startup_only,
                           meteorology_backend=meteorology_backend, legacy_turbulence_xml=legacy_turbulence_xml,
                           legacy_wrf_protocol=legacy_wrf_protocol)


def _read_wind_root(path, root, *, poisson=None, poisson_backend='cpu', compute_turbulence=True, wrf_timeout=None,
                    turbulence_gradient='cpu', native_sensor_math=False, native_wake_math=False,
                    native_turbulence_math=False, native_building_math=False, wrf_startup_only=False,
                    meteorology_backend='cpu', legacy_turbulence_xml=False, legacy_wrf_protocol=False):
    if root.tag != 'QESWindsParameters':
        raise ValueError('Expected QESWindsParameters root')
    if meteorology_backend not in ('cpu', 'cuda'):
        raise ValueError('Unknown meteorology backend')
    from .input_paths import QESInputPaths

    paths = QESInputPaths.from_input(path)
    root, xml_compatibility = normalize_wind_xml(root)
    fire = root.find('fires')
    if fire is not None:
        from .fire_case import _parse_fire_config
        _parse_fire_config(path, fire, legacy_fire_xml=legacy_turbulence_xml)
        root.remove(fire)
    unsupported = []
    sim = root.find('simulationParameters')
    if sim is None:
        raise ValueError('Missing simulationParameters')
    # Modern inputs audit every field. Legacy drivers match Boost's selective
    # parser and record unconsumed subtrees in the compatibility manifest.
    allowed = {
        'simulationParameters': {'halo_x', 'halo_y', 'domain', 'cellSize', 'verticalStretching', 'dz_value', 'totalTimeIncrements', 'maxIterations', 'tolerance', 'meshTypeFlag', 'logLawFlag', 'domainRotation', 'originFlag', 'UTMx', 'UTMy', 'UTMZone', 'UTMZoneLetter', 'reprojectShapefiles', 'reprojectDEM', 'readCoefficientsFlag', 'COEFF', 'DEM', 'DEMDistancex', 'DEMDistancey', 'WRF', 'WRFSensorSample', 'WRFCoupling', 'WRFCoordinates'},
        'metParams': {'z0_domain_flag', 'sensor', 'sensorName'},
        'sensor': {'site_coord_flag', 'site_xcoord', 'site_ycoord', 'site_UTM_x', 'site_UTM_y', 'site_UTM_zone', 'site_lon', 'site_lat', 'timeSeries'},
        'timeSeries': TIME_SERIES_FIELDS,
        'turbParams': {'sigmaConst', 'method', 'samples', 'rayBackend', 'save', 'LMfile', 'varname', 'nonLocalMixing', 'terrainWallFlag', 'buildingWallFlag', 'backgroundMixing', 'turbUpperBound'},
        'buildingsParams': {'wallRoughness', 'rooftopFlag', 'upwindCavityFlag', 'streetCanyonFlag', 'streetIntersectionFlag', 'wakeFlag', 'highRiseFlag', 'sidewallFlag', 'numBuildings', 'numPolygonNodes', 'rectangularBuilding', 'QUICBuilding', 'SHPFile', 'SHPBuildingLayer', 'SHPHeightField', 'heightFactor'},
        'rectangularBuilding': {'groupID', 'buildingType', 'height', 'baseHeight', 'xStart', 'yStart', 'length', 'width', 'buildingRotation'},
        'QUICBuilding': {'height', 'baseHeight', 'xVertex', 'yVertex'},
        'fileOptions': {'outputFlag', 'outputFields', 'massConservedFlag', 'sensorVelocityFlag', 'staggerdVelocityFlag'},
        'vegetationParams': {'wakeFlag', 'Homogeneous', 'IsolatedTree', 'Windbreak', 'ROC', 'SHPFile', 'SHPTreeLayer'},
        'Homogeneous': {'attenuationCoefficient', 'height', 'baseHeight', 'xStart', 'yStart', 'length', 'width', 'canopyRotation', 'xVertex', 'yVertex'},
        'IsolatedTree': {'attenuationCoefficient', 'height', 'zMaxLAI', 'baseHeight', 'xCenter', 'yCenter', 'width'},
        'Windbreak': {'height', 'baseHeight', 'understroyHeight', 'xStart', 'yStart', 'length', 'width', 'canopyRotation', 'opticalPorosity', 'fenceModel', 'fetch'},
        'ROC': {'height', 'baseHeight', 'understoryHeight', 'xVertex', 'yVertex', 'thinFence', 'rowSpacing',
                'rowWidth', 'rowAngle', 'LAD_eff', 'LAD_avg', 'opticalPorosity', 'tkeMax', 'upstreamSigmaW',
                'upstreamUstar', 'displacementHeightParallel'},
        'hrrrInput': {'HRRRFile', 'interpolationScheme', 'stabilityClasses'},
        'QESWindsParameters': {'simulationParameters', 'metParams', 'turbParams', 'buildingsParams',
                              'vegetationParams', 'fileOptions', 'hrrrInput'},
    }
    ignored_xml = []

    def audit(node, prefix=''):
        for child in node:
            address = prefix+'/'+child.tag
            if child.tag not in allowed.get(node.tag, set()):
                if legacy_turbulence_xml:
                    ignored_xml.append(f'ignored {address}')
                    continue
                unsupported.append(address)
            if len(child):
                audit(child, address)
    audit(root)

    def simulation_option(name, default):
        return parse_optional_number(sim, name, default, convert=type(default), legacy=legacy_turbulence_xml)

    domain_rotation_deg = simulation_option('domainRotation', 0.)
    if not np.isfinite(domain_rotation_deg):
        raise ValueError('domainRotation must be finite')
    halo = HorizontalHalo(*(simulation_option(key, 0.) for key in ('halo_x', 'halo_y')))
    coefficient_flag = simulation_option('readCoefficientsFlag', 0)
    mesh_type = simulation_option('meshTypeFlag', 0)
    wall_log_enabled = simulation_option('logLawFlag', 0) == 1
    if mesh_type not in (0, 1):
        unsupported.append(f'meshTypeFlag={mesh_type}')
    coefficient_file = None
    if coefficient_flag == 1:
        filename = sim.findtext('COEFF', '').strip()
        if not filename:
            raise ValueError('COEFF is required when readCoefficientsFlag is 1')
        coefficient_file = str(paths.resolve(filename))
    elif coefficient_flag != 0 and not legacy_turbulence_xml:
        unsupported.append(f'readCoefficientsFlag={coefficient_flag}')
    wrf_file = sim.findtext('WRF', '').strip()
    step_limit = int(sim.findtext('totalTimeIncrements', '1'))
    if legacy_turbulence_xml and not wrf_file and -(1 << 32) < step_limit < 0:
        # Native XML reads unsigned int before capping to available weather records.
        step_limit += 1 << 32
    if step_limit < 0:
        raise ValueError('totalTimeIncrements must be nonnegative')
    met = root.find('metParams')
    hrrr_node = root.find('hrrrInput')

    def meteorology_option(node, name, default):
        value = node.findtext(name, default).strip()
        return default if legacy_turbulence_xml and not value else value

    dem_file = sim.findtext('DEM', '').strip()
    wrf_only = bool(wrf_file and not dem_file)
    wrf_coordinates = sim.findtext('WRFCoordinates', 'geographic').strip()
    if wrf_coordinates not in ('geographic', 'ideal-local'):
        raise ValueError('WRFCoordinates must be geographic or ideal-local')
    if wrf_coordinates == 'ideal-local' and not (wrf_file and dem_file):
        raise ValueError('Ideal-local WRF coordinates require atmospheric WRF + DEM input')
    coupling = meteorology_option(sim, 'WRFCoupling', 'false').lower()
    if coupling not in ('0', '1', 'false', 'true'):
        raise ValueError('Invalid WRFCoupling boolean')
    live_wrf = coupling in ('1', 'true')
    if wrf_timeout is not None and not live_wrf:
        raise ValueError('WRF timeout requires WRFCoupling=true')
    if live_wrf and not wrf_only:
        unsupported.append('live WRFCoupling requires a WRF-SFIRE fire mesh')
    if live_wrf and wrf_startup_only and step_limit != 1:
        unsupported.append('coupled qes and qesFire require one WRF startup record (totalTimeIncrements=1)')
    if wrf_file and hrrr_node is not None:
        raise ValueError('Select one of WRF and HRRR')
    if met is None:
        raise ValueError('Missing metParams')
    roughness_split = int(met.findtext('z0_domain_flag', '0')) != 0
    sensors = met.findall('sensor')
    weather_source = 'WRF' if wrf_file else 'HRRR' if hrrr_node is not None else None
    for entry in met.findall('sensorName'):
        sensor_path = paths.resolve((entry.text or '').strip())
        if legacy_turbulence_xml and weather_source is not None:
            ignored_xml.append(f'ignored /metParams/sensorName/{sensor_path.name} while using {weather_source}')
            continue
        sensor = read_qes_xml(sensor_path)
        if sensor.tag != 'sensor':
            raise ValueError(f'Expected sensor root in {sensor_path}')
        xml_compatibility += normalize_sensor_xml(sensor)
        audit(sensor, f'/metParams/sensorName/{sensor_path.name}')
        sensors.append(sensor)
    xml_compatibility += tuple(ignored_xml)
    if not sensors and hrrr_node is None and not wrf_file:
        raise ValueError('At least one meteorological sensor is required')
    if sensors and hrrr_node is not None:
        raise ValueError('HRRR input generates the stations and requires an empty metParams sensor list')
    if any(int(s.findtext('site_coord_flag', '1')) not in (1, 2, 3) for s in sensors):
        unsupported.append('sensor coordinate mode other than local, UTM or geographic')
    measurements = [sensor.findall('timeSeries') for sensor in sensors]
    if any(not records for records in measurements):
        raise ValueError('Each sensor requires at least one timeSeries')
    for ts in (record for records in measurements for record in records):
        kind = int(meteorology_option(ts, 'boundaryLayerFlag', '1'))
        if kind not in (0, 1, 2, 3, 4):
            unsupported.append('sensor profile '+ts.findtext('boundaryLayerFlag'))
    # Original building geometry and parameterizations run on the CPU even
    # when the pressure solver and meteorology use CUDA.
    native_building_math = native_building_math or legacy_turbulence_xml
    bp = root.find('buildingsParams')
    # Native geometry comes from the elements and GIS records. The
    # numBuildings and numPolygonNodes fields are informational.
    building_nodes = bp.findall('rectangularBuilding') if bp is not None else []
    polygon_nodes = bp.findall('QUICBuilding') if bp is not None else []
    upwind_mode = wake_mode = rooftop_mode = sidewall_mode = street_canyon_mode = 0
    if bp is not None:
        # Native dispatch selects positive upwind and wake flags. Its upwind
        # closure reserves 1 and 3 for Rockle and HMVP and uses MVP otherwise.
        upwind_flag = int(bp.findtext('upwindCavityFlag', '0'))
        upwind_mode = upwind_flag if upwind_flag == 3 else min(2, max(0, upwind_flag))
        wake_mode = max(0, int(bp.findtext('wakeFlag', '0')))
        sidewall_mode = max(0, int(bp.findtext('sidewallFlag', '0')))
        rooftop_flag = int(bp.findtext('rooftopFlag', '0'))
        rooftop_mode = rooftop_flag if rooftop_flag in (1, 2) else 0
        canyon_flag = int(bp.findtext('streetCanyonFlag', '0'))
        street_canyon_mode = canyon_flag if canyon_flag in (1, 2) else 0
    tp = root.find('turbParams') if compute_turbulence else None
    if tp is not None and legacy_turbulence_xml:
        xml_compatibility += normalize_legacy_turbulence(tp, cutcell=mesh_type == 1)
    if tp is None and turbulence_gradient != 'cpu':
        raise ValueError('Selecting a turbulence gradient backend requires active turbParams')
    mixing_method = int(tp.findtext('method', '0')) if tp is not None else 0
    mixing_samples = int(tp.findtext('samples', '2000')) if mixing_method == 3 else 2000
    mixing_ray_backend = tp.findtext('rayBackend', 'cpu').strip() if tp is not None else 'cpu'
    if mixing_ray_backend not in ('cpu', 'cuda', 'optix') or (mixing_ray_backend != 'cpu' and mixing_method != 3):
        raise ValueError('turbParams/rayBackend must be cpu, or cuda or optix with mixing method 3')
    if tp is not None:
        if mixing_method not in (0, 1, 2, 3, 4):
            unsupported.append('turbParams/method='+tp.findtext('method').strip())
        if mixing_samples < 1:
            raise ValueError('turbParams/samples must be positive for method 3')
        if tp.findtext('nonLocalMixing', 'false').strip().lower() not in ('true', 'false', '0', '1'):
            raise ValueError('Invalid nonLocalMixing boolean')
        if (mixing_method == 4 or tp.findtext('save', 'false').strip().lower() in ['true', '1']) and not tp.findtext('LMfile', '').strip():
            raise ValueError('LMfile is required for file-based or saved mixing length')
    if unsupported:
        raise UnsupportedFeatures(unsupported)
    # Native selection happens only when the visualization output is opened.
    visualization_fields = read_output_fields(root.find('fileOptions'), validate_names=not legacy_turbulence_xml)
    stretching = simulation_option('verticalStretching', 0)
    if not legacy_turbulence_xml and stretching not in (0, 1):
        raise UnsupportedFeatures([f'verticalStretching={stretching}'])
    # The native Domain constructors always use cellSize; the XML switch
    # remains active only in the original turbulence derivative stencils.
    widths = parse_vector(sim, 'dz_value', repeated=True) if stretching and not legacy_turbulence_xml else None
    wrf = wrf_exchange = None
    if wrf_file:
        wrf_file = str(paths.resolve(wrf_file))
    if wrf_only:
        from .wrf import FireMeshInput
        timeout = 60. if wrf_timeout is None else wrf_timeout
        if live_wrf and wrf_startup_only:
            from .wrf_exchange import read_wrf_startup
            wrf = read_wrf_startup(wrf_file, timeout=timeout)
        elif live_wrf:
            from .wrf_exchange import WRFExchange
            wrf_exchange = WRFExchange.open(wrf_file, step_limit, timeout=timeout, legacy_protocol=legacy_wrf_protocol)
            wrf = wrf_exchange.grid
        else:
            wrf = FireMeshInput.read(wrf_file)
        domain = wrf.domain(halo, widths)
    else:
        # Native parsing stops on extraction failure, then checks vector length.
        cells = parse_vector(sim, 'domain', convert=int, size=3, stop_on_invalid=legacy_turbulence_xml)
        spacing = parse_vector(sim, 'cellSize', size=3, stop_on_invalid=legacy_turbulence_xml)
        domain = Domain(cells, spacing, widths)
    def parse_profile(node):
        kind = int(meteorology_option(node, 'boundaryLayerFlag', '1'))
        if kind == 4:
            from .measured_profile import MeasuredProfile
            return MeasuredProfile(*(tuple(float(value) for item in node.findall(k) for value in (item.text or '').split()) for k in ['height', 'speed', 'direction']),
                                   roughness=float(node.findtext('siteZ0', '.02')), reciprocal_length=float(node.findtext('reciprocal', '0')))
        # TimeSeries reads vectors even for single-reference profiles. Their
        # native equations use the first entry of each vector independently.
        reference_values = []
        for key in ('speed', 'direction', 'height'):
            values = [float(item.text or '') for item in node.findall(key)]
            if not values:
                raise ValueError(f'Missing timeSeries/{key}')
            reference_values.append(values[0])
        return SensorProfile(*reference_values, float(node.findtext('siteZ0', '.02')), float(node.findtext('reciprocal', '0')),
              kind=kind, canopy_height=float(node.findtext('canopyHeight', '0')),
              canopy_attenuation=float(node.findtext('attenuationCoefficient', '0')))
    reference_profile = parse_profile(measurements[0][0]) if measurements else None
    utc_at_read = QEStime()
    series = {}
    first_sensor_times = []
    initial_stations = []
    utm_origin = np.array([sim.findtext(key, '0') for key in ('UTMx', 'UTMy')], dtype=np.float32)
    utm_zone = int(sim.findtext('UTMZone', '0'))
    utm_band = sim.findtext('UTMZoneLetter', '')
    reproject = sim.findtext('reprojectShapefiles', 'false').strip().lower()
    if reproject not in ('0', '1', 'false', 'true'):
        raise ValueError('Invalid reprojectShapefiles boolean')
    reproject_dem = sim.findtext('reprojectDEM', 'false').strip().lower()
    if reproject_dem not in ('0', '1', 'false', 'true'):
        raise ValueError('Invalid reprojectDEM boolean')
    reproject_vectors = reproject in ('1', 'true')
    reproject_dem = reproject_dem in ('1', 'true')
    if reproject_dem and not dem_file:
        raise ValueError('reprojectDEM requires a DEM input')
    shared_epsg = None
    if reproject_vectors or reproject_dem:
        from .georeference import utm_epsg
        shared_epsg = utm_epsg(utm_zone, utm_band)
    target_epsg = shared_epsg if reproject_vectors else None
    for sensor_id, (sensor, records) in enumerate(zip(sensors, measurements)):
        coordinate_mode = int(sensor.findtext('site_coord_flag', '1'))
        # The legacy geographic mode uses explicitly supplied local pairs as
        # already positioned. The geographic-only extension projects them.
        legacy_local = legacy_turbulence_xml and all(
            (sensor.findtext(key) or '').strip() for key in ('site_xcoord', 'site_ycoord'))
        if coordinate_mode == 3 and not legacy_local:
            from .georeference import geographic_station
            longitude, latitude = (float(sensor.findtext(key, 'nan')) for key in ('site_lon', 'site_lat'))
            projected = geographic_station(longitude, latitude, utm_origin, utm_zone, utm_band)
            x, y = halo.translate(tuple(map(float, projected)))
        elif coordinate_mode == 2 and np.all(utm_origin != 0):
            position = np.array([sensor.findtext(key) for key in ('site_UTM_x', 'site_UTM_y')], dtype=np.float32)
            x, y = map(float, position-utm_origin)
        else:
            x, y = (float(sensor.findtext(key, 'nan')) for key in ('site_xcoord', 'site_ycoord'))
            if coordinate_mode == 1:
                x, y = halo.translate((x, y))
        if not np.isfinite([x, y]).all():
            raise ValueError('Sensor coordinates must be finite and use the configured coordinate mode')
        for record in records:
            station = Station(x, y, parse_profile(record))
            if record is records[0]:
                initial_stations.append(station)
            stamp = record.findtext('timeStamp', '').strip()
            epoch = int(meteorology_option(record, 'timeEpoch', '-1'))
            when = QEStime(stamp) if stamp else None
            if epoch != -1:
                epoch_time = QEStime(epoch)
                if when is not None and when != epoch_time:
                    raise ValueError('Sensor timeStamp and timeEpoch disagree')
                when = epoch_time
            timestamp = (when if when is not None else utc_at_read).get_timestamp()
            if sensor_id == 0:
                first_sensor_times.append(timestamp)
            # QES scans all entries and retains the last same-time record.
            series.setdefault(timestamp, {})[sensor_id] = station
    # QES seeds its timeline with every first-sensor record. It preserves
    # duplicates there and adds only new timestamps from the other sensors.
    hrrr = None
    hrrr_grid_ids = ()
    hrrr_nearest_map_positions = ()
    if wrf_only:
        reference_profile = reference_profile or SensorProfile(0., 270., 10., .1)
        schedule = (MeteorologyStep('2022-01-01T00:00:00', ()),)
    elif wrf_file:
        from .wrf import atmospheric_stations
        initial_stations = atmospheric_stations(wrf_file, domain, halo, tuple(map(float, utm_origin)),
            utm_zone, int(meteorology_option(sim, 'WRFSensorSample', '1')), coordinate_mode=wrf_coordinates,
            stored=legacy_turbulence_xml)
        reference_profile = initial_stations[0].profile
        schedule = (MeteorologyStep('1970-01-01T00:00:00', initial_stations),)
    elif hrrr_node is not None:
        from .hrrr import HRRRInput
        filename = hrrr_node.findtext('HRRRFile', '').strip()
        if not filename:
            raise ValueError('HRRRFile is required')
        hrrr = HRRRInput(str(paths.resolve(filename)),
            int(meteorology_option(hrrr_node, 'interpolationScheme', '0')),
            int(meteorology_option(hrrr_node, 'stabilityClasses', '0')),
            tuple(map(float, utm_origin)), utm_zone)
        gpu_barnes = meteorology_backend == 'cuda' and hrrr.scheme == 0
        steps, initial_stations, hrrr_grid_ids = hrrr.stations(
            domain, halo, step_limit, native_math=native_sensor_math or legacy_turbulence_xml or gpu_barnes,
            stored=legacy_turbulence_xml)
        if hrrr.scheme == 1:
            hrrr_nearest_map_positions = tuple(station.map_position for station in initial_stations)
        schedule = tuple(MeteorologyStep(timestamp, stations) for timestamp, stations in steps)
        reference_profile = initial_stations[0].profile
    else:
        timestamps = sorted(first_sensor_times + list(series.keys()-set(first_sensor_times)))
        schedule = tuple(MeteorologyStep(timestamp, tuple(series[timestamp].values()))
                         for timestamp in timestamps[:step_limit or None])
    timestamp, stations = schedule[0].timestamp, schedule[0].stations
    profile = stations[0].profile if stations else reference_profile
    buildings = []
    if bp is not None and bp.findtext('SHPFile'):
        from .gis import default_building_layer, read_building_shapefile, resolve_vector_path
        shp_path = resolve_vector_path(paths, bp.findtext('SHPFile').strip())
        layer = bp.findtext('SHPBuildingLayer', default_building_layer(shp_path))
        gis_buildings = read_building_shapefile(shp_path, height_field=bp.findtext('SHPHeightField', 'H').strip(),
            height_factor=float(bp.findtext('heightFactor', '1')), layer=None if layer is None else layer.strip(),
            origin=utm_origin, target_epsg=target_epsg,
            native_math=legacy_turbulence_xml or native_building_math, legacy_ogr=legacy_turbulence_xml)
        # The native GIS loader samples DEM heights before moving the footprint
        # by the halo. Explicit XML buildings sample at the moved coordinates.
        if halo.x or halo.y:
            gis_buildings = tuple(replace(building, terrain_sample_vertices=building.vertices)
                                  for building in gis_buildings)
        buildings.extend(gis_buildings)
    rectangle_fields = ('xStart', 'yStart', 'length', 'width', 'height', 'buildingRotation', 'baseHeight')
    # Thin cut faces amplify a one-ULP vertex error when computing wall normals.
    native_rectangle = mesh_type == 1 or native_wake_math or native_building_math
    buildings.extend(PolygonBuilding.rectangle(*(float(node.findtext(k)) for k in rectangle_fields),
                                               native_math=native_rectangle) for node in building_nodes)
    for node in polygon_nodes:
        xx = np.array([v for item in node.findall('xVertex') for v in (item.text or '').split()], dtype=np.float32)
        yy = np.array([v for item in node.findall('yVertex') for v in (item.text or '').split()], dtype=np.float32)
        if xx.shape != yy.shape or len(xx) < 3:
            raise ValueError('QUIC polygon vertices require at least three paired coordinates')
        height, base = float(node.findtext('height')), float(node.findtext('baseHeight'))
        vertices = np.c_[xx, yy]
        parts = vertex_parts(vertices)
        native_polygon = legacy_turbulence_xml or native_building_math
        if native_polygon and (len(parts) > 1 or parts[0][1]):
            (exterior, holes), *detached_parts = parts
            buildings.append(PolygonBuilding(exterior, height, base, holes,
                                            native_closure_vertices=tuple(map(tuple, vertices))+(tuple(vertices[0]),),
                                            detached_parts=tuple(detached_parts)))
            continue
        if native_polygon and np.array_equal(vertices[0], vertices[-1]):
            # PolygonQUICBuilding appends the first XML vertex even when the
            # supplied ring is already closed. The extra vertex affects wakes.
            exterior = tuple(map(tuple, vertices))
            parts = ((exterior+(exterior[0],), ()),)
        buildings.extend(PolygonBuilding(exterior, height, base, holes) for exterior, holes in parts)
    buildings = tuple(translate_footprint(building, halo) for building in buildings)
    model = None
    if tp is not None:
        explicit_sigma = tp.find('sigmaConst') is not None
        sigma = (parse_vector(tp, 'sigmaConst', size=3, stop_on_invalid=legacy_turbulence_xml)
                 if explicit_sigma else (2.5, 2., 1.3))
        building_wall = parse_optional_number(tp, 'buildingWallFlag', 2, convert=int, legacy=legacy_turbulence_xml)
        terrain_wall = parse_optional_number(tp, 'terrainWallFlag', 2, convert=int, legacy=legacy_turbulence_xml)
        if mesh_type == 1:
            # Both native cut-cell wall constructors select finite differences.
            building_wall = terrain_wall = 2
        model = LocalTurbulence(gradient_backend=turbulence_gradient, native_math=native_turbulence_math,
             force_nonuniform_z=legacy_turbulence_xml and stretching > 0,
             native_zero_mixing_nan=legacy_turbulence_xml and (mesh_type == 1 or coefficient_flag not in (0, 1)),
             explicit_sigma=explicit_sigma, sigma=sigma,
             upper_bound=parse_optional_number(tp, 'turbUpperBound', 20., convert=float, legacy=legacy_turbulence_xml),
             background=parse_optional_number(tp, 'backgroundMixing', 0., convert=float, legacy=legacy_turbulence_xml),
             building_wall=building_wall, terrain_wall=terrain_wall,
             wall_roughness=float(bp.findtext('wallRoughness', '.01')) if bp is not None else .01,
             terrain_roughness=reference_profile.roughness,
             nonlocal_mixing=tp.findtext('nonLocalMixing', 'false').strip().lower() in ('true', '1'))
    if poisson is None:
        from .contracts import NumericPolicy
        policy_type = NativeWindPolicy if poisson_backend in NATIVE_WIND_SOLVERS else NumericPolicy
        policy = policy_type(relative_tolerance=simulation_option('tolerance', policy_type.relative_tolerance),
                             max_iterations=simulation_option('maxIterations', policy_type.max_iterations))
        poisson = make_poisson(poisson_backend, policy, native_cuda_boundary=legacy_turbulence_xml)
    mixing_file = None
    if tp is not None and tp.findtext('LMfile', '').strip():
        mixing_file = Path(tp.findtext('LMfile').strip())
        if not mixing_file.is_absolute():
            mixing_file = paths.resolve(mixing_file, require_parent=mixing_method == 4)
        mixing_file = str(mixing_file)
    terrain = wrf
    vegetation = None
    vp = root.find('vegetationParams')
    if vp is not None:
        from .vegetation import HomogeneousCanopy,Vegetation
        from .canopy_tree import IsolatedTree
        from .canopy_windbreak import Windbreak
        from .canopy_roc import RowCanopy
        canopies = []
        if vp.findtext('SHPFile', '').strip():
            from .gis import read_tree_shapefile, resolve_vector_path
            shp_path = resolve_vector_path(paths, vp.findtext('SHPFile').strip())
            canopies.extend(read_tree_shapefile(shp_path, layer=vp.findtext('SHPTreeLayer', 'trees').strip(),
                origin=utm_origin, target_epsg=target_epsg, legacy_ogr=legacy_turbulence_xml))
        for node in vp.findall('Homogeneous'):
            parameters = dict(height=float(node.findtext('height')), attenuation=float(node.findtext('attenuationCoefficient')), base_height=float(node.findtext('baseHeight')))
            if float(node.findtext('length', '0')) > 0 and float(node.findtext('width', '0')) > 0:
                canopy = HomogeneousCanopy.rectangle(*(float(node.findtext(key, '0')) for key in ['xStart', 'yStart', 'length', 'width']),
                    **parameters, rotation=float(node.findtext('canopyRotation', '0')))
            else:
                xx, yy = [np.array([float(v) for item in node.findall(key) for v in (item.text or '').split()], dtype=np.float32) for key in ['xVertex', 'yVertex']]
                if len(xx) < 3 or xx.shape != yy.shape:
                    raise ValueError('Canopy polygon requires at least three paired coordinates')
                canopies.extend(HomogeneousCanopy(exterior, holes=holes, **parameters)
                                for exterior, holes in vertex_parts(np.c_[xx, yy]))
                continue
            canopies.append(canopy)
        for node in vp.findall('IsolatedTree'):
            keys = ('xCenter', 'yCenter', 'width', 'height', 'attenuationCoefficient', 'zMaxLAI', 'baseHeight')
            canopies.append(IsolatedTree.circle(*(float(node.findtext(key)) for key in keys)))
        for node in vp.findall('Windbreak'):
            canopies.append(Windbreak.rectangle(*(float(node.findtext(key)) for key in ['xStart', 'yStart', 'length', 'width', 'height', 'opticalPorosity', 'understroyHeight', 'canopyRotation']),
                base_height=float(node.findtext('baseHeight', '0')), model=int(node.findtext('fenceModel', '2')), fetch=float(node.findtext('fetch', '7'))))
        for node in vp.findall('ROC'):
            xx, yy = [np.array([float(v) for item in node.findall(key) for v in (item.text or '').split()],
                               dtype=np.float32) for key in ('xVertex', 'yVertex')]
            if xx.shape != (4,) or yy.shape != (4,):
                raise ValueError('ROC requires four paired coordinates')
            vertices = np.c_[xx, yy]
            center = np.asarray(halo.translate(vertices)).mean(axis=0)
            sensor = min(initial_stations, key=lambda s: (s.x-center[0])**2+(s.y-center[1])**2).profile
            thin = node.findtext('thinFence', '').strip().lower()
            if thin not in ('0', '1', 'true', 'false'):
                raise ValueError('Invalid ROC thinFence boolean')
            names = dict(height='height', understory='understoryHeight', row_spacing='rowSpacing',
                         row_width='rowWidth', row_angle='rowAngle', lad_effective='LAD_eff',
                         lad_average='LAD_avg', porosity='opticalPorosity', tke_max='tkeMax',
                         upstream_ustar='upstreamUstar')
            parameters = {name: float(np.float32(node.findtext(key))) for name, key in names.items()}
            canopies.append(RowCanopy(tuple(map(tuple, np.r_[vertices, vertices[:1]].astype(float))),
                sigma_w=float(node.findtext('upstreamSigmaW', '0')),
                displacement=float(node.findtext('displacementHeightParallel', '0')),
                reciprocal=sensor.reciprocal_length, roughness=sensor.roughness,
                thin_fence=thin in ('1', 'true'), base_height=float(node.findtext('baseHeight', '0')), **parameters))
        wake_enabled = int(vp.findtext('wakeFlag', '1')) == 1
        vegetation = Vegetation(tuple(translate_footprint(canopy, halo) for canopy in canopies), wake_enabled)
    if dem_file:
        from .terrain import DEMTerrain
        terrain = DEMTerrain.from_input(paths, dem_file, origin_flag=simulation_option('originFlag', 0),
            utm_x=float(sim.findtext('UTMx', '0')), utm_y=float(sim.findtext('UTMy', '0')),
            distance_x=simulation_option('DEMDistancex', 0.), distance_y=simulation_option('DEMDistancey', 0.),
            target_epsg=shared_epsg if reproject_dem else None)
    building_roughness = float(bp.findtext('wallRoughness', '.01')) if bp is not None else .01
    wall_log_roughness = building_roughness if bp is not None else (.1 if roughness_split else reference_profile.roughness)
    return WindCase(domain, profile, buildings, upwind_mode, model, poisson,
                    wake_mode=wake_mode, rooftop_mode=rooftop_mode, sidewall_mode=sidewall_mode,
                    high_rise=bp is not None and int(bp.findtext('highRiseFlag', '0')) == 1,
                    wall_roughness=building_roughness,
                    stations=stations, street_canyon_mode=street_canyon_mode,
                    timestamp=timestamp, mixing_method=mixing_method, mixing_samples=mixing_samples,
                    mixing_ray_backend=mixing_ray_backend,
                    mixing_file=mixing_file, mixing_variable=tp.findtext('varname', 'mixlength').strip() if tp is not None else 'mixlength',
                    save_mixing=tp.findtext('save', 'false').strip().lower() in ['true', '1'] if tp is not None else False,
                    terrain=terrain, vegetation=vegetation,
                    schedule=schedule if len(schedule) > 1 else (), reference_roughness=reference_profile.roughness,
                    coefficient_file=coefficient_file, coefficient_flag=coefficient_flag,
                    roughness_split=roughness_split, halo=halo, mesh_type=mesh_type,
                    hrrr=hrrr, hrrr_grid_ids=hrrr_grid_ids, hrrr_nearest_map_positions=hrrr_nearest_map_positions,
                    wrf=wrf, native_sensor_math=native_sensor_math,
                    native_wake_math=native_wake_math, native_building_math=native_building_math,
                    wrf_atmosphere_path=wrf_file if wrf_file and not wrf_only else None, wrf_exchange=wrf_exchange,
                    wrf_atmosphere_coordinates=wrf_coordinates if wrf_file and not wrf_only else None,
                    visualization_fields=visualization_fields,
                    wall_log=WallLogCorrection(wall_log_roughness) if wall_log_enabled else None,
                    domain_rotation_deg=domain_rotation_deg,
                    meteorology_backend=meteorology_backend,
                    meteorology_variant='native-gpu' if meteorology_backend == 'cuda' else 'native-cpu',
                    native_netcdf_inputs=legacy_turbulence_xml, allow_unset_wake=True,
                    xml_compatibility=xml_compatibility,
                    street_intersection_flag=int(bp.findtext('streetIntersectionFlag', '0')) if bp is not None else 0,
                    wrf_zero_steps=legacy_wrf_protocol and wrf_only and step_limit == 0)
