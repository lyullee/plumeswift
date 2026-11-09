# PlumeSwift module index

Signatures follow the Python source at this release. See the [README input/output table](../README.md#모듈과-입출력) for units and [the guide](USER_GUIDE.md) for workflows. Unannotated return types are described in the linked implementations. This includes internal compatibility modules that may change.

### `plumeswift.__init__`

[Source](../python_qes/plumeswift/__init__.py)

Python QES port in progress: consult port_inventory.json for actual coverage.


### `plumeswift.__main__`

[Source](../python_qes/plumeswift/__main__.py)

Small inspection/property CLI, not a fake replacement for all native flags.

- `main(argv=None, *, launcher=None)`
- `qes()`
- `qes_winds()`
- `qes_plume()`
- `qes_fire()`

### `plumeswift.analytical_cases`

[Source](../python_qes/plumeswift/analytical_cases.py)

Prescribed continuous-release test cases from QES testCases (GPL-3.0).

- Class `Case`
  - Field `stem: str`
  - Field `lengths: tuple[int, int, int]`
  - Field `dz: float`
  - Field `source: tuple[int, int, int]`
  - Field `lower: tuple[int, int, int]`
  - Field `upper: tuple[int, int, int]`
  - Field `boxes: tuple[int, int, int]`
  - Field `duration: int`
  - Field `average_start: int`
  - Field `courant: float`
  - Field `wall: str`
- `prepare_analytical_case(name, output, *, steps=None, particles_per_step=200)`
- `run_analytical_case(name, output, *, prepare_only=False, steps=None, particles_per_step=200, seed=0)`

### `plumeswift.analytical_results`

[Source](../python_qes/plumeswift/analytical_results.py)

Profile and error diagnostics from the QES continuous-release examples.

- `analyze_analytical_case(name, collection, *, record=-1, xml=None)`
- `write_analytical_results(name, collection, output, *, record=-1, xml=None, plot=False)`

### `plumeswift.argument_demo`

[Source](../python_qes/plumeswift/argument_demo.py)

Python counterpart to the public scratch/argparser.cpp utility.

- `argument_demo(arguments)`

### `plumeswift.assembly`

[Source](../python_qes/plumeswift/assembly.py)

Explicit dependency injection; imports do not construct models or read env vars.

- Class `Registry`
  - `__init__(self)`
  - `register(self, name: str, factory: Callable)`
  - `create(self, name: str, **kwargs)`
  - `names(self)` → tuple[str, ...]
- Class `ComponentChoice`
  - Field `implementation: str`
  - Field `parameters: dict[str, Any] = field(default_factory=dict)`
- Class `AssemblyConfig`
  - Field `thermodynamics: ComponentChoice`
  - Field `poisson_solver: ComponentChoice`
  - Field `numeric: NumericPolicy = field(default_factory=NumericPolicy)`
  - Field `physics: PhysicalCoefficients = field(default_factory=PhysicalCoefficients)`
  - Field `seed: int = 20261107`
- Class `Components`
  - Field `config: AssemblyConfig`
  - Field `thermo: Any`
  - Field `poisson: Any`
  - `manifest(self)`
- `default_registry()`
- `assemble(config: AssemblyConfig, registry: Registry | None=None)` → Components

### `plumeswift.boundaries`

[Source](../python_qes/plumeswift/boundaries.py)

Domain boundary policies, independent of particle integration and physics.

- Class `AxisBoundary`
  - Field `lower: float`
  - Field `upper: float`
  - Field `mode: str = 'exiting'`
  - `apply(self, position, fluctuation)`
- Class `DomainBoundaries`
  - `__init__(self, axes)`
  - `apply(self, positions, fluctuations)`
- Class `NativeAxisBoundary`
  - `apply(self, position, fluctuation)`
- Class `NativeDomainBoundaries`
  - `apply(self, positions, fluctuations, *, active=True)`

### `plumeswift.building_frame`

[Source](../python_qes/plumeswift/building_frame.py)

Shared geometric state for the original building parameterizations.

- `face_angles(local, *, native_math=False, wrap_slack=0.0)`
- Class `BuildingFrame`
  - Field `vertices: np.ndarray`
  - Field `center: np.ndarray`
  - Field `local: np.ndarray`
  - Field `direction: float`
  - Field `reference: tuple[float, float]`
  - Field `k_start: int`
  - Field `k_end: int`
  - Field `horizontal_bounds: tuple[int, int, int, int]`
  - Field `width: float`
  - Field `length: float`
  - Field `rectangular: bool`
  - `rooftop_faces(self, *, wrap_slack=0.0, native_math=False)`
  - `create(cls, workspace, building, *, native_math=False, native_dy_alias=False)`

### `plumeswift.building_pipeline`

[Source](../python_qes/plumeswift/building_pipeline.py)

Replaceable building closures with explicit native call order and state.

- Class `BuildingClosure`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `BuildingContext`
  - Field `wakes: dict = field(default_factory=dict)`
  - Field `max_velocity: float = 0.0`
  - Field `ordered_buildings: tuple = ()`
  - Field `allow_unset_wake: bool = False`
  - `wake_state(self, number, error)`
- Class `UpwindClosure`
  - Field `mode: int = 2`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `WakeClosure`
  - Field `mode: int = 2`
  - Field `rooftop_mode: int = 2`
  - Field `high_rise: bool = False`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `SidewallClosure`
  - Field `roughness: float = 0.01`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `RooftopClosure`
  - Field `mode: int = 2`
  - Field `roughness: float = 0.01`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `StreetCanyonClosure`
  - Field `roughness: float = 0.01`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `ModifiedStreetCanyonClosure`
  - Field `roughness: float = 0.01`
  - Field `native_math: bool = False`
  - `apply(self, workspace, ordered_buildings, context)`
- Class `BuildingPipeline`
  - Field `stages: tuple[BuildingClosure, ...]`
  - Field `allow_unset_wake: bool = False`
  - `apply(self, workspace, ordered_buildings)`
  - `native_order(cls, upwind=2, wake=2, sidewall=1, rooftop=2, high_rise=False, roughness=0.01, street_canyon=0, *, native_wake_math=False, native_building_math=False, allow_unset_wake=False)`

### `plumeswift.canopy_cionco`

[Source](../python_qes/plumeswift/canopy_cionco.py)

Legacy domain-wide Cionco canopy routine from Canopy.cpp (GPL-3).

- `apply_cionco(workspace, state)`

### `plumeswift.canopy_roc`

[Source](../python_qes/plumeswift/canopy_roc.py)

Row-organized canopy equations from QES CanopyROC.cpp (GPL-3).

- `stability(height, reciprocal)`
- Class `RowCanopy`
  - Field `vertices: tuple`
  - Field `height: float`
  - Field `understory: float`
  - Field `row_spacing: float`
  - Field `row_width: float`
  - Field `row_angle: float`
  - Field `lad_effective: float`
  - Field `lad_average: float`
  - Field `porosity: float`
  - Field `tke_max: float`
  - Field `sigma_w: float`
  - Field `upstream_ustar: float`
  - Field `displacement: float`
  - Field `reciprocal: float`
  - Field `roughness: float`
  - Field `thin_fence: bool = False`
  - Field `base_height: float = 0.0`
  - `flag(self, workspace, state, number)`
  - `apply_rows(self, workspace, state)`
  - `apply_wake(self, workspace, state, number, _defects)`
- Class `RowFlow`
  - `__init__(self, canopy, shape)`
  - `signed(self, x, y)`
  - `cross(self, u, v)`
  - `along(self, u, v)`
  - `rotate(self, u, v)`
  - `unrotate(self, u, v)`
  - `row_offset(self, distance)`
  - `shelter(self, below_mid, relative, rows, stream, local, *, all_entry_rows=False)`
  - `set_parallel_reference(self, reference)`
  - `parallel_profile(self, relative)`
  - `displacement_zone(self, z, gap, cross, flags)`
  - `set_tke(self, index, model_cross, model_parallel, cross, parallel)`

### `plumeswift.canopy_roc_turbulence`

[Source](../python_qes/plumeswift/canopy_roc_turbulence.py)

Row canopy turbulent viscosity and canopy-top stress.

- `apply_viscosity(workspace, flow, mixing, energy, viscosity)`
- `apply_stress(workspace, flow, velocity, energy, viscosity, stress)`

### `plumeswift.canopy_tree`

[Source](../python_qes/plumeswift/canopy_tree.py)

QES isolated tree attenuation and elliptical wake (GPL-3).

- Class `IsolatedTree`
  - Field `vertices: tuple[tuple[float, float], ...]`
  - Field `height: float`
  - Field `leaf_area_index: float`
  - Field `peak_fraction: float`
  - Field `base_height: float = 0.0`
  - Field `holes: tuple[tuple[tuple[float, float], ...], ...] = ()`
  - `circle(cls, x, y, width, height, attenuation, peak_fraction, base_height=0.0)`
  - `flag(self, workspace, state, number)`
  - `apply(self, workspace, state, columns)`
  - `center(self, dx)`
  - `wake_reference(self, workspace, top, columns=())`
  - `apply_wake(self, workspace, state, number, defects)`

### `plumeswift.canopy_windbreak`

[Source](../python_qes/plumeswift/canopy_windbreak.py)

QES Guan porous windbreak and wake, CanopyWindbreak.cpp (GPL-3).

- Class `Windbreak`
  - Field `vertices: tuple[tuple[float, float], ...]`
  - Field `height: float`
  - Field `porosity: float`
  - Field `understory_height: float`
  - Field `base_height: float = 0.0`
  - Field `model: int = 2`
  - Field `fetch: float = 7.0`
  - `rectangle(cls, x, y, length, width, height, porosity, understory_height, rotation, base_height=0.0, model=2, fetch=7.0)`
  - `aerodynamic_porosity(self)`
  - `flag(self, workspace, state, number)`
  - `apply(self, workspace, state, columns)`
  - `apply_wake(self, workspace, state, number, defects)`

### `plumeswift.concentration_output`

[Source](../python_qes/plumeswift/concentration_output.py)

NetCDF collection output shared by plume and fire smoke (GPL-3.0).

- `attach_concentration(output, coordinates, read, *, count_field='p_count')`
- `native_collection_series(path, config, origin)`

### `plumeswift.contracts`

[Source](../python_qes/plumeswift/contracts.py)

Small, SI-only contracts between independently replaceable components.

- Class `ModelDomainError`
- Class `UnportedFeature`
- Class `ThermoState`
  - Field `concentration: Array`
  - Field `temperature: Array`
  - Field `density: Array`
  - Field `volume_fraction: Array`
  - Field `vapor_quality: Array`
- Class `Thermodynamics`
  - Field `name: str`
  - Field `ambient_density: float`
  - `evaluate(self, concentration: Array)` → ThermoState
  - `describe(self)` → dict[str, Any]
- Class `PureFluidProperties`
  - Field `hydrogen_fluid: str`
  - `enthalpy(self, fluid: str, temperature: Array, pressure: float, phase: str | None=None)` → Array
  - `density(self, fluid: str, temperature: Array, pressure: float)` → Array
  - `saturation(self, fluid: str, pressure: float, quality: float)` → dict[str, float]
  - `describe(self)` → dict[str, Any]
- Class `MixtureRule`
  - `density_and_volume_fraction(self, hydrogen_mass_fraction: Array, rho_hydrogen: Array, rho_air: Array)` → tuple[Array, Array]
  - `describe(self)` → dict[str, Any]
- Class `WallReflection`
  - `reflect(self, positions: Array, displacement: Array, fluctuations: Array)` → tuple[Array, Array]
- Class `Observer`
  - `observe(self, time_start: float, dt: float, positions: Array, masses: Array)` → None
- Class `FieldProvider`
  - Field `dx: float`
  - Field `dy: float`
  - Field `zf: Array`
  - Field `mixlength: Array`
  - `sample_wind(self, positions: Array)` → Array
  - `cell_indices(self, positions: Array)` → tuple[Array, Array, Array]
  - `sample_turbulence(self, positions: Array)` → tuple[Array, Array, Array, Array | None]
  - `sample_turbulence_and_distance(self, positions: Array)` → tuple[Array, Array, Array, Array | None, Array]
- Class `SolverReport`
  - Field `converged: bool`
  - Field `iterations: int`
  - Field `residual_linf: float`
  - Field `update_linf: float | None`
  - Field `wall_s: float`
  - Field `method: str`
- Class `PoissonSolver`
  - `solve(self, stencil: Any, rhs: Array, initial: Array | None=None)` → tuple[Array, SolverReport]
- Class `NumericPolicy`
  - Field `relative_tolerance: float = 1e-07`
  - Field `absolute_tolerance: float = 1e-09`
  - Field `max_iterations: int = 10000`
  - Field `stress_invariant_tolerance: float = 1e-05`
  - Field `minimum_iterations: ClassVar[int] = 1`
- Class `PhysicalCoefficients`
  - Field `alpha_horizontal: float = 1.0`
  - Field `alpha_vertical: float = 1.0`
  - Field `gravity: float = 9.80665`
  - Field `lh2_drag_coefficient: float = 0.5`
  - Field `lh2_drag_length_m: float = 1.0`
  - Field `lh2_ground_reaction_height_m: float = 0.5`
- Class `DiagnosticLog`
  - Field `events: list[dict[str, Any]] = field(default_factory=list)`
  - `record(self, code: str, **values)`

### `plumeswift.cpu_plume_random`

[Source](../python_qes/plumeswift/cpu_plume_random.py)

Linux libstdc++ streams for QES Random and RandomSingleton (GPL-3).

- `resolve_cpu_seed(seed)`
- Class `LinuxQESRandom`
  - `__init__(self, seed=None)`
  - `random(self, size=None)`
  - `normal(self, size=None)`
- Class `NativeCPUStreams`
  - `__init__(self, seed, workers)`
  - `owners(self, indices, count)`
  - `initial_draws(self, ordinals, count)`

### `plumeswift.cuda_barnes`

[Source](../python_qes/plumeswift/cuda_barnes.py)

Python CUDA implementation of the QES two-pass Barnes wind profiler.

- `barnes_cuda(workspace, stations, *, sampled=None, device=0)`

### `plumeswift.cuda_fire_potential`

[Source](../python_qes/plumeswift/cuda_fire_potential.py)

CUDA implementation of the QES-Fire induced wind kernel (GPL-3).

- Class `CudaHeatPotential`
  - `apply(self, fire, workspace)`

### `plumeswift.cuda_gradient`

[Source](../python_qes/plumeswift/cuda_gradient.py)

Fused Python CUDA kernel for the nine staggered velocity derivatives.

- `cuda_velocity_gradient(workspace, velocity, fluid, *, device=0)`

### `plumeswift.cuda_particle_buffer`

[Source](../python_qes/plumeswift/cuda_particle_buffer.py)

Device particle storage, compaction and insertion for QES-Plume (GPL-3).

- Class `CudaParticleBuffer`
  - `__init__(self, capacity, *, device=0)`
  - `slot_counts(self)`
  - `compact_to(self, destination, *, clear_source=False)`
  - `insert_from(self, incoming, offset, count)`

### `plumeswift.cuda_partition`

[Source](../python_qes/plumeswift/cuda_partition.py)

Float32 pivot partition for the original CUDA diagnostic (GPL-3).

- `partition_cuda(values, pivot, *, device=0)`

### `plumeswift.cuda_plume_advection`

[Source](../python_qes/plumeswift/cuda_plume_advection.py)

Experimental QES CUDA plume Langevin solve and advection (GPL-3).

- `advect_particles(particles, draws, bounds, count=None, *, boundaries=('exiting',) * 3, historical_exiting=False)`

### `plumeswift.cuda_plume_boundaries`

[Source](../python_qes/plumeswift/cuda_plume_boundaries.py)

Axis policies for the experimental CUDA QES plume (GPL-3).

- `boundary_ids(modes)`
- `checked_bounds(particles, bounds, count)`
- `device_historical_exit()`
- `apply_historical_exiting(particles, bounds, count=None)`
- `device_boundary()`

### `plumeswift.cuda_plume_collection`

[Source](../python_qes/plumeswift/cuda_plume_collection.py)

QES experimental CUDA plume collection in Cartesian boxes (GPL-3).

- Class `CudaPlumeCollection`
  - `__init__(self, lower, widths, boxes, *, device=0)`
  - `collect(self, positions, states, dt)`
  - `copyback(self)`
  - `reset(self)`

### `plumeswift.cuda_plume_interpolation`

[Source](../python_qes/plumeswift/cuda_plume_interpolation.py)

Staggered wind and cell-centred turbulence interpolation for CUDA QES-Plume.

- Class `CudaPlumeInterpolation`
  - `__init__(self, velocity, turbulence, spacing, *, device=0)`
  - `wind(self)`
  - `turbulence(self)`
  - `tke(self)`
  - `uniform_fixture(cls, cell_shape, spacing, *, device=0)`
  - `load_fields(self, velocity, turbulence)`
  - `sample(self, particles, count=None, *, release=False)`

### `plumeswift.cuda_plume_random`

[Source](../python_qes/plumeswift/cuda_plume_random.py)

MTGP32 draws and named normal buffers for the CUDA plume (GPL-3).

- Class `CudaPlumeRandom`
  - `__init__(self, *, seed=1234, device=0)`
  - `create(self, key, length)`
  - `generate(self, key, mean=0.0, stddev=1.0)`
  - `uniform(self, size, *, signed=False)`
  - `uint32(self, size)`
  - `get(self, key)`
  - `destroy(self, key)`

### `plumeswift.cuda_plume_release`

[Source](../python_qes/plumeswift/cuda_plume_release.py)

Initialize and insert new experimental CUDA plume particles (GPL-3).

- `release_particles(destination, offset, count, interpolation, random, allocate_ids, *, position=(20.0, 50.0, 70.0))`

### `plumeswift.cuda_plume_run`

[Source](../python_qes/plumeswift/cuda_plume_run.py)

Standalone driver for the original experimental QES GPU plume stages.

- Class `CudaPlumeRun`
  - `__init__(self, interpolation, collection, bounds, capacity, *, seed=1234, boundaries=('exiting',) * 3)`
  - `step(self, release_count, *, position=(20.0, 50.0, 70.0), collect=True)`
- `run_cuda_plume(winds, turbulence, output, *, steps, release_count=None, capacity, source=None, boxes, seed=1234, record=0, collection_start=0, boundaries=('exiting',) * 3, wind_series=False, source_xml=None, source_model=None, source_seed=None, collection_bounds=None, transport_bounds=None)`

### `plumeswift.cuda_ray_mixing`

[Source](../python_qes/plumeswift/cuda_ray_mixing.py)

Python CUDA BVH traversal for the mixing-length ray distance queries.

- `cuda_ray_distances(mesh, points, directions, *, device=0)`

### `plumeswift.cuda_wind_fields`

[Source](../python_qes/plumeswift/cuda_wind_fields.py)

Device storage for staggered QES wind and cell-centred turbulence fields.

- Class `CudaWindFields`
  - `__init__(self, face_shape, *, device=0)`
  - `upload(self, velocity)`
  - `download(self, velocity)`
- Class `CudaTurbulenceFields`
  - `__init__(self, cell_shape, *, device=0)`

### `plumeswift.cutcell_building`

[Source](../python_qes/plumeswift/cutcell_building.py)

QES cut-cell openings for vertically extruded building polygons (GPL-3).

- `apply_cutcell_building(workspace, building, number, *, native_math=False, native_ring_joins=False)`

### `plumeswift.cutcell_building_walk`

[Source](../python_qes/plumeswift/cutcell_building_walk.py)

Legacy QES cut-cell edge traversal, adapted from CutBuilding.cpp (GPL-3.0-only).

- Class `CutColumn`
  - Field `reference: tuple`
  - Field `points: list = field(default_factory=list)`
  - Field `masks: list = field(default_factory=list)`
  - Field `has_height: bool = False`
  - Field `clear_roof: bool = False`
  - `add(self, samples, *, clear_roof=False)`
- `walk_columns(vertices, dx, nx, ny)`
- `apply_walk(workspace, vertices, number, dx, start, end, top, z, z_face, dz_array, *, ring_breaks=False)`

### `plumeswift.cutcell_terrain`

[Source](../python_qes/plumeswift/cutcell_terrain.py)

Terrain cut cells from native DEM corner heights (DTEHeightField, GPL-3).

- `apply_cutcell_terrain(workspace, corners)`

### `plumeswift.cutcell_walls`

[Source](../python_qes/plumeswift/cutcell_walls.py)

Post-process terrain/building cut-cell openings as in QES Wall.cpp (GPL-3).

- `cutcell_wall_stencil(workspace, *, native_math=False)`
- `define_cutcell_walls(workspace)`

### `plumeswift.data_transport`

[Source](../python_qes/plumeswift/data_transport.py)

Typed value and reference exchange for QES component factories.

- Class `QESDataTransport`
  - `__init__(self)`
  - `put(self, key, value, *, reference=False)`
  - `get(self, key, expected_type)`
  - `get_ref(self, key, expected_type)`
  - `contains(self, key)`
  - `clear(self)`

### `plumeswift.dem_projection`

[Source](../python_qes/plumeswift/dem_projection.py)

Resample a DEM on the explicit UTM model grid.

- `project_dem(raster, domain, epsg, origin)`

### `plumeswift.deposition`

[Source](../python_qes/plumeswift/deposition.py)

QES deposition constitutive equations in SI units (Deposition.cpp, GPL-3.0).

- Class `DepositionResult`
  - Field `remaining_kg: np.ndarray`
  - Field `deposited_kg: np.ndarray`
  - Field `cutoff_discard_kg: np.ndarray`
  - Field `inactive: np.ndarray`
- `vegetation_survival(displacement_m, velocity_m_s, density_kg_m3, diameter_m, stress, coeps)`
- `ground_survival(displacement_m, velocity_m_s, stress, eddy_viscosity_m2_s, settling_m_s, layer_width_m)`
- `apply_survival(mass_kg, survival, density_kg_m3, diameter_m)`
- Class `DepositionField`
  - `__init__(self, sampler, flags)`
  - `set_fields(self, sampler, flags)`
  - `record(self, cells, result)`
  - `apply(self, mass_kg, material, start, displacement, velocity, stress, coeps, viscosity, *, settling_m_s=0.0, record=None)`

### `plumeswift.domain`

[Source](../python_qes/plumeswift/domain.py)

Staggered QES domain, including the ground ghost cell (GPL-3.0-only).

- `native_vertical_coordinates(widths)`
- Class `HorizontalHalo`
  - Field `x: float = 0.0`
  - Field `y: float = 0.0`
  - `cells(self, spacing)`
  - `translate(self, coordinates)`
- Class `Domain`
  - Field `cells: tuple[int, int, int]`
  - Field `spacing: tuple[float, float, float]`
  - Field `vertical_widths: tuple[float, ...] | None = None`
  - `cell_shape(self)`
  - `face_shape(self)`
  - `interior(self)`
  - `workspace(self)`

### `plumeswift.epa_results`

[Source](../python_qes/plumeswift/epa_results.py)

EPA 7x11 array concentration sections from the original MATLAB workflow.

- `analyze_epa_results(collection, *, record=-1, xml=None)`
- `write_epa_results(collection, output, *, record=-1, xml=None, plot=False)`

### `plumeswift.fire_case`

[Source](../python_qes/plumeswift/fire_case.py)

Read native QES-Fire fuel and ignition configuration (GPL-3).

- Class `FireConfig`
  - Field `duration: float`
  - Field `fuel_type: int`
  - Field `moisture: float`
  - Field `cure: float`
  - Field `courant: float`
  - Field `ignitions: tuple[Ignition, ...]`
  - Field `fuel_file: Path | None = None`
  - Field `ignition_file: Path | None = None`
  - Field `native_ignition_input: bool = False`
  - `initialize(self, workspace, spacing, *, initialize_front=cpu_front, native_flame_sampling=False, native_fire_clock=False)`
- `read_fire_config(path, *, legacy_fire_xml=False)`

### `plumeswift.fire_front`

[Source](../python_qes/plumeswift/fire_front.py)

Exact initial QES fire-front distance in grid cells (GPL-3).

- `cpu_front(ignited)`
- `cuda_front(ignited)`

### `plumeswift.fire_fuels`

[Source](../python_qes/plumeswift/fire_fuels.py)

QES FuelProperties.hpp and FuelMap.cpp fuel catalog (GPL-3).

- Class `Fuel`
  - Field `one_hour: float`
  - Field `ten_hour: float`
  - Field `hundred_hour: float`
  - Field `live_herb: float`
  - Field `live_woody: float`
  - Field `sav_one_hour: float`
  - Field `sav_herb: float`
  - Field `sav_woody: float`
  - Field `depth: float`
  - Field `heat_content: float`
  - Field `density: float`
  - Field `extinction_moisture: float`
- `fuel_properties(codes, *, raster=False)`
- `read_fuel_raster(path, shape, dx, dy)`

### `plumeswift.fire_fuels_legacy`

[Source](../python_qes/plumeswift/fire_fuels_legacy.py)

Standalone catalogue from the historical QES OldFuelProperties.hpp (GPL-3).

- Class `HistoricalFuel`
  - Field `windrf: np.float32`
  - Field `fgi: np.float32`
  - Field `fueldepthm: np.float32`
  - Field `savr: int`
  - Field `fuelmce: np.float32`
  - Field `fueldens: int`
  - Field `st: np.float32`
  - Field `se: np.float32`
  - Field `weight: int`
  - Field `fci_d: np.float32`
  - Field `fct: int`
  - Field `ichap: int`
  - Field `fci: np.float32`
  - Field `fcbr: np.float32`
  - Field `hfgl: int`
  - Field `cmbcnst: int`
  - Field `fuelheat: np.float32`
  - Field `fuelmc_g: np.float32`
  - Field `fuelmc_c: int`
  - Field `slope: np.float32`

### `plumeswift.fire_grid`

[Source](../python_qes/plumeswift/fire_grid.py)

CPU QES fire front, forcing and burn-state updates (GPL-3).

- Class `Ignition`
  - Field `x: float`
  - Field `y: float`
  - Field `length: float`
  - Field `width: float`
  - Field `height: float = 0.0`
  - Field `base_height: float = 0.0`
- Class `TimedIgnition`
  - Field `time: float`
  - Field `x: float`
  - Field `y: float`
- `read_timed_ignitions(path, *, native_input=False)`
- Class `FireGrid`
  - `__init__(self, terrain, spacing, fuel_codes, ignitions=(), *, moisture, cure, courant, raster_fuel=False, timed_ignitions=(), spread=balbi, initialize_front=cpu_front, native_flame_sampling=False, native_fire_clock=False, native_ignition_input=False)`
  - `timestep(self)`
  - `forcing(self, workspace, *, narrow=True)`
  - `advance(self, workspace)`
  - `step(self, workspace, *, narrow=True)`

### `plumeswift.fire_output`

[Source](../python_qes/plumeswift/fire_output.py)

NetCDF layout of the original QES FIREOutput (GPL-3.0).

- `native_fire_series(path, domain, *, origin)`

### `plumeswift.fire_potential`

[Source](../python_qes/plumeswift/fire_potential.py)

Serial QES fire-induced velocity potential (GPL-3).

- Class `HeatPotential`
  - Field `radial: np.ndarray`
  - Field `vertical: np.ndarray`
  - Field `g: np.ndarray`
  - Field `g_prime: np.ndarray`
  - Field `radius: np.ndarray`
  - Field `height: np.ndarray`
  - Field `native_math: bool = False`
  - `read(cls, path, *, native_input=False)`
  - `apply(self, fire, workspace)`

### `plumeswift.fire_run`

[Source](../python_qes/plumeswift/fire_run.py)

Coupled fire and wind execution in the qesFireMain order (GPL-3).

- Class `FireFrame`
  - Field `time: datetime`
  - Field `fire_time_s: float`
  - Field `dt_s: float`
  - Field `fields: dict`
  - Field `coordinates: dict`
  - Field `turbulence: dict | None`
  - Field `wind_coordinates: dict`
  - Field `smoke_samples: tuple[SmokeSample, ...] = ()`
  - Field `smoke_accounting: dict | None = None`
  - Field `plume_samples: tuple[tuple[str, SmokeSample], ...] = ()`
  - Field `plume_accounting: dict | None = None`
- Class `CoupledFireCase`
  - Field `wind: WindCase`
  - Field `fire: FireConfig`
  - Field `potential: HeatPotential | None`
  - Field `smoke: SmokeConfig | None = None`
  - Field `particle_output: ParticleOutputConfig | None = None`
  - Field `initialize_front: Callable[[ndarray], ndarray] = cpu_front`
  - Field `plume_models: dict[str, ModelSpec] | None = None`
  - Field `native_threads: int | None = None`
  - Field `source_seed: int | None = None`
  - Field `native_flame_sampling: bool = False`
  - Field `native_release_indexing: bool = False`
  - Field `native_fire_clock: bool = False`
  - Field `native_full_domain_bounds: bool = False`
  - Field `narrow_band: bool = True`
  - Field `zero_wrf_steps: bool = False`
  - `frames(self, on_smoke_step=None, *, native_collection_math=False, include_gradients=False, native_fire_format=False, on_plume_step=None, progress=None, fire_progress_stream=None, plume_diagnostics=None)`
- `read_coupled_fire(path, *, heat_potential=None, compute_turbulence=False, plume=None, seed=0, particle_csv=False, poisson_backend='cpu', turbulence_gradient='cpu', heat_potential_backend='cpu', front_backend='cpu', native_sensor_math=False, native_wake_math=False, native_turbulence_math=False, native_building_math=False, native_serial=False, native_threads=None, source_seed=None, wrf_timeout=None, native_flame_sampling=False, meteorology_backend='cpu', native_release_indexing=False, legacy_turbulence_xml=False, legacy_plume_xml=False, native_fire_clock=False, native_ignition_input=False, native_full_domain_bounds=False, legacy_fire_xml=False, native_heat_potential_input=False, narrow_band=True)`
- `write_fire_run(case, output, *, native_collection_format=False, native_collection_math=False, native_turbulence_format=False, native_fire_format=False, prefix='', write_fire_output=True, write_turbulence=True, write_concentration=True, progress_stream=None, native_particle_csv=False, fire_progress_stream=None, plume_diagnostics=None)`
- `run_legacy_fire(wind_xml, plume_xml, *, basename=None, solver_type=1, fire_output=False, turbulence=False, particle_csv=False, no_fire_winds=False, heat_potential=None, seed=0, native_threads=1, source_seed=None, verbose=False, wrf_timeout=None, progress_stream=None, native_release_indexing=None, native_particle_csv=True, native_full_domain_bounds=False, cpu_serial=False)`

### `plumeswift.fire_smoke`

[Source](../python_qes/plumeswift/fire_smoke.py)

Adapt advancing fire cells to the shared plume transport (GPL-3).

- `legacy_smoke_sites(fire)`
- `read_smoke_config(path, *, seed=0)`
- `parse_smoke_config(root, *, seed=0, legacy_plume_xml=False)`
- Class `SmokePlume`
  - `__init__(self, config, mesh=None, *, track_particles=False, native_collection_math=False, specifications=None, native_threads=None, source_seed=None, native_release_indexing=False, native_full_domain_bounds=False)`
  - `advance(self, fire, workspace, turbulence, start_ms, end_ms, spacing, on_step=None)`
  - `advance_models(self, fire, workspace, turbulence, start_ms, end_ms, spacing, on_step=None, *, progress=None)`
  - `accounting(self)`
  - `model_accounting(self)`

### `plumeswift.fire_spread`

[Source](../python_qes/plumeswift/fire_spread.py)

QES-Fire Rothermel seed and iterative Balbi spread model (GPL-3).

- Class `FireProperties`
  - Field `updraft: np.ndarray`
  - Field `flame_height: np.ndarray`
  - Field `flame_depth: np.ndarray`
  - Field `spread_rate: np.ndarray`
  - Field `temperature: np.ndarray`
  - Field `residence_time: np.ndarray`
  - Field `drag: np.ndarray`
  - Field `heat_release: np.ndarray`
- `rothermel(fuel: Fuel, moisture)`
- `balbi(fuel: Fuel, wind_u, wind_v, normal_x, normal_y, slope_x, slope_y, moisture, *, cure, dx, dy)`

### `plumeswift.fire_spread_startup`

[Source](../python_qes/plumeswift/fire_spread_startup.py)

Native QES-Fire arithmetic for the calm, level startup closure (GPL-3).

- `calm_startup(fuel, normal_x, normal_y, slope_x, slope_y, moisture, *, cure, dx, dy)`

### `plumeswift.geometry`

[Source](../python_qes/plumeswift/geometry.py)

QES polygon stair-step flags and wall stencil; cut cells are separate.

- `ring_contains(vertices, x, y, *, native_ring_breaks=False)`
- `polygon_parts(rings)`
- `vertex_parts(vertices)`
- `translate_footprint(model, halo)`
- Class `PolygonBuilding`
  - Field `vertices: tuple[tuple[float, float], ...]`
  - Field `height: float`
  - Field `base_height: float = 0.0`
  - Field `holes: tuple[tuple[tuple[float, float], ...], ...] = ()`
  - Field `terrain_sample_vertices: tuple[tuple[float, float], ...] | None = None`
  - Field `native_closure_vertices: tuple[tuple[float, float], ...] | None = field(default=None, kw_only=True)`
  - Field `detached_parts: tuple[tuple[Ring, tuple[Ring, ...]], ...] = field(default=(), kw_only=True)`
  - `rectangle(cls, x, y, length, width, height, rotation=0.0, base_height=0.0, *, native_math=False)`
  - `physical_parts(self)`
  - `intersects_horizontal_domain(self, coordinates)`
  - `flag(self, workspace, number=0, *, mark_solid=True, native_dy_alias=False, native_math=False, native_wrapped_footprint=False)`
- `wall_stencil(workspace, *, native_dy_alias=False, native_outer_wrap=True, native_math=False)`
- `block_solid_neighbours(a, native_outer_wrap=True)`
- `scale_wall_coefficients(a, c, native_dy_alias, *, native_math=False)`
- `zero_solid_faces(workspace)`

### `plumeswift.georeference`

[Source](../python_qes/plumeswift/georeference.py)

Projected domain coordinates shared by geographic stations and vector GIS.

- `utm_epsg(zone, band='')`
- `transform_points(points, source_crs, target_crs)`
- `geographic_station(longitude, latitude, origin, zone, band)`

### `plumeswift.gis`

[Source](../python_qes/plumeswift/gis.py)

Projected vector polygons, ring nesting, and QES origin handling.

- `resolve_vector_path(paths, filename)`
- `default_building_layer(path)`
- `polygon_layer_extents(path, *, layer=None, target_epsg=None)`
- `read_building_shapefile(path, *, height_field='H', height_factor=1.0, origin=None, layer=None, target_epsg=None, native_math=False, legacy_ogr=False)`
- `read_tree_shapefile(path, *, origin=None, layer=None, target_epsg=None, legacy_ogr=False)`

### `plumeswift.hrrr`

[Source](../python_qes/plumeswift/hrrr.py)

HRRR NetCDF fields, stability, and QES profile 5 (GPL-3.0-only).

- Class `HRRRProfile`
  - Field `speed: float`
  - Field `direction: float`
  - Field `roughness: float`
  - Field `reciprocal_length: float`
  - Field `boundary_height: float`
  - Field `pressure_wind: tuple`
  - `evaluate(self, heights, altitude=0.0, *, native_math=False)`
- Class `HRRRStation`
  - Field `map_position: tuple[float, float] | None = None`
  - Field `map_index: int | None = None`
- Class `HRRRInput`
  - Field `path: str`
  - Field `scheme: int`
  - Field `stability: int`
  - Field `origin: tuple`
  - Field `zone: int`
  - `from_smoke_xml(cls, path, *, origin, zone, hrrr_path=None)`
  - `stations(self, domain, halo, step_limit, *, native_math=False, stored=False)`
  - `smoke_sources(self, domain, record, *, stored=False)`

### `plumeswift.hrrr_interpolation`

[Source](../python_qes/plumeswift/hrrr_interpolation.py)

Terrain-following HRRR interpolation on the staggered QES grid.

- `initialize_hrrr(workspace, stations, grid_ids, scheme, *, variant='native-cpu', datum=0.0, native_math=False, barnes_backend='cpu', nearest_map_positions=(), legacy_terrain_stride=False)`

### `plumeswift.input_paths`

[Source](../python_qes/plumeswift/input_paths.py)

Input-relative path resolution with the QES filesystem rules.

- Class `QESInputPaths`
  - Field `directory: Path`
  - `from_input(cls, filename)`
  - `resolve(self, filename, *, require_parent=True)`

### `plumeswift.interpolation`

[Source](../python_qes/plumeswift/interpolation.py)

Coordinate-based trilinear/nearest interpolation on QES staggered fields.

- Class `Trilinear`
  - Field `x: np.ndarray`
  - Field `y: np.ndarray`
  - Field `z: np.ndarray`
  - `weights(self, positions)`
- Class `CompiledTrilinear`
- Class `NativeTrilinear`
  - `__init__(self, spacing, offsets, z, *, value_dtype=np.float32)`
- Class `FieldSampler`
  - `__init__(self, coordinates, wind, turbulence, mode='trilinear', mixlength=None, *, native_spacing=None, native_math=False, compiled=False)`
  - `candidate_cell_indices(self, positions)`
  - `native_candidate_cell_indices(self, positions, spacing)`
  - `cell_indices(self, positions)`
  - `sample_distance(self, positions)`
  - `sample_wind(self, positions)`
  - `sample_cells(self, values, positions)`
  - `sample_turbulence(self, positions)`
  - `sample_initial_turbulence(self, positions)`
  - `sample_turbulence_and_distance(self, positions)`

### `plumeswift.io`

[Source](../python_qes/plumeswift/io.py)

Portable QES field and workspace NetCDF I/O.

- Class `WindWorkspace`
  - Field `arrays: dict`
  - Field `coordinates: dict`
  - Field `building_context: object | None = None`
  - Field `row_canopy_flows: tuple = ()`
  - Field `terrain_mesh: object | None = None`
  - `stencil(self, *, eliminate_solid=False)`
- `workspace_field_timed(variable, spatial_shape=None)`
- `workspace_record(nc, step, *, fields=None, positional_fields=None, native_input=False)`
- `read_wind_workspace(path: str | Path, *, step=0)` → WindWorkspace
- `netcdf_dates(dataset, *, native_timestamps=False)`
- `write_arrays(path, arrays, coordinates)`
- `read_turbulence(path, *, step=0)`
- `read_solver_coefficients(path, cell_shape, *, native_hyperslab=False, prepared_stencil=None)`
- `write_netcdf(path, arrays, coordinates, *, timestamp='2020-01-01T00:00:00', elapsed=0.0, static_fields=())`
- `netcdf_series(path, coordinates, *, static_fields=(), varying_fields=(), dimension_names=None, coordinate_names=None, time_origin=None, field_units=None, float_dtype='f8', long_names=None)`

### `plumeswift.lh2`

[Source](../python_qes/plumeswift/lh2.py)

Original research LH2 momentum coupling, thermodynamics supplied separately.

- Class `LH2Momentum`
  - Field `thermo: Thermodynamics`
  - Field `coefficients: PhysicalCoefficients = field(default_factory=PhysicalCoefficients)`
  - `update(self, positions, masses, drift, cell_ids, cell_volumes, dt)`

### `plumeswift.measured_profile`

[Source](../python_qes/plumeswift/measured_profile.py)

Native multi-height sensor profile flag 4 (WindProfilerSensorType, GPL-3).

- Class `MeasuredProfile`
  - Field `heights: tuple[float, ...]`
  - Field `speeds: tuple[float, ...]`
  - Field `directions: tuple[float, ...]`
  - Field `roughness: float = 0.02`
  - Field `reciprocal_length: float = 0.0`
  - Field `kind: int = field(default=4, init=False)`
  - Field `allow_negative_heights: InitVar[bool] = field(default=False, kw_only=True)`
  - `speed(self)`
  - `direction(self)`
  - `reference_height(self)`
  - `evaluate(self, heights, *, native_math=False)`

### `plumeswift.mesh_mixing`

[Source](../python_qes/plumeswift/mesh_mixing.py)

Legacy Mesh.cpp ray mixing and SphereDirections.h port (GPL-3).

- `legacy_terrain_mixing(workspace, domain)`
- `sphere_directions(samples=512, theta_bounds=(-1.0, 1.0), phi_bounds=(0.0, 2 * 3.14159265359))`
- Class `LegacyMeshMixing`
  - `__init__(self, mesh, directions=None)`
  - `distance(self, positions)`
  - `calculate(self, flags, spacing, initial=None)`

### `plumeswift.mesh_reflection`

[Source](../python_qes/plumeswift/mesh_reflection.py)

Python port of QES WallReflection_TriMesh::rayTraceReflect (GPL-3).

- Class `MeshReflection`
  - Field `mesh: object`
  - Field `max_bounces: int = 256`
  - `trace(self, origin, displacement, fluctuation)`
  - `trace_native(self, origin, displacement, fluctuation)`
- Class `TriangleWalls`
  - `__init__(self, reflection, endpoint_solid)`
  - `reflect(self, positions, displacement, fluctuations)`
- Class `NativeTriangleWalls`
  - `reflect_native(self, positions, displacement, fluctuations)`

### `plumeswift.meteorology`

[Source](../python_qes/plumeswift/meteorology.py)

QES sensor profiles. FROM directions are degrees clockwise from north.

- `stability_correction(z_over_l)`
- `canopy_displacement(ustar, z0, canopy_top, attenuation, vk=0.4, psi=0.0, *, native_math=False)`
- Class `SensorProfile`
  - Field `speed: float`
  - Field `direction: float`
  - Field `reference_height: float`
  - Field `roughness: float = 0.02`
  - Field `reciprocal_length: float = 0.0`
  - Field `kind: int = 1`
  - Field `canopy_height: float = 0.0`
  - Field `canopy_attenuation: float = 0.0`
  - `evaluate(self, heights)`
- `single_sensor_flat(workspace, profile)`
- Class `Station`
  - Field `x: float`
  - Field `y: float`
  - Field `profile: SensorProfile`
- `barnes_flat(workspace, stations, *, native_dy_alias=True)`

### `plumeswift.mixing_length`

[Source](../python_qes/plumeswift/mixing_length.py)

Original serial local-mixing algorithm (LocalMixingSerial.cpp, GPL-3.0).

- `save_mixing_length(path, variable, values)`
- Class `NetCDFMixing`
  - Field `path: str`
  - Field `variable: str = 'mixlength'`
  - Field `native_hyperslab: bool = False`
- Class `SerialMixing`
  - Field `native_dy_alias: bool = True`
  - Field `native_math: bool = False`

### `plumeswift.native_release_slots`

[Source](../python_qes/plumeswift/native_release_slots.py)

The original CPU release loop's slot-indexed state and ordinal lookups.

- Class `NativeReleaseSlots`
  - `__init__(self)`
  - `prepare(self, capacity, reserved, emissions, initial_active, track)`
  - `capture(self, batch)`

### `plumeswift.native_sensor`

[Source](../python_qes/plumeswift/native_sensor.py)

Mixed-precision QES CPU profiles and Barnes weights (GPL-3.0).

- `advance_canopy_station(station)`
- `evaluate_sensor(profile, heights)`
- `barnes_scale(positions)`
- `barnes_weights(positions, x, y)`
- `initialize_flat(workspace, stations, profile)`

### `plumeswift.native_source_geometry`

[Source](../python_qes/plumeswift/native_source_geometry.py)

Seedable Linux QES source geometry, independent of the transport RNG (GPL-3).

- Class `NativeSourceGeometry`
  - `__init__(self, geometry, seed)`
  - `sample(self, count, rng)`

### `plumeswift.native_staircase`

[Source](../python_qes/plumeswift/native_staircase.py)

CPU QES stair-step trajectory splitting and reflection (GPL-3.0).

- Class `NativeStaircase`
  - `__init__(self, sampler, solid)`
  - `reflect_native(self, positions, displacement, fluctuations)`

### `plumeswift.netcdf_legacy`

[Source](../python_qes/plumeswift/netcdf_legacy.py)

Direct legacy QES NetCDF dimensions, fields and hyperslabs.

- `open_legacy_input(path, *, raw=True)`
- Class `LegacyNetCDFInput`
  - `__init__(self, dataset, *, raw=True)`
  - `dimension_size(self, name)`
  - `field(self, name)`
  - `read(self, name, *, dtype=None, start=None, count=None)`
- `open_legacy_output(path)`
- Class `LegacyNetCDFOutput`
  - `__init__(self, dataset)`
  - `define_dimension(self, name, size=0)`
  - `dimension(self, name)`
  - `define_field(self, name, dimensions, dtype, *, units, long_name)`
  - `add_attribute(self, field, name, value)`
  - `write(self, name, values, *, start=None, count=None)`

### `plumeswift.nonlocal_mixing`

[Source](../python_qes/plumeswift/nonlocal_mixing.py)

QES building wake turbulence (NonLocalMixing.cpp, GPL-3).

- `apply_nonlocal_mixing(workspace, cell_velocity, stress, mixing, energy, dissipation, turbulence_flags, *, native_math=False)`

### `plumeswift.numerics.__init__`

[Source](../python_qes/plumeswift/numerics/__init__.py)

Numerical algorithms; no atmospheric or hydrogen model choices here.


### `plumeswift.numerics.compiled_gle`

[Source](../python_qes/plumeswift/numerics/compiled_gle.py)

Optional float64 3×3 GLE solve with partial pivoting and LAPACK fallback.

- `stress_needs_invariants(matrices, tolerance)`
- `solve_three_by_three(matrices, right_hand_side)`

### `plumeswift.numerics.compiled_transport`

[Source](../python_qes/plumeswift/numerics/compiled_transport.py)

Optional compiled loops for physical-coordinate transport.

- `trilinear(values, positions, x, y, z, spacing)`
- `reflect_voxels(positions, displacement, fluctuations, axes, spacing, solid, epsilon, max_crossings)`
- `uniform_spacing(axes)`

### `plumeswift.numerics.cuda_device_parallel`

[Source](../python_qes/plumeswift/numerics/cuda_device_parallel.py)

Python JIT port of the legacy QES device-launched pressure solver (GPL-3).

- Class `CudaDeviceParallel`
  - `project(self, workspace, physics=None)`

### `plumeswift.numerics.cuda_dynamic_legacy`

[Source](../python_qes/plumeswift/numerics/cuda_dynamic_legacy.py)

Deterministic port of the legacy QES dynamic GPU pressure stages (GPL-3).

- Class `CudaDynamicLegacy`
  - `__init__(self, policy=None, omega=1.78, device=0)`
  - `solve(self, stencil, rhs, initial=None)`
  - `project(self, workspace, physics=None)`

### `plumeswift.numerics.cuda_matrix3`

[Source](../python_qes/plumeswift/numerics/cuda_matrix3.py)

Device-callable float32 QES matrix operations (GPL-3).

- `device_matrix3(profile='modern')`

### `plumeswift.numerics.cuda_memory_legacy`

[Source](../python_qes/plumeswift/numerics/cuda_memory_legacy.py)

Ordered ports of the original QES CUDA global and shared memory solvers.

- Class `CudaGlobalMemory`
  - `__init__(self, policy=None, omega=1.78, device=0, *, native_boundary=False)`
- Class `CudaSharedMemory`

### `plumeswift.numerics.cuda_poisson`

[Source](../python_qes/plumeswift/numerics/cuda_poisson.py)

GPU red-black SOR with Python kernels and the QES pressure stencil.

- Class `CudaRedBlackSOR`
  - `__init__(self, policy=None, omega=1.78, device=0, check_every=8)`
  - `solve(self, stencil, rhs, initial=None)`
  - `project(self, workspace, physics=None)`

### `plumeswift.numerics.cuda_runtime`

[Source](../python_qes/plumeswift/numerics/cuda_runtime.py)

Load the optional CUDA array backend on request.

- `load_cupy(device)`

### `plumeswift.numerics.langevin`

[Source](../python_qes/plumeswift/numerics/langevin.py)

Batched generalized Langevin integration, same QES GLE equations.

- `invariants(tau)`
- `realizable(tau, tolerance=1e-05, *, compiled=False)`
- `implicit_gle(velocity_old, tau_old, tau_current, flux_divergence, coeps, dt, normal_draws, tolerance=1e-05, scheme='backward-euler', *, native_determinant_gate=False, compiled=False)`

### `plumeswift.numerics.linux_float_angles`

[Source](../python_qes/plumeswift/numerics/linux_float_angles.py)

Single-precision angle arithmetic for the Linux QES reference.

- `atanf(value)`
- `atan2f(y, x)`
- `sincosf(angle)`

### `plumeswift.numerics.linux_float_power`

[Source](../python_qes/plumeswift/numerics/linux_float_power.py)

Linux float32 wake exponent, adapted from glibc 2.35 powf.

- `powf_three_halves(ratio)`

### `plumeswift.numerics.linux_float_transcendentals`

[Source](../python_qes/plumeswift/numerics/linux_float_transcendentals.py)

Float32 log and exponential using the glibc 2.35 range reductions.

- `logf(value)`
- `expf(value)`

### `plumeswift.numerics.matrix3`

[Source](../python_qes/plumeswift/numerics/matrix3.py)

Batched three-dimensional matrix operations and QES storage layouts.

- `from_column_major(values)`
- `to_column_major(values)`
- `from_symmetric(values)`
- `to_symmetric(values)`
- `symmetric_invariants(values)`
- `inverse(values)`
- `native_inverse(values)`

### `plumeswift.numerics.native_cpu_rb`

[Source](../python_qes/plumeswift/numerics/native_cpu_rb.py)

Float32 CPU projection from QES Solver_CPU_RB.cpp and Solver_CPU.cpp (GPL-3.0).

- Class `NativeCPUSOR`
  - `__init__(self, policy=None)`
  - `solve(self, stencil, rhs, initial=None)`
  - `initial_divergence(self, workspace, physics=None)`
  - `project(self, workspace, physics=None, *, initial=None)`
- Class `NativeRedBlackSOR`
- Class `NativeSerialSOR`
- Class `HistoricalCPUSOR`

### `plumeswift.numerics.native_gle`

[Source](../python_qes/plumeswift/numerics/native_gle.py)

Single-precision CPU QES GLE for replay, from GLE_Solver.cpp and VectorMath.h.

- `native_realizable(tau, tolerance)`
- `native_implicit_gle(velocity, previous, stress, flux, coeps, dt, normal_draws, tolerance)`

### `plumeswift.numerics.poisson`

[Source](../python_qes/plumeswift/numerics/poisson.py)

Replaceable solvers for the same seven-point QES multiplier equation.

- Class `NativeWindPolicy`
  - Field `relative_tolerance: float = 1e-09`
  - Field `max_iterations: int = 500`
- `make_poisson(backend, policy=None, *, native_cuda_boundary=False)`
- Class `Stencil`
  - Field `east: np.ndarray`
  - Field `west: np.ndarray`
  - Field `north: np.ndarray`
  - Field `south: np.ndarray`
  - Field `up: np.ndarray`
  - Field `down: np.ndarray`
  - Field `fluid: np.ndarray`
  - `boundaries(self, p)`
  - `neighbours(self, p)`
  - `apply(self, p)`
  - `residual(self, p, rhs)`
- Class `RedBlackSOR`
  - `__init__(self, policy: NumericPolicy | None=None, omega=1.78)`
  - `solve(self, stencil, rhs, initial=None)`
- Class `KrylovPoisson`
  - `__init__(self, policy: NumericPolicy | None=None)`
  - `solve(self, stencil, rhs, initial=None)`

### `plumeswift.numerics.vector3`

[Source](../python_qes/plumeswift/numerics/vector3.py)

Scalar and batched QES three-component vector operations.

- `length32(vector)`
- `reflect32(vector, normal)`
- `dot(left, right)`
- `length(vector)`
- `distance(left, right)`
- `reflect(vector, normal)`

### `plumeswift.optix_ray_mixing`

[Source](../python_qes/plumeswift/optix_ray_mixing.py)

Trace method-3 mixing rays with NVIDIA OptiX via its Python host bindings.

- `load_optix()`
- `optix_ray_distances(mesh, points, directions, *, device=0)`

### `plumeswift.output_example`

[Source](../python_qes/plumeswift/output_example.py)

Python port of the QES util_output example (GPL-3.0).

- `write_output_example(path, *, start=None)`

### `plumeswift.output_v2`

[Source](../python_qes/plumeswift/output_v2.py)

Multi-source QES NetCDF output with independent field layouts (GPL-3.0).

- Class `OutputField`
  - Field `dimensions: tuple[str, ...] | str`
  - Field `dtype: str | np.dtype`
  - Field `units: str`
  - Field `long_name: str`
- `open_output(path, *, start=None, enabled=True)`
- Class `NullOutput`
  - `__init__(self)`
  - `attach(self, source, fields, read, *, prepare=None)`
  - `push(self, source, time, *, prepare=False)`
- Class `MultiSourceOutput`
  - `__init__(self, dataset, start)`
  - `define_axis(self, name, values, *, units, long_name)`
  - `define_dimension(self, name, size)`
  - `define_dimension_set(self, name, dimensions)`
  - `attach(self, source, fields, read, *, prepare=None)`
  - `remove_field(self, name)`
  - `remove_static_fields(self)`
  - `set_start_time(self, start)`
  - `new_time_entry(self, time, *, allow_duplicate=False)`
  - `push(self, source, time, *, prepare=False)`
  - `push_fields(self, time, fields=None)`
  - `save(self, time)`

### `plumeswift.particle_ids`

[Source](../python_qes/plumeswift/particle_ids.py)

Run-scoped particle IDs shared by all plume models in a calculation.

- Class `ParticleIDs`
  - `__init__(self, *, native=False, start=0)`
  - `take(self, count)`

### `plumeswift.particle_output`

[Source](../python_qes/plumeswift/particle_output.py)

QES particle snapshots with the native clock, columns and physical units.

- Class `ParticleOutputConfig`
  - Field `first_ms: int`
  - Field `period_ms: int`
- `read_particle_output(root, *, legacy_plume_xml=False)`
- Class `ParticleOutput`
  - `__init__(self, config, output, origin, *, prefix='', native_format=False)`
  - `save(self, time_ms, models)`

### `plumeswift.particle_sampling_pool`

[Source](../python_qes/plumeswift/particle_sampling_pool.py)

Process-based field sampling for a stationary QES plume workspace.

- Class `ParticleSamplingPool`
  - `__init__(self, fields, workers, *, min_particles=2048)`
  - `sample(self, positions)`

### `plumeswift.particle_slot_regression`

[Source](../python_qes/plumeswift/particle_slot_regression.py)

Deterministic counterpart of the original ManagedContainer example.

- `run_managed_container(*, steps=10000, release_count=1000, seed=0, on_step=None)`

### `plumeswift.particle_slots`

[Source](../python_qes/plumeswift/particle_slots.py)

CPU slot allocation compatible with QES ManagedContainer (GPL-3).

- `first_free_slots(occupied, count)`
- `reserve_particle_slots(occupied, count, capacity)`

### `plumeswift.particles`

[Source](../python_qes/plumeswift/particles.py)

Python particle state and orchestrator, with replaceable numerical services.

- Class `ParticleBatch`
  - Field `positions: np.ndarray`
  - Field `mass: np.ndarray`
  - Field `fluctuations: np.ndarray`
  - Field `previous_stress: np.ndarray`
  - Field `drift: np.ndarray`
  - Field `identifiers: np.ndarray`
  - Field `tracking: np.ndarray | None = None`
  - Field `material: np.ndarray | None = None`
  - Field `slots: np.ndarray | None = None`
  - Field `previous_fluctuations: np.ndarray | None = None`
  - `empty(cls)`
  - `retain(self, keep)`
  - `append(self, position, mass, fluctuation, stress, drift, identifiers, *, source=None, release_s=0.0, slots=None, previous_fluctuation=None)`
- `courant_timestep(distance, absolute_velocity, remaining, main_dt, courant, min_spacing)`
- `native_courant_timestep(distance, absolute_velocity, remaining, main_dt, courant, min_spacing)`
- Class `ParticleIntegrator`
  - `__init__(self, fields, boundaries, walls, rng, policy=None, scheme='backward-euler', courant=0.25, max_velocity=1000000.0, *, main_step=None, min_spacing=None, rogue_velocity=None, deposition=None, executor=None, native_arithmetic=False, sampling_in_grid=None, sampling_pool=None, compiled_gle=False)`
  - `advance(self, batch, dt, *, slot_capacity=None, capture=None)`
- Class `ParticleSimulation`
  - `__init__(self, sources, integrator, observers=(), momentum=None, cell_volume=None, *, initial_active=None, track_particles=False, allocate_ids=None, serial_slots=False, native_release_indexing=False, initial_in_grid=None, momentum_cell_indices=None)`
  - `step(self, dt)`
  - `run(self, duration_s, dt, progress=None)`
  - `accounting(self)`

### `plumeswift.plume_collection`

[Source](../python_qes/plumeswift/plume_collection.py)

CPU concentration sampling and accumulation for QES plumes (GPL-3.0).

- `native_box_geometry(config)`
- `native_box_centers(config)`
- Class `PlumeCollection`
  - `__init__(self, config, *, native_math=False, allow_zero_volume=False)`
  - `collect(self, positions, mass_kg, dt)`
  - `finalize(self)`
  - `reset(self)`
  - `take(self)`

### `plumeswift.plume_config`

[Source](../python_qes/plumeswift/plume_config.py)

Common settings from the QES plume XML contract (GPL-3).

- Class `PlumeConfig`
  - Field `duration_s: float`
  - Field `step_s: float`
  - Field `courant: float`
  - Field `stress_tolerance: float`
  - Field `interpolation: str`
  - Field `boundaries: tuple[str, str, str]`
  - Field `wall: str`
  - Field `collection_start_ms: int`
  - Field `collection_period_ms: int`
  - Field `lower: tuple[float, float, float]`
  - Field `upper: tuple[float, float, float]`
  - Field `boxes: tuple[int, int, int]`
  - Field `seed: int`
  - Field `progress_period_ms: int | None = None`
  - Field `allow_zero_collection_volume: bool = False`
  - `coordinates(self)`
- `parse_plume_config(root, *, seed=0, legacy_plume_xml=False)`

### `plumeswift.plume_fields`

[Source](../python_qes/plumeswift/plume_fields.py)

Stream staggered QES wind and turbulence files for standalone plume.

- `open_plume_fields(winds_path, turbulence_path, *, native_timestamps=False, allow_nonfinite_turbulence=False, native_layout=False, native_arithmetic=False, diagnostics=None, include_tke=False)`

### `plumeswift.plume_progress`

[Source](../python_qes/plumeswift/plume_progress.py)

Progress clock shared by the native-style plume entry points.

- `write_legacy_options(stream, *, plume, wind, turbulence, basename, verbose, particle_output)`
- `write_legacy_particle_summary(stream, origin, final_ms, models)`
- Class `PlumeProgress`
  - `__init__(self, origin, period_ms, stream)`
  - `begin(self, start_ms)`
  - `begin_interval(self, time_ms, end_ms, models)`
  - `end_interval(self, time_ms, models)`
  - `step(self, time_ms, end_ms, models)`

### `plumeswift.plume_regression`

[Source](../python_qes/plumeswift/plume_regression.py)

Analytical QES plume regressions (GPL-3.0-only).

- `sinusoidal_diagnostics(z, w, delta_w, step_s)`
- `run_sinusoidal_regression(*, particles=200000, duration=12.0, seed=0, native_rng=False)`
- `uniform_diagnostics(counts, coordinates, *, box_volume, step_s, averaging_s, release_rate)`
- `run_uniform_regression(*, particles_per_step=400, duration=2100.0, averaging_start=300.0, seed=0)`

### `plumeswift.plume_run`

[Source](../python_qes/plumeswift/plume_run.py)

Run the CPU QES wind-to-plume driver and export per-model concentrations.

- `run_plume(wind_path, plume_path, output, *, seed=0, turbulence_path=None, particle_csv=False, deposition=False, terrain_obj=None, poisson_backend='cpu', turbulence_gradient='cpu', native_serial=False, source_seed=None, native_threads=None, native_parallel=False, native_release_indexing=False, native_full_domain_bounds=False, native_collection_format=False, native_collection_math=False, native_workspace_format=False, native_turbulence_format=False, native_visualization_format=False, native_sensor_math=False, native_wake_math=False, native_turbulence_math=False, native_building_math=False, output_basename=None, allow_empty_models=False, write_concentration=True, terrain_output=None, wrf_timeout=None, progress_stream=None, native_timestamps=False, native_layout=False, native_arithmetic=False, legacy_plume_xml=False, native_particle_csv=False, _zero_wrf_steps=False)`
- `run_legacy_qes(wind_xml, plume_xml, *, basename=None, solver_type=1, workspace=False, turbulence=False, terrain=False, particle_csv=False, seed=0, native_threads=1, source_seed=None, verbose=False, wrf_timeout=None, progress_stream=None, native_release_indexing=False, native_particle_csv=False, native_full_domain_bounds=False, cpu_serial=False)`

### `plumeswift.plume_sources`

[Source](../python_qes/plumeswift/plume_sources.py)

Native QES XML source builders and fixed-step release semantics (GPL-3).

- Class `SourceSpec`
  - Field `source_id: int`
  - Field `geometry: Geometry | None`
  - Field `release: StepRelease`
  - Field `diameter_um: float`
  - Field `density_kg_m3: float`
  - Field `geometry_seed_index: int | None = None`
  - `create(self, boundaries, *, source_seed=None, native_full_domain_bounds=False)`
- Class `ModelSpec`
  - Field `sources: tuple[SourceSpec, ...]`
  - Field `deposition: bool`
- `read_models(root, step, *, legacy_plume_xml=False, allow_repeated_tags=False)`

### `plumeswift.plume_transport`

[Source](../python_qes/plumeswift/plume_transport.py)

Persistent QES plume transport and concentration collection (GPL-3).

- Class `PlumeSample`
  - Field `time_ms: int`
  - Field `fields: dict`
  - Field `extra_statistics: dict = field(default_factory=dict)`
- `plume_collection_series(path, config, origin, *, native_math=False)`
- Class `PlumeModel`
  - `__init__(self, config, mesh=None, *, rng=None, track_particles=False, allocate_ids=None, deposition=False, serial_slots=False, executor=None, native_collection_math=False, native_arithmetic=False, native_release_indexing=False)`
  - `next_sample(self)`
  - `set_fields(self, workspace, turbulence, spacing)`
  - `advance_step(self, end_ms)`
  - `advance(self, end_ms, on_step=None)`

### `plumeswift.powerlaw`

[Source](../python_qes/plumeswift/powerlaw.py)

Native analytic power-law test interpolation (InterpPowerLaw.cpp, GPL-3.0).

- Class `PowerLawSampler`
  - `max_fluctuation(self)`
  - `sample_wind(self, positions)`
  - `sample_initial_turbulence(self, positions)`
  - `sample_turbulence_and_distance(self, positions)`
- Class `NativePowerLawSampler`
  - `max_fluctuation(self)`
  - `sample_wind(self, positions)`
  - `sample_initial_turbulence(self, positions)`
  - `sample_turbulence_and_distance(self, positions)`

### `plumeswift.prescribed_diagnostics`

[Source](../python_qes/plumeswift/prescribed_diagnostics.py)

Vertical probability and particle statistics for prescribed-flow workflows.

- `particle_statistics(snapshot, turbulence, step, *, height=1.0, les=False, record=-1)`
- `write_particle_diagnostics(snapshot, turbulence, output, step, *, height=1.0, les=False, record=-1)`
- `write_case_diagnostics(xml, turbulence, step, duration, *, period=None, height=1.0, les=False)`

### `plumeswift.prescribed_figures`

[Source](../python_qes/plumeswift/prescribed_figures.py)

Optional figures for prescribed-flow probability and particle diagnostics.

- `write_profile_figures(output, stem, cases, *, ghosts=False, les=False)`

### `plumeswift.prescribed_plume`

[Source](../python_qes/plumeswift/prescribed_plume.py)

Shared XML and transport for the original file-based instantaneous releases.

- `write_instantaneous_config(path, step, particles, source_bounds, *, duration, output_period, collection_upper, boxes, vertical_bc)`
- `run_instantaneous_plume(wind, turbulence, xml, output, *, seed, native_serial)`

### `plumeswift.qes_xml`

[Source](../python_qes/plumeswift/qes_xml.py)

Read QES inputs with the historical lexical variants shipped upstream.

- `read_qes_xml(path)`
- `parse_optional_number(element, name, default, *, convert, legacy=False)`
- `parse_vector(element, name, *, convert=float, size=None, repeated=False, stop_on_invalid=False)`

### `plumeswift.ray_mixing`

[Source](../python_qes/plumeswift/ray_mixing.py)

Python implementation of the QES OptiX method-3 ray sampling equations.

- `ray_directions(samples)`
- `mixing_scene(workspace, buildings=(), terrain_mesh=None)`
- Class `RayMixing`
  - Field `samples: int = 2000`
  - Field `backend: str = 'cpu'`

### `plumeswift.rooftop`

[Source](../python_qes/plumeswift/rooftop.py)

Original QES logarithmic/vortex rooftop (Rooftop.cpp, GPL-3.0).

- `rooftop(workspace, building, wake_state, *, mode=2, roughness=0.01, native_dy_alias=True, native_math=False)`

### `plumeswift.runtime_info`

[Source](../python_qes/plumeswift/runtime_info.py)

QES startup and CUDA device diagnostics for the Python compatibility drivers.

- `write_runtime_banner(*, cuda=False, optix=False, output=None, diagnostics=None)`
- `report_startup_error(error, *, stream=None)`
- `report_warning(message, stream)`

### `plumeswift.sidewall`

[Source](../python_qes/plumeswift/sidewall.py)

Original QES sidewall recirculation and internal boundary layer.

- `sidewall(workspace, building, wake_state, *, roughness=0.01, native_dy_alias=True, native_math=False)`

### `plumeswift.single_building_demo`

[Source](../python_qes/plumeswift/single_building_demo.py)

Standalone 2018 QES single-building CPU and CUDA benchmarks (GPL-3.0).

- `run_single_building(output, *, backend='cpu')`
- `main()`

### `plumeswift.sinusoidal_case`

[Source](../python_qes/plumeswift/sinusoidal_case.py)

File-based Sinusoidal3D test case from QES testCases (GPL-3.0).

- `prepare_sinusoidal_case(output, *, steps=TIME_STEPS, particles=100000)`
- `run_sinusoidal_case(output, *, steps=TIME_STEPS, particles=100000, prepare_only=False, seed=0, native_serial=False, plot=False)`

### `plumeswift.sources`

[Source](../python_qes/plumeswift/sources.py)

Composable source geometry and release schedules; SI mass units throughout.

- Class `Geometry`
  - `sample(self, count: int, rng: np.random.Generator)` → np.ndarray
- Class `Point`
  - Field `position: tuple[float, float, float]`
  - `sample(self, count, rng)`
- Class `Cube`
  - Field `minimum: tuple[float, float, float]`
  - Field `maximum: tuple[float, float, float]`
  - Field `validate_dimensions: InitVar[bool] = True`
  - `sample(self, count, rng)`
- Class `Line`
  - Field `start: tuple[float, float, float]`
  - Field `end: tuple[float, float, float]`
  - `sample(self, count, rng)`
- Class `SphereShell`
  - Field `centre: tuple[float, float, float]`
  - Field `radius: float`
  - Field `validate_dimensions: InitVar[bool] = True`
  - `sample(self, count, rng)`
- Class `ContinuousRelease`
  - Field `mass_rate_kg_s: float`
  - Field `particles_per_second: float`
  - Field `start_s: float = 0.0`
  - Field `end_s: float = float('inf')`
  - `count_and_mass(self, t, dt)`
- Class `InstantaneousRelease`
  - Field `mass_kg: float`
  - Field `particles: int`
  - Field `time_s: float = 0.0`
  - `count_and_mass(self, t, dt)`
- Class `StepRelease`
  - Field `particles: int`
  - Field `mass_kg: float`
  - Field `start_s: float`
  - Field `end_s: float`
  - `count_and_mass(self, t, dt)`
- Class `Source`
  - Field `geometry: Geometry`
  - Field `release: ContinuousRelease | InstantaneousRelease | StepRelease`
  - Field `source_id: int = 0`
  - Field `initial_drift: tuple[float, float, float] = (0.0, 0.0, 0.0)`
  - Field `diameter_um: float = 0.0`
  - Field `density_kg_m3: float = 0.0`
  - `emit(self, t, dt, rng)`

### `plumeswift.statistics`

[Source](../python_qes/plumeswift/statistics.py)

Independent observers: exact box counting and time-window accounting.

- Class `StatisticsDirector`
  - `__init__(self, start_ms, period_ms)`
  - `attach(self, key, collect, finalize)`
  - `compute(self, time_ms, step_s)`
- Class `SensorBoxes`
  - Field `centres: np.ndarray`
  - Field `widths: tuple[float, float, float] = (2.0, 2.0, 0.2)`
  - `concentrations(self, positions, masses)`
- Class `TimeAveragedSensors`
  - `__init__(self, boxes, start_s, end_s, thermo=None)`
  - `observe(self, time_start, dt, positions, masses)`
  - `mean(self)`

### `plumeswift.street_canyon`

[Source](../python_qes/plumeswift/street_canyon.py)

Original QES legacy street-canyon closure (StreetCanyon.cpp, GPL-3.0).

- `street_canyon(workspace, building, wake_state, buildings_by_id, max_velocity, *, roughness=0.01, native_dy_alias=True, mode=1, native_math=False)`

### `plumeswift.street_intersection`

[Source](../python_qes/plumeswift/street_intersection.py)

Standalone legacy street intersection routines from StreetIntersection.cpp (GPL-3).

- `mark_street_intersections(workspace)`
- `smooth_street_intersections(workspace, iterations=10)`

### `plumeswift.tabulated_plume_cases`

[Source](../python_qes/plumeswift/tabulated_plume_cases.py)

QES Channel3D and BaileyLES input preprocessing and timestep series.

- Class `Case`
  - Field `stem: str`
  - Field `duration: float`
  - Field `period: float`
  - Field `steps: tuple[float, ...]`
  - Field `collection_upper: tuple[float, float, float]`
  - Field `boxes: int`
- `prepare_tabulated_case(name, profiles, output, *, steps=None, particles=100000, duration=None)`
- `run_tabulated_case(name, profiles, output, *, steps=None, particles=100000, duration=None, prepare_only=False, seed=0, native_serial=False, plot=False)`

### `plumeswift.terrain`

[Source](../python_qes/plumeswift/terrain.py)

Native DEM mesh construction and stairstep terrain (DTEHeightField, GPL-3).

- Class `GridTerrain`
  - Field `heights: object`
  - Field `halo_x: float = 0.0`
  - Field `halo_y: float = 0.0`
  - `describe(self)`
  - `mesh(self, domain)`
  - `cut_geometry(self, domain, halo)`
- Class `DEMTerrain`
  - Field `path: str`
  - Field `origin_flag: int = 0`
  - Field `utm_x: float = 0.0`
  - Field `utm_y: float = 0.0`
  - Field `distance_x: float = 0.0`
  - Field `distance_y: float = 0.0`
  - Field `target_epsg: int | None = None`
  - Field `archive_member: str | None = None`
  - Field `gzip_encoded: bool = False`
  - `from_input(cls, paths, filename, **options)`
  - `describe(self)`
  - `mesh(self, domain)`
  - `cut_geometry(self, domain, halo)`
  - `geometry_with_datum(self, domain, halo=None)`
- `apply_stairstep_terrain(workspace, mesh, *, mark_solid=True, halo_cells=(0, 0), native_sensor_math=False)`

### `plumeswift.terrain_meteorology`

[Source](../python_qes/plumeswift/terrain_meteorology.py)

Native terrain-following station shifts and two-pass Barnes interpolation.

- `station_profiles(workspace, stations, *, native_math=False)`
- `single_sensor_terrain(workspace, station, *, sampled=None, native_math=False)`
- `barnes_terrain(workspace, stations, *, variant='native-cpu', sampled=None, native_math=False)`

### `plumeswift.thermo`

[Source](../python_qes/plumeswift/thermo.py)

Interchangeable homogeneous-mixture thermodynamic providers.

- Class `EquilibriumTable`
  - `evaluate(self, concentration)`
  - `describe(self)`
- Class `LegacyEquilibrium`
  - `__init__(self, ambient_temperature=276.45, liquid_fraction=0.559, relative_humidity=75.0, pressure=101325.0, warm=False, points=10000, out_of_range='clip')`
- Class `CoolPropEquilibrium`
  - `__init__(self, ambient_temperature=276.45, liquid_fraction=0.559, relative_humidity=75.0, pressure=101325.0, minimum_temperature=85.0, points=4096, out_of_range='raise', properties=None, mixing_rule=None, moisture=None)`

### `plumeswift.thermodynamics.__init__`

[Source](../python_qes/plumeswift/thermodynamics/__init__.py)

Fine-grained property, moisture and mixture-rule components.


### `plumeswift.thermodynamics.mixing`

[Source](../python_qes/plumeswift/thermodynamics/mixing.py)

Independent mixture-density rule; thermodynamic approximation is explicit.

- Class `AmagatMixing`
  - `density_and_volume_fraction(self, hydrogen_mass_fraction, rho_hydrogen, rho_air)`
  - `describe(self)`

### `plumeswift.thermodynamics.moisture`

[Source](../python_qes/plumeswift/thermodynamics/moisture.py)

Trace-water energy module, independently replaceable from hydrogen EOS.

- `saturation_pressure_water(T)`
- Class `TraceWaterEnergy`
  - `enthalpy_change(self, temperatures, ambient_temperature, relative_humidity, pressure, properties, liquid_fraction_at_freezing=None)`
  - `describe(self)`

### `plumeswift.thermodynamics.properties`

[Source](../python_qes/plumeswift/thermodynamics/properties.py)

Pure-fluid property adapter; no dispersion, force, or mixture equations.

- Class `CoolPropProperties`
  - `__init__(self, hydrogen_fluid='Hydrogen')`
  - `enthalpy(self, fluid, temperature, pressure, phase=None)`
  - `density(self, fluid, temperature, pressure)`
  - `saturation(self, fluid, pressure, quality)`
  - `describe(self)`

### `plumeswift.timing`

[Source](../python_qes/plumeswift/timing.py)

QES UTC timestamp arithmetic and elapsed-time utilities.

- `parse_timestamp(value: str)` → datetime
- `elapsed_milliseconds(current: datetime, start: datetime)` → int
- `milliseconds(seconds)`
- `epoch_seconds(timestamp: datetime)` → int
- `timestamp_remainder(timestamp: datetime, period_s: float)` → float
- Class `QEStime`
  - `__init__(self, value=None)`
  - `datetime(self)`
  - `set_time_to_now(self)`
  - `set_timestamp(self, value)`
  - `get_timestamp(self)`
  - `get_epoch_time(self)`
  - `increment(self, seconds)`
- Class `NamedTimers`
  - `__init__(self, *, clock=perf_counter_ns)`
  - `reset(self)`
  - `start_new_timer(self, name)`
  - `reset_stored_timer(self, name)`
  - `elapsed(self, name)`
  - `print_stored_time(self, name, *, stream=None)`
- Class `Timer`
  - `__init__(self, name, *, clock=perf_counter_ns)`
  - `start(self)`
  - `stop(self)`
  - `elapsed(self)`
  - `show(self, *, stream=None)`

### `plumeswift.triangle_mesh`

[Source](../python_qes/plumeswift/triangle_mesh.py)

QES triangle height queries with a Python BVH (Triangle.cpp/BVH.cpp, GPL-3).

- Class `RayHit`
  - Field `triangle: int`
  - Field `parameter: float`
  - Field `distance: float`
  - Field `point: np.ndarray`
  - Field `normal: np.ndarray`
- Class `TriangleMesh`
  - `__init__(self, triangles, leaf_size=8)`
  - `ray_hit(self, origin, direction, *, max_parameter=np.inf, native_math=False)`
  - `ray_parameters(self, origins, directions, *, maximum=1e+16)`
  - `height(self, x, y, *, native_math=False)`
  - `read_obj(cls, path)`
  - `write_obj(self, path, *, indexed=False)`

### `plumeswift.turbulence`

[Source](../python_qes/plumeswift/turbulence.py)

QES local turbulence construction, separated from non-local closures.

- `cell_velocity(velocity, *, tangential_shell=False, native_math=False)`
- `default_mixing_distance(workspace, *, native_math=False)`
- Class `LocalTurbulence`
  - Field `sigma: tuple[float, float, float] = (2.5, 2.0, 1.3)`
  - Field `upper_bound: float = 15.0`
  - Field `background: float = 0.0`
  - Field `building_wall: int = 2`
  - Field `terrain_wall: int = 2`
  - Field `wall_roughness: float = 0.01`
  - Field `terrain_roughness: float = 0.02`
  - Field `native_quirks: bool = True`
  - Field `nonlocal_mixing: bool = False`
  - Field `gradient_backend: str = 'cpu'`
  - Field `native_math: bool = False`
  - Field `explicit_sigma: bool = True`
  - Field `native_zero_mixing_nan: bool = False`
  - Field `force_nonuniform_z: bool = False`
  - `velocity_gradient(self, workspace, velocity)`
  - `stress_divergence(self, workspace, tau)`
  - `build(self, workspace, velocity, *, sensor_roughness=0.02, building_heights=(), mixing_distance=None, include_gradients=False, previous_turbulence=None)`

### `plumeswift.turbulence_output`

[Source](../python_qes/plumeswift/turbulence_output.py)

NetCDF layout of the original QES TURBOutput (GPL-3.0).

- `native_turbulence_series(path, domain, *, origin, allow_nan=False)`

### `plumeswift.upwind`

[Source](../python_qes/plumeswift/upwind.py)

Original QES Rockle/MVP/HMVP upwind cavity (UpwindCavity.cpp, GPL-3.0).

- `upwind_cavity(workspace, building, mode=2, *, native_dy_alias=True, native_math=False)`

### `plumeswift.utm`

[Source](../python_qes/plumeswift/utm.py)

QES GIStool UTM projection on its Clarke 1866 ellipsoid.

- `to_utm(longitude, latitude, *, wrf_legacy=False, northern_hemisphere=True, native_zone=False)`
- `from_utm(x, y, zone, *, northern_hemisphere=True)`
- `from_wrf_utm(x, y, zone)`
- `grid_convergence(longitude, latitude, zone)`

### `plumeswift.vegetation`

[Source](../python_qes/plumeswift/vegetation.py)

QES canopy state, homogeneous vegetation and regression (GPL-3).

- Class `CanopyPrism`
  - `flag(self, workspace, state, number, *, cell_flag=20, attenuation=None, base_face_offset=0)`
- Class `HomogeneousCanopy`
  - Field `vertices: tuple[tuple[float, float], ...]`
  - Field `height: float`
  - Field `attenuation: float`
  - Field `base_height: float = 0.0`
  - Field `holes: tuple[tuple[tuple[float, float], ...], ...] = ()`
  - `rectangle(cls, x, y, length, width, height, attenuation, rotation=0.0, base_height=0.0)`
  - `flag(self, workspace, state, number)`
  - `apply(self, workspace, state, columns)`
- `slope_match(z0, height, attenuation)`
- Class `CanopyState`
  - `__init__(self, workspace, *, native_geometry=False)`
  - `column_displacement(self, j, i, *, sentinel_only=False, native_math=False)`
  - `regression(self, workspace)`
  - `export(self, workspace)`
- Class `Vegetation`
  - Field `canopies: tuple = ()`
  - Field `wake: bool = True`
  - `setup(self, workspace, *, native_geometry=False)`
  - `apply(self, workspace, state)`
  - `order(self)`
  - `apply_wake(self, workspace, state)`

### `plumeswift.wake`

[Source](../python_qes/plumeswift/wake.py)

Literal QES polygon cavity/far-wake model (PolygonWake.cpp, GPL-3.0).

- Class `WakeState`
  - Field `frame: BuildingFrame`
  - Field `length: float`
  - Field `height: float`
  - Field `rooftop: bool`
  - Field `tall: bool`
  - Field `length_over_height: float`
  - Field `width_over_height: float`
  - Field `mixing_frame: BuildingFrame | None = None`
  - Field `mixing_angles: np.ndarray | None = None`
- `face_wake_lengths(frame, relative, length, *, native_math=False)`
- `polygon_wake(workspace, building, building_id=0, *, mode=2, rooftop_mode=2, high_rise=False, native_dy_alias=True, native_math=False)`

### `plumeswift.walls`

[Source](../python_qes/plumeswift/walls.py)

Voxel wall reflection with exact segment traversal, separated from advection.

- Class `NoWalls`
  - `reflect(self, positions, displacement, fluctuations)`
- `endpoint_cells(sampler, positions)`
- `cell_grid_mask(indices, shape)`
- `fluid_cell_mask(indices, solid)`
- Class `SolidExitWalls`
  - `__init__(self, sampler, solid)`
  - `active_mask(self, positions)`
- Class `VoxelWalls`
  - `__init__(self, sampler, solid, max_crossings=256, *, endpoint_only=False, compiled=False)`
  - `reflect(self, positions, displacement, fluctuations)`

### `plumeswift.wind_case`

[Source](../python_qes/plumeswift/wind_case.py)

Composable wind pipeline. Unsupported XML features fail before execution.

- Class `UnsupportedFeatures`
  - `__init__(self, features)`
- Class `MeteorologyStep`
  - Field `timestamp: str`
  - Field `stations: tuple[Station, ...]`
- Class `WindCase`
  - Field `domain: Domain`
  - Field `profile: SensorProfile`
  - Field `buildings: tuple[PolygonBuilding, ...] = ()`
  - Field `upwind_mode: int = 0`
  - Field `turbulence: LocalTurbulence | None = None`
  - Field `poisson: object = field(default_factory=KrylovPoisson)`
  - Field `wake_mode: int = 0`
  - Field `rooftop_mode: int = 0`
  - Field `sidewall_mode: int = 0`
  - Field `high_rise: bool = False`
  - Field `wall_roughness: float = 0.01`
  - Field `stations: tuple[Station, ...] = ()`
  - Field `building_pipeline: BuildingPipeline | None = None`
  - Field `allow_unset_wake: bool = field(default=False, kw_only=True)`
  - Field `street_canyon_mode: int = 0`
  - Field `timestamp: str = '2020-01-01T00:00:00'`
  - Field `mixing_method: int = 0`
  - Field `mixing_samples: int = 2000`
  - Field `mixing_ray_backend: str = 'cpu'`
  - Field `mixing_file: str | None = None`
  - Field `mixing_variable: str = 'mixlength'`
  - Field `save_mixing: bool = False`
  - Field `terrain: object | None = None`
  - Field `meteorology_variant: str = 'native-cpu'`
  - Field `vegetation: object | None = None`
  - Field `schedule: tuple[MeteorologyStep, ...] = ()`
  - Field `reference_roughness: float | None = None`
  - Field `coefficient_file: str | None = None`
  - Field `roughness_split: bool = False`
  - Field `halo: HorizontalHalo = field(default_factory=HorizontalHalo)`
  - Field `mesh_type: int = 0`
  - Field `hrrr: object | None = None`
  - Field `hrrr_grid_ids: tuple = ()`
  - Field `hrrr_nearest_map_positions: tuple = field(default=(), kw_only=True)`
  - Field `wrf: object | None = None`
  - Field `wrf_atmosphere_path: str | None = None`
  - Field `wrf_atmosphere_coordinates: str | None = None`
  - Field `wrf_exchange: object | None = None`
  - Field `visualization_fields: tuple[str, ...] = VISUALIZATION_FIELDS`
  - Field `wall_log: WallLogCorrection | None = None`
  - Field `domain_rotation_deg: float = 0.0`
  - Field `street_intersection_flag: int = 0`
  - Field `xml_compatibility: tuple[str, ...] = ()`
  - Field `native_sensor_math: bool = False`
  - Field `native_wake_math: bool = False`
  - Field `native_building_math: bool = False`
  - Field `meteorology_backend: str = 'cpu'`
  - Field `native_staircase_metrics: bool = False`
  - Field `native_netcdf_inputs: bool = False`
  - Field `wrf_zero_steps: bool = False`
  - Field `coefficient_flag: int = 0`
  - `iter_steps(self)`
  - `nonlocal_turbulence_state(self, fields)`
  - `iter_results(self, *, initialize_only=False, include_gradients=False)`
  - `manifest(self)`
  - `initialize_geometry(self, *, prepare_mixing=False)`
  - `prepare(self)`
  - `initialize(self)`
  - `run(self, *, initialize_only=False, include_gradients=False, initial_multiplier=None, previous_turbulence=None)`
  - `solve(self, workspace, *, include_gradients=False, initial_multiplier=None, previous_turbulence=None)`
- `read_wind_case(path, *, poisson=None, poisson_backend='cpu', compute_turbulence=True, wrf_timeout=None, turbulence_gradient='cpu', native_sensor_math=False, native_wake_math=False, native_turbulence_math=False, native_building_math=False, wrf_startup_only=False, meteorology_backend='cpu', legacy_turbulence_xml=False, legacy_wrf_protocol=False)`

### `plumeswift.wind_compare`

[Source](../python_qes/plumeswift/wind_compare.py)

Wind workspace and solver diagnostics ported from QES comparison tools.

- `compare_velocity(reference, candidate, *, precision='native')`
- `compare_wind_files(reference, candidate, *, precision='native', by_index=False, native_layout=False, include_speed=True)`
- `compare_solvers(path, *, backends=('cpu-sor',), precision='native')`
- `compare_legacy_solvers(path, *, backends, precision='native')`

### `plumeswift.wind_compare_text`

[Source](../python_qes/plumeswift/wind_compare_text.py)

Text layouts used by the native QES wind comparison commands.

- `format_solver_tables(candidates)`
- `format_solver_series(result)`
- `format_workspace_stats(result)`

### `plumeswift.wind_projection`

[Source](../python_qes/plumeswift/wind_projection.py)

QES Euler-Lagrange wind correction separated from geometry/Poisson solvers.

- `rhs_from_workspace(workspace, physics=None, *, array_module=np)`
- `corrected_velocity(workspace, multiplier, physics=None, *, array_module=np, dtype=float)`
- `zero_solid_faces(u, v, w, flags, *, array_module=np)`
- `project(workspace, solver, physics=None, *, initial=None)`

### `plumeswift.wind_replay`

[Source](../python_qes/plumeswift/wind_replay.py)

Reproject stored QES wind fields with their original cut-cell stencil.

- `replay_wind(input_path, output_path, *, solver=None, wall_roughness=None, native_input=False)`

### `plumeswift.wind_run`

[Source](../python_qes/plumeswift/wind_run.py)

Write selected standalone wind products from the shared wind driver.

- `legacy_wind_backend(solver_type, *, cpu_serial=False)`
- `inspect_wind_input(path, *, output=None)`
- `write_wind_run(case, *, workspace_path=None, turbulence_path=None, visualization_path=None, terrain_obj=None, native_workspace_format=False, native_turbulence_format=False, native_visualization_format=False, initialize_only=False)`

### `plumeswift.wind_visualization`

[Source](../python_qes/plumeswift/wind_visualization.py)

Cell-centred visualization NetCDF with the public QES field schema.

- `validate_output_fields(fields)`
- `read_output_fields(options, *, validate_names=True)`
- `cell_fields(workspace, fields)`
- `visualization_series(path, workspace, fields, *, time_origin)`
- `native_visualization_series(path, domain, fields, *, origin)`

### `plumeswift.wind_wall_log`

[Source](../python_qes/plumeswift/wind_wall_log.py)

Post-projection logarithmic wall correction from QES Wall.cpp (GPL-3).

- Class `WallLogCorrection`
  - Field `roughness: float`

### `plumeswift.wind_workspace_output`

[Source](../python_qes/plumeswift/wind_workspace_output.py)

NetCDF layout of the original QES WINDSOutputWorkspace (GPL-3.0).

- `native_workspace_coordinates(domain)`
- `native_workspace_fields(result)`
- `native_workspace_series(path, domain, *, origin)`

### `plumeswift.wind_xml`

[Source](../python_qes/plumeswift/wind_xml.py)

Normalize historical XML names used in the distributed QES wind examples.

- `normalize_zero_wrf_startup(root)`
- `normalize_legacy_turbulence(params, *, cutcell)`
- `normalize_sensor_xml(sensor)`
- `normalize_wind_xml(root)`

### `plumeswift.wrf`

[Source](../python_qes/plumeswift/wrf.py)

WRF/SFIRE grid, wind snapshots, and atmospheric stations.

- `frame_checksum(*fields)`
- Class `FireMeshGrid`
  - Field `path: str`
  - Field `cells: tuple`
  - Field `spacing: tuple`
  - Field `terrain: np.ndarray`
  - `read(cls, path)`
  - `describe(self)`
  - `domain(self, halo, widths=None)`
  - `mesh(self, domain)`
  - `cut_geometry(self, domain, halo)`
- Class `WRFChecksumError`
- Class `FireMeshInput`
  - Field `heights: np.ndarray`
  - Field `u: np.ndarray`
  - Field `v: np.ndarray`
  - Field `w: np.ndarray`
  - Field `record: int`
  - `read(cls, path)`
  - `read_record(cls, nc, grid, record)`
  - `describe(self)`
  - `initialize(self, workspace, halo, *, native_math=False)`
- `atmospheric_stations(path, domain, halo, origin, zone, sample, *, coordinate_mode='geographic', stored=False)`

### `plumeswift.wrf_dem`

[Source](../python_qes/plumeswift/wrf_dem.py)

Export WRF or WRF-SFIRE terrain and measured wind profiles to a QES project.

- `convert_wrf_dem(wrf_path, dem_path, *, sample=12, nz=300)`

### `plumeswift.wrf_diagnostic`

[Source](../python_qes/plumeswift/wrf_diagnostic.py)

Round-trip diagnostic from the upstream WRF coupling test.

- `run_coupling_diagnostic(path, *, rounds=10, timeout=60.0)`

### `plumeswift.wrf_exchange`

[Source](../python_qes/plumeswift/wrf_exchange.py)

QES-Winds file handshake with a running WRF/SFIRE peer.

- `fire_height_winds(workspace, fire_height, halo_cells, dz)`
- `read_wrf_startup(path, *, timeout=60.0, record_time=False)`
- Class `WRFExchange`
  - Field `grid: FireMeshGrid`
  - Field `steps: int`
  - Field `timeout: float = 60.0`
  - Field `poll_interval: float = 0.25`
  - Field `legacy_protocol: bool = False`
  - `describe(self)`
  - `open(cls, path, steps, *, timeout=60.0, legacy_protocol=False)`
  - `receive(self, frame, *, record_time=True)`
  - `publish(self, frame, workspace, halo)`
  - `finish(self)`

### `plumeswift.wrf_input_demo`

[Source](../python_qes/plumeswift/wrf_input_demo.py)

Positional NetCDF input diagnostic from scratch/wrfInputTest.cpp.

- `pressure_dump(path)`

### `plumeswift.xml_demo`

[Source](../python_qes/plumeswift/xml_demo.py)

The two standalone QES scratch XML parser examples.

- Class `MissingValue`
- Class `RepeatedValueError`
- `read_xml_demo(path)`
- `read_simulation_demo(path)`
- `run_xml_demo(path, *, simulation=False)`
