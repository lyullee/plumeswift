"""Explicit dependency injection; imports do not construct models or read env vars."""
from dataclasses import dataclass,field
from typing import Any,Callable
from .contracts import NumericPolicy,PhysicalCoefficients

class Registry:
    def __init__(self): self._factories: dict[str,Callable]={}
    def register(self,name: str,factory: Callable):
        if name in self._factories:raise ValueError(f"Duplicate component: {name}")
        self._factories[name]=factory
    def create(self,name: str,**kwargs):
        if name not in self._factories:raise KeyError(f"Unknown component {name}; available={sorted(self._factories)}")
        def resolve(value):
            if isinstance(value,ComponentChoice):return self.create(value.implementation,**value.parameters)
            if isinstance(value,dict):return {k:resolve(v)for k,v in value.items()}
            if isinstance(value,list):return [resolve(v)for v in value]
            if isinstance(value,tuple):return tuple(resolve(v)for v in value)
            return value
        return self._factories[name](**{k:resolve(v)for k,v in kwargs.items()})
    def names(self)->tuple[str,...]:return tuple(sorted(self._factories))

@dataclass(frozen=True)
class ComponentChoice:
    implementation: str
    parameters: dict[str,Any]=field(default_factory=dict)

@dataclass(frozen=True)
class AssemblyConfig:
    """Select providers separately from numerical tolerances and coefficients."""
    thermodynamics: ComponentChoice
    poisson_solver: ComponentChoice
    numeric: NumericPolicy=field(default_factory=NumericPolicy)
    physics: PhysicalCoefficients=field(default_factory=PhysicalCoefficients)
    seed: int=20261107

@dataclass
class Components:
    config: AssemblyConfig
    thermo: Any
    poisson: Any
    def manifest(self):
        from dataclasses import asdict
        return {"configuration":asdict(self.config),"thermodynamics":self.thermo.describe(),
                "runtime":"Python; no native QES executable or shared library"}

def default_registry():
    from .thermo import LegacyEquilibrium,CoolPropEquilibrium
    from .numerics.poisson import RedBlackSOR,KrylovPoisson
    from .thermodynamics import CoolPropProperties,AmagatMixing,TraceWaterEnergy
    from .domain import Domain
    from .wind_case import WindCase
    from .meteorology import SensorProfile,Station
    from .turbulence import LocalTurbulence
    from .building_pipeline import BuildingPipeline,UpwindClosure,WakeClosure,SidewallClosure,RooftopClosure,StreetCanyonClosure,ModifiedStreetCanyonClosure
    registry=Registry()
    registry.register("thermo.legacy",LegacyEquilibrium)
    registry.register("thermo.coolprop",CoolPropEquilibrium)
    registry.register("poisson.redblack",RedBlackSOR)
    registry.register("poisson.krylov",KrylovPoisson)
    from .numerics.cuda_poisson import CudaRedBlackSOR
    from .numerics.cuda_dynamic_legacy import CudaDynamicLegacy
    from .numerics.cuda_device_parallel import CudaDeviceParallel
    registry.register('poisson.cuda_sor', CudaRedBlackSOR)
    registry.register('poisson.cuda_dynamic_legacy', CudaDynamicLegacy)
    registry.register('poisson.cuda_device_parallel', CudaDeviceParallel)
    registry.register("properties.coolprop",CoolPropProperties)
    registry.register("mixing.amagat",AmagatMixing)
    registry.register("moisture.trace",TraceWaterEnergy)
    for name,factory in [('grid.qes',Domain),('wind.case',WindCase),('met.profile',SensorProfile),('met.station',Station),
                         ('turbulence.local',LocalTurbulence),('building.pipeline',BuildingPipeline),
                         ('building.upwind',UpwindClosure),('building.wake',WakeClosure),
                         ('building.sidewall',SidewallClosure),('building.rooftop',RooftopClosure)]:
        registry.register(name,factory)
    registry.register('building.street_canyon',StreetCanyonClosure)
    registry.register('building.street_canyon_modified',ModifiedStreetCanyonClosure)
    from .mixing_length import SerialMixing,NetCDFMixing
    registry.register('mixing_length.serial',SerialMixing)
    registry.register('mixing_length.netcdf',NetCDFMixing)
    from .ray_mixing import RayMixing
    registry.register('mixing_length.ray',RayMixing)
    from .terrain import DEMTerrain,GridTerrain
    from .measured_profile import MeasuredProfile
    registry.register('terrain.dem',DEMTerrain)
    registry.register('terrain.height_array',GridTerrain)
    registry.register('met.measured',MeasuredProfile)
    from .hrrr import HRRRInput, HRRRProfile
    registry.register('met.hrrr', HRRRInput)
    registry.register('met.hrrr_profile', HRRRProfile)
    from .wrf import FireMeshInput
    registry.register('met.wrf_fire_mesh', FireMeshInput.read)
    from .wrf_exchange import WRFExchange
    registry.register('met.wrf_exchange', WRFExchange.open)
    from .triangle_mesh import TriangleMesh
    from .mesh_reflection import MeshReflection,TriangleWalls
    registry.register('mesh.triangles',TriangleMesh)
    registry.register('reflection.mesh',MeshReflection)
    registry.register('walls.triangle',TriangleWalls)
    from .wind_wall_log import WallLogCorrection
    registry.register('wind.wall_log', WallLogCorrection)
    from .mesh_mixing import LegacyMeshMixing
    registry.register('mixing_length.legacy_mesh',LegacyMeshMixing)
    from .vegetation import HomogeneousCanopy,Vegetation
    registry.register('canopy.homogeneous',HomogeneousCanopy)
    from .canopy_windbreak import Windbreak
    registry.register('canopy.windbreak',Windbreak)
    from .canopy_tree import IsolatedTree
    registry.register('canopy.isolated_tree',IsolatedTree)
    from .canopy_roc import RowCanopy
    registry.register('canopy.row_organized',RowCanopy)
    registry.register('vegetation.qes',Vegetation)
    from .fire_grid import FireGrid
    from .fire_potential import HeatPotential
    registry.register('fire.level_set', FireGrid)
    registry.register('fire.heat_potential', HeatPotential.read)
    return registry

def assemble(config: AssemblyConfig,registry: Registry|None=None)->Components:
    registry=registry or default_registry()
    thermo=registry.create(config.thermodynamics.implementation,**config.thermodynamics.parameters)
    poisson=registry.create(config.poisson_solver.implementation,policy=config.numeric,**config.poisson_solver.parameters)
    return Components(config,thermo,poisson)
