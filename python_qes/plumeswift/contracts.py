"""Small, SI-only contracts between independently replaceable components."""
from dataclasses import dataclass,field
from typing import Any, ClassVar, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

Array=NDArray[np.float64]

class ModelDomainError(ValueError):
    """Unsupported physical state; callers must not silently substitute models."""

class UnportedFeature(NotImplementedError):
    """A requested upstream feature has not passed Python migration yet."""

@dataclass(frozen=True)
class ThermoState:
    concentration: Array # kg H2 / m3 mixture
    temperature: Array # K
    density: Array # kg mixture / m3
    volume_fraction: Array # gas-phase H2 volume fraction, 0..1
    vapor_quality: Array # H2 vapor mass / total H2 mass

@runtime_checkable
class Thermodynamics(Protocol):
    name: str
    ambient_density: float
    def evaluate(self,concentration: Array)->ThermoState: ...
    def describe(self)->dict[str,Any]: ...

@runtime_checkable
class PureFluidProperties(Protocol):
    hydrogen_fluid: str
    def enthalpy(self,fluid: str,temperature: Array,pressure: float,phase: str|None=None)->Array: ...
    def density(self,fluid: str,temperature: Array,pressure: float)->Array: ...
    def saturation(self,fluid: str,pressure: float,quality: float)->dict[str,float]: ...
    def describe(self)->dict[str,Any]: ...

@runtime_checkable
class MixtureRule(Protocol):
    def density_and_volume_fraction(self,hydrogen_mass_fraction: Array,rho_hydrogen: Array,rho_air: Array)->tuple[Array,Array]: ...
    def describe(self)->dict[str,Any]: ...

@runtime_checkable
class WallReflection(Protocol):
    def reflect(self,positions: Array,displacement: Array,fluctuations: Array)->tuple[Array,Array]: ...

@runtime_checkable
class Observer(Protocol):
    def observe(self,time_start: float,dt: float,positions: Array,masses: Array)->None: ...

@runtime_checkable
class FieldProvider(Protocol):
    """Wind m/s, Reynolds stress m2/s2, stress divergence m/s2, CoEps m2/s3."""
    dx: float
    dy: float
    zf: Array
    mixlength: Array
    def sample_wind(self,positions: Array)->Array: ...
    def cell_indices(self,positions: Array)->tuple[Array,Array,Array]: ...
    def sample_turbulence(self,positions: Array)->tuple[Array,Array,Array,Array|None]: ...
    def sample_turbulence_and_distance(self,positions: Array)->tuple[Array,Array,Array,Array|None,Array]: ...

@dataclass(frozen=True)
class SolverReport:
    converged: bool
    iterations: int
    residual_linf: float
    update_linf: float|None
    wall_s: float
    method: str

@runtime_checkable
class PoissonSolver(Protocol):
    def solve(self,stencil: Any,rhs: Array,initial: Array|None=None)->tuple[Array,SolverReport]: ...

@dataclass(frozen=True)
class NumericPolicy:
    """Numerical controls, kept separate from physical coefficients."""
    relative_tolerance: float=1e-7
    absolute_tolerance: float=1e-9
    max_iterations: int=10000
    stress_invariant_tolerance: float=1e-5
    minimum_iterations: ClassVar[int] = 1

    def __post_init__(self):
        if not 0<self.relative_tolerance<1 or self.absolute_tolerance<=0 or self.max_iterations<self.minimum_iterations:
            raise ValueError("Invalid numerical convergence policy")

@dataclass(frozen=True)
class PhysicalCoefficients:
    """Preserved original defaults; changes must be intentional and recorded."""
    alpha_horizontal: float=1.
    alpha_vertical: float=1.
    gravity: float=9.80665
    lh2_drag_coefficient: float=.5
    lh2_drag_length_m: float=1.
    lh2_ground_reaction_height_m: float=.5
    def __post_init__(self):
        positive=[self.alpha_horizontal,self.alpha_vertical,self.gravity,self.lh2_drag_coefficient,self.lh2_drag_length_m]
        if not all(np.isfinite(x)and x>0 for x in positive)or self.lh2_ground_reaction_height_m<0:
            raise ValueError("Invalid physical coefficient or length scale")

@dataclass
class DiagnosticLog:
    events: list[dict[str,Any]]=field(default_factory=list)
    def record(self,code: str,**values):
        self.events.append({"code":code,**values})
