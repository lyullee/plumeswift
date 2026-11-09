"""Composable source geometry and release schedules; SI mass units throughout."""
from dataclasses import InitVar, dataclass, field
from typing import Protocol
import numpy as np

class Geometry(Protocol):
    def sample(self,count:int,rng:np.random.Generator)->np.ndarray: ...

@dataclass(frozen=True)
class Point:
    position:tuple[float,float,float]
    def sample(self,count,rng):return np.broadcast_to(self.position,(count,3)).astype(float).copy()

@dataclass(frozen=True)
class Cube:
    minimum:tuple[float,float,float]
    maximum:tuple[float,float,float]
    validate_dimensions: InitVar[bool] = True
    def __post_init__(self, validate_dimensions):
        if validate_dimensions and np.any(np.array(self.maximum) <= np.array(self.minimum)):
            raise ValueError("Cube must have positive volume")
    def sample(self, count, rng):
        low, high = np.asarray((self.minimum, self.maximum), dtype=float)
        return rng.random((count, 3))*(high-low)+low

@dataclass(frozen=True)
class Line:
    start:tuple[float,float,float]
    end:tuple[float,float,float]
    def sample(self,count,rng):return np.array(self.start)+rng.random((count,1))*(np.array(self.end)-self.start)

@dataclass(frozen=True)
class SphereShell:
    centre:tuple[float,float,float]
    radius:float
    validate_dimensions: InitVar[bool] = True
    def __post_init__(self, validate_dimensions):
        if validate_dimensions and self.radius <= 0:
            raise ValueError("Radius must be positive")
    def sample(self,count,rng):
        direction=rng.normal(size=(count,3));norm=np.linalg.norm(direction,axis=1)
        if np.any(norm==0):raise FloatingPointError("Zero random sphere direction")
        return self.centre+self.radius*direction/norm[:,None]

@dataclass
class ContinuousRelease:
    mass_rate_kg_s:float
    particles_per_second:float
    start_s:float=0.
    end_s:float=float('inf')
    def __post_init__(self):
        if self.mass_rate_kg_s<0 or self.particles_per_second<=0 or self.end_s<=self.start_s:raise ValueError("Invalid release")
    def count_and_mass(self,t,dt):
        # Cumulative counting keeps total mass independent of chosen main dt.
        a=max(0.,min(t,self.end_s)-self.start_s);b=max(0.,min(t+dt,self.end_s)-self.start_s)
        def cumulative_count(seconds):
            value=seconds*self.particles_per_second
            nearest=np.rint(value)
            if abs(value-nearest)<=max(1e-10,abs(value)*16*np.finfo(float).eps):value=nearest
            return int(np.floor(value))
        n=cumulative_count(b)-cumulative_count(a)
        return n,self.mass_rate_kg_s*(b-a)

@dataclass
class InstantaneousRelease:
    mass_kg:float
    particles:int
    time_s:float=0.
    def __post_init__(self):
        if self.mass_kg<0 or self.particles<1:raise ValueError("Invalid instantaneous release")
    def count_and_mass(self,t,dt):return (self.particles,self.mass_kg)if t<=self.time_s<t+dt else (0,0.)


@dataclass(frozen=True)
class StepRelease:
    """Native fixed particle count and mass per active integration step."""
    particles: int
    mass_kg: float
    start_s: float
    end_s: float

    def __post_init__(self):
        if (not isinstance(self.particles, (int, np.integer)) or self.particles < 0 or self.mass_kg < 0 or
                (self.particles == 0 and self.mass_kg != 0) or
                not np.isfinite([self.mass_kg, self.start_s, self.end_s]).all() or self.start_s > self.end_s):
            raise ValueError('Invalid fixed-count release window')

    def count_and_mass(self, t, dt):
        return (self.particles, self.mass_kg) if self.start_s <= t <= self.end_s else (0, 0.)


@dataclass
class Source:
    geometry:Geometry
    release:ContinuousRelease|InstantaneousRelease|StepRelease
    source_id:int=0
    initial_drift:tuple[float,float,float]=(0.,0.,0.)
    diameter_um:float=0.
    density_kg_m3:float=0.
    _pending_mass:float=field(default=0.,init=False,repr=False)
    def emit(self,t,dt,rng):
        count,mass=self.release.count_and_mass(t,dt);self._pending_mass+=mass
        if (not count and self._pending_mass>0 and isinstance(self.release,ContinuousRelease)
                and t<self.release.end_s<=t+dt):count=1
        if not count:return np.empty((0,3)),np.empty(0)
        positions=self.geometry.sample(count,rng)
        if not np.isfinite(positions).all():raise ValueError("Invalid source positions")
        masses=np.full(count,self._pending_mass/count);self._pending_mass=0.
        return positions,masses
