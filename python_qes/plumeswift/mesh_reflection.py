"""Python port of QES WallReflection_TriMesh::rayTraceReflect (GPL-3).

The 0.1 m pre-impact offset is an original model constant, not a new tolerance.
Iteration replaces unbounded C++ recursion without changing the bounce map.
Mesh construction, intersection, endpoint selection and reflection are separate.
"""
from dataclasses import dataclass
import numpy as np

from .numerics.vector3 import length32, reflect32


@dataclass(frozen=True)
class MeshReflection:
    mesh: object
    max_bounces: int = 256

    def __post_init__(self):
        if self.max_bounces < 1:
            raise ValueError('Reflection bounce limit must be positive')

    def trace(self, origin, displacement, fluctuation):
        x, u, fluct = (np.array(a, dtype=float, copy=True) for a in (origin, displacement, fluctuation))
        if any(a.shape != (3,) for a in (x, u, fluct)) or not np.isfinite([x, u, fluct]).all():
            raise ValueError('Reflection requires finite three-component vectors')
        end = x+u
        count = 0
        while True:
            hit = self.mesh.ray_hit(x, u)
            if hit is None or hit.distance > np.linalg.norm(u):
                return end, fluct, count
            if count >= self.max_bounces:
                raise RuntimeError('Triangle reflection exceeded bounce limit; no fallback trajectory applied')
            point = hit.point-.1*u/np.linalg.norm(u)
            remaining = end-point
            # Algebraically the same r*reflect(V2/r,N) as the native method.
            end = point+remaining-2*np.dot(remaining, hit.normal)*hit.normal
            fluct -= 2*np.dot(fluct, hit.normal)*hit.normal
            x, u = point, end-point
            count += 1

    def trace_native(self, origin, displacement, fluctuation):
        x, u, fluct = (np.asarray(a, dtype=np.float32) for a in (origin, displacement, fluctuation))
        end = x+u
        for count in range(self.max_bounces+1):
            hit = self.mesh.ray_hit(x, u, native_math=True)
            length = length32(u)
            if hit is None or hit.distance > length:
                return end, fluct, count
            if count == self.max_bounces:
                raise RuntimeError('Triangle reflection exceeded bounce limit; no fallback trajectory applied')
            point = hit.point-np.float32(.1)*u/length
            remaining = end-point
            travel = length32(remaining)
            if not travel:
                return end, fluct, count
            end = point+travel*reflect32(remaining/travel, hit.normal)
            fluct = reflect32(fluct, hit.normal)
            x, u = point, end-point


class TriangleWalls:
    """ParticleIntegrator adapter with the original solid-endpoint gate.

    endpoint_solid(points) returns a boolean per endpoint (native flags 0/2,
    including below-ground points). Domain exits remain a separate policy.
    Like QES this deliberately bypasses reflection when the endpoint is fluid,
    even if the trajectory crosses a thin obstacle. No hidden trajectory policy.
    """
    def __init__(self, reflection, endpoint_solid):
        self.reflection, self.endpoint_solid = reflection, endpoint_solid

    def reflect(self, positions, displacement, fluctuations):
        positions, displacement, fluctuations = (np.asarray(v, dtype=float) for v in (positions, displacement, fluctuations))
        if positions.ndim != 2 or positions.shape[1:] != (3,) or displacement.shape != positions.shape or fluctuations.shape != positions.shape:
            raise ValueError('Particle vectors require matching N x 3 arrays')
        if not np.isfinite([positions, displacement, fluctuations]).all():
            raise ValueError('Non-finite particle reflection vectors')
        end, fluct = positions+displacement, fluctuations.copy()
        mask = np.asarray(self.endpoint_solid(end), dtype=bool)
        if mask.shape != (len(end),):
            raise ValueError('Solid endpoint selector must return N booleans')
        for index in np.flatnonzero(mask):
            end[index], fluct[index], _ = self.reflection.trace(positions[index], displacement[index], fluct[index])
        return end, fluct


class NativeTriangleWalls(TriangleWalls):
    """Use the original rounded endpoint and displacement for mesh reflection."""

    def reflect_native(self, positions, displacement, fluctuations):
        movement = np.asarray(displacement, dtype=np.float32)
        end = np.asarray(positions, dtype=np.float32)+movement
        fluct = np.array(fluctuations, dtype=np.float32, copy=True)
        for index in np.flatnonzero(self.endpoint_solid(end)):
            origin = end[index]-movement[index]
            end[index], fluct[index], _ = self.reflection.trace_native(origin, movement[index], fluct[index])
        # The native mesh wrapper sets the state to ACTIVE before axis checks.
        return end, fluct, True
