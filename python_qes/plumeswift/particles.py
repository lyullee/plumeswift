"""Python particle state and orchestrator, with replaceable numerical services."""
from concurrent.futures import wait
from copy import copy
from dataclasses import dataclass
import numpy as np
from .numerics.langevin import implicit_gle,realizable
from .numerics.native_gle import native_implicit_gle, native_realizable
from .contracts import NumericPolicy
from .boundaries import NativeDomainBoundaries
from .cpu_plume_random import NativeCPUStreams
from .native_release_slots import NativeReleaseSlots
from .particle_ids import ParticleIDs
from .particle_slots import first_free_slots, reserve_particle_slots
from .walls import NoWalls

TRACKING_DTYPE = np.dtype([
    ('release_s', 'f8'), ('source_id', 'i8'), ('diameter_um', 'f8'),
    ('initial_position', 'f8', (3,)), ('delta_fluctuation', 'f8', (3,)),
])

@dataclass
class ParticleBatch:
    positions:np.ndarray
    mass:np.ndarray
    fluctuations:np.ndarray
    previous_stress:np.ndarray
    drift:np.ndarray
    identifiers:np.ndarray
    tracking:np.ndarray|None=None
    material:np.ndarray|None=None
    slots:np.ndarray|None=None
    previous_fluctuations:np.ndarray|None=None
    specific_enthalpy:np.ndarray|None=None
    carrier_air_mass:np.ndarray|None=None
    mixture_mass_ratio:np.ndarray|None=None
    @classmethod
    def empty(cls):return cls(np.empty((0,3)),np.empty(0),np.empty((0,3)),np.empty((0,3,3)),np.empty((0,3)),np.empty(0,dtype=np.int64))
    def retain(self,keep):
        for name in self.__dataclass_fields__:
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, values[keep])
    def append(self,position,mass,fluctuation,stress,drift,identifiers, *, source=None, release_s=0., slots=None,
               previous_fluctuation=None):
        columns = dict(positions=position, mass=mass, fluctuations=fluctuation,
                       previous_stress=stress, drift=drift, identifiers=identifiers)
        for name,value in columns.items():
            setattr(self,name,np.concatenate([getattr(self,name),value],axis=0))
        if self.slots is not None:
            self.slots = np.concatenate((self.slots, slots))
        if self.previous_fluctuations is not None:
            self.previous_fluctuations = np.concatenate((self.previous_fluctuations, previous_fluctuation))
        if self.tracking is not None:
            new = np.zeros(len(mass), dtype=TRACKING_DTYPE)
            new['release_s'] = release_s
            new['source_id'] = source.source_id
            new['diameter_um'] = np.where(mass > 0, source.diameter_um, 0.)
            new['initial_position'] = position
            self.tracking = np.concatenate((self.tracking, new))
        if self.material is not None:
            physical = np.array([1e-6*source.diameter_um, source.density_kg_m3])
            self.material = np.concatenate((self.material, (mass[:, None] > 0)*physical))
        if self.specific_enthalpy is not None:
            self.specific_enthalpy = np.concatenate((self.specific_enthalpy,
                                                     np.full(len(mass), source.specific_enthalpy_j_kg)))
        if self.carrier_air_mass is not None:
            self.carrier_air_mass = np.concatenate((self.carrier_air_mass, mass * source.carrier_air_per_h2))
        if self.mixture_mass_ratio is not None:
            self.mixture_mass_ratio = np.concatenate((self.mixture_mass_ratio, np.zeros(len(mass))))

def courant_timestep(distance,absolute_velocity,remaining,main_dt,courant,min_spacing):
    """Same adaptive distance/speed logic as PLUMEGeneralData.cpp."""
    if courant==0:return np.asarray(remaining).copy()
    speed=np.linalg.norm(absolute_velocity,axis=1)
    cn=np.where(distance>3*speed*main_dt,min(2*courant,1.),courant)
    dt=np.divide(cn*min_spacing,speed,out=np.full_like(speed,np.inf),where=speed>0)
    return np.where(distance>6*speed*main_dt,remaining,np.minimum(remaining,dt))


def native_courant_timestep(distance, absolute_velocity, remaining, main_dt, courant, min_spacing):
    """Float storage and expression order of the CPU QES vector overload."""
    remainder = np.asarray(remaining, dtype=np.float32)
    courant = np.float32(courant)
    if courant == 0:
        return remainder.copy()
    u, v, w = np.asarray(absolute_velocity, dtype=np.float32).T
    speed = np.sqrt(u*u+v*v+w*w)
    # The upstream distance thresholds use double literals.
    speed_double = speed.astype(np.float64)
    interval = float(np.float32(main_dt))
    distance = np.asarray(distance, dtype=np.float32)
    cn = np.where(distance > 3*speed_double*interval, min(2*float(courant), 1.), courant).astype(np.float32)
    step = np.divide(cn*np.float32(min_spacing), speed, out=np.full_like(speed, np.inf), where=speed > 0)
    return np.where(distance > 6*speed_double*interval, remainder, np.minimum(remainder, step))


def _substeps(alive, remaining, tolerance, serial):
    if serial:
        for particle in range(len(alive)):
            ids = np.array([particle])
            for _ in range(100001):
                if not alive[particle] or remaining[particle] <= tolerance:
                    break
                yield ids
            else:
                raise RuntimeError('Particle substep limit exceeded')
        return
    for _ in range(100001):
        ids = np.flatnonzero(alive & (remaining > tolerance))
        if not len(ids):
            return
        yield ids
    raise RuntimeError('Particle substep limit exceeded')


class ParticleIntegrator:
    def __init__(self,fields,boundaries,walls,rng,policy=None,scheme='backward-euler',courant=.25,max_velocity=1e6,
                 *, main_step=None, min_spacing=None, rogue_velocity=None, deposition=None, executor=None,
                 native_arithmetic=False, sampling_in_grid=None, sampling_pool=None, compiled_gle=False,
                 vertical_transport=None):
        self.fields=fields;self.boundaries=boundaries;self.walls=walls;self.rng=rng
        if self.fields.mixlength is None:raise ValueError("Particle timestep requires an explicit mixing-length field")
        self.policy=policy or NumericPolicy();self.scheme=scheme;self.courant=courant;self.max_velocity=max_velocity
        self.main_step = main_step
        self.min_spacing = min_spacing
        self.rogue_velocity = rogue_velocity
        self.deposition = deposition
        self.executor = executor
        self.sampling_in_grid = sampling_in_grid
        if sampling_pool is not None and sampling_pool.fields is not fields:
            raise ValueError('Sampling pool must use the integrator workspace')
        self.sampling_pool = sampling_pool
        self.vertical_transport = vertical_transport
        self.native_math = native_arithmetic or isinstance(rng, NativeCPUStreams)
        self.native_gle = self.native_math and scheme == 'backward-euler'
        self.compiled_gle = compiled_gle and not self.native_math
        self.escaped_mass_kg=0.;self.escaped_particles=0
        self.rogue_mass_kg=0.;self.rogue_particles=0
    def advance(self,batch,dt, *, slot_capacity=None, capture=None):
        if batch.slots is not None:
            batch.retain(np.argsort(batch.slots))
        replay = self.rng if isinstance(self.rng, NativeCPUStreams) else None
        owners = None if replay is None else replay.owners(batch.slots, slot_capacity)
        if self.executor is not None and owners is not None and owners.size:
            alive, rogue, cutoff = self._advance_workers(batch, dt, owners)
        else:
            alive, rogue, cutoff = self._integrate(batch, dt, self.rng, owners=owners)
        escaped = ~alive & ~rogue & ~cutoff
        self.escaped_mass_kg+=float(batch.mass[escaped].sum());self.escaped_particles+=int(escaped.sum())
        self.rogue_mass_kg+=float(batch.mass[rogue].sum());self.rogue_particles+=int(rogue.sum())
        if capture is not None:
            capture(batch)
        batch.retain(alive)

    def _advance_workers(self, batch, dt, owners):
        limits = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1])+1, owners.size]
        jobs = []
        for start, end in zip(limits[:-1], limits[1:], strict=True):
            span = slice(start, end)
            worker = owners[start]
            part = ParticleBatch(**{key: value[span].copy() for key, value in vars(batch).items()
                                    if value is not None})
            stream = copy(self.rng.streams[worker])
            deposits = []

            def record(cells, result, *, target=deposits):
                target.append((cells, result))

            future = self.executor.submit(self._integrate, part, dt, stream, record=record)
            jobs.append((span, worker, part, stream, deposits, future))
        wait([job[-1] for job in jobs])
        masks = [job[-1].result() for job in jobs]
        # Commit only after all workers succeed. Replay deposition contributions
        # in slot order to preserve the original floating-point accumulation.
        streams = list(self.rng.streams)
        for span, worker, part, stream, deposits, _ in jobs:
            streams[worker] = stream
            for key, value in vars(part).items():
                if value is not None:
                    getattr(batch, key)[span] = value
            for cells, result in deposits:
                self.deposition.record(cells, result)
        self.rng.streams = tuple(streams)
        return tuple(np.concatenate(column) for column in zip(*masks, strict=True))

    def _integrate(self, batch, dt, rng, *, owners=None, record=None):
        remaining = np.full(len(batch.mass), dt, dtype=np.float32 if self.native_math else float)
        tolerance = 0 if self.native_math else dt*1e-10
        step = native_courant_timestep if self.native_math else courant_timestep
        alive = np.ones(len(batch.mass), dtype=bool)
        rogue = np.zeros(len(batch.mass), dtype=bool)
        cutoff = np.zeros(len(batch.mass), dtype=bool)
        spacing=self.min_spacing or min(self.fields.dx,self.fields.dy,float(np.min(np.diff(self.fields.zf))))
        reflect_native = getattr(self.walls, 'reflect_native', None)
        defer_wind_validation = getattr(self.fields, 'defer_wind_validation', False)
        # Native setInactive evaluates an invalid endpoint after the GLE step,
        # so unrelated particles still advance and the random stream is retained.
        wall_rejects_invalid = (self.native_math and self.deposition is None
                                and getattr(self.walls, 'rejects_invalid_endpoints', False))
        native_zero_step = self.native_gle and self.rogue_velocity is not None
        for ids in _substeps(alive, remaining, tolerance, batch.slots is not None):
            p = batch.positions[ids]
            if self.sampling_in_grid is not None:
                in_grid = self.sampling_in_grid(p)
                alive[ids[~in_grid]] = False
                ids, p = ids[in_grid], p[in_grid]
                if not len(ids):
                    continue
            samples = None
            if self.sampling_pool is not None:
                if self.sampling_pool.fields is not self.fields:
                    raise ValueError('Sampling pool cannot use stale wind or turbulence fields')
                wind, *samples = self.sampling_pool.sample(p)
            else:
                wind = self.fields.sample_wind(p)
            mean = wind+batch.drift[ids]
            invalid_wind = not np.isfinite(mean).all()
            boundary_rejects_invalid = False
            if (invalid_wind and native_zero_step and self.courant == 0 and type(self.walls) is NoWalls
                    and self.deposition is None and isinstance(self.boundaries, NativeDomainBoundaries)):
                # With a fixed step and no wall lookup, an exiting axis can
                # remove an infinite endpoint before any subsequent axis.
                invalid = ~np.isfinite(mean)
                affected = invalid.any(axis=1)
                axes = invalid[affected].argmax(axis=1)
                exiting = np.array([policy.mode == 'exiting' for policy in self.boundaries.axes])
                boundary_rejects_invalid = np.all(np.isinf(mean[affected, axes]) & exiting[axes])
            if invalid_wind and not (defer_wind_validation or wall_rejects_invalid or boundary_rejects_invalid):
                if not native_zero_step or self.courant == 0 or np.isnan(mean).any():
                    raise FloatingPointError('Non-finite wind sampled at a particle position')
            previous = batch.fluctuations[ids]
            if self.native_math:
                mean = mean.astype(np.float32)
                previous = previous.astype(np.float32)
            tau, flux, ce, viscosity, distance = (self.fields.sample_turbulence_and_distance(p)
                                                 if samples is None else samples)
            if self.vertical_transport is not None:
                mean[:, 2] += self.vertical_transport(p, wind[:, 2], batch.drift[ids, 2], tau[:, 2, 2])
            local_dt = step(distance, abs(mean)+abs(previous), remaining[ids],
                            self.main_step or dt, self.courant, spacing)
            nonpositive_step = np.any(local_dt <= 0)
            # Infinite speed can produce a zero native Courant step. Let the
            # GLE reject it after consuming draws in the original order.
            if nonpositive_step and (not native_zero_step or np.any(local_dt < 0)):
                raise FloatingPointError('Nonpositive particle step')
            if owners is not None:
                draws = rng.streams[owners[ids[0]]].normal
            elif batch.slots is not None or self.native_gle:
                draws = rng.normal
            else:
                draws = rng.normal(size=(len(ids),3))
            if self.native_gle:
                old = previous if batch.previous_fluctuations is None else batch.previous_fluctuations[ids]
                fluct, stress = native_implicit_gle(old, batch.previous_stress[ids], tau, flux, ce, local_dt,
                                                    draws, self.policy.stress_invariant_tolerance)
            else:
                fluct,stress=implicit_gle(previous,batch.previous_stress[ids],tau,flux,ce,local_dt,
                                         draws,self.policy.stress_invariant_tolerance,self.scheme,
                                         native_determinant_gate=self.rogue_velocity is not None,
                                         compiled=self.compiled_gle)
            if self.native_math:
                fluct = fluct.astype(np.float32)
                stress = stress.astype(np.float32)
            if self.rogue_velocity is None:
                if not np.isfinite(fluct).all() or np.any(abs(fluct)>self.max_velocity):
                    raise FloatingPointError("Particle velocity threshold exceeded")
            else:
                good = np.all(np.isfinite(fluct) & (abs(fluct) < self.rogue_velocity), axis=1)
                rogue[ids[~good]] = True
                alive[ids[~good]] = False
                if batch.previous_fluctuations is not None:
                    batch.fluctuations[ids[~good]] = 0.
                    batch.previous_stress[ids[~good]] = stress[~good]
                ids, p, mean, local_dt, fluct, stress, ce = (
                    values[good] for values in (ids, p, mean, local_dt, fluct, stress, ce))
                if viscosity is not None:
                    viscosity = viscosity[good]
                if not len(ids):
                    continue
            if nonpositive_step and np.any(local_dt == 0):
                raise FloatingPointError('Nonpositive particle step')
            if invalid_wind and not (wall_rejects_invalid or boundary_rejects_invalid) and not np.isfinite(mean).all():
                raise FloatingPointError('Non-finite wind sampled at a particle position')
            velocity = mean+fluct
            displacement = velocity*local_dt[:, None]
            if self.deposition is not None:
                sink = {} if record is None else {'record': record}
                mass, removed = self.deposition.apply(batch.mass[ids], batch.material[ids], p, displacement,
                                                      velocity, stress, ce, viscosity, **sink)
                batch.mass[ids] = mass
                cutoff[ids[removed]] = True
                alive[ids[removed]] = False
                if removed.any():
                    ids, p, displacement, local_dt, fluct, stress = (
                        values[~removed] for values in (ids, p, displacement, local_dt, fluct, stress))
                    if not len(ids):
                        continue
            if reflect_native is not None:
                new_position, new_fluct, wall_active = reflect_native(p, displacement, fluct)
            else:
                if self.native_math:
                    # Round the endpoint before giving it to the geometric wall policy.
                    displacement = (p.astype(np.float32)+displacement).astype(float)-p
                new_position, new_fluct = self.walls.reflect(p, displacement, fluct)
                if self.native_math:
                    new_position = new_position.astype(np.float32)
                wall_active = self.walls.active_mask(new_position) if hasattr(self.walls, 'active_mask') else True
            if isinstance(self.boundaries, NativeDomainBoundaries):
                new_position, new_fluct, keep = self.boundaries.apply(new_position, new_fluct, active=wall_active)
            else:
                new_position, new_fluct, keep = self.boundaries.apply(new_position, new_fluct)
                keep &= wall_active
            if batch.tracking is not None:
                previous = (batch.fluctuations[ids] if batch.previous_fluctuations is None
                            else batch.previous_fluctuations[ids])
                if self.native_math:
                    previous = previous.astype(np.float32)
                batch.tracking['delta_fluctuation'][ids] = new_fluct-previous
            if batch.previous_fluctuations is not None:
                batch.previous_fluctuations[ids] = new_fluct
            batch.positions[ids]=new_position;batch.fluctuations[ids]=new_fluct;batch.previous_stress[ids]=stress
            alive[ids[~keep]]=False;remaining[ids]-=local_dt
        return alive, rogue, cutoff

class ParticleSimulation:
    """Coordinate timed release, optional momentum coupling and observations.

    Sources provide particle positions in m and mass in kg. The integrator
    advances them through the wind workspace. Each observer sees the same
    surviving state after a step and receives the step duration in seconds.
    """
    def __init__(self,sources,integrator,observers=(),momentum=None,cell_volume=None, *,
                 initial_active=None, track_particles=False, allocate_ids=None, serial_slots=False,
                 native_release_indexing=False, initial_in_grid=None, momentum_cell_indices=None,
                 track_enthalpy=False, density_entrainment=None, track_carrier_air=False,
                 track_mixture_inertia=False):
        self.sources=tuple(sources);self.integrator=integrator;self.observers=tuple(observers);self.momentum=momentum
        self.cell_volume=cell_volume;self.batch=ParticleBatch.empty();self.released_mass_kg=0.;self.released_particles=0;self.time_s=0.
        self.momentum_cell_indices = momentum_cell_indices
        self.initial_active = initial_active
        self.initial_in_grid = initial_in_grid
        self.replay = integrator.rng if isinstance(integrator.rng, NativeCPUStreams) else None
        if native_release_indexing and (self.replay is None or initial_active is None):
            raise ValueError('Original release indexing requires CPU replay with a cell mask')
        self.release_history = NativeReleaseSlots() if native_release_indexing else None
        self.allocate_ids = (ParticleIDs(native=self.replay is not None).take
                             if allocate_ids is None else allocate_ids)
        self.slot_capacity = 0
        if serial_slots or self.replay is not None:
            self.batch.slots = np.empty(0, dtype=np.int64)
        if self.release_history is not None:
            self.batch.previous_fluctuations = np.empty((0, 3), dtype=np.float32)
        if track_particles:
            self.batch.tracking = np.empty(0, dtype=TRACKING_DTYPE)
        if integrator.deposition is not None:
            self.batch.material = np.empty((0, 2))
        if track_enthalpy:
            self.batch.specific_enthalpy = np.empty(0)
        if track_mixture_inertia:
            if momentum is None or not track_enthalpy:
                raise ValueError('Mixture inertia requires momentum and enthalpy transport')
            self.batch.mixture_mass_ratio = np.empty(0)
        if track_carrier_air:
            if not track_enthalpy or momentum is None or self.replay is not None:
                raise ValueError('Carrier air requires cryogenic momentum and enthalpy transport')
            self.batch.carrier_air_mass = np.empty(0)
            self.released_carrier_air_kg = 0.
            self.escaped_carrier_air_kg = 0.
            self.released_drift_impulse_ns = np.zeros(3)
        if density_entrainment is not None and (not track_enthalpy or momentum is None or self.replay is not None):
            raise ValueError('Density entrainment requires cryogenic enthalpy transport')
        self.density_entrainment = density_entrainment
        self.inactive_on_release_mass_kg = 0.
        self.inactive_on_release_particles = 0

    def _count_inactive_release(self, mass, active):
        self.inactive_on_release_mass_kg += float(mass[~active].sum())
        self.inactive_on_release_particles += int((~active).sum())

    def _initial_state(self, positions, ordinals, new_count):
        field = self.integrator.fields
        if hasattr(field, 'sample_initial_turbulence'):
            tau, sigma = field.sample_initial_turbulence(positions)
        else:
            tau, _, _, _ = field.sample_turbulence(positions)
            sigma = np.sqrt(np.maximum(abs(np.diagonal(tau, axis1=1, axis2=2)), 1e-16))
        draws = (self.integrator.rng.normal(size=positions.shape) if self.replay is None else
                 self.replay.initial_draws(ordinals, new_count))
        initial = draws*sigma
        repair = native_realizable if self.integrator.native_gle else realizable
        tau = repair(tau, self.integrator.policy.stress_invariant_tolerance)
        if self.integrator.native_math:
            initial = initial.astype(np.float32)
            tau = tau.astype(np.float32)
        return initial, tau

    def step(self,dt):
        if dt<=0:raise ValueError("dt must be positive")
        field=self.integrator.fields;rng=self.integrator.rng
        occupied = self.batch.slots
        emissions = ((source, *source.emit(self.time_s, dt, rng)) for source in self.sources)
        history = self.release_history
        new_count = 0
        ordinals = None
        if self.replay is not None:
            emissions = tuple(emissions)
            new_count = sum(len(mass) for _, _, mass in emissions)
            reserved, self.slot_capacity = reserve_particle_slots(occupied, new_count, self.slot_capacity)
            first_ordinal = 0
            if history is not None and new_count:
                valid_ordinals, metadata = history.prepare(self.slot_capacity, reserved, emissions, self.initial_active,
                                                           self.batch.tracking is not None)
                invalid_slots = np.flatnonzero(~valid_ordinals)
                if len(invalid_slots):
                    keep = ~np.isin(occupied, invalid_slots)
                    self._count_inactive_release(self.batch.mass, keep)
                    self.batch.retain(keep)
        for source_index, (source, p, m) in enumerate(emissions):
            if not len(m):continue
            if self.integrator.native_math:
                p = p.astype(np.float32)
            slots = None
            previous_fluctuation = None
            if self.replay is not None:
                ordinals = np.arange(first_ordinal, first_ordinal+len(m), dtype=np.int64)
                slots = reserved[first_ordinal:first_ordinal+len(m)]
                first_ordinal += len(m)
            elif occupied is not None:
                slots = first_free_slots(occupied, len(m))
                occupied = np.concatenate((occupied, slots))
            ids = self.allocate_ids(len(m))
            self.released_particles+=len(m)
            self.released_mass_kg+=float(m.sum())
            if history is not None:
                initialize = valid_ordinals[ordinals]
                active = ~np.isin(slots, invalid_slots)
                if self.initial_in_grid is not None:
                    # The original ordinal check can miss an inserted
                    # position outside the allocation. Do not sample it.
                    in_grid = self.initial_in_grid(p)
                    initialize &= in_grid
                    active &= in_grid
                initialized_slots = slots[initialize]
                if len(initialized_slots):
                    initial, tau = self._initial_state(p[initialize], ordinals[initialize], new_count)
                    history.fluctuations[initialized_slots] = initial
                    history.previous_fluctuations[initialized_slots] = initial
                    history.stress[initialized_slots] = tau
                    history.delta[initialized_slots] = 0.
                self._count_inactive_release(m, active)
                p, m, ids, slots = p[active], m[active], ids[active], slots[active]
                if not len(m):
                    continue
                initial, tau = history.fluctuations[slots], history.stress[slots]
                previous_fluctuation = history.previous_fluctuations[slots]
            elif self.initial_active is not None:
                active = self.initial_active(p)
                self._count_inactive_release(m, active)
                p,m,ids = p[active],m[active],ids[active]
                if slots is not None:
                    slots = slots[active]
                if self.replay is not None:
                    ordinals = ordinals[active]
                if not len(m):continue
            if history is None:
                initial, tau = self._initial_state(p, ordinals, new_count)
            drift=np.broadcast_to(source.initial_drift,p.shape).copy()
            if self.batch.carrier_air_mass is not None:
                air = float(m.sum()) * source.carrier_air_per_h2
                self.released_carrier_air_kg += air
                self.released_drift_impulse_ns += (float(m.sum()) + air) * np.asarray(source.initial_drift)
            self.batch.append(p,m,initial,tau,drift,ids,source=source,release_s=self.time_s,slots=slots,
                              previous_fluctuation=previous_fluctuation)
            if history is not None and self.batch.tracking is not None:
                tracking = self.batch.tracking[-len(m):]
                tracking['initial_position'] = metadata[source_index][active]
                tracking['delta_fluctuation'] = history.delta[slots]
        if self.momentum is not None and len(self.batch.mass):
            locate = field.cell_indices if self.momentum_cell_indices is None else self.momentum_cell_indices
            shape = (len(field.z), len(field.y), len(field.x))
            flat = np.ravel_multi_index(locate(self.batch.positions), shape)
            volume = self.cell_volume[np.unique(flat)] if np.ndim(self.cell_volume) else self.cell_volume
            thermal = {} if self.batch.specific_enthalpy is None else {'enthalpy': self.batch.specific_enthalpy}
            if self.batch.carrier_air_mass is not None:
                thermal['carrier_air_mass'] = self.batch.carrier_air_mass
            if self.batch.mixture_mass_ratio is not None:
                thermal['mixture_mass_ratio'] = self.batch.mixture_mass_ratio
            self.batch.drift,_=self.momentum.update(self.batch.positions,self.batch.mass,self.batch.drift,flat,volume,dt,**thermal)
        air_before = None if self.batch.carrier_air_mass is None else float(self.batch.carrier_air_mass.sum())
        if self.replay is None:
            self.integrator.advance(self.batch, dt)
        else:
            self.integrator.advance(self.batch, dt, slot_capacity=self.slot_capacity,
                                    capture=None if history is None else history.capture)
        if air_before is not None:
            self.escaped_carrier_air_kg += air_before - float(self.batch.carrier_air_mass.sum())
        if self.density_entrainment is not None:
            self.density_entrainment.mix(self.batch, self.momentum.thermo, dt)
        thermal = {} if self.batch.specific_enthalpy is None else {'enthalpy': self.batch.specific_enthalpy}
        for observer in self.observers:
            observer.observe(self.time_s,dt,self.batch.positions,self.batch.mass,**thermal)
        self.time_s+=dt
    def run(self,duration_s,dt,progress=None):
        """Advance to the requested model time and return mass accounting."""
        if duration_s<self.time_s:raise ValueError("Cannot run backwards")
        next_progress=self.time_s+60
        origin=self.time_s;step_index=0
        while self.time_s<duration_s-1e-9:
            step_index+=1;target=min(duration_s,origin+step_index*dt)
            self.step(target-self.time_s)
            self.time_s=target
            if progress and self.time_s>=next_progress:
                progress(self.time_s,len(self.batch.mass));next_progress+=60
        return self.accounting()
    def accounting(self):
        active=float(self.batch.mass.sum());escaped=self.integrator.escaped_mass_kg
        rogue_mass = self.integrator.rogue_mass_kg
        result = {"released_mass_kg":self.released_mass_kg,"active_mass_kg":active,"escaped_mass_kg":escaped,
                "mass_balance_error_kg":self.released_mass_kg-active-escaped-rogue_mass-self.inactive_on_release_mass_kg,
                "released_particles":self.released_particles,
                "active_particles":len(self.batch.mass),"escaped_particles":self.integrator.escaped_particles,
                "pending_source_mass_kg":float(sum(s._pending_mass for s in self.sources))}
        if self.initial_active is not None:
            result.update(inactive_on_release_particles=self.inactive_on_release_particles,
                          inactive_on_release_mass_kg=self.inactive_on_release_mass_kg)
        if self.batch.carrier_air_mass is not None:
            result.update(released_carrier_air_kg=self.released_carrier_air_kg,
                          active_carrier_air_kg=float(self.batch.carrier_air_mass.sum()),
                          removed_carrier_air_kg=self.escaped_carrier_air_kg,
                          released_drift_impulse_ns=self.released_drift_impulse_ns.tolist())
        if self.integrator.rogue_velocity is not None:
            result.update(rogue_particles=self.integrator.rogue_particles, rogue_mass_kg=rogue_mass)
        deposition = self.integrator.deposition
        if deposition is not None:
            deposited = float(deposition.deposited_kg.sum())
            result.update(deposited_mass_kg=deposited, deposition_cutoff_mass_kg=deposition.cutoff_discard_kg,
                          deposition_cutoff_particles=deposition.cutoff_particles)
            result['mass_balance_error_kg'] -= deposited+deposition.cutoff_discard_kg
        return result
