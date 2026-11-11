"""Assemble optional cryogenic source, thermal transport and pressure models."""

from dataclasses import asdict, dataclass
from math import isclose

import numpy as np

from .lh2 import LH2Momentum
from .lh2_dense import ShallowLayerPressure
from .lh2_entrainment import DensityEntrainment
from .lh2_ground_heat import CloudGroundHeat
from .lh2_jet import Disk, JetHandoff, flash_state, integrate_crossflow, integrate_jet
from .lh2_pool import PoolRelease
from .lh2_plume_forces import PlumeSectionForces
from .sources import ContinuousRelease, Source
from .thermodynamics.properties import CoolPropProperties


def _fluid_disk(centre, normal, radius, sampler, solid):
    extent = radius * np.sqrt(np.maximum(0., 1 - np.square(normal)))
    lower, upper = centre - extent, centre + extent
    faces = (sampler.xf, sampler.yf, sampler.zf)
    if any(lo <= face[1] or hi >= face[-2] for lo, hi, face in zip(lower, upper, faces, strict=True)):
        raise ValueError('LH₂ handoff disk intersects a domain boundary')
    indices = sampler.cell_indices(np.array([lower, upper]))
    if np.any(solid[tuple(slice(lo, hi + 1) for lo, hi in indices)]):
        raise ValueError('LH₂ handoff disk bounding box intersects a solid cell')
    return Disk(centre, normal, radius)


@dataclass
class CryogenicRelease:
    sources: tuple[Source, ...]
    momentum: LH2Momentum
    jet: JetHandoff
    pool: PoolRelease | None
    rate: float
    end: float
    entrainment: DensityEntrainment | None
    track_carrier_air: bool = False
    crossflow_input: tuple | None = None
    crossflow_updates: int = 0
    crossflow_momentum_n: tuple | None = None

    @property
    def thermo(self):
        return self.momentum.thermo

    def refresh_crossflow(self, sampler, solid):
        if self.momentum.plume_forces is not None:
            self.momentum.plume_forces.set_wind(sampler)
        if self.crossflow_input is None:
            return
        config, position, enthalpy = self.crossflow_input
        result = integrate_crossflow(config, self.thermo, enthalpy, position, self.rate,
                                     float(sampler.zf[1]), sampler.sample_wind)
        if np.any(solid[sampler.cell_indices(result.path_m)]):
            raise ValueError('LH₂ crossflow jet enters a solid cell before handoff')
        jet = result.jet
        geometry = _fluid_disk(jet.centre_m, jet.direction, jet.radius_m, sampler, solid)
        source, = self.sources
        source.geometry = geometry
        source.initial_drift = result.drift_m_s
        source.carrier_air_per_h2 = jet.air_rate_kg_s / self.rate
        self.jet = jet
        self.crossflow_updates += 1
        self.crossflow_momentum_n = result.momentum_flux_n

    def diagnostics(self, elapsed, accounting):
        expected = self.rate * min(elapsed, self.end)
        liquid = 0. if self.pool is None else self.pool.inventory_kg
        total = accounting['released_mass_kg'] + accounting['pending_source_mass_kg'] + liquid
        if not isclose(expected, total, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError('Invalid total LH₂ source mass balance')
        if self.track_carrier_air and not isclose(
                accounting['released_carrier_air_kg'],
                accounting['active_carrier_air_kg'] + accounting['removed_carrier_air_kg'],
                rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError('Invalid jet carrier air balance')
        return dict(jet=asdict(self.jet), supplied_h2_kg=expected, source_balance_error_kg=expected - total,
                    crossflow_updates=self.crossflow_updates, last_crossflow_momentum_n=self.crossflow_momentum_n,
                    dense_layer=self.momentum.shallow_layer is not None,
                    pressure_active_steps=self.momentum.pressure_active_steps,
                    peak_pressure_increment_m_s=self.momentum.peak_pressure_increment_m_s,
                    ground_heat_j=self.momentum.ground_heat_j,
                    plume_forces=None if self.momentum.plume_forces is None else self.momentum.plume_forces.diagnostics(),
                    relative_humidity_pct=self.thermo.relative_humidity_pct,
                    entrainment=None if self.entrainment is None else dict(
                        exchanged_volume_m3=self.entrainment.exchanged_volume_m3,
                        transferred_h2_kg=self.entrainment.transferred_h2_kg),
                    pool=None if self.pool is None else dict(
                        fed_kg=self.pool.fed_kg, evaporated_kg=self.pool.evaporated_kg, remaining_kg=liquid,
                        peak_inventory_kg=self.pool.peak_inventory_kg,
                        evaporation_energy_j=self.pool.evaporated_kg * self.pool.latent_heat_j_kg))


def prepare_cryogenic(config, ambient_temperature, position, rate, particles_per_s, sampler, solid,
                      *, relative_humidity_pct=0.):
    try:
        properties = CoolPropProperties(config['jet'].get('hydrogen_fluid', 'Hydrogen'))
    except ImportError as error:
        raise ValueError('Install the thermodynamics extra with: python -m pip install "plumeswift[thermo]"') from error
    end = float(config.get('release_end_s', np.inf))
    if not end > 0 or not np.isfinite(particles_per_s) or particles_per_s <= 0:
        raise ValueError('Require positive release_end_s and particles_per_s')
    ground = float(sampler.zf[1])
    if len(position) != 3 or not np.isfinite(position).all() or position[2] <= ground:
        raise ValueError('The cryogenic nozzle must be above ground')
    ground_heat = None
    if 'ground_heat' in config:
        ground_heat = CloudGroundHeat(solid.shape, 1, sampler.dx * sampler.dy, **config['ground_heat'])
    mixture, enthalpy = flash_state(config['jet'], ambient_temperature, properties,
                                    relative_humidity_pct=config.get('relative_humidity_pct', relative_humidity_pct),
                                    maximum_temperature=None if ground_heat is None else ground_heat.ground_temperature_k)
    handoff = integrate_jet(config['jet'], mixture, enthalpy, position, rate, ground)
    crossflow = bool(config.get('crossflow_jet', False))
    if crossflow and ('pool' in config or abs(handoff.direction[2]) > 1e-12):
        raise ValueError('Crossflow jet requires a horizontal nozzle without a pool')
    track_carrier_air = crossflow or bool(config.get('conservative_handoff', False))
    spacing = min(sampler.dx, sampler.dy, np.diff(sampler.zf).min())
    sections = max(2, int(np.ceil(2 * handoff.distance_m / spacing)) + 1)
    centreline = np.linspace(position, handoff.centre_m, sections, endpoint=not handoff.ground_impact)
    if np.any(solid[sampler.cell_indices(centreline)]):
        raise ValueError('LH₂ jet centreline enters a solid cell before handoff')
    centre, direction = handoff.centre_m, handoff.direction
    normal = direction
    velocity = handoff.speed_m_s * np.asarray(direction)
    if handoff.ground_impact:
        centre = (*centre[:2], float((sampler.zf[1] + sampler.zf[2]) / 2))
        normal = (0., 0., 1.)
        velocity[2] = 0.

    pool = None
    sources = []
    gas_rate = rate
    if 'pool' in config:
        if not handoff.ground_impact:
            raise ValueError('Pool requires the downward jet to reach ground before its handoff distance')
        gas_rate = rate * handoff.vapor_mass_fraction
        if gas_rate < rate:
            pool_cfg = config['pool']
            pool = PoolRelease(rate - gas_rate, end, pool_cfg['radius_m'], pool_cfg['ground_temperature_k'],
                               mixture.boiling_temperature, mixture.latent_heat, pool_cfg['ground_effusivity'],
                               particles_per_s, pool_cfg.get('background_flux_w_m2', 0.))
            sources.append(Source(_fluid_disk(centre, normal, pool.radius_m, sampler, solid), pool, source_id=1))
        # Saturated liquid carries -L relative to vapor. Keep the remaining
        # enthalpy with the gas stream, including the deficit of its cold air.
        if pool is not None and gas_rate > 0:
            enthalpy = (rate * enthalpy + pool.liquid_rate_kg_s * mixture.latent_heat) / gas_rate
    if gas_rate > 0:
        disk = _fluid_disk(centre, normal, handoff.radius_m, sampler, solid)
        air_ratio = handoff.air_rate_kg_s / gas_rate if track_carrier_air else 0.
        sources.insert(0, Source(disk, ContinuousRelease(gas_rate, particles_per_s, end_s=end),
                                 initial_drift=tuple(velocity), specific_enthalpy_j_kg=enthalpy,
                                 carrier_air_per_h2=air_ratio))
    layer = None
    if 'dense_layer' in config:
        layer = ShallowLayerPressure(sampler, solid, ground, **config['dense_layer'])
    plume_forces = None
    if config.get('plume_section_forces', False):
        plume_forces = PlumeSectionForces(tuple(position), max(sampler.dx, sampler.dy),
                                          float(sampler.zf[2] - sampler.zf[1]))
    momentum = LH2Momentum(mixture, shallow_layer=layer, ground_heat=ground_heat, plume_forces=plume_forces)
    entrainment = None
    if 'density_entrainment' in config:
        entrainment = DensityEntrainment(sampler, solid, **config['density_entrainment'])
    release = CryogenicRelease(tuple(sources), momentum, handoff, pool, rate, end, entrainment, track_carrier_air,
                               (config['jet'], position, enthalpy) if crossflow else None)
    release.refresh_crossflow(sampler, solid)
    return release
