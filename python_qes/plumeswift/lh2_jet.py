"""Pressure-equilibrated H₂ flash and integral, top-hat near-field jet."""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from .lh2_energy import CryogenicMixture


@dataclass(frozen=True)
class JetHandoff:
    centre_m: tuple[float, float, float]
    direction: tuple[float, float, float]
    radius_m: float
    distance_m: float
    temperature_k: float
    vapor_mass_fraction: float
    air_rate_kg_s: float
    speed_m_s: float
    axial_momentum_n: float
    enthalpy_j_kg: float
    ground_impact: bool


@dataclass(frozen=True)
class CrossflowHandoff:
    jet: JetHandoff
    drift_m_s: tuple[float, float, float]
    momentum_flux_n: tuple[float, float, float]
    path_m: np.ndarray


@dataclass(frozen=True)
class Disk:
    centre: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float

    def sample(self, count, rng):
        normal = np.asarray(self.normal)
        basis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = np.cross(normal, basis)
        tangent /= np.linalg.norm(tangent)
        second = np.cross(normal, tangent)
        radius = self.radius * np.sqrt(rng.random(count))
        angle = rng.uniform(0., 2 * pi, count)
        return self.centre + radius[:, None] * (
            np.cos(angle)[:, None] * tangent + np.sin(angle)[:, None] * second)


def flash_state(config, ambient_temperature, properties, *, relative_humidity_pct=0., maximum_temperature=None):
    """Use pure-fluid properties once at startup, outside the particle loop."""
    p = float(config.get('ambient_pressure_pa', 101325.))
    fluid = properties.hydrogen_fluid
    liquid = properties.saturation(fluid, p, 0.)
    vapor = properties.saturation(fluid, p, 1.)
    latent = vapor['enthalpy'] - liquid['enthalpy']
    tb = vapor['temperature']
    if not np.isfinite(ambient_temperature) or ambient_temperature <= tb:
        raise ValueError('Ambient air must be warmer than saturated hydrogen')
    rho_vapor = properties.saturation_density(fluid, p, 1.)
    upper = ambient_temperature if maximum_temperature is None else max(ambient_temperature, maximum_temperature)
    temperatures = np.linspace(tb, upper, 512)
    gas_volumes = np.r_[1 / rho_vapor, 1 / properties.density(fluid, temperatures[1:], p, phase='gas')]
    mixture = CryogenicMixture(ambient_temperature, p, tb, latent,
                               properties.saturation_density(fluid, p, 0.), rho_vapor,
                               (temperatures, gas_volumes), relative_humidity_pct)
    quality = config.get('exit_vapor_mass_fraction')
    if quality is None:
        pressure, temperature = (float(config[key]) for key in ('upstream_pressure_pa', 'upstream_temperature_k'))
        if pressure < p or temperature <= 0 or not np.isfinite([pressure, temperature]).all():
            raise ValueError('Invalid upstream LH₂ pressure or temperature')
        enthalpy = float(properties.enthalpy(fluid, temperature, pressure)) - vapor['enthalpy']
    else:
        if not 0 <= quality <= 1 or {'upstream_pressure_pa', 'upstream_temperature_k'} & config.keys():
            raise ValueError('Specify either a nozzle exit vapor mass fraction or the upstream state')
        enthalpy = -(1. - quality) * latent
    if not -latent <= enthalpy <= 0. or ambient_temperature <= mixture.boiling_temperature:
        raise ValueError('The jet closure requires an ambient-pressure two-phase H₂ exit and warmer air')
    return mixture, enthalpy


def integrate_jet(config, mixture, enthalpy, origin, mass_rate, ground_z):
    """Integrate entrained air mass with constant axial momentum in still air.

    The handoff is at the requested distance or the first centreline/ground
    intersection. Wind and obstacles are resolved downstream by PlumeSwift.
    """
    direction = np.array(config['direction'], dtype=float)
    length = np.linalg.norm(direction)
    diameter = float(config['nozzle_diameter_m'])
    distance = float(config['handoff_distance_m'])
    alpha = float(config.get('entrainment_coefficient', .07))
    if (direction.shape != (3,) or not np.isfinite([length, diameter, distance, alpha, mass_rate]).all()
            or min(length, diameter, mass_rate) <= 0 or distance < 0 or alpha < 0):
        raise ValueError('Invalid cryogenic jet geometry, mass rate or entrainment coefficient')
    direction /= length
    impact_distance = (origin[2] - ground_z) / -direction[2] if direction[2] < 0 else np.inf
    ground_impact = impact_distance <= distance
    distance = min(distance, impact_distance)
    nozzle_area = pi * (diameter / 2)**2
    _, _, exit_volume_rate = mixture.jet_state(mass_rate, 0., enthalpy)
    momentum = mass_rate * exit_volume_rate / nozzle_area

    def section(air_rate):
        temperature, vapor, flow = mixture.jet_state(mass_rate, air_rate, enthalpy)
        speed = momentum / (mass_rate + air_rate)
        radius = sqrt(flow / (pi * speed))
        return temperature, vapor, speed, radius

    def entrain(air_rate):
        _, _, speed, radius = section(air_rate)
        return 2 * pi * radius * mixture.ambient_density * alpha * speed

    air_rate = 0.
    if distance and alpha:
        steps = max(32, min(4096, int(np.ceil(4 * distance / diameter))))
        ds = distance / steps
        for _ in range(steps):
            air_rate += ds * entrain(air_rate + .5 * ds * entrain(air_rate))
    temperature, vapor, speed, radius = section(air_rate)
    centre = np.asarray(origin) + distance * direction
    return JetHandoff(tuple(centre), tuple(direction), radius, distance, temperature, vapor,
                      air_rate, speed, momentum, enthalpy, bool(ground_impact))


def integrate_crossflow(config, mixture, enthalpy, origin, mass_rate, ground_z, wind_at):
    """March a horizontal round jet with shear entrainment and vector momentum.

    Entrained ambient air brings its local momentum. The mixture's density
    contrast produces a vertical force on the evolving cross-section. The
    resulting bulk velocity is converted to excess drift at the handoff.
    """
    still = integrate_jet(config, mixture, enthalpy, origin, mass_rate, ground_z)
    axis = np.array(config['direction'], dtype=float)
    axis /= np.linalg.norm(axis)
    if abs(axis[2]) > 1e-12:
        raise ValueError('Crossflow jet requires a horizontal nozzle')
    alpha = float(config.get('entrainment_coefficient', .07))
    diameter = float(config['nozzle_diameter_m'])
    state = np.r_[origin, 0., still.axial_momentum_n * axis]

    def section(values):
        air = values[3]
        temperature, vapor, flow = mixture.jet_state(mass_rate, air, enthalpy)
        bulk = values[4:] / (mass_rate + air)
        speed = float(np.linalg.norm(bulk))
        if speed <= 1e-6 or not np.isfinite(speed):
            raise ValueError('Near-field jet stalled before the crossflow handoff')
        area = flow / speed
        return temperature, vapor, bulk, speed, area

    def derivative(values):
        _, _, bulk, speed, area = section(values)
        ambient = wind_at(values[:3][None, :])[0]
        shear = np.linalg.norm(bulk - ambient)
        air_in = 2 * sqrt(pi * area) * mixture.ambient_density * alpha * shear
        density = (mass_rate + values[3]) / (area * speed)
        momentum = air_in * ambient
        momentum[2] += 9.80665 * (mixture.ambient_density - density) * area
        return np.r_[bulk / speed, air_in, momentum]

    path = [state[:3].copy()]
    distance = still.distance_m
    if distance:
        steps = max(32, min(4096, int(np.ceil(4 * distance / diameter))))
        ds = distance / steps
        for _ in range(steps):
            midpoint = state + .5 * ds * derivative(state)
            state += ds * derivative(midpoint)
            if state[2] <= ground_z:
                raise ValueError('Crossflow jet reaches ground before handoff; shorten its handoff distance')
            path.append(state[:3].copy())
    temperature, vapor, bulk, speed, area = section(state)
    jet = JetHandoff(tuple(state[:3]), tuple(bulk / speed), sqrt(area / pi), distance,
                     temperature, vapor, state[3], speed, float(state[4:] @ axis), enthalpy, False)
    drift = bulk - wind_at(state[:3][None, :])[0]
    return CrossflowHandoff(jet, tuple(drift), tuple(state[4:]), np.asarray(path))
