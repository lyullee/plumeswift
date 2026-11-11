from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from plumeswift.lh2 import LH2Momentum
from plumeswift.lh2_dense import DenseLayerExchange, ShallowLayerPressure
from plumeswift.lh2_energy import CryogenicMixture
from plumeswift.lh2_entrainment import DensityEntrainment
from plumeswift.lh2_ground_heat import CloudGroundHeat
from plumeswift.lh2_case import prepare_cryogenic
from plumeswift.lh2_jet import flash_state, integrate_jet
from plumeswift.lh2_pool import PoolRelease
from plumeswift.thermodynamics.moisture import equilibrium_water_pressure


@pytest.fixture
def mixture():
    return CryogenicMixture(285., 101325., 20.37, 445000., 70.8, 1.34)


@pytest.mark.parametrize('air_rate', [0., .2, 1., 10.])
def test_energy_closure_recovers_integral_jet_state(mixture, air_rate):
    mass_rate, enthalpy = .4, -.85 * mixture.latent_heat
    temperature, vapor, volume_rate = mixture.jet_state(mass_rate, air_rate, enthalpy)
    concentration = mass_rate / volume_rate
    state = mixture.evaluate(concentration, concentration * enthalpy)
    assert state.temperature == pytest.approx(temperature)
    assert state.vapor_quality == pytest.approx(vapor)
    assert state.density == pytest.approx((mass_rate + air_rate) / volume_rate)


def test_warm_hydrogen_volume_uses_temperature_dependent_gas_properties():
    pytest.importorskip('CoolProp')
    from plumeswift.thermodynamics.properties import CoolPropProperties

    props = CoolPropProperties()
    thermo, _ = flash_state(dict(exit_vapor_mass_fraction=.105), 285., props)
    assert 1 / thermo.vapor_specific_volume(250.) == pytest.approx(
        float(props.density('Hydrogen', 250., 101325.)), rel=3e-6)
    assert 1 / thermo.vapor_specific_volume(thermo.boiling_temperature) == pytest.approx(thermo.vapor_density)


@pytest.mark.parametrize('humidity', [0., 75.])
def test_compiled_equilibrium_matches_numpy_through_phase_boundaries(humidity):
    pytest.importorskip('numba')
    pytest.importorskip('CoolProp')
    from plumeswift.thermodynamics.properties import CoolPropProperties

    model, source_h = flash_state(dict(exit_vapor_mass_fraction=.105), 285., CoolPropProperties(),
                                 relative_humidity_pct=humidity, maximum_temperature=300.)
    rng = np.random.default_rng(12)
    c = np.r_[0., np.geomspace(1e-8, 69., 300), rng.uniform(0., .05, 500)]
    energy = c * rng.uniform(source_h, 3e6, len(c))
    near_freezing = model.vapor_enthalpy_density(c[-10:], np.linspace(273.15, 273.17, 10))
    energy[-10:] = near_freezing
    reference = replace(model, use_compiled=False).evaluate(c, energy)
    optimized = model.evaluate(c, energy)
    for name in ('temperature', 'density', 'volume_fraction', 'vapor_quality'):
        np.testing.assert_array_equal(getattr(reference, name), getattr(optimized, name))


@pytest.mark.parametrize('air_rate', [.2, 1., 10., 100.])
def test_moist_equilibrium_matches_jet_state(mixture, air_rate):
    moist = replace(mixture, relative_humidity_pct=75.)
    rate, enthalpy = .4, -.85 * mixture.latent_heat
    temperature, vapor, volume = moist.jet_state(rate, air_rate, enthalpy)
    state = moist.evaluate(rate / volume, rate * enthalpy / volume)
    assert state.temperature == pytest.approx(temperature, abs=2e-5)
    assert state.density == pytest.approx((rate + air_rate) / volume, rel=2e-6)
    assert state.vapor_quality == pytest.approx(vapor)
    assert equilibrium_water_pressure(250.) == pytest.approx(76.0, rel=.01)


def test_ground_heat_is_local_and_limited_by_equilibrium(mixture):
    heating = CloudGroundHeat((4, 3, 3), 1, 1., 275., 30.)
    mass = np.array([.01, .01])
    energy = mass * -1e5
    volume = np.ones(2)
    state = mixture.evaluate(mass, energy)
    heat = heating.energy(mixture, np.array([9, 18]), volume, mass, energy, state, 1e6)
    assert heat[0] > 0 and heat[1] == 0
    assert mixture.evaluate(mass, energy + heat).temperature[0] == pytest.approx(275.)


def test_carrier_handoff_uses_total_mass_for_cell_momentum(mixture):
    from plumeswift.lh2 import LH2Momentum

    positions = np.ones((2, 3))
    mass = np.array([1., 1.])
    air = np.array([2., 0.])
    drift = np.array([[8., 0., 0.], [0., 0., 0.]])
    # With zero time increment, only within-cell momentum averaging acts.
    result, _ = LH2Momentum(mixture).update(
        positions, mass, drift, np.array([2, 2]), 10., 0.,
        enthalpy=np.zeros(2), carrier_air_mass=air)
    np.testing.assert_allclose(result[:, 0], 6.)
    assert (result.T @ (mass + air))[0] == pytest.approx((drift.T @ (mass + air))[0])


def test_crossflow_handoff_receives_the_entrained_air_momentum(mixture):
    from plumeswift.lh2_jet import integrate_crossflow

    config = dict(direction=(1., 0., 0.), nozzle_diameter_m=.0254, handoff_distance_m=.5)
    ambient = np.array([0., 4., 0.])
    result = integrate_crossflow(config, mixture, -.895 * mixture.latent_heat,
                                 (0., 0., .5), .832, 0., lambda positions: ambient[None, :])
    assert result.jet.centre_m[1] > 0 and result.jet.air_rate_kg_s > 0
    assert result.momentum_flux_n[1] == pytest.approx(4. * result.jet.air_rate_kg_s)
    np.testing.assert_allclose((np.asarray(result.drift_m_s) + ambient) * (.832 + result.jet.air_rate_kg_s),
                               result.momentum_flux_n)
    still = integrate_jet(config, mixture, -.895 * mixture.latent_heat, (0., 0., .5), .832, 0.)
    calm = integrate_crossflow(config, mixture, -.895 * mixture.latent_heat,
                               (0., 0., .5), .832, 0., lambda positions: np.zeros_like(positions))
    assert calm.jet.air_rate_kg_s == pytest.approx(still.air_rate_kg_s, rel=.001)


def test_plume_section_forces_match_ellipse_and_preserve_near_field():
    from plumeswift.lh2_plume_forces import PlumeSectionForces

    model = PlumeSectionForces((0., 0., 0.), 2., .5)
    positions = np.array([[0., 0., 1.], [10., -2., 1.], [10., 2., 3.]])
    particles = np.ones(3)
    inverse = np.array([0, 1, 1])
    original = np.array([1.25, -.5])
    vertical = np.array([0., 1.5])
    result = model.vertical_velocity(positions, particles, inverse, np.array([1., 2.]),
                                     np.array([.8, .8]), 1.2, vertical, original, 9.81, .5, .2)
    # H₂-weighted standard deviations are 2 m and 1 m for the downstream slice.
    half_width, half_depth = 4., 2.
    inertia = .8 + 1.2 * half_width / half_depth
    forced = 1.5 + .2 * 9.81 * (1.2 - .8) / inertia
    drag = 1.2 * .5 / (np.pi * half_depth * inertia)
    assert result[0] == original[0]
    assert result[1] + .2 * drag * abs(result[1]) * result[1] == pytest.approx(forced)
    assert model.diagnostics()['mean_width_m'] == pytest.approx(8.)
    assert model.diagnostics()['mean_depth_m'] == pytest.approx(4.)


def test_density_exchange_conserves_markers_and_respects_solid_faces():
    faces = np.arange(6.)
    sampler = SimpleNamespace(xf=faces, yf=faces, zf=faces,
                              cell_indices=lambda points: tuple(np.floor(points.T[::-1]).astype(int)))
    solid = np.ones((5, 5, 5), dtype=bool)
    solid[2, 2, 2] = solid[2, 2, 3] = False
    model = DensityEntrainment(sampler, solid, coefficient=.1)
    batch = SimpleNamespace(positions=np.full((8, 3), 2.5), mass=np.full(8, .01),
                            specific_enthalpy=np.arange(8.) - 1e5)
    energy = float(batch.mass @ batch.specific_enthalpy)
    thermo = SimpleNamespace(ambient_density=1., evaluate=lambda c, e: SimpleNamespace(density=np.full(c.shape, 2.)))
    model.mix(batch, thermo, 100.)
    assert batch.positions[:, 0] == pytest.approx(np.full(8, 3.5))
    assert batch.positions[:, 1:] == pytest.approx(np.full((8, 2), 2.5))
    assert batch.mass.sum() == pytest.approx(.08)
    assert batch.mass @ batch.specific_enthalpy == pytest.approx(energy)
    assert model.transferred_h2_kg == pytest.approx(.08)


def test_ffi_nozzle_distinguishes_published_and_assumed_exit_quality():
    from plumeswift.ffi_lh2 import _cryogenic_nozzle

    position, jet, basis = _cryogenic_nozzle(dict(test=5, orientation='vertical_down', liquid_fraction=.559))
    assert position[2] == .32 and jet['exit_vapor_mass_fraction'] == .105 and 'ADREA-HF' in basis
    position, jet, _ = _cryogenic_nozzle(dict(test=6, orientation='horizontal', liquid_fraction=.559))
    assert position[2] == .5 and jet['exit_vapor_mass_fraction'] == .105
    _, jet, basis = _cryogenic_nozzle(dict(test=2, orientation='vertical_down', liquid_fraction=.7))
    assert jet['exit_vapor_mass_fraction'] == pytest.approx(.3) and 'assumption' in basis


def test_flash_and_jet_conserve_axial_momentum():
    pytest.importorskip('CoolProp')
    from plumeswift.thermodynamics.properties import CoolPropProperties

    config = dict(upstream_pressure_pa=326325., upstream_temperature_k=22., direction=(0., 0., -1.),
                  nozzle_diameter_m=.0254, handoff_distance_m=1.)
    mixture, enthalpy = flash_state(config, 285., CoolPropProperties())
    exit_state = integrate_jet(dict(config, handoff_distance_m=0.), mixture, enthalpy, (0., 0., .5), .7, 0.)
    impact = integrate_jet(config, mixture, enthalpy, (0., 0., .5), .7, 0.)
    assert 0 < exit_state.vapor_mass_fraction < impact.vapor_mass_fraction <= 1
    assert exit_state.radius_m == pytest.approx(.0254 / 2)
    assert impact.ground_impact and impact.centre_m == (0., 0., 0.)
    assert impact.radius_m > exit_state.radius_m
    assert (.7 + impact.air_rate_kg_s) * impact.speed_m_s == pytest.approx(exit_state.axial_momentum_n)


def test_pool_evaporation_during_and_after_cutoff():
    pool = PoolRelease(.7, 1.5, .5, 280., 20., 445000., 100., 10.)
    masses = [pool.count_and_mass(t, 1.)[1] for t in range(3)]
    assert all(mass > 0 for mass in masses)
    assert pool.fed_kg == pytest.approx(1.05)
    assert pool.inventory_kg > 0
    assert sum(masses) + pool.inventory_kg == pytest.approx(pool.fed_kg)
    assert pool.count_and_mass(3., 1e6)[1] == pytest.approx(pool.fed_kg - sum(masses))
    assert pool.count_and_mass(1e6 + 3., 1.) == (0, 0.)


def test_ground_impact_preserves_enthalpy_when_liquid_is_split():
    pytest.importorskip('CoolProp')
    faces = np.arange(-1., 5.)
    sampler = SimpleNamespace(dx=1., dy=1., xf=faces, yf=faces, zf=faces,
                              cell_indices=lambda points: tuple(np.searchsorted(faces, points.T[::-1],
                                                                               side='right') - 1))
    config = dict(jet=dict(exit_vapor_mass_fraction=.1, direction=(0., 0., -1.),
                           nozzle_diameter_m=.0254, handoff_distance_m=.5),
                  pool=dict(radius_m=.5, ground_temperature_k=285., ground_effusivity=1000.))
    rate = .4
    model = prepare_cryogenic(config, 285., (1.5, 1.5, .5), rate, 10, sampler, np.zeros((5, 5, 5), bool))
    vapor = model.sources[0]
    assert vapor.release.mass_rate_kg_s + model.pool.liquid_rate_kg_s == pytest.approx(rate)
    flux = (vapor.release.mass_rate_kg_s * vapor.specific_enthalpy_j_kg
            - model.pool.liquid_rate_kg_s * model.thermo.latent_heat)
    assert flux == pytest.approx(rate * model.jet.enthalpy_j_kg)


def test_shallow_pressure_spreads_outward_and_respects_solid_faces():
    axes = SimpleNamespace(xf=np.arange(4.), yf=np.arange(4.), zf=np.arange(2.))
    solid = np.zeros((1, 3, 3), dtype=bool)
    solid[0, 1, 2] = True
    model = ShallowLayerPressure(axes, solid, ground_z=0., depth_m=1.)
    points = np.array([[1.2, 1.5, .5], [1.8, 1.5, .5]])
    flat = np.array([4, 4])
    cells = np.array([4])
    inverse = np.array([0, 0])
    result = model.increment(points, flat, cells, inverse, np.array([2.]), 1., 9.81, .2)
    assert result[0, 0] < result[1, 0] < 0
    assert result[:, 1] == pytest.approx([0., 0.])
    solid[0, 1, 2] = False
    unblocked = model.increment(points, flat, cells, inverse, np.array([2.]), 1., 9.81, .2)
    assert unblocked[1, 0] == pytest.approx(-unblocked[0, 0])
    assert model.increment(points, flat, cells, inverse, np.array([1.]), 1., 9.81, .2) == pytest.approx(np.zeros((2, 2)))


def test_dense_contact_allows_descent_until_the_ground():
    thermo = SimpleNamespace(ambient_density=1., evaluate=lambda c: SimpleNamespace(
        density=np.full_like(c, 1.1), temperature=np.full_like(c, 200.)))
    args = (np.array([[.5, .5, .1], [.5, .5, .2], [.5, .5, .3]]), np.ones(3),
            np.zeros((3, 3)), np.arange(3), 1., 1.)
    original, _ = LH2Momentum(thermo).update(*args)
    contact, _ = LH2Momentum(thermo, ground_contact_z=0.).update(*args)
    np.testing.assert_array_equal(original[:, 2], 0.)
    np.testing.assert_allclose(contact[:, 2], [-.1, -.2, -.3])


def test_dilute_dense_layer_spreads_and_resists_lift_without_impeding_descent():
    axes = SimpleNamespace(xf=np.arange(4.), yf=np.arange(4.), zf=np.arange(2.),
                           cell_indices=lambda p: (np.zeros(len(p), int),
                                                   p[:, 1].astype(int), p[:, 0].astype(int)))
    solid = np.zeros((1, 3, 3), dtype=bool)
    dense = DenseLayerExchange(axes, solid, 0., 1.)
    legacy = ShallowLayerPressure(axes, solid, 0., 1.)
    points = np.array([[1.2, 1.5, .1], [1.8, 1.5, .1]])
    args = (points, np.array([4, 4]), np.array([4]), np.zeros(2, int), np.array([1.01]), 1., 9.81, .2)
    np.testing.assert_array_equal(legacy.increment(*args), 0.)
    push = dense.increment(*args)
    assert push[0, 0] < 0. < push[1, 0]
    w = dense.vertical_wind(points, np.array([.2, -.2]), np.array([.01, .01]))
    assert 0. < w[0] < .2
    assert w[1] == -.2
    assert dense.vertical_wind(np.array([[1.2, 1.5, 1.]]), np.array([.2]), np.array([.01]))[0] == .2
    supported = DenseLayerExchange(axes, solid, 0., 1., support_depth_m=.5)
    elevations = np.array([[1., 1., 0.], [1., 1., .25], [1., 1., .5], [1., 1., .25]])
    slip = np.array([-.4, -.4, -.4, .4])
    corrected = slip + supported.vertical_correction(elevations, np.zeros(4), slip, np.zeros(4))
    np.testing.assert_allclose(corrected, [0., -.2, -.4, .4])
    influence = np.zeros((3, 3))
    gated = DenseLayerExchange(axes, solid, 0., 1., wake_influence=influence)
    np.testing.assert_array_equal(gated.increment(*args), 0.)
    influence[1, 1] = 1.
    np.testing.assert_allclose(gated.increment(*args), push)
