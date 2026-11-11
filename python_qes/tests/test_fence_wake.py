import numpy as np
import pytest

from plumeswift.building_pipeline import BuildingPipeline
from plumeswift.domain import Domain
from plumeswift.fence_wake import FenceShearTurbulence, SolidFenceWake
from plumeswift.geometry import PolygonBuilding
from plumeswift.meteorology import SensorProfile, single_sensor_flat
from plumeswift.turbulence import LocalTurbulence
from plumeswift.wind_case import WindCase


@pytest.mark.parametrize(('axis', 'direction', 'fence'), [
    ('y', 0., PolygonBuilding.rectangle(3., 9., 16., 1., 2.)),
    ('x', 270., PolygonBuilding.rectangle(7., 3., 1., 18., 2.)),
])
def test_streamfunction_flux_is_conservative_before_wall_masking(axis, direction, fence):
    domain = Domain((27, 23, 38), (.75, .5, .5))
    profile = SensorProfile(3., direction, 3.)
    workspace = domain.workspace()
    single_sensor_flat(workspace, profile)
    before = {key: workspace.arrays[key+'0'].copy() for key in 'uvw'}
    pipeline = BuildingPipeline((SolidFenceWake(profile, normal_axis=axis),))
    context = pipeline.apply(workspace, ((0, fence),))
    u, v, w = (workspace.arrays[key+'0']-before[key] for key in 'uvw')
    dx, dy, dz = domain.spacing
    divergence = (np.diff(u, axis=2)[:-1, :-1]/dx +
                  np.diff(v, axis=1)[:-1, :, :-1]/dy +
                  np.diff(w, axis=0)[:, :-1, :-1]/dz)
    np.testing.assert_allclose(divergence, 0., atol=2e-14)
    assert np.any(w > 0) and np.any(w < 0)
    assert np.any(context.fence_variance[domain.coordinates['z'] > fence.height] > 0)


def test_shear_layer_changes_turbulence_without_changing_projected_wind():
    domain = Domain((24, 36, 38), (1., .5, .5))
    profile = SensorProfile(3., 0., 3.)
    fence = PolygonBuilding.rectangle(3., 12., 18., 1., 2.)
    pipeline = BuildingPipeline((SolidFenceWake(profile),))
    kwargs = dict(buildings=(fence,), building_pipeline=pipeline)
    mean = WindCase(domain, profile, turbulence=LocalTurbulence(native_quirks=False), **kwargs).run()
    full = WindCase(domain, profile, turbulence=FenceShearTurbulence(), **kwargs).run()
    assert mean['solver'].converged and full['solver'].converged
    for key in 'uvw':
        np.testing.assert_array_equal(mean['workspace'].arrays[key], full['workspace'].arrays[key])
    layer = full['turbulence']['tzz']-mean['turbulence']['tzz']
    assert np.all(layer >= 0)
    assert np.any(layer[domain.coordinates['z'] > fence.height] > 0)
    assert np.all(layer[:, domain.coordinates['y'] > 13., :] == 0)
    mixed = WindCase(domain, profile, turbulence=FenceShearTurbulence(
        vertical_shear_std=.5, shear_height=fence.height), **kwargs).run()
    for key in 'uvw':
        np.testing.assert_array_equal(mixed['workspace'].arrays[key], full['workspace'].arrays[key])
    extra = mixed['turbulence']['tzz'] - full['turbulence']['tzz']
    assert np.any(extra > 0.)
    assert np.all(extra[:, domain.coordinates['y'] > 13., :] == 0.)
    assert np.any(mixed['turbulence']['div_tau_z'] != full['turbulence']['div_tau_z'])


def test_short_gap_keeps_the_streamfunction_return_below_the_crest():
    domain = Domain((24, 36, 38), (1., .5, .5))
    profile = SensorProfile(3., 0., 3.)
    fences = tuple(enumerate(PolygonBuilding.rectangle(3., y, 18., 1., 2.) for y in (12., 8.)))
    flows = []
    for confined in (False, True):
        workspace = domain.workspace()
        single_sensor_flat(workspace, profile)
        BuildingPipeline((SolidFenceWake(profile, confined_gap=confined),)).apply(workspace, fences)
        flows.append(workspace.arrays['w0'])
    k = np.searchsorted(domain.coordinates['z_face'], 2.)
    j = np.argmin(abs(domain.coordinates['y']-10.))
    assert flows[0][k, j, 10] < 0.
    assert flows[1][k, j, 10] == 0.
    assert np.any(flows[1][1:k, j, 10] != 0.)
