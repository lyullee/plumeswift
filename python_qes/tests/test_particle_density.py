from types import SimpleNamespace

import numpy as np
import pytest

from plumeswift.lh2 import LH2Momentum
from plumeswift.thermodynamics.ammonia_moist import MoistAmmoniaVapour
from plumeswift.thermodynamics.particle_density import LeaveOneOutDensity


def test_leave_one_out_has_no_isolated_marker_self_buoyancy():
    thermo=MoistAmmoniaVapour(289.65,102000.,62.)
    model=LH2Momentum(thermo,force_density_reconstruction=LeaveOneOutDensity())
    args=(np.array([[0.,0.,2.]]),np.array([.003]),np.zeros((1,3)),np.array([0]),1.,.25)
    velocity,accounting=model.update(*args,enthalpy=np.array([-1.25e6]))
    np.testing.assert_allclose(velocity,0.,atol=1e-12)
    assert accounting['active_mass_kg']==pytest.approx(.003)
    ordinary=LH2Momentum(thermo).update(*args,enthalpy=np.array([-1.25e6]))[0]
    assert ordinary[0,2]<0.


def test_concentration_estimator_removes_exact_bernoulli_self_bias():
    # Given a tagged marker, each of the remaining N-1 markers independently
    # occupies its cell with probability p. Compute the expectation exactly.
    from scipy.stats import binom
    n,p,mass,volume=30,.04,.003,1.
    others=np.arange(n)
    probability=binom.pmf(others,n-1,p)
    full=(1+others)*mass
    class LinearDensity:
        def evaluate(self,c,e):
            return SimpleNamespace(density=1.+c)
    reconstructed=LeaveOneOutDensity().density(LinearDensity(),full,np.zeros(n),volume,1+others)-1.
    assert probability@reconstructed==pytest.approx((n-1)*p*mass/volume)
    assert probability@(full/volume)-(probability@reconstructed)==pytest.approx(mass/volume)


def test_unequal_marker_mass_rejected_and_high_count_limit():
    thermo=MoistAmmoniaVapour(289.65,102000.,0.)
    model=LH2Momentum(thermo,force_density_reconstruction=LeaveOneOutDensity())
    with pytest.raises(ValueError,match='equal marker masses'):
        model.update(np.array([[0.,0.,2.],[0.,0.,2.]]),np.array([.001,.002]),
                     np.zeros((2,3)),np.array([0,0]),1.,.25,enthalpy=np.full(2,-1.25e6))
    c=np.array([.03])
    direct=thermo.evaluate(c,c*-1.25e6).density
    estimate=LeaveOneOutDensity().density(thermo,c,c*-1.25e6,1.,np.array([1000000]))
    np.testing.assert_allclose(estimate,direct,rtol=1e-6)
