"""Optional mean-field sampling correction, independent of fluid EOS.

For an equal-weight Monte Carlo marker selected at x, the other markers are
an independent estimate of the Eulerian concentration at x. Including its
own mass gives a positive self-contribution m/V even in a vanishing plume.
Removing one cell-average marker is an unbiased concentration estimator
up to the finite population factor N/(N-1). It does not guarantee unbiased
nonlinear density, and is NOT a replacement for particle-count convergence.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeaveOneOutDensity:
    """Return density for forces; keep full cell mass/energy for heat transfer.

    Inputs are occupied-cell mass [kg], energy [J], volume [m3] and marker
    count. All markers in a cell share the reconstructed force density.
    This cell-average variant is exact for equal marker masses and enthalpies;
    different thermal histories are represented by their cell mean. Unequal
    marker weights require a per-marker estimator and are not supported here.
    """
    def density(self, thermo, mass, energy, volume, counts):
        if energy is None:
            raise ValueError('Leave-one-out density requires transported enthalpy')
        scale = 1.-1./np.asarray(counts)
        return thermo.evaluate(mass*scale/volume, energy*scale/volume).density
