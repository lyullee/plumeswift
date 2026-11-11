"""FLADIS sensitivity to density linearization and turbulent drift relaxation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np

from fladis_sensitivity import run_job
from run_fladis import ROOT, TRIALS
from plumeswift.thermodynamics.ammonia_moist import CPA, RA, RN, RW


OUTPUT = ROOT / 'drift_study'
CASES = ('density', 'velocity', 'combined')


def ambient_linear_density(thermo, ids, mass, energy, volume, counts):
    """First-order constant-pressure density about unsaturated ambient air."""
    loading = thermo.water_loading
    air_cp = (CPA + loading * 1860.) / (1. + loading)
    air_gas_constant = (RA + loading * RW) / (1. + loading)
    scale = (1. - 1. / counts) / volume
    density_anomaly = ((1. - RN / air_gas_constant) * mass
                       - energy / (air_cp * thermo.ambient_temperature)) * scale
    return thermo.ambient_density + density_anomaly


def relax_vertical_drift(sampler, positions, inverse, cell_mass, masses, velocity, acceleration, dt):
    """Relax mean vertical slip on the local GLE integral correlation time."""
    stress, _, diffusion, _ = sampler.sample_turbulence(positions)
    vertical_row = stress[:, 2, :]
    variance = vertical_row[:, 2]
    squared = np.einsum('ij,ij->i', vertical_row, vertical_row)
    rate = np.divide(diffusion * variance, 2. * squared,
                     out=np.zeros_like(variance), where=(variance > 0.) & (squared > 0.))
    weighted_rate = np.bincount(inverse, weights=masses * rate, minlength=len(cell_mass))
    rate = np.divide(weighted_rate, cell_mass, out=np.zeros_like(cell_mass), where=cell_mass > 0.)
    response = np.divide(-np.expm1(-rate * dt), rate,
                         out=np.full_like(rate, dt), where=rate > 0.)
    return np.exp(-rate * dt) * velocity + response * acceleration


def configure(case, momentum, sampler):
    if case in ('density', 'combined'):
        momentum.vertical_force_density = ambient_linear_density
    if case in ('velocity', 'combined'):
        momentum.vertical_drift_update = partial(relax_vertical_drift, sampler)


def run_case(trial, case):
    return run_job((trial, 'loo_moist_midpoint', 120), output=OUTPUT / case,
                   configure_momentum=partial(configure, case))


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(run_case, trial, case): (trial, case)
                for trial in TRIALS for case in CASES}
        for future in as_completed(jobs):
            trial, case = jobs[future]
            result = future.result()
            print(f'{trial} {case} completed in {result["wall_s"]:.1f} s', flush=True)
