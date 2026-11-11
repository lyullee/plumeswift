"""Isolate vertical buoyancy, marker count, and buoyancy-density resolution."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from fladis_sensitivity import run_job
from run_fladis import ROOT, TRIALS
from plumeswift.thermodynamics.particle_density import LeaveOneOutDensity


OUTPUT = ROOT / 'force_study'
CASES = {
    'no_buoyancy': (120, False, False),
    'more_particles': (240, True, False),
    'coarse_buoyancy': (120, True, True),
}


@dataclass(frozen=True)
class PairedVerticalCells:
    """Pool mass, enthalpy and counts across two adjacent fluid layers."""

    shape: tuple
    upper_layer_exclusive: int

    def __call__(self, thermo, ids, mass, energy, volume, counts):
        plane_size = self.shape[1] * self.shape[2]
        layer, column = np.divmod(ids, plane_size)
        if np.any((layer < 1) | (layer >= self.upper_layer_exclusive)):
            raise ValueError('Buoyancy grid requires cells inside the flat fluid domain')
        anchor = 1 + 2 * ((layer - 1) // 2)
        groups, first, inverse = np.unique(anchor * plane_size + column,
                                           return_index=True, return_inverse=True)
        pooled_volume = volume[first] * np.minimum(2, self.upper_layer_exclusive - groups // plane_size)
        density = LeaveOneOutDensity().density(
            thermo, np.bincount(inverse, weights=mass),
            np.bincount(inverse, weights=energy), pooled_volume,
            np.bincount(inverse, weights=counts))
        return density[inverse]


def run_case(trial, case):
    rate, buoyancy, coarse = CASES[case]
    return run_job((trial, 'loo_moist_midpoint', rate), output=OUTPUT / case,
                   vertical_buoyancy=buoyancy, coarsen_buoyancy=coarse)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_case, trial, case): (trial, case)
                for trial in TRIALS for case in CASES}
        for future in as_completed(jobs):
            trial, case = jobs[future]
            result = future.result()
            print(f'{trial} {case} completed in {result["wall_s"]:.1f} s', flush=True)
