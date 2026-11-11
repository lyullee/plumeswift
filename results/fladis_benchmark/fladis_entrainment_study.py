"""Compare dilution of vertical momentum with the archived FLADIS runs."""
from concurrent.futures import ProcessPoolExecutor, as_completed

from compare_fladis_force import main as compare
from fladis_sensitivity import run_job
from run_fladis import ROOT, TRIALS


OUTPUT = ROOT / 'entrainment_study'


def run_trial(trial):
    return run_job((trial, 'loo_moist_midpoint', 120), output=OUTPUT / 'entrainment',
                   track_mixture_inertia=True)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(TRIALS)) as pool:
        jobs = {pool.submit(run_trial, trial): trial for trial in TRIALS}
        for future in as_completed(jobs):
            result = future.result()
            print(f'{jobs[future]} completed in {result["wall_s"]:.1f} s', flush=True)
    compare(OUTPUT, {'reference': 'Reference', 'entrainment': 'Momentum dilution'},
            {}, profile_errors=True)
