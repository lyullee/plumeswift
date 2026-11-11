"""Compare relative concentration errors for the three FLADIS drift changes."""
from compare_fladis_force import main
from fladis_drift_study import OUTPUT


if __name__ == '__main__':
    main(OUTPUT, {
        'reference': 'Reference',
        'density': 'Linearized density',
        'velocity': 'Vertical slip relaxation',
        'combined': 'Both changes',
    }, {}, profile_errors=True)
