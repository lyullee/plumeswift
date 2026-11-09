"""Vertical probability and particle statistics for prescribed-flow workflows."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .io import _open_netcdf
from .timing import milliseconds


STATISTICS = ('mean_w', 'variance_w', 'mean_delta_w_over_dt', 'variance_delta_w_over_dt')
LES_STATISTICS = ('subfilter_tke', 'covariance_u_w', 'variance_delta_w_over_dt')


def _particle_samples(snapshot, record, *, height, les):
    if snapshot.suffix.lower() != '.nc':
        if record != -1:
            raise ValueError('A CSV particle snapshot contains one record')
        samples = np.loadtxt(snapshot, delimiter=',', skiprows=5, usecols=(11, 12, 14, 17), ndmin=2).T
        return samples[0], samples

    with _open_netcdf(snapshot) as nc:
        dimensions = nc['zPos'].dimensions
        time_axes = [i for i, name in enumerate(dimensions) if name.casefold() in {'t', 'time'}]
        if len(dimensions) != 2 or len(time_axes) != 1:
            raise ValueError('Particle NetCDF requires a particle axis and a time axis')
        selection = (record, slice(None)) if time_axes[0] == 0 else (slice(None), record)

        def read(name):
            var = nc[name]
            if var.dimensions != dimensions:
                raise ValueError(f'Particle field {name} axes differ from zPos')
            return np.ma.asarray(var[selection], dtype=float).filled(np.nan)

        z = read('zPos')
        active = (read('isActive') == 1) & (z >= 0) & (z <= height)
        u = read('uFluct') if les else np.zeros_like(z)
        return z, np.array([z, u, read('wFluct'), read('delta_wFluct')])[:, active]


def particle_statistics(snapshot, turbulence, step, *, height=1., les=False, record=-1):
    if not np.isfinite((step, height)).all() or min(step, height) <= 0:
        raise ValueError('Particle statistics require a positive timestep and domain height')
    probability_z, samples = _particle_samples(Path(snapshot), record, height=height, les=les)
    z, u, w, delta = samples
    if not len(z) or not np.isfinite(samples).all() or np.any((z < 0) | (z > height)):
        raise ValueError('Expected finite active particle samples in the vertical domain')
    z = z/height
    edges = np.linspace(0, 1, 26)
    # The original probability plot uses every particle slot. The moments use active slots.
    probabilities = np.histogram(probability_z/height, edges)[0]*(25/len(probability_z))
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero*np.log(nonzero)))
    with _open_netcdf(turbulence) as nc:
        levels = np.asarray(nc['z'][:], dtype=float)
        stress = np.asarray(nc['tzz'][0], dtype=float).mean(axis=(1, 2))
        dissipation = np.asarray(nc['CoEps'][0], dtype=float).mean(axis=(1, 2))
        shear = np.asarray(nc['txz'][0], dtype=float).mean(axis=(1, 2)) if les else None
    gradient = np.gradient(stress, levels[1]-levels[0], edge_order=2)
    reference = np.array([np.zeros_like(levels), stress, gradient, dissipation])
    centers, observed, occupied = _moments(z, u, w, delta, step, 20, les=False)
    theory = np.array([np.interp(centers, levels/height, row) for row in reference])
    rmse = np.sqrt(np.mean((observed-theory)**2, axis=1))
    report = dict(entropy=entropy, occupied_statistics_bins=occupied,
                  rmse={key: float(value) if np.isfinite(value) else None
                        for key, value in zip(STATISTICS, rmse, strict=True)})
    arrays = dict(probability_edges=edges, relative_probability=probabilities, statistics_z=centers,
                  observed=observed, expected=theory, statistics=STATISTICS,
                  reference_z=levels/height, reference_profiles=reference)
    if les:
        centers_les, observed_les, occupied_les = _moments(z, u, w, delta, step, 100, les=True)
        profiles = np.array([stress, shear, dissipation])
        report['occupied_les_statistics_bins'] = occupied_les
        arrays.update(les_statistics_z=centers_les, les_observed=observed_les,
                      les_reference_profiles=profiles, les_statistics=LES_STATISTICS)
    return report, arrays


def write_particle_diagnostics(snapshot, turbulence, output, step, *, height=1., les=False, record=-1):
    report, arrays = particle_statistics(snapshot, turbulence, step, height=height, les=les, record=record)
    output = Path(output).with_suffix('.npz')
    if output.resolve() in (Path(snapshot).resolve(), Path(turbulence).resolve()):
        raise ValueError('Particle diagnostic output must differ from the input files')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return dict(report, arrays=str(output))


def _moments(z, u, w, delta, step, count, *, les):
    edges = np.linspace(0, 1, count+1)
    centers = (edges[:-1]+edges[1:])/2
    bin_ids = np.minimum(np.searchsorted(edges, z, side='right')-1, count-1)
    occupied = np.unique(bin_ids)
    observed = np.full((len(LES_STATISTICS if les else STATISTICS), count), np.nan)
    for bin_id in occupied:
        mask = bin_ids == bin_id
        u_bin, w_bin, delta_bin = u[mask], w[mask], delta[mask]
        ddof = int(len(w_bin) > 1)
        if les:
            covariance = np.sum((u_bin-u_bin.mean())*(w_bin-w_bin.mean()))/(len(w_bin)-ddof)
            observed[:, bin_id] = w_bin.var(ddof=ddof), covariance, delta_bin.var(ddof=ddof)/step
        else:
            observed[:, bin_id] = (w_bin.mean(), w_bin.var(ddof=ddof),
                                   delta_bin.mean()/step, delta_bin.var(ddof=ddof)/step)
    return centers, observed, int(len(occupied))


def write_case_diagnostics(xml, turbulence, step, duration, *, period=None, height=1., les=False):
    # Each output is written on the first model step at or after its target.
    end, increment = milliseconds(duration), milliseconds(step)
    interval = end if period is None else milliseconds(period)
    target_ms = end//interval*interval
    snapshot_ms = min(end, (target_ms+increment-1)//increment*increment)
    stamp = datetime(2022, 1, 1)+timedelta(milliseconds=snapshot_ms)
    snapshot = xml.with_name(f"{xml.stem}_base_particleData_{stamp.isoformat().replace(':', '-')}.csv")
    target = xml.with_name(f'{xml.stem}_diagnostics.npz')
    return write_particle_diagnostics(snapshot, turbulence, target, step, height=height, les=les)
