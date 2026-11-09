"""EPA 7x11 array concentration sections from the original MATLAB workflow."""
import json
from pathlib import Path

import numpy as np

from .analytical_results import _continuous_normalization, _read_collection


def analyze_epa_results(collection, *, record=-1, xml=None):
    """Evaluate the original three y and five z section selections."""
    coordinates, counts, averaging = _read_collection(collection, record)
    if xml is None:
        source, step, particles, averaging = (50, 200, 1), 1., 200, 1800.
    else:
        source, step, particles, averaging = _continuous_normalization(xml, averaging)
    height, speed = 15., 2.83
    x = (coordinates['x']-source[0])/height
    y = (coordinates['y']-source[1])/height
    z = coordinates['z']/height
    y_indices = np.array([np.searchsorted(y, 0), np.searchsorted(y, .2), np.searchsorted(y, 1, side='right')])
    z_indices = np.array([np.searchsorted(z, .25), np.searchsorted(z, .50), np.searchsorted(z, .75),
                          np.searchsorted(z, 1, side='right'), np.searchsorted(z, 1.25)])
    if max(y_indices) >= len(y) or max(z_indices) >= len(z):
        raise ValueError('Collection grid does not cover the original EPA section positions')
    volume = float(np.prod([np.mean(np.diff(axis)) for axis in coordinates.values()]))
    factor = (step/averaging/volume)*(speed*height**2/(particles/step))
    # Stored arrays are section, vertical/horizontal axis, downwind axis.
    vertical = counts[:, y_indices, :].transpose(1, 0, 2)*factor
    horizontal = counts[z_indices]*factor
    arrays = dict(x_over_h=x, y_over_h=y, z_over_h=z, y_sections_over_h=y[y_indices],
                  z_sections_over_h=z[z_indices], vertical_cstar=vertical, horizontal_cstar=horizontal)
    report = dict(record=record, normalization='matlab' if xml is None else 'xml', source_m=source,
                  time_step_s=step, particles_per_step=particles, averaging_s=averaging,
                  y_sections_over_h=y[y_indices].astype(float).tolist(),
                  z_sections_over_h=z[z_indices].astype(float).tolist(),
                  maximum_cstar=float(max(vertical.max(), horizontal.max())))
    return report, arrays


def _write_section_figures(output, arrays):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    files = []
    for orientation, sections, coords, ylabel, xlim, ylim, size in (
        ('vertical', 'y_sections_over_h', 'z_over_h', 'z / H', (-1, 19), (0, 4), (9, 3)),
        ('horizontal', 'z_sections_over_h', 'y_over_h', 'y / H', (-2, 22), (-12, 12), (7, 7)),
    ):
        for i, level in enumerate(arrays[sections]):
            figure = Figure(figsize=size, layout='constrained')
            FigureCanvasAgg(figure)
            axis = figure.add_subplot()
            values = arrays[f'{orientation}_cstar'][i][:-1, :-1]
            mesh = axis.pcolormesh(arrays['x_over_h'], arrays[coords], np.ma.masked_less_equal(values, 0),
                                   shading='flat', norm=LogNorm(.001, 1), cmap='bone_r', rasterized=True)
            axis.set(xlabel='x / H', ylabel=ylabel, xlim=xlim, ylim=ylim, title=f'{sections[0]} / H = {level:.3f}')
            figure.colorbar(mesh, ax=axis, label='C*')
            path = output/f'EPA_{orientation}_{i+1}.pdf'
            figure.savefig(path)
            files.append(str(path))
    return files


def write_epa_results(collection, output, *, record=-1, xml=None, plot=False):
    report, arrays = analyze_epa_results(collection, record=record, xml=xml)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/'EPA_sections.npz'
    np.savez_compressed(archive, **arrays)
    report['sections'] = str(archive)
    if plot:
        report['plots'] = _write_section_figures(output, arrays)
    (output/'EPA_sections.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
