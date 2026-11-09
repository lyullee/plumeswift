"""Profile and error diagnostics from the QES continuous-release examples."""
import json
from pathlib import Path

import numpy as np
from scipy.special import i0e

from .analytical_cases import CASES
from .io import _open_netcdf


_PROFILE_TARGETS = {
    'uniform-x': (.4, .8, 1.),
    'uniform-y': (.393, .464, .964),
    'powerlaw-x': (6., 10., 19.),
    'powerlaw-y': (6., 10., 19.),
}


def _read_collection(path, record):
    with _open_netcdf(path) as dataset:
        count_key = 'p_count' if 'p_count' in dataset.variables else 'pBox'
        count_var = dataset[count_key]
        axis_keys = {axis: axis+'_c' if axis+'_c' in dataset.variables else axis for axis in 'zyx'}
        if any(dataset[key].ndim != 1 for key in axis_keys.values()):
            raise ValueError('Collection axes must be coordinate vectors')
        dimensions = tuple(dataset[key].dimensions[0] for key in axis_keys.values())
        stored = tuple(dim for dim in count_var.dimensions if dim != 't')
        if len(stored) != 3 or len(set(dimensions)) != 3 or set(dimensions) != set(stored):
            raise ValueError('Collection counts must use the three coordinate dimensions')
        other = set(count_var.dimensions)-set(dimensions)
        if other:
            if other != {'t'}:
                raise ValueError('Unexpected particle-count dimension')
            index = tuple(record if dim == 't' else slice(None) for dim in count_var.dimensions)
        else:
            if record not in (-1, 0):
                raise IndexError('Collection has one record')
            index = (...,)
        values = count_var[index]
        coordinates = {axis: dataset[key][:] for axis, key in axis_keys.items()}
        time_var = dataset.variables.get('t_avg')
        averaging = None if time_var is None else time_var[record if time_var.ndim else ...]
    fields = [values, *coordinates.values()] + ([] if averaging is None else [averaging])
    if any(np.ma.is_masked(field) or not np.isfinite(field).all() for field in fields):
        raise ValueError('Missing or nonfinite collection data')
    coords = {axis: np.asarray(values) for axis, values in coordinates.items()}
    if any(axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0) for axis in coords.values()):
        raise ValueError('Collection axes must be increasing vectors with at least two cells')
    values = np.asarray(values).transpose(*(stored.index(dim) for dim in dimensions))
    if np.any(values < 0):
        raise ValueError('Expected nonnegative particle counts')
    return coords, values, None if averaging is None else float(averaging)


def _normalization(name, averaging, xml):
    case = CASES[name]
    if xml is None:
        # The uniform X MATLAB postprocessor uses 400, while its case XML uses 200.
        return 1., 400 if name == 'uniform-x' else 200, case.duration-case.average_start
    point, step, particles, averaging = _continuous_normalization(xml, averaging)
    if point != case.source:
        raise ValueError('Point source coordinates differ from the analytical example')
    return step, particles, averaging


def _continuous_normalization(xml, averaging):
    from .qes_xml import read_qes_xml

    root = read_qes_xml(xml)
    params = root.find('plumeParameters')
    release, = root.findall('.//source/releaseType_continuous')
    if len(root.findall('.//source')) != 1:
        raise ValueError('Expected one continuous point source')
    source, = root.findall('.//source/sourceGeometry_Point')
    point = tuple(float(source.findtext('pos'+axis)) for axis in 'XYZ')
    step = float(params.findtext('timeStep'))
    particles = int(release.findtext('particlePerTimestep'))
    if averaging is None:
        averaging = float(root.findtext('collectionParameters/timeAvgFreq'))
    if not np.isfinite([step, averaging]).all() or min(step, averaging, particles) <= 0:
        raise ValueError('Normalization requires positive timestep, averaging time and particle count')
    return point, step, particles, averaging


def _reference_profiles(name, distance, lateral, height, *, profile=False):
    """Return separable factors for dimensionless concentration, C U H²/Q."""
    h = CASES[name].source[2]
    if name.startswith('uniform'):
        zi = 1000 if name.endswith('-x') else 10000
        time = distance/2
        spread = .174*time/(1+np.sqrt(time*2.5*.174/zi))
        sy, sz = 1.78*spread, (1/(3*.4*.4))**(1/3)*spread
        lat = np.exp(-.5*(lateral/sy)**2)
        vert = np.exp(-.5*((height-h)/sz)**2)
        if profile and name == 'uniform-x':
            vert += np.exp(-.5*((height+h)/sz)**2)
        return lat*h*h/(2*np.pi*sy*sz), vert
    a, b, alpha = 4.8, .08, 1.15
    sy = .32*distance**.78
    # n=1 in both originals. Combine the exponential and I0 scaling
    # analytically to avoid overflow and cancellation near the source.
    zr, hr = height**(alpha/2), h**(alpha/2)
    scale = a/(b*alpha**2*distance)
    vert = np.exp(-scale*(zr-hr)**2)*i0e(2*scale*zr*hr)/(b*alpha*distance)
    lat = np.exp(-.5*(lateral/sy)**2)*(a*h**.15)*h*h/(np.sqrt(2*np.pi)*sy)
    return lat, vert


def _matlab_round(values):
    return np.copysign(np.floor(np.abs(values)+.5), values)


def analyze_analytical_case(name, collection, *, record=-1, xml=None):
    """Return the report and arrays used in the four original MATLAB analyses."""
    coords, counts, averaging = _read_collection(collection, record)
    step, particles, averaging = _normalization(name, averaging, xml)
    case = CASES[name]
    uniform = name.startswith('uniform')
    x_direction = name.endswith('-x')
    h = case.source[2]
    source_lateral = case.source[1 if x_direction else 0]
    source_stream = case.source[0 if x_direction else 1]
    stream, lateral = 'xy' if x_direction else 'yx'
    distance_m = coords[stream]-source_stream
    downwind = distance_m/h
    crosswind, z = coords[lateral].astype(float), coords['z'].astype(float)
    if not uniform and z[0] < 0:
        raise ValueError('Power-law reference requires nonnegative heights')
    volume = float(np.prod([np.mean(np.diff(axis)) for axis in coords.values()]))
    speed = 2. if uniform else 4.8*h**.15
    # (streamwise station, transverse coordinate, vertical coordinate)
    order = (2, 1, 0) if x_direction else (1, 2, 0)
    values = counts.transpose(order)*(step/averaging/volume)*(speed*h*h/(particles/step))

    def stations(targets, scale):
        result = np.searchsorted(_matlab_round(downwind*scale), _matlab_round(np.asarray(targets)*scale))
        if np.any(result >= len(downwind)) or np.any(downwind[result] <= 0):
            raise ValueError('Collection grid does not cover the downstream profile stations')
        return result

    indices = stations(downwind[downwind > (.2 if uniform else 2.)], 100)
    if not len(indices):
        raise ValueError('Collection grid has no cells in the downstream comparison range')
    prediction = np.ascontiguousarray(values[indices])
    reference = np.stack([np.multiply.outer(*_reference_profiles(name, float(distance_m[i]),
                                                                 crosswind-source_lateral, z))
                          for i in indices])
    difference = prediction-reference
    residual = float(np.sum(difference**2))
    total = float(np.sum((prediction-prediction.mean())**2))
    peak = reference.max(axis=(1, 2))
    max_relative = (np.max(abs(difference), axis=(1, 2))/peak).tolist()
    report = dict(case=name, record=record, normalization='matlab' if xml is None else 'xml',
                  averaging_s=averaging, time_step_s=step, particles_per_step=particles,
                  sample_cells=int(prediction.size), r_squared=1-residual/total if total else None,
                  rmse=float(np.sqrt(residual/prediction.size)), max_relative_error=max(max_relative),
                  max_relative_error_by_station=max_relative)

    profile_indices = stations(_PROFILE_TARGETS[name], 1000 if uniform else 100)
    if uniform:
        model_lateral = np.linspace(-h, h, 20*h+1)
        model_z = np.linspace(0, 2*h, 20*h+1)
        cross_index = int(np.argmin(abs(crosswind-source_lateral)))
        z_index = int(np.argmin(abs(z-h)))
        if crosswind[cross_index] != source_lateral or z[z_index] != h:
            raise ValueError('Uniform example requires collection centers at the source height and crosswind location')
        model_cross_index = model_z_index = 10*h
    else:
        resolution = 100 if x_direction else 10
        model_lateral = np.linspace(-50, 50, 100*resolution+1)
        model_z = np.linspace(0, 5*h, 5*h*resolution+1)
        cross_index = np.searchsorted(crosswind, source_lateral)
        z_index = np.searchsorted(z, h)-int(x_direction)
        if cross_index >= len(crosswind) or not 0 <= z_index < len(z):
            raise ValueError('Power-law collection grid does not cover the profile center')
        model_cross_index = np.searchsorted(model_lateral-model_lateral[0], crosswind[cross_index])
        model_z_index = np.searchsorted(model_z, z[z_index])
        if model_cross_index >= len(model_lateral) or model_z_index >= len(model_z):
            raise ValueError('Power-law collection grid exceeds the analytical profile axes')
    model_profiles = [_reference_profiles(name, float(distance_m[i]), model_lateral, model_z, profile=True)
                      for i in profile_indices]
    centered = name == 'uniform-x'
    arrays = dict(
        profile_downwind_over_h=downwind[profile_indices], metrics_downwind_over_h=downwind[indices],
        crosswind_over_h=(crosswind-(source_lateral if centered else 0))/h,
        model_crosswind_over_h=(model_lateral+(0 if centered else source_lateral))/h,
        z_over_h=z/h, model_z_over_h=model_z/h,
        lateral_qes=values[profile_indices, :, z_index], vertical_qes=values[profile_indices, cross_index, :],
        lateral_reference=np.stack([lat*vert[model_z_index] for lat, vert in model_profiles]),
        vertical_reference=np.stack([lat[model_cross_index]*vert for lat, vert in model_profiles]),
        comparison_qes=prediction, comparison_reference=reference)
    return report, arrays


def _plot_profiles(output, name, arrays, report):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    profiles = Figure(figsize=(10, 6), layout='constrained')
    FigureCanvasAgg(profiles)
    for k, distance in enumerate(arrays['profile_downwind_over_h']):
        ax = profiles.add_subplot(2, 3, k+1)
        ax.plot(arrays['model_crosswind_over_h'], arrays['lateral_reference'][k], label='Analytical')
        ax.plot(arrays['crosswind_over_h'], arrays['lateral_qes'][k], 's:', markersize=3, label='QES')
        ax.set(xlabel='Crosswind / H', ylabel='C*', title=f'Downwind / H = {distance:.3f}')
        if k == 0:
            ax.legend()
        ax = profiles.add_subplot(2, 3, k+4)
        ax.plot(arrays['vertical_reference'][k], arrays['model_z_over_h'])
        ax.plot(arrays['vertical_qes'][k], arrays['z_over_h'], 's:', markersize=3)
        ax.set(xlabel='C*', ylabel='Height / H')
    profiles_path = output/f'{name}_ModelComp.pdf'
    profiles.savefig(profiles_path)

    scatter = Figure(figsize=(6, 6), layout='constrained')
    FigureCanvasAgg(scatter)
    ax = scatter.add_subplot()
    x, y = arrays['comparison_reference'].ravel(), arrays['comparison_qes'].ravel()
    limit = max(x.max(), y.max())
    ax.scatter(x, y, s=3, alpha=.4, rasterized=True)
    ax.plot([0, limit], [0, limit], color='black', linewidth=1)
    bounds = (0, 1.04*limit)
    ax.set(xlabel='Analytical C*', ylabel='QES C*', title=f"RMSE = {report['rmse']:.4g}",
           xlim=bounds, ylim=bounds, aspect='equal')
    scatter_path = output/f'{name}_1to1.pdf'
    scatter.savefig(scatter_path)
    return [str(profiles_path), str(scatter_path)]


def write_analytical_results(name, collection, output, *, record=-1, xml=None, plot=False):
    report, arrays = analyze_analytical_case(name, collection, record=record, xml=xml)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    array_path = output/f'{name}_profiles.npz'
    np.savez_compressed(array_path, **arrays)
    report['profiles'] = str(array_path)
    if plot:
        report['plots'] = _plot_profiles(output, name, arrays, report)
    (output/f'{name}_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
