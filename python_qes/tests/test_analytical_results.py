from math import exp, pi, sqrt

import numpy as np
import pytest
from netCDF4 import Dataset
from scipy.special import iv

from plumeswift.analytical_cases import CASES, _write_config
from plumeswift.analytical_results import analyze_analytical_case


@pytest.mark.parametrize('name', CASES)
def test_analytical_diagnostics_match_original_formulas(name, tmp_path):
    case = CASES[name]
    coordinates = {axis: lo+(np.arange(n)+.5)*(hi-lo)/n
                   for axis, lo, hi, n in zip('xyz', case.lower, case.upper, case.boxes, strict=True)}
    counts = np.arange(np.prod(case.boxes), dtype=np.int32).reshape(case.boxes) % 101
    path = tmp_path/'old.nc'
    with Dataset(path, 'w') as nc:
        for axis, values in coordinates.items():
            nc.createDimension(axis, len(values))
            nc.createVariable(axis, 'f4', (axis,))[:] = values
        nc.createVariable('pBox', 'i4', ('x', 'y', 'z'))[:] = counts

    report, arrays = analyze_analytical_case(name, path)
    assert report['particles_per_step'] == (400 if name == 'uniform-x' else 200)
    h = case.source[2]
    distance = float(arrays['metrics_downwind_over_h'][0])*h
    x_direction = name.endswith('-x')
    stream, lateral = ('x', 'y') if x_direction else ('y', 'x')
    first = np.argmin(abs(coordinates[stream]-case.source['xyz'.index(stream)]-distance))
    distance = float(np.float32(coordinates[stream][first])-np.float32(20.))
    iy, iz = len(coordinates[lateral])//2, len(coordinates['z'])//2
    cross = coordinates[lateral][iy]-50
    z = coordinates['z'][iz]
    if name.startswith('uniform'):
        zi = 1000 if x_direction else 10000
        travel = distance/2
        reduction = 1/(1+sqrt(travel/(zi/(2.5*.174))))
        sy = 1.78*.174*travel*reduction
        sz = (1/(3*.4*.4))**(1/3)*.174*travel*reduction
        reference = h*h/(2*pi*sy*sz)*exp(-.5*(cross/sy)**2-.5*((z-h)/sz)**2)
        speed = 2
    else:
        a, b, alpha = 4.8, .08, 1.15
        sy = .32*distance**.78
        speed = a*h**.15
        reference = (exp(-.5*(cross/sy)**2)/(sqrt(2*pi)*sy)
                     * exp(-a*(z**alpha+h**alpha)/(b*alpha**2*distance))
                     * iv(0, 2*a*(z*h)**(.5*alpha)/(b*alpha**2*distance))/(b*alpha*distance)
                     * speed*h*h)
    assert arrays['comparison_reference'][0, iy, iz] == pytest.approx(reference)
    volume = np.prod([(hi-lo)/n for lo, hi, n in zip(case.lower, case.upper, case.boxes, strict=True)])
    count = counts[first, iy, iz] if x_direction else counts[iy, first, iz]
    expected = count*speed*h*h/(volume*report['averaging_s']*report['particles_per_step'])
    assert arrays['comparison_qes'][0, iy, iz] == pytest.approx(expected, rel=2e-6)
    assert arrays['lateral_qes'].shape == (3, len(coordinates[lateral]))
    assert arrays['vertical_qes'].shape == (3, len(coordinates['z']))

    modern = tmp_path/'new.nc'
    with Dataset(modern, 'w') as nc:
        nc.createDimension('t', 2)
        for axis, values in coordinates.items():
            dim = axis+'_c'
            nc.createDimension(dim, len(values))
            nc.createVariable(dim, 'f4', (dim,))[:] = values
        nc.createVariable('p_count', 'i4', ('t', 'z_c', 'y_c', 'x_c'))[:] = [counts.T*2, counts.T]
        nc.createVariable('t_avg', 'f4', ('t',))[:] = [8, 4]
    actual, modern_arrays = analyze_analytical_case(name, modern)
    assert actual == report
    for key in arrays:
        np.testing.assert_array_equal(modern_arrays[key], arrays[key])
    xml = tmp_path/'plume.xml'
    _write_config(xml, case, name.startswith('uniform'), steps=8, particles_per_step=12)
    modern_report, normalized = analyze_analytical_case(name, modern, record=0, xml=xml)
    assert modern_report['averaging_s'] == 8 and modern_report['particles_per_step'] == 12
    factor = 2*report['averaging_s']*report['particles_per_step']/(8*12)
    np.testing.assert_allclose(normalized['comparison_qes'], factor*arrays['comparison_qes'], rtol=1e-14)
