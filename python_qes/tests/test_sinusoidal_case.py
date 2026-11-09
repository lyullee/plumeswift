from xml.etree import ElementTree as ET

import numpy as np

from plumeswift.io import _open_netcdf
from plumeswift.plume_config import parse_plume_config
from plumeswift.sinusoidal_case import TIME_STEPS, prepare_sinusoidal_case, run_sinusoidal_case


def test_sinusoidal_files_preserve_original_axes_and_timestep_series(tmp_path):
    wind, turbulence, configs = prepare_sinusoidal_case(tmp_path)
    assert tuple(configs) == TIME_STEPS
    with _open_netcdf(wind) as nc:
        np.testing.assert_allclose(nc['x'][:], (np.arange(20)+.5)/20)
        np.testing.assert_allclose(nc['z_face'][:], (np.arange(52)-1)/49)
        assert nc['u'].shape == (1, 52, 21, 21)
        assert (nc['u'][:] == 0).all() and (nc['icellflag'][:] == 1).all()
    with _open_netcdf(turbulence) as nc:
        z = (np.arange(51)-.5)/49
        stress = 1.1+np.sin(2*np.pi*z)
        np.testing.assert_array_equal(nc['tzz'][0, :, 0, 0], stress.astype('f4'))
        np.testing.assert_array_equal(nc['CoEps'][0, :, 0, 0], (4*stress**1.5).astype('f4'))
    for step, path in configs.items():
        xml = ET.parse(path).getroot()
        config = parse_plume_config(xml)
        assert (config.duration_s, config.step_s, config.boxes) == (12, step, (5, 5, 20))
        assert config.boundaries == ('periodic',)*3
        assert xml.findtext('.//numPar') == '100000'


def test_sinusoidal_file_transport_and_snapshot_diagnostics(tmp_path):
    report = run_sinusoidal_case(tmp_path, steps=[4.], particles=96, seed=17)
    case = report['cases']['4']
    plume = case['plume']
    assert plume['particle_output_records'] == plume['models']['base']['records'] == 3
    assert plume['models']['base']['active_particles'] == 96
    with np.load(case['diagnostics']['arrays']) as arrays:
        assert arrays['observed'].shape == arrays['expected'].shape == (4, 20)
        np.testing.assert_allclose(arrays['relative_probability'].sum(), 25)
        np.testing.assert_array_equal(arrays['expected'][0], 0.)
