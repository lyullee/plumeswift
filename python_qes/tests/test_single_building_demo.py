from pathlib import Path

import numpy as np
import pytest

from plumeswift.single_building_demo import run_single_building


ROOT = Path(__file__).resolve().parents[2]


def test_single_building_cpu_diagnostic_writes_both_native_tables(tmp_path):
    report = run_single_building(tmp_path)
    assert report['iterations'] == 360
    assert report['mean_update'] == pytest.approx(9.90107e-10, rel=1e-6)
    assert report['reduced_error'] == pytest.approx(8.39279e-5, rel=1e-6)
    corrected = ROOT/'results/python_port/single_building_demo'
    shipped = ROOT/'external/QES-Public/scratch/solverTestCases/Single Building/CPU'
    for field, rows in [('face_table', 41*41*42), ('cell_table', 40*40*41)]:
        path = Path(report[field])
        actual = np.loadtxt(path)
        assert actual.shape == (rows, 9)
        assert np.any(actual[:, 7:] != 0)
        if (corrected/path.name).is_file():
            np.testing.assert_array_equal(actual, np.loadtxt(corrected/path.name))
        elif (shipped/path.name).is_file():
            # The bundled output has an out-of-bounds velocity write on the
            # highest lateral edge; compare its defined interior values.
            interior = (actual[:, 0] < 39) & (actual[:, 1] < 39) & (actual[:, 2] < 40)
            np.testing.assert_allclose(actual[interior], np.loadtxt(shipped/path.name)[interior],
                                       atol=2e-5, rtol=0)


def test_single_building_cuda_diagnostic_matches_corrected_native(tmp_path):
    cp = pytest.importorskip('cupy')
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip('CUDA hardware is required for this test')
    try:
        cp.asarray([1.]).copy()
    except cp.cuda.compiler.CompileException as exc:
        if 'cannot open source file "cupy/complex.cuh"' not in str(exc):
            raise
        pytest.skip('CuPy headers are unavailable to NVRTC from this installation path')

    report = run_single_building(tmp_path, backend='cuda')
    assert report['backend'] == 'cuda'
    assert report['iterations'] == 158
    assert report['mean_update'] == pytest.approx(8.91806355e-10, rel=1e-6)
    corrected = ROOT/'results/python_port/single_building_cuda'
    for field, rows in [('face_table', 41*41*42), ('cell_table', 40*40*41)]:
        path = Path(report[field])
        actual = np.loadtxt(path)
        assert actual.shape == (rows, 9)
        assert np.isfinite(actual).all()
        assert np.any(actual[:, 7:] != 0)
        if (corrected/path.name).is_file():
            np.testing.assert_array_equal(actual, np.loadtxt(corrected/path.name))
