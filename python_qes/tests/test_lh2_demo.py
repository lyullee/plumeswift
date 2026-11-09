import json
import subprocess
import sys

import numpy as np
import pytest


def test_packaged_lh2_example_runs_and_exports_3d_results(tmp_path):
    pytest.importorskip('plotly')
    pytest.importorskip('skimage')
    command = [sys.executable, '-m', 'plumeswift', 'lh2-demo']
    subprocess.run([*command, '--write-input', 'example.json'], cwd=tmp_path, check=True)
    config = json.loads((tmp_path / 'example.json').read_text(encoding='utf-8'))
    result = subprocess.run([*command, '--input', 'example.json', '--output', 'result'],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    assert summary['averaged_s'] == pytest.approx(config['duration_s'] - config['average_from_s'])
    assert summary['accounting']['released_mass_kg'] == pytest.approx(
        config['release_rate_kg_s'] * config['duration_s'])
    assert abs(summary['accounting']['mass_balance_error_kg']) < 1e-6
    assert summary['rendered_isosurfaces_volpct'] == config['isosurfaces_volpct']
    assert json.loads((tmp_path / 'result/case.json').read_text(encoding='utf-8')) == config
    with np.load(tmp_path / 'result/concentration.npz') as data:
        field, solid = data['h2_mean_volpct'], data['solid']
        assert field.shape == tuple(reversed(config['grid_cells']))
        assert solid.any() and np.array_equal(np.isnan(field), solid)
        assert np.nanmax(field) == pytest.approx(summary['peak_mean_volpct'])
    html = (tmp_path / 'result/plume.html').read_text(encoding='utf-8')
    assert 'plotly.js v' in html
    plot_args = html.rsplit('Plotly.newPlot(', 1)[1]
    traces, _ = json.JSONDecoder().raw_decode(plot_args.split(',', 1)[1].lstrip())
    meshes = [trace for trace in traces if trace['type'] == 'mesh3d']
    assert [mesh['name'] for mesh in meshes] == ['0.5 vol%', '2 vol%', '4 vol%', 'Obstacle']
