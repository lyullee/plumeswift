import pytest

from plumeswift.input_paths import QESInputPaths


def test_input_paths_use_independent_config_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('first', 'second'):
        (tmp_path/name/'assets').mkdir(parents=True)
    first = QESInputPaths.from_input('first/winds.xml')
    second = QESInputPaths.from_input('second/winds.xml')
    assert second.resolve('assets/terrain.tif') == tmp_path/'second/assets/terrain.tif'
    assert first.resolve('assets/../assets/terrain.tif') == tmp_path/'first/assets/terrain.tif'
    unchanged = tmp_path/'first/../first/assets/terrain.tif'
    assert first.resolve(unchanged) == unchanged
    with pytest.raises(FileNotFoundError):
        first.resolve('missing/grid.nc')
    assert first.resolve('missing/grid.nc', require_parent=False) == tmp_path/'first/missing/grid.nc'
