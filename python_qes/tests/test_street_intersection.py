import numpy as np
import pytest

from plumeswift.domain import Domain
from plumeswift.street_intersection import mark_street_intersections, smooth_street_intersections


def test_intersection_marker_requires_canyon_runs_on_both_axes():
    workspace = Domain((9, 9, 5), (1., 1., 1.)).workspace()
    flags = workspace.arrays['icellflag']
    flags[2, 2:7, 2:7] = 6
    flags[2, 3:6, 3:6] = 3
    flags[3, 2:7, 2:7] = 3
    flags[3, 2:7, 2] = flags[3, 2:7, 6] = 6
    mask = mark_street_intersections(workspace)
    expected = np.zeros_like(flags, dtype=bool)
    expected[2, 3:6, 3:6] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(flags == 12, expected)


def test_intersection_relaxation_updates_only_shared_faces():
    workspace = Domain((7, 7, 5), (1., 1., 1.)).workspace()
    arrays = workspace.arrays
    arrays['icellflag'][2, 3, 2:4] = 12
    u = arrays['u0']
    u[2, 3, 4] = 12.
    smooth_street_intersections(workspace, iterations=1)
    assert u[2, 3, 3] == 2.
    assert u[2, 3, 4] == 12.
    assert np.count_nonzero(u) == 2
    assert not np.any(arrays['v0']) and not np.any(arrays['w0'])
    for name in 'efghmn':
        arrays[name][2, 3, 3] = 0.
    with pytest.raises(ValueError, match='positive finite total aperture'):
        smooth_street_intersections(workspace)
    assert u[2, 3, 3] == 2.
