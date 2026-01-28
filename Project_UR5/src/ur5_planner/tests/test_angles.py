import numpy as np
from ur5_planner.common.angles import wrap_to_pi, angle_diff

def test_wrap_to_pi_basic():   # wrap_to_pi always returns values in the intended range
    x = np.array([0.0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi])
    y = wrap_to_pi(x)
    assert np.all(y < np.pi + 1e-12)
    assert np.all(y >= -np.pi - 1e-12)

def test_angle_diff_prefers_small(): # angle_diff returns the shortest circular difference
    a = np.array([2*np.pi - 0.1])
    b = np.array([0.0])
    d = angle_diff(a, b)
    assert np.allclose(d, np.array([-0.1]), atol=1e-9)
