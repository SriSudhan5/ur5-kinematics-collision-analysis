import numpy as np
from ur5_planner.planning.costs import edge_cost

def test_edge_cost_sanity(): # no movement → zero cost
    q1 = np.zeros(6)
    q2 = np.zeros(6)
    assert edge_cost(q1, q2, dlam=0.1) == 0.0

def test_edge_cost_wrap():  # edge_cost correctly uses wrapped differences, and normalization behaves as expected
    q1 = np.zeros(6)
    q2 = np.array([2*np.pi - 0.1, 0,0,0,0,0], dtype=float)
    c = edge_cost(q1, q2, dlam=0.01)
    assert abs(c - 1.0) < 1e-6
