import numpy as np
from ur5_planner.common.angles import angle_diff

def edge_cost(q_from: np.ndarray, q_to: np.ndarray, dlam: float) -> float:
    """
    c = ||wrapToPi(q_to - q_from)||^2 / dlam
    """
    if dlam <= 0:
        raise ValueError("dlam must be > 0")

    dq = angle_diff(q_to, q_from)
    return float(dq @ dq) / float(dlam)
