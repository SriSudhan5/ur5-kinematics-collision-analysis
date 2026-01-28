import numpy as np
from ur5_planner.common.angles import angle_diff

def edge_cost(q_from: np.ndarray, q_to: np.ndarray, dlam: float) -> float:
    """
    Paper-style edge cost (discrete version):

      c = ||wrapToPi(q_to - q_from)||^2 / dlam

    where dlam is the path parameter step (constant if uniformly sampled).

    Parameters
    ----------
    q_from, q_to : (6,) arrays
        Joint configurations in radians.
    dlam : float
        Positive step in path parameter.

    Returns
    -------
    float
        Edge cost.
    """
    if dlam <= 0:
        raise ValueError("dlam must be > 0")
    q_from = np.asarray(q_from, dtype=float).reshape(-1)
    q_to = np.asarray(q_to, dtype=float).reshape(-1)
    if q_from.shape != q_to.shape:
        raise ValueError("q_from and q_to must have same shape")

    dq = angle_diff(q_to, q_from)
    return float(dq @ dq) / float(dlam)

def is_edge_feasible_by_velocity(
    q_from: np.ndarray,
    q_to: np.ndarray,
    dt: float,
    qd_max: np.ndarray,
) -> bool:
    """
    Hard feasibility gate based on per-joint velocity limits.

    We approximate:
      qd ≈ wrapToPi(q_to - q_from) / dt

    and require:
      |qd_j| <= qd_max_j for all joints.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    qd_max : (6,) array
        Max allowed joint velocity (rad/s) for each joint.

    Returns
    -------
    bool
        True if edge respects velocity limits.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    q_from = np.asarray(q_from, dtype=float).reshape(-1)
    q_to = np.asarray(q_to, dtype=float).reshape(-1)
    qd_max = np.asarray(qd_max, dtype=float).reshape(-1)

    if q_from.shape != q_to.shape:
        raise ValueError("q_from and q_to must have same shape")
    if qd_max.shape != q_from.shape:
        raise ValueError("qd_max must match q shape")

    dq = angle_diff(q_to, q_from)
    qd = dq / float(dt)
    return bool(np.all(np.abs(qd) <= qd_max + 1e-12))
