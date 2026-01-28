import numpy as np

def yoshikawa_manipulability(J: np.ndarray) -> float:
    """
    Yoshikawa manipulability: sqrt(det(J J^T)).
    Works for 3x6 or 6x6 Jacobians.
    """
    JJ = J @ J.T
    val = float(np.linalg.det(JJ))
    return float(np.sqrt(max(val, 0.0)))
