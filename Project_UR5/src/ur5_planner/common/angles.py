import numpy as np

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """
    Wrap each element of x into [-pi, pi).

    Works for scalars or arrays.
    """
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimal signed angle difference a - b wrapped to [-pi, pi).
    """
    return wrap_to_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
