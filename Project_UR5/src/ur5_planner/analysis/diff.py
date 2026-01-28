import numpy as np

def central_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute dx/dt using central differences for interior points and
    one-sided differences at the boundaries.

    x: (N, D) or (N,) array
    t: (N,) strictly increasing time stamps
    returns: same shape as x
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)

    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must have same length along axis 0")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")

    # ensure 2D for unified handling
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True

    N, D = x.shape
    dx = np.zeros((N, D), dtype=float)

    # forward difference at start
    dt0 = t[1] - t[0]
    dx[0] = (x[1] - x[0]) / dt0

    # central differences
    for k in range(1, N - 1):
        dt = t[k + 1] - t[k - 1]
        dx[k] = (x[k + 1] - x[k - 1]) / dt

    # backward difference at end
    dtn = t[-1] - t[-2]
    dx[-1] = (x[-1] - x[-2]) / dtn

    if squeeze:
        return dx[:, 0]
    return dx
