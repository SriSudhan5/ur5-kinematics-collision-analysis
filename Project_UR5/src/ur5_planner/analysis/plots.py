import numpy as np
import matplotlib.pyplot as plt

def plot_joint_timeseries(t: np.ndarray, Q: np.ndarray, title: str, ylabel: str):
    t = np.asarray(t, dtype=float).reshape(-1)
    Q = np.asarray(Q, dtype=float)
    if Q.shape[0] != t.shape[0]:
        raise ValueError("Q and t length mismatch")

    plt.figure()
    for j in range(Q.shape[1]):
        plt.plot(t, Q[:, j], label=f"J{j+1}")
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
