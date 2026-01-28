import os
import numpy as np

from eaik.IK_URDF import UrdfRobot

from ur5_planner.planning.layered_dp import solve_layered_shortest_path
from ur5_planner.analysis.diff import central_diff
from ur5_planner.analysis.plots import plot_joint_timeseries
from ur5_planner.analysis.manipulability import yoshikawa_manipulability


def approx_Jpos_fd(bot: UrdfRobot, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Finite-difference translational Jacobian Jpos (3x6) using FK position.
    """
    q = np.asarray(q, dtype=float).reshape(6,)
    T0 = bot.fwdKin(q)
    p0 = np.asarray(T0[:3, 3], dtype=float)

    J = np.zeros((3, 6), dtype=float)
    for j in range(6):
        dq = np.zeros(6, dtype=float)
        dq[j] = eps
        Tp = bot.fwdKin(q + dq)
        pp = np.asarray(Tp[:3, 3], dtype=float)
        J[:, j] = (pp - p0) / eps
    return J


def load_xyz_path(csv_path: str):
    """
    Read t,x,y,z from CSV with header.
    Returns:
        t   : (K,)
        xyz : (K,3)
    CSV header must include: t,x,y,z
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    x = np.asarray(data["x"], dtype=float).reshape(-1)
    y = np.asarray(data["y"], dtype=float).reshape(-1)
    z = np.asarray(data["z"], dtype=float).reshape(-1)
    xyz = np.stack([x, y, z], axis=1)
    return t, xyz


def layers_from_xyz_fixed_orientation(
    bot: UrdfRobot,
    t: np.ndarray,
    xyz: np.ndarray,
    q_seed: np.ndarray,
):
    """
    Build poses using XYZ from file and orientation fixed from q_seed FK.
    Then compute IK layers with EAIK.

    Returns:
        Ts     : list of 4x4 poses
        layers : list (len K) of list of (6,) joint solutions
    """
    q_seed = np.asarray(q_seed, dtype=float).reshape(6,)
    Tseed = bot.fwdKin(q_seed)
    R = np.asarray(Tseed[:3, :3], dtype=float)

    layers = []
    Ts = []
    for k in range(len(t)):
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = xyz[k]
        Ts.append(T)

        sol = bot.IK(T)
        Qs = [np.asarray(q, dtype=float).reshape(6,) for q in sol.Q]
        layers.append(Qs)

    return Ts, layers


def main():
    # --- URDF path for EAIK ---
    urdf_path = os.environ.get("UR5_URDF_PATH", "")
    if not urdf_path or not os.path.isfile(urdf_path):
        raise RuntimeError(f"UR5_URDF_PATH must point to a URDF file. Got: {urdf_path}")

    bot = UrdfRobot(urdf_path)

    # Seed pose defines fixed tool orientation (welding-style)
    q_seed = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0], dtype=float)

    # ===== INPUT (workspace coordinates) =====
    csv_in = "path_xyz.csv"
    if not os.path.isfile(csv_in):
        raise RuntimeError(f"Missing {csv_in}. Create it first.")
    t, xyz = load_xyz_path(csv_in)

    K = len(t)
    if K < 2:
        raise RuntimeError("Need at least 2 samples in path")

    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, atol=1e-9):
        raise RuntimeError("t must be uniformly sampled for this pipeline")

    dlam = 1.0 / (K - 1)

    # Build task-space poses with fixed orientation, then IK layers
    Ts, layers = layers_from_xyz_fixed_orientation(bot, t, xyz, q_seed)

    print("layer sizes (first 10):", [len(L) for L in layers[:10]])
    if any(len(L) == 0 for L in layers):
        bad = [i for i, L in enumerate(layers) if len(L) == 0]
        raise RuntimeError(f"Some points had no IK solutions at indices: {bad}. Adjust XYZ or orientation.")

    # --- manipulability node costs (minimize => use negative manipulability) ---
    node_costs = []
    for L in layers:
        costs_k = []
        for q in L:
            Jpos = approx_Jpos_fd(bot, q, eps=1e-5)
            w = yoshikawa_manipulability(Jpos)
            costs_k.append(-w)  # minimize -w == maximize w
        node_costs.append(np.asarray(costs_k, dtype=float))

    # --- Constraints ---
    qd_max = np.ones(6) * 4.0  # rad/s

    # --- Node cost weight ---
    # For verification: run once with 0.0, then with 500.0 and compare chosen_idx and manipulability.
    w_node = 500.0

    chosen_q, chosen_idx, total_cost = solve_layered_shortest_path(
        layers,
        dlam=dlam,
        dt=dt,
        qd_max=qd_max,
        cost_threshold=None,
        node_costs=node_costs,
        w_node=w_node,
    )

    print("w_node =", w_node)
    print("chosen_idx[:10] =", chosen_idx[:10])

    chosen_manip = [-float(node_costs[k][chosen_idx[k]]) for k in range(K)]
    print("Manipulability (mean/min):", float(np.mean(chosen_manip)), float(np.min(chosen_manip)))

    # Convert list -> matrix for analysis
    Q = np.vstack(chosen_q)  # (K,6)
    Qd = central_diff(Q, t)
    Qdd = central_diff(Qd, t)

    print("Total cost:", float(total_cost))
    print("Max |qd| per joint:", np.max(np.abs(Qd), axis=0))
    print("Max |qdd| per joint:", np.max(np.abs(Qdd), axis=0))

    # Save for ROS playback
    np.savez("trajectory_eaik.npz", t=t, Q=Q)
    print("Wrote: trajectory_eaik.npz")

    # Plots
    plot_joint_timeseries(t, Q, "Joint positions", "rad")
    plot_joint_timeseries(t, Qd, "Joint velocities", "rad/s")
    plot_joint_timeseries(t, Qdd, "Joint accelerations", "rad/s^2")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()

