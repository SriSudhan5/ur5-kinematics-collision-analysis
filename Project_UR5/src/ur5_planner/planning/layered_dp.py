from __future__ import annotations
import numpy as np

from ur5_planner.planning.costs import edge_cost, is_edge_feasible_by_velocity


def solve_layered_shortest_path(
    layers: list[list[np.ndarray]],
    *,
    dlam: float,
    dt: float,
    qd_max: np.ndarray | None = None,
    cost_threshold: float | None = None,
    node_costs: list[np.ndarray] | None = None,
    w_node: float = 0.0,
) -> tuple[list[np.ndarray], list[int], float]:
    """
    Solve a shortest path problem on a layered graph.

    Each layer k contains candidate joint solutions q[k][i] (e.g., all IK solutions
    for pose k). We want one node per layer that minimizes sum of edge costs.

    Edge cost uses:
        c = ||wrapToPi(q_to - q_from)||^2 / dlam

    Feasibility gating (optional):
      - velocity limits using dt and qd_max
      - maximum cost threshold

    Returns
    -------
    chosen_q : list of np.ndarray
        One chosen joint vector per layer.
    chosen_idx : list[int]
        Index chosen at each layer.
    total_cost : float
        Total accumulated cost.
    """
    if len(layers) < 2:
        raise ValueError("Need at least 2 layers")
    if dlam <= 0:
        raise ValueError("dlam must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    K = len(layers)
    n_per = [len(L) for L in layers]
    if any(n == 0 for n in n_per):
        bad = [k for k, n in enumerate(n_per) if n == 0]
        raise ValueError(f"Empty IK layer(s): {bad}")

    if qd_max is not None:
        qd_max = np.asarray(qd_max, dtype=float).reshape(-1)

    if node_costs is not None:
        if len(node_costs) != K:
            raise ValueError("node_costs must have length K (one array per layer)")
        for k in range(K):
            c = np.asarray(node_costs[k], dtype=float).reshape(-1)
            if c.shape[0] != n_per[k]:
                raise ValueError(f"node_costs[{k}] length must match layer size")
        if w_node < 0:
            raise ValueError("w_node must be >= 0")

    # DP tables:
    # dp_cost[k][i] = best cost to reach node i in layer k
    # dp_prev[k][i] = predecessor index in layer k-1
    dp_cost = [np.full(n, np.inf) for n in n_per]
    dp_prev = [np.full(n, -1, dtype=int) for n in n_per]

    # Start anywhere in layer 0 with preferably zero cost 
    dp_cost[0][:] = 0.5     # lower cost means smoother and smaller joint changes

    def edge_ok(q_from: np.ndarray, q_to: np.ndarray) -> tuple[bool, float]:
        """Check feasibility + compute cost once."""
        c = edge_cost(q_from, q_to, dlam=dlam)

        if cost_threshold is not None and c > cost_threshold:
            return False, c

        if qd_max is not None:
            if not is_edge_feasible_by_velocity(q_from, q_to, dt=dt, qd_max=qd_max):
                return False, c

        return True, c

    # Forward DP (only k -> k+1 transitions)
    for k in range(K - 1):
        for i, q_from in enumerate(layers[k]):
            base = dp_cost[k][i]
            if not np.isfinite(base):
                continue

            for j, q_to in enumerate(layers[k + 1]):
                ok, c = edge_ok(q_from, q_to)
                if not ok:
                    continue

                node_c = 0.0
                if node_costs is not None and w_node > 150:   # make it prefer high manipubality (between 100 and 500)
                    node_c = float(node_costs[k + 1][j])
                new_cost = base + c + w_node * node_c
                if new_cost < dp_cost[k + 1][j]:
                    dp_cost[k + 1][j] = new_cost
                    dp_prev[k + 1][j] = i

    # Choose best end node
    end_layer = K - 1
    end_idx = int(np.argmin(dp_cost[end_layer]))
    total_cost = float(dp_cost[end_layer][end_idx])

    if not np.isfinite(total_cost):
        raise RuntimeError("No feasible path found: graph disconnected under constraints")

    # Backtrack
    chosen_idx = [-1] * K
    chosen_idx[end_layer] = end_idx

    k = end_layer
    while k > 0:
        p = int(dp_prev[k][chosen_idx[k]])
        if p == -1:
            raise RuntimeError("Backtracking failed: missing predecessor")
        chosen_idx[k - 1] = p
        k -= 1

    chosen_q = [layers[k][chosen_idx[k]] for k in range(K)]
    return chosen_q, chosen_idx, total_cost
