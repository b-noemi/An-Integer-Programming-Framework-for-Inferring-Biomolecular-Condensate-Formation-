"""Stage 3: Detection of protein complexes (integer linear program for the highest scoring cluster)."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


def variable_layout(n, k, m, M):
    """ILP variables: v_i for i in V, e_ij for (i,j) in E, and g_c, s_c, a_c for c in [m, M]"""
    C = M - m + 1
    return {
        "v": np.arange(0, n),
        "e": np.arange(n, n + k),
        "g": np.arange(n + k, n + k + C),
        "s": np.arange(n + k + C, n + k + 2 * C),
        "a": np.arange(n + k + 2 * C, n + k + 3 * C),
        "c": np.arange(m, M + 1),
        "size": n + k + 3 * C,
    }


def objective(layout, L1, gamma):
    """Objective: max sum_{(i,j) in E} L1_ij e_ij + gamma * (C(R,2) - T), with C(R,2) = (sq - R) / 2, R = sum_i v_i (1), T = sum_ij e_ij (2)"""
    obj = np.zeros(layout["size"])
    obj[layout["v"]] = -gamma / 2.0
    obj[layout["e"]] = np.asarray(L1) - gamma
    obj[layout["a"]] = gamma * layout["c"] ** 2 / 2.0
    return obj


def edge_indicator_constraints(layout, edges):
    """Constraint (3): v_i + v_j - 1 <= e_ij <= v_i, v_j for all (i,j) in E"""
    v, e = layout["v"], layout["e"]
    k = len(edges)
    q = np.arange(k)
    rows = np.concatenate([q, q, k + q, k + q, 2 * k + q, 2 * k + q, 2 * k + q])
    cols = np.concatenate([e, v[edges[:, 0]], e, v[edges[:, 1]], v[edges[:, 0]], v[edges[:, 1]], e])
    vals = np.concatenate([np.ones(k), -np.ones(k), np.ones(k), -np.ones(k), np.ones(k), np.ones(k), -np.ones(k)])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(3 * k, layout["size"]))
    lb = np.full(3 * k, -np.inf)
    ub = np.concatenate([np.zeros(2 * k), np.ones(k)])
    return A, lb, ub


def connectivity_constraints(layout, edges):
    """Constraint (4): sum_{j in N(i)} e_ij >= (R - 1)/2 - (1 - v_i) |V| for all i in V"""
    v, e = layout["v"], layout["e"]
    n, k = len(v), len(edges)
    q = np.arange(k)
    rows = [edges[:, 0], edges[:, 1], np.repeat(np.arange(n), n), np.arange(n)]
    cols = [e[q], e[q], np.tile(v, n), v]
    vals = [np.ones(k), np.ones(k), np.full(n * n, -0.5), np.full(n, -float(n))]
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, layout["size"]))
    lb = np.full(n, -float(n) - 0.5)
    ub = np.full(n, np.inf)
    return A, lb, ub


def size_constraints(layout, m, M):
    """Cluster size range: m <= R <= M, with R = sum_i v_i"""
    A = sp.coo_matrix((np.ones(len(layout["v"])), (np.zeros(len(layout["v"]), dtype=int), layout["v"])), shape=(1, layout["size"]))
    return A, np.array([m]), np.array([M])


def square_linearization_constraints(layout, M):
    """Constraints (5)-(7): (R-c+1)/(cM) <= g_c <= R/c, (-R+c+1)/c <= s_c <= (M+c-R)/M, s_c + g_c - 1 <= a_c <= s_c, g_c for all c in [m, M]; (8): sq = sum_{m<=c<=M} c^2 a_c"""
    v, g, s, a, cs = layout["v"], layout["g"], layout["s"], layout["a"], layout["c"]
    blocks, ubs = [], []
    for idx, c in enumerate(cs):
        A = np.zeros((7, layout["size"]))
        A[0, v], A[0, g[idx]] = 1.0, -c * M
        A[1, g[idx]], A[1, v] = c, -1.0
        A[2, v], A[2, s[idx]] = -1.0, -c
        A[3, v], A[3, s[idx]] = 1.0, M
        A[4, s[idx]], A[4, g[idx]], A[4, a[idx]] = 1.0, 1.0, -1.0
        A[5, a[idx]], A[5, s[idx]] = 1.0, -1.0
        A[6, a[idx]], A[6, g[idx]] = 1.0, -1.0
        blocks.append(A)
        ubs.append([c - 1, 0, -c - 1, M + c, 1, 0, 0])
    ub = np.concatenate(ubs).astype(float)
    return sp.csr_matrix(np.vstack(blocks)), np.full(len(ub), -np.inf), ub


def find_top_cluster(n, edges, L1, gamma, m=4, M=20, time_limit=300, mip_gap=0.01):
    """Highest scoring cluster: argmax_C sum_{(i,j) in E[C]} L1_ij + gamma * (C(R,2) - T) subject to (1)-(8)"""
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if n < m or len(edges) == 0:
        return None, None
    layout = variable_layout(n, len(edges), m, M)
    blocks = [
        edge_indicator_constraints(layout, edges),
        connectivity_constraints(layout, edges),
        size_constraints(layout, m, M),
        square_linearization_constraints(layout, M),
    ]
    A = sp.vstack([b[0] for b in blocks]).tocsr()
    lb = np.concatenate([b[1] for b in blocks])
    ub = np.concatenate([b[2] for b in blocks])
    res = milp(
        c=-objective(layout, L1, gamma),
        constraints=LinearConstraint(A, lb, ub),
        integrality=np.ones(layout["size"]),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": False},
    )
    if res.x is None:
        return None, None
    return np.flatnonzero(np.round(res.x[layout["v"]]) == 1), -res.fun


def find_clusters(nodes, edges, L1, gamma, n_clusters=10, m=4, M=20, time_limit=300, mip_gap=0.01):
    """Iterative cluster detection: C_k = highest scoring cluster in H minus (V[C_1] u ... u V[C_{k-1}]), k = 1..10"""
    nodes = np.asarray(nodes)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    L1 = np.asarray(L1, dtype=float)
    remaining = np.ones(len(nodes), dtype=bool)
    clusters = []
    for _ in range(n_clusters):
        keep = np.flatnonzero(remaining)
        local = -np.ones(len(nodes), dtype=int)
        local[keep] = np.arange(len(keep))
        mask = remaining[edges[:, 0]] & remaining[edges[:, 1]]
        sub_edges = local[edges[mask]]
        chosen, score = find_top_cluster(len(keep), sub_edges, L1[mask], gamma, m, M, time_limit, mip_gap)
        if chosen is None:
            break
        clusters.append((nodes[keep[chosen]], score))
        remaining[keep[chosen]] = False
    return clusters
