"""Stage 2: Likelihood scores (protein complex model vs. degree-preserving random model)."""

import numpy as np


def random_edge_probability(d_i, d_j, t):
    """Random model interaction probability: p_ij = d_i d_j / 2t"""
    return np.asarray(d_i, dtype=float) * np.asarray(d_j, dtype=float) / (2.0 * t)


def matrix_edge_probability(u_names, v_names, score_genes, score_values, return_source=False, eps=1e-6):
    """Condensate random model: p_ij = S_ij, the IDR relative interaction score (NEW_log_uniform_relative_scores.csv), for protein names u_i, v_i

    Interim fill for pairs missing from S (no annotated IDR), done in log space because S is log-uniform on [0.01, 1]:
        one endpoint j scored  -> p_ij = exp(median_k log S_jk), the scored partner's geometric row median
        neither endpoint scored (or S_ij not finite) -> p_ij = exp(median_{k<l} log S_kl), the global geometric median (~0.10)
    p is clipped to [eps, 1 - eps] so that L1 = log(beta / p) and L0 = log((1 - beta) / (1 - p)) stay finite.
    With return_source=True also returns source: int array, 2 = both scored, 1 = row-median fill, 0 = global fill.
    """
    index = {g: k for k, g in enumerate(score_genes)}
    logS = np.log(np.clip(np.asarray(score_values, dtype=float), eps, 1.0))
    np.fill_diagonal(logS, np.nan)
    row_fill = np.exp(np.nanmedian(logS, axis=1))
    global_fill = float(np.exp(np.nanmedian(logS[np.triu_indices_from(logS, k=1)])))
    p = np.empty(len(u_names))
    source = np.empty(len(u_names), dtype=int)
    for q, (u, v) in enumerate(zip(u_names, v_names)):
        a, b = index.get(u), index.get(v)
        if a is not None and b is not None and np.isfinite(logS[a, b]):
            p[q], source[q] = np.exp(logS[a, b]), 2
        elif a is not None and b is None:
            p[q], source[q] = row_fill[a], 1
        elif b is not None and a is None:
            p[q], source[q] = row_fill[b], 1
        else:
            p[q], source[q] = global_fill, 0
    p = np.clip(p, eps, 1 - eps)
    return (p, source) if return_source else p


def edge_likelihood(p, beta=0.9):
    """Likelihood score of an interaction in C: L1_ij = log(beta / p_ij)"""
    return np.log(beta / np.asarray(p, dtype=float))


def non_edge_likelihood(p, beta=0.9):
    """Likelihood score of a non-interaction in C: L0_ij = log((1 - beta) / (1 - p_ij))"""
    return np.log((1 - beta) / (1 - np.asarray(p, dtype=float)))


def constant_non_edge_penalty(beta=0.9):
    """Constant missing-edge penalty: gamma = L0_ij ~ log(1 - beta) (= -2.3 for beta = 0.9)"""
    return float(np.log(1 - beta))


def cluster_likelihood(nodes, adjacency, degree, t, beta=0.9, pair_probability=None):
    """Likelihood score of C: L(C) = sum_{(i,j) in E[C]} L1_ij + sum_{i,j in V[C], (i,j) not in E} L0_ij

    pair_probability: optional callable (i, j) -> p_ij; defaults to the degree-preserving random model d_i d_j / 2t.
    """
    nodes = list(nodes)
    total = 0.0
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            i, j = nodes[a], nodes[b]
            p = pair_probability(i, j) if pair_probability else random_edge_probability(degree[i], degree[j], t)
            total += edge_likelihood(p, beta) if adjacency[i, j] else non_edge_likelihood(p, beta)
    return float(total)
