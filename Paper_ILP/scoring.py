"""Stage 2: Likelihood scores (protein complex model vs. degree-preserving random model)."""

import numpy as np


def random_edge_probability(d_i, d_j, t):
    """Random model interaction probability: p_ij = d_i d_j / 2t"""
    return np.asarray(d_i, dtype=float) * np.asarray(d_j, dtype=float) / (2.0 * t)


def edge_likelihood(p, beta=0.9):
    """Likelihood score of an interaction in C: L1_ij = log(beta / p_ij)"""
    return np.log(beta / np.asarray(p, dtype=float))


def non_edge_likelihood(p, beta=0.9):
    """Likelihood score of a non-interaction in C: L0_ij = log((1 - beta) / (1 - p_ij))"""
    return np.log((1 - beta) / (1 - np.asarray(p, dtype=float)))


def constant_non_edge_penalty(beta=0.9):
    """Constant missing-edge penalty: gamma = L0_ij ~ log(1 - beta) (= -2.3 for beta = 0.9)"""
    return float(np.log(1 - beta))


def cluster_likelihood(nodes, adjacency, degree, t, beta=0.9):
    """Likelihood score of C: L(C) = sum_{(i,j) in E[C]} L1_ij + sum_{i,j in V[C], (i,j) not in E} L0_ij"""
    nodes = list(nodes)
    total = 0.0
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            i, j = nodes[a], nodes[b]
            p = random_edge_probability(degree[i], degree[j], t)
            total += edge_likelihood(p, beta) if adjacency[i, j] else non_edge_likelihood(p, beta)
    return float(total)
