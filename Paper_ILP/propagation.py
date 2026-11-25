"""Stage 1: Constructing a disease-relevant subnetwork (network propagation + empirical P-value filtering)."""

import numpy as np
import scipy.sparse as sp


def normalize_weights(W):
    """Normalized weight function w: W' = D^{-1/2} W D^{-1/2}, where D_ii = sum_j W_ij"""
    d = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(d)
    inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
    D = sp.diags(inv_sqrt)
    return (D @ W @ D).tocsr()


def prior_vector(n, prior_idx):
    """Prior knowledge function Y: V -> R, Y(v) = 1 if v is a prior (disease-causing) protein, else 0"""
    Y = np.zeros(n)
    Y[prior_idx] = 1.0
    return Y


def propagate(W_norm, Y, alpha=0.5, tol=1e-6, max_iter=1000):
    """Network propagation: F(v) = alpha * sum_{u in N(v)} F(u) w(u,v) + (1 - alpha) * Y(v)"""
    F = Y.copy()
    for _ in range(max_iter):
        F_next = alpha * (W_norm @ F) + (1 - alpha) * Y
        if np.abs(F_next - F).max() < tol:
            return F_next
        F = F_next
    return F


def empirical_pvalues(W_norm, F, prior_size, alpha=0.5, n_random=1000, batch=100, seed=0):
    """Empirical P-value: p(v) = #{random priors S, |S| = |prior|, v not in S : F_S(v) >= F(v)} / #{random priors S : v not in S}"""
    n = W_norm.shape[0]
    rng = np.random.default_rng(seed)
    hits = np.zeros(n)
    valid = np.zeros(n)
    for start in range(0, n_random, batch):
        b = min(batch, n_random - start)
        Y = np.zeros((n, b))
        for col in range(b):
            Y[rng.choice(n, prior_size, replace=False), col] = 1.0
        F_rand = propagate(W_norm, Y, alpha)
        not_in_prior = Y == 0
        hits += ((F_rand >= F[:, None]) & not_in_prior).sum(axis=1)
        valid += not_in_prior.sum(axis=1)
    return hits / np.maximum(valid, 1)


def select_relevant(pvalues, threshold=0.01):
    """P-value thresholding: V_H = {v in V : p(v) < 0.01}"""
    return np.flatnonzero(pvalues < threshold)
