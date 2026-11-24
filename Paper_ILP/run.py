"""Paper ILP: reimplementation of Mazza et al. 2016, "An integer programming framework for inferring disease complexes from network data"
(Bioinformatics 32:i271). Pipeline: data processing -> network propagation -> likelihood scoring -> ILP cluster detection.

Files:
    run.py                              data loading + full pipeline (run_framework) + command line entry
    propagation.py                      stage 1: disease-relevant subnetwork (network propagation + empirical P-values)
    scoring.py                          stage 2: likelihood scores (protein complex model vs. degree-preserving random model)
    ilp.py                              stage 3: detection of protein complexes (integer linear program)
    data/hippie_medium.txt              HIPPIE PPI network used in the paper (medium/high confidence, 148,441 lines)
    data/priors/long_QT_syndrome.txt    prior (known disease) genes for long QT syndrome, the test case

Requirements: Python 3.9+, numpy, scipy >= 1.9 (scipy.optimize.milp uses the open-source HiGHS solver).

Command line, from inside this directory:
    python run.py NETWORK_FILE PRIOR_FILE [options]

Quick test (~20 s):
    python run.py data/hippie_medium.txt data/priors/long_QT_syndrome.txt --n-random 100 --n-clusters 2 --time-limit 60
    expected clusters:
        1  CACNA1C*,CALM1*,CALM2*,CALM3*,KCNQ2,KCNQ3,KCNQ5,KCNRG,TRIM13
        2  ADRA1D,DGKZ,DTNA,SCN5A*,SNTA1*,SNTB1,SNTB2

Full paper settings (~2 min):
    python run.py data/hippie_medium.txt data/priors/long_QT_syndrome.txt --out long_qt.tsv

Input files:
    NETWORK_FILE   tab-separated lines "geneA <tab> confidence <tab> geneB" (format of data/hippie_medium.txt)
    PRIOR_FILE     one gene name per line, same naming as the network (format of data/priors/long_QT_syndrome.txt)

Options (paper defaults): --alpha 0.5, --pvalue 0.01, --n-random 1000, --beta 0.9, --min-size 4, --max-size 20,
    --n-clusters 10, --max-degree 500, --time-limit 300 (seconds per ILP), --mip-gap 0.01, --seed 0, --out FILE

Output: TSV with one row per cluster (rank, size, edges, density, ILP objective, exact L(C), genes with prior genes marked *),
    printed to stdout and optionally written to --out.

From Python, with data loaded any way you like:
    from run import run_framework, format_clusters
    clusters = run_framework(genes, edges, weights, prior_genes, n_random=200)
    print(format_clusters(clusters))
    genes: list[str]; edges: int array (k, 2) of indices into genes; weights: float array (k,); prior_genes: list[str]

To swap a stage, replace propagation.py, scoring.py or ilp.py with a module exposing the same function names and signatures.
To use a different data format, write loaders that return the same structures as load_network and load_priors.
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp

import ilp
import propagation
import scoring


def load_network(path):
    """PPI network: G = (V, E, w), w(i,j) = interaction confidence

    Returns (genes, edges, weights):
        genes: list[str] of length n, gene name at each index 0..n-1
        edges: np.ndarray of int, shape (k, 2), each undirected edge (i, j) listed once, no self-loops, indices into genes
        weights: np.ndarray of float, shape (k,), w(i,j) for each row of edges
    """
    best = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[0] == parts[2]:
            continue
        a, w, b = parts[0], float(parts[1]), parts[2]
        key = (a, b) if a < b else (b, a)
        best[key] = max(w, best.get(key, 0.0))
    genes = sorted({g for key in best for g in key})
    index = {g: i for i, g in enumerate(genes)}
    edges = np.array([(index[a], index[b]) for a, b in best], dtype=int)
    weights = np.array(list(best.values()), dtype=float)
    return genes, edges, weights


def load_priors(path):
    """Gene-disease association retrieval: prior set P = {genes known to be associated with the disease}

    Returns prior_genes: list[str], prior gene names (same naming as the network's genes)
    """
    return [line.strip() for line in open(path) if line.strip()]


def weight_matrix(n, edges, weights):
    """Weighted adjacency of G: W_ij = W_ji = w(i,j) for (i,j) in E, else 0"""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.coo_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n, n)).tocsr()


def degrees(n, edges):
    """Node degree in G: d_v = |N(v)|, t = |E|"""
    return np.bincount(edges.ravel(), minlength=n), len(edges)


def disease_subnetwork(relevant, edges, degree, max_degree=500):
    """Disease-relevant subnetwork: H = G[V_H \\ {v : d_v > 500}]"""
    keep = np.zeros(len(degree), dtype=bool)
    keep[relevant] = True
    keep &= degree <= max_degree
    mask = keep[edges[:, 0]] & keep[edges[:, 1]]
    return np.flatnonzero(keep), edges[mask]


def run_framework(genes, edges, weights, prior_genes, alpha=0.5, pvalue=0.01, n_random=1000, beta=0.9,
                  min_size=4, max_size=20, n_clusters=10, max_degree=500, time_limit=300, mip_gap=0.01, seed=0, verbose=True):
    """Framework: (a) PPI + prior -> (b) propagation and P-value thresholding -> (c) likelihood scores -> (d) ILP cluster detection

    Takes the outputs of load_network (genes, edges, weights) and load_priors (prior_genes) plus the paper's parameters.
    Returns list[dict], one per cluster in rank order, with keys:
        rank (int), size (int, R), edges (int, T), density (float, T / C(R,2)), ilp_objective (float), likelihood (float, exact L(C)),
        genes (list[str]), in_prior (list[bool], aligned with genes)
    """
    log = print if verbose else (lambda *a, **k: None)
    n = len(genes)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float)
    index = {g: i for i, g in enumerate(genes)}
    prior_idx = np.array(sorted({index[g] for g in prior_genes if g in index}), dtype=int)
    log(f"network: {n} genes, {len(edges)} interactions; prior genes in network: {len(prior_idx)} / {len(set(prior_genes))}")

    t0 = time.time()
    W_norm = propagation.normalize_weights(weight_matrix(n, edges, weights))
    F = propagation.propagate(W_norm, propagation.prior_vector(n, prior_idx), alpha)
    p = propagation.empirical_pvalues(W_norm, F, len(prior_idx), alpha, n_random, seed=seed)
    relevant = propagation.select_relevant(p, pvalue)
    log(f"[stage 1] {len(relevant)} genes with p < {pvalue} ({time.time() - t0:.1f}s)")

    degree, t = degrees(n, edges)
    nodes, sub_edges = disease_subnetwork(relevant, edges, degree, max_degree)
    local = -np.ones(n, dtype=int)
    local[nodes] = np.arange(len(nodes))
    p_ij = scoring.random_edge_probability(degree[sub_edges[:, 0]], degree[sub_edges[:, 1]], t)
    L1 = scoring.edge_likelihood(p_ij, beta)
    gamma = scoring.constant_non_edge_penalty(beta)
    log(f"[stage 2] subnetwork H: {len(nodes)} nodes, {len(sub_edges)} edges, gamma = {gamma:.3f}")

    t0 = time.time()
    found = ilp.find_clusters(nodes, local[sub_edges], L1, gamma, n_clusters, min_size, max_size, time_limit, mip_gap)
    log(f"[stage 3] {len(found)} clusters ({time.time() - t0:.1f}s)")

    adjacency = weight_matrix(n, edges, np.ones(len(edges))).astype(bool)
    prior_set = set(prior_idx.tolist())
    clusters = []
    for rank, (members, objective) in enumerate(found, 1):
        R = len(members)
        T = int(adjacency[members][:, members].sum() // 2)
        clusters.append({
            "rank": rank,
            "size": R,
            "edges": T,
            "density": T / (R * (R - 1) / 2),
            "ilp_objective": float(objective),
            "likelihood": scoring.cluster_likelihood(members, adjacency, degree, t, beta),
            "genes": [genes[i] for i in members],
            "in_prior": [int(i) in prior_set for i in members],
        })
    return clusters


def format_clusters(clusters):
    """Cluster report: TSV with rank, R, T, density, ILP objective, L(C), genes (prior genes marked *)"""
    lines = ["rank\tsize\tedges\tdensity\tilp_objective\tL(C)\tgenes"]
    for c in clusters:
        names = ",".join(g + ("*" if p else "") for g, p in zip(c["genes"], c["in_prior"]))
        lines.append(f"{c['rank']}\t{c['size']}\t{c['edges']}\t{c['density']:.2f}\t{c['ilp_objective']:.2f}\t{c['likelihood']:.2f}\t{names}")
    return "\n".join(lines)


def main():
    """Command line entry: load NETWORK_FILE and PRIOR_FILE, run the framework, report the clusters"""
    parser = argparse.ArgumentParser(description="ILP framework for inferring disease complexes (Mazza et al. 2016)")
    parser.add_argument("network_file")
    parser.add_argument("prior_file")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pvalue", type=float, default=0.01)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--max-degree", type=int, default=500)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = vars(parser.parse_args())

    network_file, prior_file, out = args.pop("network_file"), args.pop("prior_file"), args.pop("out")
    genes, edges, weights = load_network(network_file)
    prior_genes = load_priors(prior_file)
    report = format_clusters(run_framework(genes, edges, weights, prior_genes, **args))
    print(report)
    if out:
        with open(out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
