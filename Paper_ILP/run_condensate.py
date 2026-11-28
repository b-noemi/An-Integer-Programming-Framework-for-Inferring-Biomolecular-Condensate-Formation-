"""Condensate run: the Mazza et al. 2016 pipeline (run.run_framework) on the team's co-phase-separation graph.

Inputs (defaults are the team's files, relative to the repo root):
    data/graph/edges.tsv                  co-phase-separation pairs from make_graph.py; only label == 1 pairs become edges,
                                          label == 0 (tested, did not phase separate) pairs are treated like untested pairs
    data/priors.txt                       default prior: both seed proteins; per-condensate priors (one run each) are
                                          data/priors/G3BP1.txt (Q13283) and data/priors/FUS.txt (P35637)
    NEW_log_uniform_relative_scores.csv   IDR relative interaction scores, used as p_ij (pij_method="matrix")
    annotated_idr_features.csv            only used to print protein abbreviations next to UniProt IDs

Command line, from inside Paper_ILP/:
    python run_condensate.py                        matrix p_ij, normalized propagation, paper defaults otherwise
    python run_condensate.py --pij paper            degree-preserving p_ij, for comparison
    python run_condensate.py --no-normalize --alpha 0.15
                                                    raw-weight propagation (needs alpha * rho(W) < 1; rho ~ 5.36 here)
All options of run.py are accepted (--alpha, --pvalue, --n-random, --beta, --min-size, --max-size, --n-clusters, ...).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from run import TERMS, format_clusters, load_priors, load_score_matrix, run_framework

REPO = Path(__file__).resolve().parent.parent
EDGES = REPO / "data" / "graph" / "edges.tsv"
PRIORS = REPO / "data" / "priors.txt"
SCORES = REPO / "NEW_log_uniform_relative_scores.csv"
FEATURES = REPO / "annotated_idr_features.csv"


def load_condensate_graph(path, weight_column=None):
    """Co-phase-separation graph: G = (V, E, w), E = {(u, v) : label(u, v) = 1}, w = 1 (or the given column, e.g. n_yes)

    Returns (genes, edges, weights) in the same format as run.load_network. Proteins with no label-1 pair are not in V.
    """
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["label"] == "1" and row["u"] != row["v"]:
                w = float(row[weight_column]) if weight_column else 1.0
                pairs.append((min(row["u"], row["v"]), max(row["u"], row["v"]), w))
    best = {}
    for a, b, w in pairs:
        best[(a, b)] = max(w, best.get((a, b), 0.0))
    genes = sorted({g for key in best for g in key})
    index = {g: i for i, g in enumerate(genes)}
    edges = np.array([(index[a], index[b]) for a, b in best], dtype=int)
    weights = np.array(list(best.values()), dtype=float)
    return genes, edges, weights


def load_abbreviations(path):
    """UniProt ID -> abbreviation, for display only"""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return {r["uniprot_id"]: r["abbreviation"] for r in csv.DictReader(f)}


def label_genes(clusters, names):
    """Report helper: rename each gene to 'UNIPROT(abbr)' when an abbreviation is known"""
    for c in clusters:
        c["genes"] = [f"{g}({names[g]})" if g in names else g for g in c["genes"]]
    return clusters


def main():
    """Command line entry: load the team's graph, priors and score matrix, run the framework, report the clusters"""
    parser = argparse.ArgumentParser(description="ILP framework (Mazza et al. 2016) on the condensate co-phase-separation graph")
    parser.add_argument("--edges", default=str(EDGES))
    parser.add_argument("--priors", default=str(PRIORS))
    parser.add_argument("--score-matrix", default=str(SCORES))
    parser.add_argument("--weight-column", default=None, help="edges.tsv column to use as edge weight (default: all 1)")
    parser.add_argument("--pij", dest="pij_method", choices=["paper", "matrix"], default="matrix")
    parser.add_argument("--no-normalize", dest="normalize_edges", action="store_false")
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
    parser.add_argument("--terminology", choices=sorted(TERMS), default="condensate")
    parser.add_argument("--out")
    args = vars(parser.parse_args())

    edges_file, prior_file, matrix_file = args.pop("edges"), args.pop("priors"), args.pop("score_matrix")
    weight_column, out = args.pop("weight_column"), args.pop("out")
    genes, edges, weights = load_condensate_graph(edges_file, weight_column)
    prior_genes = load_priors(prior_file)
    args["score_matrix"] = load_score_matrix(matrix_file) if args["pij_method"] == "matrix" else None
    clusters = run_framework(genes, edges, weights, prior_genes, **args)
    report = format_clusters(label_genes(clusters, load_abbreviations(FEATURES)), args["terminology"])
    print(report)
    if out:
        with open(out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
