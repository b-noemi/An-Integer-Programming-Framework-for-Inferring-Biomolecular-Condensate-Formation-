"""Build a graph from llps_human_protein_clean.csv.

Outputs (data/graph/):
  adjacency.tsv  symmetric matrix over all proteins:
                 1 = phase-separated together, -1 = tested pairwise and didn't, 0 = untested
  edges.tsv    every tested pair with evidence counts and its label (1 or 0)
  nodes.tsv    uniprot, degree, evidence counts
"""

import itertools
from pathlib import Path
import pandas as pd

CLEAN_CSV = Path("llps_human_protein_clean.csv")
OUT_DIR = Path("data/graph")


def load_experiments(path):
    """One row per experiment, with the deduplicated, sorted UniProt set."""
    exp = pd.read_csv(path, dtype={"PMID": str})
    exp["uniprots"] = exp["UniProt ID"].str.split(";").apply(lambda ids: sorted(set(ids)))
    exp["n_proteins"] = exp["uniprots"].apply(len)
    return exp


def experiment_pairs(exp):
    """Expand each experiment into one row per protein pair (u < v)."""
    rows = []
    for e in exp.itertuples(index=False):
        for u, v in itertools.combinations(e.uniprots, 2):
            rows.append((u, v, e.phase_separated, e.n_proteins, e.PSID, e.PMID))
    return pd.DataFrame(rows, columns=["u", "v", "phase_separated", "n_proteins", "psid", "pmid"])


def summarize_pairs(pairs):
    """Per-pair evidence counts and the 1 / 0 / untested label."""
    yes = pairs["phase_separated"] == 1
    pairwise = pairs["n_proteins"] == 2
    g = pairs.assign(
        yes=yes,
        no=~yes,
        no_pairwise=~yes & pairwise,
    ).groupby(["u", "v"])
    edges = g.agg(
        n_yes=("yes", "sum"),
        n_no=("no", "sum"),
        n_no_pairwise=("no_pairwise", "sum"),
        psids=("psid", lambda s: ";".join(sorted(set(s)))),
        pmids=("pmid", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    edges["label"] = pd.NA
    edges.loc[edges["n_no_pairwise"] > 0, "label"] = 0
    edges.loc[edges["n_yes"] > 0, "label"] = 1
    edges["conflict"] = (edges["n_yes"] > 0) & (edges["n_no"] > 0)
    return edges


def build_nodes(exp, edges):
    """Node table: every protein in the cleaned data, with degree in the binary graph."""
    proteins = sorted({u for ids in exp["uniprots"] for u in ids})
    pos = edges[edges["label"] == 1]
    neg = edges[edges["label"] == 0]
    degree = pd.concat([pos["u"], pos["v"]]).value_counts()
    n_neg = pd.concat([neg["u"], neg["v"]]).value_counts()
    long = exp.explode("uniprots")
    n_exp = long["uniprots"].value_counts()
    n_exp_yes = long.loc[long["phase_separated"] == 1, "uniprots"].value_counts()

    nodes = pd.DataFrame({"uniprot": proteins})
    nodes["degree"] = nodes["uniprot"].map(degree).fillna(0).astype(int)
    nodes["n_negative_partners"] = nodes["uniprot"].map(n_neg).fillna(0).astype(int)
    nodes["n_experiments"] = nodes["uniprot"].map(n_exp).astype(int)
    nodes["n_experiments_yes"] = nodes["uniprot"].map(n_exp_yes).fillna(0).astype(int)
    return nodes.sort_values(["degree", "uniprot"], ascending=[False, True])


def connected_components(nodes, edge_list):
    """Component sizes via union-find (networkx isn't installed in this venv)."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edge_list:
        parent[find(u)] = find(v)
    sizes = pd.Series([find(n) for n in nodes]).value_counts()
    return sorted(sizes.tolist(), reverse=True)


exp = load_experiments(CLEAN_CSV)
pairs = experiment_pairs(exp)
edges = summarize_pairs(pairs)
nodes = build_nodes(exp, edges)

tested = edges[edges["label"].notna()].copy()
tested["label"] = tested["label"].astype(int)
untested = edges[edges["label"].isna()]

proteins = sorted(nodes["uniprot"])
adjacency = pd.DataFrame(0, index=proteins, columns=proteins)
for u, v, label in tested[["u", "v", "label"]].itertuples(index=False):
    value = 1 if label == 1 else -1
    adjacency.loc[u, v] = value
    adjacency.loc[v, u] = value

OUT_DIR.mkdir(parents=True, exist_ok=True)
adjacency.to_csv(OUT_DIR / "adjacency.tsv", sep="\t", index_label="uniprot")
cols = ["u", "v", "label", "n_yes", "n_no", "n_no_pairwise", "conflict", "psids", "pmids"]
tested[cols].sort_values(["label", "n_yes"], ascending=False).to_csv(
    OUT_DIR / "edges.tsv", sep="\t", index=False)
nodes.to_csv(OUT_DIR / "nodes.tsv", sep="\t", index=False)

print("Wrote outputs to data/graph")