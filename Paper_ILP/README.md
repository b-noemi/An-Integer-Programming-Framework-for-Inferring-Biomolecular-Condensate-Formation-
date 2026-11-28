# Paper ILP

## How to run the program:

```
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

```

## How to change up things for our project

The data is read in falls into two data structures, an interaction graph and a disease. They are represented as follows:

Graph:
 - genes: list[str]
 - edges: int array (k, 2) of indices into genes
 - weights: float array (k,)

 Disease:
  - prior_genes: list[str]

For our condensate program, we might need to change up the graph representation to distinguish RNA and proteins. I think 
the disease should stay the same as just a list of things that are involved in the condensate.

```
from run import run_framework, format_clusters
clusters = run_framework(genes, edges, weights, prior_genes, n_random=200)
print(format_clusters(clusters))
```

From what I surmised, we'll have to change the calculation for pij as well. This is calculated in the function
`random_edge_probability` in `scoring.py`, and called from line 27 of `run_framework` in `run.py`.

Currently, its calculation is just based off of simple graph metrics (degree of nodes in disease subgraph). If its value 
varies more subtly (like based on RNA vs protein) we might have to change some other parts of the code, but if it remains
that simple it should be a plug and play.



**p_ij decision (interim):** `pij_method="matrix"` (`scoring.matrix_edge_probability`) uses `NEW_log_uniform_relative_scores.csv` as p_ij; for the 32 proteins in the co-phase-separation graph with no annotated IDR we fill p_ij in log space with the scored partner protein's geometric row median, or the global geometric median (~0.10) if neither protein is scored, rather than dropping them (48/91 co-phase-separation pairs, incl. tau, MED1, CAPRIN1) or using 0/1 (L1 = log(β/p) → +∞ or ≤ 0, which would force or forbid them in a candidate condensate by annotation artifact alone).
**Why per-protein:** ~84% of the variance in log p_ij is explained by per-protein "stickiness" effects, so the partner protein's row carries most of the signal, and a median fill gives L1 ≈ +2.2 ≈ −γ, keeping imputed pairs near break-even; this is temporary until IDR features are computed for the missing proteins (see `imputed` column in the output).

## Choosing alpha and normalization (co-phase-separation graph)

- **No normalization (`--no-normalize`).** The paper's D^-1/2 W D^-1/2 discounts pairs involving highly connected proteins, but our seed proteins (FUS, G3BP1) *are* the most connected proteins, so it weakens the seed signal: only proteins with a single co-phase-separation partner enter the seed-relevant subgraph, and those can never satisfy the ILP's connectivity constraint (0 candidate condensates). The empirical P-value against random seed-protein sets already corrects for connectivity, so we keep only that correction.
- **alpha = 0.5 / rho(W) ≈ 0.093** (rho = largest eigenvalue of the raw co-phase-separation adjacency = 5.36). The quantity that matters is alpha·rho (per-hop decay); normalized W has rho = 1, so this matches the paper's alpha = 0.5. Near 1 the scores lose seed-protein specificity and raw propagation diverges at alpha·rho ≥ 1. Recompute rho whenever the graph changes.

## Condensate runs (from Paper_ILP/)

One independent run per condensate, as the paper runs once per disease. Each prior is a single seed protein, so we use `--pvalue 0.05`: with one seed, a protein's empirical P-value can't go below ~1/(n−1) ≈ 0.012, so p < 0.01 would keep only the seed protein.

```
$ python run_condensate.py --no-normalize --alpha 0.093 --pvalue 0.05 --priors ../data/priors/G3BP1.txt
co-phase-separation graph: 88 proteins, 91 co-phase-separation pairs; seed proteins in graph: 1 / 1
[stage 1] 37 proteins with p < 0.05
[stage 2] p_ij from score matrix: 24 scored, 18 row-median fill, 3 global fill
[stage 2] seed-relevant subgraph H: 37 proteins, 45 pairs, gamma = -2.303
[stage 3] 1 candidate condensates
rank  size  edges  imputed  density  ilp_objective  L(C)  proteins
1     4     5      3        0.83     6.61           7.76  P09651(hnRNPA1),P10636(tau),P31483(TIA1),Q13283(G3BP1)*

$ python run_condensate.py --no-normalize --alpha 0.093 --pvalue 0.05 --priors ../data/priors/FUS.txt
co-phase-separation graph: 88 proteins, 91 co-phase-separation pairs; seed proteins in graph: 1 / 1
[stage 1] 30 proteins with p < 0.05
[stage 2] p_ij from score matrix: 22 scored, 15 row-median fill, 0 global fill
[stage 2] seed-relevant subgraph H: 30 proteins, 37 pairs, gamma = -2.303
[stage 3] 1 candidate condensates
rank  size  edges  imputed  density  ilp_objective  L(C)  proteins
1     4     6      3        1.00     7.33           7.33  P22626(hnRNPA2),P35637(FUS)*,P52597(hnRNPF),Q14008(chTOG)
```
Timings are omitted; protein names are added here for UniProt IDs missing from `annotated_idr_features.csv`. `--terminology paper` prints the original gene/network wording. Half of each candidate condensate's co-phase-separation pairs use imputed p_ij.
