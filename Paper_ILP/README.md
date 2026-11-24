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


