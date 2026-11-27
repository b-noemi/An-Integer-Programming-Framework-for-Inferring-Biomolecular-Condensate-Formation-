#!/usr/bin/env python3
"""Build a transparent, heuristic IDR sequence-complementarity matrix.

This is a ranking score, NOT KH, a binding free energy, Kd, or calibrated probability.
Requires: pandas, numpy.
"""
import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

FEATURES = [
    "idr_fraction_positive", "idr_fraction_negative",
    "idr_fraction_aromatic_FYW", "idr_mean_kd_hydropathy",
]

def zscore(series):
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="protein_features.csv")
    ap.add_argument("--outdir", default="idr_interaction_mvp")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    idr = data.loc[
        (data["idr_annotation_status"] == "annotated")
        & data[FEATURES].notna().all(axis=1)
    ].drop_duplicates("uniprot_id").copy().reset_index(drop=True)

    rows = []
    for i, a in idr.iterrows():
        for j in range(i + 1, len(idr)):
            b = idr.iloc[j]
            rows.append({
                "protein_A_uniprot": a.uniprot_id,
                "protein_A_abbreviation": a.abbreviation,
                "protein_A_name": a.protein_name,
                "protein_B_uniprot": b.uniprot_id,
                "protein_B_abbreviation": b.abbreviation,
                "protein_B_name": b.protein_name,
                "cross_charge_attraction_raw":
                    a.idr_fraction_positive*b.idr_fraction_negative
                    + a.idr_fraction_negative*b.idr_fraction_positive,
                "same_charge_repulsion_raw":
                    a.idr_fraction_positive*b.idr_fraction_positive
                    + a.idr_fraction_negative*b.idr_fraction_negative,
                "aromatic_pairing_raw":
                    a.idr_fraction_aromatic_FYW*b.idr_fraction_aromatic_FYW,
                "cation_pi_raw":
                    a.idr_fraction_positive*b.idr_fraction_aromatic_FYW
                    + b.idr_fraction_positive*a.idr_fraction_aromatic_FYW,
                "hydropathy_product_raw":
                    a.idr_mean_kd_hydropathy*b.idr_mean_kd_hydropathy,
            })

    pairs = pd.DataFrame(rows)
    for raw, z in [
        ("cross_charge_attraction_raw", "z_cross_charge"),
        ("same_charge_repulsion_raw", "z_same_charge"),
        ("aromatic_pairing_raw", "z_aromatic"),
        ("cation_pi_raw", "z_cation_pi"),
        ("hydropathy_product_raw", "z_hydropathy"),
    ]:
        pairs[z] = zscore(pairs[raw])

    pairs["interaction_propensity_score"] = (
        pairs.z_cross_charge - 0.75*pairs.z_same_charge
        + pairs.z_aromatic + 0.75*pairs.z_cation_pi
        + 0.25*pairs.z_hydropathy
    )
    pairs = pairs.sort_values("interaction_propensity_score", ascending=False)
    pairs["score_rank"] = pairs["interaction_propensity_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    ids = idr.uniprot_id.tolist()
    mat = pd.DataFrame(np.nan, index=ids, columns=ids)
    for r in pairs.itertuples(index=False):
        mat.loc[r.protein_A_uniprot, r.protein_B_uniprot] = r.interaction_propensity_score
        mat.loc[r.protein_B_uniprot, r.protein_A_uniprot] = r.interaction_propensity_score
    np.fill_diagonal(mat.values, 0.0)

    idr.to_csv(out/"annotated_idr_features.csv", index=False)
    pairs.to_csv(out/"idr_pairwise_scores.csv", index=False)
    mat.to_csv(out/"idr_interaction_score_matrix.csv")
    print(f"Proteins scored: {len(idr)}")
    print(f"Unique pairs: {len(pairs)}")
    print(f"Outputs: {out.resolve()}")

if __name__ == "__main__":
    main()
