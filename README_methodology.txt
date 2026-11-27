# Sequence-based IDR interaction propensity MVP

## Scope
This is a heuristic, composition-based ranking model. It is NOT a KH force-field calculation,
a binding free-energy calculation, a Kd predictor, or a calibrated binding probability.

## Input coverage
- Original database rows: 380
- Proteins with explicitly annotated IDRs and complete required descriptors: 195
- Unique unordered pairs scored: 18915

Proteins without an annotated IDR are excluded. Missing IDR annotation is not interpreted as
evidence that the protein has no disordered region.

## Pairwise terms
For IDRs A and B:
- Cross-charge attraction = f+_A*f-_B + f-_A*f+_B
- Same-charge repulsion = f+_A*f+_B + f-_A*f-_B
- Aromatic pairing = f_arom_A*f_arom_B
- Cation-pi proxy = f+_A*f_arom_B + f+_B*f_arom_A
- Hydropathy proxy = mean_hydropathy_A * mean_hydropathy_B

Each term is z-scored across all evaluated pairs. The composite score is:
S = z(cross-charge) - 0.75*z(same-charge) + z(aromatic)
    + 0.75*z(cation-pi) + 0.25*z(hydropathy)

Weights are illustrative, chosen for a first presentation-ready hypothesis-generating score.
They are not fitted to experimental affinity data and should be varied in sensitivity analyses.

## Interpretation
Higher scores indicate stronger predicted sequence-feature complementarity under this heuristic.
Scores are relative to this dataset and have arbitrary units. Do not interpret them as ΔG, Kd,
or probabilities. The model ignores residue positions/patterning, chain conformations,
folded-domain interfaces, solvent, salt, pH, concentration, and binding entropy.

## Files
- annotated_idr_features.csv: feature table for proteins included in this analysis
- idr_pairwise_scores.csv: ranked pair list and component terms
- idr_interaction_score_matrix.csv: symmetric score matrix
