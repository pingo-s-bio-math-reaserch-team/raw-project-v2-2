# Phase 4: Subtype-Specific Counterfactual Sheaf Geometry

This package implements Phase 4 of the glioma sheaf project.

## New technical layer

Phase 4 builds subtype-specific/reference sheaves for groups such as IDH/codel subtype and grade. Each group has its own learned restriction maps, coboundary matrix, and sheaf Laplacian. Patients are then scored under every group sheaf, producing a counterfactual energy vector.

A patient is geometrically close to group g if the patient has low energy under the sheaf learned from group g.

## Main outputs

- `results/phase4_laplacian_divergences.csv`: pairwise distances between group-specific sheaf Laplacians.
- `results/phase4_permutation_divergence_tests.csv`: permutation tests for whether group sheaves differ beyond random labels.
- `results/phase4_counterfactual_accuracy_metrics.csv`: baseline, minimum-energy sheaf, and hybrid model accuracy.
- `results/phase4_counterfactual_patient_energies.csv`: patient-by-patient counterfactual energies under each group sheaf.
- `results/phase4_group_edge_energy_summary.csv`: dominant edge residuals by group.
- `figures/`: divergence heatmaps and accuracy plots.

## Best delta rows

| task              | protocol                           | method                                       |   delta_accuracy |   delta_balanced_accuracy |   delta_macro_f1 |   baseline_accuracy |   method_accuracy |   baseline_balanced_accuracy |   method_balanced_accuracy |
|:------------------|:-----------------------------------|:---------------------------------------------|-----------------:|--------------------------:|-----------------:|--------------------:|------------------:|-----------------------------:|---------------------------:|
| grade_label       | strict_no_idh_no_grade_no_clusters | hybrid_logistic_features_plus_sheaf_energies |       0.0785714  |                0.048423   |       0.0566387  |            0.657143 |          0.735714 |                     0.608163 |                   0.656586 |
| grade4_status     | strict_no_grade_no_clusters        | hybrid_logistic_features_plus_sheaf_energies |       0.0047619  |                0.0130612  |       0.00710294 |            0.866667 |          0.871429 |                     0.858776 |                   0.871837 |
| idh_codel_subtype | full_geometry                      | hybrid_logistic_features_plus_sheaf_energies |      -0.00241546 |               -0.001287   |      -0.00223994 |            0.980676 |          0.978261 |                     0.964471 |                   0.963184 |
| grade_label       | strict_no_grade_no_clusters        | hybrid_logistic_features_plus_sheaf_energies |      -0.0047619  |               -0.00921459 |      -0.00236283 |            0.72381  |          0.719048 |                     0.648114 |                   0.638899 |
| idh_codel_subtype | strict_no_idh_no_grade_no_clusters | hybrid_logistic_features_plus_sheaf_energies |      -0.0169082  |               -0.0321961  |      -0.0253637  |            0.905797 |          0.888889 |                     0.908203 |                   0.876007 |

## Run

```bash
pip install -r requirements.txt
python src/run_phase4.py
```
