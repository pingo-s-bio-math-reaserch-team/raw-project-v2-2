# Phase 4 Technical Report: Subtype-Specific Counterfactual Sheaf Geometry

## Purpose

Phases 1--3 established fixed and learned sheaf residuals, then tested subtype and survival associations. Phase 4 adds a more novel geometric layer: instead of learning one global tumor sheaf, we learn one sheaf per biological class and compare the resulting regulatory geometries.

## Method

For each label group g, Phase 4 fits restriction maps

r_{uv}^g(p) = W_{uv}^g x_u(p) - x_v(p),

for edges D->R, D->C, and R->C. The group-specific coboundary B_g and Laplacian L_g define the group regulatory geometry.

The counterfactual energy of patient p under group g is

E_g(p) = x_p^T L_g x_p.

The vector (E_g(p))_g is used for counterfactual assignment, margin analysis, and hybrid prediction.

## Why this is different from standard multi-omics GNNs

Standard multi-omics graph models usually learn embeddings or attention weights. Phase 4 instead compares learned local-to-global consistency laws. This produces interpretable geometric statements such as: an IDH-wildtype tumor is not only different in features; it is far from the IDH-mutant regulatory sheaf.

## Leakage-aware protocols

The package includes:

- full_geometry: biological discovery, not strict prediction;
- strict_no_idh_no_clusters: removes IDH and cluster-risk variables for subtype analysis;
- strict_no_grade_no_clusters: removes grade-risk and cluster-risk variables for grade analysis;
- strict_no_idh_no_grade_no_clusters: strictest protocol.

## Results summary

Permutation tests:

| label_col         | protocol                    |   n_perm |   observed_mean_pairwise_frobenius |   perm_mean |   perm_std |   permutation_p_value |   z_score_vs_permutation |
|:------------------|:----------------------------|---------:|-----------------------------------:|------------:|-----------:|----------------------:|-------------------------:|
| idh_codel_subtype | full_geometry               |      100 |                           0.821207 |    0.459715 |  0.0406448 |            0.00990099 |                  8.89392 |
| idh_codel_subtype | strict_no_idh_no_clusters   |      100 |                           0.811664 |    0.45127  |  0.0391866 |            0.00990099 |                  9.19688 |
| grade_label       | strict_no_grade_no_clusters |      100 |                           0.710364 |    0.480862 |  0.045171  |            0.00990099 |                  5.08074 |
| grade4_status     | strict_no_grade_no_clusters |      100 |                           0.705078 |    0.381304 |  0.0558715 |            0.00990099 |                  5.79498 |

Best accuracy deltas:

| task              | protocol                           | method                                       |   delta_accuracy |   delta_balanced_accuracy |   delta_macro_f1 |   baseline_accuracy |   method_accuracy |   baseline_balanced_accuracy |   method_balanced_accuracy |
|:------------------|:-----------------------------------|:---------------------------------------------|-----------------:|--------------------------:|-----------------:|--------------------:|------------------:|-----------------------------:|---------------------------:|
| grade_label       | strict_no_idh_no_grade_no_clusters | hybrid_logistic_features_plus_sheaf_energies |       0.0785714  |                0.048423   |       0.0566387  |            0.657143 |          0.735714 |                     0.608163 |                   0.656586 |
| grade4_status     | strict_no_grade_no_clusters        | hybrid_logistic_features_plus_sheaf_energies |       0.0047619  |                0.0130612  |       0.00710294 |            0.866667 |          0.871429 |                     0.858776 |                   0.871837 |
| idh_codel_subtype | full_geometry                      | hybrid_logistic_features_plus_sheaf_energies |      -0.00241546 |               -0.001287   |      -0.00223994 |            0.980676 |          0.978261 |                     0.964471 |                   0.963184 |
| grade_label       | strict_no_grade_no_clusters        | hybrid_logistic_features_plus_sheaf_energies |      -0.0047619  |               -0.00921459 |      -0.00236283 |            0.72381  |          0.719048 |                     0.648114 |                   0.638899 |
| idh_codel_subtype | strict_no_idh_no_grade_no_clusters | hybrid_logistic_features_plus_sheaf_energies |      -0.0169082  |               -0.0321961  |      -0.0253637  |            0.905797 |          0.888889 |                     0.908203 |                   0.876007 |

## Interpretation

The strongest Phase 4 novelty is not simply a classifier. It is a subtype-specific sheaf geometry: each tumor group induces a different sheaf Laplacian, and patients can be scored by their energy under each counterfactual regulatory law. This creates a new set of features and biological interpretations that ordinary feature-concatenation or graph-aggregation models do not provide.

## Main limitation

The current data are still one-cohort internal data. Phase 4 improves the technical framework and adds rigorous permutation/accuracy analyses, but external cohort validation remains necessary before claiming definitive state-of-the-art performance.
