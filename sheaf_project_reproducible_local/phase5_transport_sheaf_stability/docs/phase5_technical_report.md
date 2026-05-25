# Phase 5 Technical Report: Transport-Calibrated Sheaf Stability

## Core idea
Phase 5 adds an optimal-transport layer on top of Phase 4 counterfactual sheaf geometry.  Phase 4 asks whether group-specific sheaf Laplacians differ; Phase 5 asks whether the residual signatures remain stable when patient distributions are optimally transported between biological groups.

## Main objects
- Pairwise transport plans between groups.
- Transport Sheaf Discrepancy (TSD).
- Edge-level transport stability for D->R, D->C, and R->C residuals.
- Cross-fitted transport-to-reference features for strict prediction tests.

## Best balanced-accuracy delta
{
  "task": "grade_label",
  "protocol": "strict_no_idh_no_grade_no_clusters",
  "method": "baseline_plus_phase5_transport_features",
  "delta_accuracy": 0.02619047619047621,
  "delta_balanced_accuracy": 0.03110698824984548,
  "delta_macro_f1": 0.03153609600120355,
  "delta_auroc": NaN,
  "delta_auprc": NaN,
  "delta_auroc_ovr_weighted": 0.004984613898798229,
  "delta_auroc_ovr_macro": 0.006860466022337031
}

## Permutation tests
             task                           protocol  observed_mean_pairwise_sheaf_transport_gap  null_mean  null_sd   z_score  permutation_p_value_high_gap  n_permutations
idh_codel_subtype strict_no_idh_no_grade_no_clusters                                    4.837469   4.216627 0.101582  6.111755                      0.009901             100
      grade_label strict_no_idh_no_grade_no_clusters                                    4.916524   4.132369 0.071637 10.946260                      0.009901             100
    grade4_status        strict_no_grade_no_clusters                                    4.471006   3.640631 0.042058 19.743775                      0.009901             100

## Interpretation
This phase should be presented as a robustness and geometry contribution.  It does not claim that optimal transport alone solves classification.  Instead, it quantifies whether sheaf residual signatures define non-random group-level transport geometry and whether transport distances provide incremental predictive signal under strict controls.
