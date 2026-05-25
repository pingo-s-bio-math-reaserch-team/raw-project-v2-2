# Phase 7 Technical Report: Publication Synthesis and Claim Calibration

Phase 7 consolidates Phases 1-6 into an integrated evidence ledger. It separates what is methodologically new, what is empirically supported internally, and what still requires external validation.

## Cohort summary

```json
{
  "n_patients": 420,
  "events": 318,
  "grade_counts": {
    "2.0": 77,
    "3.0": 98,
    "4.0": 245
  },
  "subtype_counts": {
    "IDHwt": 259,
    "IDHmut-non-codel": 100,
    "IDHmut-codel": 55,
    "nan": 6
  }
}
```

## Highest-value internal evidence

```json
{
  "largest_positive_delta": {
    "phase": "Phase 4",
    "task": "grade_label",
    "protocol": "strict_no_idh_no_grade_no_clusters",
    "method": "hybrid_logistic_features_plus_sheaf_energies",
    "metric": "accuracy",
    "delta": 0.0785714285714286,
    "evidence_type": "counterfactual subtype sheaf"
  },
  "max_permutation_z": {
    "phase": "Phase 5",
    "evidence": "transport sheaf gap",
    "task": "grade4_status",
    "protocol": "strict_no_grade_no_clusters",
    "statistic": "mean_pairwise_sheaf_transport_gap",
    "observed": 4.47100637090664,
    "null_mean": 3.6406309575407994,
    "null_sd": 0.0420575789999523,
    "z_score": 19.743775399506653,
    "p_value": 0.0099009900990099
  },
  "best_survival_model": {
    "model": "Clinical + molecular + Phase2 edges [reference_bio_constrained_lowrisk]",
    "cv_c_index": 0.79806985992553
  }
}
```

## Manuscript claim guidance

| claim_id   | claim                                                                                                                | status            | support                                                                                                      |
|:-----------|:---------------------------------------------------------------------------------------------------------------------|:------------------|:-------------------------------------------------------------------------------------------------------------|
| Safe-1     | We introduce a biologically constrained cellular-sheaf framework for glioma multi-omics inconsistency.               | safe              | constructed SRIS, learned maps, Laplacians, and edge residuals across Phases 1-2                             |
| Safe-2     | Subtype and grade groups exhibit statistically non-random sheaf Laplacian geometry under internal permutation tests. | safe internal     | Phase 4 permutation tests, p=0.0099 across strict protocols                                                  |
| Safe-3     | Sheaf residual signatures exhibit non-random OT-calibrated transport gaps between biological groups.                 | safe internal     | Phase 5 transport permutation z-scores 6.11-19.74, p=0.0099                                                  |
| Safe-4     | Consensus sheaf features add measurable strict grade-classification signal in internal cross-validation.             | safe internal     | Phase 6 strict grade-label macro-F1 +0.0377 and balanced accuracy +0.0308                                    |
| Caution-1  | The framework improves survival prediction over clinical-molecular baselines.                                        | caution           | Phase 3 C-index gain is only about +0.0011; describe as survival-associated, not clinically superior         |
| Unsafe-1   | The method is state of the art for glioma survival prediction.                                                       | not supported yet | requires external TCGA-to-CGGA validation and stronger C-index/AUROC gains                                   |
| Unsafe-2   | This is the first use of sheaves in machine learning.                                                                | false             | sheaf neural networks/neural sheaf diffusion already exist; our novelty is glioma-specific residual geometry |


## State-of-art contribution matrix

| contribution_id   | name                                           |   evidence_weighted_score_0_10 | claim_tier                     |
|:------------------|:-----------------------------------------------|-------------------------------:|:-------------------------------|
| C1                | Sheaf Regulatory Inconsistency Score (SRIS)    |                          5.99  | Exploratory contribution       |
| C2                | Learned/reference biological restriction maps  |                          6.058 | Moderate internal contribution |
| C3                | Subtype-specific counterfactual sheaf geometry |                          6.924 | Moderate internal contribution |
| C4                | Transport-Calibrated Sheaf Stability           |                          7.076 | Strong internal contribution   |
| C5                | Consensus Sheaf Discovery and Reliability      |                          7.156 | Strong internal contribution   |
| C6                | Publication-grade claim calibration layer      |                          6.624 | Moderate internal contribution |