# Phase 2 Sharpened Accuracy Report

## Purpose

This extension adds a reviewer-facing accuracy layer to Phase 2. The learned sheaf remains an inconsistency model; this benchmark tests whether the resulting residuals add measurable predictive signal for clinically meaningful endpoints.

## Accuracy Protocol

All classification metrics are computed from stratified out-of-fold logistic predictions. The classifier uses median imputation, standard scaling, class-balanced logistic regression, and a 0.5 threshold for hard labels. Reported metrics include accuracy, balanced accuracy, precision, recall, F1, AUROC, and AUPRC.

## Two Validation Protocols

1. **Broad benchmark:** includes available clinical/molecular covariates, including RNA/methylation/transcriptome cluster risk fields. This is useful but can be optimistic because clusters may encode endpoint structure.
2. **Strict no-cluster benchmark:** removes potentially label-derived cluster fields from the baseline and should be treated as the cleaner accuracy table for manuscript claims.

## Strict No-Cluster Increment Summary

| endpoint          | model                                | predictor_set                   |   n |   accuracy |   balanced_accuracy |       f1 |    auroc |    auprc |   delta_auroc_vs_strict_baseline |   delta_balanced_accuracy_vs_strict_baseline |
|:------------------|:-------------------------------------|:--------------------------------|----:|-----------:|--------------------:|---------:|---------:|---------:|---------------------------------:|---------------------------------------------:|
| 24-month death    | reference_bio_constrained_lowrisk    | strict baseline + SRIS          | 419 |   0.852029 |            0.845796 | 0.873984 | 0.902848 | 0.889763 |                      -0.00135615 |                                   0.00127432 |
| Grade 4 status    | no_grade/bio_constrained_nonnegative | strict baseline + edge energies | 420 |   0.869048 |            0.864082 | 0.888438 | 0.909434 | 0.89732  |                       0.0195685  |                                   0.0102041  |
| IDH mutant status | no_idh/bio_constrained_nonnegative   | strict baseline + sheaf full    | 420 |   0.940476 |            0.943473 | 0.922601 | 0.986815 | 0.976089 |                       0.00245241 |                                   0.0120921  |

## Broad Increment Summary

| endpoint          | model                                | predictor_set            |   n |   accuracy |   balanced_accuracy |       f1 |    auroc |    auprc |   delta_auroc_vs_baseline |   delta_balanced_accuracy_vs_baseline |
|:------------------|:-------------------------------------|:-------------------------|----:|-----------:|--------------------:|---------:|---------:|---------:|--------------------------:|--------------------------------------:|
| 24-month death    | bio_constrained_nonnegative          | baseline + edge energies | 419 |   0.854415 |            0.847854 | 0.876268 | 0.911382 | 0.918929 |                0.00222129 |                           -0.0183198  |
| Grade 4 status    | no_grade/bio_constrained_nonnegative | baseline + edge energies | 420 |   0.869048 |            0.864082 | 0.888438 | 0.921703 | 0.920007 |                0.0252362  |                            0.00734694 |
| IDH mutant status | no_idh/bio_constrained_nonnegative   | baseline + SRIS          | 420 |   1        |            1        | 1        | 1        | 1        |                0          |                            0          |

## Interpretation

- The clearest incremental accuracy gain is for **Grade 4 status**, where strict baseline + edge energies improves AUROC from the strict baseline by about 0.0196 and balanced accuracy by about 0.0102.
- For **IDH mutant status**, baseline features already predict very strongly. The sheaf adds a small gain in strict balanced accuracy, but this should be framed conservatively.
- For **24-month death**, the strongest baseline is already high. Sheaf residuals are useful as a diagnostic and interpretability layer, but the current binary endpoint does not yet show a large accuracy gain. The next statistically stronger version should use Cox modeling / survival C-index rather than a 24-month binary endpoint.

## Manuscript-Safe Claim

Phase 2 now supports the following cautious claim: learned sheaf residuals provide endpoint-specific incremental predictive value, most clearly for high-grade phenotype classification, while also giving interpretable residual decompositions. Survival improvement remains preliminary and should be evaluated with time-to-event models in Phase 3.
