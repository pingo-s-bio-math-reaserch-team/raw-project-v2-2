# Phase 3 Survival and Clinical Validation Package

This package validates Phase 1/2 sheaf residuals against right-censored overall survival. It treats age as an external covariate, not as part of SRIS.

## Run

```bash
cd phase3_survival_package
python src/run_phase3.py
```

## Main outputs

- `results/phase3_survival_model_summary.csv`: Cox and out-of-fold C-index summary.
- `results/phase3_cox_coefficients.csv`: adjusted hazard ratios per 1 SD.
- `results/phase3_likelihood_ratio_tests.csv`: nested model tests for added sheaf terms.
- `results/phase3_time_horizon_accuracy.csv`: 24-month and 60-month AUROC/AUPRC.
- `results/phase3_out_of_fold_risks.csv`: patient-level out-of-fold risk scores.
- `figures/`: C-index comparison, hazard ratio forest plot, KM-style stratification, horizon AUROC.
- `paper/phase3_mathematics_only.pdf`: mathematical specification.

## Main internal result

The best internal model was the reference biologically constrained Phase 2 edge model. This produced a small out-of-fold C-index gain over the clinical+molecular baseline. Phase 1 edge residuals were statistically significant by nested likelihood-ratio testing but did not improve cross-validated C-index in this internal run, so external validation is necessary before strong publication claims.
