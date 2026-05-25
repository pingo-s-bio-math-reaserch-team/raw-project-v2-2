# Phase 1 Sheaf Residual Engine

This package implements the Phase 1 sheaf residual backbone for glioma multi-omics regulatory inconsistency.

## Core correction in this version

Age is excluded from the sheaf Laplacian and from SRIS. It is preserved only as an external covariate/result variable for downstream association and adjustment analyses.

## Run

```bash
python run_phase1.py
```

## Outputs

- `phase1_outputs/phase1_clean_encoded.csv`
- `phase1_outputs/phase1_sris_results.csv`
- `phase1_outputs/phase1_sheaf_metadata.json`
- `phase1_outputs/phase1_summary.json`

## Main equation

SRIS is computed as:

```text
SRIS(p) = ||B_F x_p||^2 = x_p^T L_F x_p
```

where `B_F` is the sheaf coboundary and `L_F = B_F.T @ B_F`.
