# Phase 2 Learned Sheaf Package

This package upgrades Phase 1 from fixed scalar residuals to learned vector-stalk restriction maps.

## Core equation

For each edge `u -> v`, the edge stalk is the target-node space and

```math
r_{uv}(p)=W_{uv}x_u(p)-x_v(p), \qquad \mathrm{SRIS}(p)=\sum_{u\to v}\frac{\|r_{uv}(p)\|_2^2}{\dim \mathcal F(uv)}.
```

## Models

- `ridge_unconstrained`: cross-fitted ridge restriction maps.
- `bio_constrained_nonnegative`: cross-fitted nonnegative/risk-oriented ridge maps.
- `identity_projection`: fixed projection negative control.
- `random_projection`: random map negative control.
- `shuffled_target_ridge`: target-permuted negative control.

## Run

```bash
cd phase2_learned_sheaf_package
python src/run_phase2.py
```

Outputs are written to `results/`.

## Leakage controls

The package also computes diagnostic variants:

- `no_idh`: removes IDH-derived feature from the sheaf for IDH validation.
- `no_grade`: removes grade from the sheaf for grade validation.
- `no_grade_no_idh`: removes both.

Age and survival remain external to SRIS.

## Reference sheaf methods

The package also includes:

- `reference_ridge_lowrisk`: learns restriction maps only on the low-risk/coherent reference subset, then scores all patients.
- `reference_bio_constrained_lowrisk`: same reference design, but with nonnegative risk-oriented maps.

These are important because cohort-wide learned maps can absorb the average tumor structure and reduce biologically meaningful residual signal.

## Main result files

- `results/phase2_sris_all_models.csv`
- `results/phase2_model_summary.csv`
- `results/phase2_group_tests.csv`
- `results/phase2_validation_metrics.csv`
- `results/phase2_reconstruction_metrics.csv`
- `results/phase2_maps_and_laplacians.json`
- `paper/phase2_mathematics_only.pdf`
- `paper/phase2_mathematics_only.tex`

## Current interpretation caution

The reference-sheaf variants generate stronger subtype/grade separation than cohort-wide learned maps. However, some negative controls can also correlate with clinical endpoints because the clinical phenotype node contains grade/KPS/purity. For final claims, use the leakage-controlled variants and external validation.
