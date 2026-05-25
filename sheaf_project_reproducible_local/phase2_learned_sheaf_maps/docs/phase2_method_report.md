# Phase 2 Method Report

## What Phase 2 adds

Phase 2 replaces fixed scalar residuals with learned vector-stalk restriction maps. For each sheaf edge `u -> v`, the edge stalk is the target-node space, so the residual is

```math
r_{uv}(p)=W_{uv}x_u(p)-x_v(p).
```

This avoids the trivial degeneracy of freely learning both sides of the edge, because the target restriction is fixed as the identity.

## Implemented models

1. `ridge_unconstrained`: cross-fitted ridge maps.
2. `bio_constrained_nonnegative`: cross-fitted nonnegative risk-oriented maps.
3. `reference_ridge_lowrisk`: maps learned on a low-risk/coherent reference subset and evaluated on all patients.
4. `reference_bio_constrained_lowrisk`: low-risk reference maps with nonnegative constraints.
5. `identity_projection`: fixed negative control.
6. `random_projection`: random negative control.
7. `shuffled_target_ridge`: target-permuted negative control.

## Why reference learning matters

When maps are learned on the full tumor cohort, they learn the average cohort law and reduce residual differences. The reference sheaf instead learns a low-risk regulatory law and scores all tumors by deviation from that law.

## Current output interpretation

The strongest biological separation in this run comes from the reference-sheaf variants, especially `reference_ridge_lowrisk`, which shows strong SRIS differences across IDH and grade groups. This is promising but should not be overclaimed yet because grade/KPS/purity live in the clinical phenotype node. Use `no_grade`, `no_idh`, and `no_grade_no_idh` diagnostic variants for leakage-controlled validation.

## Next necessary technical step

Phase 3 should add gene/pathway-level graph nodes so that the sheaf residuals are not only patient-level D/R/C residuals but also edge-resolved pathway residuals.
