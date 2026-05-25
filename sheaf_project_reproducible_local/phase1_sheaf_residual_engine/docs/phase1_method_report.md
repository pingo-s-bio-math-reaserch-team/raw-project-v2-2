# Phase 1 Implementation Report: Formal Sheaf Residual Engine

## What Phase 1 now does

The current prototype has been converted into a formal three-node cellular sheaf:

- `D`: DNA/genomic risk state
- `R`: regulatory/transcriptomic risk state
- `C`: clinical phenotype state, excluding survival outcome labels by default

For every patient `p`, the engine builds a stacked vector

\[
x_p = (x_D(p), x_R(p), x_C(p))
\]

and computes edge residuals

\[
r_e(p) = A_e x_u(p) - B_e x_v(p).
\]

The resulting Sheaf Regulatory Inconsistency Score is

\[
\operatorname{SRIS}(p)=\sum_{e \in \{D\to R,D\to C,R\to C\}} \|r_e(p)\|_2^2.
\]

Equivalently, after constructing the sheaf coboundary matrix `B_F`, the code computes

\[
\operatorname{SRIS}(p)=x_p^T L_{\mathcal F} x_p,\qquad L_{\mathcal F}=B_{\mathcal F}^T B_{\mathcal F}.
\]

## Important rigor improvement

Survival months and vital status are preserved as downstream validation labels, but they are not included in the default `C` node. This avoids survival leakage. The score can later be tested against survival using Cox models rather than trivially encoding survival into the score itself.

## Current Phase 1 sanity-check results

On the uploaded 420-patient table:

- Number of patients: 420
- Number of sheaf edges: 3
- Number of sheaf features: 20
- Mean SRIS: 4.1363
- Median SRIS: 2.7167
- SRIS is higher in IDH-wildtype than IDH-mutant cases in this pilot run.
- Mann--Whitney test for SRIS by IDH status: p = 3.10e-05
- Kruskal--Wallis test for SRIS by grade: p = 4.15e-08

These are not final paper claims yet, because the fixed maps still need learned-map ablations and survival validation. However, they show that the Phase 1 sheaf residual engine is producing biologically nontrivial structure rather than random noise.

## Files produced

- `phase1_sheaf_engine.py`: reusable implementation
- `run_phase1.py`: one-command runner
- `phase1_outputs/phase1_clean_encoded.csv`: encoded patient features
- `phase1_outputs/phase1_sris_results.csv`: residuals, edge energies, edge fractions, SRIS
- `phase1_outputs/phase1_sheaf_metadata.json`: formal coboundary matrix and sheaf Laplacian
- `phase1_outputs/phase1_summary.json`: sanity-check statistics

## Next technical step

The next phase should add learned biologically constrained restriction maps and ablations:

1. fixed sheaf maps versus learned constrained maps;
2. identity sheaf ablation;
3. random sheaf ablation;
4. survival Cox model using SRIS while controlling for age, grade, IDH, MGMT, and purity;
5. subtype-specific sheaf Laplacian divergence.
