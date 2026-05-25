# Phase 8 Technical Report: Publishability Upgrade

## Purpose

Phase 8 converts the project from a promising internal method into a publication-ready experimental framework. The goal is to make the paper credible for IEEE BIBM by adding strict validation discipline, external-validation readiness, benchmark structure, and a route to gene/pathway-level biological discovery.

## What was completed

- A locked internal holdout validation layer was run on the current TCGA-style data.
- Strict feature sets were constructed to avoid direct label leakage.
- Phase 1, Phase 4, Phase 5, and Phase 6 sheaf features were evaluated against the strict clinical/molecular baseline.
- A CGGA adapter was written to standardize external clinical files.
- A gene/pathway-level regulatory sheaf scaffold was written for future full multi-omics matrices.
- A BIBM manuscript skeleton and claim ledger were prepared.

## Important limitation

CGGA external validation has not yet been run because raw CGGA files were not present in the workspace. This package is designed so the team can drop external data into `cgga_dropbox/` and run the same pipeline.

## Strongest immediate paper upgrade

The paper should now be framed as a sheaf residual geometry with rigorous validation safeguards, not as a survival SOTA claim. The next decisive step is external validation on CGGA.
