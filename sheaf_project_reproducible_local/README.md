# Sheaf Multi-Omics Analysis Code: Phases 1–7

This archive collects the analysis code produced across Phases 1–7 of the sheaf-theoretic glioma multi-omics project.

## Folder map

- `phase1_sheaf_residual_engine/`: fixed/reference D-R-C sheaf residual engine, SRIS, coboundary/Laplacian-compatible outputs.
- `phase2_learned_sheaf_maps/`: learned restriction maps, constrained/reference sheaves, accuracy benchmarks, leakage-control variants.
- `phase3_survival_validation/`: Cox models, likelihood-ratio tests, C-index, time-horizon validation.
- `phase4_subtype_sheaf_geometry/`: subtype-specific sheaf Laplacians, counterfactual energies, divergence/permutation tests.
- `phase5_transport_sheaf_stability/`: OT-calibrated sheaf residual stability and transport features.
- `phase6_consensus_sheaf_discovery/`: consensus feature discovery, stability/ranking, reliability scoring.
- `phase7_publication_synthesis/`: integrated evidence ledger, claim ledger, contribution matrix, external validation schema.
- `build_scripts/`: generator/build scripts used to create later phase packages and reports.

## Suggested run order

Run phases in order, because later phases consume outputs from earlier phases:

```bash
python phase1_sheaf_residual_engine/scripts/run_phase1.py
python phase2_learned_sheaf_maps/src/run_phase2.py
python phase3_survival_validation/src/run_phase3.py
python phase4_subtype_sheaf_geometry/src/run_phase4.py
python phase5_transport_sheaf_stability/src/run_phase5.py
python phase6_consensus_sheaf_discovery/src/run_phase6.py
python phase7_publication_synthesis/src/run_phase7.py
```

Depending on where data/results are stored, paths may need to be adjusted. The original full phase packages include data/result CSVs; this archive focuses on code and technical documentation.

## Notes

- This is not yet an external-validation package. CGGA or another external cohort must be added separately.
- Clinical claims should remain decision-support oriented until external validation is completed.
