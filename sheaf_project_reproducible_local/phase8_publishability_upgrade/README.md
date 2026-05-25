# Phase 8 Publishability Upgrade Package

This package is the next step toward an IEEE BIBM-quality paper. It provides:

1. A locked holdout validation audit using the current TCGA-style dataset.
2. A CGGA schema adapter for external validation once CGGA files are added.
3. A gene/pathway-level regulatory sheaf scaffold.
4. Publication-readiness checklists, claim controls, and BIBM manuscript skeleton.

Important: this package does **not** claim CGGA external results, because CGGA raw files were not present in the workspace.

## Main outputs

- `results/phase8_lockbox_holdout_metrics.csv`
- `results/phase8_best_lockbox_deltas.csv`
- `results/phase8_external_validation_requirements.csv`
- `results/phase8_publishability_checklist.csv`
- `src/cgga_schema_adapter.py`
- `src/pathway_sheaf_scaffold.py`

## Rebuild

Run from the original environment:

```bash
python /mnt/data/build_phase8.py
```

## CGGA external validation workflow

1. Place files in `cgga_dropbox/`.
2. Standardize clinical data:

```bash
python src/cgga_schema_adapter.py --clinical cgga_dropbox/cgga_clinical.csv --out data/processed_cgga_clinical.csv
```

3. Add matched expression/methylation/microRNA/CNV files.
4. Run external sheaf scoring using TCGA-trained restriction maps.
5. Report CGGA AUROC/C-index/pathway-stability without retraining on CGGA labels.
