
# Phase 5: Transport-Calibrated Sheaf Stability

This package extends the Phase 1--4 glioma sheaf pipeline by adding optimal-transport robustness geometry.
It tests whether patient-level sheaf residual signatures remain stable under cross-group distributional transport and whether transport-to-reference features add predictive value under strict leakage controls.

Run:

```bash
cd phase5_transport_sheaf_stability_package/src
python run_phase5.py
```

Main outputs:

- `results/phase5_pairwise_transport_metrics.csv`
- `results/phase5_permutation_transport_tests.csv`
- `results/phase5_transport_prediction_metrics.csv`
- `results/phase5_transport_accuracy_deltas.csv`
- `results/phase5_patient_transport_features.csv`
- `results/phase5_summary.json`
- `figures/*.png`

Novelty target: transport-calibrated sheaf residual stability and counterfactual reference distances for glioma multi-omics.
