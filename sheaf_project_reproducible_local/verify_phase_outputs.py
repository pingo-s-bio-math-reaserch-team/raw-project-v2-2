#!/usr/bin/env python3
"""Verify that the expected local outputs from Phases 1--7 exist."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
expected = {
    1: ["phase1_clean_encoded.csv", "phase1_sris_results.csv", "phase1_summary.json"],
    2: ["phase2_sris_all_models.csv", "phase2_model_summary.csv", "phase2_summary.json"],
    3: ["phase3_survival_model_summary.csv", "phase3_likelihood_ratio_tests.csv", "phase3_summary.json"],
    4: ["phase4_counterfactual_accuracy_metrics.csv", "phase4_laplacian_divergences.csv", "phase4_summary.json"],
    5: ["phase5_pairwise_transport_metrics.csv", "phase5_patient_transport_features.csv", "phase5_summary.json"],
    6: ["phase6_consensus_feature_discovery.csv", "phase6_prediction_metrics.csv", "phase6_summary.json"],
    7: ["phase7_local_run_note.txt"],
}
missing = []
for p, names in expected.items():
    d = ROOT / "combined_results" / f"phase{p}"
    for name in names:
        if not (d / name).exists():
            missing.append(str(d / name))
if missing:
    print("Missing outputs:")
    for m in missing:
        print(" -", m)
    sys.exit(1)
print("All expected phase outputs are present.")
manifest = ROOT / "combined_results" / "run_manifest.json"
if manifest.exists():
    data = json.loads(manifest.read_text())
    print("Manifest phases:", [p["phase"] for p in data.get("phases", [])])
