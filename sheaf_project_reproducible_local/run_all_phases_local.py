#!/usr/bin/env python3
"""Run all local reproducibility analyses for Phases 1--7.

This script is designed for a teammate's machine. It wires the phase packages
into a single reproducible pipeline by copying each phase's outputs into the
next phase's expected data directory, then running the official phase scripts.

Run from the project root:
    python run_all_phases_local.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMBINED = ROOT / "combined_results"


def ensure_file(path: Path, label: str | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label or 'required file'}: {path}")


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy(src: Path, dst: Path) -> None:
    ensure_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_python(args: list[str], cwd: Path, extra_pythonpath: Iterable[Path] = ()) -> None:
    env = os.environ.copy()
    paths = [str(p) for p in extra_pythonpath]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    cmd = [sys.executable] + args
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("CWD:", cwd)
    print("=" * 90)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def phase1() -> dict:
    phase = ROOT / "phase1_sheaf_residual_engine"
    out = phase / "results"
    clean_dir(out)
    ensure_file(DATA / "data.txt", "raw data.txt")
    code = (
        "from pathlib import Path; "
        "from phase1_sheaf_engine import run_phase1; "
        "outs=run_phase1(Path('../data/data.txt'), Path('results')); "
        "print('Phase 1 outputs:'); "
        "[print(k, v) for k,v in outs.items()]"
    )
    run_python(["-c", code], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase1_clean_encoded.csv",
        out / "phase1_sris_results.csv",
        out / "phase1_sheaf_metadata.json",
        out / "phase1_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 1, "output_dir": str(out), "files": [str(f) for f in expected]}


def phase2() -> dict:
    phase = ROOT / "phase2_learned_sheaf_maps"
    data = phase / "data"
    out = phase / "results"
    clean_dir(data); clean_dir(out)
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", data / "phase1_clean_encoded.csv")
    run_python(["src/run_phase2.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase2_sris_all_models.csv",
        out / "phase2_model_summary.csv",
        out / "phase2_group_tests.csv",
        out / "phase2_validation_metrics.csv",
        out / "phase2_reconstruction_metrics.csv",
        out / "phase2_diagnostic_sris.csv",
        out / "phase2_maps_and_laplacians.json",
        out / "phase2_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 2, "output_dir": str(out), "files": [str(f) for f in expected]}


def phase3() -> dict:
    phase = ROOT / "phase3_survival_validation"
    data = phase / "data"; out = phase / "results"; fig = phase / "figures"
    clean_dir(data); clean_dir(out); clean_dir(fig)
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", data / "phase1_sris_results.csv")
    copy(ROOT / "phase2_learned_sheaf_maps/results/phase2_sris_all_models.csv", data / "phase2_sris_all_models.csv")
    run_python(["src/run_phase3.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase3_survival_model_summary.csv",
        out / "phase3_cox_coefficients.csv",
        out / "phase3_likelihood_ratio_tests.csv",
        out / "phase3_time_horizon_accuracy.csv",
        out / "phase3_out_of_fold_risks.csv",
        out / "phase3_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 3, "output_dir": str(out), "figure_dir": str(fig), "files": [str(f) for f in expected]}


def phase4() -> dict:
    phase = ROOT / "phase4_subtype_sheaf_geometry"
    data = phase / "data"; out = phase / "results"; fig = phase / "figures"
    clean_dir(data); clean_dir(out); clean_dir(fig)
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", data / "phase1_clean_encoded.csv")
    run_python(["src/run_phase4.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase4_laplacian_divergences.csv",
        out / "phase4_permutation_divergence_tests.csv",
        out / "phase4_counterfactual_accuracy_metrics.csv",
        out / "phase4_counterfactual_patient_energies.csv",
        out / "phase4_accuracy_deltas.csv",
        out / "phase4_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 4, "output_dir": str(out), "figure_dir": str(fig), "files": [str(f) for f in expected]}


def phase5() -> dict:
    phase = ROOT / "phase5_transport_sheaf_stability"
    data = phase / "data"; out = phase / "results"; fig = phase / "figures"
    clean_dir(data); clean_dir(out); clean_dir(fig)
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", data / "phase1_clean_encoded.csv")
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", data / "phase1_sris_results.csv")
    copy(ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv", data / "phase4_counterfactual_patient_energies.csv")
    run_python(["src/run_phase5.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase5_pairwise_transport_metrics.csv",
        out / "phase5_permutation_transport_tests.csv",
        out / "phase5_transport_prediction_metrics.csv",
        out / "phase5_patient_transport_features.csv",
        out / "phase5_transport_accuracy_deltas.csv",
        out / "phase5_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 5, "output_dir": str(out), "figure_dir": str(fig), "files": [str(f) for f in expected]}


def phase6() -> dict:
    phase = ROOT / "phase6_consensus_sheaf_discovery"
    data = phase / "data"; out = phase / "results"; fig = phase / "figures"
    clean_dir(data); clean_dir(out); clean_dir(fig)
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", data / "phase1_clean_encoded.csv")
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", data / "phase1_sris_results.csv")
    copy(ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv", data / "phase4_counterfactual_patient_energies.csv")
    copy(ROOT / "phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv", data / "phase5_patient_transport_features.csv")
    copy(ROOT / "phase5_transport_sheaf_stability/results/phase5_pairwise_transport_metrics.csv", data / "phase5_pairwise_transport_metrics.csv")
    run_python(["src/run_phase6.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    expected = [
        out / "phase6_consensus_feature_discovery.csv",
        out / "phase6_prediction_metrics.csv",
        out / "phase6_metric_deltas.csv",
        out / "phase6_summary.json",
    ]
    for f in expected:
        ensure_file(f)
    return {"phase": 6, "output_dir": str(out), "figure_dir": str(fig), "files": [str(f) for f in expected]}


def phase7() -> dict:
    phase = ROOT / "phase7_publication_synthesis"
    out = phase / "results"
    clean_dir(out)
    # This phase is a synthesis / claim-ledger phase. The shipped script prints the safe claim ledger.
    run_python(["src/run_phase7.py"], cwd=phase, extra_pythonpath=[phase / "src"])
    # Produce a local manifest summarizing generated prior-phase outputs.
    ledger = out / "phase7_local_run_note.txt"
    ledger.write_text(
        "Phase 7 synthesis script executed. It prints the claim ledger and should be combined "
        "with Phase 1-6 result tables for manuscript synthesis.\n",
        encoding="utf-8",
    )
    return {"phase": 7, "output_dir": str(out), "files": [str(ledger)]}


def collect_combined_outputs(manifest: list[dict]) -> None:
    COMBINED.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        phase_no = item["phase"]
        dst_dir = COMBINED / f"phase{phase_no}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in item.get("files", []):
            src = Path(f)
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
    write_json(COMBINED / "run_manifest.json", {"generated_at": datetime.now().isoformat(), "phases": manifest})


def main() -> None:
    print("Sheaf multi-omics local reproducibility runner")
    print("Project root:", ROOT)
    print("Python:", sys.version)
    for f in [DATA / "data.txt", DATA / "twentyFourAndUp.xlsx"]:
        ensure_file(f)
    manifest = []
    for fn in [phase1, phase2, phase3, phase4, phase5, phase6, phase7]:
        manifest.append(fn())
    collect_combined_outputs(manifest)
    print("\nAll phases completed successfully.")
    print("Combined key outputs:", COMBINED)
    print("Manifest:", COMBINED / "run_manifest.json")


if __name__ == "__main__":
    main()
