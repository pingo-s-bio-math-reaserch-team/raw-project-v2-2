#!/usr/bin/env python3
"""Run one phase at a time with data handoff checks.

Examples:
    python run_one_phase.py 1
    python run_one_phase.py 4
    python run_one_phase.py all
    python run_one_phase.py 8   # optional publishability upgrade
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def ensure(path: Path, msg: str | None = None):
    if not path.exists():
        raise FileNotFoundError(msg or f"Missing required file: {path}")

def copy(src: Path, dst: Path):
    ensure(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def run(cmd: list[str], cwd: Path, pythonpath: Path | None = None):
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
    print("\n" + "="*80)
    print("RUN:", " ".join([sys.executable] + cmd))
    print("CWD:", cwd)
    print("="*80)
    subprocess.run([sys.executable] + cmd, cwd=str(cwd), env=env, check=True)

def phase1():
    p = ROOT / "phase1_sheaf_residual_engine"
    ensure(DATA / "data.txt", "Phase 1 requires data/data.txt")
    code = "from pathlib import Path; from phase1_sheaf_engine import run_phase1; print(run_phase1(Path('../data/data.txt'), Path('results')))"
    run(["-c", code], p, p / "src")

def phase2():
    p = ROOT / "phase2_learned_sheaf_maps"
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", p / "data/phase1_clean_encoded.csv")
    run(["src/run_phase2.py"], p, p / "src")

def phase3():
    p = ROOT / "phase3_survival_validation"
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", p / "data/phase1_sris_results.csv")
    copy(ROOT / "phase2_learned_sheaf_maps/results/phase2_sris_all_models.csv", p / "data/phase2_sris_all_models.csv")
    run(["src/run_phase3.py"], p, p / "src")

def phase4():
    p = ROOT / "phase4_subtype_sheaf_geometry"
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", p / "data/phase1_clean_encoded.csv")
    run(["src/run_phase4.py"], p, p / "src")

def phase5():
    p = ROOT / "phase5_transport_sheaf_stability"
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", p / "data/phase1_clean_encoded.csv")
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", p / "data/phase1_sris_results.csv")
    copy(ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv", p / "data/phase4_counterfactual_patient_energies.csv")
    run(["src/run_phase5.py"], p, p / "src")

def phase6():
    p = ROOT / "phase6_consensus_sheaf_discovery"
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", p / "data/phase1_clean_encoded.csv")
    copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", p / "data/phase1_sris_results.csv")
    copy(ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv", p / "data/phase4_counterfactual_patient_energies.csv")
    copy(ROOT / "phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv", p / "data/phase5_patient_transport_features.csv")
    copy(ROOT / "phase5_transport_sheaf_stability/results/phase5_pairwise_transport_metrics.csv", p / "data/phase5_pairwise_transport_metrics.csv")
    run(["src/run_phase6.py"], p, p / "src")

def phase7():
    p = ROOT / "phase7_publication_synthesis"
    # The shipped Phase 7 local script is claim-ledger focused. Full tables are already included
    # in the original phase7 package outputs and combined_results.
    (p / "results").mkdir(exist_ok=True)
    run(["src/run_phase7.py"], p, p / "src")
    (p / "results/phase7_local_run_note.txt").write_text("Phase 7 claim ledger script executed locally.\n", encoding="utf-8")

def phase8():
    p = ROOT / "phase8_publishability_upgrade"
    ensure(p, "Phase 8 folder is optional but should be included in this package.")
    # Refresh data from latest local outputs where available.
    if (ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv").exists():
        copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv", p / "data/phase1_clean_encoded.csv")
    if (ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv").exists():
        copy(ROOT / "phase1_sheaf_residual_engine/results/phase1_sris_results.csv", p / "data/phase1_sris_results.csv")
    if (ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv").exists():
        copy(ROOT / "phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv", p / "data/phase4_counterfactual_patient_energies.csv")
    if (ROOT / "phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv").exists():
        copy(ROOT / "phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv", p / "data/phase5_patient_transport_features.csv")
    run(["src/run_phase8.py"], p, p / "src")

PHASES = {"1": phase1, "2": phase2, "3": phase3, "4": phase4, "5": phase5, "6": phase6, "7": phase7, "8": phase8}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", help="1-8 or all. Phase 8 is optional publishability upgrade.")
    args = ap.parse_args()
    if args.phase == "all":
        for k in ["1","2","3","4","5","6","7"]:
            PHASES[k]()
        print("\nPhases 1-7 completed. Run `python run_one_phase.py 8` for optional publishability upgrade.")
    elif args.phase in PHASES:
        PHASES[args.phase]()
        print(f"\nPhase {args.phase} completed.")
    else:
        raise SystemExit("Use phase 1-8 or all")

if __name__ == "__main__":
    main()
