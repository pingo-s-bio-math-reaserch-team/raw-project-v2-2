"""Convenience runner for the Phase 1 sheaf residual engine."""

from pathlib import Path
from phase1_sheaf_engine import run_phase1

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    outputs = run_phase1(root / "data.txt", root / "phase1_outputs")
    print("Phase 1 complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
