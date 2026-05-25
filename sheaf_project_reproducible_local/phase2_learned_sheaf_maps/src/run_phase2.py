from pathlib import Path
from phase2_learned_sheaf import Phase2Config, run_phase2

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    encoded = root / "data" / "phase1_clean_encoded.csv"
    output_dir = root / "results"
    paths = run_phase2(encoded, output_dir, Phase2Config(alpha=1.0, n_splits=5))
    print("Phase 2 complete.")
    for key, value in paths.items():
        print(f"{key}: {value}")
