
from pathlib import Path
from phase5_transport_sheaf_stability import run_phase5

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    summary = run_phase5(root / 'data', root, n_perm=100)
    print(summary)
