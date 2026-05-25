from pathlib import Path
from phase6_consensus_sheaf_discovery import run_phase6

root = Path(__file__).resolve().parents[1]
summary = run_phase6(root/'data', root/'results', root/'figures')
print('Phase 6 complete. Summary keys:', list(summary.keys()))
