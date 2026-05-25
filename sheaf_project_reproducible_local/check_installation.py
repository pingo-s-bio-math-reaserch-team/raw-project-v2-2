#!/usr/bin/env python3
"""Fast setup/data/code check. Does not recompute expensive phases."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
required_files = [
    ROOT/'data/data.txt',
    ROOT/'data/twentyFourAndUp.xlsx',
    ROOT/'phase1_sheaf_residual_engine/src/phase1_sheaf_engine.py',
    ROOT/'phase2_learned_sheaf_maps/src/phase2_learned_sheaf.py',
    ROOT/'phase3_survival_validation/src/phase3_survival_analysis.py',
    ROOT/'phase4_subtype_sheaf_geometry/src/phase4_subtype_sheaf_geometry.py',
    ROOT/'phase5_transport_sheaf_stability/src/phase5_transport_sheaf_stability.py',
    ROOT/'phase6_consensus_sheaf_discovery/src/phase6_consensus_sheaf_discovery.py',
    ROOT/'phase7_publication_synthesis/src/phase7_publication_synthesis.py',
]
missing = [str(p) for p in required_files if not p.exists()]
if missing:
    print('Missing files:')
    for m in missing: print(' -', m)
    sys.exit(1)

modules = ['numpy','pandas','scipy','sklearn','matplotlib','openpyxl']
missing_mods=[]
for m in modules:
    if importlib.util.find_spec(m) is None:
        missing_mods.append(m)
if missing_mods:
    print('Missing Python packages:', ', '.join(missing_mods))
    print('Install with: pip install -r requirements_all_phases.txt')
    sys.exit(1)
print('Installation check passed: data files, phase code, and Python packages are present.')
