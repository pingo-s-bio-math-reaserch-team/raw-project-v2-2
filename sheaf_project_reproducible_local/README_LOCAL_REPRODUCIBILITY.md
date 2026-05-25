# Sheaf-Theoretic Glioma Multi-Omics Project: Local Reproducibility Package

Use this package to show every Phase 1--7 analysis working locally.

Start here:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_all_phases.txt statsmodels
python run_all_phases_local.py
python verify_phase_outputs.py
```

For details, read `RUN_ON_YOUR_MACHINE.md`.
