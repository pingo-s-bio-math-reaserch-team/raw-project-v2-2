# Quick Start: Running the Sheaf Multi-Omics Project Locally

This folder is a runnable reproduction package for the internal Phases 1--7 analyses, plus the optional Phase 8 publishability upgrade scaffold. It includes the patient table (`data/data.txt`), the original spreadsheet (`data/twentyFourAndUp.xlsx`), phase code, phase outputs, and instructions.

## 0. What this package can and cannot prove

This package reproduces the **internal TCGA-style analyses** that have already been built. It does **not** include CGGA external-validation data. Therefore, do not claim clinical state-of-the-art diagnosis yet. The correct claim is: the framework is a novel, internally validated sheaf-residual approach that still needs independent external validation before clinical/SOTA claims.

## 1. Set up Python

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install --upgrade pip
pip install -r requirements_all_phases.txt
```

## 2. Check installation without rerunning heavy analyses

```bash
python check_installation.py
```

Expected:

```text
Installation check passed: data files, phase code, and Python packages are present.
```

## 3. Verify the shipped outputs

```bash
python verify_phase_outputs.py
```

Expected:

```text
All expected phase outputs are present.
Manifest phases: [1, 2, 3, 4, 5, 6, 7]
```

## 4. Run one phase at a time

This is the safest way for teammates to work.

```bash
python run_one_phase.py 1
python run_one_phase.py 2
python run_one_phase.py 3
python run_one_phase.py 4
python run_one_phase.py 5
python run_one_phase.py 6
python run_one_phase.py 7
```

Optional publishability upgrade:

```bash
python run_one_phase.py 8
```

## 5. Run all internal phases

```bash
python run_one_phase.py all
```

Phases 4--6 include permutation, transport, and consensus computations, so a full rerun can take a while. If you only want to confirm the repo is intact, use `check_installation.py` and `verify_phase_outputs.py` instead.

## 6. Important output folders

```text
combined_results/                         key copied outputs from all phases
phase1_sheaf_residual_engine/results/     fixed sheaf SRIS outputs
phase2_learned_sheaf_maps/results/        learned-map outputs and accuracy benchmarks
phase3_survival_validation/results/       Cox/survival outputs
phase4_subtype_sheaf_geometry/results/    subtype sheaf geometry outputs
phase5_transport_sheaf_stability/results/ OT stability outputs
phase6_consensus_sheaf_discovery/results/ consensus feature outputs
phase7_publication_synthesis/results/     claim-ledger outputs
phase8_publishability_upgrade/results/    optional lockbox/publishability outputs
```

## 7. Clinical interpretation warning

This project is designed for potential clinical decision support, not autonomous diagnosis. Until external validation is complete, write: "may support subtype/risk stratification after external validation," not "diagnoses brain tumors."
