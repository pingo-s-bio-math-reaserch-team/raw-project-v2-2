# Running the sheaf multi-omics code locally

This package contains the analysis code and data needed to reproduce the Phase 1--7 internal analyses.
It does **not** contain CGGA external-validation data. External validation must be added separately before making final clinical/SOTA claims.

## 1. Unzip and enter the folder

```bash
unzip sheaf_project_reproducible_local.zip
cd sheaf_project_reproducible_local
```

## 2. Create a clean Python environment

Recommended Python: 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell
pip install --upgrade pip
pip install -r requirements_all_phases.txt
```

The requirements are:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
lifelines
POT
```

Some scripts use `statsmodels`; install it too if your environment asks for it:

```bash
pip install statsmodels
```

## 3. Run every phase in order

```bash
python run_all_phases_local.py
```

This runner handles all data copying between phases. It runs:

```text
Phase 1: fixed sheaf residual engine / SRIS
Phase 2: learned sheaf restriction maps and accuracy benchmarks
Phase 3: survival/Cox validation
Phase 4: subtype-specific sheaf geometry
Phase 5: OT-calibrated sheaf stability
Phase 6: consensus sheaf discovery
Phase 7: publication synthesis / claim ledger
```

## 4. Verify outputs

```bash
python verify_phase_outputs.py
```

Expected final folder:

```text
combined_results/
  phase1/
  phase2/
  phase3/
  phase4/
  phase5/
  phase6/
  phase7/
  run_manifest.json
```

## 5. Data included

```text
data/data.txt                         extracted patient table used by Phase 1
data/twentyFourAndUp.xlsx             original spreadsheet input
data/txtMaker.py                      provenance script that created data.txt
data/Main.py                          original prototype script
data/drive-download-20260523T...zip   original uploaded archive
```

## 6. Clinical/publishability note

These analyses are internal. The method is technically novel as a sheaf-residual framework, but final clinical-diagnosis and state-of-the-art claims require external validation, ideally TCGA-to-CGGA or another independent glioma cohort.
