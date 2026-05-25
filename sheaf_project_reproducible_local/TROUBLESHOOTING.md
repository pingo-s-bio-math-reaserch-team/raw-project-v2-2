# Troubleshooting

## `ModuleNotFoundError: No module named ot`

Install POT:

```bash
pip install POT
```

The import name is `ot`, but the package name is `POT`.

## `ModuleNotFoundError: No module named lifelines`

Install requirements:

```bash
pip install -r requirements_all_phases.txt
```

## A full rerun takes too long

Use individual phases:

```bash
python run_one_phase.py 1
python run_one_phase.py 2
```

or verify shipped outputs without recomputing:

```bash
python check_installation.py
python verify_phase_outputs.py
```

## Phase 5 prints many `Mean of empty slice` warnings

These warnings come from edge cases in group-wise transport summaries. The script still completes and writes outputs. They should be suppressed/cleaned before final public release, but they do not necessarily indicate a failed run.

## Windows path issues

Use PowerShell from the project root. Activate the environment with:

```powershell
.venv\Scripts\activate
python check_installation.py
```

## Clinical claim warning

Do not present these outputs as a diagnostic medical device. This is research code for a computational biology project. Final clinical claims require independent cohort validation, prospective evaluation, and clinical review.
