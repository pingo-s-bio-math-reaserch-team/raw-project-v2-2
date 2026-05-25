
"""Phase 8: publication-grade lockbox validation and external-validation readiness.

This module evaluates strict held-out performance on the current TCGA-style table
and provides reusable feature-building utilities for later external validation.
It intentionally does not claim external CGGA results unless external files are
provided.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

# See build_phase8.py in the project root for the complete reproducible build script.
# The package contains results generated from that script in results/.

def load_clean_table(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / 'phase1_clean_encoded.csv')

def compute_binary_grade4_label(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df['grade'], errors='coerce') >= 4).astype(int)

def strict_baseline_columns(task: str, protocol: str) -> list[str]:
    cols = ['age','kps','purity','mutation_count','tmb','aneuploidy','immune_score','stromal_score','estimate_score',
            'mgmt_methylated','mgmt_unmethylated','atrx_mutant','atrx_wt','tert_promoter_mutant','tert_expr','tert_expressed',
            'chr7_gain_chr10_loss','egfr_amp']
    if task != 'idh_codel_subtype' and 'no_idh' not in protocol:
        cols += ['idh_mutant','idh_wt']
    if task not in ['grade_label','grade4_status'] and 'no_grade' not in protocol:
        cols += ['grade_risk']
    return cols

def make_elasticnet_logistic_model() -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, solver='saga', penalty='elasticnet', l1_ratio=0.5, C=0.5, class_weight='balanced'))
    ])
