
"""CGGA schema adapter for external validation.

Usage:
    python src/cgga_schema_adapter.py --clinical cgga_clinical.csv --out processed_cgga_clinical.csv

This adapter is intentionally conservative. It standardizes column names and
creates the minimum clinical/molecular fields needed by the sheaf pipeline. It
does not impute external multi-omics measurements silently; missingness flags
must be tracked in downstream analysis.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def norm_text(x):
    if pd.isna(x): return None
    return str(x).strip().lower().replace('-', '').replace('_','').replace(' ','')

def find_col(df, candidates):
    cmap = {norm_text(c): c for c in df.columns}
    for cand in candidates:
        key = norm_text(cand)
        if key in cmap:
            return cmap[key]
    # loose contains
    for c in df.columns:
        nc = norm_text(c)
        if any(norm_text(cand) in nc for cand in candidates):
            return c
    return None

def encode_idh(v):
    s = norm_text(v)
    if s is None or s in {'na','nan','unknown'}: return np.nan
    if 'mut' in s and 'wild' not in s and 'wt' not in s: return 1.0
    if s in {'wt','wildtype','wild'} or 'wild' in s: return 0.0
    return np.nan

def encode_grade(v):
    s = norm_text(v)
    if s is None: return np.nan
    for g in ['4','3','2','1']:
        if g in s: return float(g)
    try: return float(v)
    except Exception: return np.nan

def encode_status(v):
    s = norm_text(v)
    if s is None: return np.nan
    if any(k in s for k in ['dead','deceased','1']): return 1.0
    if any(k in s for k in ['alive','living','0']): return 0.0
    return np.nan

def adapt_clinical(path):
    df = pd.read_csv(path)
    colmap = {
        'sample_id': find_col(df, ['sample_id','sample','cgga_id','case_id','patient_id']),
        'age': find_col(df, ['age','diagnosis_age','age_at_diagnosis']),
        'grade': find_col(df, ['grade','who_grade','histology_grade']),
        'idh': find_col(df, ['idh','idh_status','idh_mutation_status']),
        'os_months': find_col(df, ['os_months','overall_survival_months','survival_months','os']),
        'deceased': find_col(df, ['status','vital_status','dead','censor','event']),
        'mgmt': find_col(df, ['mgmt','mgmt_status','mgmt_methylation']),
        'sex': find_col(df, ['sex','gender'])
    }
    out = pd.DataFrame()
    out['sample_id'] = df[colmap['sample_id']] if colmap['sample_id'] else np.arange(len(df)).astype(str)
    out['age'] = pd.to_numeric(df[colmap['age']], errors='coerce') if colmap['age'] else np.nan
    out['grade'] = df[colmap['grade']].map(encode_grade) if colmap['grade'] else np.nan
    out['idh_mutant'] = df[colmap['idh']].map(encode_idh) if colmap['idh'] else np.nan
    out['os_months'] = pd.to_numeric(df[colmap['os_months']], errors='coerce') if colmap['os_months'] else np.nan
    out['deceased'] = df[colmap['deceased']].map(encode_status) if colmap['deceased'] else np.nan
    if colmap['mgmt']:
        out['mgmt_methylated'] = df[colmap['mgmt']].astype(str).str.lower().str.contains('meth').astype(float)
    else:
        out['mgmt_methylated'] = np.nan
    out['source_dataset'] = 'CGGA_external'
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clinical', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = adapt_clinical(args.clinical)
    out.to_csv(args.out, index=False)
    print(f'Wrote {args.out} with shape {out.shape}')

if __name__ == '__main__':
    main()
