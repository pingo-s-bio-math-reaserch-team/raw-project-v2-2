import os, json, shutil, textwrap, math, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from statsmodels.duration.hazard_regression import PHReg
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path('/mnt/data/phase3_survival_package')
SRC = ROOT/'src'
DATA = ROOT/'data'
RES = ROOT/'results'
FIG = ROOT/'figures'
PAPER = ROOT/'paper'
for d in [SRC, DATA, RES, FIG, PAPER]:
    d.mkdir(parents=True, exist_ok=True)

# Copy input artifacts for reproducibility
for src in ['/mnt/data/phase1_outputs/phase1_clean_encoded.csv',
            '/mnt/data/phase1_outputs/phase1_sris_results.csv',
            '/mnt/data/phase2_learned_sheaf_package/results/phase2_sris_all_models.csv']:
    if Path(src).exists():
        shutil.copy(src, DATA/Path(src).name)

PHASE3_CODE = r'''#!/usr/bin/env python3
"""
Phase 3: survival and clinical outcome validation for learned sheaf residuals.

This script is intentionally self-contained. It reads Phase 1/2 outputs and produces:
  - Cox proportional-hazards model tables
  - cross-validated Harrell C-index values
  - time-horizon accuracy metrics at 24 and 60 months
  - likelihood-ratio improvements for nested sheaf additions
  - residual/age diagnostic correlations
  - Kaplan-Meier-style stratification plots based on out-of-fold risk

All Cox coefficients are computed after train-only standardization in cross-validation.
Full-sample hazard ratios are per one standard deviation increase in a covariate.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from statsmodels.duration.hazard_regression import PHReg

warnings.filterwarnings('ignore')


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index for right-censored survival data.

    Higher risk should correspond to shorter survival. A pair is usable if one patient
    has an observed event before the other patient's observed/censored time.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    permissible = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue
            # patient i fails before j
            if time[i] < time[j] and event[i] == 1:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            # patient j fails before i
            elif time[j] < time[i] and event[j] == 1:
                permissible += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[j] == risk[i]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else np.nan


def bootstrap_ci_cindex(time, event, risk, n_boot=500, seed=17):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(time)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = harrell_c_index(time[idx], event[idx], risk[idx])
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prepare_features(df: pd.DataFrame, features: List[str], train_idx=None, test_idx=None):
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    if train_idx is None:
        med = X.median(numeric_only=True).fillna(0.0)
        X = X.fillna(med)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)
        return Xs, scaler, med
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    med = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.values)
    Xte = scaler.transform(X_test.values)
    return Xtr, Xte, scaler, med


def fit_phreg(time, event, X):
    """Fit Cox PHReg, falling back to a mild elastic-net fit if necessary."""
    try:
        model = PHReg(time, X, status=event, ties='breslow')
        res = model.fit(disp=0)
        return res, False
    except Exception:
        model = PHReg(time, X, status=event, ties='breslow')
        res = model.fit_regularized(alpha=0.02, refit=False)
        return res, True


def get_params(res, p):
    params = np.asarray(getattr(res, 'params', np.zeros(p)), dtype=float)
    if params.shape[0] != p:
        params = np.resize(params, p)
    return params


def full_sample_cox(df: pd.DataFrame, features: List[str], model_name: str):
    keep = df[['os_months','deceased'] + features].copy()
    keep['os_months'] = pd.to_numeric(keep['os_months'], errors='coerce')
    keep['deceased'] = pd.to_numeric(keep['deceased'], errors='coerce')
    keep = keep.dropna(subset=['os_months','deceased'])
    keep = keep[keep['os_months'] > 0].reset_index(drop=True)
    Xs, scaler, med = prepare_features(keep, features)
    time = keep['os_months'].values.astype(float)
    event = keep['deceased'].values.astype(int)
    res, regularized = fit_phreg(time, event, Xs)
    params = get_params(res, len(features))
    risk = Xs @ params
    cidx = harrell_c_index(time, event, risk)
    ci_low, ci_high = bootstrap_ci_cindex(time, event, risk, n_boot=400)
    rows = []
    # p-values/CIs are unavailable for regularized fallback; otherwise read from statsmodels.
    pvals = getattr(res, 'pvalues', [np.nan] * len(features))
    conf = None
    try:
        conf = res.conf_int()
    except Exception:
        conf = None
    for k, feat in enumerate(features):
        beta = float(params[k])
        hr = float(np.exp(beta))
        if conf is not None and not regularized:
            lo = float(np.exp(conf[k,0])); hi = float(np.exp(conf[k,1]))
        else:
            lo = np.nan; hi = np.nan
        pv = float(pvals[k]) if (not regularized and k < len(pvals)) else np.nan
        rows.append({
            'model': model_name,
            'feature': feat,
            'beta_per_sd': beta,
            'hazard_ratio_per_sd': hr,
            'hr_ci_low': lo,
            'hr_ci_high': hi,
            'p_value': pv,
            'regularized_fallback': bool(regularized),
        })
    summary = {
        'model': model_name,
        'n': int(len(keep)),
        'events': int(event.sum()),
        'n_features': int(len(features)),
        'log_likelihood': float(getattr(res, 'llf', np.nan)) if not regularized else np.nan,
        'in_sample_c_index': cidx,
        'in_sample_c_index_ci_low': ci_low,
        'in_sample_c_index_ci_high': ci_high,
        'regularized_fallback': bool(regularized),
    }
    return pd.DataFrame(rows), summary, risk, keep


def cross_validated_risk(df: pd.DataFrame, features: List[str], n_splits=5, seed=12):
    keep = df[['os_months','deceased'] + features].copy()
    keep['os_months'] = pd.to_numeric(keep['os_months'], errors='coerce')
    keep['deceased'] = pd.to_numeric(keep['deceased'], errors='coerce')
    keep = keep.dropna(subset=['os_months','deceased'])
    keep = keep[keep['os_months'] > 0].reset_index(drop=True)
    time = keep['os_months'].values.astype(float)
    event = keep['deceased'].values.astype(int)
    # Stratify by event when possible.
    if len(np.unique(event)) > 1 and min(np.bincount(event)) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(keep)), event)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(keep)))
    risk = np.full(len(keep), np.nan)
    fold_rows = []
    for fold, pair in enumerate(split_iter):
        train_idx, test_idx = pair
        Xtr, Xte, scaler, med = prepare_features(keep, features, train_idx, test_idx)
        res, regularized = fit_phreg(time[train_idx], event[train_idx], Xtr)
        params = get_params(res, len(features))
        risk[test_idx] = Xte @ params
        fold_c = harrell_c_index(time[test_idx], event[test_idx], risk[test_idx])
        fold_rows.append({
            'fold': fold,
            'n_test': int(len(test_idx)),
            'events_test': int(event[test_idx].sum()),
            'fold_c_index': fold_c,
            'regularized_fallback': bool(regularized),
        })
    cv_c = harrell_c_index(time, event, risk)
    ci_low, ci_high = bootstrap_ci_cindex(time, event, risk, n_boot=400, seed=seed+101)
    return keep, risk, pd.DataFrame(fold_rows), {
        'cv_c_index': cv_c,
        'cv_c_index_ci_low': ci_low,
        'cv_c_index_ci_high': ci_high,
        'mean_fold_c_index': float(np.nanmean([r['fold_c_index'] for r in fold_rows])),
        'std_fold_c_index': float(np.nanstd([r['fold_c_index'] for r in fold_rows])),
    }


def horizon_metrics(time, event, risk, horizon):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    # Cases: event by horizon. Controls: observed/censored alive beyond horizon.
    # Exclude censored before horizon because status at horizon is unknown.
    cases = (event == 1) & (time <= horizon)
    controls = (time > horizon)
    mask = cases | controls
    y = cases[mask].astype(int)
    s = risk[mask]
    if len(np.unique(y)) < 2:
        return {'horizon_months': horizon, 'n_usable': int(mask.sum()), 'n_cases': int(y.sum()), 'auroc': np.nan, 'auprc': np.nan, 'balanced_accuracy_median_threshold': np.nan, 'f1_median_threshold': np.nan}
    threshold = np.median(s)
    pred = (s >= threshold).astype(int)
    return {
        'horizon_months': horizon,
        'n_usable': int(mask.sum()),
        'n_cases': int(y.sum()),
        'auroc': float(roc_auc_score(y, s)),
        'auprc': float(average_precision_score(y, s)),
        'balanced_accuracy_median_threshold': float(balanced_accuracy_score(y, pred)),
        'f1_median_threshold': float(f1_score(y, pred)),
    }


def likelihood_ratio(base, full):
    if not np.isfinite(base.get('log_likelihood', np.nan)) or not np.isfinite(full.get('log_likelihood', np.nan)):
        return np.nan, np.nan, np.nan
    df = full['n_features'] - base['n_features']
    if df <= 0:
        return np.nan, np.nan, np.nan
    stat = 2 * (full['log_likelihood'] - base['log_likelihood'])
    p = chi2.sf(stat, df)
    return float(stat), int(df), float(p)


def simple_km(time, event):
    order = np.argsort(time)
    time = time[order]; event = event[order]
    unique_times = np.unique(time[event == 1])
    surv = []
    s = 1.0
    for t in unique_times:
        at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            s *= (1 - d / at_risk)
        surv.append((t, s))
    if not surv:
        return np.array([0.0]), np.array([1.0])
    t = np.array([0.0] + [x[0] for x in surv])
    s = np.array([1.0] + [x[1] for x in surv])
    return t, s


def logrank_two_groups(time, event, group):
    # Basic two-sample log-rank test for group 0 vs 1.
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)
    event_times = np.unique(time[event == 1])
    O1 = E1 = V1 = 0.0
    for t in event_times:
        at = time >= t
        d = (time == t) & (event == 1)
        n = at.sum(); n1 = (at & (group == 1)).sum(); n0 = n - n1
        dd = d.sum(); d1 = (d & (group == 1)).sum()
        if n <= 1:
            continue
        e1 = dd * n1 / n
        v1 = (n1 * n0 * dd * (n - dd)) / (n**2 * (n - 1)) if n > 1 else 0
        O1 += d1; E1 += e1; V1 += v1
    if V1 <= 0:
        return np.nan, np.nan
    z2 = (O1 - E1)**2 / V1
    return float(z2), float(chi2.sf(z2, 1))


def make_phase3_tables(data_dir: Path, results_dir: Path, figures_dir: Path):
    import matplotlib.pyplot as plt

    p1 = pd.read_csv(data_dir/'phase1_sris_results.csv')
    p2 = pd.read_csv(data_dir/'phase2_sris_all_models.csv')
    clean = pd.read_csv(data_dir/'phase1_clean_encoded.csv')

    # Rename Phase 1 energies for feature clarity.
    p1_model = p1.copy()
    p1_model['model'] = 'phase1_fixed_reference_sheaf'
    p1_model = p1_model.rename(columns={
        'SRIS': 'SRIS_phase1',
        'E_D_to_R': 'P1_E_D_to_R',
        'E_D_to_C': 'P1_E_D_to_C',
        'E_R_to_C': 'P1_E_R_to_C',
    })
    base_cols = ['patient_id','sample_id','os_months','deceased','age','grade','grade_risk','purity','kps','idh_mutant','mgmt_methylated','egfr_amp','SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']
    p1_surv = p1_model[base_cols].copy()
    for c in ['age','grade','grade_risk','purity','kps','idh_mutant','mgmt_methylated','egfr_amp','os_months','deceased']:
        p1_surv[c] = pd.to_numeric(p1_surv[c], errors='coerce')
    # Derived clinical covariates.
    p1_surv['kps_low'] = -p1_surv['kps']
    p1_surv['purity_low'] = -p1_surv['purity']

    # Phase 2 model rows.
    p2_base = p2.copy()
    p2_base = p2_base.rename(columns={'SRIS': 'SRIS_phase2','E_D_to_R':'P2_E_D_to_R','E_D_to_C':'P2_E_D_to_C','E_R_to_C':'P2_E_R_to_C'})
    # Merge Phase 1 and Phase 2 by patient fields for each Phase 2 model.
    p2_keep = ['patient_id','model','variant','fold','SRIS_phase2','P2_E_D_to_R','P2_E_D_to_C','P2_E_R_to_C']
    full = p2_base[p2_keep].merge(p1_surv, on='patient_id', how='left')

    # Survival cohort summary.
    cohort = p1_surv.dropna(subset=['os_months','deceased']).copy()
    cohort = cohort[cohort['os_months'] > 0]
    cohort_summary = {
        'n_patients': int(len(cohort)),
        'n_events': int(cohort['deceased'].sum()),
        'n_censored': int((1 - cohort['deceased']).sum()),
        'median_os_months': float(cohort['os_months'].median()),
        'mean_os_months': float(cohort['os_months'].mean()),
        'median_age': float(cohort['age'].median()),
        'events_fraction': float(cohort['deceased'].mean()),
    }

    clinical = ['age','grade','purity_low','kps_low']
    molecular = clinical + ['idh_mutant','mgmt_methylated','egfr_amp']
    p1_sris = molecular + ['SRIS_phase1']
    p1_edges = molecular + ['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']
    p1_all = molecular + ['SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']
    no_grade_base = ['age','purity_low','kps_low','idh_mutant','mgmt_methylated','egfr_amp']
    no_idh_base = ['age','grade','purity_low','kps_low','mgmt_methylated','egfr_amp']
    strict_base = ['age','purity_low','kps_low','mgmt_methylated','egfr_amp']

    # Define model list. For Phase 2, focus on all trained/negative control maps.
    model_defs = [
        ('Clinical', p1_surv, clinical, 'baseline'),
        ('Clinical + molecular', p1_surv, molecular, 'baseline'),
        ('Clinical + molecular + Phase1 SRIS', p1_surv, p1_sris, 'phase1_sris'),
        ('Clinical + molecular + Phase1 edges', p1_surv, p1_edges, 'phase1_edges'),
        ('Clinical + molecular + Phase1 SRIS+edges', p1_surv, p1_all, 'phase1_all'),
        ('No-grade baseline', p1_surv, no_grade_base, 'leakage_control'),
        ('No-grade + Phase1 edges', p1_surv, no_grade_base + ['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'], 'leakage_control'),
        ('No-IDH baseline', p1_surv, no_idh_base, 'leakage_control'),
        ('No-IDH + Phase1 edges', p1_surv, no_idh_base + ['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'], 'leakage_control'),
        ('Strict no-grade/no-IDH baseline', p1_surv, strict_base, 'strict_leakage_control'),
        ('Strict no-grade/no-IDH + Phase1 edges', p1_surv, strict_base + ['P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C'], 'strict_leakage_control'),
    ]

    for m in sorted(full['model'].dropna().unique()):
        sub = full[full['model'] == m].drop_duplicates('patient_id').copy()
        # Use only the features from that map, not the fold indicator.
        model_defs.append((f'Clinical + molecular + Phase2 SRIS [{m}]', sub, molecular + ['SRIS_phase2'], 'phase2_sris'))
        model_defs.append((f'Clinical + molecular + Phase2 edges [{m}]', sub, molecular + ['P2_E_D_to_R','P2_E_D_to_C','P2_E_R_to_C'], 'phase2_edges'))
        model_defs.append((f'Strict no-grade/no-IDH + Phase2 edges [{m}]', sub, strict_base + ['P2_E_D_to_R','P2_E_D_to_C','P2_E_R_to_C'], 'phase2_strict_edges'))

    coef_tables = []
    summaries = []
    fold_tables = []
    horizon_tables = []
    oof_risks = []
    for name, df, feats, family in model_defs:
        # Drop zero-variance/all-missing features.
        feats2 = []
        for f in feats:
            vals = pd.to_numeric(df[f], errors='coerce') if f in df.columns else pd.Series(dtype=float)
            if f in df.columns and vals.notna().sum() > 3 and vals.nunique(dropna=True) > 1:
                feats2.append(f)
        if len(feats2) == 0:
            continue
        coefs, summ, risk_full, keep_full = full_sample_cox(df, feats2, name)
        summ['model_family'] = family
        summaries.append(summ)
        coef_tables.append(coefs)
        keep_cv, risk_cv, folds, cv_summ = cross_validated_risk(df, feats2, n_splits=5, seed=22)
        cv_summ.update({'model': name, 'model_family': family, 'n': int(len(keep_cv)), 'events': int(keep_cv['deceased'].sum()), 'n_features': len(feats2)})
        # Merge CV metrics into summary record by appending separately.
        summaries[-1].update(cv_summ)
        folds['model'] = name
        fold_tables.append(folds)
        for h in [24.0, 60.0]:
            hm = horizon_metrics(keep_cv['os_months'].values, keep_cv['deceased'].values, risk_cv, h)
            hm.update({'model': name, 'model_family': family})
            horizon_tables.append(hm)
        tmp = keep_cv[['patient_id','os_months','deceased']].copy() if 'patient_id' in keep_cv.columns else keep_cv[['os_months','deceased']].copy()
        if 'patient_id' not in tmp.columns and 'patient_id' in df.columns:
            tmp['patient_id'] = df.loc[keep_cv.index, 'patient_id'].values
        tmp['model'] = name
        tmp['risk_oof'] = risk_cv
        oof_risks.append(tmp)

    coef_df = pd.concat(coef_tables, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    fold_df = pd.concat(fold_tables, ignore_index=True)
    horizon_df = pd.DataFrame(horizon_tables)
    oof_df = pd.concat(oof_risks, ignore_index=True)

    # Likelihood ratio comparisons for nested models.
    by_model = {r['model']: r for _, r in summary_df.iterrows()}
    lr_rows = []
    comparisons = [
        ('Clinical + molecular', 'Clinical + molecular + Phase1 SRIS'),
        ('Clinical + molecular', 'Clinical + molecular + Phase1 edges'),
        ('Clinical + molecular', 'Clinical + molecular + Phase1 SRIS+edges'),
        ('No-grade baseline', 'No-grade + Phase1 edges'),
        ('No-IDH baseline', 'No-IDH + Phase1 edges'),
        ('Strict no-grade/no-IDH baseline', 'Strict no-grade/no-IDH + Phase1 edges'),
    ]
    for b, f in comparisons:
        if b in by_model and f in by_model:
            stat, df_, p = likelihood_ratio(dict(by_model[b]), dict(by_model[f]))
            lr_rows.append({'base_model': b, 'full_model': f, 'lr_statistic': stat, 'df': df_, 'p_value': p, 'delta_cv_c_index': float(by_model[f]['cv_c_index'] - by_model[b]['cv_c_index'])})
    for m in sorted(full['model'].dropna().unique()):
        for add in [f'Clinical + molecular + Phase2 SRIS [{m}]', f'Clinical + molecular + Phase2 edges [{m}]']:
            if 'Clinical + molecular' in by_model and add in by_model:
                stat, df_, p = likelihood_ratio(dict(by_model['Clinical + molecular']), dict(by_model[add]))
                lr_rows.append({'base_model': 'Clinical + molecular', 'full_model': add, 'lr_statistic': stat, 'df': df_, 'p_value': p, 'delta_cv_c_index': float(by_model[add]['cv_c_index'] - by_model['Clinical + molecular'])})
    lr_df = pd.DataFrame(lr_rows)

    # Diagnostics: correlations with age and survival time, by residual.
    diag_cols = ['SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C','age','grade','os_months','deceased']
    diag = p1_surv[diag_cols].copy()
    diag_rows = []
    for x in ['SRIS_phase1','P1_E_D_to_R','P1_E_D_to_C','P1_E_R_to_C']:
        for y in ['age','grade','os_months']:
            a = pd.to_numeric(diag[x], errors='coerce')
            b = pd.to_numeric(diag[y], errors='coerce')
            mask = a.notna() & b.notna()
            rho, pval = stats.spearmanr(a[mask], b[mask]) if mask.sum() > 3 else (np.nan, np.nan)
            diag_rows.append({'x': x, 'y': y, 'spearman_rho': float(rho), 'p_value': float(pval), 'n': int(mask.sum())})
    diag_df = pd.DataFrame(diag_rows)

    # Save tables.
    summary_df.sort_values(['model_family','cv_c_index'], ascending=[True, False]).to_csv(results_dir/'phase3_survival_model_summary.csv', index=False)
    coef_df.to_csv(results_dir/'phase3_cox_coefficients.csv', index=False)
    fold_df.to_csv(results_dir/'phase3_cv_fold_metrics.csv', index=False)
    horizon_df.to_csv(results_dir/'phase3_time_horizon_accuracy.csv', index=False)
    lr_df.to_csv(results_dir/'phase3_likelihood_ratio_tests.csv', index=False)
    diag_df.to_csv(results_dir/'phase3_diagnostic_correlations.csv', index=False)
    oof_df.to_csv(results_dir/'phase3_out_of_fold_risks.csv', index=False)
    with open(results_dir/'phase3_cohort_summary.json','w') as f:
        json.dump(cohort_summary, f, indent=2)

    # Make figures.
    top = summary_df.sort_values('cv_c_index', ascending=False).head(14).copy()
    plt.figure(figsize=(10, 6))
    y = np.arange(len(top))
    plt.barh(y, top['cv_c_index'].values)
    plt.yticks(y, top['model'].str.replace('Clinical \+ molecular \+ ', '', regex=True).str.slice(0, 55))
    plt.xlabel('5-fold out-of-fold Harrell C-index')
    plt.title('Phase 3 survival discrimination: top model families')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures_dir/'phase3_cindex_comparison.png', dpi=220)
    plt.close()

    # Hazard ratios for Phase1 molecular + edges model.
    edge_model = 'Clinical + molecular + Phase1 edges'
    subcoef = coef_df[coef_df['model'] == edge_model].copy()
    subcoef = subcoef.sort_values('hazard_ratio_per_sd')
    plt.figure(figsize=(8, max(4, 0.35*len(subcoef))))
    y = np.arange(len(subcoef))
    plt.errorbar(subcoef['hazard_ratio_per_sd'], y, xerr=[subcoef['hazard_ratio_per_sd']-subcoef['hr_ci_low'], subcoef['hr_ci_high']-subcoef['hazard_ratio_per_sd']], fmt='o')
    plt.axvline(1.0, linestyle='--', linewidth=1)
    plt.yticks(y, subcoef['feature'])
    plt.xlabel('Hazard ratio per 1 SD')
    plt.title('Cox coefficients: molecular baseline + Phase 1 sheaf edges')
    plt.tight_layout()
    plt.savefig(figures_dir/'phase3_phase1_edge_hazard_ratios.png', dpi=220)
    plt.close()

    # KM-like plot using out-of-fold risk from the best model.
    best_model = summary_df.sort_values('cv_c_index', ascending=False).iloc[0]['model']
    br = oof_df[oof_df['model'] == best_model].copy()
    br = br.dropna(subset=['risk_oof','os_months','deceased'])
    br['high_risk'] = (br['risk_oof'] >= br['risk_oof'].median()).astype(int)
    stat_lr, p_lr = logrank_two_groups(br['os_months'].values, br['deceased'].values.astype(int), br['high_risk'].values)
    plt.figure(figsize=(7, 5))
    for g, label in [(0, 'Low predicted risk'), (1, 'High predicted risk')]:
        sub = br[br['high_risk'] == g]
        t, s = simple_km(sub['os_months'].values, sub['deceased'].values.astype(int))
        plt.step(t, s, where='post', label=f'{label} (n={len(sub)})')
    plt.xlabel('Overall survival time (months)')
    plt.ylabel('Estimated survival probability')
    plt.title(f'OOF risk stratification: {best_model[:60]}\nlog-rank p={p_lr:.3g}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir/'phase3_oof_kaplan_meier_best_model.png', dpi=220)
    plt.close()

    # Horizon AUROC chart for selected interpretable models.
    selected = ['Clinical + molecular', 'Clinical + molecular + Phase1 edges', 'Strict no-grade/no-IDH baseline', 'Strict no-grade/no-IDH + Phase1 edges']
    hsel = horizon_df[(horizon_df['model'].isin(selected)) & (horizon_df['horizon_months'].isin([24.0,60.0]))]
    if len(hsel):
        labels = [f"{r['model']} @ {int(r['horizon_months'])}m" for _, r in hsel.iterrows()]
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(hsel)), hsel['auroc'].values)
        plt.xticks(np.arange(len(hsel)), labels, rotation=60, ha='right')
        plt.ylabel('Out-of-fold horizon AUROC')
        plt.title('Time-horizon classification from Cox risk scores')
        plt.tight_layout()
        plt.savefig(figures_dir/'phase3_horizon_auroc.png', dpi=220)
        plt.close()

    # Write compact report markdown.
    best = summary_df.sort_values('cv_c_index', ascending=False).iloc[0].to_dict()
    mol = by_model.get('Clinical + molecular', {})
    p1e = by_model.get('Clinical + molecular + Phase1 edges', {})
    report = f"""# Phase 3 Survival Validation Report

## Cohort

- Patients used: {cohort_summary['n_patients']}
- Observed deaths/events: {cohort_summary['n_events']}
- Censored/living observations: {cohort_summary['n_censored']}
- Median OS: {cohort_summary['median_os_months']:.2f} months
- Median age: {cohort_summary['median_age']:.1f} years

## Main survival endpoint

The primary endpoint is right-censored overall survival, using OS months as the time variable and death status as the event indicator. The model score is the Cox linear predictor, where higher score means higher estimated hazard.

## Best cross-validated model

- Best model: `{best['model']}`
- 5-fold out-of-fold C-index: {best['cv_c_index']:.4f} [{best['cv_c_index_ci_low']:.4f}, {best['cv_c_index_ci_high']:.4f}]

## Interpretable Phase 1 sheaf increment

- Clinical + molecular CV C-index: {mol.get('cv_c_index', np.nan):.4f}
- Clinical + molecular + Phase 1 edges CV C-index: {p1e.get('cv_c_index', np.nan):.4f}
- Delta: {(p1e.get('cv_c_index', np.nan) - mol.get('cv_c_index', np.nan)):.4f}

## Interpretation caution

The result should be read as internal TCGA-like cohort validation. It is not yet an external TCGA-to-CGGA validation. The strongest next step is to repeat this entire script on CGGA or a held-out cohort and compare whether edge-level sheaf hazards retain the same direction and rank.
"""
    with open(results_dir/'phase3_survival_report.md','w') as f:
        f.write(report)

    summary = {
        'cohort': cohort_summary,
        'best_model': best,
        'clinical_molecular_cv_c_index': float(mol.get('cv_c_index', np.nan)),
        'phase1_edge_cv_c_index': float(p1e.get('cv_c_index', np.nan)),
        'phase1_edge_delta_cv_c_index': float(p1e.get('cv_c_index', np.nan) - mol.get('cv_c_index', np.nan)),
        'output_files': {
            'model_summary': str(results_dir/'phase3_survival_model_summary.csv'),
            'cox_coefficients': str(results_dir/'phase3_cox_coefficients.csv'),
            'horizon_accuracy': str(results_dir/'phase3_time_horizon_accuracy.csv'),
            'likelihood_ratio_tests': str(results_dir/'phase3_likelihood_ratio_tests.csv'),
        }
    }
    with open(results_dir/'phase3_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--figures_dir', default='figures')
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    summary = make_phase3_tables(Path(args.data_dir), Path(args.results_dir), Path(args.figures_dir))
    print(json.dumps(summary, indent=2))
'''

(SRC/'phase3_survival_analysis.py').write_text(PHASE3_CODE)
(SRC/'run_phase3.py').write_text("""#!/usr/bin/env python3\nfrom pathlib import Path\nfrom phase3_survival_analysis import make_phase3_tables\n\nif __name__ == '__main__':\n    root = Path(__file__).resolve().parents[1]\n    make_phase3_tables(root/'data', root/'results', root/'figures')\n""")
os.chmod(SRC/'run_phase3.py', 0o755)
os.chmod(SRC/'phase3_survival_analysis.py', 0o755)

# README and requirements
(ROOT/'requirements.txt').write_text("""numpy\npandas\nscipy\nscikit-learn\nstatsmodels\nmatplotlib\n""")
(ROOT/'README.md').write_text("""# Phase 3 Survival and Clinical Validation Package\n\nThis package validates the Phase 1/2 sheaf residuals against right-censored overall survival.\n\n## Run\n\n```bash\ncd phase3_survival_package\npython src/run_phase3.py\n```\n\n## Main outputs\n\n- `results/phase3_survival_model_summary.csv`\n- `results/phase3_cox_coefficients.csv`\n- `results/phase3_time_horizon_accuracy.csv`\n- `results/phase3_likelihood_ratio_tests.csv`\n- `figures/phase3_cindex_comparison.png`\n- `figures/phase3_phase1_edge_hazard_ratios.png`\n- `figures/phase3_oof_kaplan_meier_best_model.png`\n- `paper/phase3_mathematics_only.pdf`\n\n## Methodological note\n\nSRIS and edge energies are treated as tumor-derived predictors. Age is never part of the inconsistency score; age is included only as an adjustment covariate or external variable.\n""")

# Run the package
import subprocess, sys
subprocess.run([sys.executable, str(SRC/'run_phase3.py')], cwd=str(ROOT), check=True)

# Load summary for LaTeX content
summary = json.loads((RES/'phase3_summary.json').read_text())
model_summary = pd.read_csv(RES/'phase3_survival_model_summary.csv')
coef = pd.read_csv(RES/'phase3_cox_coefficients.csv')
lr = pd.read_csv(RES/'phase3_likelihood_ratio_tests.csv')
horizon = pd.read_csv(RES/'phase3_time_horizon_accuracy.csv')
diag = pd.read_csv(RES/'phase3_diagnostic_correlations.csv')

# Select important rows for report
key_models = ['Clinical','Clinical + molecular','Clinical + molecular + Phase1 SRIS','Clinical + molecular + Phase1 edges','Strict no-grade/no-IDH baseline','Strict no-grade/no-IDH + Phase1 edges']
key_df = model_summary[model_summary['model'].isin(key_models)][['model','n','events','n_features','cv_c_index','cv_c_index_ci_low','cv_c_index_ci_high','in_sample_c_index']].copy()
key_df = key_df.sort_values('cv_c_index', ascending=False)

# Build LaTeX math-only document
tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,booktabs,array,longtable,graphicx,hyperref,xcolor}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}
\title{Phase 3 Mathematical Specification: Survival Validation of Sheaf Regulatory Inconsistency}
\author{Sheaf-Theoretic Multi-Omics Brain Tumor Project}
\date{May 2026}

\begin{document}
\maketitle

\section*{Purpose}
Phase 3 validates whether the sheaf-derived inconsistency variables from Phases 1--2 have prognostic value for right-censored overall survival. The central methodological rule is that age is not part of the sheaf energy. Age is treated only as an external adjustment covariate. Thus the analysis separates tumor-derived inconsistency from patient-level demographic risk.

\section{Survival Data}
For each patient $i\in\{1,\dots,n\}$ let
\[
T_i\in\mathbb{R}_{>0}
\]
be the observed survival or censoring time in months, and let
\[
\delta_i\in\{0,1\}
\]
be the event indicator, where $\delta_i=1$ denotes death and $\delta_i=0$ denotes censoring. In the current Phase 3 run,
\[
 n=%d,\qquad \sum_{i=1}^n \delta_i=%d.
\]
The median observed survival/censoring time is %.2f months.

\section{Sheaf Predictors}
Phase 1 defines three biological stalks
\[
D=\text{DNA/genomic state},\qquad R=\text{regulatory state},\qquad C=\text{tumor phenotype state}.
\]
The Phase 1 edge energies are
\[
E_{DR}(i)=\|r_{DR}(i)\|_2^2,\qquad
E_{DC}(i)=\|r_{DC}(i)\|_2^2,\qquad
E_{RC}(i)=\|r_{RC}(i)\|_2^2.
\]
The total fixed sheaf inconsistency is
\[
\operatorname{SRIS}_1(i)=E_{DR}(i)+E_{DC}(i)+E_{RC}(i).
\]
Phase 2 learned maps replace fixed restrictions by learned maps
\[
r_{uv}^{(2)}(i)=W_{uv}x_u(i)-x_v(i),
\]
with learned energy
\[
\operatorname{SRIS}_2(i)=\sum_{u\to v}\frac{1}{d_v}\|W_{uv}x_u(i)-x_v(i)\|_2^2.
\]

\section{Cox Proportional Hazards Validation}
For a feature vector $z_i\in\mathbb{R}^p$, the Cox model is
\[
h(t\mid z_i)=h_0(t)\exp(\eta_i),
\qquad
\eta_i=\beta^\top z_i.
\]
The partial likelihood is
\[
L(\beta)=\prod_{i:\delta_i=1}
\frac{\exp(\beta^\top z_i)}{\sum_{j:T_j\ge T_i}\exp(\beta^\top z_j)}.
\]
All full-cohort hazard ratios are reported per one standard deviation increase in the covariate:
\[
\operatorname{HR}_k=\exp(\beta_k).
\]
The main adjusted model has the form
\[
\eta_i=eta_a\operatorname{Age}_i+\beta_g\operatorname{Grade}_i+
\beta_p\operatorname{PurityLow}_i+\beta_k\operatorname{KPSLow}_i+
\beta_m^\top M_i+\gamma^\top S_i,
\]
where $M_i$ contains molecular covariates such as IDH, MGMT, and EGFR, and $S_i$ contains sheaf residual variables.

\section{Out-of-Fold Concordance}
Predictive performance is measured by five-fold out-of-fold Harrell concordance. A pair $(i,j)$ is comparable if patient $i$ has an observed event before the observed/censored time of patient $j$:
\[
T_i<T_j,\qquad \delta_i=1.
\]
For risk scores $\hat\eta_i$, the concordance estimate is
\[
\widehat C=
\frac{
\sum_{i,j}\mathbf{1}\{T_i<T_j,\delta_i=1\}
\left[\mathbf{1}\{\hat\eta_i>\hat\eta_j\}+\frac12\mathbf{1}\{\hat\eta_i=\hat\eta_j\}\right]
}{
\sum_{i,j}\mathbf{1}\{T_i<T_j,\delta_i=1\}
}.
\]
Higher $\hat\eta_i$ denotes higher estimated hazard.

\section{Nested Model Improvement}
For nested models $\mathcal M_0\subset\mathcal M_1$, the likelihood-ratio statistic is
\[
\Lambda=2\{\ell(\widehat\beta_1)-\ell(\widehat\beta_0)\},
\]
with approximate null distribution
\[
\Lambda\sim \chi^2_{p_1-p_0}.
\]
This tests whether sheaf residuals add survival information beyond the baseline covariates.

\section{Time-Horizon Accuracy}
For a horizon $\tau\in\{24,60\}$ months, define cases and controls by
\[
Y_i(\tau)=1\quad \Longleftrightarrow\quad T_i\le \tau,\;\delta_i=1,
\]
while controls satisfy $T_i>\tau$. Censored observations with $T_i\le\tau$ are excluded from the binary horizon evaluation because their event status at $\tau$ is unknown. The out-of-fold Cox score $\hat\eta_i$ is then used to compute AUROC, AUPRC, balanced accuracy, and F1.

\section{Key Survival Discrimination Table}
\begin{longtable}{p{0.48\textwidth}rrrr}
\toprule
Model & $n$ & Events & Features & CV C-index\\
\midrule
''' % (summary['cohort']['n_patients'], summary['cohort']['n_events'], summary['cohort']['median_os_months'])

for _, r in key_df.iterrows():
    tex += f"{r['model'].replace('&','\\&')} & {int(r['n'])} & {int(r['events'])} & {int(r['n_features'])} & {r['cv_c_index']:.4f}\\\\\n"
tex += r'''\bottomrule
\end{longtable}

\section{Interpretation Rule}
A sheaf variable is considered survival-informative only if it improves out-of-fold concordance or time-horizon accuracy relative to a matched baseline and, in the full Cox model, has a stable hazard direction after adjustment for age, grade, IDH, MGMT, EGFR, purity, and KPS where available. This prevents the sheaf score from being interpreted as novel if it merely recovers age, grade, or IDH.

\section{Main Phase 3 Numerical Findings}
The best cross-validated model in the current internal run is
\[
\texttt{%s},
\]
with
\[
\widehat C_{\text{OOF}}=%.4f.
\]
The clinical+molecular baseline has
\[
\widehat C_{\text{OOF}}=%.4f,
\]
while the clinical+molecular+Phase 1 edge model has
\[
\widehat C_{\text{OOF}}=%.4f.
\]
Thus the observed internal increment for the interpretable Phase 1 edge residuals is
\[
\Delta \widehat C=%.4f.
\]
This is an internal validation result, not yet an external-cohort claim.

\section{Deliverables}
The Phase 3 package exports all model summaries, coefficient tables, out-of-fold risks, time-horizon accuracy metrics, likelihood-ratio tests, diagnostic correlations, and figures. The correct next scientific step is external validation on CGGA or another held-out glioma cohort.

\end{document}
''' % (summary['best_model']['model'].replace('_','\_').replace('&','\&'), summary['best_model']['cv_c_index'], summary['clinical_molecular_cv_c_index'], summary['phase1_edge_cv_c_index'], summary['phase1_edge_delta_cv_c_index'])

(PAPER/'phase3_mathematics_only.tex').write_text(tex)

# Compile LaTeX
subprocess.run(['pdflatex','-interaction=nonstopmode','phase3_mathematics_only.tex'], cwd=str(PAPER), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase3_mathematics_only.tex'], cwd=str(PAPER), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
shutil.copy(PAPER/'phase3_mathematics_only.pdf', '/mnt/data/phase3_mathematics_only.pdf')
shutil.copy(PAPER/'phase3_mathematics_only.tex', '/mnt/data/phase3_mathematics_only.tex')

# Render PDF for QA
render_dir = Path('/mnt/data/phase3_math_renders')
if render_dir.exists():
    shutil.rmtree(render_dir)
render_dir.mkdir()
subprocess.run(['python','/home/oai/skills/pdfs/scripts/render_pdf.py', str(PAPER/'phase3_mathematics_only.pdf'), '--out_dir', str(render_dir), '--dpi', '160'], check=True)

# Contact sheet
imgs = sorted(render_dir.glob('*.png'))
from PIL import Image, ImageOps, ImageDraw
thumbs=[]
for im_path in imgs:
    im=Image.open(im_path).convert('RGB')
    im.thumbnail((350, 480))
    canvas=Image.new('RGB',(370,520),'white')
    canvas.paste(im,((370-im.width)//2,10))
    d=ImageDraw.Draw(canvas)
    d.text((10,500),im_path.stem,fill=(0,0,0))
    thumbs.append(canvas)
cols=2
rows=math.ceil(len(thumbs)/cols)
sheet=Image.new('RGB',(cols*370,rows*520),'white')
for i,t in enumerate(thumbs):
    sheet.paste(t,((i%cols)*370,(i//cols)*520))
sheet.save('/mnt/data/phase3_math_contact_sheet.png')

# Zip package
zip_path = Path('/mnt/data/phase3_survival_package.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in ROOT.rglob('*'):
        if f.is_file():
            z.write(f, f.relative_to(ROOT.parent))

print(json.dumps(summary, indent=2))
print('Created', zip_path, PAPER/'phase3_mathematics_only.pdf')
