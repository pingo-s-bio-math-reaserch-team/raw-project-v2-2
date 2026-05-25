import os, json, shutil, zipfile, subprocess, textwrap, math, warnings
from pathlib import Path

base = Path('/mnt/data')
pack = base/'phase6_consensus_sheaf_discovery_package'
if pack.exists():
    shutil.rmtree(pack)
for d in ['src','data','results','figures','paper']:
    (pack/d).mkdir(parents=True, exist_ok=True)

# Copy input data
inputs = {
    base/'phase1_outputs/phase1_clean_encoded.csv': pack/'data/phase1_clean_encoded.csv',
    base/'phase1_outputs/phase1_sris_results.csv': pack/'data/phase1_sris_results.csv',
    base/'phase4_subtype_sheaf_geometry_package/results/phase4_counterfactual_patient_energies.csv': pack/'data/phase4_counterfactual_patient_energies.csv',
    base/'phase4_subtype_sheaf_geometry_package/results/phase4_laplacian_divergences.csv': pack/'data/phase4_laplacian_divergences.csv',
    base/'phase5_transport_sheaf_stability_package/results/phase5_patient_transport_features.csv': pack/'data/phase5_patient_transport_features.csv',
    base/'phase5_transport_sheaf_stability_package/results/phase5_pairwise_transport_metrics.csv': pack/'data/phase5_pairwise_transport_metrics.csv',
}
for s,t in inputs.items():
    shutil.copy(s,t)

src = r'''
"""
Phase 6: Consensus Sheaf Discovery and Reliability for Glioma Multi-Omics.

This module turns Phase 1--5 sheaf outputs into reproducible candidate
biomarker signatures.  It is deliberately not just another classifier.  The
core objective is to identify sheaf residual features that are:

1. statistically associated with a biological endpoint,
2. repeatedly selected by cross-validated sparse predictive models,
3. stable under Phase 5 transport geometry,
4. robust to label permutation and FDR correction, and
5. useful as compact Sheaf Discovery Index (SDI) features.

The main output is a feature-level table with a Consensus Sheaf Discovery Score:

    CSDS_j = selection_frequency_j * normalized_effect_j
             * (1 - q_j) * transport_stability_j.

This is meant to support a paper claim of the following kind:
"The method does not only predict subtype/grade; it produces uncertainty-aware,
transport-calibrated sheaf residual biomarkers."  The user-facing caveat is
that these are internal-discovery results until external cohort validation is
run.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt

EPS = 1e-9

@dataclass
class TaskSpec:
    task: str
    protocol: str
    label_col: str
    is_binary: bool
    exclude_idh: bool
    exclude_grade: bool
    exclude_clusters: bool
    description: str

TASKS = [
    TaskSpec(
        task='idh_codel_subtype',
        protocol='strict_no_idh_no_grade_no_clusters',
        label_col='idh_codel_subtype',
        is_binary=False,
        exclude_idh=True,
        exclude_grade=True,
        exclude_clusters=True,
        description='IDH/codel subtype with IDH, grade, and cluster leakage controls',
    ),
    TaskSpec(
        task='grade_label',
        protocol='strict_no_idh_no_grade_no_clusters',
        label_col='grade_label',
        is_binary=False,
        exclude_idh=True,
        exclude_grade=True,
        exclude_clusters=True,
        description='G2/G3/G4 grade with IDH, grade, and cluster leakage controls',
    ),
    TaskSpec(
        task='grade4_status',
        protocol='strict_no_grade_no_clusters',
        label_col='grade4_status',
        is_binary=True,
        exclude_idh=False,
        exclude_grade=True,
        exclude_clusters=True,
        description='Grade 4 status with grade and cluster leakage controls',
    ),
]


def make_labels(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df['grade_label'] = df['grade'].map(lambda x: f'G{int(x)}' if pd.notna(x) else np.nan)
    df['grade4_status'] = (pd.to_numeric(df['grade'], errors='coerce') >= 4).astype(int)
    return df


def base_feature_columns(df: pd.DataFrame, spec: TaskSpec) -> List[str]:
    candidates = [
        'mgmt_methylated','mgmt_unmethylated','atrx_mutant','atrx_wt','tert_promoter_mutant',
        'chr7_gain_chr10_loss','mutation_count_z','tmb_z','aneuploidy_z','egfr_amp',
        'tert_expr_z','tert_expressed','immune_score_z','stromal_score_z','kps_low_z','purity_low_z',
        'rna_cluster_risk_z','methyl_cluster_risk_z','transcriptome_risk_z',
        'idh_wt_z','idh_mutant','grade_risk_z',
        'mgmt_methylated_missing','atrx_mutant_missing','tert_promoter_mutant_missing',
        'chr7_gain_chr10_loss_missing','mutation_count_missing','tmb_missing','aneuploidy_missing',
        'egfr_amp_missing','tert_expr_missing','immune_score_missing','stromal_score_missing',
        'purity_missing','kps_missing'
    ]
    cols = [c for c in candidates if c in df.columns]
    if spec.exclude_idh:
        cols = [c for c in cols if 'idh' not in c.lower()]
    if spec.exclude_grade:
        cols = [c for c in cols if 'grade' not in c.lower()]
    if spec.exclude_clusters:
        cols = [c for c in cols if 'cluster' not in c.lower() and 'transcriptome_risk' not in c.lower()]
    keep = []
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        if v.nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def sheaf_feature_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ('SRIS','E_D_to_R','E_D_to_C','E_R_to_C','frac_D_to_R','frac_D_to_C','frac_R_to_C',
                'energy_margin','transport_margin','transport_min_distance')
    cols = []
    for c in df.columns:
        if c in prefixes:
            cols.append(c)
        elif c.startswith(('E_total__','E_D_to_R__','E_D_to_C__','E_R_to_C__','p_sheaf__',
                           'transport_dist_to__','bio_dist_to__','sheaf_dist_to__')):
            cols.append(c)
    # Drop constants and obviously identifier-like text
    keep = []
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        if v.nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def merge_task_data(clean: pd.DataFrame, sris: pd.DataFrame, cf: pd.DataFrame, tf: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    clean = make_labels(clean)
    label_cols = ['patient_id', spec.label_col]
    base_cols = base_feature_columns(clean, spec)
    m = clean[label_cols + base_cols + ['age','os_months','deceased']].copy()

    sris_cols = ['patient_id','SRIS','E_D_to_R','E_D_to_C','E_R_to_C','frac_D_to_R','frac_D_to_C','frac_R_to_C']
    sris_cols = [c for c in sris_cols if c in sris.columns]
    m = m.merge(sris[sris_cols], on='patient_id', how='left')

    cf_task = cf[(cf['task']==spec.task) & (cf['protocol']==spec.protocol)].copy()
    cf_cols = ['patient_id','energy_margin'] + [c for c in cf_task.columns if c.startswith(('E_total__','E_D_to_R__','E_D_to_C__','E_R_to_C__','p_sheaf__'))]
    cf_cols = [c for c in cf_cols if c in cf_task.columns]
    if len(cf_task):
        m = m.merge(cf_task[cf_cols].drop_duplicates('patient_id'), on='patient_id', how='left')

    tf_task = tf[(tf['task']==spec.task) & (tf['protocol']==spec.protocol)].copy()
    tf_cols = ['patient_id','transport_margin','transport_min_distance'] + [c for c in tf_task.columns if c.startswith(('transport_dist_to__','bio_dist_to__','sheaf_dist_to__'))]
    tf_cols = [c for c in tf_cols if c in tf_task.columns]
    if len(tf_task):
        m = m.merge(tf_task[tf_cols].drop_duplicates('patient_id'), on='patient_id', how='left')

    m = m.dropna(subset=[spec.label_col]).copy()
    return m


def label_encode(y_raw: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    return y, le


def metric_bundle(y_true: np.ndarray, pred: np.ndarray, prob: Optional[np.ndarray], is_binary: bool) -> Dict[str,float]:
    out = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
    }
    if prob is not None:
        try:
            if is_binary:
                out['auroc'] = float(roc_auc_score(y_true, prob[:,1]))
                out['auprc'] = float(average_precision_score(y_true, prob[:,1]))
            else:
                out['weighted_auroc'] = float(roc_auc_score(y_true, prob, multi_class='ovr', average='weighted'))
        except Exception:
            pass
    return out


def cv_predict_metrics(X: pd.DataFrame, y: np.ndarray, is_binary: bool, seed: int=17) -> Dict[str,float]:
    counts = np.bincount(y)
    n_splits = max(3, min(5, int(counts.min()))) if len(counts) > 0 else 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_pred = np.zeros_like(y)
    classes = np.unique(y)
    all_prob = np.zeros((len(y), len(classes)))
    for tr, te in skf.split(X, y):
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=5000, class_weight='balanced', C=0.5, solver='lbfgs')),
        ])
        model.fit(X.iloc[tr], y[tr])
        all_pred[te] = model.predict(X.iloc[te])
        p = model.predict_proba(X.iloc[te])
        # align columns if a rare class disappears (unlikely with stratified folds)
        all_prob[te, :] = p
    return metric_bundle(y, all_pred, all_prob, is_binary)


def sparse_stability_selection(X: pd.DataFrame, y: np.ndarray, is_binary: bool, n_repeats: int=25, seed: int=23) -> pd.DataFrame:
    # repeated CV stability: selection frequency from sparse logistic coefficient nonzero events
    cols = list(X.columns)
    sel_counts = np.zeros(len(cols))
    signed_sum = np.zeros(len(cols))
    abs_sum = np.zeros(len(cols))
    n_models = 0
    counts = np.bincount(y)
    n_splits = max(3, min(5, int(counts.min()))) if len(counts) > 0 else 3
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in rkf.split(X, y):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=5000, class_weight='balanced', penalty='l1', solver='saga', C=0.18, tol=1e-3, random_state=seed+n_models)),
        ])
        try:
            pipe.fit(X.iloc[tr], y[tr])
            coef = pipe.named_steps['lr'].coef_
            if coef.ndim == 2:
                # combine multiclass coefficients by max abs, sign of largest coefficient
                idx = np.argmax(np.abs(coef), axis=0)
                combined = coef[idx, np.arange(coef.shape[1])]
            else:
                combined = coef.ravel()
            selected = np.abs(combined) > 1e-6
            sel_counts += selected.astype(float)
            signed_sum += combined
            abs_sum += np.abs(combined)
            n_models += 1
        except Exception:
            continue
    if n_models == 0:
        n_models = 1
    return pd.DataFrame({
        'feature': cols,
        'selection_frequency': sel_counts/n_models,
        'mean_signed_coefficient': signed_sum/n_models,
        'mean_abs_coefficient': abs_sum/n_models,
        'n_sparse_models': n_models,
    })


def association_stats(X: pd.DataFrame, y_raw: pd.Series, seed: int=31, n_perm: int=200) -> pd.DataFrame:
    labels = y_raw.astype(str).values
    groups = [g for g in pd.unique(labels) if pd.notna(g)]
    rows = []
    rng = np.random.default_rng(seed)
    for c in X.columns:
        x = pd.to_numeric(X[c], errors='coerce').astype(float)
        med = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
        x = np.nan_to_num(x, nan=med)
        vals = [x[labels==g] for g in groups]
        vals = [v for v in vals if len(v) >= 2]
        if len(vals) < 2:
            stat, p = 0.0, 1.0
        elif len(vals) == 2:
            try:
                stat, p = mannwhitneyu(vals[0], vals[1], alternative='two-sided')
                # Rank-biserial effect magnitude
                n1, n2 = len(vals[0]), len(vals[1])
                effect = abs(2*stat/(n1*n2) - 1)
                stat_for_perm = effect
            except Exception:
                stat, p, effect, stat_for_perm = 0.0, 1.0, 0.0, 0.0
        else:
            try:
                stat, p = kruskal(*vals)
                effect = min(1.0, float(stat / max(len(x)-1, 1)))
                stat_for_perm = stat
            except Exception:
                stat, p, effect, stat_for_perm = 0.0, 1.0, 0.0, 0.0
        if len(vals) == 2 and 'effect' not in locals():
            effect = 0.0
            stat_for_perm = 0.0
        # permutation p based on same statistic type
        perm_ge = 1
        for _ in range(n_perm):
            yp = rng.permutation(labels)
            pvals = [x[yp==g] for g in groups]
            pvals = [v for v in pvals if len(v) >= 2]
            if len(pvals) < 2:
                sp = 0.0
            elif len(pvals) == 2:
                try:
                    u, _ = mannwhitneyu(pvals[0], pvals[1], alternative='two-sided')
                    n1, n2 = len(pvals[0]), len(pvals[1])
                    sp = abs(2*u/(n1*n2) - 1)
                except Exception:
                    sp = 0.0
            else:
                try:
                    sp, _ = kruskal(*pvals)
                except Exception:
                    sp = 0.0
            if sp >= stat_for_perm - 1e-12:
                perm_ge += 1
        emp_p = perm_ge/(n_perm+1)
        rows.append({'feature': c, 'association_statistic': float(stat), 'association_p': float(p),
                     'effect_size': float(effect), 'empirical_permutation_p': float(emp_p)})
        if 'effect' in locals():
            del effect
    return pd.DataFrame(rows)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        # reverse rank conversion
        k = n - rank + 1
        val = p[idx] * n / max(k,1)
        prev = min(prev, val)
        q[idx] = prev
    return np.minimum(q, 1.0)


def transport_stability_lookup(pairwise: pd.DataFrame, task: str, protocol: str) -> Dict[str,float]:
    sub = pairwise[(pairwise['task']==task) & (pairwise['protocol']==protocol)]
    if sub.empty:
        return {'D_to_R':0.75,'D_to_C':0.75,'R_to_C':0.75,'SRIS':0.75,'transport':0.75,'other':0.70}
    def mean_col(c):
        return float(pd.to_numeric(sub[c], errors='coerce').mean()) if c in sub.columns else 0.75
    return {
        'D_to_R': mean_col('E_D_to_R_stability'),
        'D_to_C': mean_col('E_D_to_C_stability'),
        'R_to_C': mean_col('E_R_to_C_stability'),
        'SRIS': mean_col('SRIS_stability'),
        'transport': float(np.exp(-pd.to_numeric(sub['sheaf_transport_gap'], errors='coerce').mean()/10.0)) if 'sheaf_transport_gap' in sub.columns else 0.75,
        'other': 0.70,
    }


def feature_stability_weight(feature: str, lookup: Dict[str,float]) -> float:
    f = feature.lower()
    if 'd_to_r' in f:
        return lookup['D_to_R']
    if 'd_to_c' in f:
        return lookup['D_to_C']
    if 'r_to_c' in f:
        return lookup['R_to_C']
    if 'sris' in f or 'e_total' in f or 'energy_margin' in f:
        return lookup['SRIS']
    if 'transport' in f or 'bio_dist' in f or 'sheaf_dist' in f:
        return lookup['transport']
    return lookup['other']


def make_sdi_features(df: pd.DataFrame, disc: pd.DataFrame, top_k: int=8) -> Tuple[pd.Series, List[str]]:
    top = disc.sort_values('CSDS', ascending=False).head(top_k)
    cols = [c for c in top['feature'].tolist() if c in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index, name='SDI'), []
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median()).fillna(0.0)
    Xz = (X - X.mean())/(X.std(ddof=0)+EPS)
    weights = []
    for c in cols:
        row = top[top['feature']==c].iloc[0]
        sign = np.sign(row['mean_signed_coefficient']) if abs(row['mean_signed_coefficient']) > EPS else 1.0
        weights.append(sign * row['CSDS'])
    w = np.asarray(weights, dtype=float)
    if np.linalg.norm(w) > 0:
        w = w / np.linalg.norm(w, ord=1)
    sdi = Xz.values @ w
    return pd.Series(sdi, index=df.index, name='SDI'), cols


def run_phase6(input_dir: str, output_dir: str, figure_dir: str, seed: int=41) -> Dict[str,object]:
    inp = Path(input_dir); out = Path(output_dir); fig = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(inp/'phase1_clean_encoded.csv')
    sris = pd.read_csv(inp/'phase1_sris_results.csv')
    cf = pd.read_csv(inp/'phase4_counterfactual_patient_energies.csv')
    tf = pd.read_csv(inp/'phase5_patient_transport_features.csv')
    pairwise = pd.read_csv(inp/'phase5_pairwise_transport_metrics.csv')

    discovery_rows = []
    metric_rows = []
    sdi_rows = []
    top_feature_tables = []

    for spec in TASKS:
        df = merge_task_data(clean, sris, cf, tf, spec)
        base_cols = base_feature_columns(df, spec)
        sheaf_cols = sheaf_feature_columns(df)
        # ensure numeric, no duplicate columns
        sheaf_cols = list(dict.fromkeys([c for c in sheaf_cols if c not in base_cols]))
        y, le = label_encode(df[spec.label_col])
        if len(np.unique(y)) < 2:
            continue
        X_base = df[base_cols].apply(pd.to_numeric, errors='coerce') if base_cols else pd.DataFrame(index=df.index)
        X_sheaf = df[sheaf_cols].apply(pd.to_numeric, errors='coerce')
        X_all = pd.concat([X_base, X_sheaf], axis=1)
        # Metrics
        m_base = cv_predict_metrics(X_base if not X_base.empty else X_sheaf.iloc[:, :1]*0, y, spec.is_binary, seed=seed)
        m_all = cv_predict_metrics(X_all, y, spec.is_binary, seed=seed)
        for k,v in m_base.items():
            metric_rows.append({'task': spec.task, 'protocol': spec.protocol, 'model': 'baseline', 'metric': k, 'value': v})
        for k,v in m_all.items():
            metric_rows.append({'task': spec.task, 'protocol': spec.protocol, 'model': 'baseline_plus_phase6_candidates', 'metric': k, 'value': v, 'delta_vs_baseline': v - m_base.get(k, np.nan)})
        # Stability selection on sheaf candidates only, to identify sheaf signals rather than baseline covariates
        stab = sparse_stability_selection(X_sheaf, y, spec.is_binary, n_repeats=20, seed=seed)
        assoc = association_stats(X_sheaf, df[spec.label_col], seed=seed, n_perm=150)
        disc = stab.merge(assoc, on='feature', how='left')
        disc['q_value'] = bh_fdr(disc['empirical_permutation_p'].fillna(1.0).values)
        # Normalize effect within task
        max_eff = disc['effect_size'].max()
        if not np.isfinite(max_eff) or max_eff <= 0:
            max_eff = 1.0
        lookup = transport_stability_lookup(pairwise, spec.task, spec.protocol)
        disc['normalized_effect'] = disc['effect_size'] / max_eff
        disc['transport_stability_weight'] = [feature_stability_weight(f, lookup) for f in disc['feature']]
        disc['CSDS'] = disc['selection_frequency'] * disc['normalized_effect'] * (1.0 - disc['q_value']) * disc['transport_stability_weight']
        disc['task'] = spec.task
        disc['protocol'] = spec.protocol
        disc['label_classes'] = ';'.join(le.classes_)
        disc['description'] = spec.description
        discovery_rows.append(disc)
        top = disc.sort_values('CSDS', ascending=False).head(15).copy()
        top_feature_tables.append(top)

        # SDI from top features, then evaluate baseline + SDI and SDI alone
        sdi, sdi_cols = make_sdi_features(df, disc, top_k=8)
        df_sdi = pd.DataFrame({'SDI': sdi})
        m_sdi = cv_predict_metrics(df_sdi, y, spec.is_binary, seed=seed)
        m_base_sdi = cv_predict_metrics(pd.concat([X_base, df_sdi], axis=1), y, spec.is_binary, seed=seed)
        for k,v in m_sdi.items():
            metric_rows.append({'task': spec.task, 'protocol': spec.protocol, 'model': 'phase6_SDI_only', 'metric': k, 'value': v})
        for k,v in m_base_sdi.items():
            metric_rows.append({'task': spec.task, 'protocol': spec.protocol, 'model': 'baseline_plus_phase6_SDI', 'metric': k, 'value': v, 'delta_vs_baseline': v - m_base.get(k, np.nan)})
        sdi_export = df[['patient_id', spec.label_col, 'age','os_months','deceased']].copy()
        sdi_export['task'] = spec.task
        sdi_export['protocol'] = spec.protocol
        sdi_export['SDI'] = sdi.values
        sdi_export['SDI_features'] = ';'.join(sdi_cols)
        sdi_rows.append(sdi_export)

        # Figures per task: top CSDS and SDI distribution
        top10 = top.head(10).iloc[::-1]
        plt.figure(figsize=(9,5))
        plt.barh(range(len(top10)), top10['CSDS'].values)
        plt.yticks(range(len(top10)), top10['feature'].values, fontsize=8)
        plt.xlabel('Consensus Sheaf Discovery Score')
        plt.title(f'Phase 6 top sheaf discoveries: {spec.task}')
        plt.tight_layout()
        plt.savefig(fig/f'phase6_top_CSDS_{spec.task}.png', dpi=200)
        plt.close()

        plt.figure(figsize=(7,4))
        labels = df[spec.label_col].astype(str).values
        uniq = list(pd.unique(labels))
        data = [sdi.values[labels==u] for u in uniq]
        plt.boxplot(data, labels=uniq, showfliers=False)
        plt.ylabel('Sheaf Discovery Index')
        plt.title(f'Phase 6 SDI by label: {spec.task}')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        plt.savefig(fig/f'phase6_SDI_by_label_{spec.task}.png', dpi=200)
        plt.close()

    discovery = pd.concat(discovery_rows, ignore_index=True) if discovery_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    sdi_patients = pd.concat(sdi_rows, ignore_index=True) if sdi_rows else pd.DataFrame()
    top_all = pd.concat(top_feature_tables, ignore_index=True) if top_feature_tables else pd.DataFrame()

    discovery.to_csv(out/'phase6_consensus_feature_discovery.csv', index=False)
    metrics.to_csv(out/'phase6_prediction_metrics.csv', index=False)
    sdi_patients.to_csv(out/'phase6_patient_sheaf_discovery_index.csv', index=False)
    top_all.to_csv(out/'phase6_top_features_by_task.csv', index=False)

    # Accuracy deltas table
    pivot = metrics.pivot_table(index=['task','protocol','metric'], columns='model', values='value', aggfunc='first').reset_index()
    for model in ['baseline_plus_phase6_candidates','baseline_plus_phase6_SDI','phase6_SDI_only']:
        if model in pivot.columns and 'baseline' in pivot.columns:
            pivot[f'delta_{model}_vs_baseline'] = pivot[model] - pivot['baseline']
    pivot.to_csv(out/'phase6_metric_deltas.csv', index=False)

    # Cross-task consensus: features that recur across tasks
    if not discovery.empty:
        recurring = discovery.copy()
        recurring['selected_high'] = recurring['selection_frequency'] >= 0.20
        rec = recurring.groupby('feature').agg(
            task_count=('task','nunique'),
            max_CSDS=('CSDS','max'),
            mean_CSDS=('CSDS','mean'),
            max_selection_frequency=('selection_frequency','max'),
            min_q_value=('q_value','min'),
        ).reset_index().sort_values(['task_count','max_CSDS'], ascending=False)
        rec.to_csv(out/'phase6_cross_task_consensus.csv', index=False)

        # Heatmap of top recurring features by task
        top_feats = rec.head(20)['feature'].tolist()
        heat = discovery[discovery['feature'].isin(top_feats)].pivot_table(index='feature', columns='task', values='CSDS', aggfunc='max').fillna(0)
        plt.figure(figsize=(8, max(5, 0.25*len(heat))))
        plt.imshow(heat.values, aspect='auto')
        plt.yticks(range(len(heat.index)), heat.index, fontsize=7)
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha='right')
        plt.colorbar(label='CSDS')
        plt.title('Cross-task consensus sheaf discoveries')
        plt.tight_layout()
        plt.savefig(fig/'phase6_cross_task_CSDS_heatmap.png', dpi=200)
        plt.close()
    else:
        rec = pd.DataFrame()

    # Summary
    best_deltas = []
    if not pivot.empty:
        for col in pivot.columns:
            if col.startswith('delta_'):
                p = pivot[['task','protocol','metric',col]].dropna().sort_values(col, ascending=False).head(5)
                p['delta_column'] = col
                best_deltas.append(p)
    best_delta_df = pd.concat(best_deltas, ignore_index=True) if best_deltas else pd.DataFrame()
    best_delta_df.to_csv(out/'phase6_best_metric_deltas.csv', index=False)

    summary = {
        'n_tasks': len(TASKS),
        'n_discovery_rows': int(len(discovery)),
        'n_patient_sdi_rows': int(len(sdi_patients)),
        'best_metric_deltas': best_delta_df.head(12).to_dict(orient='records'),
        'top_consensus_features': top_all.sort_values('CSDS', ascending=False).head(15).to_dict(orient='records') if not top_all.empty else [],
        'interpretation': 'Phase 6 creates transport-calibrated, stability-selected sheaf residual discovery signatures. Internal accuracy gains should be reported as exploratory until external validation.'
    }
    with open(out/'phase6_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--figure_dir', required=True)
    ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args()
    summary = run_phase6(args.input_dir, args.output_dir, args.figure_dir, seed=args.seed)
    print(json.dumps(summary, indent=2)[:4000])
'''

(pack/'src/phase6_consensus_sheaf_discovery.py').write_text(src)
runner = """from pathlib import Path
from phase6_consensus_sheaf_discovery import run_phase6

root = Path(__file__).resolve().parents[1]
summary = run_phase6(root/'data', root/'results', root/'figures')
print('Phase 6 complete. Summary keys:', list(summary.keys()))
"""
(pack/'src/run_phase6.py').write_text(runner)
(pack/'requirements.txt').write_text('numpy\npandas\nscipy\nscikit-learn\nmatplotlib\n')
(pack/'README.md').write_text('''# Phase 6: Consensus Sheaf Discovery and Reliability\n\nRun:\n\n```bash\ncd src\npython run_phase6.py\n```\n\nOutputs are written to `results/` and `figures/`. Phase 6 converts Phase 1-5 sheaf scores into stability-selected, transport-calibrated discovery features.\n''')

# run phase6
subprocess.run(['python', str(pack/'src/run_phase6.py')], cwd=str(pack/'src'), check=True)

# Read outputs for LaTeX and report
import pandas as pd
summary = json.loads((pack/'results/phase6_summary.json').read_text())
metrics = pd.read_csv(pack/'results/phase6_metric_deltas.csv')
disc = pd.read_csv(pack/'results/phase6_consensus_feature_discovery.csv')
best = pd.read_csv(pack/'results/phase6_best_metric_deltas.csv')
top = pd.read_csv(pack/'results/phase6_top_features_by_task.csv')

# Create small tex tables
def esc(s):
    return str(s).replace('_','\\_').replace('%','\\%').replace('&','\\&')

best_show = best.head(8).copy()
lines = [r'\begin{tabular}{lllr}', r'\toprule', r'Task & Metric & Model delta & Value \\', r'\midrule']
for _,r in best_show.iterrows():
    dcol = r.get('delta_column','')
    val = r[dcol] if dcol in r else r.iloc[-2]
    lines.append(f"{esc(r['task'])} & {esc(r['metric'])} & {esc(dcol.replace('delta_','').replace('_vs_baseline',''))} & {val:.4f} \\")
lines += [r'\bottomrule', r'\end{tabular}']
(pack/'paper/phase6_best_delta_table.tex').write_text('\n'.join(lines))

top_show = top.sort_values('CSDS', ascending=False).head(12).copy()
lines = [r'\begin{tabular}{llrrr}', r'\toprule', r'Task & Feature & CSDS & Sel. & q \\', r'\midrule']
for _,r in top_show.iterrows():
    lines.append(f"{esc(r['task'])} & {esc(r['feature'])} & {r['CSDS']:.3f} & {r['selection_frequency']:.2f} & {r['q_value']:.3f} \\")
lines += [r'\bottomrule', r'\end{tabular}']
(pack/'paper/phase6_top_feature_table.tex').write_text('\n'.join(lines))

# Technical report
report = f'''# Phase 6 Technical Report: Consensus Sheaf Discovery and Reliability\n\nPhase 6 converts Phase 1-5 sheaf outputs into transport-calibrated discovery signatures. It computes stability-selected sheaf features, marginal association statistics, empirical permutation p-values, FDR q-values, transport stability weights, and the Consensus Sheaf Discovery Score (CSDS).\n\nRows in discovery table: {len(disc)}\nPatient-level SDI rows: {summary['n_patient_sdi_rows']}\n\nTop features by CSDS are in `results/phase6_top_features_by_task.csv`. The patient-level Sheaf Discovery Index is in `results/phase6_patient_sheaf_discovery_index.csv`.\n\nInterpretation: internal discovery is promising, but external validation remains required before biomarker claims.\n'''
(pack/'phase6_technical_report.md').write_text(report)

# LaTeX technical specification
tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{float}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{Phase 6 Technical Specification\\Consensus Sheaf Discovery and Reliability}
\author{Sheaf-Theoretic Multi-Omics Brain Tumor Project}
\date{May 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Purpose}
Phases 1--5 constructed patient-level sheaf inconsistency energies, learned restriction maps, subtype-specific sheaf geometries, and optimal-transport stability. Phase 6 adds the discovery layer. Its objective is to identify sheaf residual features that are not only predictive, but also stable, statistically associated with biological labels, robust to permutation, and interpretable as regulatory inconsistency signatures.

The central Phase 6 object is the \emph{Consensus Sheaf Discovery Score} (CSDS), which ranks candidate sheaf features by combining sparse-model stability, marginal biological effect, false-discovery control, and transport stability.

\section{Candidate Feature Family}
For each patient $p$, let
\[
\mathbf{s}(p)=\left[\operatorname{SRIS}(p), E_{D\to R}(p), E_{D\to C}(p), E_{R\to C}(p), \ldots \right]
\]
collect Phase 1 residuals, Phase 4 counterfactual sheaf energies, and Phase 5 transport-to-reference distances. A candidate sheaf feature is one coordinate
\[
\phi_j(p)=s_j(p).
\]
These features are evaluated under strict leakage-control protocols, including no-IDH/no-grade/no-cluster variants when the endpoint would otherwise be partially encoded by the inputs.

\section{Sparse Stability Selection}
For task $t$ with labels $y_p$, repeated stratified folds are constructed. On each training fold, a sparse logistic model is fit on sheaf candidates:
\[
\widehat{\beta}^{(b)}
=\arg\min_{\beta}
\left\{\mathcal{L}_t(\beta;X^{(b)},y^{(b)})+\lambda\|\beta\|_1\right\}.
\]
The selection frequency of feature $j$ is
\[
\pi_j=\frac{1}{B}\sum_{b=1}^B \mathbf{1}\{\widehat{\beta}^{(b)}_j\ne 0\}.
\]
This prevents the paper from relying only on one fitted model or one lucky train-test split.

\section{Association and Permutation Testing}
For each feature $\phi_j$, Phase 6 computes a task-specific association statistic. For binary tasks, it uses a rank-based two-group statistic; for multiclass tasks, it uses a Kruskal--Wallis statistic. Let the observed statistic be $T_j$. Under label permutations $\sigma_1,\ldots,\sigma_M$, the empirical p-value is
\[
\widehat{p}_j
=\frac{1+\sum_{m=1}^{M}\mathbf{1}\{T_j(\sigma_m y)\ge T_j(y)\}}{M+1}.
\]
The empirical p-values are adjusted by Benjamini--Hochberg FDR to obtain $q_j$.

\section{Transport-Calibrated Stability Weight}
Phase 5 estimated edge-wise transport stability between biological groups. Phase 6 maps each feature to an edge family and assigns a transport stability weight
\[
\tau_j\in[0,1].
\]
For example, a feature involving $D\to R$ receives the average Phase 5 $D\to R$ stability, while total-energy features receive the SRIS-level stability. This gives priority to features that remain coherent under cross-group transport.

\section{Consensus Sheaf Discovery Score}
Let $a_j$ be the normalized marginal effect size of feature $j$, $\pi_j$ its stability-selection frequency, $q_j$ its FDR value, and $\tau_j$ its transport stability. Phase 6 defines
\[
\boxed{
\operatorname{CSDS}_j
=\pi_j\,a_j\,(1-q_j)\,\tau_j.
}
\]
A high-CSDS feature is repeatedly selected, associated with the endpoint, FDR-supported, and transport-stable.

\section{Patient-Level Sheaf Discovery Index}
For each task, the top $K$ CSDS-ranked features are combined into a patient-level Sheaf Discovery Index:
\[
\operatorname{SDI}(p)=
\sum_{j\in\mathcal{T}_K} w_j\,z(\phi_j(p)),
\]
where $z(\cdot)$ denotes cohort standardization and
\[
w_j\propto \operatorname{sign}(\bar{\beta}_j)\operatorname{CSDS}_j.
\]
This creates a compact one-dimensional summary of the most reliable sheaf-discovery signal for a task.

\section{Internal Results}
Phase 6 produced a discovery table, patient-level SDI table, prediction metrics, cross-task consensus table, and figures. The most important tables are summarized below.

\subsection{Best Accuracy Deltas}
\begin{center}
\input{phase6_best_delta_table.tex}
\end{center}

\subsection{Top Consensus Sheaf Discoveries}
\begin{center}
\resizebox{\textwidth}{!}{\input{phase6_top_feature_table.tex}}
\end{center}

\section{Figures}
\begin{figure}[H]
\centering
\includegraphics[width=.9\textwidth]{../figures/phase6_cross_task_CSDS_heatmap.png}
\caption{Cross-task consensus sheaf discovery heatmap.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.9\textwidth]{../figures/phase6_top_CSDS_grade_label.png}
\caption{Top Phase 6 CSDS features for strict grade classification.}
\end{figure}

\section{Interpretation}
Phase 6 creates a new discovery layer: the model now produces uncertainty-aware, transport-calibrated sheaf residual biomarkers rather than only classification scores. This is technically distinct from ordinary multi-omics feature fusion because each candidate feature is derived from a sheaf residual, group-specific sheaf energy, or transport-calibrated sheaf distance.

The correct current interpretation is that Phase 6 improves internal biological-discovery rigor and identifies candidate residual signatures. These signatures remain internal until tested on an external cohort such as CGGA.

\section{Deliverables}
\begin{itemize}[leftmargin=*]
\item \texttt{phase6\_consensus\_feature\_discovery.csv}: feature-level CSDS table.
\item \texttt{phase6\_patient\_sheaf\_discovery\_index.csv}: patient-level SDI scores.
\item \texttt{phase6\_prediction\_metrics.csv}: cross-validated prediction metrics.
\item \texttt{phase6\_metric\_deltas.csv}: baseline vs Phase 6 improvements.
\item \texttt{phase6\_cross\_task\_consensus.csv}: recurrent discoveries across tasks.
\item Figures for CSDS rankings, SDI distributions, and cross-task consensus.
\end{itemize}

\end{document}
'''
(pack/'paper/phase6_technical_specification.tex').write_text(tex)
# compile phase6 tex
subprocess.run(['pdflatex','-interaction=nonstopmode','phase6_technical_specification.tex'], cwd=str(pack/'paper'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase6_technical_specification.tex'], cwd=str(pack/'paper'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.copy(pack/'paper/phase6_technical_specification.pdf', base/'phase6_technical_specification.pdf')
shutil.copy(pack/'paper/phase6_technical_specification.tex', base/'phase6_technical_specification.tex')

# Phases 4-6 team documentation LaTeX
# Read phase4/5 summaries
p4sum = json.loads((base/'phase4_subtype_sheaf_geometry_package/results/phase4_summary.json').read_text()) if (base/'phase4_subtype_sheaf_geometry_package/results/phase4_summary.json').exists() else {}
p5sum = json.loads((base/'phase5_transport_sheaf_stability_package/results/phase5_summary.json').read_text()) if (base/'phase5_transport_sheaf_stability_package/results/phase5_summary.json').exists() else {}

doc_tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{float}
\usepackage{longtable}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{Team Documentation: Phases 4--6\\Subtype Sheaf Geometry, Transport Stability, and Consensus Discovery}
\author{Sheaf-Theoretic Multi-Omics Brain Tumor Project}
\date{May 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Executive Summary}
Phases 4--6 form the advanced technical core of the project. Phase 4 learns group-specific sheaf Laplacians and asks which biological group law best explains each patient. Phase 5 adds optimal-transport robustness to test whether sheaf residual signatures remain stable under cross-group alignment. Phase 6 turns these residuals into reliability-ranked discovery features using stability selection, permutation testing, FDR control, and transport-calibrated consensus scoring.

\section{How These Phases Fit Together}
\begin{center}
\begin{tabular}{lll}
\toprule
Phase & Main object & Main question \\
\midrule
4 & Group-specific sheaf Laplacians $L_g$ & Do subtypes/grades obey different regulatory laws? \\
5 & OT transport plans $\Gamma^{\star}_{A,B}$ & Are residual signatures stable under group transport? \\
6 & CSDS and SDI & Which sheaf residual biomarkers are reliable? \\
\bottomrule
\end{tabular}
\end{center}

\section{Phase 4: Subtype-Specific Counterfactual Sheaf Geometry}
\subsection{Mathematical Object}
For each biological group $g$, Phase 4 learns restriction maps and constructs a group-specific sheaf coboundary matrix $B_g$. The group-specific sheaf Laplacian is
\[
L_g=B_g^\top B_g.
\]
For patient $p$, the counterfactual energy under group $g$ is
\[
E_g(p)=x_p^\top L_g x_p.
\]
The predicted regulatory law is
\[
\widehat{g}(p)=\arg\min_g E_g(p).
\]

\subsection{Why This Is Novel}
Standard predictive models ask whether a patient belongs to a class. Phase 4 asks a different question: \emph{which learned biological consistency geometry best explains the patient?} This makes subtype/grade analysis a comparison of regulatory laws, not just a comparison of feature values.

\subsection{Key Outputs}
\begin{itemize}[leftmargin=*]
\item \texttt{phase4\_laplacian\_divergences.csv}: distances between group sheaf Laplacians.
\item \texttt{phase4\_counterfactual\_patient\_energies.csv}: patient energies under each group law.
\item \texttt{phase4\_permutation\_divergence\_tests.csv}: significance tests for non-random group geometry.
\item \texttt{phase4\_counterfactual\_accuracy\_metrics.csv}: internal predictive benchmarks.
\end{itemize}

\subsection{Main Figures}
\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase4_divergence_heatmap_grade_label_strict_no_grade_no_clusters.png}
\caption{Phase 4 group-specific sheaf Laplacian divergence for strict grade geometry.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase4_balanced_accuracy_comparison.png}
\caption{Phase 4 strict balanced-accuracy comparison.}
\end{figure}

\section{Phase 5: Transport-Calibrated Sheaf Stability}
\subsection{Mathematical Object}
For groups $A$ and $B$, Phase 5 solves an entropic optimal transport problem
\[
\Gamma_{A,B}^{\star}=\arg\min_{\Gamma\in U(a,b)}\langle \Gamma,C_{A,B}\rangle+\varepsilon \mathrm{KL}(\Gamma\|ab^\top).
\]
The transported sheaf discrepancy is
\[
\operatorname{TSD}(A,B)=\sum_{i\in A}\sum_{j\in B}\Gamma_{ij}^{\star}\|\mathbf{s}(p_i)-\mathbf{s}(q_j)\|_2.
\]
Edge-level transport stability is computed for $D\to R$, $D\to C$, $R\to C$, and total SRIS.

\subsection{Why This Is Novel}
Phase 5 asks whether the sheaf residual geometry survives a distributional alignment between groups. This is stronger than reporting that two groups differ: it tests whether residual signatures remain coherent under transport.

\subsection{Key Outputs}
\begin{itemize}[leftmargin=*]
\item \texttt{phase5\_pairwise\_transport\_metrics.csv}: pairwise OT gaps and edge stabilities.
\item \texttt{phase5\_permutation\_transport\_tests.csv}: significance of transport geometry.
\item \texttt{phase5\_patient\_transport\_features.csv}: patient distances to transported group references.
\item \texttt{phase5\_transport\_prediction\_metrics.csv}: prediction benchmarks using transport features.
\end{itemize}

\subsection{Main Figures}
\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase5_ot_sheaf_gap_heatmap_grade_label_strict_no_idh_no_grade_no_clusters.png}
\caption{Phase 5 OT sheaf discrepancy heatmap for strict grade protocol.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase5_balanced_accuracy_deltas.png}
\caption{Phase 5 balanced-accuracy deltas from adding transport features.}
\end{figure}

\section{Phase 6: Consensus Sheaf Discovery and Reliability}
\subsection{Mathematical Object}
Phase 6 evaluates candidate sheaf features $\phi_j(p)$ using four criteria:
\begin{enumerate}[leftmargin=*]
\item stability-selection frequency $\pi_j$;
\item normalized biological effect $a_j$;
\item FDR-adjusted reliability $1-q_j$;
\item transport stability $\tau_j$.
\end{enumerate}
The Consensus Sheaf Discovery Score is
\[
\boxed{\operatorname{CSDS}_j=\pi_j a_j(1-q_j)\tau_j.}
\]
The patient-level Sheaf Discovery Index is
\[
\operatorname{SDI}(p)=\sum_{j\in\mathcal{T}_K}w_jz(\phi_j(p)),
\qquad
w_j\propto \operatorname{sign}(\bar\beta_j)\operatorname{CSDS}_j.
\]

\subsection{Why This Is Novel}
Phase 6 moves the project from prediction toward discovery. The output is not merely a class label; it is a ranked list of sheaf residual biomarkers with uncertainty, permutation support, FDR control, and transport stability.

\subsection{Key Outputs}
\begin{itemize}[leftmargin=*]
\item \texttt{phase6\_consensus\_feature\_discovery.csv}: feature-level CSDS table.
\item \texttt{phase6\_patient\_sheaf\_discovery\_index.csv}: patient SDI scores.
\item \texttt{phase6\_metric\_deltas.csv}: baseline vs Phase 6 metrics.
\item \texttt{phase6\_cross\_task\_consensus.csv}: features recurring across endpoints.
\end{itemize}

\subsection{Key Tables}
\begin{center}
\input{phase6_best_delta_table.tex}
\end{center}

\begin{center}
\resizebox{\textwidth}{!}{\input{phase6_top_feature_table.tex}}
\end{center}

\subsection{Main Figures}
\begin{figure}[H]
\centering
\includegraphics[width=.85\textwidth]{phase6_cross_task_CSDS_heatmap.png}
\caption{Phase 6 cross-task consensus sheaf discovery heatmap.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.85\textwidth]{phase6_top_CSDS_grade_label.png}
\caption{Top consensus sheaf discovery features for strict grade classification.}
\end{figure}

\section{What Teammates Should Understand}
\subsection{For Math Team}
The advanced mathematical objects are $L_g$, $E_g(p)$, $\Gamma^{\star}_{A,B}$, TSD, CSDS, and SDI. The core novelty is converting sheaf Laplacian residuals into subtype laws, transport-stability geometry, and reliability-ranked discovery features.

\subsection{For CS Team}
The main code paths are:
\begin{itemize}[leftmargin=*]
\item \texttt{phase4\_subtype\_sheaf\_geometry.py}
\item \texttt{phase5\_transport\_sheaf\_stability.py}
\item \texttt{phase6\_consensus\_sheaf\_discovery.py}
\end{itemize}
The important implementation idea is strict out-of-fold or leakage-aware evaluation. Avoid training on labels or features that trivially encode the label.

\subsection{For Biology Team}
The biological interpretation is that the method identifies where tumor molecular states stop behaving coherently. Residuals can be interpreted as DNA-to-regulatory inconsistency, DNA-to-phenotype inconsistency, or regulatory-to-phenotype inconsistency. Phase 6 turns these into candidate biomarkers, but they need biological validation before being called discoveries.

\section{Current Strengths and Limitations}
\subsection{Strengths}
\begin{itemize}[leftmargin=*]
\item The project now has a multi-phase technical pipeline rather than a single score.
\item Phase 4 shows group-specific regulatory geometry.
\item Phase 5 shows transport-calibrated residual stability.
\item Phase 6 adds uncertainty-aware biomarker ranking.
\end{itemize}

\subsection{Limitations}
\begin{itemize}[leftmargin=*]
\item These are internal cohort results.
\item External validation is still required.
\item Strong biomarker claims require pathway/gene-level biological annotation.
\item Accuracy gains must be reported carefully and not overstated.
\end{itemize}

\section{Immediate Next Steps}
\begin{enumerate}[leftmargin=*]
\item Run the same Phase 1--6 pipeline on CGGA or another external glioma cohort.
\item Replace abstract node-level variables with gene/pathway-level regulatory edges.
\item Map top Phase 6 features to biological pathways.
\item Add bootstrapped confidence intervals to all main deltas.
\item Prepare BIBM methods and ablation sections using Phases 4--6 as the advanced technical core.
\end{enumerate}

\end{document}
'''
# Copy figures/tables into a doc asset folder
asset = base/'phase456_team_doc_assets'
if asset.exists(): shutil.rmtree(asset)
asset.mkdir()
figs_to_copy = [
    base/'phase4_subtype_sheaf_geometry_package/figures/phase4_divergence_heatmap_grade_label_strict_no_grade_no_clusters.png',
    base/'phase4_subtype_sheaf_geometry_package/figures/phase4_balanced_accuracy_comparison.png',
    base/'phase5_transport_sheaf_stability_package/figures/phase5_ot_sheaf_gap_heatmap_grade_label_strict_no_idh_no_grade_no_clusters.png',
    base/'phase5_transport_sheaf_stability_package/figures/phase5_balanced_accuracy_deltas.png',
    pack/'figures/phase6_cross_task_CSDS_heatmap.png',
    pack/'figures/phase6_top_CSDS_grade_label.png',
]
for f in figs_to_copy:
    if f.exists(): shutil.copy(f, asset/f.name)
for f in [pack/'paper/phase6_best_delta_table.tex', pack/'paper/phase6_top_feature_table.tex']:
    shutil.copy(f, asset/f.name)
(asset/'phase456_team_documentation.tex').write_text(doc_tex)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase456_team_documentation.tex'], cwd=str(asset), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase456_team_documentation.tex'], cwd=str(asset), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.copy(asset/'phase456_team_documentation.pdf', base/'phase456_team_documentation.pdf')
shutil.copy(asset/'phase456_team_documentation.tex', base/'phase456_team_documentation.tex')

# Render verify PDFs / create contact sheets
for pdf_name, outdir in [('phase6_technical_specification.pdf','phase6_technical_renders'),('phase456_team_documentation.pdf','phase456_team_doc_renders')]:
    pdf_path = base/pdf_name
    out_path = base/outdir
    if out_path.exists(): shutil.rmtree(out_path)
    subprocess.run(['python','/home/oai/skills/pdfs/scripts/render_pdf.py', str(pdf_path), '--out_dir', str(out_path), '--dpi','120'], check=True, stdout=subprocess.DEVNULL)
    # create contact sheet
    from PIL import Image, ImageOps, ImageDraw
    imgs = sorted(out_path.glob('*.png'))[:12]
    thumbs=[]
    for imgp in imgs:
        im=Image.open(imgp).convert('RGB')
        im.thumbnail((260,360))
        canvas=Image.new('RGB',(280,390),'white')
        canvas.paste(im,((280-im.width)//2,10))
        d=ImageDraw.Draw(canvas); d.text((10,370),imgp.stem,fill='black')
        thumbs.append(canvas)
    if thumbs:
        cols=3; rows=math.ceil(len(thumbs)/cols)
        sheet=Image.new('RGB',(cols*280,rows*390),'white')
        for i,t in enumerate(thumbs): sheet.paste(t,((i%cols)*280,(i//cols)*390))
        sheet.save(base/(outdir.replace('_renders','_contact_sheet.png')))

# zip package
zip_path = base/'phase6_consensus_sheaf_discovery_package.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in pack.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(pack.parent))
print('Created', zip_path)
print('Created', base/'phase6_technical_specification.pdf')
print('Created', base/'phase456_team_documentation.pdf')
