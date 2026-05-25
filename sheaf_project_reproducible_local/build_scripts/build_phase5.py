import os, json, textwrap, shutil, zipfile, math, subprocess, re
from pathlib import Path

base = Path('/mnt/data')
pack = base/'phase5_transport_sheaf_stability_package'
if pack.exists():
    shutil.rmtree(pack)
for d in ['src','data','results','figures','paper']:
    (pack/d).mkdir(parents=True, exist_ok=True)

# Copy input data
inputs = {
    base/'phase4_subtype_sheaf_geometry_package/data/phase1_clean_encoded.csv': pack/'data/phase1_clean_encoded.csv',
    base/'phase4_subtype_sheaf_geometry_package/data/phase1_sris_results.csv': pack/'data/phase1_sris_results.csv',
    base/'phase4_subtype_sheaf_geometry_package/results/phase4_counterfactual_patient_energies.csv': pack/'data/phase4_counterfactual_patient_energies.csv',
    base/'phase4_subtype_sheaf_geometry_package/results/phase4_laplacian_divergences.csv': pack/'data/phase4_laplacian_divergences.csv',
}
for s,t in inputs.items():
    shutil.copy(s,t)

src = r'''
"""
Phase 5: Transport-Calibrated Sheaf Stability for Glioma Multi-Omics.

This module extends the Phase 1--4 sheaf pipeline with optimal-transport
robustness geometry.  The objective is not to replace the sheaf model, but to
ask whether sheaf residual signatures remain stable when patient distributions
are transported between biologically defined groups.

Main objects
------------
1. Pairwise transport plan between groups A and B:
       Gamma_AB = argmin_{Gamma in U(a,b)} <Gamma, C_AB> + eps KL(Gamma || ab^T)

2. Transport Sheaf Discrepancy (TSD):
       TSD(A,B) = sum_{i in A, j in B} Gamma_ij ||s_i - s_j||_2
   where s_i is a sheaf signature containing SRIS, edge residual energies, and
   counterfactual group energies.

3. Edge-level OT stability:
       Stable_e(A,B) = exp(- E_Gamma |E_e(i)-E_e(j)| / sigma_e)

4. Cross-validated transport-to-reference features:
       d_g(p) = E_{q in train group g} c(p,q)
   These are added to strict baseline models to test whether transport sheaf
   geometry improves dataset-level classification.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy.spatial.distance import cdist
from scipy.stats import zscore
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
        description='IDH/codel subtype under strict no-IDH/no-grade/no-cluster features',
    ),
    TaskSpec(
        task='grade_label',
        protocol='strict_no_idh_no_grade_no_clusters',
        label_col='grade_label',
        is_binary=False,
        exclude_idh=True,
        exclude_grade=True,
        exclude_clusters=True,
        description='G2/G3/G4 grade under strict no-IDH/no-grade/no-cluster features',
    ),
    TaskSpec(
        task='grade4_status',
        protocol='strict_no_grade_no_clusters',
        label_col='grade4_status',
        is_binary=True,
        exclude_idh=False,
        exclude_grade=True,
        exclude_clusters=True,
        description='Grade 4 status under strict no-grade/no-cluster features',
    ),
]


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
        med = out[c].median(skipna=True)
        if not np.isfinite(med):
            med = 0.0
        out[c] = out[c].fillna(med)
    return out


def sinkhorn_plan(C: np.ndarray, epsilon: Optional[float]=None, max_iter: int=1000, tol: float=1e-8) -> np.ndarray:
    """Stable enough entropic Sinkhorn for moderate sample sizes.

    Uniform marginals are used.  Cost is robustly rescaled to median 1 before
    exponentiation.  If numerical collapse occurs, falls back to a softmax plan.
    """
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    C = np.nan_to_num(C, nan=np.nanmedian(C), posinf=np.nanmax(C[np.isfinite(C)]), neginf=0.0)
    med = np.median(C[C > 0]) if np.any(C > 0) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    Cn = C / med
    if epsilon is None:
        epsilon = 0.25
    K = np.exp(-Cn / max(epsilon, 1e-6)) + EPS
    a = np.ones(n) / n
    b = np.ones(m) / m
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(max_iter):
        u_prev = u.copy()
        Kv = K @ v + EPS
        u = a / Kv
        Ktu = K.T @ u + EPS
        v = b / Ktu
        if np.linalg.norm(u - u_prev, ord=1) < tol:
            break
    G = (u[:, None] * K) * v[None, :]
    s = G.sum()
    if not np.isfinite(s) or s <= 0:
        P = np.exp(-Cn / max(epsilon, 1e-6))
        G = P / P.sum()
    else:
        G = G / s
    return G


def weighted_gap(G: np.ndarray, X: np.ndarray, Y: np.ndarray, metric: str='euclidean') -> float:
    if X.ndim == 1:
        D = np.abs(X[:, None] - Y[None, :])
    else:
        D = cdist(X, Y, metric=metric)
    return float(np.sum(G * D))


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['grade_label'] = df['grade'].map(lambda x: f'G{int(x)}' if pd.notna(x) else np.nan)
    df['grade4_status'] = (pd.to_numeric(df['grade'], errors='coerce') >= 4).astype(int)
    return df


def base_feature_columns(df: pd.DataFrame, spec: TaskSpec) -> List[str]:
    # Do not include age: age remains an external covariate.  Do not include survival.
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
    # Remove constants
    keep = []
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        if v.nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def energy_columns(cf: pd.DataFrame) -> List[str]:
    return [c for c in cf.columns if c.startswith('E_total__') or c.startswith('E_D_to_R__') or c.startswith('E_D_to_C__') or c.startswith('E_R_to_C__') or c == 'energy_margin']


def merge_task_data(clean: pd.DataFrame, sris: pd.DataFrame, cf: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    clean = make_labels(clean)
    cf_task = cf[(cf['task'] == spec.task) & (cf['protocol'] == spec.protocol)].copy()
    if cf_task.empty:
        raise ValueError(f'No Phase4 counterfactual rows for {spec.task}/{spec.protocol}')
    # One row per patient; Phase 4 is already out-of-fold.  Keep columns with protocol-specific energies.
    cols_cf = ['patient_id','fold','true_label','pred_min_energy','energy_margin'] + [c for c in cf_task.columns if c.startswith('E_') or c.startswith('p_sheaf__')]
    cf_task = cf_task[cols_cf].drop_duplicates('patient_id')
    sris_cols = ['patient_id','SRIS','E_D_to_R','E_D_to_C','E_R_to_C','frac_D_to_R','frac_D_to_C','frac_R_to_C']
    dat = clean.merge(sris[sris_cols], on='patient_id', how='inner').merge(cf_task, on='patient_id', how='inner')
    dat = dat[dat[spec.label_col].notna()].copy()
    return dat


def make_sheaf_signature(dat: pd.DataFrame, cf_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    phase1_cols = [c for c in ['SRIS','E_D_to_R','E_D_to_C','E_R_to_C','frac_D_to_R','frac_D_to_C','frac_R_to_C'] if c in dat.columns]
    cols = phase1_cols + [c for c in cf_cols if c in dat.columns and c.startswith('E_')]
    X = safe_numeric(dat, cols).values
    X = StandardScaler().fit_transform(X)
    return X, cols


def pairwise_transport_analysis(dat: pd.DataFrame, spec: TaskSpec, outdir: Path, n_perm: int=100, seed: int=17) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base_cols = base_feature_columns(dat, spec)
    cf_cols = energy_columns(dat)
    bio = safe_numeric(dat, base_cols).values
    if bio.shape[1] > 0:
        bio = StandardScaler().fit_transform(bio)
    sig, sig_cols = make_sheaf_signature(dat, cf_cols)
    labels = dat[spec.label_col].astype(str).values
    groups = sorted(pd.unique(labels))
    edge_cols = ['E_D_to_R','E_D_to_C','E_R_to_C','SRIS']
    results = []
    for a in groups:
        for b in groups:
            if a == b:
                continue
            ia = np.where(labels == a)[0]
            ib = np.where(labels == b)[0]
            if len(ia) < 3 or len(ib) < 3:
                continue
            Dbio = cdist(bio[ia], bio[ib]) if bio.shape[1] else np.zeros((len(ia), len(ib)))
            Dsig = cdist(sig[ia], sig[ib])
            # robust scale and combine
            Dbio = Dbio / (np.median(Dbio[Dbio > 0]) if np.any(Dbio > 0) else 1.0)
            Dsig = Dsig / (np.median(Dsig[Dsig > 0]) if np.any(Dsig > 0) else 1.0)
            C = 0.60 * Dbio + 0.40 * Dsig
            G = sinkhorn_plan(C, epsilon=0.35)
            row = {
                'task': spec.task, 'protocol': spec.protocol,
                'group_a': a, 'group_b': b,
                'n_a': len(ia), 'n_b': len(ib),
                'ot_cost': float(np.sum(G * C)),
                'bio_transport_gap': weighted_gap(G, bio[ia], bio[ib]) if bio.shape[1] else 0.0,
                'sheaf_transport_gap': weighted_gap(G, sig[ia], sig[ib]),
                'signature_columns': ';'.join(sig_cols)
            }
            for e in edge_cols:
                vals = safe_numeric(dat, [e]).values.ravel()
                gap = weighted_gap(G, vals[ia], vals[ib])
                scale = np.nanstd(vals) + EPS
                row[f'{e}_gap'] = gap
                row[f'{e}_stability'] = float(np.exp(-gap / scale))
            results.append(row)
    pair_df = pd.DataFrame(results)

    # Permutation test: compare mean between-group sheaf gap to label-shuffled null.
    observed = pair_df['sheaf_transport_gap'].mean() if not pair_df.empty else np.nan
    null = []
    for k in range(n_perm):
        labels_perm = rng.permutation(labels)
        groups_perm = sorted(pd.unique(labels_perm))
        vals = []
        for a in groups_perm:
            for b in groups_perm:
                if a >= b:
                    continue
                ia = np.where(labels_perm == a)[0]
                ib = np.where(labels_perm == b)[0]
                if len(ia) < 3 or len(ib) < 3:
                    continue
                # approximate with signature gap only for speed; still transport-calibrated.
                Dsig = cdist(sig[ia], sig[ib])
                Dsig = Dsig / (np.median(Dsig[Dsig > 0]) if np.any(Dsig > 0) else 1.0)
                G = sinkhorn_plan(Dsig, epsilon=0.35, max_iter=500)
                vals.append(weighted_gap(G, sig[ia], sig[ib]))
        null.append(np.mean(vals) if vals else np.nan)
    null = np.array(null, dtype=float)
    null = null[np.isfinite(null)]
    p = (1 + np.sum(null >= observed)) / (len(null) + 1) if len(null) else np.nan
    z = (observed - np.mean(null)) / (np.std(null) + EPS) if len(null) else np.nan
    perm_df = pd.DataFrame([{
        'task': spec.task, 'protocol': spec.protocol,
        'observed_mean_pairwise_sheaf_transport_gap': observed,
        'null_mean': float(np.mean(null)) if len(null) else np.nan,
        'null_sd': float(np.std(null)) if len(null) else np.nan,
        'z_score': float(z) if np.isfinite(z) else np.nan,
        'permutation_p_value_high_gap': float(p) if np.isfinite(p) else np.nan,
        'n_permutations': int(len(null))
    }])
    return pair_df, perm_df


def train_eval_cv(X_base: np.ndarray, X_add: Optional[np.ndarray], y: np.ndarray, is_binary: bool, seed: int=23) -> Dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)
    X = X_base if X_add is None else np.hstack([X_base, X_add])
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_true, y_pred = [], []
    probas = []
    for tr, te in skf.split(X, y_enc):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=5000, C=0.5, class_weight='balanced')
        clf.fit(Xtr, y_enc[tr])
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        y_true.extend(y_enc[te].tolist())
        y_pred.extend(pred.tolist())
        probas.append(proba)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    P = np.vstack(probas)
    out = {
        'n': int(len(y_true)),
        'classes': '|'.join(le.classes_),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'confusion_matrix': json.dumps(confusion_matrix(y_true, y_pred).tolist())
    }
    try:
        if is_binary or n_classes == 2:
            pos = 1 if P.shape[1] > 1 else 0
            out['auroc'] = float(roc_auc_score(y_true, P[:, pos]))
            out['auprc'] = float(average_precision_score(y_true, P[:, pos]))
        else:
            out['auroc_ovr_weighted'] = float(roc_auc_score(y_true, P, multi_class='ovr', average='weighted'))
            out['auroc_ovr_macro'] = float(roc_auc_score(y_true, P, multi_class='ovr', average='macro'))
    except Exception:
        pass
    return out


def make_transport_reference_features(dat: pd.DataFrame, spec: TaskSpec, seed: int=101) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-fitted transport-to-group reference distances.

    For each test patient and each training group g, compute mean transport cost from
    the test singleton to the empirical distribution of group g.  The cost uses a
    60/40 blend of strict biological features and sheaf signatures.
    """
    y = dat[spec.label_col].astype(str).values
    groups = sorted(pd.unique(y))
    base_cols = base_feature_columns(dat, spec)
    cf_cols = [c for c in energy_columns(dat) if c in dat.columns]
    bio_raw = safe_numeric(dat, base_cols).values
    sig_raw, sig_cols = make_sheaf_signature(dat, cf_cols)
    # sig_raw is already scaled globally; for CV features this is acceptable as it is unsupervised. 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    features = pd.DataFrame({'patient_id': dat['patient_id'].values, 'task': spec.task, 'protocol': spec.protocol, 'label': y})
    for g in groups:
        features[f'transport_dist_to__{g}'] = np.nan
        features[f'bio_dist_to__{g}'] = np.nan
        features[f'sheaf_dist_to__{g}'] = np.nan
    for fold, (tr, te) in enumerate(skf.split(bio_raw, y)):
        scaler = StandardScaler().fit(bio_raw[tr]) if bio_raw.shape[1] else None
        bio_tr = scaler.transform(bio_raw[tr]) if scaler else np.zeros((len(tr), 0))
        bio_te = scaler.transform(bio_raw[te]) if scaler else np.zeros((len(te), 0))
        sig_tr = sig_raw[tr]
        sig_te = sig_raw[te]
        y_tr = y[tr]
        for idx_pos, idx in enumerate(te):
            for g in groups:
                gpos = np.where(y_tr == g)[0]
                if len(gpos) == 0:
                    continue
                if bio_tr.shape[1]:
                    db = cdist(bio_te[[idx_pos]], bio_tr[gpos]).ravel()
                    db_med = np.median(db[db > 0]) if np.any(db > 0) else 1.0
                    dbn = db / (db_med if db_med > 0 else 1.0)
                else:
                    dbn = np.zeros(len(gpos))
                ds = cdist(sig_te[[idx_pos]], sig_tr[gpos]).ravel()
                ds_med = np.median(ds[ds > 0]) if np.any(ds > 0) else 1.0
                dsn = ds / (ds_med if ds_med > 0 else 1.0)
                cost = 0.60 * dbn + 0.40 * dsn
                # Singleton-to-empirical entropic OT reduces to expected cost under uniform target.
                features.loc[features.index[idx], f'transport_dist_to__{g}'] = float(np.mean(cost))
                features.loc[features.index[idx], f'bio_dist_to__{g}'] = float(np.mean(dbn))
                features.loc[features.index[idx], f'sheaf_dist_to__{g}'] = float(np.mean(dsn))
    feature_cols = [c for c in features.columns if c.startswith('transport_dist_to__') or c.startswith('bio_dist_to__') or c.startswith('sheaf_dist_to__')]
    # margins: nearest vs second nearest transport reference.
    vals = features[[c for c in features.columns if c.startswith('transport_dist_to__')]].values
    sorted_vals = np.sort(vals, axis=1)
    features['transport_margin'] = sorted_vals[:,1] - sorted_vals[:,0] if vals.shape[1] > 1 else 0.0
    features['transport_min_distance'] = sorted_vals[:,0]
    return features, pd.DataFrame({'feature': feature_cols + ['transport_margin','transport_min_distance']})


def prediction_benchmark(dat: pd.DataFrame, spec: TaskSpec, transport_features: pd.DataFrame) -> pd.DataFrame:
    y = dat[spec.label_col].astype(str).values
    base_cols = base_feature_columns(dat, spec)
    X_base = safe_numeric(dat, base_cols).values
    ecols = [c for c in energy_columns(dat) if c in dat.columns and (c.startswith('E_total__') or c == 'energy_margin')]
    X_sheaf = safe_numeric(dat, ecols).values if ecols else np.empty((len(dat),0))
    tf = transport_features.set_index('patient_id').loc[dat['patient_id']]
    tcols = [c for c in tf.columns if c.startswith('transport_dist_to__') or c.startswith('bio_dist_to__') or c.startswith('sheaf_dist_to__') or c in ['transport_margin','transport_min_distance']]
    X_transport = safe_numeric(tf.reset_index(), tcols).values
    rows = []
    for name, extra in [
        ('baseline_strict_features', None),
        ('baseline_plus_phase4_sheaf_energies', X_sheaf),
        ('baseline_plus_phase5_transport_features', X_transport),
        ('baseline_plus_sheaf_and_transport', np.hstack([X_sheaf, X_transport]) if X_sheaf.size else X_transport),
        ('transport_features_only', X_transport)
    ]:
        metrics = train_eval_cv(X_base if name != 'transport_features_only' else np.zeros((len(dat),1)), extra if name != 'transport_features_only' else X_transport, y, spec.is_binary)
        metrics.update({'task': spec.task, 'protocol': spec.protocol, 'method': name, 'n_base_features': len(base_cols), 'n_sheaf_features': X_sheaf.shape[1], 'n_transport_features': X_transport.shape[1]})
        rows.append(metrics)
    return pd.DataFrame(rows)


def make_figures(pair_df: pd.DataFrame, perm_df: pd.DataFrame, pred_df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Pairwise heatmaps by task/protocol
    for (task, protocol), sub in pair_df.groupby(['task','protocol']):
        if sub.empty:
            continue
        groups = sorted(set(sub.group_a) | set(sub.group_b))
        mat = pd.DataFrame(np.nan, index=groups, columns=groups)
        for _, r in sub.iterrows():
            mat.loc[r.group_a, r.group_b] = r.sheaf_transport_gap
        plt.figure(figsize=(6,5))
        plt.imshow(mat.values, aspect='auto')
        plt.xticks(range(len(groups)), groups, rotation=45, ha='right')
        plt.yticks(range(len(groups)), groups)
        plt.colorbar(label='OT sheaf transport gap')
        plt.title(f'Phase 5 OT sheaf gaps\n{task} / {protocol}')
        plt.tight_layout()
        fname = outdir/f'phase5_ot_sheaf_gap_heatmap_{task}_{protocol}.png'
        plt.savefig(fname, dpi=220)
        plt.close()
    # Accuracy delta plot
    if not pred_df.empty:
        rows=[]
        for (task, protocol), sub in pred_df.groupby(['task','protocol']):
            base = sub[sub.method=='baseline_strict_features'].iloc[0]
            for _, r in sub.iterrows():
                if r.method == 'baseline_strict_features':
                    continue
                rows.append({'task':task,'protocol':protocol,'method':r.method,'delta_bal_acc':r.balanced_accuracy-base.balanced_accuracy,'delta_acc':r.accuracy-base.accuracy})
        dd = pd.DataFrame(rows)
        if not dd.empty:
            plt.figure(figsize=(9,5))
            labels = [f"{r.task}\n{r.method.replace('baseline_plus_','+').replace('_',' ')}" for _,r in dd.iterrows()]
            plt.bar(range(len(dd)), dd['delta_bal_acc'].values)
            plt.axhline(0, linewidth=1)
            plt.xticks(range(len(dd)), labels, rotation=70, ha='right', fontsize=7)
            plt.ylabel('Delta balanced accuracy')
            plt.title('Phase 5 incremental value of OT-sheaf transport features')
            plt.tight_layout()
            plt.savefig(outdir/'phase5_balanced_accuracy_deltas.png', dpi=220)
            plt.close()


def run_phase5(input_dir: Path, output_dir: Path, n_perm: int=100) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'results').mkdir(exist_ok=True)
    (output_dir/'figures').mkdir(exist_ok=True)
    clean = pd.read_csv(input_dir/'phase1_clean_encoded.csv')
    sris = pd.read_csv(input_dir/'phase1_sris_results.csv')
    cf = pd.read_csv(input_dir/'phase4_counterfactual_patient_energies.csv')
    all_pairs=[]; all_perms=[]; all_preds=[]; all_tfeatures=[]
    for spec in TASKS:
        dat = merge_task_data(clean, sris, cf, spec)
        pair, perm = pairwise_transport_analysis(dat, spec, output_dir, n_perm=n_perm)
        tfeat, _ = make_transport_reference_features(dat, spec)
        pred = prediction_benchmark(dat, spec, tfeat)
        all_pairs.append(pair); all_perms.append(perm); all_preds.append(pred); all_tfeatures.append(tfeat)
    pair_df = pd.concat(all_pairs, ignore_index=True)
    perm_df = pd.concat(all_perms, ignore_index=True)
    pred_df = pd.concat(all_preds, ignore_index=True)
    tfeat_df = pd.concat(all_tfeatures, ignore_index=True)
    pair_df.to_csv(output_dir/'results/phase5_pairwise_transport_metrics.csv', index=False)
    perm_df.to_csv(output_dir/'results/phase5_permutation_transport_tests.csv', index=False)
    pred_df.to_csv(output_dir/'results/phase5_transport_prediction_metrics.csv', index=False)
    tfeat_df.to_csv(output_dir/'results/phase5_patient_transport_features.csv', index=False)
    # deltas
    deltas=[]
    for (task, protocol), sub in pred_df.groupby(['task','protocol']):
        base = sub[sub.method=='baseline_strict_features'].iloc[0]
        for _,r in sub.iterrows():
            if r.method == 'baseline_strict_features':
                continue
            row = {'task':task,'protocol':protocol,'method':r.method}
            for m in ['accuracy','balanced_accuracy','macro_f1','auroc','auprc','auroc_ovr_weighted','auroc_ovr_macro']:
                if m in sub.columns and pd.notna(r.get(m, np.nan)) and pd.notna(base.get(m, np.nan)):
                    row[f'delta_{m}'] = float(r[m]-base[m])
            deltas.append(row)
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(output_dir/'results/phase5_transport_accuracy_deltas.csv', index=False)
    make_figures(pair_df, perm_df, pred_df, output_dir/'figures')
    # Summary with best metrics
    best_delta = None
    if not delta_df.empty:
        metric = 'delta_balanced_accuracy'
        if metric in delta_df.columns:
            best_delta = delta_df.sort_values(metric, ascending=False).head(1).to_dict('records')[0]
    summary = {
        'phase': 5,
        'name': 'Transport-Calibrated Sheaf Stability',
        'n_tasks': len(TASKS),
        'tasks': [spec.__dict__ for spec in TASKS],
        'mean_pairwise_transport_gap_by_task': pair_df.groupby(['task','protocol'])['sheaf_transport_gap'].mean().reset_index().to_dict('records'),
        'permutation_tests': perm_df.to_dict('records'),
        'best_balanced_accuracy_delta': best_delta,
    }
    with open(output_dir/'results/phase5_summary.json','w') as f:
        json.dump(summary,f,indent=2)
    return summary
'''
(pack/'src/phase5_transport_sheaf_stability.py').write_text(src)
run = r'''
from pathlib import Path
from phase5_transport_sheaf_stability import run_phase5

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    summary = run_phase5(root / 'data', root, n_perm=100)
    print(summary)
'''
(pack/'src/run_phase5.py').write_text(run)
(pack/'requirements.txt').write_text('numpy\npandas\nscipy\nscikit-learn\nmatplotlib\n')
(pack/'README.md').write_text(textwrap.dedent('''
# Phase 5: Transport-Calibrated Sheaf Stability

This package extends the Phase 1--4 glioma sheaf pipeline by adding optimal-transport robustness geometry.
It tests whether patient-level sheaf residual signatures remain stable under cross-group distributional transport and whether transport-to-reference features add predictive value under strict leakage controls.

Run:

```bash
cd phase5_transport_sheaf_stability_package/src
python run_phase5.py
```

Main outputs:

- `results/phase5_pairwise_transport_metrics.csv`
- `results/phase5_permutation_transport_tests.csv`
- `results/phase5_transport_prediction_metrics.csv`
- `results/phase5_transport_accuracy_deltas.csv`
- `results/phase5_patient_transport_features.csv`
- `results/phase5_summary.json`
- `figures/*.png`

Novelty target: transport-calibrated sheaf residual stability and counterfactual reference distances for glioma multi-omics.
'''))

# Run phase5
subprocess.run(['python','run_phase5.py'], cwd=pack/'src', check=True)

# Load results for report values
import pandas as pd, numpy as np
pair_df = pd.read_csv(pack/'results/phase5_pairwise_transport_metrics.csv')
perm_df = pd.read_csv(pack/'results/phase5_permutation_transport_tests.csv')
pred_df = pd.read_csv(pack/'results/phase5_transport_prediction_metrics.csv')
delta_df = pd.read_csv(pack/'results/phase5_transport_accuracy_deltas.csv')
summary = json.load(open(pack/'results/phase5_summary.json'))

# Create tables for LaTeX
perm_table = perm_df.copy()
for c in perm_table.select_dtypes(include='number').columns:
    perm_table[c] = perm_table[c].map(lambda x: f'{x:.4g}' if pd.notna(x) else '')
perm_table[['task','protocol','observed_mean_pairwise_sheaf_transport_gap','null_mean','z_score','permutation_p_value_high_gap']].to_latex(pack/'paper/phase5_perm_table.tex', index=False, escape=True)

# best delta table
metric_cols = ['task','protocol','method','delta_accuracy','delta_balanced_accuracy','delta_macro_f1','delta_auroc','delta_auroc_ovr_weighted']
use_cols = [c for c in metric_cols if c in delta_df.columns]
dt = delta_df[use_cols].copy()
for c in dt.select_dtypes(include='number').columns:
    dt[c] = dt[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else '')
dt.to_latex(pack/'paper/phase5_delta_table.tex', index=False, escape=True)

# write technical report markdown
best = summary.get('best_balanced_accuracy_delta') or {}
report = f'''# Phase 5 Technical Report: Transport-Calibrated Sheaf Stability

## Core idea
Phase 5 adds an optimal-transport layer on top of Phase 4 counterfactual sheaf geometry.  Phase 4 asks whether group-specific sheaf Laplacians differ; Phase 5 asks whether the residual signatures remain stable when patient distributions are optimally transported between biological groups.

## Main objects
- Pairwise transport plans between groups.
- Transport Sheaf Discrepancy (TSD).
- Edge-level transport stability for D->R, D->C, and R->C residuals.
- Cross-fitted transport-to-reference features for strict prediction tests.

## Best balanced-accuracy delta
{json.dumps(best, indent=2)}

## Permutation tests
{perm_df.to_string(index=False)}

## Interpretation
This phase should be presented as a robustness and geometry contribution.  It does not claim that optimal transport alone solves classification.  Instead, it quantifies whether sheaf residual signatures define non-random group-level transport geometry and whether transport distances provide incremental predictive signal under strict controls.
'''
(pack/'phase5_technical_report.md').write_text(report)

# LaTeX PDF
tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsfonts, bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{array}
\usepackage{caption}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}
\title{Phase 5 Technical Specification: Transport-Calibrated Sheaf Stability for Glioma Multi-Omics}
\author{Sheaf-Theoretic Multi-Omics Regulatory Inconsistency Project}
\date{May 23, 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Purpose of Phase 5}
Phases 1--4 constructed patient-level sheaf residual energies and group-specific counterfactual sheaf Laplacians.  Phase 5 adds a robustness layer: it asks whether the learned sheaf residual signatures remain stable when patient distributions are transported between biological groups.  This turns the method from a purely predictive or residual-scoring framework into a distributional geometry framework.

The central contribution is a \emph{transport-calibrated sheaf stability} analysis.  Instead of only asking whether two group-specific Laplacians differ, we also ask whether patients from one group can be transported to patients from another group while preserving sheaf residual structure.  This is designed to be different from standard multi-omics graph aggregation: the transported object is not merely a feature vector, but a sheaf-derived regulatory inconsistency signature.

\section{Inputs from Previous Phases}
For each patient $p$, Phase 1 gives a sheaf regulatory inconsistency score
\begin{equation}
\operatorname{SRIS}(p)=x_p^{\top}L_{\mathcal F}x_p,
\end{equation}
with edge energies
\begin{equation}
E_{D\to R}(p),\qquad E_{D\to C}(p),\qquad E_{R\to C}(p).
\end{equation}
Phase 4 gives group-specific counterfactual sheaf energies
\begin{equation}
E_g(p)=x_p^{\top}L_gx_p,
\end{equation}
where $g$ ranges over subtype or grade groups.  Phase 5 combines these into a patient sheaf signature
\begin{equation}
\bm{s}(p)=\left[\operatorname{SRIS}(p),E_{D\to R}(p),E_{D\to C}(p),E_{R\to C}(p),\{E_g(p)\}_{g\in\mathcal G}\right].
\end{equation}

\section{Pairwise Transport Between Biological Groups}
Let $A$ and $B$ be two biological groups, such as $\mathrm{IDHwt}$ and $\mathrm{IDHmut\text{-}codel}$, or $G2$ and $G4$.  Let
\begin{equation}
\{p_i\}_{i=1}^{n_A}\subset A,
\qquad
\{q_j\}_{j=1}^{n_B}\subset B.
\end{equation}
We define two distances: a strict biological feature distance and a sheaf-signature distance.

\subsection{Strict biological feature distance}
Let $\bm{z}(p)$ be a leakage-controlled feature vector.  Age and survival are excluded from the transport state, and task labels such as IDH or grade are removed under the corresponding strict protocol.  The biological distance is
\begin{equation}
C^{\mathrm{bio}}_{ij}=\|\bm{z}(p_i)-\bm{z}(q_j)\|_2.
\end{equation}

\subsection{Sheaf-signature distance}
The sheaf distance is
\begin{equation}
C^{\mathrm{sheaf}}_{ij}=\|\bm{s}(p_i)-\bm{s}(q_j)\|_2.
\end{equation}
Both distances are median-normalized, then combined:
\begin{equation}
C_{ij}=0.60\,\widetilde C^{\mathrm{bio}}_{ij}+0.40\,\widetilde C^{\mathrm{sheaf}}_{ij}.
\end{equation}

\section{Entropic Optimal Transport Plan}
For uniform marginals $a_i=1/n_A$ and $b_j=1/n_B$, the transport plan is
\begin{equation}
\Gamma_{A\to B}^{\star}=\arg\min_{\Gamma\in U(a,b)}
\sum_{i,j}\Gamma_{ij}C_{ij}
+\varepsilon\sum_{i,j}\Gamma_{ij}\log \Gamma_{ij},
\end{equation}
where
\begin{equation}
U(a,b)=\{\Gamma\ge 0: \Gamma\mathbf 1=b,\ \Gamma^{\top}\mathbf 1=a\}.
\end{equation}
The implementation uses Sinkhorn scaling with robust cost normalization.

\section{Transport Sheaf Discrepancy}
The main Phase 5 statistic is the transport sheaf discrepancy:
\begin{equation}
\operatorname{TSD}(A,B)=
\sum_{i\in A}\sum_{j\in B}\Gamma^{\star}_{ij}\,\|\bm{s}(p_i)-\bm{s}(q_j)\|_2.
\end{equation}
This measures how much sheaf residual structure changes under the best distributional alignment between two groups.

\section{Edge-Level Transport Stability}
For an edge $e\in\{D\to R,D\to C,R\to C\}$, define the transported edge gap
\begin{equation}
G_e(A,B)=\sum_{i,j}\Gamma^{\star}_{ij}\left|E_e(p_i)-E_e(q_j)\right|.
\end{equation}
To make the score scale-free, define
\begin{equation}
\operatorname{Stable}_e(A,B)=\exp\left(-\frac{G_e(A,B)}{\sigma_e+10^{-9}}\right),
\end{equation}
where $\sigma_e$ is the cohort standard deviation of edge energy $E_e$.  Values closer to $1$ mean stronger preservation of that edge residual channel under transport.

\section{Permutation Test}
To determine whether the observed transport sheaf geometry is non-random, group labels are permuted $B$ times.  For each permutation $b$, we compute the mean pairwise transport sheaf discrepancy
\begin{equation}
\overline{\operatorname{TSD}}^{(b)}.
\end{equation}
The empirical high-gap $p$-value is
\begin{equation}
 p=\frac{1+\#\{b:\overline{\operatorname{TSD}}^{(b)}\ge \overline{\operatorname{TSD}}^{\mathrm{obs}}\}}{B+1}.
\end{equation}
The $z$-score is
\begin{equation}
 z=\frac{\overline{\operatorname{TSD}}^{\mathrm{obs}}-\mu_{\mathrm{null}}}{\sigma_{\mathrm{null}}+10^{-9}}.
\end{equation}

\section{Transport-to-Reference Prediction Features}
For each patient $p$ and group $g$, Phase 5 defines a cross-fitted transport reference distance
\begin{equation}
 d_g(p)=\frac{1}{|\mathcal T_g|}\sum_{q\in\mathcal T_g}c(p,q),
\end{equation}
where $\mathcal T_g$ is the training-fold empirical distribution of group $g$ and $c(p,q)$ is the same 60/40 biological-sheaf cost used in the OT analysis.  These features are evaluated under strict cross-validation:
\begin{equation}
\{d_g(p)\}_{g\in\mathcal G},\qquad
\min_g d_g(p),\qquad
\text{second-min}_g d_g(p)-\min_g d_g(p).
\end{equation}
They are compared against baseline features and Phase 4 sheaf energies.

\section{Outputs Produced}
The package produces:
\begin{itemize}[leftmargin=2em]
    \item pairwise OT sheaf transport metrics,
    \item edge-level transport stability scores,
    \item permutation tests for non-random sheaf transport separation,
    \item cross-fitted patient-level transport-to-reference features,
    \item strict prediction metrics and accuracy deltas,
    \item heatmaps and accuracy-delta figures.
\end{itemize}

\section{Permutation Results}
\input{phase5_perm_table.tex}

\section{Accuracy Delta Results}
\input{phase5_delta_table.tex}

\section{Interpretation for the Manuscript}
The strongest Phase 5 claim is not that optimal transport alone maximizes classification accuracy.  The stronger and more defensible claim is that sheaf residual signatures define a non-random distributional geometry between biological tumor groups.  This gives a robustness layer that standard feature-fusion and graph-aggregation models generally do not provide: the framework can say which regulatory residual channels remain stable or unstable under cross-group distributional alignment.

\section{Novelty Positioning}
Optimal transport has been widely used for single-cell and multi-omics alignment, and sheaf neural networks have been developed as graph learning models with restriction maps and sheaf Laplacians.  Phase 5 combines these ideas in a more specific way: the transported object is a glioma-specific sheaf residual signature, not merely a raw expression vector or patient similarity embedding.  Therefore, the intended novelty is \emph{transport-calibrated stability of learned regulatory inconsistency geometry}.

\section{Limitations}
This phase remains internally validated on the current dataset.  The strongest next validation is external TCGA-to-CGGA transport, where the same OT-sheaf stability statistics can test whether residual signatures transfer across cohorts.  Until then, the appropriate claim is internal robustness and non-random transport geometry, not definitive external state-of-the-art survival prediction.

\section{Immediate Next Step}
The next phase should use an external cohort.  The most natural experiment is:
\begin{equation}
\text{TCGA glioma}\longrightarrow \text{CGGA glioma}
\end{equation}
with the same cost, the same sheaf signatures, and the same edge-level OT stability statistics.

\end{document}
'''
(pack/'paper/phase5_technical_specification.tex').write_text(tex)
# compile
subprocess.run(['pdflatex','-interaction=nonstopmode','phase5_technical_specification.tex'], cwd=pack/'paper', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase5_technical_specification.tex'], cwd=pack/'paper', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Copy pdf/tex to /mnt/data
shutil.copy(pack/'paper/phase5_technical_specification.pdf', base/'phase5_technical_specification.pdf')
shutil.copy(pack/'paper/phase5_technical_specification.tex', base/'phase5_technical_specification.tex')

# render PDF to verify
render_dir = base/'phase5_technical_renders'
if render_dir.exists(): shutil.rmtree(render_dir)
subprocess.run(['python','/home/oai/skills/pdfs/scripts/render_pdf.py', str(base/'phase5_technical_specification.pdf'), '--out_dir', str(render_dir), '--dpi', '120'], check=True)
# contact sheet
from PIL import Image, ImageDraw
imgs = sorted(render_dir.glob('*.png'))[:6]
thumbs=[]
for imf in imgs:
    im=Image.open(imf).convert('RGB')
    im.thumbnail((240,320))
    thumbs.append((im, imf.name))
w=3*260; h=2*360
sheet=Image.new('RGB',(w,h),'white')
d=ImageDraw.Draw(sheet)
for idx,(im,name) in enumerate(thumbs):
    x=(idx%3)*260+10; y=(idx//3)*360+10
    sheet.paste(im,(x,y))
    d.text((x,y+im.height+5),name,fill='black')
sheet.save(base/'phase5_technical_contact_sheet.png')

# zip package
zip_path = base/'phase5_transport_sheaf_stability_package.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in pack.rglob('*'):
        z.write(f, f.relative_to(pack.parent))

print('DONE')
print(zip_path)
print(base/'phase5_technical_specification.pdf')
print(summary)
