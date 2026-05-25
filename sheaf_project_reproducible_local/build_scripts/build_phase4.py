import os, json, zipfile, shutil, math, textwrap, warnings, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from scipy import stats
from scipy.special import softmax
from numpy.linalg import eigvalsh, eigh

warnings.filterwarnings('ignore')
BASE = Path('/mnt/data')
PKG = BASE / 'phase4_subtype_sheaf_geometry_package'
if PKG.exists(): shutil.rmtree(PKG)
for d in ['src','data','results','figures','paper']:
    (PKG/d).mkdir(parents=True, exist_ok=True)

# ---------------- core module code ----------------
module = r'''
import numpy as np
import pandas as pd
from numpy.linalg import solve, pinv, eigvalsh, eigh
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from scipy.special import softmax
from scipy import stats

EPS = 1e-8

D_ALL = [
    'idh_wt_z', 'mgmt_unmethylated_z', 'atrx_wt_z', 'tert_promoter_mutant_z',
    'chr7_gain_chr10_loss_z', 'mutation_count_z', 'tmb_z', 'aneuploidy_z'
]
R_ALL = [
    'egfr_amp_z', 'tert_expr_z', 'tert_expressed_z', 'immune_score_z', 'stromal_score_z',
    'rna_cluster_risk_z', 'methyl_cluster_risk_z', 'transcriptome_risk_z'
]
C_ALL = ['grade_risk_z', 'kps_low_z', 'purity_low_z']

PROTOCOLS = {
    'full_geometry': {
        'D': D_ALL,
        'R': R_ALL,
        'C': C_ALL,
        'description': 'All Phase 1 tumor state features. Used for biological geometry discovery, not leakage-free prediction.'
    },
    'strict_no_idh_no_clusters': {
        'D': [x for x in D_ALL if x != 'idh_wt_z'],
        'R': [x for x in R_ALL if x not in ['rna_cluster_risk_z','methyl_cluster_risk_z','transcriptome_risk_z']],
        'C': C_ALL,
        'description': 'Removes IDH indicator and cluster risk variables for stricter subtype analysis.'
    },
    'strict_no_grade_no_clusters': {
        'D': D_ALL,
        'R': [x for x in R_ALL if x not in ['rna_cluster_risk_z','methyl_cluster_risk_z','transcriptome_risk_z']],
        'C': [x for x in C_ALL if x != 'grade_risk_z'],
        'description': 'Removes grade risk and cluster risk variables for stricter grade analysis.'
    },
    'strict_no_idh_no_grade_no_clusters': {
        'D': [x for x in D_ALL if x != 'idh_wt_z'],
        'R': [x for x in R_ALL if x not in ['rna_cluster_risk_z','methyl_cluster_risk_z','transcriptome_risk_z']],
        'C': [x for x in C_ALL if x != 'grade_risk_z'],
        'description': 'Strictest leakage-aware protocol.'
    },
}
EDGES = [('D','R'), ('D','C'), ('R','C')]

class SubtypeReferenceSheaf:
    def __init__(self, protocol='full_geometry', ridge=1.0):
        self.protocol = protocol
        self.features = PROTOCOLS[protocol]
        self.ridge = float(ridge)
        self.maps = {}
        self.group = None
        self.node_dims = {node: len(self.features[node]) for node in ['D','R','C']}
        self.offsets = {}
        off = 0
        for node in ['D','R','C']:
            self.offsets[node] = off
            off += self.node_dims[node]
        self.total_dim = off

    def _X(self, df, node):
        cols = self.features[node]
        return df[cols].astype(float).fillna(0.0).values

    @staticmethod
    def _ridge_map(Xs, Xt, ridge):
        # returns W with shape source_dim x target_dim so Xs @ W predicts Xt
        ds = Xs.shape[1]
        A = Xs.T @ Xs + ridge * np.eye(ds)
        B = Xs.T @ Xt
        try:
            W = solve(A, B)
        except Exception:
            W = pinv(A) @ B
        return W

    def fit(self, df, group_name=None):
        self.group = group_name
        XD, XR, XC = self._X(df,'D'), self._X(df,'R'), self._X(df,'C')
        self.maps[('D','R')] = self._ridge_map(XD, XR, self.ridge)
        self.maps[('D','C')] = self._ridge_map(XD, XC, self.ridge)
        self.maps[('R','C')] = self._ridge_map(XR, XC, self.ridge)
        return self

    def edge_energies(self, df):
        X = {'D': self._X(df,'D'), 'R': self._X(df,'R'), 'C': self._X(df,'C')}
        out = {}
        for u,v in EDGES:
            pred = X[u] @ self.maps[(u,v)]
            res = pred - X[v]
            out[f'E_{u}_to_{v}'] = np.mean(res**2, axis=1)
            out[f'norm_{u}_to_{v}'] = np.sqrt(np.sum(res**2, axis=1))
        total = sum(out[f'E_{u}_to_{v}'] for u,v in EDGES)
        out['SRIS4'] = total
        return pd.DataFrame(out, index=df.index)

    def coboundary(self):
        rows = []
        for u,v in EDGES:
            W = self.maps[(u,v)] # source_dim x target_dim
            dv = self.node_dims[v]
            block = np.zeros((dv, self.total_dim))
            ou, ov = self.offsets[u], self.offsets[v]
            du = self.node_dims[u]
            # rho_source = W.T, rho_target = I
            block[:, ou:ou+du] = W.T
            block[:, ov:ov+dv] = -np.eye(dv)
            rows.append(block)
        return np.vstack(rows)

    def laplacian(self):
        B = self.coboundary()
        return B.T @ B

    def normalized_laplacian(self):
        L = self.laplacian()
        return L / (np.linalg.norm(L, 'fro') + EPS)


def fit_group_sheaves(df, label_col, protocol, ridge=1.0, min_n=20):
    sheaves = {}
    for g, sub in df.groupby(label_col):
        if pd.isna(g) or len(sub) < min_n:
            continue
        sheaves[str(g)] = SubtypeReferenceSheaf(protocol, ridge).fit(sub, str(g))
    return sheaves


def score_under_sheaves(df, sheaves):
    parts = []
    for g, sh in sheaves.items():
        e = sh.edge_energies(df).copy()
        e.columns = [f'{c}__{g}' for c in e.columns]
        parts.append(e)
    return pd.concat(parts, axis=1)


def pairwise_laplacian_divergence(sheaves, harmonic_dim=3):
    groups = list(sheaves)
    rows = []
    for a in groups:
        for b in groups:
            La = sheaves[a].normalized_laplacian()
            Lb = sheaves[b].normalized_laplacian()
            fro = float(np.linalg.norm(La-Lb, 'fro'))
            ea = np.sort(eigvalsh(La))
            eb = np.sort(eigvalsh(Lb))
            k = min(10, len(ea), len(eb))
            spec = float(np.linalg.norm(ea[:k]-eb[:k]))
            # harmonic/low-energy subspace distance via projector difference
            h = min(harmonic_dim, La.shape[0])
            _, Va = eigh(La)
            _, Vb = eigh(Lb)
            Pa = Va[:,:h] @ Va[:,:h].T
            Pb = Vb[:,:h] @ Vb[:,:h].T
            subspace = float(np.linalg.norm(Pa-Pb, 'fro'))
            rows.append({'group_a':a,'group_b':b,'frobenius_divergence':fro,'spectral_low10_distance':spec,'harmonic_projector_distance':subspace})
    return pd.DataFrame(rows)


def mean_pairwise_divergence(div_df):
    x = div_df[div_df.group_a != div_df.group_b]['frobenius_divergence'].values
    return float(np.mean(x)) if len(x) else np.nan


def permutation_divergence_test(df, label_col, protocol, ridge=1.0, n_perm=100, seed=7, min_n=20):
    rng = np.random.default_rng(seed)
    work = df.dropna(subset=[label_col]).copy()
    obs_sheaves = fit_group_sheaves(work, label_col, protocol, ridge, min_n)
    obs_div = pairwise_laplacian_divergence(obs_sheaves)
    obs = mean_pairwise_divergence(obs_div)
    vals = []
    y = work[label_col].astype(str).values.copy()
    for _ in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        tmp = work.copy()
        tmp['_perm_label'] = yp
        sh = fit_group_sheaves(tmp, '_perm_label', protocol, ridge, min_n)
        if len(sh) < 2:
            continue
        vals.append(mean_pairwise_divergence(pairwise_laplacian_divergence(sh)))
    vals = np.array(vals, dtype=float)
    p = (1 + np.sum(vals >= obs)) / (len(vals) + 1) if len(vals) else np.nan
    z = (obs - float(np.mean(vals))) / (float(np.std(vals)) + EPS) if len(vals) else np.nan
    return {
        'label_col': label_col, 'protocol': protocol, 'n_perm': int(len(vals)),
        'observed_mean_pairwise_frobenius': obs,
        'perm_mean': float(np.mean(vals)) if len(vals) else np.nan,
        'perm_std': float(np.std(vals)) if len(vals) else np.nan,
        'permutation_p_value': float(p),
        'z_score_vs_permutation': float(z)
    }, obs_div


def _metric_row(task, protocol, method, y_true, y_pred, proba=None, classes=None):
    row = {
        'task':task, 'protocol':protocol, 'method':method, 'n':len(y_true),
        'accuracy':accuracy_score(y_true,y_pred),
        'balanced_accuracy':balanced_accuracy_score(y_true,y_pred),
        'macro_f1':f1_score(y_true,y_pred,average='macro'),
    }
    try:
        row['confusion_matrix'] = str(confusion_matrix(y_true,y_pred).tolist())
    except Exception:
        row['confusion_matrix'] = ''
    if proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # ensure positive class is last label encoder class
                row['auroc'] = roc_auc_score(y_true, proba[:,1])
                row['auprc'] = average_precision_score(y_true, proba[:,1])
            else:
                row['auroc_ovr_weighted'] = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
                row['auroc_ovr_macro'] = roc_auc_score(y_true, proba, multi_class='ovr', average='macro')
        except Exception:
            pass
    return row


def _baseline_features(protocol):
    f = PROTOCOLS[protocol]
    return f['D'] + f['R'] + f['C']


def crossvalidated_counterfactual_assignment(df, label_col, protocol, ridge=1.0, n_splits=5, seed=13, min_n=15):
    work = df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    work[label_col] = work[label_col].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(work[label_col].values)
    y_labels = le.classes_
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_rows = []
    pred_energy = np.zeros(len(work), dtype=int)
    proba_energy = np.zeros((len(work), len(y_labels)))
    pred_hybrid = np.zeros(len(work), dtype=int)
    proba_hybrid = np.zeros((len(work), len(y_labels)))
    pred_base = np.zeros(len(work), dtype=int)
    proba_base = np.zeros((len(work), len(y_labels)))
    feature_cols = _baseline_features(protocol)
    for fold, (tr, te) in enumerate(skf.split(work, y)):
        train = work.iloc[tr].copy(); test = work.iloc[te].copy()
        sheaves = fit_group_sheaves(train, label_col, protocol, ridge=ridge, min_n=min_n)
        # If any class missing due min_n, set with lower min_n fallback
        if set(sheaves.keys()) != set(y_labels.astype(str)):
            sheaves = fit_group_sheaves(train, label_col, protocol, ridge=ridge, min_n=2)
        # energy matrices train/test in class order
        E_train, E_test = [], []
        all_energy_cols = []
        for cls in y_labels:
            cls = str(cls)
            sh = sheaves[cls]
            etr = sh.edge_energies(train)
            ete = sh.edge_energies(test)
            for edge_col in ['SRIS4','E_D_to_R','E_D_to_C','E_R_to_C']:
                name = f'{edge_col}__{cls}'
                all_energy_cols.append(name)
            E_train.append(etr[['SRIS4','E_D_to_R','E_D_to_C','E_R_to_C']].values)
            E_test.append(ete[['SRIS4','E_D_to_R','E_D_to_C','E_R_to_C']].values)
        E_train = np.stack(E_train, axis=1)  # n_train x n_classes x 4
        E_test = np.stack(E_test, axis=1)
        total_train = E_train[:,:,0]
        total_test = E_test[:,:,0]
        # minimum energy assignment
        pred_energy[te] = np.argmin(total_test, axis=1)
        tau = np.median(np.std(total_train, axis=0)) + 1e-6
        proba_energy[te] = softmax(-total_test / tau, axis=1)
        # baseline logistic
        Xtr = train[feature_cols].astype(float).fillna(0).values
        Xte = test[feature_cols].astype(float).fillna(0).values
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        base.fit(Xtr, y[tr])
        pred_base[te] = base.predict(Xte)
        proba_base[te] = base.predict_proba(Xte)
        # hybrid logistic with energy features
        Xetr = E_train.reshape(E_train.shape[0], -1)
        Xete = E_test.reshape(E_test.shape[0], -1)
        Xhtr = np.hstack([Xtr, Xetr])
        Xhte = np.hstack([Xte, Xete])
        hyb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        hyb.fit(Xhtr, y[tr])
        pred_hybrid[te] = hyb.predict(Xhte)
        proba_hybrid[te] = hyb.predict_proba(Xhte)
        # patient long rows
        for local_idx, global_idx in enumerate(te):
            row = {
                'patient_id': work.loc[global_idx,'patient_id'],
                'task': label_col, 'protocol': protocol, 'fold': fold,
                'true_label': work.loc[global_idx,label_col],
                'pred_min_energy': y_labels[pred_energy[global_idx]],
            }
            sortedE = np.sort(total_test[local_idx])
            row['energy_margin'] = float(sortedE[1] - sortedE[0]) if len(sortedE)>1 else np.nan
            for ci, cls in enumerate(y_labels):
                row[f'E_total__{cls}'] = float(E_test[local_idx, ci, 0])
                row[f'E_D_to_R__{cls}'] = float(E_test[local_idx, ci, 1])
                row[f'E_D_to_C__{cls}'] = float(E_test[local_idx, ci, 2])
                row[f'E_R_to_C__{cls}'] = float(E_test[local_idx, ci, 3])
                row[f'p_sheaf__{cls}'] = float(proba_energy[global_idx, ci])
            patient_rows.append(row)
    metrics = []
    metrics.append(_metric_row(label_col, protocol, 'baseline_logistic_features', y, pred_base, proba_base, y_labels))
    metrics.append(_metric_row(label_col, protocol, 'counterfactual_min_energy_sheaf', y, pred_energy, proba_energy, y_labels))
    metrics.append(_metric_row(label_col, protocol, 'hybrid_logistic_features_plus_sheaf_energies', y, pred_hybrid, proba_hybrid, y_labels))
    # decode preds for convenience
    patient = pd.DataFrame(patient_rows)
    return pd.DataFrame(metrics), patient


def group_edge_summary(df, label_col, protocol, ridge=1.0):
    work = df.dropna(subset=[label_col]).copy()
    sh = fit_group_sheaves(work, label_col, protocol, ridge, min_n=10)
    rows = []
    for g, sheaf in sh.items():
        sub = work[work[label_col].astype(str)==str(g)]
        e = sheaf.edge_energies(sub)
        rows.append({
            'label_col':label_col,'protocol':protocol,'group':g,'n':len(sub),
            'mean_SRIS4':float(e['SRIS4'].mean()),
            'mean_E_D_to_R':float(e['E_D_to_R'].mean()),
            'mean_E_D_to_C':float(e['E_D_to_C'].mean()),
            'mean_E_R_to_C':float(e['E_R_to_C'].mean()),
            'dominant_edge': ['D_to_R','D_to_C','R_to_C'][int(np.argmax([e['E_D_to_R'].mean(), e['E_D_to_C'].mean(), e['E_R_to_C'].mean()]))]
        })
    return pd.DataFrame(rows)
'''
(PKG/'src/phase4_subtype_sheaf_geometry.py').write_text(module)

# runner code
runner = r'''
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from phase4_subtype_sheaf_geometry import (
    PROTOCOLS, fit_group_sheaves, pairwise_laplacian_divergence, permutation_divergence_test,
    crossvalidated_counterfactual_assignment, group_edge_summary, score_under_sheaves
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/phase1_clean_encoded.csv'
OUT = ROOT/'results'
FIG = ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
# Clean labels
for col in ['idh_codel_subtype','transcriptome_subtype','methylation_cluster','rna_cluster']:
    if col in df.columns:
        df[col] = df[col].astype(object)
df['grade_label'] = df['grade'].apply(lambda x: f'G{int(x)}' if pd.notna(x) else np.nan)
df['grade4_status'] = df['grade'].apply(lambda x: 'G4' if pd.notna(x) and int(x)==4 else ('G2G3' if pd.notna(x) else np.nan))

# Main protocols and tasks
analysis_specs = [
    ('idh_codel_subtype','full_geometry'),
    ('idh_codel_subtype','strict_no_idh_no_clusters'),
    ('grade_label','strict_no_grade_no_clusters'),
    ('grade4_status','strict_no_grade_no_clusters'),
    ('idh_codel_subtype','strict_no_idh_no_grade_no_clusters'),
    ('grade_label','strict_no_idh_no_grade_no_clusters'),
]

# Divergence and permutation tests
all_divs=[]; perm_rows=[]; edge_rows=[]
for label_col, protocol in analysis_specs[:4]:
    work=df.dropna(subset=[label_col]).copy()
    sheaves=fit_group_sheaves(work,label_col,protocol,ridge=1.0,min_n=20)
    div=pairwise_laplacian_divergence(sheaves)
    div['label_col']=label_col; div['protocol']=protocol
    all_divs.append(div)
    er=group_edge_summary(work,label_col,protocol,ridge=1.0)
    edge_rows.append(er)
    # fewer perms by default for runtime; enough for signal sanity, increase in paper revision
    ptest, _ = permutation_divergence_test(work,label_col,protocol,ridge=1.0,n_perm=100,seed=31,min_n=20)
    perm_rows.append(ptest)

pd.concat(all_divs,ignore_index=True).to_csv(OUT/'phase4_laplacian_divergences.csv',index=False)
pd.DataFrame(perm_rows).to_csv(OUT/'phase4_permutation_divergence_tests.csv',index=False)
pd.concat(edge_rows,ignore_index=True).to_csv(OUT/'phase4_group_edge_energy_summary.csv',index=False)

# Cross-validated prediction/assignment
metric_parts=[]; patient_parts=[]
for label_col, protocol in analysis_specs:
    m,p = crossvalidated_counterfactual_assignment(df,label_col,protocol,ridge=1.0,n_splits=5,seed=19,min_n=15)
    metric_parts.append(m); patient_parts.append(p)
metrics=pd.concat(metric_parts,ignore_index=True)
patients=pd.concat(patient_parts,ignore_index=True)
metrics.to_csv(OUT/'phase4_counterfactual_accuracy_metrics.csv',index=False)
patients.to_csv(OUT/'phase4_counterfactual_patient_energies.csv',index=False)

# Deltas relative to baseline features
rows=[]
for (task,protocol), sub in metrics.groupby(['task','protocol']):
    base=sub[sub.method=='baseline_logistic_features']
    if base.empty: continue
    base=base.iloc[0]
    for _,r in sub.iterrows():
        if r['method']=='baseline_logistic_features': continue
        rows.append({
            'task':task,'protocol':protocol,'method':r['method'],
            'delta_accuracy':r['accuracy']-base['accuracy'],
            'delta_balanced_accuracy':r['balanced_accuracy']-base['balanced_accuracy'],
            'delta_macro_f1':r['macro_f1']-base['macro_f1'],
            'baseline_accuracy':base['accuracy'], 'method_accuracy':r['accuracy'],
            'baseline_balanced_accuracy':base['balanced_accuracy'], 'method_balanced_accuracy':r['balanced_accuracy'],
        })
pd.DataFrame(rows).to_csv(OUT/'phase4_accuracy_deltas.csv',index=False)

# Figure 1: divergence heatmaps for key tasks
for label_col, protocol in [('idh_codel_subtype','strict_no_idh_no_clusters'),('grade_label','strict_no_grade_no_clusters')]:
    divs=pd.read_csv(OUT/'phase4_laplacian_divergences.csv')
    sub=divs[(divs.label_col==label_col)&(divs.protocol==protocol)]
    groups=sorted(sub.group_a.unique())
    mat=np.zeros((len(groups),len(groups)))
    for i,a in enumerate(groups):
        for j,b in enumerate(groups):
            val=sub[(sub.group_a==a)&(sub.group_b==b)].frobenius_divergence.iloc[0]
            mat[i,j]=val
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(mat)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups,rotation=30,ha='right')
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
    ax.set_title(f'Subtype-specific sheaf Laplacian divergence\n{label_col} | {protocol}')
    for i in range(len(groups)):
        for j in range(len(groups)):
            ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG/f'phase4_divergence_heatmap_{label_col}_{protocol}.png',dpi=180)
    plt.close(fig)

# Figure 2: accuracy comparison bars
plot_metrics=metrics.copy()
plot_metrics['label']=plot_metrics['task']+'\n'+plot_metrics['protocol']
fig,ax=plt.subplots(figsize=(11,6))
sub=plot_metrics[plot_metrics.method.isin(['baseline_logistic_features','counterfactual_min_energy_sheaf','hybrid_logistic_features_plus_sheaf_energies'])]
# choose compact tasks
sub=sub[sub['task'].isin(['idh_codel_subtype','grade_label','grade4_status'])]
# grouped bars manually
labels=list(dict.fromkeys(sub['label']))
methods=['baseline_logistic_features','counterfactual_min_energy_sheaf','hybrid_logistic_features_plus_sheaf_energies']
x=np.arange(len(labels)); width=0.25
for k,method in enumerate(methods):
    vals=[]
    for lab in labels:
        r=sub[(sub.label==lab)&(sub.method==method)]
        vals.append(float(r.balanced_accuracy.iloc[0]) if not r.empty else np.nan)
    ax.bar(x+(k-1)*width, vals, width, label=method.replace('_',' '))
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=8)
ax.set_ylabel('Balanced accuracy')
ax.set_title('Phase 4 leakage-aware counterfactual sheaf assignment and hybrid accuracy')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'phase4_balanced_accuracy_comparison.png',dpi=180); plt.close(fig)

# Figure 3 energy margins
pdat=patients[(patients.task=='idh_codel_subtype') & (patients.protocol=='strict_no_idh_no_clusters')]
if len(pdat):
    groups=sorted(pdat.true_label.unique())
    data=[pdat[pdat.true_label==g]['energy_margin'].dropna().values for g in groups]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.set_title('Counterfactual sheaf energy margin by true subtype')
    ax.set_ylabel('Second-lowest energy - lowest energy')
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout(); fig.savefig(FIG/'phase4_energy_margin_by_subtype.png',dpi=180); plt.close(fig)

summary={
    'n_patients': int(len(df)),
    'analysis_specs': [{'task':a,'protocol':b} for a,b in analysis_specs],
    'protocols': {k:v['description'] for k,v in PROTOCOLS.items()},
    'best_balanced_accuracy_rows': metrics.sort_values('balanced_accuracy',ascending=False).head(8).to_dict(orient='records'),
    'best_delta_rows': pd.read_csv(OUT/'phase4_accuracy_deltas.csv').sort_values('delta_balanced_accuracy',ascending=False).head(8).to_dict(orient='records'),
    'permutation_tests': pd.read_csv(OUT/'phase4_permutation_divergence_tests.csv').to_dict(orient='records'),
}
(OUT/'phase4_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2)[:4000])
'''
(PKG/'src/run_phase4.py').write_text(runner)

# Copy data
shutil.copy('/mnt/data/phase1_outputs/phase1_clean_encoded.csv', PKG/'data/phase1_clean_encoded.csv')
shutil.copy('/mnt/data/phase1_outputs/phase1_sris_results.csv', PKG/'data/phase1_sris_results.csv')
# Write requirements
(PKG/'requirements.txt').write_text('numpy\npandas\nscikit-learn\nscipy\nmatplotlib\n')

# Execute runner
subprocess.run([sys.executable, str(PKG/'src/run_phase4.py')], cwd=str(PKG/'src'), check=True)

# Load outputs for report
metrics=pd.read_csv(PKG/'results/phase4_counterfactual_accuracy_metrics.csv')
deltas=pd.read_csv(PKG/'results/phase4_accuracy_deltas.csv')
perms=pd.read_csv(PKG/'results/phase4_permutation_divergence_tests.csv')
edges=pd.read_csv(PKG/'results/phase4_group_edge_energy_summary.csv')
divs=pd.read_csv(PKG/'results/phase4_laplacian_divergences.csv')
summary=json.loads((PKG/'results/phase4_summary.json').read_text())

# Create README and method report
best_delta = deltas.sort_values('delta_balanced_accuracy', ascending=False).head(5)
readme = f'''# Phase 4: Subtype-Specific Counterfactual Sheaf Geometry

This package implements Phase 4 of the glioma sheaf project.

## New technical layer

Phase 4 builds subtype-specific/reference sheaves for groups such as IDH/codel subtype and grade. Each group has its own learned restriction maps, coboundary matrix, and sheaf Laplacian. Patients are then scored under every group sheaf, producing a counterfactual energy vector.

A patient is geometrically close to group g if the patient has low energy under the sheaf learned from group g.

## Main outputs

- `results/phase4_laplacian_divergences.csv`: pairwise distances between group-specific sheaf Laplacians.
- `results/phase4_permutation_divergence_tests.csv`: permutation tests for whether group sheaves differ beyond random labels.
- `results/phase4_counterfactual_accuracy_metrics.csv`: baseline, minimum-energy sheaf, and hybrid model accuracy.
- `results/phase4_counterfactual_patient_energies.csv`: patient-by-patient counterfactual energies under each group sheaf.
- `results/phase4_group_edge_energy_summary.csv`: dominant edge residuals by group.
- `figures/`: divergence heatmaps and accuracy plots.

## Best delta rows

{best_delta.to_markdown(index=False)}

## Run

```bash
pip install -r requirements.txt
python src/run_phase4.py
```
'''
(PKG/'README.md').write_text(readme)

report = f'''# Phase 4 Technical Report: Subtype-Specific Counterfactual Sheaf Geometry

## Purpose

Phases 1--3 established fixed and learned sheaf residuals, then tested subtype and survival associations. Phase 4 adds a more novel geometric layer: instead of learning one global tumor sheaf, we learn one sheaf per biological class and compare the resulting regulatory geometries.

## Method

For each label group g, Phase 4 fits restriction maps

r_{{uv}}^g(p) = W_{{uv}}^g x_u(p) - x_v(p),

for edges D->R, D->C, and R->C. The group-specific coboundary B_g and Laplacian L_g define the group regulatory geometry.

The counterfactual energy of patient p under group g is

E_g(p) = x_p^T L_g x_p.

The vector (E_g(p))_g is used for counterfactual assignment, margin analysis, and hybrid prediction.

## Why this is different from standard multi-omics GNNs

Standard multi-omics graph models usually learn embeddings or attention weights. Phase 4 instead compares learned local-to-global consistency laws. This produces interpretable geometric statements such as: an IDH-wildtype tumor is not only different in features; it is far from the IDH-mutant regulatory sheaf.

## Leakage-aware protocols

The package includes:

- full_geometry: biological discovery, not strict prediction;
- strict_no_idh_no_clusters: removes IDH and cluster-risk variables for subtype analysis;
- strict_no_grade_no_clusters: removes grade-risk and cluster-risk variables for grade analysis;
- strict_no_idh_no_grade_no_clusters: strictest protocol.

## Results summary

Permutation tests:

{perms.to_markdown(index=False)}

Best accuracy deltas:

{best_delta.to_markdown(index=False)}

## Interpretation

The strongest Phase 4 novelty is not simply a classifier. It is a subtype-specific sheaf geometry: each tumor group induces a different sheaf Laplacian, and patients can be scored by their energy under each counterfactual regulatory law. This creates a new set of features and biological interpretations that ordinary feature-concatenation or graph-aggregation models do not provide.

## Main limitation

The current data are still one-cohort internal data. Phase 4 improves the technical framework and adds rigorous permutation/accuracy analyses, but external cohort validation remains necessary before claiming definitive state-of-the-art performance.
'''
(PKG/'phase4_technical_report.md').write_text(report)

# Create LaTeX doc
tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,array,longtable}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{microtype}
\usepackage{listings}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\pagestyle{fancy}
\fancyhf{}
\lhead{Phase 4: Counterfactual Sheaf Geometry}
\rhead{Glioma Multi-Omics Project}
\cfoot{\thepage}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}
\newcommand{\R}{\mathbb{R}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\E}{\mathcal{E}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\title{Phase 4 Technical Specification\\Subtype-Specific Counterfactual Sheaf Geometry for Glioma Multi-Omics}
\author{Sheaf-Theoretic Quantification of Multi-Omics Regulatory Inconsistency in Brain Tumors}
\date{May 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Purpose of Phase 4}
Phases 1--3 built a patient-level sheaf regulatory inconsistency score, learned restriction maps, and tested clinical/survival endpoints. Phase 4 adds the strongest geometric novelty so far: rather than fitting one global tumor sheaf, we fit a separate cellular sheaf for each biological group and compare their learned regulatory geometries.

The central claim is not merely that a model predicts a label. The central claim is that distinct glioma groups induce distinct local-to-global consistency laws. A tumor can then be scored under each counterfactual group law, creating a new interpretable feature system.

\section{State Variables}
For each patient $p$, define node states
\[
  x_D(p)\in\R^{d_D},\qquad x_R(p)\in\R^{d_R},\qquad x_C(p)\in\R^{d_C},
\]
where $D$ is the genomic/DNA state, $R$ is the regulatory/transcriptomic state, and $C$ is the tumor phenotype state. The sheaf graph has directed biological edges
\[
  D\to R,\qquad D\to C,\qquad R\to C.
\]

\section{Group-Specific Reference Sheaves}
Let $G$ be a biological grouping variable, such as IDH/codel subtype or grade. For each group $g$, let
\[
  \mathcal{I}_g=\{p: G(p)=g\}
\]
be the set of patients in that group.

\begin{definition}[Group-specific restriction map]
For an edge $u\to v$ and group $g$, the Phase 4 restriction map is a ridge-estimated linear map
\[
  W_{uv}^{(g)}
  =\arg\min_W
  \sum_{p\in\mathcal{I}_g}
  \norm{W x_u(p)-x_v(p)}_2^2
  +\lambda\norm{W}_F^2.
\]
The group-specific residual is
\[
  r_{uv}^{(g)}(p)=W_{uv}^{(g)}x_u(p)-x_v(p).
\]
\end{definition}

In implementation, if row vectors are used, $x_uW_{uv}^{(g)}$ predicts $x_v$. The mathematics above uses column-vector notation.

\section{Coboundary and Sheaf Laplacian}
For each group $g$, define the coboundary matrix
\[
B_g=
\begin{bmatrix}
\left(W_{DR}^{(g)}\right)^T & -I_R & 0\\
\left(W_{DC}^{(g)}\right)^T & 0 & -I_C\\
0 & \left(W_{RC}^{(g)}\right)^T & -I_C
\end{bmatrix}.
\]
The group-specific sheaf Laplacian is
\[
  L_g=B_g^T B_g.
\]

\begin{definition}[Counterfactual sheaf energy]
The energy of patient $p$ under the regulatory law of group $g$ is
\[
  E_g(p)
  =x_p^T L_gx_p
  =\sum_{u\to v}\norm{r_{uv}^{(g)}(p)}_2^2,
\]
with edge-normalized implementation
\[
  E_g(p)=
  \sum_{u\to v}\frac{1}{d_v}
  \norm{W_{uv}^{(g)}x_u(p)-x_v(p)}_2^2.
\]
\end{definition}

The vector
\[
  \Phi(p)=\big(E_g(p)\big)_{g\in\mathcal{G}}
\]
is the Phase 4 counterfactual sheaf embedding.

\section{Counterfactual Assignment}
The minimum-energy assignment rule is
\[
  \widehat{G}(p)=\arg\min_{g\in\mathcal{G}} E_g(p).
\]
The energy margin is
\[
  m(p)=E_{(2)}(p)-E_{(1)}(p),
\]
where $E_{(1)}(p)$ and $E_{(2)}(p)$ are the lowest and second-lowest group energies. A large margin means that the patient's regulatory state is geometrically much more compatible with one group sheaf than with the alternatives.

A soft assignment is also defined by
\[
  \pi_g(p)=
  \frac{\exp(-E_g(p)/\tau)}
  {\sum_{h\in\mathcal{G}}\exp(-E_h(p)/\tau)}.
\]

\section{Laplacian Divergence Between Biological Groups}
To compare biological groups directly, normalize each Laplacian by
\[
  \widetilde L_g=\frac{L_g}{\norm{L_g}_F+\varepsilon}.
\]
Then define
\[
  \Delta_F(g,h)=\norm{\widetilde L_g-\widetilde L_h}_F.
\]
We also compute a low-spectrum distance
\[
  \Delta_{\lambda}(g,h)=
  \left(\sum_{k=1}^{K}(\lambda_k(\widetilde L_g)-\lambda_k(\widetilde L_h))^2\right)^{1/2}
\]
and a low-energy projector distance
\[
  \Delta_P(g,h)=\norm{P_g-P_h}_F,
\]
where $P_g$ projects onto the span of the lowest-energy eigenvectors of $\widetilde L_g$.

\section{Permutation Significance Test}
The null hypothesis is that group-specific sheaf geometries are no more separated than what would be obtained from random labels. Define the observed statistic
\[
  T_{obs}=\frac{1}{|\mathcal{G}|(|\mathcal{G}|-1)}
  \sum_{g\ne h}\Delta_F(g,h).
\]
For $B$ random label permutations, compute $T_b$. The permutation $p$-value is
\[
  p_{perm}=\frac{1+\#\{b:T_b\ge T_{obs}\}}{B+1}.
\]

\section{Leakage-Aware Protocols}
Phase 4 uses multiple protocols:
\begin{itemize}[leftmargin=1.5em]
\item \textbf{full geometry}: uses all Phase 1 state features for biological geometry discovery.
\item \textbf{strict no-IDH/no-cluster}: removes IDH and cluster-risk features for subtype analysis.
\item \textbf{strict no-grade/no-cluster}: removes grade-risk and cluster-risk features for grade analysis.
\item \textbf{strict no-IDH/no-grade/no-cluster}: strictest diagnostic protocol.
\end{itemize}
This separation is crucial: full geometry is useful for discovery, but strict protocols are needed for predictive claims.

\section{Accuracy Models}
The package reports three cross-validated methods:
\begin{enumerate}[leftmargin=1.5em]
\item baseline logistic regression on node features;
\item counterfactual minimum-energy sheaf assignment;
\item hybrid logistic regression using node features plus counterfactual sheaf energies.
\end{enumerate}
The metrics are accuracy, balanced accuracy, macro F1, and where meaningful, AUROC/AUPRC.

\section{Results Snapshot}
The following values are generated from the current internal cohort and should be treated as internal validation rather than final external performance claims.

\subsection{Permutation Tests}
\begin{center}
\small
\input{phase4_perm_table.tex}
\end{center}

\subsection{Best Accuracy Improvements}
\begin{center}
\small
\input{phase4_delta_table.tex}
\end{center}

\section{Figures}
\begin{figure}[h!]
\centering
\includegraphics[width=0.92\linewidth]{../figures/phase4_divergence_heatmap_idh_codel_subtype_strict_no_idh_no_clusters.png}
\caption{Strict subtype-specific sheaf Laplacian divergence.}
\end{figure}

\begin{figure}[h!]
\centering
\includegraphics[width=0.92\linewidth]{../figures/phase4_divergence_heatmap_grade_label_strict_no_grade_no_clusters.png}
\caption{Strict grade-specific sheaf Laplacian divergence.}
\end{figure}

\begin{figure}[h!]
\centering
\includegraphics[width=0.98\linewidth]{../figures/phase4_balanced_accuracy_comparison.png}
\caption{Cross-validated balanced accuracy for baseline, counterfactual sheaf assignment, and hybrid sheaf-energy models.}
\end{figure}

\section{Interpretation}
Phase 4 creates a new object: a biological class is represented not only by feature means or classifier coefficients, but by a sheaf Laplacian $L_g$ encoding its learned consistency law. Patients are evaluated by their counterfactual energy under each law. This moves the project from feature integration toward geometric comparison of regulatory regimes.

\section{Limitations}
The current Phase 4 results are internal. They strengthen the method and generate new features, but external CGGA-style validation is still needed before making definitive state-of-the-art performance claims. The strongest current claim is methodological: subtype-specific sheaf Laplacians and counterfactual sheaf energies provide an interpretable alternative to ordinary graph aggregation.

\section{Next Required Work}
\begin{enumerate}[leftmargin=1.5em]
\item Add an external cohort and compute $E_g(p)$ under TCGA-trained sheaves.
\item Test whether subgroup Laplacian divergences replicate across cohorts.
\item Add pathway-level or gene-level nodes to move beyond the current three-node macro-sheaf.
\item Use permutation and bootstrap confidence intervals for every reported gain.
\end{enumerate}

\end{document}
'''
# Generate tex tables
perm_slim=perms.copy()
perm_slim=perm_slim[['label_col','protocol','observed_mean_pairwise_frobenius','perm_mean','permutation_p_value','z_score_vs_permutation']]
perm_slim.columns=['Task','Protocol','Observed','Null mean','p-value','z']
# format for latex
def df_to_latex_input(df, max_rows=10):
    df=df.head(max_rows).copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c]=df[c].map(lambda x: f'{x:.4g}' if pd.notna(x) else '')
    return df.to_latex(index=False, escape=True, column_format='p{0.18\linewidth}p{0.28\linewidth}rrrr')
(PKG/'paper/phase4_perm_table.tex').write_text(df_to_latex_input(perm_slim, 10))

delta_slim=deltas.sort_values('delta_balanced_accuracy',ascending=False).head(8)
delta_slim=delta_slim[['task','protocol','method','baseline_balanced_accuracy','method_balanced_accuracy','delta_balanced_accuracy','delta_macro_f1']]
delta_slim.columns=['Task','Protocol','Method','Base BA','Method BA','Delta BA','Delta F1']
(PKG/'paper/phase4_delta_table.tex').write_text(df_to_latex_input(delta_slim,8).replace('p{0.18\\linewidth}p{0.28\\linewidth}rrrr','p{0.14\\linewidth}p{0.24\\linewidth}p{0.22\\linewidth}rrrr'))
(PKG/'paper/phase4_technical_specification.tex').write_text(tex)
# Compile LaTeX
subprocess.run(['pdflatex','-interaction=nonstopmode','phase4_technical_specification.tex'], cwd=str(PKG/'paper'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase4_technical_specification.tex'], cwd=str(PKG/'paper'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Copy PDF/TEX to /mnt/data root for link ease
if (PKG/'paper/phase4_technical_specification.pdf').exists():
    shutil.copy(PKG/'paper/phase4_technical_specification.pdf', BASE/'phase4_technical_specification.pdf')
shutil.copy(PKG/'paper/phase4_technical_specification.tex', BASE/'phase4_technical_specification.tex')

# zip package
zip_path = BASE/'phase4_subtype_sheaf_geometry_package.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in PKG.rglob('*'):
        z.write(f, f.relative_to(PKG.parent))

print('PACKAGE', zip_path)
print('PDF', BASE/'phase4_technical_specification.pdf')
print('METRICS')
print(metrics.to_string(max_rows=20))
print('DELTAS')
print(deltas.sort_values('delta_balanced_accuracy', ascending=False).head(12).to_string())
print('PERMS')
print(perms.to_string())
