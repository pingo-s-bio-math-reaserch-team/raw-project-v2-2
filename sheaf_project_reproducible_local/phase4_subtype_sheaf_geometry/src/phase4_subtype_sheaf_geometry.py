
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
