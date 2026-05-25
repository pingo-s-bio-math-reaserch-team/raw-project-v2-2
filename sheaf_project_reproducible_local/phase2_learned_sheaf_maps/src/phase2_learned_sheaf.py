"""
Phase 2 learned sheaf module for glioma regulatory inconsistency.

This module takes the Phase 1 encoded table and upgrades the fixed scalar-edge
sheaf into a learned vector-stalk sheaf.  For each edge u -> v, the edge stalk is
chosen to be the target-node space, and the residual is

    r_{uv}(p) = W_{uv} x_u(p) - x_v(p).

This avoids the trivial zero-map degeneracy that would occur if both sides of an
edge were freely learned.  The module supports cross-fitted learned maps,
biologically constrained nonnegative/risk-oriented maps, identity/random/shuffled
negative controls, and explicit coboundary/Laplacian construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


RNG_SEED = 20260523


@dataclass(frozen=True)
class NodeSpec:
    name: str
    features: Tuple[str, ...]


@dataclass(frozen=True)
class EdgeSpec:
    name: str
    source: str
    target: str


@dataclass(frozen=True)
class LearnedSheafSpec:
    nodes: Mapping[str, NodeSpec]
    edges: Tuple[EdgeSpec, ...]


@dataclass
class Phase2Config:
    alpha: float = 1.0
    n_splits: int = 5
    random_seed: int = RNG_SEED
    edge_normalize: bool = True


def default_feature_sets(exclude: Optional[Iterable[str]] = None) -> LearnedSheafSpec:
    """Return the Phase 2 three-node feature layout.

    exclude can remove feature groups used for leakage-controlled validation:
    - 'idh': removes idh_wt_z / idh_mutant-derived signal from D.
    - 'grade': removes grade_risk_z from C.
    - 'purity': removes purity_low_z from C.
    - 'kps': removes kps_low_z from C.
    """
    exclude = set(exclude or [])
    D = [
        "idh_wt_z", "mgmt_unmethylated_z", "atrx_wt_z", "tert_promoter_mutant_z",
        "chr7_gain_chr10_loss_z", "mutation_count_z", "tmb_z", "aneuploidy_z",
    ]
    R = [
        "egfr_amp_z", "tert_expr_z", "tert_expressed_z", "immune_score_z",
        "stromal_score_z", "rna_cluster_risk_z", "methyl_cluster_risk_z",
        "transcriptome_risk_z",
    ]
    C = ["grade_risk_z", "kps_low_z", "purity_low_z"]

    if "idh" in exclude:
        D = [f for f in D if f != "idh_wt_z"]
    if "grade" in exclude:
        C = [f for f in C if f != "grade_risk_z"]
    if "purity" in exclude:
        C = [f for f in C if f != "purity_low_z"]
    if "kps" in exclude:
        C = [f for f in C if f != "kps_low_z"]

    nodes = {
        "D": NodeSpec("D", tuple(D)),
        "R": NodeSpec("R", tuple(R)),
        "C": NodeSpec("C", tuple(C)),
    }
    edges = (
        EdgeSpec("D_to_R", "D", "R"),
        EdgeSpec("D_to_C", "D", "C"),
        EdgeSpec("R_to_C", "R", "C"),
    )
    return LearnedSheafSpec(nodes=nodes, edges=edges)


def _check_features(df: pd.DataFrame, spec: LearnedSheafSpec) -> None:
    missing = []
    for node in spec.nodes.values():
        missing.extend([f for f in node.features if f not in df.columns])
    if missing:
        raise KeyError(f"Encoded table is missing required Phase 2 features: {missing}")


def _node_matrix(df: pd.DataFrame, spec: LearnedSheafSpec, node: str) -> np.ndarray:
    X = df[list(spec.nodes[node].features)].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def ridge_map(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit W minimizing ||X W^T - Y||_F^2 + alpha ||W||_F^2.

    Returns W with shape (target_dim, source_dim), so predictions are X @ W.T.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_source = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_source)
    B = X.T @ Y
    Wt = np.linalg.solve(A, B)
    return Wt.T


def nonnegative_ridge_map(X: np.ndarray, Y: np.ndarray, alpha: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fit a target-by-source nonnegative ridge map via augmented NNLS.

    The variables are constrained W[j, k] >= 0 after Phase 1's risk-oriented
    feature transformations.  If mask[j, k] = 0, the coefficient is fixed to 0.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_source = X.shape[1]
    n_target = Y.shape[1]
    W = np.zeros((n_target, n_source), dtype=float)
    sqrt_alpha = math.sqrt(max(alpha, 0.0))

    for j in range(n_target):
        active = np.ones(n_source, dtype=bool) if mask is None else mask[j].astype(bool)
        if active.sum() == 0:
            continue
        X_active = X[:, active]
        X_aug = np.vstack([X_active, sqrt_alpha * np.eye(active.sum())])
        y_aug = np.concatenate([Y[:, j], np.zeros(active.sum())])
        coef, _ = nnls(X_aug, y_aug)
        W[j, active] = coef
    return W


def biological_mask(edge: EdgeSpec, spec: LearnedSheafSpec) -> np.ndarray:
    """Return a conservative structural mask for the constrained learned sheaf.

    The current Phase 2 variables are all risk-oriented proxies; therefore the
    constrained version uses nonnegative maps but does not over-prune target
    channels.  The mask is still explicit so future biological priors can be
    swapped in without changing the learning API.
    """
    target_dim = len(spec.nodes[edge.target].features)
    source_dim = len(spec.nodes[edge.source].features)
    mask = np.ones((target_dim, source_dim), dtype=int)

    # Conservative prior: mutation burden and TMB are nearly redundant, so do not
    # allow both to dominate the same map without regularization. Keep both active
    # here; the penalty handles redundancy. This placeholder is intentionally
    # explicit for extension in Phase 3 gene-level graphs.
    return mask


def identity_projection_map(source_dim: int, target_dim: int) -> np.ndarray:
    W = np.zeros((target_dim, source_dim), dtype=float)
    m = min(source_dim, target_dim)
    W[:m, :m] = np.eye(m)
    return W


def random_map(source_dim: int, target_dim: int, rng: np.random.Generator) -> np.ndarray:
    scale = 1.0 / math.sqrt(max(source_dim, 1))
    return rng.normal(loc=0.0, scale=scale, size=(target_dim, source_dim))




def reference_mask(df: pd.DataFrame) -> np.ndarray:
    """Low-risk/coherent reference subset for reference-sheaf learning.

    The reference group is not used as a prediction label; it is used to learn a
    baseline regulatory law against which all tumors are scored.  Priority is
    given to IDH-mutant/codeleted and grade-2 tumors, which are clinically lower
    risk within diffuse glioma.
    """
    mask = pd.Series(False, index=df.index)
    if "idh_codel_subtype" in df:
        mask = mask | (df["idh_codel_subtype"].astype(str) == "IDHmut-codel")
    if "grade" in df and "idh_mutant" in df:
        mask = mask | ((pd.to_numeric(df["grade"], errors="coerce") <= 2.0) & (pd.to_numeric(df["idh_mutant"], errors="coerce") == 1.0))
    # Fallback if labels are unavailable or too few.
    if int(mask.sum()) < 30:
        if "grade" in df:
            mask = pd.to_numeric(df["grade"], errors="coerce") <= 2.0
    if int(mask.sum()) < 20:
        mask[:] = True
    return mask.to_numpy(dtype=bool)


def fit_edge_map(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    edge: EdgeSpec,
    spec: LearnedSheafSpec,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if method in {"ridge_unconstrained", "reference_ridge_lowrisk"}:
        return ridge_map(X, Y, alpha=alpha)
    if method in {"bio_constrained_nonnegative", "reference_bio_constrained_lowrisk"}:
        return nonnegative_ridge_map(X, Y, alpha=alpha, mask=biological_mask(edge, spec))
    if method == "identity_projection":
        return identity_projection_map(X.shape[1], Y.shape[1])
    if method == "random_projection":
        return random_map(X.shape[1], Y.shape[1], rng)
    if method == "shuffled_target_ridge":
        perm = rng.permutation(len(Y))
        return ridge_map(X, Y[perm], alpha=alpha)
    raise ValueError(f"Unknown Phase 2 method: {method}")


def edge_residuals(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = X @ W.T - Y
    # Mean squared residual per edge so higher-dimensional edge stalks do not dominate.
    E = np.mean(R ** 2, axis=1)
    return R, E


def crossfit_sris(
    df: pd.DataFrame,
    spec: LearnedSheafSpec,
    method: str,
    config: Phase2Config,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Compute cross-fitted SRIS for a model/method.

    For learned methods, each patient's residuals are evaluated on a fold not used
    to fit the restriction maps.  For random/identity negative controls, maps are
    deterministic and no data fitting occurs.
    """
    _check_features(df, spec)
    n = len(df)
    rng = np.random.default_rng(config.random_seed + abs(hash(method)) % 100000)

    learned = method in {"ridge_unconstrained", "bio_constrained_nonnegative", "shuffled_target_ridge"}
    reference_learned = method in {"reference_ridge_lowrisk", "reference_bio_constrained_lowrisk"}
    fold_ids = np.zeros(n, dtype=int)
    edge_energy = {edge.name: np.zeros(n, dtype=float) for edge in spec.edges}
    edge_norm = {edge.name: np.zeros(n, dtype=float) for edge in spec.edges}
    edge_dim = {}
    maps_by_fold: Dict[str, List[List[List[float]]]] = {edge.name: [] for edge in spec.edges}

    if learned:
        splitter = KFold(n_splits=min(config.n_splits, n), shuffle=True, random_state=config.random_seed)
        folds = list(splitter.split(np.arange(n)))
    elif reference_learned:
        ref_idx = np.where(reference_mask(df))[0]
        folds = [(ref_idx, np.arange(n))]
    else:
        folds = [(np.arange(n), np.arange(n))]

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        for edge in spec.edges:
            X_all = _node_matrix(df, spec, edge.source)
            Y_all = _node_matrix(df, spec, edge.target)
            edge_dim[edge.name] = int(Y_all.shape[1])
            X_train, Y_train = X_all[train_idx], Y_all[train_idx]
            X_test, Y_test = X_all[test_idx], Y_all[test_idx]
            W = fit_edge_map(X_train, Y_train, method, edge, spec, alpha=config.alpha, rng=rng)
            R_test, E_test = edge_residuals(X_test, Y_test, W)
            edge_energy[edge.name][test_idx] = E_test
            edge_norm[edge.name][test_idx] = np.linalg.norm(R_test, axis=1)
            fold_ids[test_idx] = fold_idx
            maps_by_fold[edge.name].append(W.tolist())

    sris = np.zeros(n, dtype=float)
    for edge in spec.edges:
        sris += edge_energy[edge.name]

    out = pd.DataFrame({
        "patient_id": df.get("patient_id", pd.Series(range(n))).values,
        "sample_id": df.get("sample_id", pd.Series([None] * n)).values,
        "model": method,
        "variant": "core",
        "fold": fold_ids,
        "SRIS": sris,
    })
    for edge in spec.edges:
        out[f"E_{edge.name}"] = edge_energy[edge.name]
        out[f"norm_{edge.name}"] = edge_norm[edge.name]
        out[f"frac_{edge.name}"] = np.where(sris > 0, edge_energy[edge.name] / sris, 0.0)

    for col in [
        "idh_mutant", "idh_wt", "idh_codel_subtype", "grade", "grade_risk", "mgmt_methylated",
        "egfr_amp", "os_months", "deceased", "age", "kps", "purity", "transcriptome_subtype",
        "methylation_cluster", "rna_cluster",
    ]:
        if col in df.columns:
            out[col] = df[col].values

    # Fit full-data maps for exportable Laplacian metadata.
    full_maps: Dict[str, List[List[float]]] = {}
    for edge in spec.edges:
        X_all = _node_matrix(df, spec, edge.source)
        Y_all = _node_matrix(df, spec, edge.target)
        full_rng = np.random.default_rng(config.random_seed + 1007 + abs(hash(method + edge.name)) % 100000)
        W = fit_edge_map(X_all, Y_all, method, edge, spec, alpha=config.alpha, rng=full_rng)
        full_maps[edge.name] = W.tolist()

    metadata = {
        "method": method,
        "alpha": config.alpha,
        "n_splits": config.n_splits,
        "edge_dim": edge_dim,
        "crossfit": learned,
        "reference_learned": reference_learned,
        "n_reference_train": int(reference_mask(df).sum()) if reference_learned else None,
        "maps_by_fold": maps_by_fold,
        "full_data_maps": full_maps,
    }
    return out, metadata


def build_coboundary_from_maps(spec: LearnedSheafSpec, full_maps: Mapping[str, Sequence[Sequence[float]]]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Construct vector-stalk coboundary matrix from full-data edge maps."""
    feature_names: List[str] = []
    node_slices: Dict[str, slice] = {}
    start = 0
    for node_name in ["D", "R", "C"]:
        feats = list(spec.nodes[node_name].features)
        feature_names.extend(feats)
        node_slices[node_name] = slice(start, start + len(feats))
        start += len(feats)

    rows = []
    row_names = []
    for edge in spec.edges:
        W = np.asarray(full_maps[edge.name], dtype=float)
        target_dim, source_dim = W.shape
        row_block = np.zeros((target_dim, len(feature_names)), dtype=float)
        s_slice = node_slices[edge.source]
        t_slice = node_slices[edge.target]
        row_block[:, s_slice] = W
        row_block[:, t_slice] = -np.eye(target_dim)
        rows.append(row_block)
        row_names.extend([f"{edge.name}[{j}]" for j in range(target_dim)])
    B = np.vstack(rows) if rows else np.zeros((0, len(feature_names)))
    return B, feature_names, row_names


def model_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in results.groupby("model"):
        row = {
            "model": model,
            "n": len(group),
            "sris_mean": group["SRIS"].mean(),
            "sris_median": group["SRIS"].median(),
            "sris_std": group["SRIS"].std(),
        }
        for edge in ["D_to_R", "D_to_C", "R_to_C"]:
            col = f"E_{edge}"
            if col in group:
                row[f"mean_{col}"] = group[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def group_tests(results: pd.DataFrame) -> pd.DataFrame:
    """Exploratory model diagnostics; not final causal claims."""
    rows = []
    group_cols = ["model"] + (["variant"] if "variant" in results.columns else [])
    for keys, g in results.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        if "idh_mutant" in g and g["idh_mutant"].nunique() == 2:
            a = g.loc[g["idh_mutant"] == 0, "SRIS"]
            b = g.loc[g["idh_mutant"] == 1, "SRIS"]
            if len(a) > 1 and len(b) > 1:
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append({**base, "test": "SRIS by IDH mutant status", "statistic": u.statistic, "pvalue": u.pvalue})
        if "grade" in g and g["grade"].nunique() > 2:
            groups = [h["SRIS"].to_numpy() for _, h in g.groupby("grade") if len(h) > 1]
            if len(groups) > 2:
                kw = stats.kruskal(*groups)
                rows.append({**base, "test": "SRIS by grade", "statistic": kw.statistic, "pvalue": kw.pvalue})
        if "age" in g:
            valid = g[["SRIS", "age"]].dropna()
            if len(valid) > 3:
                sp = stats.spearmanr(valid["SRIS"], valid["age"])
                rows.append({**base, "test": "SRIS vs age external Spearman", "statistic": sp.statistic, "pvalue": sp.pvalue})
    return pd.DataFrame(rows)


def make_survival_endpoint(df: pd.DataFrame, horizon_months: float = 24.0) -> pd.DataFrame:
    """Create a conservative binary endpoint for quick validation.

    y=1: deceased within horizon.
    y=0: observed beyond horizon, regardless of final status.
    Exclude living/censored before horizon.
    """
    tmp = df.copy()
    months = pd.to_numeric(tmp["os_months"], errors="coerce")
    deceased = pd.to_numeric(tmp["deceased"], errors="coerce")
    y = pd.Series(np.nan, index=tmp.index, dtype=float)
    y[(deceased == 1) & (months <= horizon_months)] = 1.0
    y[months > horizon_months] = 0.0
    tmp["event_within_24m"] = y
    return tmp.dropna(subset=["event_within_24m"])




def reconstruction_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in results.groupby("model"):
        for edge in ["D_to_R", "D_to_C", "R_to_C"]:
            col = f"E_{edge}"
            if col not in g:
                continue
            mse = float(g[col].mean())
            rows.append({"model": model, "edge": edge, "crossfit_or_reference_mse": mse, "median_edge_energy": float(g[col].median())})
        rows.append({"model": model, "edge": "ALL", "crossfit_or_reference_mse": float(g["SRIS"].mean()), "median_edge_energy": float(g["SRIS"].median())})
    return pd.DataFrame(rows)


def validation_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Quick no-survival-leak validation using the 24-month event endpoint.

    This is not a replacement for Cox modeling; it is an immediate Phase 2 sanity
    check because lifelines/survival libraries may not be available in every runtime.
    """
    rows = []
    for model, g in results.groupby("model"):
        tmp = make_survival_endpoint(g)
        if len(tmp) < 20 or tmp["event_within_24m"].nunique() != 2:
            continue
        y = tmp["event_within_24m"].astype(int).to_numpy()
        sris = tmp[["SRIS"]].to_numpy()
        auc_raw = roc_auc_score(y, tmp["SRIS"])
        auc_raw_flipped = max(auc_raw, 1.0 - auc_raw)
        rows.append({"model": model, "endpoint": "24m death", "predictors": "SRIS only", "n": len(tmp), "AUROC_orientation_free": auc_raw_flipped})

        # Cross-validated logistic: age only vs age + SRIS, when both classes have enough cases.
        if "age" in tmp and tmp["age"].notna().all() and min(np.bincount(y)) >= 5:
            cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=RNG_SEED)
            X_age = tmp[["age"]].to_numpy(dtype=float)
            X_age_sris = tmp[["age", "SRIS"]].to_numpy(dtype=float)
            for name, X in [("age only", X_age), ("age + SRIS", X_age_sris)]:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=RNG_SEED))
                try:
                    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
                    auc = roc_auc_score(y, prob)
                    rows.append({"model": model, "endpoint": "24m death", "predictors": name, "n": len(tmp), "AUROC_orientation_free": auc})
                except Exception as exc:
                    rows.append({"model": model, "endpoint": "24m death", "predictors": name, "n": len(tmp), "AUROC_orientation_free": np.nan, "note": str(exc)})
    return pd.DataFrame(rows)


def run_phase2(encoded_path: str | Path, output_dir: str | Path, config: Optional[Phase2Config] = None) -> Dict[str, str]:
    config = config or Phase2Config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(encoded_path)
    spec = default_feature_sets()
    methods = [
        "ridge_unconstrained",
        "bio_constrained_nonnegative",
        "reference_ridge_lowrisk",
        "reference_bio_constrained_lowrisk",
        "identity_projection",
        "random_projection",
        "shuffled_target_ridge",
    ]
    all_results = []
    all_metadata = {"config": config.__dict__, "models": {}, "spec": {
        "nodes": {name: list(node.features) for name, node in spec.nodes.items()},
        "edges": [edge.__dict__ for edge in spec.edges],
    }}

    for method in methods:
        res, meta = crossfit_sris(df, spec, method, config)
        all_results.append(res)
        B, feature_names, row_names = build_coboundary_from_maps(spec, meta["full_data_maps"])
        L = B.T @ B
        meta["full_data_coboundary_shape"] = list(B.shape)
        meta["full_data_laplacian_shape"] = list(L.shape)
        meta["full_data_feature_names"] = feature_names
        meta["full_data_coboundary_row_names"] = row_names
        meta["full_data_coboundary_matrix"] = B.tolist()
        meta["full_data_laplacian_matrix"] = L.tolist()
        all_metadata["models"][method] = meta

    results = pd.concat(all_results, ignore_index=True)
    summary = model_summary(results)
    tests = group_tests(results)
    metrics = validation_metrics(results)
    recon = reconstruction_metrics(results)

    # Include leakage-controlled IDH/grade variants as diagnostic outputs.
    # These are not the main Phase 2 learned maps, but they prevent overclaiming.
    diagnostic_rows = []
    for variant, exclude in [("no_idh", ["idh"]), ("no_grade", ["grade"]), ("no_grade_no_idh", ["grade", "idh"] )]:
        variant_spec = default_feature_sets(exclude=exclude)
        for method in ["ridge_unconstrained", "bio_constrained_nonnegative"]:
            res, _ = crossfit_sris(df, variant_spec, method, config)
            res["variant"] = variant
            diagnostic_rows.append(res)
    diagnostic_results = pd.concat(diagnostic_rows, ignore_index=True)
    diagnostic_tests = group_tests(diagnostic_results)
    diagnostic_summary = model_summary(diagnostic_results)

    paths = {
        "phase2_sris_all_models": output_dir / "phase2_sris_all_models.csv",
        "phase2_model_summary": output_dir / "phase2_model_summary.csv",
        "phase2_group_tests": output_dir / "phase2_group_tests.csv",
        "phase2_validation_metrics": output_dir / "phase2_validation_metrics.csv",
        "phase2_reconstruction_metrics": output_dir / "phase2_reconstruction_metrics.csv",
        "phase2_diagnostic_sris": output_dir / "phase2_diagnostic_sris.csv",
        "phase2_diagnostic_tests": output_dir / "phase2_diagnostic_tests.csv",
        "phase2_diagnostic_summary": output_dir / "phase2_diagnostic_summary.csv",
        "phase2_maps_and_laplacians": output_dir / "phase2_maps_and_laplacians.json",
    }
    results.to_csv(paths["phase2_sris_all_models"], index=False)
    summary.to_csv(paths["phase2_model_summary"], index=False)
    tests.to_csv(paths["phase2_group_tests"], index=False)
    metrics.to_csv(paths["phase2_validation_metrics"], index=False)
    recon.to_csv(paths["phase2_reconstruction_metrics"], index=False)
    diagnostic_results.to_csv(paths["phase2_diagnostic_sris"], index=False)
    diagnostic_tests.to_csv(paths["phase2_diagnostic_tests"], index=False)
    diagnostic_summary.to_csv(paths["phase2_diagnostic_summary"], index=False)
    paths["phase2_maps_and_laplacians"].write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")

    report = {
        "n_patients": int(len(df)),
        "core_models": methods,
        "sris_summary": summary.to_dict(orient="records"),
        "group_tests": tests.to_dict(orient="records"),
        "validation_metrics": metrics.to_dict(orient="records"),
        "reconstruction_metrics": recon.to_dict(orient="records"),
        "diagnostic_note": "no_idh/no_grade variants are provided to prevent leakage in downstream label-specific validation.",
    }
    report_path = output_dir / "phase2_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["phase2_summary"] = report_path

    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 learned sheaf experiments.")
    parser.add_argument("--encoded", default="data/phase1_clean_encoded.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    paths = run_phase2(args.encoded, args.output_dir, Phase2Config(alpha=args.alpha, n_splits=args.n_splits))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
