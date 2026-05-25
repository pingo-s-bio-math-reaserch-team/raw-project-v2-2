"""
Phase 2 accuracy benchmark layer for learned regulatory sheaves.

This module evaluates whether learned sheaf residuals add predictive information
for clinically meaningful glioma endpoints.  It is intentionally separate from
the residual construction module so the paper can distinguish:

    (i) unsupervised / reference-sheaf inconsistency scoring, and
    (ii) supervised validation of those scores.

All reported metrics are out-of-fold cross-validated metrics from a simple
logistic meta-model.  Label-leakage controls are used for endpoint-specific
analyses:
    - IDH endpoint: use no_idh diagnostic sheaf variants and omit IDH features.
    - Grade endpoint: use no_grade diagnostic sheaf variants and omit grade features.
    - 24-month death endpoint: keep age/grade/IDH as covariates but do not use
      survival time/status in SRIS construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 20260523
EDGE_FEATURES = ["E_D_to_R", "E_D_to_C", "E_R_to_C"]
FRACTION_FEATURES = ["frac_D_to_R", "frac_D_to_C", "frac_R_to_C"]
SHEAF_FEATURES = ["SRIS"] + EDGE_FEATURES + FRACTION_FEATURES


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    target_column: str
    variant_filter: Optional[str]
    baseline_features: Tuple[str, ...]
    description: str


def make_24m_endpoint(df: pd.DataFrame, horizon: float = 24.0) -> pd.DataFrame:
    """Conservative 24-month death endpoint.

    y=1: deceased within horizon.
    y=0: observed beyond horizon.
    Exclude living/censored at or before horizon.
    """
    out = df.copy()
    months = pd.to_numeric(out["os_months"], errors="coerce")
    deceased = pd.to_numeric(out["deceased"], errors="coerce")
    y = pd.Series(np.nan, index=out.index, dtype=float)
    y[(deceased == 1.0) & (months <= horizon)] = 1.0
    y[months > horizon] = 0.0
    out["death_24m"] = y
    return out


def add_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = make_24m_endpoint(df)
    out["idh_mutant_endpoint"] = pd.to_numeric(out.get("idh_mutant"), errors="coerce")
    grade = pd.to_numeric(out.get("grade"), errors="coerce")
    out["grade4_endpoint"] = np.where(grade.notna(), (grade >= 4).astype(float), np.nan)
    return out


def available_features(df: pd.DataFrame, features: Sequence[str]) -> List[str]:
    return [f for f in features if f in df.columns]


def endpoint_specs(df: pd.DataFrame) -> List[EndpointSpec]:
    # Baselines are intentionally simple and reproducible.  They avoid the label
    # being tested but keep medically relevant non-leaking covariates.
    molecular_no_idh = (
        "age", "grade", "kps", "purity",
        "mgmt_methylated", "egfr_amp", "atrx_wt", "tert_promoter_mutant",
        "chr7_gain_chr10_loss", "mutation_count", "tmb", "aneuploidy",
        "tert_expr", "tert_expressed", "immune_score", "stromal_score",
        "rna_cluster_risk", "methyl_cluster_risk", "transcriptome_risk",
    )
    molecular_no_grade = (
        "age", "kps", "purity", "idh_mutant", "mgmt_methylated", "egfr_amp",
        "atrx_wt", "tert_promoter_mutant", "chr7_gain_chr10_loss",
        "mutation_count", "tmb", "aneuploidy", "tert_expr", "tert_expressed",
        "immune_score", "stromal_score", "rna_cluster_risk", "methyl_cluster_risk",
        "transcriptome_risk",
    )
    survival_baseline = (
        "age", "grade", "kps", "purity", "idh_mutant", "mgmt_methylated", "egfr_amp",
        "atrx_wt", "tert_promoter_mutant", "chr7_gain_chr10_loss", "mutation_count", "tmb",
        "aneuploidy", "tert_expr", "immune_score", "stromal_score", "rna_cluster_risk",
        "methyl_cluster_risk", "transcriptome_risk",
    )
    return [
        EndpointSpec(
            name="IDH mutant status",
            target_column="idh_mutant_endpoint",
            variant_filter="no_idh",
            baseline_features=tuple(available_features(df, molecular_no_idh)),
            description="Leakage-controlled endpoint: IDH features are removed from the sheaf and baseline.",
        ),
        EndpointSpec(
            name="Grade 4 status",
            target_column="grade4_endpoint",
            variant_filter="no_grade",
            baseline_features=tuple(available_features(df, molecular_no_grade)),
            description="Leakage-controlled endpoint: grade features are removed from the sheaf and baseline.",
        ),
        EndpointSpec(
            name="24-month death",
            target_column="death_24m",
            variant_filter=None,
            baseline_features=tuple(available_features(df, survival_baseline)),
            description="Conservative binary survival endpoint; living/censored before 24 months are excluded.",
        ),
    ]


def make_classifier() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RNG_SEED,
        )),
    ])


def cv_predictions(X: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y.astype(int))
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError("Not enough class support for stratified CV.")
    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_classifier()
    proba = cross_val_predict(clf, X, y.astype(int), cv=cv, method="predict_proba")[:, 1]
    label = (proba >= 0.5).astype(int)
    return proba, label


def metric_ci(y: np.ndarray, proba: np.ndarray, label: np.ndarray, metric_name: str, n_boot: int = 0, seed: int = RNG_SEED) -> Tuple[float, float]:
    # Confidence intervals are intentionally disabled in the default fast benchmark.
    # The out-of-fold point estimates are the primary Phase 2 accuracy quantities.
    return np.nan, np.nan


def compute_one_metric(y: np.ndarray, proba: np.ndarray, label: np.ndarray, metric_name: str) -> float:
    if metric_name == "accuracy":
        return accuracy_score(y, label)
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y, label)
    if metric_name == "precision":
        return precision_score(y, label, zero_division=0)
    if metric_name == "recall":
        return recall_score(y, label, zero_division=0)
    if metric_name == "f1":
        return f1_score(y, label, zero_division=0)
    if metric_name == "auroc":
        return roc_auc_score(y, proba)
    if metric_name == "auprc":
        return average_precision_score(y, proba)
    raise KeyError(metric_name)


def compute_metrics(y: np.ndarray, proba: np.ndarray, label: np.ndarray, n_features: int) -> Dict[str, float]:
    out: Dict[str, float] = {"n": int(len(y)), "positive_rate": float(np.mean(y)), "n_features": int(n_features)}
    for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
        val = compute_one_metric(y, proba, label, m)
        lo, hi = metric_ci(y, proba, label, m)
        out[m] = float(val)
        out[f"{m}_ci_low"] = lo
        out[f"{m}_ci_high"] = hi
    return out


def construct_design(df: pd.DataFrame, predictor_set: str, baseline_features: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    if predictor_set == "SRIS only":
        cols = ["SRIS"]
    elif predictor_set == "edge energies only":
        cols = available_features(df, EDGE_FEATURES)
    elif predictor_set == "sheaf full":
        cols = available_features(df, SHEAF_FEATURES)
    elif predictor_set == "baseline only":
        cols = available_features(df, baseline_features)
    elif predictor_set == "baseline + SRIS":
        cols = available_features(df, list(baseline_features) + ["SRIS"])
    elif predictor_set == "baseline + edge energies":
        cols = available_features(df, list(baseline_features) + EDGE_FEATURES)
    elif predictor_set == "baseline + sheaf full":
        cols = available_features(df, list(baseline_features) + SHEAF_FEATURES)
    else:
        raise ValueError(f"Unknown predictor set: {predictor_set}")
    if not cols:
        raise ValueError(f"No columns available for predictor set {predictor_set}")
    X = df[cols].to_numpy(dtype=float)
    return X, cols


def evaluate_endpoint_model(df: pd.DataFrame, endpoint: EndpointSpec, model_name: str, predictor_set: str, seed_offset: int = 0) -> Optional[Dict[str, object]]:
    tmp = df.dropna(subset=[endpoint.target_column]).copy()
    if tmp.empty:
        return None
    y = tmp[endpoint.target_column].astype(int).to_numpy()
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 3:
        return None
    X, cols = construct_design(tmp, predictor_set, endpoint.baseline_features)
    try:
        proba, label = cv_predictions(X, y, seed=RNG_SEED + seed_offset)
        metrics = compute_metrics(y, proba, label, n_features=len(cols))
    except Exception as exc:
        return {
            "endpoint": endpoint.name,
            "model": model_name,
            "predictor_set": predictor_set,
            "status": "failed",
            "note": str(exc),
        }
    return {
        "endpoint": endpoint.name,
        "model": model_name,
        "predictor_set": predictor_set,
        "status": "ok",
        "features": ";".join(cols),
        **metrics,
    }


def load_and_merge(encoded_path: Path, core_results_path: Path, diagnostic_results_path: Path) -> pd.DataFrame:
    encoded = add_endpoint_columns(pd.read_csv(encoded_path))
    core = pd.read_csv(core_results_path)
    diag = pd.read_csv(diagnostic_results_path)
    # Preserve all encoded columns; result files already carry patient_id and model-specific residuals.
    encoded_cols_to_add = [c for c in encoded.columns if c not in core.columns or c.endswith("_endpoint") or c == "death_24m"]
    core_merged = core.merge(encoded[["patient_id"] + [c for c in encoded_cols_to_add if c != "patient_id"]], on="patient_id", how="left")
    diag_merged = diag.merge(encoded[["patient_id"] + [c for c in encoded_cols_to_add if c != "patient_id"]], on="patient_id", how="left")
    return core_merged, diag_merged, encoded


def run_accuracy_benchmark(
    encoded_path: str | Path,
    core_results_path: str | Path,
    diagnostic_results_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, str]:
    encoded_path = Path(encoded_path)
    core_results_path = Path(core_results_path)
    diagnostic_results_path = Path(diagnostic_results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core, diagnostic, encoded = load_and_merge(encoded_path, core_results_path, diagnostic_results_path)
    specs = endpoint_specs(encoded)
    rows = []

    predictor_sets = [
        "SRIS only",
        "edge energies only",
        "sheaf full",
        "baseline only",
        "baseline + SRIS",
        "baseline + edge energies",
        "baseline + sheaf full",
    ]

    for endpoint in specs:
        if endpoint.variant_filter is None:
            candidate = core.copy()
            model_groups = list(candidate.groupby("model"))
        else:
            candidate = diagnostic[diagnostic["variant"] == endpoint.variant_filter].copy()
            model_groups = list(candidate.groupby(["variant", "model"]))
        for key, g in model_groups:
            model_name = key if isinstance(key, str) else "/".join(map(str, key))
            for j, predictor_set in enumerate(predictor_sets):
                # Avoid meaningless baseline repeated across every sheaf model? Keep it intentionally so delta can be model-local.
                row = evaluate_endpoint_model(g, endpoint, model_name, predictor_set, seed_offset=37 * j)
                if row is not None:
                    row["endpoint_description"] = endpoint.description
                    rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / "phase2_accuracy_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Add baseline-relative deltas for the primary accuracy metrics within each endpoint/model.
    delta_rows = []
    ok = metrics[metrics["status"] == "ok"].copy()
    metric_cols = ["accuracy", "balanced_accuracy", "f1", "auroc", "auprc"]
    for (endpoint_name, model), group in ok.groupby(["endpoint", "model"]):
        base = group[group["predictor_set"] == "baseline only"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            rec = {"endpoint": endpoint_name, "model": model, "predictor_set": row["predictor_set"]}
            for m in metric_cols:
                rec[f"delta_{m}_vs_baseline"] = float(row[m] - base_row[m])
            delta_rows.append(rec)
    deltas = pd.DataFrame(delta_rows)
    deltas_path = output_dir / "phase2_accuracy_deltas.csv"
    deltas.to_csv(deltas_path, index=False)

    # Best rows by endpoint/metric, emphasizing balanced accuracy and AUROC.
    best_rows = []
    for endpoint_name, group in ok.groupby("endpoint"):
        for metric in ["balanced_accuracy", "auroc", "auprc", "f1"]:
            idx = group[metric].astype(float).idxmax()
            row = group.loc[idx]
            best_rows.append({
                "endpoint": endpoint_name,
                "selection_metric": metric,
                "best_model": row["model"],
                "best_predictor_set": row["predictor_set"],
                "value": float(row[metric]),
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "f1": float(row["f1"]),
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
            })
    best = pd.DataFrame(best_rows)
    best_path = output_dir / "phase2_best_accuracy_rows.csv"
    best.to_csv(best_path, index=False)

    # Compact manuscript-oriented table: best non-baseline sheaf-increment row per endpoint/model.
    increment_rows = []
    for endpoint_name, group in ok.groupby("endpoint"):
        non_baseline = group[group["predictor_set"].isin(["baseline + SRIS", "baseline + edge energies", "baseline + sheaf full"])]
        if non_baseline.empty:
            continue
        idx = non_baseline["auroc"].astype(float).idxmax()
        row = non_baseline.loc[idx]
        base = group[(group["model"] == row["model"]) & (group["predictor_set"] == "baseline only")]
        delta_auc = np.nan
        delta_bal = np.nan
        if not base.empty:
            delta_auc = float(row["auroc"] - base.iloc[0]["auroc"])
            delta_bal = float(row["balanced_accuracy"] - base.iloc[0]["balanced_accuracy"])
        increment_rows.append({
            "endpoint": endpoint_name,
            "model": row["model"],
            "predictor_set": row["predictor_set"],
            "n": int(row["n"]),
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "f1": float(row["f1"]),
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
            "delta_auroc_vs_baseline": delta_auc,
            "delta_balanced_accuracy_vs_baseline": delta_bal,
        })
    inc = pd.DataFrame(increment_rows)
    inc_path = output_dir / "phase2_accuracy_increment_summary.csv"
    inc.to_csv(inc_path, index=False)

    report = {
        "n_encoded_patients": int(len(encoded)),
        "metric_protocol": "Stratified cross-validated logistic meta-model; threshold=0.5; class_weight=balanced; CI columns are disabled in the fast run and left blank.",
        "endpoints": [spec.__dict__ for spec in specs],
        "files": {
            "metrics": str(metrics_path),
            "deltas": str(deltas_path),
            "best_rows": str(best_path),
            "increment_summary": str(inc_path),
        },
        "increment_summary": inc.to_dict(orient="records"),
    }
    summary_path = output_dir / "phase2_accuracy_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "metrics": str(metrics_path),
        "deltas": str(deltas_path),
        "best_rows": str(best_path),
        "increment_summary": str(inc_path),
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 cross-validated accuracy benchmark.")
    parser.add_argument("--encoded", required=True)
    parser.add_argument("--core-results", required=True)
    parser.add_argument("--diagnostic-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = run_accuracy_benchmark(args.encoded, args.core_results, args.diagnostic_results, args.output_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
