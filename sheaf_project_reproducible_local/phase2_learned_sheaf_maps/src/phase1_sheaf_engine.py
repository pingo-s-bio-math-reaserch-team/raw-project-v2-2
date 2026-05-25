"""
Phase 1 sheaf residual engine for glioma multi-omics inconsistency.

This module replaces the early rule-based Consistancy prototype with a formal
cellular-sheaf implementation over three biological nodes:

    D = DNA / genomic state
    R = regulatory / transcriptomic state
    C = tumor/clinical phenotype state, excluding survival outcome labels and patient age by default

For each patient p, the engine constructs node vectors x_D(p), x_R(p), x_C(p),
builds a sheaf coboundary matrix B_F from biologically oriented restriction maps,
and computes

    SRIS(p) = ||B_F x_p||_2^2 = x_p^T L_F x_p,   L_F = B_F^T B_F.

The goal is not to claim final biological discovery yet. The goal is to create a
rigorous, reproducible Phase 1 backbone that can support subtype validation,
survival testing, learned maps, ablations, and OT robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import argparse
import json
import math
import re

import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "NA", "N/A", "NONE", "NULL", "NAN", "nan", "None", None}


@dataclass(frozen=True)
class NodeDefinition:
    """A node/stalk in the three-node Phase 1 sheaf."""

    name: str
    features: Tuple[str, ...]
    description: str


@dataclass(frozen=True)
class EdgeDefinition:
    """A sheaf edge with scalar edge stalk and two restriction maps.

    The source_weights and target_weights dictionaries define the row vectors
    A_e and B_e for the residual

        r_e(p) = A_e x_source(p) - B_e x_target(p).

    The vectors are normalized to unit L2 norm when the coboundary matrix is
    built, making edge energies more comparable across differently sized nodes.
    """

    name: str
    source: str
    target: str
    source_weights: Mapping[str, float]
    target_weights: Mapping[str, float]
    rationale: str


@dataclass
class SheafSpec:
    nodes: Dict[str, NodeDefinition]
    edges: List[EdgeDefinition]


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in MISSING_STRINGS:
        return True
    return False


def _clean_str(value) -> Optional[str]:
    if _is_missing(value):
        return None
    return str(value).strip()


def _to_float(value) -> float:
    if _is_missing(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _binary_from_allowed(value, positive: Iterable[str], negative: Iterable[str]) -> float:
    """Return 1/0/NaN from string-coded categories."""
    value = _clean_str(value)
    if value is None:
        return np.nan
    v = value.lower().replace(" ", "")
    pos = {p.lower().replace(" ", "") for p in positive}
    neg = {n.lower().replace(" ", "") for n in negative}
    if v in pos:
        return 1.0
    if v in neg:
        return 0.0
    return np.nan


def _parse_grade(value) -> float:
    s = _clean_str(value)
    if s is None:
        return np.nan
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    g = float(m.group(1))
    return g if g in {1.0, 2.0, 3.0, 4.0} else np.nan


def _parse_cluster_risk(value: object, prefix: str, max_index: int) -> float:
    """Map LGm1/LGr1/... style clusters to [0,1] ordinal risk as a Phase 1 proxy.

    This is deliberately marked as a proxy: it should be replaced or ablated in
    later phases with learned maps or known subtype labels.
    """
    s = _clean_str(value)
    if s is None:
        return np.nan
    m = re.search(re.escape(prefix) + r"(\d+)", s, flags=re.IGNORECASE)
    if not m:
        return np.nan
    idx = int(m.group(1))
    if idx < 1 or idx > max_index:
        return np.nan
    if max_index == 1:
        return 0.0
    return (idx - 1) / (max_index - 1)


def _status_deceased(value) -> float:
    s = _clean_str(value)
    if s is None:
        return np.nan
    sl = s.lower()
    if "deceased" in sl or sl.startswith("1"):
        return 1.0
    if "living" in sl or sl.startswith("0"):
        return 0.0
    return np.nan


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mu = values.mean(skipna=True)
    sigma = values.std(skipna=True)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mu) / sigma


def _mean_impute(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mu = values.mean(skipna=True)
    if not np.isfinite(mu):
        mu = 0.0
    return values.fillna(mu)


def _binary_impute(series: pd.Series, fill: float = 0.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(fill)


def load_raw_table(path: str | Path) -> pd.DataFrame:
    """Load either the whitespace-separated data.txt or a CSV exported later."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=r"\s+", engine="python")


def encode_glioma_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a reproducible Phase 1 encoded table from the raw glioma table.

    The encoded table keeps survival labels and patient age separate from the sheaf phenotype
    node. This prevents leakage/confounding when SRIS is later tested against
    survival or age-associated biology.
    """
    df = pd.DataFrame(index=raw.index)

    # Identifiers and labels.
    df["patient_id"] = raw.get("PatientID")
    df["sample_id"] = raw.get("SampleID")
    df["idh_codel_subtype"] = raw.get("IDH/codelsubtype")
    df["transcriptome_subtype"] = raw.get("TranscriptomeSubtype")
    df["methylation_cluster"] = raw.get("Pan-GliomaDNAMethylationCluster")
    df["rna_cluster"] = raw.get("Pan-GliomaRNAExpressionCluster")

    # Direct clinical/outcome labels retained for downstream validation. Age is
    # intentionally not included in the default C node because it is a patient-level
    # covariate/result to analyze against SRIS, not a tumor inconsistency variable.
    df["age"] = raw.get("DiagnosisAge", pd.Series(index=raw.index)).map(_to_float)
    df["grade"] = raw.get("NeoplasmHistologicGrade", pd.Series(index=raw.index)).map(_parse_grade)
    df["grade_risk"] = ((df["grade"] - 2.0) / 2.0).clip(lower=0.0, upper=1.0)
    df["os_months"] = raw.get("Months", pd.Series(index=raw.index)).map(_to_float)
    df["deceased"] = raw.get("Status", pd.Series(index=raw.index)).map(_status_deceased)
    df["kps"] = raw.get("KarnofskyPerformanceScore", pd.Series(index=raw.index)).map(_to_float)
    df["kps_low"] = 1.0 - (df["kps"] / 100.0).clip(lower=0.0, upper=1.0)
    df["purity"] = raw.get("AbsolutePurity", pd.Series(index=raw.index)).map(_to_float)
    df["purity_low"] = 1.0 - df["purity"].clip(lower=0.0, upper=1.0)

    # DNA/genomic variables.
    df["idh_mutant"] = raw.get("IDH", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"Mutant"}, negative={"WT", "Wildtype", "Wild-type"})
    )
    df["idh_wt"] = 1.0 - df["idh_mutant"]
    df["mgmt_methylated"] = raw.get("MGMT", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"Methylated"}, negative={"Unmethylated"})
    )
    df["mgmt_unmethylated"] = 1.0 - df["mgmt_methylated"]
    df["atrx_mutant"] = raw.get("ATRXstatus", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"Mutant"}, negative={"WT"})
    )
    df["atrx_wt"] = 1.0 - df["atrx_mutant"]
    df["tert_promoter_mutant"] = raw.get("TERTpromoterstatus", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"Mutant"}, negative={"WT"})
    )
    df["chr7_gain_chr10_loss"] = raw.get("Chr7gain/Chr10loss", pd.Series(index=raw.index)).map(
        lambda x: 1.0 if isinstance(x, str) and "gainchr7" in x.lower() and "losschr10" in x.lower() else (0.0 if not _is_missing(x) else np.nan)
    )
    df["mutation_count"] = raw.get("MutationCount", pd.Series(index=raw.index)).map(_to_float)
    df["tmb"] = raw.get("TMB(nonsynonymous)", pd.Series(index=raw.index)).map(_to_float)
    df["aneuploidy"] = raw.get("Percentaneuploidy", pd.Series(index=raw.index)).map(_to_float)

    # Regulatory/transcriptomic variables.
    df["egfr_amp"] = raw.get("EGFR", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"amp", "amplified"}, negative={"normal", "WT"})
    )
    df["tert_expr"] = raw.get("TERTexpression(log2)", pd.Series(index=raw.index)).map(_to_float)
    df["tert_expressed"] = raw.get("TERTexpressionstatus", pd.Series(index=raw.index)).map(
        lambda x: _binary_from_allowed(x, positive={"Expressed"}, negative={"Notexpressed", "Not expressed"})
    )
    df["immune_score"] = raw.get("ESTIMATEimmunescore", pd.Series(index=raw.index)).map(_to_float)
    df["stromal_score"] = raw.get("ESTIMATEstromalscore", pd.Series(index=raw.index)).map(_to_float)
    df["estimate_score"] = raw.get("ESTIMATEcombinedscore", pd.Series(index=raw.index)).map(_to_float)
    df["rna_cluster_risk"] = raw.get("Pan-GliomaRNAExpressionCluster", pd.Series(index=raw.index)).map(
        lambda x: _parse_cluster_risk(x, prefix="LGr", max_index=4)
    )
    df["methyl_cluster_risk"] = raw.get("Pan-GliomaDNAMethylationCluster", pd.Series(index=raw.index)).map(
        lambda x: _parse_cluster_risk(x, prefix="LGm", max_index=6)
    )
    df["transcriptome_risk"] = raw.get("TranscriptomeSubtype", pd.Series(index=raw.index)).map(
        lambda x: {"PN": 0.20, "NE": 0.45, "CL": 0.75, "ME": 0.85}.get(_clean_str(x), np.nan)
    )

    # Missingness flags for transparency and later ablations.
    core_raw_features = [
        "age", "grade", "kps", "purity", "idh_mutant", "mgmt_methylated",
        "atrx_mutant", "tert_promoter_mutant", "chr7_gain_chr10_loss",
        "mutation_count", "tmb", "aneuploidy", "egfr_amp", "tert_expr",
        "tert_expressed", "immune_score", "stromal_score", "rna_cluster_risk",
        "methyl_cluster_risk", "transcriptome_risk",
    ]
    for col in core_raw_features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Impute raw encoded values before standardization. Binary unknowns get 0.5;
    # continuous values get cohort means. This keeps all patients in Phase 1 while
    # still exposing missingness flags.
    binary_cols = [
        "idh_mutant", "idh_wt", "mgmt_methylated", "mgmt_unmethylated",
        "atrx_mutant", "atrx_wt", "tert_promoter_mutant", "chr7_gain_chr10_loss",
        "egfr_amp", "tert_expressed", "deceased",
    ]
    for col in binary_cols:
        if col in df:
            df[col] = _binary_impute(df[col], fill=0.5)

    continuous_cols = [
        "age", "grade", "grade_risk", "os_months", "kps", "kps_low", "purity", "purity_low",
        "mutation_count", "tmb", "aneuploidy", "tert_expr", "immune_score",
        "stromal_score", "estimate_score", "rna_cluster_risk", "methyl_cluster_risk",
        "transcriptome_risk",
    ]
    for col in continuous_cols:
        if col in df:
            df[col] = _mean_impute(df[col])

    # Standardized risk-oriented feature columns used in the sheaf. Suffix _z means
    # standardized after imputation, not necessarily original z-scored assay data.
    sheaf_base_cols = [
        "idh_wt", "mgmt_unmethylated", "atrx_wt", "tert_promoter_mutant",
        "chr7_gain_chr10_loss", "mutation_count", "tmb", "aneuploidy",
        "egfr_amp", "tert_expr", "tert_expressed", "immune_score", "stromal_score",
        "rna_cluster_risk", "methyl_cluster_risk", "transcriptome_risk",
        "grade_risk", "age", "kps_low", "purity_low",
    ]
    for col in sheaf_base_cols:
        df[f"{col}_z"] = _zscore(df[col]).fillna(0.0)

    return df


def build_fixed_phase1_sheaf() -> SheafSpec:
    """Build a biologically oriented fixed-map sheaf for Phase 1.

    The maps are intended as an interpretable baseline. They are deliberately
    simple and normalized. Later phases should learn/ablate these maps.
    """
    nodes = {
        "D": NodeDefinition(
            name="D",
            description="DNA/genomic risk state",
            features=(
                "idh_wt_z", "mgmt_unmethylated_z", "atrx_wt_z", "tert_promoter_mutant_z",
                "chr7_gain_chr10_loss_z", "mutation_count_z", "tmb_z", "aneuploidy_z",
            ),
        ),
        "R": NodeDefinition(
            name="R",
            description="regulatory/transcriptomic risk state",
            features=(
                "egfr_amp_z", "tert_expr_z", "tert_expressed_z", "immune_score_z",
                "stromal_score_z", "rna_cluster_risk_z", "methyl_cluster_risk_z",
                "transcriptome_risk_z",
            ),
        ),
        "C": NodeDefinition(
            name="C",
            description="tumor/clinical phenotype state without survival or age leakage",
            features=("grade_risk_z", "kps_low_z", "purity_low_z"),
        ),
    }

    # Weights are intentionally explicit so reviewers can see the biological
    # assumptions and so later ablations can test/learn them.
    edges = [
        EdgeDefinition(
            name="D_to_R",
            source="D",
            target="R",
            source_weights={
                "idh_wt_z": 0.40,
                "mgmt_unmethylated_z": 0.15,
                "atrx_wt_z": 0.10,
                "tert_promoter_mutant_z": 0.20,
                "chr7_gain_chr10_loss_z": 0.25,
                "mutation_count_z": 0.10,
                "tmb_z": 0.10,
                "aneuploidy_z": 0.15,
            },
            target_weights={
                "egfr_amp_z": 0.35,
                "tert_expr_z": 0.20,
                "tert_expressed_z": 0.15,
                "immune_score_z": 0.10,
                "stromal_score_z": 0.10,
                "rna_cluster_risk_z": 0.15,
                "methyl_cluster_risk_z": 0.15,
                "transcriptome_risk_z": 0.15,
            },
            rationale="Genomic lesions should constrain the observed regulatory/transcriptomic state.",
        ),
        EdgeDefinition(
            name="D_to_C",
            source="D",
            target="C",
            source_weights={
                "idh_wt_z": 0.45,
                "mgmt_unmethylated_z": 0.15,
                "tert_promoter_mutant_z": 0.20,
                "chr7_gain_chr10_loss_z": 0.25,
                "mutation_count_z": 0.10,
                "tmb_z": 0.10,
                "aneuploidy_z": 0.20,
            },
            target_weights={
                "grade_risk_z": 0.60,
                "kps_low_z": 0.30,
                "purity_low_z": 0.10,
            },
            rationale="Genomic risk should align with tumor/clinical phenotype severity, excluding survival outcome and patient age.",
        ),
        EdgeDefinition(
            name="R_to_C",
            source="R",
            target="C",
            source_weights={
                "egfr_amp_z": 0.35,
                "tert_expr_z": 0.20,
                "tert_expressed_z": 0.15,
                "immune_score_z": 0.15,
                "stromal_score_z": 0.15,
                "rna_cluster_risk_z": 0.20,
                "methyl_cluster_risk_z": 0.15,
                "transcriptome_risk_z": 0.15,
            },
            target_weights={
                "grade_risk_z": 0.60,
                "kps_low_z": 0.30,
                "purity_low_z": 0.10,
            },
            rationale="Regulatory state should align with tumor/clinical phenotype severity, excluding patient age.",
        ),
    ]
    return SheafSpec(nodes=nodes, edges=edges)


def _normalized_weight_vector(features: Sequence[str], weights: Mapping[str, float]) -> np.ndarray:
    vec = np.array([float(weights.get(f, 0.0)) for f in features], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Restriction map weight vector cannot be all zeros.")
    return vec / norm


def construct_feature_layout(spec: SheafSpec) -> Tuple[List[str], Dict[str, slice]]:
    feature_names: List[str] = []
    node_slices: Dict[str, slice] = {}
    start = 0
    for node_name, node in spec.nodes.items():
        feature_names.extend(node.features)
        stop = start + len(node.features)
        node_slices[node_name] = slice(start, stop)
        start = stop
    return feature_names, node_slices


def construct_coboundary(spec: SheafSpec) -> Tuple[np.ndarray, List[str], List[str], Dict[str, slice]]:
    """Construct scalar-edge-stalk coboundary matrix B_F."""
    feature_names, node_slices = construct_feature_layout(spec)
    feature_to_pos = {name: idx for idx, name in enumerate(feature_names)}
    B = np.zeros((len(spec.edges), len(feature_names)), dtype=float)

    for row, edge in enumerate(spec.edges):
        source_features = spec.nodes[edge.source].features
        target_features = spec.nodes[edge.target].features
        a = _normalized_weight_vector(source_features, edge.source_weights)
        b = _normalized_weight_vector(target_features, edge.target_weights)

        for local_idx, feat in enumerate(source_features):
            B[row, feature_to_pos[feat]] = a[local_idx]
        for local_idx, feat in enumerate(target_features):
            B[row, feature_to_pos[feat]] = -b[local_idx]

    return B, feature_names, [edge.name for edge in spec.edges], node_slices


def compute_sris(encoded: pd.DataFrame, spec: Optional[SheafSpec] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Compute edge residuals, edge energies, fractions, SRIS, and Laplacian metadata."""
    if spec is None:
        spec = build_fixed_phase1_sheaf()

    B, feature_names, edge_names, node_slices = construct_coboundary(spec)
    missing = [f for f in feature_names if f not in encoded.columns]
    if missing:
        raise KeyError(f"Encoded table is missing sheaf features: {missing}")

    X = encoded[feature_names].to_numpy(dtype=float)
    residuals = X @ B.T
    energies = residuals ** 2
    sris = energies.sum(axis=1)
    denom = np.where(sris > 0, sris, np.nan)
    fractions = energies / denom[:, None]
    fractions = np.nan_to_num(fractions, nan=0.0)

    out = pd.DataFrame({
        "patient_id": encoded.get("patient_id", pd.Series(range(len(encoded)))).values,
        "sample_id": encoded.get("sample_id", pd.Series([None] * len(encoded))).values,
        "SRIS": sris,
    })

    for i, edge in enumerate(edge_names):
        out[f"r_{edge}"] = residuals[:, i]
        out[f"E_{edge}"] = energies[:, i]
        out[f"frac_{edge}"] = fractions[:, i]

    # Attach main labels/covariates for immediate analysis.
    attach_cols = [
        "idh_mutant", "idh_wt", "idh_codel_subtype", "grade", "grade_risk",
        "mgmt_methylated", "mgmt_unmethylated", "egfr_amp", "os_months",
        "deceased", "age", "kps", "purity", "transcriptome_subtype",
        "methylation_cluster", "rna_cluster",
    ]
    for col in attach_cols:
        if col in encoded.columns:
            out[col] = encoded[col].values

    laplacian = B.T @ B
    metadata = {
        "edge_names": edge_names,
        "feature_names": feature_names,
        "node_slices": {k: [v.start, v.stop] for k, v in node_slices.items()},
        "coboundary_shape": list(B.shape),
        "laplacian_shape": list(laplacian.shape),
        "coboundary_matrix": B.tolist(),
        "laplacian_matrix": laplacian.tolist(),
        "nodes": {name: {"features": list(node.features), "description": node.description} for name, node in spec.nodes.items()},
        "edges": [
            {
                "name": e.name,
                "source": e.source,
                "target": e.target,
                "rationale": e.rationale,
                "source_weights": dict(e.source_weights),
                "target_weights": dict(e.target_weights),
            }
            for e in spec.edges
        ],
    }
    return out, metadata


def summarize_phase1(encoded: pd.DataFrame, results: pd.DataFrame, metadata: Dict[str, object]) -> Dict[str, object]:
    """Create a compact JSON-serializable Phase 1 sanity-check summary."""
    summary: Dict[str, object] = {
        "n_patients": int(len(results)),
        "n_edges": int(len(metadata["edge_names"])),
        "n_sheaf_features": int(len(metadata["feature_names"])),
        "sris_mean": float(results["SRIS"].mean()),
        "sris_median": float(results["SRIS"].median()),
        "sris_std": float(results["SRIS"].std()),
        "top_10_high_sris_patients": results.sort_values("SRIS", ascending=False)[
            ["patient_id", "SRIS"] + [f"E_{e}" for e in metadata["edge_names"]]
        ].head(10).to_dict(orient="records"),
    }

    if "idh_codel_subtype" in results:
        summary["sris_by_idh_codel_subtype"] = (
            results.groupby("idh_codel_subtype")["SRIS"].agg(["count", "mean", "median", "std"]).reset_index().to_dict(orient="records")
        )
    if "grade" in results:
        summary["sris_by_grade"] = (
            results.groupby("grade")["SRIS"].agg(["count", "mean", "median", "std"]).reset_index().to_dict(orient="records")
        )
    if "idh_mutant" in results:
        summary["sris_by_idh_mutant"] = (
            results.groupby("idh_mutant")["SRIS"].agg(["count", "mean", "median", "std"]).reset_index().to_dict(orient="records")
        )

    missing_cols = [c for c in encoded.columns if c.endswith("_missing")]
    summary["missingness_rates"] = {
        c.replace("_missing", ""): float(encoded[c].mean()) for c in missing_cols
    }

    if "age" in results:
        summary["age_handling"] = {
            "included_in_sheaf_laplacian": False,
            "role": "external covariate/result for downstream association and adjustment analyses",
            "mean_age": float(pd.to_numeric(results["age"], errors="coerce").mean()),
        }

    # Optional hypothesis-test sanity checks, if scipy is available.
    try:
        from scipy import stats  # type: ignore

        if "idh_mutant" in results and results["idh_mutant"].nunique() == 2:
            a = results.loc[results["idh_mutant"] == 0, "SRIS"]
            b = results.loc[results["idh_mutant"] == 1, "SRIS"]
            if len(a) > 1 and len(b) > 1:
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                summary["mann_whitney_sris_idh_mutant"] = {"statistic": float(u.statistic), "pvalue": float(u.pvalue)}
        if "grade" in results and results["grade"].nunique() > 2:
            groups = [g["SRIS"].to_numpy() for _, g in results.groupby("grade") if len(g) > 1]
            if len(groups) > 2:
                kw = stats.kruskal(*groups)
                summary["kruskal_sris_grade"] = {"statistic": float(kw.statistic), "pvalue": float(kw.pvalue)}
        if "age" in results:
            valid = results[["SRIS", "age"]].dropna()
            if len(valid) > 2:
                sp = stats.spearmanr(valid["SRIS"], valid["age"])
                summary["spearman_sris_age_external_analysis"] = {"statistic": float(sp.statistic), "pvalue": float(sp.pvalue)}
    except Exception as exc:  # pragma: no cover - scipy optional
        summary["hypothesis_tests_note"] = f"Skipped optional scipy tests: {exc}"

    return summary


def run_phase1(input_path: str | Path, output_dir: str | Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_table(input_path)
    encoded = encode_glioma_table(raw)
    spec = build_fixed_phase1_sheaf()
    results, metadata = compute_sris(encoded, spec=spec)
    summary = summarize_phase1(encoded, results, metadata)

    encoded_path = output_dir / "phase1_clean_encoded.csv"
    results_path = output_dir / "phase1_sris_results.csv"
    metadata_path = output_dir / "phase1_sheaf_metadata.json"
    summary_path = output_dir / "phase1_summary.json"

    encoded.to_csv(encoded_path, index=False)
    results.to_csv(results_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "encoded": encoded_path,
        "results": results_path,
        "metadata": metadata_path,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 sheaf residual engine on glioma table.")
    parser.add_argument("--input", default="data.txt", help="Input data.txt or CSV path.")
    parser.add_argument("--output-dir", default="phase1_outputs", help="Directory for Phase 1 outputs.")
    args = parser.parse_args()

    paths = run_phase1(args.input, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
