"""Strict accuracy benchmark for Phase 2 learned sheaves.

This companion benchmark avoids potentially label-derived cluster fields when
estimating incremental predictive value.  It should be reported alongside the
broad benchmark because the broad benchmark is medically informative but can be
optimistic when supervised methylation/RNA clusters encode the target.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import argparse, json
import numpy as np
import pandas as pd

from phase2_accuracy_benchmark import (
    add_endpoint_columns,
    EDGE_FEATURES,
    FRACTION_FEATURES,
    SHEAF_FEATURES,
    available_features,
    cv_predictions,
    compute_metrics,
)

STRICT_NO_IDH = (
    "age", "grade", "kps", "purity", "mgmt_methylated", "egfr_amp", "atrx_wt",
    "tert_promoter_mutant", "chr7_gain_chr10_loss", "mutation_count", "tmb", "aneuploidy",
    "tert_expr", "tert_expressed", "immune_score", "stromal_score",
)
STRICT_NO_GRADE = (
    "age", "kps", "purity", "idh_mutant", "mgmt_methylated", "egfr_amp", "atrx_wt",
    "tert_promoter_mutant", "chr7_gain_chr10_loss", "mutation_count", "tmb", "aneuploidy",
    "tert_expr", "tert_expressed", "immune_score", "stromal_score",
)
STRICT_SURVIVAL = (
    "age", "grade", "kps", "purity", "idh_mutant", "mgmt_methylated", "egfr_amp", "atrx_wt",
    "tert_promoter_mutant", "chr7_gain_chr10_loss", "mutation_count", "tmb", "aneuploidy",
    "tert_expr", "immune_score", "stromal_score",
)


def make_24m_endpoint(df: pd.DataFrame, horizon: float = 24.0) -> pd.DataFrame:
    out = df.copy()
    months = pd.to_numeric(out["os_months"], errors="coerce")
    deceased = pd.to_numeric(out["deceased"], errors="coerce")
    y = pd.Series(np.nan, index=out.index, dtype=float)
    y[(deceased == 1.0) & (months <= horizon)] = 1.0
    y[months > horizon] = 0.0
    out["death_24m"] = y
    return out


def merge_results(encoded_path: Path, core_path: Path, diagnostic_path: Path):
    encoded = add_endpoint_columns(pd.read_csv(encoded_path))
    core = pd.read_csv(core_path)
    diag = pd.read_csv(diagnostic_path)
    add_cols = [c for c in encoded.columns if c not in core.columns or c.endswith("_endpoint") or c == "death_24m"]
    core = core.merge(encoded[["patient_id"] + [c for c in add_cols if c != "patient_id"]], on="patient_id", how="left")
    diag = diag.merge(encoded[["patient_id"] + [c for c in add_cols if c != "patient_id"]], on="patient_id", how="left")
    return encoded, core, diag


def design(df: pd.DataFrame, feature_mode: str, baseline_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    if feature_mode == "SRIS only":
        cols = ["SRIS"]
    elif feature_mode == "edge energies only":
        cols = available_features(df, EDGE_FEATURES)
    elif feature_mode == "sheaf full":
        cols = available_features(df, SHEAF_FEATURES)
    elif feature_mode == "strict baseline":
        cols = available_features(df, baseline_cols)
    elif feature_mode == "strict baseline + SRIS":
        cols = available_features(df, list(baseline_cols)+["SRIS"])
    elif feature_mode == "strict baseline + edge energies":
        cols = available_features(df, list(baseline_cols)+EDGE_FEATURES)
    elif feature_mode == "strict baseline + sheaf full":
        cols = available_features(df, list(baseline_cols)+SHEAF_FEATURES)
    else:
        raise ValueError(feature_mode)
    return df[cols].to_numpy(dtype=float), cols


def eval_row(df: pd.DataFrame, target_col: str, model_name: str, predictor_set: str, baseline_cols: Sequence[str]):
    tmp = df.dropna(subset=[target_col]).copy()
    y = tmp[target_col].astype(int).to_numpy()
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 3:
        return None
    X, cols = design(tmp, predictor_set, baseline_cols)
    proba, label = cv_predictions(X, y)
    return {"model": model_name, "predictor_set": predictor_set, "n": len(y), "features": ";".join(cols), **compute_metrics(y, proba, label, len(cols))}


def run_strict_benchmark(encoded_path, core_path, diagnostic_path, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    encoded, core, diag = merge_results(Path(encoded_path), Path(core_path), Path(diagnostic_path))
    endpoints = [
        ("IDH mutant status", "idh_mutant_endpoint", diag[diag["variant"]=="no_idh"], STRICT_NO_IDH, "no label-derived cluster fields; no IDH feature"),
        ("Grade 4 status", "grade4_endpoint", diag[diag["variant"]=="no_grade"], STRICT_NO_GRADE, "no label-derived cluster fields; no grade feature"),
        ("24-month death", "death_24m", core, STRICT_SURVIVAL, "no label-derived cluster fields; survival labels excluded from SRIS"),
    ]
    predictor_sets = ["SRIS only", "edge energies only", "sheaf full", "strict baseline", "strict baseline + SRIS", "strict baseline + edge energies", "strict baseline + sheaf full"]
    rows=[]
    for ep, target, data, baseline, note in endpoints:
        groupby = ["variant", "model"] if "variant" in data.columns and ep != "24-month death" else ["model"]
        for key, g in data.groupby(groupby):
            model = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)
            for ps in predictor_sets:
                try:
                    r=eval_row(g, target, model, ps, baseline)
                    if r:
                        r={"endpoint": ep, "protocol": "strict_no_cluster_baseline", "note": note, **r}
                        rows.append(r)
                except Exception as e:
                    rows.append({"endpoint": ep, "model": model, "predictor_set": ps, "status": "failed", "note": str(e)})
    metrics=pd.DataFrame(rows)
    metrics_path=output_dir/'phase2_strict_accuracy_metrics.csv'
    metrics.to_csv(metrics_path,index=False)
    # Deltas versus strict baseline
    status_col = metrics['status'] if 'status' in metrics.columns else pd.Series(['ok'] * len(metrics), index=metrics.index)
    ok=metrics[status_col.fillna('ok')!='failed'].copy()
    deltas=[]
    for (ep, model), g in ok.groupby(['endpoint','model']):
        base=g[g['predictor_set']=='strict baseline']
        if base.empty: continue
        b=base.iloc[0]
        for _, r in g.iterrows():
            rec={'endpoint':ep,'model':model,'predictor_set':r['predictor_set']}
            for m in ['accuracy','balanced_accuracy','f1','auroc','auprc']:
                rec[f'delta_{m}_vs_strict_baseline']=float(r[m]-b[m])
            deltas.append(rec)
    deltas=pd.DataFrame(deltas)
    deltas_path=output_dir/'phase2_strict_accuracy_deltas.csv'
    deltas.to_csv(deltas_path,index=False)
    # Best non-baseline increment per endpoint by AUROC
    inc=[]
    for ep,g in ok.groupby('endpoint'):
        candidates=g[g['predictor_set'].str.contains('strict baseline \\+', regex=True)]
        if candidates.empty: continue
        idx=candidates['auroc'].astype(float).idxmax(); r=candidates.loc[idx]
        base=g[(g['model']==r['model']) & (g['predictor_set']=='strict baseline')]
        delta_auc=float(r['auroc']-base.iloc[0]['auroc']) if not base.empty else np.nan
        delta_bal=float(r['balanced_accuracy']-base.iloc[0]['balanced_accuracy']) if not base.empty else np.nan
        inc.append({'endpoint':ep,'model':r['model'],'predictor_set':r['predictor_set'],'n':int(r['n']),'accuracy':float(r['accuracy']),'balanced_accuracy':float(r['balanced_accuracy']),'f1':float(r['f1']),'auroc':float(r['auroc']),'auprc':float(r['auprc']),'delta_auroc_vs_strict_baseline':delta_auc,'delta_balanced_accuracy_vs_strict_baseline':delta_bal})
    inc=pd.DataFrame(inc)
    inc_path=output_dir/'phase2_strict_accuracy_increment_summary.csv'
    inc.to_csv(inc_path,index=False)
    summ={'protocol':'strict_no_cluster_baseline','metrics':str(metrics_path),'deltas':str(deltas_path),'increment_summary':str(inc_path),'increment_rows':inc.to_dict(orient='records')}
    summary_path=output_dir/'phase2_strict_accuracy_summary.json'
    summary_path.write_text(json.dumps(summ,indent=2),encoding='utf-8')
    return {'metrics':str(metrics_path),'deltas':str(deltas_path),'increment_summary':str(inc_path),'summary':str(summary_path)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--encoded',required=True)
    ap.add_argument('--core-results',required=True)
    ap.add_argument('--diagnostic-results',required=True)
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    for k,v in run_strict_benchmark(args.encoded,args.core_results,args.diagnostic_results,args.output_dir).items(): print(k+': '+v)
if __name__=='__main__': main()
