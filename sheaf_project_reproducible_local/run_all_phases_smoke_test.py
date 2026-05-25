#!/usr/bin/env python3
"""Fast local smoke test for Phases 1--7.

This verifies that the code/data pipeline works end-to-end using smaller
permutation counts and fewer cross-validation folds for the expensive geometry
phases. Use `run_all_phases_local.py` for the fuller internal run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COMBINED = ROOT / "combined_results_smoke"


def ensure(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy(src: Path, dst: Path) -> None:
    ensure(src); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


def run(args, cwd, py_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(py_path), env.get("PYTHONPATH", "")])
    print("\nRUN", " ".join([sys.executable] + args), "in", cwd, flush=True)
    subprocess.run([sys.executable] + args, cwd=str(cwd), env=env, check=True)


def phase1():
    phase = ROOT/'phase1_sheaf_residual_engine'; out = phase/'results'; clean(out)
    code = "from pathlib import Path; from phase1_sheaf_engine import run_phase1; print(run_phase1(Path('../data/data.txt'), Path('results')))"
    run(['-c', code], phase, phase/'src')
    return out


def phase2():
    phase = ROOT/'phase2_learned_sheaf_maps'; data = phase/'data'; out = phase/'results'; clean(data); clean(out)
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv', data/'phase1_clean_encoded.csv')
    run(['src/run_phase2.py'], phase, phase/'src')
    return out


def phase3():
    phase = ROOT/'phase3_survival_validation'; data=phase/'data'; out=phase/'results'; fig=phase/'figures'; clean(data); clean(out); clean(fig)
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_sris_results.csv', data/'phase1_sris_results.csv')
    copy(ROOT/'phase2_learned_sheaf_maps/results/phase2_sris_all_models.csv', data/'phase2_sris_all_models.csv')
    run(['src/run_phase3.py'], phase, phase/'src')
    return out


def phase4_fast():
    phase = ROOT/'phase4_subtype_sheaf_geometry'; data=phase/'data'; out=phase/'results'; fig=phase/'figures'; clean(data); clean(out); clean(fig)
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv', data/'phase1_clean_encoded.csv')
    sys.path.insert(0, str(phase/'src'))
    from phase4_subtype_sheaf_geometry import (
        PROTOCOLS, fit_group_sheaves, pairwise_laplacian_divergence,
        permutation_divergence_test, crossvalidated_counterfactual_assignment,
        group_edge_summary,
    )
    df = pd.read_csv(data/'phase1_clean_encoded.csv')
    for col in ['idh_codel_subtype','transcriptome_subtype','methylation_cluster','rna_cluster']:
        if col in df.columns: df[col] = df[col].astype(object)
    df['grade_label'] = df['grade'].apply(lambda x: f'G{int(x)}' if pd.notna(x) else np.nan)
    df['grade4_status'] = df['grade'].apply(lambda x: 'G4' if pd.notna(x) and int(x)==4 else ('G2G3' if pd.notna(x) else np.nan))
    analysis_specs = [
        ('idh_codel_subtype','strict_no_idh_no_grade_no_clusters'),
        ('grade_label','strict_no_idh_no_grade_no_clusters'),
        ('grade4_status','strict_no_grade_no_clusters'),
    ]
    all_divs=[]; perm_rows=[]; edge_rows=[]; metric_parts=[]; patient_parts=[]
    for label_col, protocol in analysis_specs:
        work = df.dropna(subset=[label_col]).copy()
        sheaves = fit_group_sheaves(work, label_col, protocol, ridge=1.0, min_n=10)
        div = pairwise_laplacian_divergence(sheaves); div['label_col']=label_col; div['protocol']=protocol; all_divs.append(div)
        edge_rows.append(group_edge_summary(work, label_col, protocol, ridge=1.0))
        ptest, _ = permutation_divergence_test(work, label_col, protocol, ridge=1.0, n_perm=5, seed=31, min_n=10)
        perm_rows.append(ptest)
        m, p = crossvalidated_counterfactual_assignment(df, label_col, protocol, ridge=1.0, n_splits=2, seed=19, min_n=10)
        metric_parts.append(m); patient_parts.append(p)
    pd.concat(all_divs, ignore_index=True).to_csv(out/'phase4_laplacian_divergences.csv', index=False)
    pd.DataFrame(perm_rows).to_csv(out/'phase4_permutation_divergence_tests.csv', index=False)
    pd.concat(edge_rows, ignore_index=True).to_csv(out/'phase4_group_edge_energy_summary.csv', index=False)
    metrics = pd.concat(metric_parts, ignore_index=True); patients = pd.concat(patient_parts, ignore_index=True)
    metrics.to_csv(out/'phase4_counterfactual_accuracy_metrics.csv', index=False)
    patients.to_csv(out/'phase4_counterfactual_patient_energies.csv', index=False)
    rows=[]
    for (task, protocol), sub in metrics.groupby(['task','protocol']):
        base=sub[sub.method=='baseline_logistic_features']
        if base.empty: continue
        base=base.iloc[0]
        for _, r in sub.iterrows():
            if r['method']=='baseline_logistic_features': continue
            rows.append({'task':task,'protocol':protocol,'method':r['method'],
                         'delta_accuracy':r['accuracy']-base['accuracy'],
                         'delta_balanced_accuracy':r['balanced_accuracy']-base['balanced_accuracy'],
                         'delta_macro_f1':r['macro_f1']-base['macro_f1']})
    pd.DataFrame(rows).to_csv(out/'phase4_accuracy_deltas.csv', index=False)
    (out/'phase4_summary.json').write_text(json.dumps({'phase':4,'mode':'smoke','analysis_specs':analysis_specs,'protocols':{k:v['description'] for k,v in PROTOCOLS.items()}}, indent=2))
    return out


def phase5_fast():
    phase = ROOT/'phase5_transport_sheaf_stability'; data=phase/'data'; out=phase/'results'; clean(data); clean(out); clean(phase/'figures')
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv', data/'phase1_clean_encoded.csv')
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_sris_results.csv', data/'phase1_sris_results.csv')
    copy(ROOT/'phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv', data/'phase4_counterfactual_patient_energies.csv')
    code = "from pathlib import Path; from phase5_transport_sheaf_stability import run_phase5; print(run_phase5(Path('data'), Path('.'), n_perm=5))"
    run(['-c', code], phase, phase/'src')
    return out


def phase6():
    phase = ROOT/'phase6_consensus_sheaf_discovery'; data=phase/'data'; out=phase/'results'; clean(data); clean(out); clean(phase/'figures')
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv', data/'phase1_clean_encoded.csv')
    copy(ROOT/'phase1_sheaf_residual_engine/results/phase1_sris_results.csv', data/'phase1_sris_results.csv')
    copy(ROOT/'phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv', data/'phase4_counterfactual_patient_energies.csv')
    copy(ROOT/'phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv', data/'phase5_patient_transport_features.csv')
    copy(ROOT/'phase5_transport_sheaf_stability/results/phase5_pairwise_transport_metrics.csv', data/'phase5_pairwise_transport_metrics.csv')
    run(['src/run_phase6.py'], phase, phase/'src')
    return out


def phase7():
    phase = ROOT/'phase7_publication_synthesis'; out=phase/'results'; clean(out)
    run(['src/run_phase7.py'], phase, phase/'src')
    (out/'phase7_local_run_note.txt').write_text('Phase 7 claim ledger script executed in smoke mode.\n')
    return out


def collect(outputs):
    clean(COMBINED)
    manifest=[]
    key_files = {
        1:['phase1_clean_encoded.csv','phase1_sris_results.csv','phase1_summary.json'],
        2:['phase2_sris_all_models.csv','phase2_model_summary.csv','phase2_summary.json'],
        3:['phase3_survival_model_summary.csv','phase3_summary.json'],
        4:['phase4_counterfactual_accuracy_metrics.csv','phase4_counterfactual_patient_energies.csv','phase4_summary.json'],
        5:['phase5_pairwise_transport_metrics.csv','phase5_patient_transport_features.csv','phase5_summary.json'],
        6:['phase6_consensus_feature_discovery.csv','phase6_prediction_metrics.csv','phase6_summary.json'],
        7:['phase7_local_run_note.txt'],
    }
    for i, out in enumerate(outputs, start=1):
        d = COMBINED/f'phase{i}'; clean(d); files=[]
        for name in key_files[i]:
            src=Path(out)/name
            if src.exists(): copy(src, d/name); files.append(str(d/name))
        manifest.append({'phase':i,'source_output_dir':str(out),'copied_files':files})
    (COMBINED/'run_manifest.json').write_text(json.dumps({'mode':'smoke','generated_at':datetime.now().isoformat(),'phases':manifest}, indent=2))


def main():
    ensure(DATA/'data.txt'); ensure(DATA/'twentyFourAndUp.xlsx')
    outputs=[phase1(), phase2(), phase3(), phase4_fast(), phase5_fast(), phase6(), phase7()]
    collect(outputs)
    print('\nSMOKE TEST COMPLETE. Verify with: python verify_phase_outputs.py')
    print('Outputs:', COMBINED)

if __name__ == '__main__':
    main()
