
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
