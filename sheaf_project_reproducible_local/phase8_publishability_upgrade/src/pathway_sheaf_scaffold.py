
"""Gene/pathway-level regulatory sheaf scaffold.

This module defines a schema for moving beyond patient-level summary sheaves.
Inputs expected:
  - omics tensor table with columns sample_id,gene,RNA,methylation,CNV,mutation,miRNA_activity
  - regulatory_edges.csv with columns source,target,edge_type,sign,pathway
  - optional pathway_membership.csv with pathway,gene

The main residual is
    r_{p,e} = rho_{u,e} x_{p,u} - rho_{v,e} x_{p,v}
where x_{p,g} is an omics vector for patient p and gene g.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class RegulatoryEdge:
    source: str
    target: str
    edge_type: str
    sign: float = 1.0
    pathway: str | None = None

OMIC_ORDER = ['RNA','methylation','CNV','mutation','miRNA_activity']

def edge_restriction_projection(edge_type: str, sign: float = 1.0):
    # First-pass interpretable maps into 1D expected regulatory activity.
    # These can later be replaced by learned constrained maps.
    w_src = np.zeros(len(OMIC_ORDER)); w_tgt = np.zeros(len(OMIC_ORDER))
    if edge_type in {'TF_activation','TF_target_activation'}:
        w_src[0] = sign; w_tgt[0] = 1.0
    elif edge_type in {'TF_repression','miRNA_target'}:
        w_src[0] = -abs(sign); w_tgt[0] = 1.0
    elif edge_type in {'methylation_expression','promoter_methylation'}:
        w_src[1] = -1.0; w_tgt[0] = 1.0
    elif edge_type in {'cnv_expression','copy_number_expression'}:
        w_src[2] = 1.0; w_tgt[0] = 1.0
    else:
        w_src[0] = sign; w_tgt[0] = 1.0
    return w_src, w_tgt

def compute_edge_residuals(omics_long: pd.DataFrame, edges: pd.DataFrame):
    # omics_long: sample_id,gene,RNA,methylation,CNV,mutation,miRNA_activity
    records=[]
    by_sample = {s: g.set_index('gene') for s,g in omics_long.groupby('sample_id')}
    for sample_id, mat in by_sample.items():
        for _, e in edges.iterrows():
            u, v = e['source'], e['target']
            if u not in mat.index or v not in mat.index: continue
            xu = mat.loc[u, OMIC_ORDER].astype(float).values
            xv = mat.loc[v, OMIC_ORDER].astype(float).values
            sign = float(e.get('sign',1.0)) if pd.notna(e.get('sign',1.0)) else 1.0
            w_u, w_v = edge_restriction_projection(str(e.get('edge_type','generic')), sign)
            r = float(w_u @ xu - w_v @ xv)
            records.append({'sample_id':sample_id,'source':u,'target':v,'edge_type':e.get('edge_type','generic'),
                            'pathway':e.get('pathway','NA'),'residual':r,'energy':r*r})
    return pd.DataFrame(records)

def pathway_sris(edge_residuals: pd.DataFrame):
    return edge_residuals.groupby(['sample_id','pathway'])['energy'].sum().reset_index(name='pathway_SRIS')
