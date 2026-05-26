"""
pathway_sheaf.py -- Gene/pathway-level sheaf (PLIS) for the regulatory-sheaf glioma project.

Resolves the data-availability constraint that previously blocked Gate 4: pulls gene-level
mutation + CNV from cBioPortal (see fetch_gene_data.py) and combines with the team's
clinical-summary table to populate per-patient node states for the Bio-Lead-approved
pathway graph.

Inputs
------
  phase1_clean_encoded.csv         team's TCGA clinical-summary table (420 patients)
  tcga_gene_status.csv             29 driver genes, mutation + GISTIC (this session)
  tcga_tert_expression.csv         TCGA TERT expression (already pulled earlier)
  pathway_node_list.csv            Bio-Lead-approved 110 nodes
  pathway_graph_edges.csv          Bio-Lead-approved 104 edges

Method
------
1. NODE STATE.  For each node in the pathway graph that maps to a measurable
   per-patient quantity, derive a binary perturbation state state_u(p) in {0,1}.
   - Gene oncogenes (EGFR, PDGFRA, MET, ...) -> mutated OR amplified
   - Tumor suppressors (TP53, PTEN, CDKN2A, RB1, ...) -> mutated OR deleted
   - IDH1/IDH2 -> mutated (kept as-is; sheaf inconsistency, not risk score)
   - Methylation/promoter events -> use team's binary fields where available
   - Clinical phenotypes (grade>=3, mesenchymal subtype, etc.) -> threshold

2. PATHWAY MEMBERSHIP.  For each pathway node P in the graph, M(P) = the set of
   nodes u with an edge u -> P.  These are the pathway's input drivers.

3. PATHWAY-LOCAL INCONSISTENCY (PLIS).  Defined as the within-pathway variance of
   member-node perturbation states (matches the team plan's
   PLIS_P(p) = sum_{e in E(P)} || r_e(p) ||^2 when r is the deviation from the
   pathway-mean state):

        z_P(p) = (1/|M(P)|) * sum_{u in M(P)} state_u(p)
        PLIS_P(p) = sum_{u in M(P)} ( state_u(p) - z_P(p) )^2

   PLIS is HIGH when pathway members disagree (some perturbed, some not) and LOW
   when the pathway is internally consistent (all perturbed or all quiescent).

Outputs (in results/)
---------------------
  pathway_sheaf_results.csv        patient x pathway PLIS matrix + summary cols
  top_residual_pathways.csv        ranked pathways by mean PLIS and subtype separation
  pathway_membership_map.csv       which nodes mapped to which data source (audit trail)
  pathway_sheaf_summary.json       coverage statistics + group-test outcomes
  pathway_plis_by_subtype.png      figure: PLIS distributions by IDH/codel subtype

Run: python pathway_sheaf.py --data-dir . --output-dir results
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Node -> per-patient state derivation
# --------------------------------------------------------------------------- #
ONCOGENES = {"EGFR", "PDGFRA", "MET", "FGFR1", "PIK3CA", "PIK3R1", "MDM2", "MDM4",
             "BRAF", "KRAS", "CDK4", "CDK6", "PPM1D", "NOTCH1"}
TUMOR_SUPPRESSORS = {"TP53", "ATRX", "DAXX", "CIC", "FUBP1", "CDKN2A", "CDKN2B",
                     "RB1", "PTEN", "NF1", "SETD2", "SUZ12"}


def derive_node_states(team_df: pd.DataFrame, gene_df: pd.DataFrame,
                       tert_df: Optional[pd.DataFrame],
                       node_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (patient x node) binary perturbation matrix and an audit table."""
    df = team_df.merge(gene_df, on="patient_id", how="left")
    if tert_df is not None and "patient_id" in tert_df.columns:
        df = df.merge(tert_df, on="patient_id", how="left")

    states = pd.DataFrame({"patient_id": df["patient_id"]})
    audit_rows: List[dict] = []

    def add(node: str, series: pd.Series, source: str, note: str = ""):
        s = pd.to_numeric(series, errors="coerce")
        s = (s > 0).astype("Int64").where(s.notna(), pd.NA)
        states[node] = s
        n_pos = int((s == 1).sum())
        n_obs = int(s.notna().sum())
        audit_rows.append({"node": node, "source": source, "n_observed": n_obs,
                           "n_perturbed": n_pos, "perturbation_rate": round(n_pos / max(n_obs, 1), 3),
                           "note": note})

    # Direct gene mutations / CNVs from cBioPortal pull -------------------------
    for gene in ONCOGENES | TUMOR_SUPPRESSORS | {"IDH1", "IDH2", "H3F3A"}:
        if f"{gene}_mutated" not in df.columns:
            continue
        mutated = df[f"{gene}_mutated"].astype(float).fillna(0)
        if gene in ONCOGENES:
            amp = df.get(f"{gene}_amplified", pd.Series(0, index=df.index)).astype(float).fillna(0)
            state = ((mutated > 0) | (amp > 0)).astype(int)
            note = "mutated OR amplified"
        elif gene in TUMOR_SUPPRESSORS:
            deleted = df.get(f"{gene}_deleted", pd.Series(0, index=df.index)).astype(float).fillna(0)
            state = ((mutated > 0) | (deleted > 0)).astype(int)
            note = "mutated OR homozygously deleted"
        else:
            state = (mutated > 0).astype(int)
            note = "mutated"
        if gene in node_names:
            add(gene, state, "cBioPortal lgggbm_tcga_pub", note)

    # Special: H3F3A K27M / G34 variants (mutation calls don't distinguish variant)
    if "H3F3A_K27M" in node_names and "H3F3A_mutated" in df.columns:
        add("H3F3A_K27M", df["H3F3A_mutated"], "cBioPortal", "H3F3A any mutation (variant not resolved)")
    if "H3F3A_G34" in node_names and "H3F3A_mutated" in df.columns:
        add("H3F3A_G34", df["H3F3A_mutated"], "cBioPortal", "H3F3A any mutation (variant not resolved)")

    # CNV alteration nodes -----------------------------------------------------
    if "EGFR_amplification" in node_names and "EGFR_amplified" in df.columns:
        add("EGFR_amplification", df["EGFR_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "PDGFRA_amplification" in node_names and "PDGFRA_amplified" in df.columns:
        add("PDGFRA_amplification", df["PDGFRA_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "MET_amplification" in node_names and "MET_amplified" in df.columns:
        add("MET_amplification", df["MET_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "CDK4_amplification" in node_names and "CDK4_amplified" in df.columns:
        add("CDK4_amplification", df["CDK4_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "CDK6_amplification" in node_names and "CDK6_amplified" in df.columns:
        add("CDK6_amplification", df["CDK6_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "MDM2_amplification" in node_names and "MDM2_amplified" in df.columns:
        add("MDM2_amplification", df["MDM2_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "MDM4_amplification" in node_names and "MDM4_amplified" in df.columns:
        add("MDM4_amplification", df["MDM4_amplified"].fillna(0), "cBioPortal GISTIC", "GISTIC == 2")
    if "Chr9p21_loss" in node_names and "CDKN2A_deleted" in df.columns:
        add("Chr9p21_loss", df["CDKN2A_deleted"].fillna(0), "cBioPortal GISTIC", "proxy: CDKN2A homozygous deletion")
    if "Chr13q_loss" in node_names and "RB1_deleted" in df.columns:
        add("Chr13q_loss", df["RB1_deleted"].fillna(0), "cBioPortal GISTIC", "proxy: RB1 homozygous deletion")
    if "Chr17p_loss" in node_names and "TP53_deleted" in df.columns:
        add("Chr17p_loss", df["TP53_deleted"].fillna(0), "cBioPortal GISTIC", "proxy: TP53 homozygous deletion")

    # From the team's clinical-summary table -----------------------------------
    if "Chr7_gain" in node_names and "chr7_gain_chr10_loss" in df.columns:
        add("Chr7_gain", df["chr7_gain_chr10_loss"], "team clinical", "co-occurring Chr7+/Chr10-")
    if "Chr10_loss" in node_names and "chr7_gain_chr10_loss" in df.columns:
        add("Chr10_loss", df["chr7_gain_chr10_loss"], "team clinical", "co-occurring Chr7+/Chr10-")
    if "1p_loss" in node_names and "idh_codel_subtype" in df.columns:
        codel = (df["idh_codel_subtype"] == "IDHmut-codel").astype(int)
        add("1p_loss", codel, "derived from idh_codel_subtype", "IDHmut-codel = 1p/19q codeletion")
    if "19q_loss" in node_names and "idh_codel_subtype" in df.columns:
        codel = (df["idh_codel_subtype"] == "IDHmut-codel").astype(int)
        add("19q_loss", codel, "derived from idh_codel_subtype", "IDHmut-codel = 1p/19q codeletion")
    if "MGMT_promoter" in node_names and "mgmt_methylated" in df.columns:
        add("MGMT_promoter", df["mgmt_methylated"], "team clinical", "1 = methylated (silenced)")
    if "TERT_promoter" in node_names and "tert_promoter_mutant" in df.columns:
        add("TERT_promoter", df["tert_promoter_mutant"], "team clinical", "promoter mutation status")
    if "EGFRvIII_variant" in node_names and "EGFR_amplified" in df.columns:
        # EGFRvIII not directly in the panel; use amplification as a weak proxy
        add("EGFRvIII_variant", df["EGFR_amplified"].fillna(0), "cBioPortal (proxy)",
            "weak proxy: EGFR amplification (vIII status not separately panel-typed here)")

    # Pathway-level abstractions derived from clinical clusters ----------------
    if "G_CIMP_hypermethylation" in node_names and "idh_mutant" in df.columns:
        add("G_CIMP_hypermethylation", df["idh_mutant"], "team clinical (proxy)",
            "G-CIMP-high tracks closely with IDH mutation")
    if "IDH_2HG_metabolism" in node_names and "idh_mutant" in df.columns:
        add("IDH_2HG_metabolism", df["idh_mutant"], "team clinical", "aberrant activity = IDH mutation")
    if "ALT_telomere_maintenance" in node_names:
        atrx = df.get("atrx_mutant", df.get("ATRX_mutated", pd.Series(0, index=df.index)))
        add("ALT_telomere_maintenance", atrx.fillna(0), "team clinical + cBioPortal",
            "ALT driven by ATRX loss-of-function")
    if "Telomere_maintenance_pathway" in node_names and "tert_promoter_mutant" in df.columns:
        # active when EITHER TERT promoter mutated OR ATRX lost
        atrx = df.get("atrx_mutant", df.get("ATRX_mutated", pd.Series(0, index=df.index))).fillna(0)
        active = ((df["tert_promoter_mutant"] > 0) | (atrx > 0)).astype(int)
        add("Telomere_maintenance_pathway", active, "team + cBioPortal",
            "active = TERT promoter mut OR ATRX loss")
    if "Mesenchymal_transcription_program" in node_names and "transcriptome_subtype" in df.columns:
        active = (df["transcriptome_subtype"].astype(str).str.upper() == "ME").astype(int)
        add("Mesenchymal_transcription_program", active, "team clinical",
            "Mesenchymal transcriptome subtype")
    if "Proneural_transcription_program" in node_names and "transcriptome_subtype" in df.columns:
        active = (df["transcriptome_subtype"].astype(str).str.upper() == "PN").astype(int)
        add("Proneural_transcription_program", active, "team clinical", "Proneural transcriptome")
    if "Transcriptome_subtype" in node_names and "transcriptome_subtype" in df.columns:
        aggressive = df["transcriptome_subtype"].astype(str).str.upper().isin(["ME", "CL"]).astype(int)
        add("Transcriptome_subtype", aggressive, "team clinical", "aggressive subtype (ME or CL)")
    if "Pan_Glioma_DNA_methylation_cluster" in node_names and "methylation_cluster" in df.columns:
        add("Pan_Glioma_DNA_methylation_cluster",
            df["methylation_cluster"].notna().astype(int), "team clinical", "cluster assigned")
    if "Pan_Glioma_RNA_expression_cluster" in node_names and "rna_cluster" in df.columns:
        add("Pan_Glioma_RNA_expression_cluster",
            df["rna_cluster"].notna().astype(int), "team clinical", "cluster assigned")
    if "ESTIMATE_immune_score" in node_names and "immune_score" in df.columns:
        s = pd.to_numeric(df["immune_score"], errors="coerce")
        add("ESTIMATE_immune_score", (s > s.median()).astype(int),
            "team clinical", "above-median = perturbed")
    if "ESTIMATE_stromal_score" in node_names and "stromal_score" in df.columns:
        s = pd.to_numeric(df["stromal_score"], errors="coerce")
        add("ESTIMATE_stromal_score", (s > s.median()).astype(int),
            "team clinical", "above-median = perturbed")
    if "Neoplasm_histologic_grade" in node_names and "grade" in df.columns:
        add("Neoplasm_histologic_grade", (df["grade"].astype(float) >= 3).astype(int),
            "team clinical", "high grade (G3/G4) = perturbed")
    if "Genomic_instability_phenotype" in node_names and "aneuploidy" in df.columns:
        s = pd.to_numeric(df["aneuploidy"], errors="coerce")
        add("Genomic_instability_phenotype", (s > s.median()).astype(int),
            "team clinical", "above-median aneuploidy")
    if "Proliferation_phenotype" in node_names and "grade" in df.columns:
        add("Proliferation_phenotype", (df["grade"].astype(float) >= 3).astype(int),
            "team clinical (proxy: grade)", "")
    if "Cell_cycle_phenotype" in node_names and "grade" in df.columns:
        add("Cell_cycle_phenotype", (df["grade"].astype(float) >= 3).astype(int),
            "team clinical (proxy: grade)", "")

    audit = pd.DataFrame(audit_rows).sort_values("node").reset_index(drop=True)
    return states, audit


# --------------------------------------------------------------------------- #
# Pathway-local inconsistency
# --------------------------------------------------------------------------- #
def compute_plis(states: pd.DataFrame, edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute PLIS per pathway per patient. Returns (patient x pathway, membership table)."""
    pathway_targets = edges["target"][edges["target_type"] == "pathway"].unique()
    # Also include phenotype targets that act as pathway-level summaries
    phenotype_targets = edges["target"][edges["target_type"] == "phenotype"].unique()
    targets = list(pathway_targets) + [p for p in phenotype_targets if p not in pathway_targets]

    membership_rows: List[dict] = []
    plis = pd.DataFrame({"patient_id": states["patient_id"]})

    for P in targets:
        members = edges[edges["target"] == P]["source"].unique().tolist()
        observed_members = [m for m in members if m in states.columns]
        if len(observed_members) < 2:
            membership_rows.append({"pathway": P, "n_total_members": len(members),
                                    "n_observed_members": len(observed_members),
                                    "members_observed": ";".join(observed_members),
                                    "members_unobserved": ";".join(m for m in members if m not in states.columns)})
            continue
        M = states[observed_members].apply(pd.to_numeric, errors="coerce")
        # Per-patient: mean across observed members (skip NaN), then sum of squared deviations
        Mmean = M.mean(axis=1, skipna=True)
        sq = (M.sub(Mmean, axis=0) ** 2).sum(axis=1, skipna=True)
        plis[P] = sq
        membership_rows.append({"pathway": P, "n_total_members": len(members),
                                "n_observed_members": len(observed_members),
                                "members_observed": ";".join(observed_members),
                                "members_unobserved": ";".join(m for m in members if m not in states.columns)})
    membership = pd.DataFrame(membership_rows)
    return plis, membership


# --------------------------------------------------------------------------- #
# Group testing + top-pathway ranking
# --------------------------------------------------------------------------- #
def group_tests(plis: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis test of PLIS by IDH/codel subtype and by grade, per pathway."""
    from scipy.stats import kruskal
    merged = plis.merge(team_df[["patient_id", "idh_codel_subtype", "grade"]], on="patient_id", how="left")
    rows = []
    pathways = [c for c in plis.columns if c != "patient_id"]
    for P in pathways:
        vals = pd.to_numeric(merged[P], errors="coerce")
        # subtype
        subtype_p = np.nan
        try:
            grps = [vals[merged["idh_codel_subtype"] == s].dropna().to_numpy()
                    for s in merged["idh_codel_subtype"].dropna().unique()]
            grps = [g for g in grps if len(g) > 2]
            if len(grps) >= 2:
                subtype_p = float(kruskal(*grps).pvalue)
        except Exception:
            pass
        # grade
        grade_p = np.nan
        try:
            grps = [vals[merged["grade"].astype(float) == g].dropna().to_numpy()
                    for g in sorted(merged["grade"].dropna().unique())]
            grps = [g for g in grps if len(g) > 2]
            if len(grps) >= 2:
                grade_p = float(kruskal(*grps).pvalue)
        except Exception:
            pass
        rows.append({"pathway": P, "mean_plis": float(vals.mean()),
                     "std_plis": float(vals.std()),
                     "kw_subtype_p": subtype_p, "kw_grade_p": grade_p,
                     "n_observed": int(vals.notna().sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figure(plis: pd.DataFrame, team_df: pd.DataFrame, top_pathways: List[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    merged = plis.merge(team_df[["patient_id", "idh_codel_subtype"]], on="patient_id", how="left")
    subs = ["IDHmut-codel", "IDHmut-non-codel", "IDHwt"]
    n = min(len(top_pathways), 6)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, P in zip(axes, top_pathways[:n]):
        data = [pd.to_numeric(merged[P][merged["idh_codel_subtype"] == s],
                              errors="coerce").dropna().to_numpy() for s in subs]
        ax.boxplot(data, tick_labels=[s.replace("IDHmut-", "Imut-") for s in subs])
        ax.set_title(P.replace("_", " ")[:32], fontsize=9)
        ax.set_ylabel("PLIS")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle("Top pathway-local inconsistency (PLIS) by IDH/codel subtype")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team-clean", default="phase1_clean_encoded.csv",
                   help="Path to the team's phase1_clean_encoded.csv")
    p.add_argument("--gene-status", default="tcga_gene_status.csv",
                   help="Path to tcga_gene_status.csv (produced by fetch_gene_data.py)")
    p.add_argument("--tert", default=None, help="Optional TCGA TERT expression CSV")
    p.add_argument("--graph-nodes", default="pathway_node_list.csv")
    p.add_argument("--graph-edges", default="pathway_graph_edges.csv")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    team_df = pd.read_csv(args.team_clean)
    gene_df = pd.read_csv(args.gene_status)
    tert_df = pd.read_csv(args.tert) if args.tert else None
    nodes = pd.read_csv(args.graph_nodes)
    edges = pd.read_csv(args.graph_edges)

    node_names = nodes["node_name"].tolist()
    # Exclude Overall_survival from any computation (leakage guard)
    edges = edges[edges["target"] != "Overall_survival"].copy()

    states, audit = derive_node_states(team_df, gene_df, tert_df, node_names)
    audit.to_csv(out / "pathway_membership_map.csv", index=False)

    plis, membership = compute_plis(states, edges)
    plis.to_csv(out / "pathway_sheaf_results.csv", index=False)
    membership.to_csv(out / "pathway_membership_detail.csv", index=False)

    tests = group_tests(plis, team_df)
    # rank pathways: prioritise ones whose PLIS strongly separates subtypes
    tests["log10_subtype_p"] = -np.log10(tests["kw_subtype_p"].clip(lower=1e-300))
    tests["log10_grade_p"] = -np.log10(tests["kw_grade_p"].clip(lower=1e-300))
    tests["combined_signal"] = tests["log10_subtype_p"].fillna(0) + tests["log10_grade_p"].fillna(0)
    tests = tests.sort_values(["combined_signal", "mean_plis"], ascending=[False, False])
    tests.to_csv(out / "top_residual_pathways.csv", index=False)

    top = tests["pathway"].head(6).tolist()
    make_figure(plis, team_df, top, out / "pathway_plis_by_subtype.png")

    summary = {
        "n_patients": int(len(plis)),
        "n_pathways_computed": int(len(tests)),
        "n_pathways_with_>=2_observed_members": int(membership["n_observed_members"].ge(2).sum()),
        "top_5_by_subtype_separation": tests[["pathway", "mean_plis", "kw_subtype_p", "kw_grade_p", "n_observed"]].head(5).to_dict("records"),
        "node_coverage": int(audit["n_observed"].gt(0).sum()),
        "audit_rows": int(len(audit)),
        "outputs": {
            "patient_pathway_plis": "pathway_sheaf_results.csv",
            "ranked_pathways": "top_residual_pathways.csv",
            "node_data_audit": "pathway_membership_map.csv",
            "pathway_membership_detail": "pathway_membership_detail.csv",
            "figure": "pathway_plis_by_subtype.png",
        },
    }
    (out / "pathway_sheaf_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== PATHWAY SHEAF RESULTS ({len(plis)} patients, {len(tests)} pathways) ===\n")
    print(tests[["pathway", "mean_plis", "kw_subtype_p", "kw_grade_p", "n_observed"]].head(10).to_string(index=False))
    print(f"\nnodes with measurable state: {summary['node_coverage']}/{len(audit)}")
    print(f"pathways with >=2 observed members: {summary['n_pathways_with_>=2_observed_members']}")
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
