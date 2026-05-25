import os, json, shutil, zipfile, subprocess, math
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
base=Path('/mnt/data')
pack=base/'phase6_consensus_sheaf_discovery_package'
(pack/'paper').mkdir(exist_ok=True)
summary=json.loads((pack/'results/phase6_summary.json').read_text())
metrics=pd.read_csv(pack/'results/phase6_metric_deltas.csv')
disc=pd.read_csv(pack/'results/phase6_consensus_feature_discovery.csv')
best=pd.read_csv(pack/'results/phase6_best_metric_deltas.csv')
top=pd.read_csv(pack/'results/phase6_top_features_by_task.csv')

def esc(s):
    return str(s).replace('_','\\_').replace('%','\\%').replace('&','\\&')

best_show=best.head(8).copy()
lines=[r'\begin{tabular}{lllr}', r'\toprule', r'Task & Metric & Model delta & Value \\', r'\midrule']
for _,r in best_show.iterrows():
    dcol=r.get('delta_column','')
    val=r[dcol] if dcol in r else 0.0
    lines.append(f"{esc(r['task'])} & {esc(r['metric'])} & {esc(dcol.replace('delta_','').replace('_vs_baseline',''))} & {float(val):.4f} \\")
lines += [r'\bottomrule', r'\end{tabular}']
(pack/'paper/phase6_best_delta_table.tex').write_text('\n'.join(lines))

top_show=top.sort_values('CSDS', ascending=False).head(12).copy()
lines=[r'\begin{tabular}{llrrr}', r'\toprule', r'Task & Feature & CSDS & Sel. & q \\', r'\midrule']
for _,r in top_show.iterrows():
    lines.append(f"{esc(r['task'])} & {esc(r['feature'])} & {float(r['CSDS']):.3f} & {float(r['selection_frequency']):.2f} & {float(r['q_value']):.3f} \\")
lines += [r'\bottomrule', r'\end{tabular}']
(pack/'paper/phase6_top_feature_table.tex').write_text('\n'.join(lines))

report=f'''# Phase 6 Technical Report: Consensus Sheaf Discovery and Reliability\n\nPhase 6 converts Phase 1-5 sheaf outputs into transport-calibrated discovery signatures. It computes stability-selected sheaf features, marginal association statistics, empirical permutation p-values, FDR q-values, transport stability weights, and the Consensus Sheaf Discovery Score (CSDS).\n\nRows in discovery table: {len(disc)}\nPatient-level SDI rows: {summary['n_patient_sdi_rows']}\n\nTop features by CSDS are in `results/phase6_top_features_by_task.csv`. The patient-level Sheaf Discovery Index is in `results/phase6_patient_sheaf_discovery_index.csv`.\n\nInterpretation: internal discovery is promising, but external validation remains required before biomarker claims.\n'''
(pack/'phase6_technical_report.md').write_text(report)

tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{float}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\title{Phase 6 Technical Specification\\Consensus Sheaf Discovery and Reliability}
\author{Sheaf-Theoretic Multi-Omics Brain Tumor Project}
\date{May 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Purpose}
Phases 1--5 constructed patient-level sheaf inconsistency energies, learned restriction maps, subtype-specific sheaf geometries, and optimal-transport stability. Phase 6 adds the discovery layer. Its objective is to identify sheaf residual features that are not only predictive, but also stable, statistically associated with biological labels, robust to permutation, and interpretable as regulatory inconsistency signatures.

The central Phase 6 object is the \emph{Consensus Sheaf Discovery Score} (CSDS), which ranks candidate sheaf features by combining sparse-model stability, marginal biological effect, false-discovery control, and transport stability.

\section{Candidate Feature Family}
For each patient $p$, let
\[
\mathbf{s}(p)=\left[\operatorname{SRIS}(p), E_{D\to R}(p), E_{D\to C}(p), E_{R\to C}(p), \ldots \right]
\]
collect Phase 1 residuals, Phase 4 counterfactual sheaf energies, and Phase 5 transport-to-reference distances. A candidate sheaf feature is one coordinate
\[
\phi_j(p)=s_j(p).
\]
These features are evaluated under strict leakage-control protocols, including no-IDH/no-grade/no-cluster variants when the endpoint would otherwise be partially encoded by the inputs.

\section{Sparse Stability Selection}
For task $t$ with labels $y_p$, repeated stratified folds are constructed. On each training fold, a sparse logistic model is fit on sheaf candidates:
\[
\widehat{\beta}^{(b)}
=\arg\min_{\beta}
\left\{\mathcal{L}_t(\beta;X^{(b)},y^{(b)})+\lambda\|\beta\|_1\right\}.
\]
The selection frequency of feature $j$ is
\[
\pi_j=\frac{1}{B}\sum_{b=1}^B \mathbf{1}\{\widehat{\beta}^{(b)}_j\ne 0\}.
\]
This prevents the paper from relying only on one fitted model or one lucky train-test split.

\section{Association and Permutation Testing}
For each feature $\phi_j$, Phase 6 computes a task-specific association statistic. For binary tasks, it uses a rank-based two-group statistic; for multiclass tasks, it uses a Kruskal--Wallis statistic. Let the observed statistic be $T_j$. Under label permutations $\sigma_1,\ldots,\sigma_M$, the empirical p-value is
\[
\widehat{p}_j
=\frac{1+\sum_{m=1}^{M}\mathbf{1}\{T_j(\sigma_m y)\ge T_j(y)\}}{M+1}.
\]
The empirical p-values are adjusted by Benjamini--Hochberg FDR to obtain $q_j$.

\section{Transport-Calibrated Stability Weight}
Phase 5 estimated edge-wise transport stability between biological groups. Phase 6 maps each feature to an edge family and assigns a transport stability weight
\[
\tau_j\in[0,1].
\]
For example, a feature involving $D\to R$ receives the average Phase 5 $D\to R$ stability, while total-energy features receive the SRIS-level stability. This gives priority to features that remain coherent under cross-group transport.

\section{Consensus Sheaf Discovery Score}
Let $a_j$ be the normalized marginal effect size of feature $j$, $\pi_j$ its stability-selection frequency, $q_j$ its FDR value, and $\tau_j$ its transport stability. Phase 6 defines
\[
\boxed{
\operatorname{CSDS}_j
=\pi_j\,a_j\,(1-q_j)\,\tau_j.
}
\]
A high-CSDS feature is repeatedly selected, associated with the endpoint, FDR-supported, and transport-stable.

\section{Patient-Level Sheaf Discovery Index}
For each task, the top $K$ CSDS-ranked features are combined into a patient-level Sheaf Discovery Index:
\[
\operatorname{SDI}(p)=
\sum_{j\in\mathcal{T}_K} w_j\,z(\phi_j(p)),
\]
where $z(\cdot)$ denotes cohort standardization and
\[
w_j\propto \operatorname{sign}(\bar{\beta}_j)\operatorname{CSDS}_j.
\]
This creates a compact one-dimensional summary of the most reliable sheaf-discovery signal for a task.

\section{Internal Results}
Phase 6 produced a discovery table, patient-level SDI table, prediction metrics, cross-task consensus table, and figures. The most important tables are summarized below.

\subsection{Best Accuracy Deltas}
\begin{center}
\input{phase6_best_delta_table.tex}
\end{center}

\subsection{Top Consensus Sheaf Discoveries}
\begin{center}
\resizebox{\textwidth}{!}{\input{phase6_top_feature_table.tex}}
\end{center}

\section{Figures}
\begin{figure}[H]
\centering
\includegraphics[width=.9\textwidth]{../figures/phase6_cross_task_CSDS_heatmap.png}
\caption{Cross-task consensus sheaf discovery heatmap.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.9\textwidth]{../figures/phase6_top_CSDS_grade_label.png}
\caption{Top Phase 6 CSDS features for strict grade classification.}
\end{figure}

\section{Interpretation}
Phase 6 creates a new discovery layer: the model now produces uncertainty-aware, transport-calibrated sheaf residual biomarkers rather than only classification scores. This is technically distinct from ordinary multi-omics feature fusion because each candidate feature is derived from a sheaf residual, group-specific sheaf energy, or transport-calibrated sheaf distance.

The correct current interpretation is that Phase 6 improves internal biological-discovery rigor and identifies candidate residual signatures. These signatures remain internal until tested on an external cohort such as CGGA.

\section{Deliverables}
\begin{itemize}[leftmargin=*]
\item \texttt{phase6\_consensus\_feature\_discovery.csv}: feature-level CSDS table.
\item \texttt{phase6\_patient\_sheaf\_discovery\_index.csv}: patient-level SDI scores.
\item \texttt{phase6\_prediction\_metrics.csv}: cross-validated prediction metrics.
\item \texttt{phase6\_metric\_deltas.csv}: baseline vs Phase 6 improvements.
\item \texttt{phase6\_cross\_task\_consensus.csv}: recurrent discoveries across tasks.
\item Figures for CSDS rankings, SDI distributions, and cross-task consensus.
\end{itemize}

\end{document}
'''
(pack/'paper/phase6_technical_specification.tex').write_text(tex)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase6_technical_specification.tex'], cwd=str(pack/'paper'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase6_technical_specification.tex'], cwd=str(pack/'paper'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.copy(pack/'paper/phase6_technical_specification.pdf', base/'phase6_technical_specification.pdf')
shutil.copy(pack/'paper/phase6_technical_specification.tex', base/'phase6_technical_specification.tex')

# Create documentation assets and PDF for phases 4-6
asset=base/'phase456_team_doc_assets'
if asset.exists(): shutil.rmtree(asset)
asset.mkdir()
figs=[
base/'phase4_subtype_sheaf_geometry_package/figures/phase4_divergence_heatmap_grade_label_strict_no_grade_no_clusters.png',
base/'phase4_subtype_sheaf_geometry_package/figures/phase4_balanced_accuracy_comparison.png',
base/'phase5_transport_sheaf_stability_package/figures/phase5_ot_sheaf_gap_heatmap_grade_label_strict_no_idh_no_grade_no_clusters.png',
base/'phase5_transport_sheaf_stability_package/figures/phase5_balanced_accuracy_deltas.png',
pack/'figures/phase6_cross_task_CSDS_heatmap.png',
pack/'figures/phase6_top_CSDS_grade_label.png']
for f in figs:
    if f.exists(): shutil.copy(f, asset/f.name)
for f in [pack/'paper/phase6_best_delta_table.tex', pack/'paper/phase6_top_feature_table.tex']:
    shutil.copy(f, asset/f.name)

doc_tex = r'''
\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{float}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{Team Documentation: Phases 4--6\\Subtype Sheaf Geometry, Transport Stability, and Consensus Discovery}
\author{Sheaf-Theoretic Multi-Omics Brain Tumor Project}
\date{May 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage

\section{Executive Summary}
Phases 4--6 form the advanced technical core of the project. Phase 4 learns group-specific sheaf Laplacians and asks which biological group law best explains each patient. Phase 5 adds optimal-transport robustness to test whether sheaf residual signatures remain stable under cross-group alignment. Phase 6 turns these residuals into reliability-ranked discovery features using stability selection, permutation testing, FDR control, and transport-calibrated consensus scoring.

\section{How These Phases Fit Together}
\begin{center}
\begin{tabular}{lll}
\toprule
Phase & Main object & Main question \\
\midrule
4 & Group-specific sheaf Laplacians $L_g$ & Do subtypes/grades obey different regulatory laws? \\
5 & OT transport plans $\Gamma^{\star}_{A,B}$ & Are residual signatures stable under group transport? \\
6 & CSDS and SDI & Which sheaf residual biomarkers are reliable? \\
\bottomrule
\end{tabular}
\end{center}

\section{Phase 4: Subtype-Specific Counterfactual Sheaf Geometry}
For each biological group $g$, Phase 4 learns restriction maps and constructs a group-specific sheaf coboundary matrix $B_g$. The group-specific sheaf Laplacian is
\[
L_g=B_g^\top B_g.
\]
For patient $p$, the counterfactual energy under group $g$ is
\[
E_g(p)=x_p^\top L_g x_p,
\qquad
\widehat{g}(p)=\arg\min_g E_g(p).
\]
Phase 4 asks which learned biological consistency geometry best explains each patient, not merely which label a classifier predicts.

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase4_divergence_heatmap_grade_label_strict_no_grade_no_clusters.png}
\caption{Phase 4 group-specific sheaf Laplacian divergence for strict grade geometry.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase4_balanced_accuracy_comparison.png}
\caption{Phase 4 strict balanced-accuracy comparison.}
\end{figure}

\section{Phase 5: Transport-Calibrated Sheaf Stability}
For groups $A$ and $B$, Phase 5 solves an entropic optimal transport problem
\[
\Gamma_{A,B}^{\star}=\arg\min_{\Gamma\in U(a,b)}\langle \Gamma,C_{A,B}\rangle+\varepsilon \mathrm{KL}(\Gamma\|ab^\top).
\]
The transported sheaf discrepancy is
\[
\operatorname{TSD}(A,B)=\sum_{i\in A}\sum_{j\in B}\Gamma_{ij}^{\star}\|\mathbf{s}(p_i)-\mathbf{s}(q_j)\|_2.
\]
This tests whether residual signatures remain coherent after cross-group distributional alignment.

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase5_ot_sheaf_gap_heatmap_grade_label_strict_no_idh_no_grade_no_clusters.png}
\caption{Phase 5 OT sheaf discrepancy heatmap for strict grade protocol.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.82\textwidth]{phase5_balanced_accuracy_deltas.png}
\caption{Phase 5 balanced-accuracy deltas from adding transport features.}
\end{figure}

\section{Phase 6: Consensus Sheaf Discovery and Reliability}
Phase 6 evaluates candidate sheaf features $\phi_j(p)$ using four criteria: stability-selection frequency $\pi_j$, normalized biological effect $a_j$, FDR-adjusted reliability $1-q_j$, and transport stability $\tau_j$. The Consensus Sheaf Discovery Score is
\[
\boxed{\operatorname{CSDS}_j=\pi_j a_j(1-q_j)\tau_j.}
\]
The patient-level Sheaf Discovery Index is
\[
\operatorname{SDI}(p)=\sum_{j\in\mathcal{T}_K}w_jz(\phi_j(p)),
\qquad
w_j\propto \operatorname{sign}(\bar\beta_j)\operatorname{CSDS}_j.
\]
Phase 6 turns sheaf residuals into candidate biomarkers with uncertainty and transport-calibrated reliability.

\subsection{Key Tables}
\begin{center}
\input{phase6_best_delta_table.tex}
\end{center}

\begin{center}
\resizebox{\textwidth}{!}{\input{phase6_top_feature_table.tex}}
\end{center}

\begin{figure}[H]
\centering
\includegraphics[width=.85\textwidth]{phase6_cross_task_CSDS_heatmap.png}
\caption{Phase 6 cross-task consensus sheaf discovery heatmap.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.85\textwidth]{phase6_top_CSDS_grade_label.png}
\caption{Top consensus sheaf discovery features for strict grade classification.}
\end{figure}

\section{What Teammates Should Understand}
\subsection{For Math Team}
The advanced mathematical objects are $L_g$, $E_g(p)$, $\Gamma^{\star}_{A,B}$, TSD, CSDS, and SDI. The core novelty is converting sheaf Laplacian residuals into subtype laws, transport-stability geometry, and reliability-ranked discovery features.

\subsection{For CS Team}
The main code paths are:
\begin{itemize}[leftmargin=*]
\item \texttt{phase4\_subtype\_sheaf\_geometry.py}
\item \texttt{phase5\_transport\_sheaf\_stability.py}
\item \texttt{phase6\_consensus\_sheaf\_discovery.py}
\end{itemize}
The important implementation idea is strict out-of-fold or leakage-aware evaluation. Avoid training on labels or features that trivially encode the label.

\subsection{For Biology Team}
The biological interpretation is that the method identifies where tumor molecular states stop behaving coherently. Residuals can be interpreted as DNA-to-regulatory inconsistency, DNA-to-phenotype inconsistency, or regulatory-to-phenotype inconsistency. Phase 6 turns these into candidate biomarkers, but they need biological validation before being called discoveries.

\section{Current Strengths and Limitations}
\subsection{Strengths}
\begin{itemize}[leftmargin=*]
\item The project now has a multi-phase technical pipeline rather than a single score.
\item Phase 4 shows group-specific regulatory geometry.
\item Phase 5 shows transport-calibrated residual stability.
\item Phase 6 adds uncertainty-aware biomarker ranking.
\end{itemize}

\subsection{Limitations}
\begin{itemize}[leftmargin=*]
\item These are internal cohort results.
\item External validation is still required.
\item Strong biomarker claims require pathway/gene-level biological annotation.
\item Accuracy gains must be reported carefully and not overstated.
\end{itemize}

\section{Immediate Next Steps}
\begin{enumerate}[leftmargin=*]
\item Run the same Phase 1--6 pipeline on CGGA or another external glioma cohort.
\item Replace abstract node-level variables with gene/pathway-level regulatory edges.
\item Map top Phase 6 features to biological pathways.
\item Add bootstrapped confidence intervals to all main deltas.
\item Prepare BIBM methods and ablation sections using Phases 4--6 as the advanced technical core.
\end{enumerate}

\end{document}
'''
(asset/'phase456_team_documentation.tex').write_text(doc_tex)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase456_team_documentation.tex'], cwd=str(asset), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['pdflatex','-interaction=nonstopmode','phase456_team_documentation.tex'], cwd=str(asset), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.copy(asset/'phase456_team_documentation.pdf', base/'phase456_team_documentation.pdf')
shutil.copy(asset/'phase456_team_documentation.tex', base/'phase456_team_documentation.tex')

# Render PDFs and contact sheets to verify
for pdf_name,outdir in [('phase6_technical_specification.pdf','phase6_technical_renders'),('phase456_team_documentation.pdf','phase456_team_doc_renders')]:
    pdf_path=base/pdf_name
    out_path=base/outdir
    if out_path.exists(): shutil.rmtree(out_path)
    subprocess.run(['python','/home/oai/skills/pdfs/scripts/render_pdf.py', str(pdf_path),'--out_dir',str(out_path),'--dpi','120'], check=True, stdout=subprocess.DEVNULL)
    imgs=sorted(out_path.glob('*.png'))[:12]
    thumbs=[]
    for imgp in imgs:
        im=Image.open(imgp).convert('RGB')
        im.thumbnail((260,360))
        canvas=Image.new('RGB',(280,390),'white')
        canvas.paste(im,((280-im.width)//2,10))
        d=ImageDraw.Draw(canvas); d.text((10,370),imgp.stem,fill='black')
        thumbs.append(canvas)
    if thumbs:
        cols=3; rows=math.ceil(len(thumbs)/cols)
        sheet=Image.new('RGB',(cols*280,rows*390),'white')
        for i,t in enumerate(thumbs): sheet.paste(t,((i%cols)*280,(i//cols)*390))
        sheet.save(base/(outdir.replace('_renders','_contact_sheet.png')))

# zip package
zip_path=base/'phase6_consensus_sheaf_discovery_package.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in pack.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(pack.parent))
print('Created', zip_path)
print('Created', base/'phase6_technical_specification.pdf')
print('Created', base/'phase456_team_documentation.pdf')
