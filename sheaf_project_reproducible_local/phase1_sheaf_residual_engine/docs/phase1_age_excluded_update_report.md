# Phase 1 Update: Age Excluded From SRIS

## Design correction

Age has been removed from the sheaf phenotype node. This is the correct modeling choice because patient age is a baseline covariate and potential downstream association variable, not a tumor-intrinsic regulatory inconsistency feature.

The updated Phase 1 model now computes SRIS from:

- **D node**: DNA/genomic risk state
- **R node**: regulatory/transcriptomic risk state
- **C node**: tumor/clinical phenotype state, excluding survival and excluding age

Age is preserved in the output table for downstream analyses such as:

- SRIS-age correlation
- age-adjusted subtype classification
- age-adjusted survival modeling
- covariate adjustment in Cox regression

## Updated C node

The updated C node is:

\[
C = [\text{grade risk},\; \text{low KPS},\; \text{low purity}].
\]

Age is not included in:

\[
B_{\mathcal F}, \quad L_{\mathcal F}, \quad \operatorname{SRIS}(p).
\]

The updated sheaf energy remains:

\[
\operatorname{SRIS}(p)=x_p^T L_{\mathcal F}x_p=\sum_e \|r_e(p)\|^2.
\]

## Updated pilot output

- Patients: 420
- Edges: 3
- Sheaf features: 19
- Mean SRIS: 4.3622
- Median SRIS: 2.8213

## External age analysis

Age is now treated as an external variable. The pilot Spearman association between SRIS and age is:

- Spearman rho: 0.0920
- p-value: 0.0597

This should be reported as a diagnostic association, not as part of the inconsistency definition.

## Why this improves the framework

This avoids a major confounding problem: older patients are more likely to have aggressive glioma phenotypes, but age itself is not a regulatory contradiction among omics layers. Removing age makes SRIS closer to a tumor-derived inconsistency score and makes future claims stronger, especially if SRIS remains predictive after age adjustment.

The key downstream test is now:

\[
h(t|p)=h_0(t)\exp(\beta_1\operatorname{SRIS}(p)+\beta_2\text{age}+\beta_3\text{grade}+\beta_4\text{IDH}+\cdots).
\]

If \(\beta_1\) remains significant, then SRIS adds independent information beyond age, grade, and IDH.
