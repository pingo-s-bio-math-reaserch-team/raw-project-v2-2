# Phase-by-Phase Runbook

## Phase 1: Fixed Sheaf Residual Engine

**Purpose:** Build the first rigorous sheaf residual representation over genomic, regulatory, and phenotype states.

**Command:**

```bash
python run_one_phase.py 1
```

**Inputs:**

```text
data/data.txt
```

**Main code:**

```text
phase1_sheaf_residual_engine/src/phase1_sheaf_engine.py
```

**Main outputs:**

```text
phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv
phase1_sheaf_residual_engine/results/phase1_sris_results.csv
phase1_sheaf_residual_engine/results/phase1_sheaf_metadata.json
phase1_sheaf_residual_engine/results/phase1_summary.json
```

**What it computes:**

\[
\operatorname{SRIS}(p)=x(p)^T L_{\mathcal F}x(p).
\]

Age is excluded from the inconsistency score and preserved only as an external covariate.

---

## Phase 2: Learned Sheaf Restriction Maps

**Purpose:** Replace purely fixed residual rules with learned restriction maps.

**Command:**

```bash
python run_one_phase.py 2
```

**Inputs:**

```text
phase1_sheaf_residual_engine/results/phase1_clean_encoded.csv
```

**Main code:**

```text
phase2_learned_sheaf_maps/src/phase2_learned_sheaf.py
phase2_learned_sheaf_maps/src/phase2_accuracy_benchmark.py
phase2_learned_sheaf_maps/src/phase2_strict_accuracy_benchmark.py
```

**Main outputs:**

```text
phase2_learned_sheaf_maps/results/phase2_sris_all_models.csv
phase2_learned_sheaf_maps/results/phase2_model_summary.csv
phase2_learned_sheaf_maps/results/phase2_validation_metrics.csv
phase2_learned_sheaf_maps/results/phase2_maps_and_laplacians.json
```

**What it computes:**

\[
r_{uv}(p)=W_{uv}x_u(p)-x_v(p),\qquad
\operatorname{SRIS}_2(p)=\sum_{u\to v}\frac{1}{d_v}\|W_{uv}x_u(p)-x_v(p)\|_2^2.
\]

---

## Phase 3: Survival Validation

**Purpose:** Test whether sheaf residuals have survival/risk signal under Cox-style modeling and cross-validated C-index.

**Command:**

```bash
python run_one_phase.py 3
```

**Inputs:**

```text
phase1_sheaf_residual_engine/results/phase1_sris_results.csv
phase2_learned_sheaf_maps/results/phase2_sris_all_models.csv
```

**Main outputs:**

```text
phase3_survival_validation/results/phase3_survival_model_summary.csv
phase3_survival_validation/results/phase3_cox_coefficients.csv
phase3_survival_validation/results/phase3_likelihood_ratio_tests.csv
phase3_survival_validation/results/phase3_out_of_fold_risks.csv
```

**Interpretation:** If SRIS is statistically significant but C-index gain is small, the claim should be "survival-associated and interpretable," not "survival SOTA."

---

## Phase 4: Subtype-Specific Sheaf Geometry

**Purpose:** Learn group-specific sheaf Laplacians and compare subtype/grade regulatory geometries.

**Command:**

```bash
python run_one_phase.py 4
```

**Main outputs:**

```text
phase4_subtype_sheaf_geometry/results/phase4_laplacian_divergences.csv
phase4_subtype_sheaf_geometry/results/phase4_permutation_divergence_tests.csv
phase4_subtype_sheaf_geometry/results/phase4_counterfactual_accuracy_metrics.csv
phase4_subtype_sheaf_geometry/results/phase4_counterfactual_patient_energies.csv
```

**Core object:**

\[
E_g(p)=x(p)^T L_g x(p),\qquad \widehat g(p)=\arg\min_g E_g(p).
\]

---

## Phase 5: Transport-Calibrated Sheaf Stability

**Purpose:** Use OT to test whether sheaf residual signatures are stable across biological group shifts.

**Command:**

```bash
python run_one_phase.py 5
```

**Main outputs:**

```text
phase5_transport_sheaf_stability/results/phase5_pairwise_transport_metrics.csv
phase5_transport_sheaf_stability/results/phase5_patient_transport_features.csv
phase5_transport_sheaf_stability/results/phase5_permutation_transport_tests.csv
```

**Core metric:**

\[
\operatorname{TSD}(A,B)=\sum_{i,j}\Gamma_{ij}^{\star}\|\mathbf{s}(p_i)-\mathbf{s}(q_j)\|_2.
\]

---

## Phase 6: Consensus Sheaf Discovery

**Purpose:** Rank reliable sheaf-derived signatures using stability, prediction, and transport evidence.

**Command:**

```bash
python run_one_phase.py 6
```

**Main outputs:**

```text
phase6_consensus_sheaf_discovery/results/phase6_consensus_feature_discovery.csv
phase6_consensus_sheaf_discovery/results/phase6_prediction_metrics.csv
phase6_consensus_sheaf_discovery/results/phase6_cross_task_consensus.csv
phase6_consensus_sheaf_discovery/results/phase6_summary.json
```

**Interpretation:** Phase 6 produces candidate sheaf residual biomarkers/signatures, not final clinical biomarkers.

---

## Phase 7: Publication Synthesis / Claim Ledger

**Purpose:** Consolidate what can safely be claimed and what still needs external validation.

**Command:**

```bash
python run_one_phase.py 7
```

**Main outputs:**

```text
phase7_publication_synthesis/results/phase7_local_run_note.txt
combined_results/phase7/phase7_local_run_note.txt
```

The full publication synthesis tables are also in the original Phase 7 package and in prior generated outputs.

---

## Optional Phase 8: Publishability Upgrade

**Purpose:** Lockbox-style internal validation, CGGA adapter scaffold, pathway-sheaf scaffold, and IEEE BIBM manuscript skeleton.

**Command:**

```bash
python run_one_phase.py 8
```

**Main outputs:**

```text
phase8_publishability_upgrade/results/phase8_lockbox_holdout_metrics.csv
phase8_publishability_upgrade/results/phase8_best_lockbox_deltas.csv
phase8_publishability_upgrade/results/phase8_publishability_checklist.csv
phase8_publishability_upgrade/manuscript/IEEE_BIBM_skeleton.tex
```

**Note:** Phase 8 does not replace true external validation. It prepares the project for CGGA or another held-out cohort.
