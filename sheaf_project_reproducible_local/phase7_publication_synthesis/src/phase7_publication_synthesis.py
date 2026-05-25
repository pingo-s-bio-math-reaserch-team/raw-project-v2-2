#!/usr/bin/env python3
"""Phase 7: publication synthesis and claim calibration.

This module reads Phases 1-6 outputs and creates an integrated evidence ledger.
It does not invent new biological results. Instead, it formalizes which results are
safe to claim, which remain internal-only, and what external validation is needed.
"""
from pathlib import Path
import pandas as pd
import numpy as np

WEIGHTS = {
    "mechanistic_novelty": 0.20,
    "validation_strictness": 0.20,
    "predictive_gain": 0.18,
    "robustness": 0.16,
    "interpretability": 0.16,
    "external_validation": 0.10,
}

def evidence_weighted_score(row, weights=WEIGHTS):
    """Compute an internal evidence-weighted score from dimension scores in [0,1]."""
    return 10.0 * sum(float(row[k]) * w for k, w in weights.items())

def claim_tier(score, external_validation):
    """Cap claims when external validation has not yet been performed."""
    if external_validation <= 0 and score >= 8:
        return "strong internal methodological contribution; external validation required"
    if score >= 7:
        return "strong internal contribution"
    if score >= 6:
        return "moderate internal contribution"
    return "exploratory contribution"

def safe_claim_ledger():
    """Return the manuscript claim ledger used in the Phase 7 report."""
    return pd.DataFrame([
        ("Safe-1", "We introduce a biologically constrained cellular-sheaf framework for glioma multi-omics inconsistency.", "safe"),
        ("Safe-2", "Subtype and grade groups exhibit statistically non-random sheaf Laplacian geometry under internal permutation tests.", "safe internal"),
        ("Safe-3", "Sheaf residual signatures exhibit non-random OT-calibrated transport gaps between biological groups.", "safe internal"),
        ("Caution-1", "The framework improves survival prediction over clinical-molecular baselines.", "caution"),
        ("Unsafe-1", "The method is state of the art for glioma survival prediction.", "not supported yet"),
    ], columns=["claim_id", "claim", "status"])
