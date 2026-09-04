#!/usr/bin/env python3
"""Compare saved atomic and compound logistic-regression coefficient directions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ATOMIC_PATH = ROOT / "results" / "qwen_union_probe_gate" / "selected_probe.npz"
COMPOUND_PATH = ROOT / "results" / "r1" / "compound_probe_control_probe.npz"
PREDICTIONS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "r1" / "mechanism" / "probe_direction_similarity.json"
EXPECTED_DIMENSION = 3584
EXPECTED_LAYER = 22


def main() -> None:
    atomic = np.load(ATOMIC_PATH, allow_pickle=False)
    compound = np.load(COMPOUND_PATH, allow_pickle=False)
    w_atomic = np.asarray(atomic["coef"], dtype=float).reshape(-1)
    w_compound = np.asarray(compound["coef"], dtype=float).reshape(-1)
    atomic_intercept = float(np.asarray(atomic["intercept"]).reshape(-1)[0])
    compound_intercept = float(np.asarray(compound["intercept"]).reshape(-1)[0])
    atomic_layer = int(atomic["layer"])
    compound_layer = int(compound["layer"])

    checks = {
        "same_dimension": w_atomic.shape == w_compound.shape,
        "dimension_is_3584": w_atomic.size == EXPECTED_DIMENSION,
        "atomic_entries_finite": bool(np.isfinite(w_atomic).all()),
        "compound_entries_finite": bool(np.isfinite(w_compound).all()),
        "intercepts_finite": bool(np.isfinite([atomic_intercept, compound_intercept]).all()),
        "layers_match_expected_22": atomic_layer == compound_layer == EXPECTED_LAYER,
        "atomic_classes_are_false_true": np.array_equal(atomic["classes"], np.array([0, 1])),
        "compound_classes_are_false_true": np.array_equal(compound["classes"], np.array([0, 1])),
    }
    if not all(checks.values()):
        raise ValueError(f"Probe archive sanity check failed: {checks}")

    norm_atomic = float(np.linalg.norm(w_atomic))
    norm_compound = float(np.linalg.norm(w_compound))
    checks["atomic_norm_nonzero"] = norm_atomic > 0
    checks["compound_norm_nonzero"] = norm_compound > 0
    if not checks["atomic_norm_nonzero"] or not checks["compound_norm_nonzero"]:
        raise ValueError("At least one probe coefficient vector has zero norm")

    # Confirm the saved, unmodified score orientations rank compound truth above falsehood.
    heldout = pd.read_csv(PREDICTIONS_PATH)
    labels = heldout["global_truth"].to_numpy(dtype=int)
    atomic_auc = float(roc_auc_score(labels, heldout["atomic_union_to_compounds_score"]))
    compound_auc = float(roc_auc_score(labels, heldout["compound_trained_to_compounds_score"]))
    checks["atomic_larger_score_is_more_true"] = atomic_auc > 0.5
    checks["compound_larger_score_is_more_true"] = compound_auc > 0.5
    sign_flip_required = not (
        checks["atomic_larger_score_is_more_true"]
        and checks["compound_larger_score_is_more_true"]
    )
    if sign_flip_required:
        raise RuntimeError(
            "A saved probe score orientation does not rank true labels above false labels; "
            "stopping without flipping either coefficient vector."
        )

    dot_product = float(w_atomic @ w_compound)
    cosine_similarity = float(dot_product / (norm_atomic * norm_compound))
    angle_degrees = float(np.degrees(np.arccos(np.clip(cosine_similarity, -1.0, 1.0))))
    sanity_checks_passed = all(checks.values()) and not sign_flip_required
    output = {
        "layer": EXPECTED_LAYER,
        "dimension": int(w_atomic.size),
        "atomic_norm": norm_atomic,
        "compound_norm": norm_compound,
        "dot_product": dot_product,
        "cosine_similarity": cosine_similarity,
        "angle_degrees": angle_degrees,
        "atomic_intercept": atomic_intercept,
        "compound_intercept": compound_intercept,
        "sign_flip_required": sign_flip_required,
        "sanity_checks_passed": sanity_checks_passed,
        "sanity_checks": checks,
        "orientation_evidence": {
            "atomic_heldout_compound_truth_auroc": atomic_auc,
            "compound_heldout_compound_truth_auroc": compound_auc,
            "classes_convention": "classes=[0,1], so positive decision scores favor truth label 1",
        },
        "sources": {
            "atomic_probe": str(ATOMIC_PATH.relative_to(ROOT)),
            "compound_probe": str(COMPOUND_PATH.relative_to(ROOT)),
            "heldout_orientation_check": str(PREDICTIONS_PATH.relative_to(ROOT)),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
