#!/usr/bin/env python3
"""Evaluate the compound-probe component orthogonal to the atomic direction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
DECOMPOSITION_PATH = MECHANISM_DIR / "probe_direction_decomposition.npz"
COMPOUND_PROBE_PATH = ROOT / "results" / "r1" / "compound_probe_control_probe.npz"
ACTIVATIONS_PATH = ROOT / "acts" / "r1_quadruples.npy"
ACTIVATION_SIDECAR_PATH = ROOT / "acts" / "r1_quadruples.csv"
SCORE_TABLE_PATH = ROOT / "results" / "r1" / "r1_score_table.csv"
SPLIT_PATH = ROOT / "results" / "r1" / "g_c_cg" / "pair_split.csv"
HELDOUT_REFERENCE_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
TASK2_SCRIPT = Path(__file__).with_name("02_atomic_pairwise_auc.py")
OUTPUT_PAIRWISE = MECHANISM_DIR / "pairwise_auc_perp.csv"
OUTPUT_SUMMARY = MECHANISM_DIR / "perp_auc_summary.json"
OUTPUT_SCORES = MECHANISM_DIR / "perp_test_scores.csv"
LAYER = 22
DIMENSION = 3584
TOLERANCE = 1e-12


def load_task2_pairwise_function():
    spec = importlib.util.spec_from_file_location("task2_atomic_pairwise", TASK2_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Task 2 implementation from {TASK2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grouped_pairwise_auc


def main() -> None:
    grouped_pairwise_auc = load_task2_pairwise_function()
    decomposition = np.load(DECOMPOSITION_PATH, allow_pickle=False)
    compound_probe = np.load(COMPOUND_PROBE_PATH, allow_pickle=False)
    w_perp = np.asarray(decomposition["w_perp"], dtype=float).reshape(-1)
    w_parallel = np.asarray(decomposition["w_parallel"], dtype=float).reshape(-1)
    compound_intercept = float(np.asarray(compound_probe["intercept"]).reshape(-1)[0])

    table = pd.read_csv(SCORE_TABLE_PATH)
    sidecar = pd.read_csv(ACTIVATION_SIDECAR_PATH)
    split = pd.read_csv(SPLIT_PATH)
    reference = pd.read_csv(HELDOUT_REFERENCE_PATH)
    acts = np.load(ACTIVATIONS_PATH, mmap_mode="r")

    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not table[metadata_columns].equals(sidecar[metadata_columns]):
        raise ValueError("Compound score metadata is not row-aligned to the activation sidecar")
    if acts.shape != (8000, 28, DIMENSION):
        raise ValueError(f"Unexpected compound activation shape: {acts.shape}")
    if w_perp.shape != (DIMENSION,) or w_parallel.shape != (DIMENSION,):
        raise ValueError("Direction decomposition vectors are not 3,584-dimensional")

    train_pairs = set(split.loc[split["partition"].eq("train"), "entity_pair_id"])
    test_pairs = set(split.loc[split["partition"].eq("test"), "entity_pair_id"])
    if len(train_pairs) != 400 or len(test_pairs) != 100 or train_pairs & test_pairs:
        raise ValueError("Saved canonical-pair split is not disjoint 400/100")
    test_mask = table["entity_pair_id"].isin(test_pairs).to_numpy()
    test_indices = np.flatnonzero(test_mask)
    if len(test_indices) != 1600 or table.loc[test_mask, "entity_pair_id"].nunique() != 100:
        raise ValueError("Held-out set is not exactly 1,600 rows / 100 canonical pairs")
    if not np.array_equal(test_indices, reference["r1_row_index"].to_numpy(dtype=int)):
        raise ValueError("Held-out row order differs from the direct-transfer evaluation")

    heldout = table.loc[test_mask].copy().reset_index(drop=True)
    heldout["connective"] = heldout["connective"].str.upper()
    expected_counts = pd.Series(
        200,
        index=pd.MultiIndex.from_product(
            [["AND", "OR"], ["TT", "TF", "FT", "FF"]],
            names=["connective", "cell"],
        ),
    )
    observed_counts = heldout.groupby(["connective", "cell"]).size().reindex(expected_counts.index)
    if not observed_counts.equals(expected_counts):
        raise ValueError(f"Held-out connective/cell counts are not all 200:\n{observed_counts}")

    X = np.asarray(acts[test_indices, LAYER, :])
    if X.shape != (1600, DIMENSION) or not np.isfinite(X).all():
        raise ValueError("Held-out layer-22 activations have wrong shape or non-finite entries")
    score_perp = X @ w_perp
    score_parallel = X @ w_parallel
    reconstructed_compound = score_parallel + score_perp + compound_intercept
    saved_compound = reference["compound_trained_to_compounds_score"].to_numpy(dtype=float)
    reconstruction_abs = np.abs(reconstructed_compound - saved_compound)
    reconstruction_max_abs_error = float(reconstruction_abs.max())
    reconstruction_mean_abs_error = float(reconstruction_abs.mean())
    if not np.isfinite(score_perp).all() or not np.isfinite(reconstructed_compound).all():
        raise ValueError("Perpendicular or reconstructed compound scores are non-finite")
    if reconstruction_max_abs_error > TOLERANCE:
        raise ValueError(
            f"Compound-score reconstruction error {reconstruction_max_abs_error} exceeds {TOLERANCE}"
        )

    heldout["score_perp"] = score_perp
    comparisons = [
        ("TT", "TF", ("TT",), ("TF",)),
        ("TT", "FT", ("TT",), ("FT",)),
        ("TT", "FF", ("TT",), ("FF",)),
        ("TF", "FF", ("TF",), ("FF",)),
        ("FT", "FF", ("FT",), ("FF",)),
        ("TF", "FT", ("TF",), ("FT",)),
        ("TT", "mixed", ("TT",), ("TF", "FT")),
        ("mixed", "FF", ("TF", "FT"), ("FF",)),
    ]
    pairwise_rows: list[dict[str, object]] = []
    for connective in ("AND", "OR"):
        subset = heldout.loc[heldout["connective"].eq(connective)]
        for high_name, low_name, high_cells, low_cells in comparisons:
            n_high, n_low, auc = grouped_pairwise_auc(
                subset, "score_perp", high_cells, low_cells
            )
            pairwise_rows.append({
                "direction": "perp",
                "connective": connective,
                "higher_cell": high_name,
                "lower_cell": low_name,
                "n_higher": n_high,
                "n_lower": n_low,
                "auroc": auc,
            })
    pairwise = pd.DataFrame(pairwise_rows)

    direct_overall = float(roc_auc_score(heldout["global_truth"], score_perp))
    direct: dict[str, float] = {}
    reconstructed: dict[str, float] = {}
    differences: dict[str, float] = {}
    pooled: dict[str, dict[str, float]] = {}
    formulas = {
        "AND": [("TT", "TF"), ("TT", "FT"), ("TT", "FF")],
        "OR": [("TT", "FF"), ("TF", "FF"), ("FT", "FF")],
    }
    for connective in ("AND", "OR"):
        subset = heldout.loc[heldout["connective"].eq(connective)]
        direct[connective] = float(roc_auc_score(subset["global_truth"], subset["score_perp"]))
        lookup = pairwise.loc[pairwise["connective"].eq(connective)].set_index(
            ["higher_cell", "lower_cell"]
        )["auroc"]
        reconstructed[connective] = float(np.mean([lookup.loc[key] for key in formulas[connective]]))
        differences[connective] = reconstructed[connective] - direct[connective]
        pooled[connective] = {
            "TT_vs_mixed": float(lookup.loc[("TT", "mixed")]),
            "mixed_vs_FF": float(lookup.loc[("mixed", "FF")]),
        }
        if abs(differences[connective]) > TOLERANCE:
            raise ValueError(f"{connective} direct/reconstructed AUROC mismatch")

    score_output = pd.DataFrame({
        "r1_row_index": test_indices,
        "entity_pair_id": heldout["entity_pair_id"],
        "topic": heldout["topic"],
        "connective": heldout["connective"],
        "cell": heldout["cell"],
        "ordering": heldout["ordering"],
        "global_truth": heldout["global_truth"].astype(int),
        "score_perp": score_perp,
    })
    sanity_checks = {
        "activation_dimension_is_3584": X.shape[1] == DIMENSION,
        "w_perp_dimension_is_3584": w_perp.size == DIMENSION,
        "activations_finite": bool(np.isfinite(X).all()),
        "scores_finite": bool(np.isfinite(score_perp).all()),
        "n_heldout_rows_is_1600": len(heldout) == 1600,
        "n_heldout_pairs_is_100": heldout["entity_pair_id"].nunique() == 100,
        "each_connective_cell_has_200_rows": bool(observed_counts.eq(200).all()),
        "zero_train_test_pair_overlap": not bool(train_pairs & test_pairs),
        "heldout_order_matches_saved_evaluation": True,
        "compound_score_reconstruction_within_tolerance": reconstruction_max_abs_error <= TOLERANCE,
        "and_auc_decomposition_within_tolerance": abs(differences["AND"]) <= TOLERANCE,
        "or_auc_decomposition_within_tolerance": abs(differences["OR"]) <= TOLERANCE,
    }
    passed = all(sanity_checks.values())
    summary = {
        "status": "PASS" if passed else "FAIL",
        "layer": LAYER,
        "dimension": DIMENSION,
        "direct_overall_auroc": direct_overall,
        "direct_AND_auroc": direct["AND"],
        "direct_OR_auroc": direct["OR"],
        "reconstructed_AND_auroc": reconstructed["AND"],
        "reconstructed_OR_auroc": reconstructed["OR"],
        "reconstruction_difference_reconstructed_minus_direct": differences,
        "TT_vs_mixed_auroc": {key: value["TT_vs_mixed"] for key, value in pooled.items()},
        "mixed_vs_FF_auroc": {key: value["mixed_vs_FF"] for key, value in pooled.items()},
        "compound_score_reconstruction_max_abs_error": reconstruction_max_abs_error,
        "compound_score_reconstruction_mean_abs_error": reconstruction_mean_abs_error,
        "tolerance": TOLERANCE,
        "sanity_checks": sanity_checks,
        "sources": {
            "decomposition": str(DECOMPOSITION_PATH.relative_to(ROOT)),
            "compound_activations": str(ACTIVATIONS_PATH.relative_to(ROOT)),
            "activation_sidecar": str(ACTIVATION_SIDECAR_PATH.relative_to(ROOT)),
            "pair_split": str(SPLIT_PATH.relative_to(ROOT)),
            "heldout_reference": str(HELDOUT_REFERENCE_PATH.relative_to(ROOT)),
            "pairwise_implementation": str(TASK2_SCRIPT.relative_to(ROOT)),
        },
    }

    MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(OUTPUT_PAIRWISE, index=False)
    score_output.to_csv(OUTPUT_SCORES, index=False)
    with OUTPUT_SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(pairwise.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Perpendicular-direction AUROC validation failed")


if __name__ == "__main__":
    main()
