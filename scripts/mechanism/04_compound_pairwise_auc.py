#!/usr/bin/env python3
"""Compute and verify held-out truth-cell AUROCs for the compound-trained probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
PREDICTIONS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
BALANCE_PATH = MECHANISM_DIR / "cell_balance_summary.json"
ATOMIC_PAIRWISE_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
TASK2_SCRIPT = Path(__file__).with_name("02_atomic_pairwise_auc.py")
OUTPUT_CSV = MECHANISM_DIR / "pairwise_auc_compound.csv"
OUTPUT_JSON = MECHANISM_DIR / "compound_auroc_decomposition.json"
SCORE_COLUMN = "compound_trained_to_compounds_score"
TOLERANCE = 1e-12


def load_task2_pairwise_function():
    """Load the exact row-pooled implementation used for Task 2."""
    spec = importlib.util.spec_from_file_location("task2_atomic_pairwise", TASK2_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Task 2 implementation from {TASK2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grouped_pairwise_auc


def main() -> None:
    grouped_pairwise_auc = load_task2_pairwise_function()
    with BALANCE_PATH.open() as handle:
        balance = json.load(handle)
    if not (
        balance["n_test_pairs"] == 100
        and balance["n_test_rows"] == 1600
        and balance["zero_pair_overlap"]
        and balance["exactly_balanced_within_connective"]
        and balance["expected_rows_per_connective_cell"] == 200
    ):
        raise ValueError("Task 1 balance prerequisites failed")

    heldout = pd.read_csv(PREDICTIONS_PATH)
    heldout["connective"] = heldout["connective"].str.upper()
    required = {"entity_pair_id", "connective", "cell", "global_truth", SCORE_COLUMN}
    missing = required.difference(heldout.columns)
    if missing:
        raise ValueError(f"Missing held-out columns: {sorted(missing)}")
    if len(heldout) != 1600 or heldout["entity_pair_id"].nunique() != 100:
        raise ValueError("Held-out subset mismatch")
    if not np.isfinite(heldout[SCORE_COLUMN]).all():
        raise ValueError("Compound-trained scores contain NaN or infinity")

    expected_truth = np.where(
        heldout["connective"].eq("AND"),
        heldout["cell"].eq("TT"),
        ~heldout["cell"].eq("FF"),
    ).astype(int)
    if not np.array_equal(expected_truth, heldout["global_truth"].to_numpy(dtype=int)):
        raise ValueError("Global-truth labels disagree with connective/cell semantics")

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
    result_rows: list[dict[str, object]] = []
    for connective in ("AND", "OR"):
        subset = heldout.loc[heldout["connective"].eq(connective)]
        counts = subset.groupby("cell").size().to_dict()
        if counts != {"FF": 200, "FT": 200, "TF": 200, "TT": 200}:
            raise ValueError(f"{connective} cells are not exactly balanced: {counts}")
        for high_name, low_name, high_cells, low_cells in comparisons:
            n_high, n_low, auc = grouped_pairwise_auc(
                subset, SCORE_COLUMN, high_cells, low_cells
            )
            result_rows.append(
                {
                    "probe": "compound_trained",
                    "connective": connective,
                    "cell_high": high_name,
                    "cell_low": low_name,
                    "n_high": n_high,
                    "n_low": n_low,
                    "auroc": auc,
                }
            )
    results = pd.DataFrame(result_rows)

    formulas = {
        "AND": [("TT", "TF"), ("TT", "FT"), ("TT", "FF")],
        "OR": [("TT", "FF"), ("TF", "FF"), ("FT", "FF")],
    }
    decomposition: dict[str, dict[str, object]] = {}
    pooled: dict[str, dict[str, float]] = {}
    for connective, terms in formulas.items():
        subset = heldout.loc[heldout["connective"].eq(connective)]
        lookup = results.loc[results["connective"].eq(connective)].set_index(
            ["cell_high", "cell_low"]
        )["auroc"]
        direct = float(roc_auc_score(subset["global_truth"], subset[SCORE_COLUMN]))
        term_values = {f"A({high},{low})": float(lookup.loc[(high, low)]) for high, low in terms}
        reconstructed = float(np.mean(list(term_values.values())))
        signed_difference = reconstructed - direct
        absolute_difference = abs(signed_difference)
        decomposition[connective] = {
            "direct_auroc": direct,
            "reconstructed_auroc": reconstructed,
            "signed_difference_reconstructed_minus_direct": signed_difference,
            "absolute_difference": absolute_difference,
            "pairwise_terms": term_values,
            "pass": absolute_difference <= TOLERANCE,
        }
        pooled[connective] = {
            "TT_vs_mixed": float(lookup.loc[("TT", "mixed")]),
            "mixed_vs_FF": float(lookup.loc[("mixed", "FF")]),
        }

    atomic = pd.read_csv(ATOMIC_PAIRWISE_PATH)
    atomic["connective"] = atomic["connective"].str.upper()
    comparison: dict[str, dict[str, float]] = {}
    for connective in ("AND", "OR"):
        atomic_lookup = atomic.loc[atomic["connective"].eq(connective)].set_index(
            ["cell_high", "cell_low"]
        )["auroc"]
        comparison[connective] = {
            "atomic_TT_vs_mixed": float(atomic_lookup.loc[("TT", "mixed")]),
            "compound_TT_vs_mixed": pooled[connective]["TT_vs_mixed"],
            "atomic_mixed_vs_FF": float(atomic_lookup.loc[("mixed", "FF")]),
            "compound_mixed_vs_FF": pooled[connective]["mixed_vs_FF"],
        }

    overall_pass = all(bool(values["pass"]) for values in decomposition.values())
    output = {
        "status": "PASS" if overall_pass else "FAIL",
        "tolerance": TOLERANCE,
        "difference_orientation": "reconstructed_minus_direct",
        "heldout_predictions_source": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "balance_summary_source": str(BALANCE_PATH.relative_to(ROOT)),
        "task2_implementation_source": str(TASK2_SCRIPT.relative_to(ROOT)),
        "atomic_pairwise_source": str(ATOMIC_PAIRWISE_PATH.relative_to(ROOT)),
        "score_column": SCORE_COLUMN,
        "n_test_pairs": int(heldout["entity_pair_id"].nunique()),
        "n_test_rows": int(len(heldout)),
        "pooled_aurocs": pooled,
        "decomposition": decomposition,
        "atomic_compound_pooled_comparison": comparison,
        "bug_found": False if overall_pass else None,
    }

    MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(results.to_string(index=False))
    print(json.dumps(output, indent=2))
    if not overall_pass:
        raise RuntimeError("FAILED: direct and reconstructed compound AUROCs disagree")


if __name__ == "__main__":
    main()
