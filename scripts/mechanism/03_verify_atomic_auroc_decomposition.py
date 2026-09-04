#!/usr/bin/env python3
"""Verify the balanced-cell decomposition of held-out atomic-probe AUROC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
PREDICTIONS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
PAIRWISE_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
BALANCE_PATH = MECHANISM_DIR / "cell_balance_summary.json"
OUTPUT_PATH = MECHANISM_DIR / "atomic_auroc_decomposition.json"
SCORE_COLUMN = "atomic_union_to_compounds_score"
TOLERANCE = 1e-12


def main() -> None:
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
    pairwise = pd.read_csv(PAIRWISE_PATH)
    heldout["connective"] = heldout["connective"].str.upper()
    pairwise["connective"] = pairwise["connective"].str.upper()

    required = {"entity_pair_id", "connective", "cell", "global_truth", SCORE_COLUMN}
    missing = required.difference(heldout.columns)
    if missing:
        raise ValueError(f"Missing held-out columns: {sorted(missing)}")
    if len(heldout) != 1600 or heldout["entity_pair_id"].nunique() != 100:
        raise ValueError("Held-out subset mismatch")
    if not np.isfinite(heldout[SCORE_COLUMN]).all():
        raise ValueError("Atomic scores contain NaN or infinity")

    expected_truth = np.where(
        heldout["connective"].eq("AND"),
        heldout["cell"].eq("TT"),
        ~heldout["cell"].eq("FF"),
    ).astype(int)
    if not np.array_equal(expected_truth, heldout["global_truth"].to_numpy(dtype=int)):
        raise ValueError("Global-truth labels disagree with connective/cell semantics")

    formulas = {
        "AND": [("TT", "TF"), ("TT", "FT"), ("TT", "FF")],
        "OR": [("TT", "FF"), ("TF", "FF"), ("FT", "FF")],
    }
    results: dict[str, dict[str, object]] = {}
    for connective, terms in formulas.items():
        rows = heldout.loc[heldout["connective"].eq(connective)]
        counts = rows.groupby("cell").size().to_dict()
        if counts != {"FF": 200, "FT": 200, "TF": 200, "TT": 200}:
            raise ValueError(f"{connective} cells are not exactly balanced: {counts}")

        direct = float(roc_auc_score(rows["global_truth"], rows[SCORE_COLUMN]))
        term_values: dict[str, float] = {}
        for high, low in terms:
            match = pairwise.loc[
                pairwise["connective"].eq(connective)
                & pairwise["cell_high"].eq(high)
                & pairwise["cell_low"].eq(low)
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one saved pairwise value for {connective} {high}>{low}")
            term_values[f"A({high},{low})"] = float(match.iloc[0]["auroc"])
        reconstructed = float(np.mean(list(term_values.values())))
        signed_difference = reconstructed - direct
        absolute_difference = abs(signed_difference)
        passed = absolute_difference <= TOLERANCE
        results[connective] = {
            "direct_auroc": direct,
            "reconstructed_auroc": reconstructed,
            "signed_difference_reconstructed_minus_direct": signed_difference,
            "absolute_difference": absolute_difference,
            "pairwise_terms": term_values,
            "pass": passed,
        }

    overall_pass = all(bool(item["pass"]) for item in results.values())
    output = {
        "status": "PASS" if overall_pass else "FAIL",
        "tolerance": TOLERANCE,
        "difference_orientation": "reconstructed_minus_direct",
        "heldout_predictions_source": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "pairwise_auc_source": str(PAIRWISE_PATH.relative_to(ROOT)),
        "balance_summary_source": str(BALANCE_PATH.relative_to(ROOT)),
        "score_column": SCORE_COLUMN,
        "results": results,
        "bug_found": False if overall_pass else None,
    }
    MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))

    if not overall_pass:
        raise RuntimeError("FAILED: direct and reconstructed AUROCs disagree beyond tolerance")


if __name__ == "__main__":
    main()
