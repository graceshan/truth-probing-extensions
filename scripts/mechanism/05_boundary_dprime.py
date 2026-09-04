#!/usr/bin/env python3
"""Compute descriptive standardized boundary separations on held-out R1 rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
PREDICTIONS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
BALANCE_PATH = MECHANISM_DIR / "cell_balance_summary.json"
ATOMIC_AUC_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
COMPOUND_AUC_PATH = MECHANISM_DIR / "pairwise_auc_compound.csv"
OUTPUT_PATH = MECHANISM_DIR / "boundary_dprime.csv"

PROBES = {
    "Atomic": "atomic_union_to_compounds_score",
    "Compound": "compound_trained_to_compounds_score",
}


def dprime(x: np.ndarray, y: np.ndarray) -> float:
    """Return pooled-variance descriptive separation using sample variances."""
    denominator = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    if denominator == 0:
        raise ValueError("Cannot compute d-prime when both sample variances are zero")
    return float((np.mean(x) - np.mean(y)) / denominator)


def values_for_cells(
    frame: pd.DataFrame, score_column: str, cells: Sequence[str]
) -> np.ndarray:
    """Pool and return all row-level scores belonging to the requested cells."""
    values = frame.loc[frame["cell"].isin(cells), score_column].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No rows found for cells {cells}")
    return values


def main() -> None:
    with BALANCE_PATH.open() as handle:
        balance = json.load(handle)
    if not (
        balance["n_test_pairs"] == 100
        and balance["n_test_rows"] == 1600
        and balance["zero_pair_overlap"]
        and balance["exactly_balanced_within_connective"]
    ):
        raise ValueError("Task 1 held-out balance prerequisites failed")

    heldout = pd.read_csv(PREDICTIONS_PATH)
    heldout["connective"] = heldout["connective"].str.upper()
    required = {"entity_pair_id", "connective", "cell", *PROBES.values()}
    missing = required.difference(heldout.columns)
    if missing:
        raise ValueError(f"Missing held-out columns: {sorted(missing)}")
    if len(heldout) != 1600 or heldout["entity_pair_id"].nunique() != 100:
        raise ValueError("Held-out subset mismatch")
    if not np.isfinite(heldout[list(PROBES.values())].to_numpy()).all():
        raise ValueError("Probe scores contain NaN or infinity")

    comparisons = [
        ("TT_vs_mixed", "TT", "mixed", ("TT",), ("TF", "FT")),
        ("mixed_vs_FF", "mixed", "FF", ("TF", "FT"), ("FF",)),
        ("TT_vs_TF", "TT", "TF", ("TT",), ("TF",)),
        ("TT_vs_FT", "TT", "FT", ("TT",), ("FT",)),
        ("TT_vs_FF", "TT", "FF", ("TT",), ("FF",)),
        ("TF_vs_FF", "TF", "FF", ("TF",), ("FF",)),
        ("FT_vs_FF", "FT", "FF", ("FT",), ("FF",)),
        ("TF_vs_FT", "TF", "FT", ("TF",), ("FT",)),
    ]
    output_rows: list[dict[str, object]] = []
    for probe, score_column in PROBES.items():
        for connective in ("AND", "OR"):
            subset = heldout.loc[heldout["connective"].eq(connective)]
            counts = subset.groupby("cell").size().to_dict()
            if counts != {"FF": 200, "FT": 200, "TF": 200, "TT": 200}:
                raise ValueError(f"{connective} cells are not balanced: {counts}")
            for comparison, high_name, low_name, high_cells, low_cells in comparisons:
                high = values_for_cells(subset, score_column, high_cells)
                low = values_for_cells(subset, score_column, low_cells)
                output_rows.append(
                    {
                        "probe": probe,
                        "connective": connective,
                        "comparison": comparison,
                        "group_high": high_name,
                        "group_low": low_name,
                        "n_high": len(high),
                        "n_low": len(low),
                        "mean_high": float(np.mean(high)),
                        "mean_low": float(np.mean(low)),
                        "sd_high": float(np.std(high, ddof=1)),
                        "sd_low": float(np.std(low, ddof=1)),
                        "dprime": dprime(high, low),
                    }
                )

    output = pd.DataFrame(output_rows)
    if len(output) != 32 or not np.isfinite(output.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Unexpected or non-finite d-prime output")
    output.to_csv(OUTPUT_PATH, index=False)

    # Read back the already-computed AUROCs solely for the printed reference table.
    auc_frames = []
    for probe, path in (("Atomic", ATOMIC_AUC_PATH), ("Compound", COMPOUND_AUC_PATH)):
        auc = pd.read_csv(path)
        auc["probe_display"] = probe
        auc["connective"] = auc["connective"].str.upper()
        auc_frames.append(auc)
    aucs = pd.concat(auc_frames, ignore_index=True)
    reference = aucs.loc[
        ((aucs["cell_high"] == "TT") & (aucs["cell_low"] == "mixed"))
        | ((aucs["cell_high"] == "mixed") & (aucs["cell_low"] == "FF")),
        ["probe_display", "connective", "cell_high", "cell_low", "auroc"],
    ]
    print(output.to_string(index=False))
    print("\nSaved pairwise-AUROC references:\n" + reference.to_string(index=False))


if __name__ == "__main__":
    main()
