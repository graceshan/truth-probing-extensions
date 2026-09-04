#!/usr/bin/env python3
"""Compute prespecified truth-cell AUROCs for the frozen atomic union probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
PREDICTIONS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
CONNECTIVE_METRICS_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_by_connective.csv"
BALANCE_SUMMARY_PATH = MECHANISM_DIR / "cell_balance_summary.json"
OUTPUT_CSV = MECHANISM_DIR / "pairwise_auc_atomic.csv"
OUTPUT_JSON = MECHANISM_DIR / "pairwise_auc_atomic_summary.json"

PROBE = "atomic_union"
SCORE_COLUMN = "atomic_union_to_compounds_score"
PREDICTION = (
    "The atomic probe should separate TT from TF/FT strongly, while TF/FT versus FF "
    "should be substantially weaker. Given direct OR AUROC ≈ 0.679, if TT-versus-FF "
    "is around 0.95, the average mixed-versus-FF AUROC would need to be around 0.54."
)


def grouped_pairwise_auc(
    frame: pd.DataFrame,
    score_column: str,
    high_cells: Sequence[str],
    low_cells: Sequence[str],
) -> tuple[int, int, float]:
    """Return P(score from high group > score from low group), with ties split equally.

    Rows are pooled within each supplied cell group; cell scores are never averaged first.
    This function is score-column agnostic so it can be reused for another probe.
    """
    high = frame.loc[frame["cell"].isin(high_cells), score_column]
    low = frame.loc[frame["cell"].isin(low_cells), score_column]
    if high.empty or low.empty:
        raise ValueError(f"Empty comparison group: high={high_cells}, low={low_cells}")
    labels = np.concatenate((np.ones(len(high), dtype=int), np.zeros(len(low), dtype=int)))
    scores = np.concatenate((high.to_numpy(), low.to_numpy()))
    return len(high), len(low), float(roc_auc_score(labels, scores))


def main() -> None:
    with BALANCE_SUMMARY_PATH.open() as handle:
        balance = json.load(handle)
    if balance["n_test_pairs"] != 100 or balance["n_test_rows"] != 1600:
        raise ValueError("Saved held-out balance metadata is not the expected 100-pair/1,600-row split")
    if not balance["zero_pair_overlap"] or not balance["exactly_balanced_within_connective"]:
        raise ValueError("Saved held-out split did not pass the prerequisite balance checks")

    heldout = pd.read_csv(PREDICTIONS_PATH)
    required = {"entity_pair_id", "connective", "cell", SCORE_COLUMN}
    missing = required.difference(heldout.columns)
    if missing:
        raise ValueError(f"Missing held-out prediction columns: {sorted(missing)}")
    heldout["connective"] = heldout["connective"].str.upper()
    if len(heldout) != 1600 or heldout["entity_pair_id"].nunique() != 100:
        raise ValueError("Prediction file does not contain the saved 1,600-row/100-pair test subset")
    if not np.isfinite(heldout[SCORE_COLUMN]).all():
        raise ValueError(f"Non-finite values found in {SCORE_COLUMN}")

    expected_counts = pd.Series(200, index=pd.MultiIndex.from_product(
        [["AND", "OR"], ["TT", "TF", "FT", "FF"]], names=["connective", "cell"]
    ))
    observed_counts = heldout.groupby(["connective", "cell"]).size().reindex(expected_counts.index)
    if not observed_counts.equals(expected_counts):
        raise ValueError(f"Held-out connective/cell counts changed:\n{observed_counts}")

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
    rows: list[dict[str, object]] = []
    for connective in ("AND", "OR"):
        connective_rows = heldout.loc[heldout["connective"] == connective]
        for high_name, low_name, high_cells, low_cells in comparisons:
            n_high, n_low, auc = grouped_pairwise_auc(
                connective_rows, SCORE_COLUMN, high_cells, low_cells
            )
            rows.append(
                {
                    "probe": PROBE,
                    "connective": connective,
                    "cell_high": high_name,
                    "cell_low": low_name,
                    "n_high": n_high,
                    "n_low": n_low,
                    "auroc": auc,
                }
            )

    results = pd.DataFrame(rows)
    direct = pd.read_csv(CONNECTIVE_METRICS_PATH)
    direct = direct.loc[direct["probe"].str.lower().str.contains("atomic")]
    direct["connective"] = direct["connective"].str.upper()
    direct_auc = direct.set_index("connective")["auroc"].to_dict()
    if set(direct_auc) != {"AND", "OR"}:
        raise ValueError("Could not uniquely import the existing atomic direct AND/OR AUROCs")

    pooled: dict[str, dict[str, float]] = {}
    qualitative: dict[str, dict[str, object]] = {}
    for connective in ("AND", "OR"):
        sub = results.loc[results["connective"] == connective]
        lookup = sub.set_index(["cell_high", "cell_low"])["auroc"]
        pooled[connective] = {
            "TT_vs_mixed": float(lookup.loc[("TT", "mixed")]),
            "mixed_vs_FF": float(lookup.loc[("mixed", "FF")]),
        }
        upper = float(np.mean([lookup.loc[("TT", "TF")], lookup.loc[("TT", "FT")]]))
        lower = float(np.mean([lookup.loc[("TF", "FF")], lookup.loc[("FT", "FF")]]))
        qualitative[connective] = {
            "mean_TT_vs_TF_FT": upper,
            "mean_TF_FT_vs_FF": lower,
            "upper_minus_lower": upper - lower,
            "matches_expected_ordering": bool(upper > lower),
        }

    summary = {
        "probe": PROBE,
        "score_column": SCORE_COLUMN,
        "prediction": PREDICTION,
        "heldout_predictions_source": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "balance_summary_source": str(BALANCE_SUMMARY_PATH.relative_to(ROOT)),
        "direct_connective_metrics_source": str(CONNECTIVE_METRICS_PATH.relative_to(ROOT)),
        "n_test_pairs": int(heldout["entity_pair_id"].nunique()),
        "n_test_rows": int(len(heldout)),
        "direct_AND_AUROC": float(direct_auc["AND"]),
        "direct_OR_AUROC": float(direct_auc["OR"]),
        "pooled_TT_vs_mixed_AUROC": {key: value["TT_vs_mixed"] for key, value in pooled.items()},
        "pooled_mixed_vs_FF_AUROC": {key: value["mixed_vs_FF"] for key, value in pooled.items()},
        "qualitative_prediction_check": qualitative,
    }

    MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
