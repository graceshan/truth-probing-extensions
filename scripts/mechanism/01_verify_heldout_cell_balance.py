"""Checkpoint 1: verify balance of the existing held-out R1 partition only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_PATH = Path("results/r1/g_c_cg/pair_split.csv")
PREDICTIONS_PATH = Path("results/r1/probe_transfer_comparison_predictions.csv")
SCORE_TABLE_PATH = Path("results/r1/r1_score_table.csv")
OUTPUT_DIR = Path("results/r1/mechanism")


def main():
    split = pd.read_csv(SPLIT_PATH)
    heldout = pd.read_csv(PREDICTIONS_PATH)
    score_table = pd.read_csv(SCORE_TABLE_PATH)

    required_split = {"entity_pair_id", "partition", "n_rows"}
    required_heldout = {
        "r1_row_index", "entity_pair_id", "connective", "ordering", "cell"
    }
    if not required_split.issubset(split.columns):
        raise ValueError(f"split missing columns: {sorted(required_split - set(split.columns))}")
    if not required_heldout.issubset(heldout.columns):
        raise ValueError(
            f"held-out predictions missing columns: {sorted(required_heldout - set(heldout.columns))}"
        )

    train_pairs = set(split.loc[split.partition == "train", "entity_pair_id"])
    test_pairs = set(split.loc[split.partition == "test", "entity_pair_id"])
    overlap = train_pairs & test_pairs
    if len(split) != 500 or len(train_pairs) != 400 or len(test_pairs) != 100:
        raise AssertionError("saved split is not exactly 500 = 400 train + 100 test pairs")
    if not (split.n_rows == 16).all():
        raise AssertionError("saved split metadata does not report 16 variants per pair")

    if len(heldout) != 1600 or heldout.entity_pair_id.nunique() != 100:
        raise AssertionError("transfer evaluation is not exactly 100 pairs / 1600 rows")
    if set(heldout.entity_pair_id) != test_pairs:
        raise AssertionError("transfer prediction pairs differ from the saved test partition")
    if not (heldout.groupby("entity_pair_id").size() == 16).all():
        raise AssertionError("not every held-out pair contributes all 16 variants")

    row_indices = heldout.r1_row_index.to_numpy(dtype=int)
    if len(np.unique(row_indices)) != 1600:
        raise AssertionError("held-out prediction row indices are not unique")
    if row_indices.min() < 0 or row_indices.max() >= len(score_table):
        raise AssertionError("held-out prediction row index is outside the R1 score table")
    comparison_columns = ["entity_pair_id", "connective", "ordering", "cell"]
    expected_rows = score_table.loc[row_indices, comparison_columns].reset_index(drop=True)
    if not expected_rows.equals(heldout[comparison_columns].reset_index(drop=True)):
        raise AssertionError("held-out rows/order do not match indexed rows in R1 score table")

    connective_cell = (
        heldout.groupby(["connective", "cell"]).size().rename("n_rows").reset_index()
    )
    connective_ordering_cell = (
        heldout.groupby(["connective", "ordering", "cell"])
        .size().rename("n_rows").reset_index()
    )
    connective_order = {"and": 0, "or": 1}
    ordering_order = {"AB": 0, "BA": 1}
    cell_order = {"TT": 0, "TF": 1, "FT": 2, "FF": 3}
    connective_cell = connective_cell.sort_values(
        ["connective", "cell"],
        key=lambda x: x.map(connective_order if x.name == "connective" else cell_order),
    ).reset_index(drop=True)
    connective_ordering_cell = connective_ordering_cell.sort_values(
        ["connective", "ordering", "cell"],
        key=lambda x: x.map(
            connective_order if x.name == "connective"
            else ordering_order if x.name == "ordering"
            else cell_order
        ),
    ).reset_index(drop=True)

    expected_per_connective_cell = 200
    expected_per_connective_ordering_cell = 100
    balanced = (
        len(connective_cell) == 8
        and (connective_cell.n_rows == expected_per_connective_cell).all()
        and len(connective_ordering_cell) == 16
        and (
            connective_ordering_cell.n_rows == expected_per_connective_ordering_cell
        ).all()
    )

    output_counts = pd.concat(
        [
            connective_cell.assign(aggregation="connective_x_cell", ordering="all"),
            connective_ordering_cell.assign(
                aggregation="connective_x_ordering_x_cell"
            ),
        ],
        ignore_index=True,
    )[["aggregation", "connective", "ordering", "cell", "n_rows"]]
    summary = {
        "source_files": {
            "saved_pair_split": str(SPLIT_PATH),
            "exact_heldout_transfer_rows": str(PREDICTIONS_PATH),
            "row_order_cross_check": str(SCORE_TABLE_PATH),
        },
        "n_total_pairs": 500,
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "n_test_rows": len(heldout),
        "rows_per_test_pair": 16,
        "zero_pair_overlap": len(overlap) == 0,
        "exactly_balanced_within_connective": bool(balanced),
        "expected_rows_per_connective_cell": expected_per_connective_cell,
        "expected_rows_per_connective_ordering_cell": expected_per_connective_ordering_cell,
        "heldout_row_order_matches_transfer_predictions": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_counts.to_csv(OUTPUT_DIR / "cell_counts.csv", index=False)
    with (OUTPUT_DIR / "cell_balance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("CONNECTIVE x CELL")
    print(connective_cell.to_string(index=False))
    print("\nCONNECTIVE x ORDERING x CELL")
    print(connective_ordering_cell.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
