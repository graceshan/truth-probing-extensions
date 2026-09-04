#!/usr/bin/env python3
"""Fit strict truth-count plus connective-offset diagnostics on saved R1 scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "r1"
MECHANISM_DIR = RESULTS / "mechanism"
SPLIT_PATH = RESULTS / "g_c_cg" / "pair_split.csv"
FULL_SCORES_PATH = RESULTS / "compound_probe_control_predictions.csv"
HELDOUT_REFERENCE_PATH = RESULTS / "probe_transfer_comparison_predictions.csv"
METRICS_PATH = MECHANISM_DIR / "count_connective_metrics.csv"
RESIDUALS_PATH = MECHANISM_DIR / "count_connective_cell_residuals.csv"
PREDICTIONS_PATH = MECHANISM_DIR / "count_connective_test_predictions.csv"

PROBES = {
    "Atomic": "score_compound",
    "Compound": "compound_trained_score",
}
CELL_TO_COUNT = {"FF": 0, "TF": 1, "FT": 1, "TT": 2}


def main() -> None:
    split = pd.read_csv(SPLIT_PATH)
    scores = pd.read_csv(FULL_SCORES_PATH)
    heldout_reference = pd.read_csv(HELDOUT_REFERENCE_PATH)

    if len(split) != 500 or split["entity_pair_id"].nunique() != 500:
        raise ValueError("Saved split does not contain exactly 500 canonical pairs")
    train_ids = set(split.loc[split["partition"].eq("train"), "entity_pair_id"])
    test_ids = set(split.loc[split["partition"].eq("test"), "entity_pair_id"])
    if len(train_ids) != 400 or len(test_ids) != 100 or train_ids & test_ids:
        raise ValueError("Saved split is not a disjoint 400/100 canonical-pair split")

    split_map = split[["entity_pair_id", "partition"]].rename(
        columns={"partition": "authoritative_partition"}
    )
    data = scores.merge(split_map, on="entity_pair_id", how="left", validate="many_to_one")
    if data["authoritative_partition"].isna().any():
        raise ValueError("Some score rows lack a saved pair partition")
    if "partition" in data and not data["partition"].equals(data["authoritative_partition"]):
        raise ValueError("Full-score partition column disagrees with authoritative pair split")
    if len(data) != 8000 or not data.groupby("entity_pair_id").size().eq(16).all():
        raise ValueError("Expected exactly 16 rows for each of 500 canonical pairs")

    data["true_count"] = data["cell"].map(CELL_TO_COUNT)
    data["is_OR"] = data["connective"].str.lower().eq("or").astype(int)
    if data["true_count"].isna().any():
        raise ValueError("Unexpected truth cell")
    if not np.array_equal(data["true_count"].to_numpy(), data["labelA"] + data["labelB"]):
        raise ValueError("Truth-count mapping disagrees with constituent labels")

    train = data.loc[data["authoritative_partition"].eq("train")].copy()
    test = data.loc[data["authoritative_partition"].eq("test")].copy()
    if len(train) != 6400 or len(test) != 1600:
        raise ValueError("Expected 6,400 training and 1,600 held-out rows")
    if set(train["entity_pair_id"]) & set(test["entity_pair_id"]):
        raise ValueError("Train/test canonical-pair overlap detected")

    # Require exact held-out row identity/order and score equality with the direct evaluation.
    reference_columns = [
        "r1_row_index", "entity_pair_id", "topic", "connective", "cell", "ordering",
        "atomic_union_to_compounds_score", "compound_trained_to_compounds_score",
    ]
    ref = heldout_reference[reference_columns].copy()
    check = test.merge(ref, on="r1_row_index", how="outer", suffixes=("", "_ref"), validate="one_to_one")
    if len(check) != 1600 or check["entity_pair_id_ref"].isna().any():
        raise ValueError("Held-out rows do not match the direct-transfer evaluation")
    for column in ("entity_pair_id", "topic", "connective", "cell", "ordering"):
        if not check[column].equals(check[f"{column}_ref"]):
            raise ValueError(f"Held-out {column} mismatch against direct evaluation")
    if not np.array_equal(
        check["score_compound"].to_numpy(),
        check["atomic_union_to_compounds_score"].to_numpy(),
    ):
        raise ValueError("Atomic held-out scores differ from direct evaluation")
    if not np.array_equal(
        check["compound_trained_score"].to_numpy(),
        check["compound_trained_to_compounds_score"].to_numpy(),
    ):
        raise ValueError("Compound-trained held-out scores differ from direct evaluation")

    features = ["true_count", "is_OR"]
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    spacing_rows: list[dict[str, object]] = []
    for probe, score_column in PROBES.items():
        model = LinearRegression(fit_intercept=True)
        model.fit(train[features], train[score_column])
        predicted = model.predict(test[features])
        observed = test[score_column].to_numpy()
        metrics_rows.append({
            "probe": probe,
            "intercept": float(model.intercept_),
            "true_count_coef": float(model.coef_[0]),
            "is_OR_coef": float(model.coef_[1]),
            "heldout_r2": float(r2_score(observed, predicted)),
            "heldout_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "heldout_mae": float(mean_absolute_error(observed, predicted)),
            "n_train_pairs": 400,
            "n_train_rows": 6400,
            "n_test_pairs": 100,
            "n_test_rows": 1600,
        })

        probe_predictions = pd.DataFrame({
            "probe": probe,
            "pair_id": test["entity_pair_id"].to_numpy(),
            "r1_row_index": test["r1_row_index"].to_numpy(),
            "topic": test["topic"].to_numpy(),
            "connective": test["connective"].str.upper().to_numpy(),
            "cell": test["cell"].to_numpy(),
            "ordering": test["ordering"].to_numpy(),
            "true_count": test["true_count"].to_numpy(dtype=int),
            "is_OR": test["is_OR"].to_numpy(dtype=int),
            "observed_score": observed,
            "predicted_score": predicted,
            "residual": observed - predicted,
        })
        prediction_frames.append(probe_predictions)

        for connective, is_or in (("AND", 0), ("OR", 1)):
            fitted_ff = float(model.predict(pd.DataFrame([[0, is_or]], columns=features))[0])
            fitted_mixed = float(model.predict(pd.DataFrame([[1, is_or]], columns=features))[0])
            fitted_tt = float(model.predict(pd.DataFrame([[2, is_or]], columns=features))[0])
            lower_spacing = fitted_mixed - fitted_ff
            upper_spacing = fitted_tt - fitted_mixed
            if not np.isclose(lower_spacing, upper_spacing, atol=1e-12, rtol=0):
                raise ValueError(f"Equal-spacing check failed for {probe} {connective}")
            spacing_rows.append({
                "probe": probe,
                "connective": connective,
                "predicted_FF": fitted_ff,
                "predicted_mixed": fitted_mixed,
                "predicted_TT": fitted_tt,
                "lower_spacing": lower_spacing,
                "upper_spacing": upper_spacing,
            })

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != 3200 or not np.isfinite(
        predictions[["observed_score", "predicted_score", "residual"]].to_numpy()
    ).all():
        raise ValueError("Unexpected or non-finite held-out predictions")

    residuals = (
        predictions.groupby(["probe", "connective", "cell"], sort=False)["residual"]
        .agg(n="size", mean_residual="mean", sd_residual="std")
        .reset_index()
    )
    rmse = (
        predictions.assign(squared_residual=predictions["residual"] ** 2)
        .groupby(["probe", "connective", "cell"], sort=False)["squared_residual"]
        .mean().pow(0.5).rename("rmse").reset_index()
    )
    residuals = residuals.merge(rmse, on=["probe", "connective", "cell"], validate="one_to_one")
    if len(residuals) != 16 or not residuals["n"].eq(200).all():
        raise ValueError("Expected 200 held-out rows in each probe/connective/cell residual group")

    # TF and FT necessarily share the same fitted value within a connective.
    fitted_uniques = predictions.groupby(["probe", "connective", "true_count"])["predicted_score"].nunique()
    if not fitted_uniques.eq(1).all():
        raise ValueError("Rows sharing probe/connective/true_count received different fitted values")

    MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    residuals.to_csv(RESIDUALS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print("METRICS\n" + metrics.to_string(index=False))
    print("\nRESIDUALS\n" + residuals.to_string(index=False))
    print("\nEQUAL SPACING\n" + pd.DataFrame(spacing_rows).to_string(index=False))


if __name__ == "__main__":
    main()
