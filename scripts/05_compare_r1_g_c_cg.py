"""Prespecified held-out comparison of R1 Models G, C, and C+G only."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.probes import split_indices


INPUT = "results/r1/r1_score_table.csv"
OUTPUT_DIR = Path("results/r1/g_c_cg")
SEED = 0
TEST_SIZE = 0.2
TARGET = "score_compound"
MODEL_FEATURES = {
    "G": ["global_truth"],
    "C": ["score_A", "score_B", "is_or"],
    "C+G": ["score_A", "score_B", "is_or", "global_truth"],
}


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(INPUT)
    required = {
        "topic", "entity_pair_id", "connective", "global_truth",
        "score_A", "score_B", TARGET,
    }
    if len(df) != 8000 or not required.issubset(df.columns):
        raise ValueError("R1 score table has unexpected rows or columns")
    numeric = ["global_truth", "score_A", "score_B", TARGET]
    if df[numeric].isna().any().any() or not np.isfinite(df[numeric]).all().all():
        raise ValueError("R1 score table contains NaN or infinite inputs")
    if set(df["connective"]) != {"and", "or"}:
        raise ValueError("connective must contain exactly lowercase 'and' and 'or'")
    df["is_or"] = df["connective"].eq("or").astype(int)
    if not np.array_equal(df["is_or"].to_numpy(), (df["connective"] == "or").astype(int)):
        raise AssertionError("OR indicator coding is inconsistent")

    pair_sizes = df.groupby("entity_pair_id").size()
    if len(pair_sizes) != 500 or not (pair_sizes == 16).all():
        raise AssertionError("expected exactly 500 canonical pairs with 16 rows each")
    pair_topics = df.groupby("entity_pair_id")["topic"].nunique()
    if not (pair_topics == 1).all():
        raise AssertionError("an entity pair appears under multiple topics")

    pair_ids = np.array(sorted(pair_sizes.index), dtype=object)
    pair_train_idx, pair_test_idx = split_indices(
        len(pair_ids), test_size=TEST_SIZE, random_state=SEED, groups=pair_ids
    )
    train_pairs = set(pair_ids[pair_train_idx])
    test_pairs = set(pair_ids[pair_test_idx])
    if train_pairs & test_pairs:
        raise AssertionError("train/test entity-pair overlap is nonempty")
    if len(train_pairs) != 400 or len(test_pairs) != 100:
        raise AssertionError("expected 400 train and 100 test entity pairs")

    partitions = np.where(df["entity_pair_id"].isin(train_pairs), "train", "test")
    train_mask = partitions == "train"
    test_mask = partitions == "test"
    if train_mask.sum() != 6400 or test_mask.sum() != 1600:
        raise AssertionError("expected 6400 train and 1600 test rows")
    if not all(
        len(set(partitions[df.entity_pair_id == pair_id])) == 1 for pair_id in pair_ids
    ):
        raise AssertionError("at least one pair was split across partitions")

    pair_split = (
        df.assign(partition=partitions)
        .groupby("entity_pair_id", as_index=False)
        .agg(topic=("topic", "first"), partition=("partition", "first"), n_rows=("topic", "size"))
        .sort_values("entity_pair_id")
    )

    y = df[TARGET].to_numpy()
    overall_rows = []
    coefficient_rows = []
    connective_rows = []
    topic_rows = []
    predictions = df.loc[test_mask, [
        "topic", "entity_pair_id", "conjunctA", "conjunctB", "labelA", "labelB",
        "cell", "connective", "ordering", "global_truth", "score_A", "score_B", TARGET,
    ]].copy()
    predictions.insert(0, "r1_row_index", df.index[test_mask].astype(int))
    heldout_row_indices = predictions["r1_row_index"].to_numpy()

    fitted = {}
    for model_name, features in MODEL_FEATURES.items():
        X = df[features].to_numpy()
        model = LinearRegression(fit_intercept=True).fit(X[train_mask], y[train_mask])
        train_prediction = model.predict(X[train_mask])
        test_prediction = model.predict(X[test_mask])
        if len(test_prediction) != 1600 or not np.isfinite(test_prediction).all():
            raise AssertionError(f"{model_name}: invalid held-out predictions")
        fitted[model_name] = {"model": model, "test_prediction": test_prediction}

        test_metrics = metrics(y[test_mask], test_prediction)
        overall_rows.append({
            "model": model_name, "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs), "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
            "train_r2": float(r2_score(y[train_mask], train_prediction)),
            "test_r2": test_metrics["r2"], "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
        })
        coefficient_rows.append({
            "model": model_name, "term": "intercept", "coefficient": float(model.intercept_)
        })
        coefficient_rows.extend(
            {"model": model_name, "term": feature, "coefficient": float(coef)}
            for feature, coef in zip(features, model.coef_)
        )

        test_df = df.loc[test_mask]
        test_y = y[test_mask]
        for connective in ["and", "or"]:
            subset = test_df["connective"].eq(connective).to_numpy()
            subset_metrics = metrics(test_y[subset], test_prediction[subset])
            connective_rows.append({
                "model": model_name, "connective": connective,
                "n_pairs": int(test_df.loc[subset, "entity_pair_id"].nunique()),
                "n_rows": int(subset.sum()), "r2": subset_metrics["r2"],
                "rmse": subset_metrics["rmse"],
            })
        for topic in sorted(df["topic"].unique()):
            subset = test_df["topic"].eq(topic).to_numpy()
            subset_metrics = metrics(test_y[subset], test_prediction[subset])
            topic_rows.append({
                "model": model_name, "topic": topic,
                "n_pairs": int(test_df.loc[subset, "entity_pair_id"].nunique()),
                "n_rows": int(subset.sum()), "r2": subset_metrics["r2"],
                "rmse": subset_metrics["rmse"],
            })
        safe_name = model_name.lower().replace("+", "_plus_")
        predictions[f"prediction_{safe_name}"] = test_prediction
        predictions[f"residual_{safe_name}"] = y[test_mask] - test_prediction

    overall = pd.DataFrame(overall_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    per_connective = pd.DataFrame(connective_rows)
    per_topic = pd.DataFrame(topic_rows)
    by_model = overall.set_index("model")
    improvements = {
        "r2_G_to_C": float(by_model.loc["C", "test_r2"] - by_model.loc["G", "test_r2"]),
        "r2_C_to_C_plus_G": float(by_model.loc["C+G", "test_r2"] - by_model.loc["C", "test_r2"]),
        "rmse_reduction_G_to_C": float(by_model.loc["G", "test_rmse"] - by_model.loc["C", "test_rmse"]),
        "rmse_reduction_C_to_C_plus_G": float(by_model.loc["C", "test_rmse"] - by_model.loc["C+G", "test_rmse"]),
    }

    prediction_columns = [c for c in predictions if c.startswith("prediction_")]
    if len(prediction_columns) != 3 or len(predictions) != 1600:
        raise AssertionError("all models must have predictions on the same 1600 held-out rows")
    if not np.array_equal(predictions["r1_row_index"].to_numpy(), heldout_row_indices):
        raise AssertionError("held-out row identity changed across models")
    output_numeric = [
        overall.select_dtypes(include=[np.number]).to_numpy(),
        coefficients[["coefficient"]].to_numpy(),
        per_connective.select_dtypes(include=[np.number]).to_numpy(),
        per_topic.select_dtypes(include=[np.number]).to_numpy(),
        predictions[prediction_columns].to_numpy(),
        np.array(list(improvements.values())),
    ]
    if not all(np.isfinite(values).all() for values in output_numeric):
        raise AssertionError("saved metrics contain NaN or infinite values")

    verification = {
        "input": INPUT, "split_method": "GroupShuffleSplit on sorted canonical pair IDs",
        "seed": SEED, "test_size": TEST_SIZE, "canonical_pairs": len(pair_ids),
        "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
        "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
        "train_test_pair_intersection": [], "rows_per_pair": 16,
        "identical_heldout_rows_all_models": True,
        "heldout_metrics_use_test_pairs_only": True,
        "or_indicator_rule": "1 iff connective == 'or'; otherwise 0",
        "or_indicator_values": sorted(df["is_or"].unique().astype(int).tolist()),
        "all_inputs_outputs_finite": True,
        "models": {name: features for name, features in MODEL_FEATURES.items()},
        "fit": "ordinary sklearn LinearRegression with intercept",
        "improvement_convention": {
            "r2": "new minus old", "rmse": "old minus new (positive is improvement)"
        },
        "improvements": improvements,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_split.to_csv(OUTPUT_DIR / "pair_split.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    per_connective.to_csv(OUTPUT_DIR / "per_connective_metrics.csv", index=False)
    per_topic.to_csv(OUTPUT_DIR / "per_topic_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "heldout_predictions.csv", index=False)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    print("OVERALL")
    print(overall.to_string(index=False))
    print("\nCOEFFICIENTS")
    print(coefficients.to_string(index=False))
    print("\nPER CONNECTIVE")
    print(per_connective.to_string(index=False))
    print("\nPER TOPIC")
    print(per_topic.to_string(index=False))
    print("\nIMPROVEMENTS")
    print(json.dumps(improvements, indent=2))
    print(f"\nsaved outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
