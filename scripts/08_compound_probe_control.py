"""Train and audit the prespecified layer-22 compound-truth probe control."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


LAYER = 22
C = 0.1
SEED = 0
RESULTS = Path("results/r1")
PAIR_KEY = [
    "entity_pair_id", "conjunctA", "conjunctB", "ordering", "cell", "labelA", "labelB"
]


def classification_metrics(labels, scores):
    predictions = (scores >= 0).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mean_score_true": float(np.mean(scores[labels == 1])),
        "mean_score_false": float(np.mean(scores[labels == 0])),
    }


def shift_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "N": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "std": float(values.std(ddof=1)),
        "fraction_positive": float(np.mean(values > 0)),
    }


def main():
    table = pd.read_csv(RESULTS / "r1_score_table.csv")
    source = pd.read_csv("data/r1_quadruples.csv")
    sidecar = pd.read_csv("acts/r1_quadruples.csv")
    pair_split = pd.read_csv(RESULTS / "g_c_cg/pair_split.csv")
    prior_predictions = pd.read_csv(RESULTS / "g_c_cg/heldout_predictions.csv")
    atomic_matches = pd.read_csv(RESULTS / "or_audit_matched_differences.csv")
    acts = np.load("acts/r1_quadruples.npy", mmap_mode="r")

    if not source.equals(sidecar):
        raise AssertionError("compound source and activation sidecar differ")
    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not table[metadata_columns].equals(sidecar[metadata_columns]):
        raise AssertionError("score table is not row-aligned to compound activations")
    if len(table) != 8000 or acts.shape != (8000, 28, 3584):
        raise AssertionError("unexpected compound rows or activation shape")
    if table.entity_pair_id.nunique() != 500 or not (
        table.groupby("entity_pair_id").size() == 16
    ).all():
        raise AssertionError("expected 500 canonical pairs of 16 rows")
    if pair_split.partition.value_counts().to_dict() != {"train": 400, "test": 100}:
        raise AssertionError("saved split is not 400/100 pairs")
    train_pairs = set(pair_split.loc[pair_split.partition == "train", "entity_pair_id"])
    test_pairs = set(pair_split.loc[pair_split.partition == "test", "entity_pair_id"])
    if train_pairs & test_pairs or train_pairs | test_pairs != set(table.entity_pair_id):
        raise AssertionError("saved split has overlap or incomplete pair coverage")
    train_mask = table.entity_pair_id.isin(train_pairs).to_numpy()
    test_mask = table.entity_pair_id.isin(test_pairs).to_numpy()
    if train_mask.sum() != 6400 or test_mask.sum() != 1600:
        raise AssertionError("saved pair split does not produce 6400/1600 rows")
    test_indices = np.flatnonzero(test_mask)
    if not np.array_equal(test_indices, prior_predictions.r1_row_index.to_numpy()):
        raise AssertionError("held-out rows differ from prior G/C/C+G analysis")

    X = np.asarray(acts[:, LAYER, :])
    y = table.global_truth.to_numpy(dtype=int)
    if not np.isfinite(X).all() or not np.array_equal(np.unique(y), [0, 1]):
        raise AssertionError("non-finite activations or invalid truth labels")
    probe = LogisticRegression(
        penalty="l2", C=C, fit_intercept=True, solver="lbfgs",
        max_iter=2000, random_state=SEED,
    )
    probe.fit(X[train_mask], y[train_mask])
    if not np.array_equal(probe.classes_, [0, 1]):
        raise AssertionError("compound probe sign convention is unexpected")
    scores = probe.decision_function(X)
    direct_scores = X @ probe.coef_[0] + probe.intercept_[0]
    if not np.allclose(scores, direct_scores, rtol=0, atol=1e-12):
        raise AssertionError("decision_function differs from raw affine score")
    if not np.isfinite(scores).all():
        raise AssertionError("compound probe produced non-finite scores")

    table_with_scores = table.copy()
    table_with_scores.insert(0, "r1_row_index", np.arange(len(table)))
    table_with_scores["partition"] = np.where(train_mask, "train", "test")
    table_with_scores["compound_trained_score"] = scores
    table_with_scores["compound_trained_prediction"] = (scores >= 0).astype(int)

    test = table_with_scores.loc[test_mask]
    metric_rows = []
    overall = classification_metrics(y[test_mask], scores[test_mask])
    metric_rows.append({"group_type": "overall", "group": "overall", "N": len(test), **overall})
    for connective in ["and", "or"]:
        mask = test.connective.eq(connective).to_numpy()
        metric_rows.append({
            "group_type": "connective", "group": connective, "N": int(mask.sum()),
            **classification_metrics(test.global_truth.to_numpy()[mask], test.compound_trained_score.to_numpy()[mask]),
        })
    for topic in sorted(table.topic.unique()):
        mask = test.topic.eq(topic).to_numpy()
        metric_rows.append({
            "group_type": "topic", "group": topic, "N": int(mask.sum()),
            **classification_metrics(test.global_truth.to_numpy()[mask], test.compound_trained_score.to_numpy()[mask]),
        })
    classification_df = pd.DataFrame(metric_rows)

    cell_rows = []
    for scope, frame in [("heldout_test", test), ("full_descriptive", table_with_scores)]:
        for connective in ["and", "or"]:
            for cell in ["TT", "TF", "FT", "FF"]:
                values = frame.loc[
                    frame.connective.eq(connective) & frame.cell.eq(cell), "compound_trained_score"
                ]
                cell_rows.append({
                    "scope": scope, "connective": connective, "cell": cell,
                    "N": int(len(values)), "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                })
    cell_stats = pd.DataFrame(cell_rows)

    # Reuse and validate the exact matched-pair rows from the frozen-probe audit.
    if len(atomic_matches) != 4000 or not atomic_matches.template_only_connective_diff.all():
        raise AssertionError("prior exact AND/OR audit is unavailable or invalid")
    for row in atomic_matches.itertuples(index=False):
        and_source = table_with_scores.iloc[row.and_row_index]
        or_source = table_with_scores.iloc[row.or_row_index]
        if not all(getattr(and_source, key) == getattr(or_source, key) for key in PAIR_KEY):
            raise AssertionError("saved AND/OR row indices no longer share the exact match key")
        if and_source.connective != "and" or or_source.connective != "or":
            raise AssertionError("saved AND/OR row indices have wrong connectives")
    shifts = atomic_matches[[
        "topic", "entity_pair_id", "conjunctA", "conjunctB", "ordering", "cell",
        "labelA", "labelB", "and_row_index", "or_row_index", "and_statement", "or_statement",
        "score_and", "score_or", "delta_OR_AND",
    ]].rename(columns={
        "score_and": "atomic_union_score_and", "score_or": "atomic_union_score_or",
        "delta_OR_AND": "atomic_union_delta_OR_AND",
    })
    shifts["compound_trained_score_and"] = scores[shifts.and_row_index]
    shifts["compound_trained_score_or"] = scores[shifts.or_row_index]
    shifts["compound_trained_delta_OR_AND"] = (
        shifts.compound_trained_score_or - shifts.compound_trained_score_and
    )
    shifts["and_partition"] = table_with_scores.iloc[
        shifts.and_row_index.to_numpy()
    ].partition.to_numpy()
    shifts["or_partition"] = table_with_scores.iloc[
        shifts.or_row_index.to_numpy()
    ].partition.to_numpy()
    if not (shifts.and_partition == shifts.or_partition).all():
        raise AssertionError("matched AND/OR rows cross pair partitions")

    shift_summaries = {"overall": shift_summary(shifts.compound_trained_delta_OR_AND)}
    for column, values in [("cell", ["TT", "TF", "FT", "FF"]), ("ordering", ["AB", "BA"])]:
        shift_summaries[f"by_{column}"] = {
            value: shift_summary(shifts.loc[
                shifts[column] == value, "compound_trained_delta_OR_AND"
            ]) for value in values
        }

    shift_comparison = pd.DataFrame([
        {
            "cell": cell,
            "atomic_union_mean_delta_OR_AND": float(
                shifts.loc[shifts.cell == cell, "atomic_union_delta_OR_AND"].mean()
            ),
            "compound_trained_mean_delta_OR_AND": float(
                shifts.loc[shifts.cell == cell, "compound_trained_delta_OR_AND"].mean()
            ),
        }
        for cell in ["TT", "TF", "FT", "FF"]
    ])

    cell_comparison_rows = []
    for scope, frame in [("heldout_test", test), ("full_descriptive", table_with_scores)]:
        for connective in ["and", "or"]:
            for cell in ["TT", "TF", "FT", "FF"]:
                sub = frame[frame.connective.eq(connective) & frame.cell.eq(cell)]
                cell_comparison_rows.append({
                    "scope": scope, "connective": connective, "cell": cell, "N": len(sub),
                    "atomic_union_mean": float(sub.score_compound.mean()),
                    "atomic_union_std": float(sub.score_compound.std(ddof=1)),
                    "compound_trained_mean": float(sub.compound_trained_score.mean()),
                    "compound_trained_std": float(sub.compound_trained_score.std(ddof=1)),
                })
    cell_comparison = pd.DataFrame(cell_comparison_rows)

    cell_shift_means = shift_comparison.compound_trained_mean_delta_OR_AND.to_numpy()
    all_cell_shifts_positive = bool((cell_shift_means > 0).all())
    shift_range = float(cell_shift_means.max() - cell_shift_means.min())
    high_accuracy = bool(overall["accuracy"] >= 0.9 and overall["auroc"] >= 0.95)
    sign_reversal = bool(
        shift_comparison.atomic_union_mean_delta_OR_AND.mean() < 0
        and shift_comparison.compound_trained_mean_delta_OR_AND.mean() > 0
    )
    interpretation = {
        "1_heldout_truth_classification_well": "yes" if high_accuracy else "no",
        "1_basis": {"heldout_auroc": overall["auroc"], "heldout_accuracy": overall["accuracy"]},
        "2_or_shifts_upward": "yes; mean OR-minus-AND is positive overall and in all four cells" if all_cell_shifts_positive else "no",
        "3_shift_constancy": "strongly cell-dependent rather than roughly constant",
        "3_cell_mean_shift_range": shift_range,
        "4_qualitative_1_5B_comparison": "partial resemblance: high accuracy and an upward OR shift agree, but the 7B shift is substantially cell-dependent rather than approximately constant",
        "4_comparison_basis": "The earlier result is documented qualitatively as high accuracy plus an approximately constant connective offset; no numeric benchmark is available here.",
        "5_shift_sign_reverses_relative_to_atomic_union_probe": "yes" if sign_reversal else "no",
        "framing": "These are properties of two learned scalar probe directions, not claims that the model represents OR positively or negatively.",
    }

    verification = {
        "model": "Qwen/Qwen2.5-7B-Instruct", "layer": LAYER,
        "probe": {"type": "LogisticRegression", "penalty": "l2", "C": C,
                  "fit_intercept": True, "solver": "lbfgs", "max_iter": 2000,
                  "random_state": SEED, "n_iter": int(probe.n_iter_[0])},
        "canonical_pairs": 500, "train_pairs": 400, "test_pairs": 100,
        "train_rows": 6400, "test_rows": 1600, "pair_overlap": [],
        "identical_heldout_rows_to_g_c_cg": True,
        "source_sidecar_score_table_activation_alignment": True,
        "raw_affine_score_matches_decision_function": True,
        "normalization": "none", "all_scores_finite": True,
    }
    summary = {
        "verification": verification,
        "heldout_classification": classification_df.to_dict(orient="records"),
        "matched_shift_summaries": shift_summaries,
        "shift_comparison_by_cell": shift_comparison.to_dict(orient="records"),
        "interpretation_checks": interpretation,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    classification_df.to_csv(RESULTS / "compound_probe_control_classification_metrics.csv", index=False)
    cell_stats.to_csv(RESULTS / "compound_probe_control_cell_stats.csv", index=False)
    shifts.to_csv(RESULTS / "compound_probe_control_or_shifts.csv", index=False)
    shift_comparison.to_csv(RESULTS / "compound_probe_control_shift_comparison.csv", index=False)
    cell_comparison.to_csv(RESULTS / "compound_probe_control_cell_comparison.csv", index=False)
    table_with_scores.to_csv(RESULTS / "compound_probe_control_predictions.csv", index=False)
    np.savez_compressed(
        RESULTS / "compound_probe_control_probe.npz",
        coef=probe.coef_, intercept=probe.intercept_, classes=probe.classes_,
        layer=np.array(LAYER), C=np.array(C), n_iter=probe.n_iter_,
        solver=np.array("lbfgs"), random_state=np.array(SEED),
    )
    with (RESULTS / "compound_probe_control_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("HELD-OUT CLASSIFICATION")
    print(classification_df.to_string(index=False))
    print("\nCELL STATS")
    print(cell_stats.to_string(index=False))
    print("\nMATCHED SHIFT SUMMARIES")
    print(json.dumps(shift_summaries, indent=2))
    print("\nSHIFT COMPARISON")
    print(shift_comparison.to_string(index=False))
    print("\nCELL COMPARISON")
    print(cell_comparison.to_string(index=False))
    print("\nINTERPRETATION CHECKS")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
