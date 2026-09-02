"""Compare frozen atomic-union and compound-trained probes on the same R1 test rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


LAYER = 22
RESULTS = Path("results/r1")
PROBES = {
    "atomic_union_to_compounds": {
        "label": "Atomic union → compounds",
        "path": Path("results/qwen_union_probe_gate/selected_probe.npz"),
        "stored_column": "score_compound",
    },
    "compound_trained_to_compounds": {
        "label": "Compound-trained → compounds",
        "path": RESULTS / "compound_probe_control_probe.npz",
        "stored_column": "compound_trained_score",
    },
}


def metrics(labels, scores):
    predicted = (scores >= 0).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "average_precision": float(average_precision_score(labels, scores)),
        "mean_score_true": float(scores[labels == 1].mean()),
        "mean_score_false": float(scores[labels == 0].mean()),
    }


def main():
    table = pd.read_csv(RESULTS / "r1_score_table.csv")
    split = pd.read_csv(RESULTS / "g_c_cg/pair_split.csv")
    prior_test = pd.read_csv(RESULTS / "g_c_cg/heldout_predictions.csv")
    compound_stored = pd.read_csv(RESULTS / "compound_probe_control_predictions.csv")
    source = pd.read_csv("data/r1_quadruples.csv")
    sidecar = pd.read_csv("acts/r1_quadruples.csv")
    acts = np.load("acts/r1_quadruples.npy", mmap_mode="r")

    if not source.equals(sidecar) or len(table) != 8000 or acts.shape != (8000, 28, 3584):
        raise AssertionError("R1 source/sidecar/table/activation alignment failed")
    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not table[metadata_columns].equals(sidecar[metadata_columns]):
        raise AssertionError("score table metadata is not activation-row aligned")
    if split.partition.value_counts().to_dict() != {"train": 400, "test": 100}:
        raise AssertionError("saved split is not the registered 400/100 split")
    train_pairs = set(split.loc[split.partition == "train", "entity_pair_id"])
    test_pairs = set(split.loc[split.partition == "test", "entity_pair_id"])
    if train_pairs & test_pairs or len(train_pairs) != 400 or len(test_pairs) != 100:
        raise AssertionError("pair split overlap or count failure")
    test_mask = table.entity_pair_id.isin(test_pairs).to_numpy()
    test_indices = np.flatnonzero(test_mask)
    if len(test_indices) != 1600 or table.loc[test_mask, "entity_pair_id"].nunique() != 100:
        raise AssertionError("held-out set must be 100 pairs / 1600 rows")
    if not (table.loc[test_mask].groupby("entity_pair_id").size() == 16).all():
        raise AssertionError("held-out pairs do not contain all 16 variants")
    if not np.array_equal(test_indices, prior_test.r1_row_index.to_numpy()):
        raise AssertionError("test row order differs from prior G/C/C+G evaluation")
    if not np.array_equal(compound_stored.r1_row_index.to_numpy(), np.arange(8000)):
        raise AssertionError("stored compound-probe predictions lost source row order")

    X_test = np.asarray(acts[test_indices, LAYER, :])
    test = table.loc[test_mask].copy()
    test.insert(0, "r1_row_index", test_indices)
    labels = test.global_truth.to_numpy(dtype=int)
    recomputation_differences = {}
    for probe_name, config in PROBES.items():
        saved = np.load(config["path"])
        if int(saved["layer"]) != LAYER or not np.array_equal(saved["classes"], [0, 1]):
            raise AssertionError(f"{probe_name}: wrong layer or stored orientation")
        score = X_test @ saved["coef"][0] + saved["intercept"][0]
        if not np.isfinite(score).all():
            raise AssertionError(f"{probe_name}: non-finite recomputed score")
        if probe_name == "atomic_union_to_compounds":
            stored_score = table.loc[test_mask, config["stored_column"]].to_numpy()
        else:
            stored_score = compound_stored.loc[test_indices, config["stored_column"]].to_numpy()
        difference = np.abs(score - stored_score)
        if not np.allclose(score, stored_score, rtol=0, atol=1e-12):
            raise AssertionError(f"{probe_name}: recomputed score differs from stored score")
        recomputation_differences[probe_name] = float(difference.max())
        test[f"{probe_name}_score"] = score
        test[f"{probe_name}_prediction_at_0"] = (score >= 0).astype(int)

    overall_rows = []
    connective_rows = []
    topic_rows = []
    cell_rows = []
    for probe_name, config in PROBES.items():
        score_column = f"{probe_name}_score"
        score = test[score_column].to_numpy()
        overall_rows.append({
            "probe": probe_name, "probe_label": config["label"], "N": len(test),
            **metrics(labels, score),
        })
        for connective in ["and", "or"]:
            mask = test.connective.eq(connective).to_numpy()
            connective_rows.append({
                "probe": probe_name, "probe_label": config["label"],
                "connective": connective, "N": int(mask.sum()),
                **metrics(labels[mask], score[mask]),
            })
        for topic in ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]:
            mask = test.topic.eq(topic).to_numpy()
            topic_metrics = metrics(labels[mask], score[mask])
            topic_rows.append({
                "probe": probe_name, "probe_label": config["label"], "topic": topic,
                "test_entity_pairs": int(test.loc[mask, "entity_pair_id"].nunique()),
                "N": int(mask.sum()), "auroc": topic_metrics["auroc"],
                "accuracy": topic_metrics["accuracy"],
                "balanced_accuracy": topic_metrics["balanced_accuracy"],
            })
        for connective in ["and", "or"]:
            for cell in ["TT", "TF", "FT", "FF"]:
                mask = test.connective.eq(connective) & test.cell.eq(cell)
                values = test.loc[mask, score_column].to_numpy()
                cell_rows.append({
                    "probe": probe_name, "probe_label": config["label"],
                    "connective": connective, "cell": cell, "N": int(len(values)),
                    "mean_score": float(values.mean()), "std_score": float(values.std(ddof=1)),
                    "fraction_classified_positive": float(np.mean(values >= 0)),
                })

    overall = pd.DataFrame(overall_rows)
    by_connective = pd.DataFrame(connective_rows)
    by_topic = pd.DataFrame(topic_rows)
    by_cell = pd.DataFrame(cell_rows)
    primary_rows = []
    for row in overall.itertuples(index=False):
        conn = by_connective[by_connective.probe == row.probe].set_index("connective")
        primary_rows.append({
            "probe": row.probe_label, "overall_auroc": row.auroc,
            "accuracy": row.accuracy, "balanced_accuracy": row.balanced_accuracy,
            "pr_auc_average_precision": row.average_precision,
            "and_auroc": float(conn.loc["and", "auroc"]),
            "or_auroc": float(conn.loc["or", "auroc"]),
        })
    primary = pd.DataFrame(primary_rows)
    overall_indexed = overall.set_index("probe")
    connective_indexed = by_connective.set_index(["probe", "connective"])
    differences = {
        "overall": {
            "auroc_compound_minus_atomic": float(
                overall_indexed.loc["compound_trained_to_compounds", "auroc"]
                - overall_indexed.loc["atomic_union_to_compounds", "auroc"]
            ),
            "accuracy_compound_minus_atomic": float(
                overall_indexed.loc["compound_trained_to_compounds", "accuracy"]
                - overall_indexed.loc["atomic_union_to_compounds", "accuracy"]
            ),
        },
        "and": {
            "auroc_compound_minus_atomic": float(
                connective_indexed.loc[("compound_trained_to_compounds", "and"), "auroc"]
                - connective_indexed.loc[("atomic_union_to_compounds", "and"), "auroc"]
            ),
            "accuracy_compound_minus_atomic": float(
                connective_indexed.loc[("compound_trained_to_compounds", "and"), "accuracy"]
                - connective_indexed.loc[("atomic_union_to_compounds", "and"), "accuracy"]
            ),
        },
        "or": {
            "auroc_compound_minus_atomic": float(
                connective_indexed.loc[("compound_trained_to_compounds", "or"), "auroc"]
                - connective_indexed.loc[("atomic_union_to_compounds", "or"), "auroc"]
            ),
            "accuracy_compound_minus_atomic": float(
                connective_indexed.loc[("compound_trained_to_compounds", "or"), "accuracy"]
                - connective_indexed.loc[("atomic_union_to_compounds", "or"), "accuracy"]
            ),
        },
    }

    atomic_overall = overall_indexed.loc["atomic_union_to_compounds"]
    atomic_and = connective_indexed.loc[("atomic_union_to_compounds", "and")]
    atomic_or = connective_indexed.loc[("atomic_union_to_compounds", "or")]
    interpretation = {
        "atomic_union_overall": (
            f"Atomic-union transfer has AUROC {atomic_overall.auroc:.4f} and threshold-0 "
            f"accuracy {atomic_overall.accuracy:.4f}."
        ),
        "threshold_vs_ranking": (
            "Threshold-0 accuracy and AUROC are reported separately; no calibration, "
            "threshold tuning, rescaling, or sign flipping was applied."
        ),
        "connective_difference": (
            f"Atomic-union AND AUROC is {atomic_and.auroc:.4f}, versus OR AUROC "
            f"{atomic_or.auroc:.4f}; AND/OR behavior is therefore reported separately."
        ),
    }
    verification = {
        "layer": LAYER, "canonical_pairs_total": 500, "train_pairs": 400,
        "test_pairs": 100, "test_rows": 1600, "rows_per_test_pair": 16,
        "train_test_pair_overlap": [], "identical_ordered_test_rows_for_both_probes": True,
        "identical_test_rows_to_g_c_cg": True, "normalization": "none",
        "threshold": 0.0, "score_signs_flipped": False, "recalibrated": False,
        "max_abs_recomputed_vs_stored_score": recomputation_differences,
        "all_scores_and_outputs_finite": True,
    }
    summary = {
        "verification": verification,
        "overall": overall.to_dict(orient="records"),
        "by_connective": by_connective.to_dict(orient="records"),
        "differences": differences,
        "interpretation": interpretation,
    }

    overall.to_csv(RESULTS / "probe_transfer_comparison_overall.csv", index=False)
    primary.to_csv(RESULTS / "probe_transfer_comparison_primary.csv", index=False)
    by_connective.to_csv(RESULTS / "probe_transfer_comparison_by_connective.csv", index=False)
    by_topic.to_csv(RESULTS / "probe_transfer_comparison_by_topic.csv", index=False)
    by_cell.to_csv(RESULTS / "probe_transfer_comparison_by_cell.csv", index=False)
    test.to_csv(RESULTS / "probe_transfer_comparison_predictions.csv", index=False)
    with (RESULTS / "probe_transfer_comparison_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("PRIMARY")
    print(primary.to_string(index=False))
    print("\nOVERALL")
    print(overall.to_string(index=False))
    print("\nBY CONNECTIVE")
    print(by_connective.to_string(index=False))
    print("\nBY TOPIC")
    print(by_topic.to_string(index=False))
    print("\nBY CELL")
    print(by_cell.to_string(index=False))
    print("\nDIFFERENCES")
    print(json.dumps(differences, indent=2))
    print("\nINTERPRETATION")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
