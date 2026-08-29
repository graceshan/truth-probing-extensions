"""Build the validated Step-1 R1 row-level score table; fit no models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import entity_key_from_statements


TOPICS = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]
LAYER = 22


def raw_probe_scores(X, coef, intercept):
    """Frozen logistic decision function: X @ coef + intercept, unnormalized."""
    return np.asarray(X) @ coef + intercept


def assert_sidecars_equal(left, right, name):
    if not left.equals(right):
        raise AssertionError(f"{name}: extraction sidecar is not exactly row-aligned metadata")


def load_probe(path):
    saved = np.load(path)
    required = {"layer", "coef", "intercept", "classes"}
    if not required.issubset(saved.files):
        raise ValueError(f"probe missing fields: {sorted(required - set(saved.files))}")
    layer = int(saved["layer"])
    if layer != LAYER:
        raise ValueError(f"frozen probe says layer {layer}, expected {LAYER}")
    if not np.array_equal(saved["classes"], [0, 1]):
        raise ValueError(f"unexpected probe classes {saved['classes']}")
    coef = saved["coef"]
    if coef.shape[0] != 1:
        raise ValueError(f"expected one binary coefficient row, got {coef.shape}")
    return coef[0], float(saved["intercept"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", default="data/r1_quadruples.csv")
    parser.add_argument("--datasets-dir", default="data/tiu_datasets")
    parser.add_argument("--acts-dir", default="acts")
    parser.add_argument(
        "--probe", default="results/qwen_union_probe_gate/selected_probe.npz"
    )
    parser.add_argument("--output", default="results/r1/r1_score_table.csv")
    parser.add_argument(
        "--summary-output", default="results/r1/step1_descriptive_summary.json"
    )
    parser.add_argument(
        "--examples-output", default="results/r1/step1_random_examples.csv"
    )
    args = parser.parse_args()

    coef, intercept = load_probe(args.probe)
    acts_dir = Path(args.acts_dir)

    # Compound metadata must be the exact extraction sidecar and row order.
    r1 = pd.read_csv(args.r1)
    compound_meta = pd.read_csv(acts_dir / "r1_quadruples.csv")
    assert_sidecars_equal(r1, compound_meta, "R1 compound")
    compound_acts = np.load(acts_dir / "r1_quadruples.npy", mmap_mode="r")
    if compound_acts.shape != (8000, 28, len(coef)):
        raise ValueError(f"unexpected R1 compound activation shape {compound_acts.shape}")

    # Repair sidecar must exactly equal the extraction input manifest.
    repair_source = pd.read_csv("data/r1_missing_constituents.csv")
    repair_meta = pd.read_csv(acts_dir / "r1_missing_constituents.csv")
    assert_sidecars_equal(repair_source, repair_meta, "repair atomic")
    repair_acts = np.load(acts_dir / "r1_missing_constituents.npy", mmap_mode="r")
    if repair_acts.shape != (len(repair_meta), 28, len(coef)):
        raise ValueError(f"unexpected repair activation shape {repair_acts.shape}")
    if repair_meta.duplicated(["topic", "statement"]).any():
        raise AssertionError("repair metadata is not unique by (topic, exact statement)")

    original = {}
    duplicate_resolution = []
    for topic in TOPICS:
        meta = pd.read_csv(Path(args.datasets_dir) / f"{topic}.csv")
        acts = np.load(acts_dir / f"{topic}.npy", mmap_mode="r")
        if acts.shape != (len(meta), 28, len(coef)):
            raise ValueError(f"{topic}: incompatible original atomic shape {acts.shape}")
        for statement, group in meta.groupby("statement", sort=False):
            indices = group.index.to_numpy(dtype=int)
            labels = group["label"].unique()
            if len(labels) != 1:
                raise AssertionError(f"{topic}: duplicate exact text has conflicting labels")
            if len(indices) > 1:
                reference = np.asarray(acts[indices[0], LAYER, :])
                if not all(
                    np.array_equal(reference, np.asarray(acts[idx, LAYER, :]))
                    for idx in indices[1:]
                ):
                    raise AssertionError(f"{topic}: duplicate exact text has ambiguous activations")
                duplicate_resolution.append({
                    "topic": topic, "statement": statement,
                    "candidate_rows": indices.tolist(), "chosen_row": int(indices.min()),
                    "rule": "lowest row after label and layer-22 vector equality",
                })
            idx = int(indices.min())
            key = (topic, statement)
            original[key] = {
                "label": int(labels[0]),
                "entity": str(entity_key_from_statements(topic, [statement])[0]),
                "score": float(raw_probe_scores(acts[idx, LAYER, :], coef, intercept)),
                "source": "original",
            }

    repair = {}
    for idx, row in repair_meta.iterrows():
        key = (row["topic"], row["statement"])
        if key in original:
            raise AssertionError(f"repair record overlaps preferred original record: {key}")
        score = float(raw_probe_scores(repair_acts[idx, LAYER, :], coef, intercept))
        repair[key] = {
            "label": int(row["truth_label"]), "entity": str(row["entity"]),
            "score": score, "source": "repair",
        }

    combined = {**repair, **original}  # explicit original preference
    score_A, score_B, entity_pair_ids = [], [], []
    used_repair_keys = set()
    for row in r1.itertuples(index=False):
        resolved = []
        for side in ("A", "B"):
            statement = getattr(row, f"conjunct{side}")
            expected_label = int(getattr(row, f"label{side}"))
            key = (row.topic, statement)
            if key not in combined:
                raise AssertionError(f"missing exact atomic mapping: {key}")
            item = combined[key]
            if item["label"] != expected_label:
                raise AssertionError(f"atomic/R1 label disagreement: {key}")
            if item["source"] == "repair":
                used_repair_keys.add(key)
            resolved.append(item)
        score_A.append(resolved[0]["score"])
        score_B.append(resolved[1]["score"])
        canonical_entities = sorted([resolved[0]["entity"], resolved[1]["entity"]])
        entity_pair_ids.append(
            f"{row.topic}::{json.dumps(canonical_entities, ensure_ascii=False, separators=(',', ':'))}"
        )

    if used_repair_keys != set(repair):
        raise AssertionError(
            f"repair coverage mismatch: used {len(used_repair_keys)} of {len(repair)} records"
        )

    cell_from_labels = r1["labelA"].map({0: "F", 1: "T"}) + r1["labelB"].map({0: "F", 1: "T"})
    if not cell_from_labels.equals(r1["cell"]):
        raise AssertionError("R1 cell does not agree with labelA/labelB")
    global_truth = np.where(
        r1["connective"].eq("and"), r1["cell"].eq("TT"), ~r1["cell"].eq("FF")
    ).astype(int)
    if not r1["connective"].isin(["and", "or"]).all():
        raise AssertionError("unexpected connective in R1")

    score_compound = raw_probe_scores(compound_acts[:, LAYER, :], coef, intercept)
    output = pd.DataFrame({
        "topic": r1["topic"], "entity_pair_id": entity_pair_ids,
        "conjunctA": r1["conjunctA"], "conjunctB": r1["conjunctB"],
        "labelA": r1["labelA"], "labelB": r1["labelB"], "cell": r1["cell"],
        "connective": r1["connective"], "ordering": r1["ordering"],
        "global_truth": global_truth, "score_A": score_A, "score_B": score_B,
        "score_compound": score_compound,
    })

    score_cols = ["score_A", "score_B", "score_compound"]
    if len(output) != 8000 or len(score_compound) != 8000:
        raise AssertionError("R1 output/compound score count is not exactly 8000")
    if output[score_cols].isna().any().any() or not np.isfinite(output[score_cols]).all().all():
        raise AssertionError("R1 score table contains NaN or infinite values")
    pair_sizes = output.groupby("entity_pair_id").size()
    if len(pair_sizes) != 500 or not (pair_sizes == 16).all():
        raise AssertionError("canonical pair IDs do not identify 500 groups of 16 variants")

    atomic = {
        "true_A": float(output.loc[output.labelA == 1, "score_A"].mean()),
        "false_A": float(output.loc[output.labelA == 0, "score_A"].mean()),
        "true_B": float(output.loc[output.labelB == 1, "score_B"].mean()),
        "false_B": float(output.loc[output.labelB == 0, "score_B"].mean()),
    }
    global_means = {
        "globally_true": float(output.loc[output.global_truth == 1, "score_compound"].mean()),
        "globally_false": float(output.loc[output.global_truth == 0, "score_compound"].mean()),
    }
    cell_stats_df = (
        output.groupby(["connective", "cell"])["score_compound"]
        .agg(N="size", mean="mean", std="std").reset_index()
    )
    connective_order = {"and": 0, "or": 1}
    cell_order = {"TT": 0, "TF": 1, "FT": 2, "FF": 3}
    cell_stats_df = cell_stats_df.sort_values(
        ["connective", "cell"],
        key=lambda col: col.map(connective_order if col.name == "connective" else cell_order),
    ).reset_index(drop=True)
    cell_stats = cell_stats_df.to_dict(orient="records")
    examples = output.sample(n=10, random_state=0)[[
        "conjunctA", "conjunctB", "labelA", "labelB", "connective",
        "global_truth", "score_A", "score_B", "score_compound",
    ]]

    verification = {
        "r1_rows": len(output), "score_A_count": int(output.score_A.count()),
        "score_B_count": int(output.score_B.count()),
        "score_compound_count": int(output.score_compound.count()),
        "missing_mappings": 0, "ambiguous_unresolved_mappings": 0,
        "all_scores_finite": True, "global_truth_semantics_valid": True,
        "repair_records_available": len(repair),
        "repair_records_used": len(used_repair_keys),
        "compound_sidecar_exactly_aligned": True,
        "repair_sidecar_exactly_aligned": True,
        "canonical_entity_pairs": int(output.entity_pair_id.nunique()),
        "duplicate_original_resolution": duplicate_resolution,
        "probe_layer": LAYER, "score_scale": "raw frozen-probe decision function",
        "normalization": "none",
    }
    summary = {
        "verification": verification, "atomic_constituent_means": atomic,
        "compound_global_truth_means": global_means, "compound_cell_statistics": cell_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    examples_path = Path(args.examples_output)
    examples.to_csv(examples_path, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nRANDOM EXAMPLES")
    print(examples.to_string(index=False))
    print(f"\nsaved {output_path}")
    print(f"saved {summary_path}")
    print(f"saved {examples_path}")


if __name__ == "__main__":
    main()
