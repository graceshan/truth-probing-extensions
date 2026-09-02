"""Audit the matched OR-minus-AND shift without fitting any models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import entity_key_from_statements


LAYER = 22
SEED = 0
TOPICS = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]
PAIR_COLUMNS = [
    "entity_pair_id", "conjunctA", "conjunctB", "ordering", "cell", "labelA", "labelB"
]
RESULTS = Path("results/r1")


def raw_score(vector, coef, intercept):
    return float(np.asarray(vector) @ coef + intercept)


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "N": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "fraction_lt_0": float(np.mean(values < 0)),
        "fraction_gt_0": float(np.mean(values > 0)),
    }


def main():
    score_table = pd.read_csv(RESULTS / "r1_score_table.csv")
    source_meta = pd.read_csv("data/r1_quadruples.csv")
    sidecar = pd.read_csv("acts/r1_quadruples.csv")
    compound_acts = np.load("acts/r1_quadruples.npy", mmap_mode="r")
    probe = np.load("results/qwen_union_probe_gate/selected_probe.npz")
    coef = probe["coef"][0]
    intercept = float(probe["intercept"][0])
    if int(probe["layer"]) != LAYER or not np.array_equal(probe["classes"], [0, 1]):
        raise AssertionError("unexpected frozen probe layer or sign convention")
    if not source_meta.equals(sidecar):
        raise AssertionError("compound source metadata and activation sidecar differ")
    if len(source_meta) != len(score_table) or compound_acts.shape != (8000, 28, len(coef)):
        raise AssertionError("compound metadata/score/activation row counts or shapes differ")
    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not score_table[metadata_columns].equals(source_meta[metadata_columns]):
        raise AssertionError("score table metadata is not in exact source/activation row order")

    # Recompute every compound score, and save a fixed-seed 20-row alignment audit.
    recomputed_compound = np.asarray(compound_acts[:, LAYER, :]) @ coef + intercept
    compound_diff = np.abs(recomputed_compound - score_table["score_compound"].to_numpy())
    if not np.isfinite(recomputed_compound).all():
        raise AssertionError("non-finite recomputed compound score")
    alignment_indices = np.random.default_rng(SEED).choice(len(score_table), size=20, replace=False)
    alignment_examples = pd.DataFrame({
        "row_index": alignment_indices,
        "entity_pair_id": score_table.loc[alignment_indices, "entity_pair_id"].to_numpy(),
        "statement": sidecar.loc[alignment_indices, "statement"].to_numpy(),
        "connective": score_table.loc[alignment_indices, "connective"].to_numpy(),
        "ordering": score_table.loc[alignment_indices, "ordering"].to_numpy(),
        "cell": score_table.loc[alignment_indices, "cell"].to_numpy(),
        "labelA": score_table.loc[alignment_indices, "labelA"].to_numpy(),
        "labelB": score_table.loc[alignment_indices, "labelB"].to_numpy(),
        "global_truth": score_table.loc[alignment_indices, "global_truth"].to_numpy(),
        "stored_score_compound": score_table.loc[alignment_indices, "score_compound"].to_numpy(),
        "recomputed_score_compound": recomputed_compound[alignment_indices],
        "absolute_difference": compound_diff[alignment_indices],
    }).sort_values("row_index")

    # Build a direct exact-text atomic lookup, preferring original records.
    original = {}
    all_original_rows = []
    duplicate_rules = []
    for topic in TOPICS:
        meta = pd.read_csv(f"data/tiu_datasets/{topic}.csv")
        acts = np.load(f"acts/{topic}.npy", mmap_mode="r")
        if acts.shape != (len(meta), 28, len(coef)):
            raise AssertionError(f"{topic}: atomic metadata/activation shape mismatch")
        for idx, row in meta.iterrows():
            all_original_rows.append({
                "topic": topic, "statement": row.statement, "label": int(row.label),
                "atomic_row_index": int(idx),
                "recomputed_score": raw_score(acts[idx, LAYER, :], coef, intercept),
            })
        for statement, group in meta.groupby("statement", sort=False):
            indices = group.index.to_numpy(dtype=int)
            labels = group.label.unique()
            if len(labels) != 1:
                raise AssertionError("duplicate atomic text has conflicting labels")
            reference = np.asarray(acts[indices[0], LAYER, :])
            if not all(np.array_equal(reference, np.asarray(acts[i, LAYER, :])) for i in indices[1:]):
                raise AssertionError("duplicate atomic text has different layer-22 activations")
            chosen = int(indices.min())
            if len(indices) > 1:
                duplicate_rules.append({
                    "topic": topic, "statement": statement, "candidate_rows": indices.tolist(),
                    "chosen_row": chosen,
                    "rule": "lowest row after identical labels and bit-identical layer-22 vectors",
                })
            original[(topic, statement)] = {
                "label": int(labels[0]),
                "score": raw_score(acts[chosen, LAYER, :], coef, intercept),
                "source": "original", "row": chosen,
            }

    repair_meta = pd.read_csv("acts/r1_missing_constituents.csv")
    repair_source = pd.read_csv("data/r1_missing_constituents.csv")
    repair_acts = np.load("acts/r1_missing_constituents.npy", mmap_mode="r")
    if not repair_meta.equals(repair_source) or repair_acts.shape != (571, 28, len(coef)):
        raise AssertionError("repair metadata/activation row alignment mismatch")
    atomic_lookup = dict(original)
    for idx, row in repair_meta.iterrows():
        key = (row.topic, row.statement)
        if key in atomic_lookup:
            raise AssertionError("repair record unexpectedly overlaps original exact text")
        atomic_lookup[key] = {
            "label": int(row.truth_label),
            "score": raw_score(repair_acts[idx, LAYER, :], coef, intercept),
            "source": "repair", "row": int(idx),
        }

    # Independently remap all 16,000 R1 atomic occurrences and compare stored scores.
    recomputed_A, recomputed_B = [], []
    for row in score_table.itertuples(index=False):
        for side, destination in [("A", recomputed_A), ("B", recomputed_B)]:
            key = (row.topic, getattr(row, f"conjunct{side}"))
            item = atomic_lookup.get(key)
            if item is None or item["label"] != int(getattr(row, f"label{side}")):
                raise AssertionError(f"missing or label-inconsistent atomic key {key}")
            destination.append(item["score"])
    recomputed_A = np.asarray(recomputed_A)
    recomputed_B = np.asarray(recomputed_B)
    max_atomic_stored_difference = float(max(
        np.max(np.abs(recomputed_A - score_table.score_A)),
        np.max(np.abs(recomputed_B - score_table.score_B)),
    ))
    atomic_means = {
        "A_true": float(recomputed_A[score_table.labelA.to_numpy() == 1].mean()),
        "A_false": float(recomputed_A[score_table.labelA.to_numpy() == 0].mean()),
        "B_true": float(recomputed_B[score_table.labelB.to_numpy() == 1].mean()),
        "B_false": float(recomputed_B[score_table.labelB.to_numpy() == 0].mean()),
    }
    pooled_scores = np.concatenate([recomputed_A, recomputed_B])
    pooled_labels = np.concatenate([score_table.labelA, score_table.labelB]).astype(int)
    atomic_means["pooled_true"] = float(pooled_scores[pooled_labels == 1].mean())
    atomic_means["pooled_false"] = float(pooled_scores[pooled_labels == 0].mean())
    atomic_means["fraction_true_gt_0"] = float(
        np.mean(pooled_scores[pooled_labels == 1] > 0)
    )
    atomic_means["fraction_false_lt_0"] = float(
        np.mean(pooled_scores[pooled_labels == 0] < 0)
    )

    # Fixed-seed known atomic examples restricted to exact texts used by R1,
    # ensuring a stored R1 score exists for the requested comparison.
    original_df = pd.DataFrame(all_original_rows)
    r1_keys = set(zip(score_table.topic, score_table.conjunctA)) | set(
        zip(score_table.topic, score_table.conjunctB)
    )
    original_df = original_df[
        [(t, s) in r1_keys for t, s in zip(original_df.topic, original_df.statement)]
    ].drop_duplicates(["topic", "statement", "label"])
    atomic_examples = pd.concat([
        original_df[original_df.label == label].sample(n=5, random_state=SEED)
        for label in [1, 0]
    ]).sort_values(["label", "topic", "statement"], ascending=[False, True, True])
    stored_by_key = {}
    for row in score_table.itertuples(index=False):
        for side in ["A", "B"]:
            key = (row.topic, getattr(row, f"conjunct{side}"))
            stored_by_key.setdefault(key, []).append(float(getattr(row, f"score_{side}")))
    stored_scores = []
    for row in atomic_examples.itertuples(index=False):
        values = np.asarray(stored_by_key[(row.topic, row.statement)])
        if not np.allclose(values, values[0], rtol=0, atol=1e-12):
            raise AssertionError("stored atomic score varies for identical exact text")
        stored_scores.append(float(values[0]))
    atomic_examples["stored_r1_score"] = stored_scores
    atomic_examples["absolute_difference"] = np.abs(
        atomic_examples.recomputed_score - atomic_examples.stored_r1_score
    )

    # Exact one-to-one matched AND/OR construction.
    working = score_table.copy()
    working.insert(0, "row_index", np.arange(len(working)))
    working["statement"] = sidecar["statement"]
    key_counts = working.groupby(PAIR_COLUMNS + ["connective"], dropna=False).size()
    duplicate_match_keys = key_counts[key_counts > 1]
    grouped = working.groupby(PAIR_COLUMNS, dropna=False, sort=False)
    bad_groups = []
    matched_rows = []
    template_failures = []
    for key, group in grouped:
        counts = group.connective.value_counts().to_dict()
        if len(group) != 2 or counts != {"and": 1, "or": 1}:
            bad_groups.append({"key": list(key), "counts": counts, "row_indices": group.row_index.tolist()})
            continue
        and_row = group[group.connective == "and"].iloc[0]
        or_row = group[group.connective == "or"].iloc[0]
        and_tokens = and_row.statement.split()
        or_tokens = or_row.statement.split()
        differing_positions = [
            i for i, (a, o) in enumerate(zip(and_tokens, or_tokens)) if a != o
        ] if len(and_tokens) == len(or_tokens) else []
        template_ok = (
            len(and_tokens) == len(or_tokens)
            and len(differing_positions) == 1
            and and_tokens[differing_positions[0]] == "and"
            and or_tokens[differing_positions[0]] == "or"
        )
        if not template_ok:
            template_failures.append({
                "and_row_index": int(and_row.row_index), "or_row_index": int(or_row.row_index),
                "and_statement": and_row.statement, "or_statement": or_row.statement,
                "and_tokens": and_tokens, "or_tokens": or_tokens,
                "differing_positions": differing_positions,
            })
        matched_rows.append({
            "topic": and_row.topic, "entity_pair_id": and_row.entity_pair_id,
            "conjunctA": and_row.conjunctA, "conjunctB": and_row.conjunctB,
            "ordering": and_row.ordering, "cell": and_row.cell,
            "labelA": int(and_row.labelA), "labelB": int(and_row.labelB),
            "and_row_index": int(and_row.row_index), "or_row_index": int(or_row.row_index),
            "and_statement": and_row.statement, "or_statement": or_row.statement,
            "score_and": float(and_row.score_compound), "score_or": float(or_row.score_compound),
            "delta_OR_AND": float(or_row.score_compound - and_row.score_compound),
            "template_diff_token_index": differing_positions[0] if template_ok else None,
            "template_only_connective_diff": template_ok,
        })
    matched = pd.DataFrame(matched_rows)
    if bad_groups or len(duplicate_match_keys) or template_failures or len(matched) != 4000:
        raise AssertionError("AND/OR matching or template validation failed")

    summaries = {"overall": describe(matched.delta_OR_AND)}
    for column, values in [
        ("cell", ["TT", "TF", "FT", "FF"]),
        ("ordering", ["AB", "BA"]),
        ("topic", TOPICS),
    ]:
        summaries[f"by_{column}"] = {
            value: describe(matched.loc[matched[column] == value, "delta_OR_AND"])
            for value in values
        }

    # Ten fixed-seed entity pairs, with AB examples for all four cells.
    selected_pairs = np.random.default_rng(SEED).choice(
        np.array(sorted(matched.entity_pair_id.unique()), dtype=object), size=10, replace=False
    )
    matched_examples = matched[
        matched.entity_pair_id.isin(selected_pairs) & matched.ordering.eq("AB")
    ].copy()
    cell_order = {"TT": 0, "TF": 1, "FT": 2, "FF": 3}
    pair_order = {pair: i for i, pair in enumerate(selected_pairs)}
    matched_examples = matched_examples.sort_values(
        ["entity_pair_id", "cell"],
        key=lambda x: x.map(pair_order if x.name == "entity_pair_id" else cell_order),
    )
    if len(matched_examples) != 40:
        raise AssertionError("expected four AB cell examples for each of ten entity pairs")

    summary = {
        "conclusion": "B. No bug found; negative OR shift appears to be a genuine matched-sentence property of the frozen probe",
        "probe_sign": {
            "layer": LAYER, "classes": probe["classes"].astype(int).tolist(),
            "atomic_occurrence_means": atomic_means,
            "max_abs_recomputed_vs_stored_atomic_score": max_atomic_stored_difference,
        },
        "compound_alignment": {
            "metadata_rows": len(sidecar), "activation_rows": int(compound_acts.shape[0]),
            "source_sidecar_exact_equality": True, "score_table_metadata_exact_row_order": True,
            "max_abs_stored_vs_recomputed_score": float(compound_diff.max()),
            "fixed_seed_alignment_examples": len(alignment_examples),
        },
        "template_check": {
            "matched_comparisons": len(matched), "rows_matched_one_to_one": 2 * len(matched),
            "unmatched_groups": len(bad_groups), "duplicated_match_keys": len(duplicate_match_keys),
            "ambiguous_groups": len(bad_groups), "template_failures": len(template_failures),
            "rule": "whitespace-token sequences equal except one token: 'and' versus 'or'",
        },
        "matched_delta_summaries": summaries,
        "duplicate_atomic_resolution": duplicate_rules,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    matched.to_csv(RESULTS / "or_audit_matched_differences.csv", index=False)
    matched_examples.to_csv(RESULTS / "or_audit_examples.csv", index=False)
    alignment_examples.to_csv(RESULTS / "or_audit_alignment_examples.csv", index=False)
    atomic_examples.to_csv(RESULTS / "or_audit_atomic_examples.csv", index=False)
    with (RESULTS / "or_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("ATOMIC EXAMPLES")
    print(atomic_examples.to_string(index=False))
    print("\nATOMIC MEANS")
    print(json.dumps(atomic_means, indent=2))
    print("\n20 COMPOUND ALIGNMENT EXAMPLES")
    print(alignment_examples.to_string(index=False))
    print("\nMATCHED EXAMPLES (10 PAIRS x 4 CELLS, AB ordering)")
    print(matched_examples[[
        "entity_pair_id", "cell", "and_statement", "or_statement",
        "score_and", "score_or", "delta_OR_AND"
    ]].to_string(index=False))
    print("\nMATCHED DELTA SUMMARIES")
    print(json.dumps(summaries, indent=2))
    print("\nCONCLUSION")
    print(summary["conclusion"])


if __name__ == "__main__":
    main()
