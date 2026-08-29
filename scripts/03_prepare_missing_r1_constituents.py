"""Audit and save exact R1 constituent texts missing atomic activations.

This is local preparation only: it reads metadata and existing checkpoints,
does not load a model, extract activations, score examples, or fit analyses.
"""

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
ABSENCE_REASON = "generated_wrong_object_text_absent_from_source_atomic_metadata"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", default="data/r1_quadruples.csv")
    parser.add_argument("--datasets-dir", default="data/tiu_datasets")
    parser.add_argument("--acts-dir", default="acts")
    parser.add_argument("--output", default="data/r1_missing_constituents.csv")
    parser.add_argument(
        "--audit-output", default="results/r1_missing_constituents_audit.json"
    )
    args = parser.parse_args()

    r1 = pd.read_csv(args.r1)
    if len(r1) != 8000:
        raise ValueError(f"expected 8000 R1 rows, found {len(r1)}")

    existing = {}
    duplicate_collapses = []
    for topic in TOPICS:
        meta = pd.read_csv(Path(args.datasets_dir) / f"{topic}.csv")
        acts = np.load(Path(args.acts_dir) / f"{topic}.npy", mmap_mode="r")
        if len(meta) != acts.shape[0] or acts.shape[1] <= 22:
            raise ValueError(f"{topic}: incompatible atomic metadata/activation shape")
        for statement, group in meta.groupby("statement", sort=False):
            indices = group.index.to_numpy(dtype=int)
            labels = group["label"].unique()
            if len(labels) != 1:
                raise AssertionError(f"{topic}: duplicate text has conflicting labels: {statement}")
            if len(indices) > 1:
                reference = np.asarray(acts[indices[0], 22, :])
                if not all(
                    np.array_equal(reference, np.asarray(acts[idx, 22, :]))
                    for idx in indices[1:]
                ):
                    raise AssertionError(
                        f"{topic}: duplicate text has different layer-22 activations: {statement}"
                    )
                duplicate_collapses.append({
                    "topic": topic,
                    "statement": statement,
                    "atomic_row_indices": indices.tolist(),
                    "truth_label": int(labels[0]),
                    "layer_22_activations_identical": True,
                    "collapse_rule": "choose lowest atomic row index",
                    "chosen_atomic_row_index": int(indices.min()),
                })
            existing[(topic, statement)] = {
                "label": int(labels[0]), "atomic_row_index": int(indices.min())
            }

    occurrences = []
    for r1_row, row in r1.iterrows():
        for side in ("A", "B"):
            occurrences.append({
                "r1_row_index": int(r1_row),
                "side": side,
                "topic": row["topic"],
                "statement": row[f"conjunct{side}"],
                "truth_label": int(row[f"label{side}"]),
            })
    occurrences = pd.DataFrame(occurrences)
    occurrences["exists"] = [
        (r.topic, r.statement) in existing for r in occurrences.itertuples()
    ]

    missing_occurrences = occurrences.loc[~occurrences["exists"]].copy()
    records = []
    for (topic, statement), group in missing_occurrences.groupby(
        ["topic", "statement"], sort=True
    ):
        labels = group["truth_label"].unique()
        if len(labels) != 1:
            raise AssertionError(f"missing text has conflicting R1 labels: {topic}: {statement}")
        refs = [
            f"{int(row.r1_row_index)}:{row.side}"
            for row in group.sort_values(["r1_row_index", "side"]).itertuples()
        ]
        row_indices = sorted(group["r1_row_index"].astype(int).unique().tolist())
        entity = entity_key_from_statements(topic, [statement])[0]
        records.append({
            "statement": statement,
            "topic": topic,
            "truth_label": int(labels[0]),
            "entity": entity,
            "dedup_identity": f"{topic}::{statement}",
            "source_r1_row_indices": json.dumps(row_indices),
            "source_r1_conjunct_refs": json.dumps(refs),
            "n_r1_rows": len(row_indices),
            "n_r1_conjunct_occurrences": len(group),
            "absence_reason": ABSENCE_REASON,
        })
    missing = pd.DataFrame(records).sort_values(["topic", "statement"]).reset_index(drop=True)

    if missing.duplicated(["topic", "statement"]).any():
        raise AssertionError("missing output is not unique by (topic, exact statement)")
    if not (missing["truth_label"] == 0).all():
        raise AssertionError("expected missing generated wrong-object statements to be false")

    missing_keys = set(zip(missing["topic"], missing["statement"]))
    resolution_counts = []
    for row in occurrences.itertuples():
        key = (row.topic, row.statement)
        resolution_counts.append(int(key in existing) + int(key in missing_keys))
    if set(resolution_counts) != {1}:
        raise AssertionError("some R1 conjunct does not resolve by exactly one route")

    resolved = occurrences.assign(resolution_count=resolution_counts)
    per_row = resolved.groupby("r1_row_index").agg(
        n_conjuncts=("side", "size"),
        n_deterministic_mappings=("resolution_count", "sum"),
    )
    if not ((per_row.n_conjuncts == 2) & (per_row.n_deterministic_mappings == 2)).all():
        raise AssertionError("not all 8000 R1 rows have two deterministic mappings")

    # These are the three known duplicated Spanish true statements. Fail if
    # that changes, since the deterministic collapse rule is evidence-based.
    if len(duplicate_collapses) != 3 or {d["topic"] for d in duplicate_collapses} != {"sp_en_trans"}:
        raise AssertionError(
            f"expected exactly three duplicated Spanish texts, found {len(duplicate_collapses)}"
        )

    affected_rows = int(missing_occurrences["r1_row_index"].nunique())
    audit = {
        "deduplication_identity": ["topic", "exact statement"],
        "missing_unique_statements": int(len(missing)),
        "missing_counts_by_topic": {
            topic: int(count) for topic, count in missing.groupby("topic").size().items()
        },
        "missing_counts_by_truth_label": {
            str(int(label)): int(count)
            for label, count in missing.groupby("truth_label").size().items()
        },
        "missing_conjunct_occurrences": int(len(missing_occurrences)),
        "affected_r1_rows": affected_rows,
        "r1_rows_total": int(len(r1)),
        "r1_rows_with_two_deterministic_mappings_after_hypothetical_addition": int(
            len(per_row)
        ),
        "absence_reason": ABSENCE_REASON,
        "duplicate_existing_text_collapses": duplicate_collapses,
        "verification": {
            "every_conjunct_resolves_by_exactly_one_route": True,
            "every_r1_row_has_score_A_and_score_B_mapping": True,
            "missing_rows_unique_by_topic_and_exact_statement": True,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(output, index=False)
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    with audit_output.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"saved {output}")
    print(f"saved {audit_output}")


if __name__ == "__main__":
    main()
