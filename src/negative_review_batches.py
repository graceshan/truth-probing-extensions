"""One unverified factual-review candidate per unavailable validation entity."""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

from src.clean_compounds import check
from src.entity_partitions import TOPICS, csv_bytes, sha256
from src.validated_negatives import ACCEPTED, json_bytes, split_rows

BATCH_VERSION = "validation-negative-review-batch-v1"
FIELDS = ("topic", "entity_id", "entity", "candidate_rank", "candidate_object", "statement",
          "known_true_objects", "source_classification", "fact_id", "validation_status")
PROVENANCE_KEYS = ("registry_version", "negative_candidate_seed", "ranking_version",
                   "source_audit_sha256", "entity_manifest_sha256", "entity_manifest_version")


def validation_only(split):
    check(split == "validation", "review batches are validation-only; train/test candidates are forbidden")


def load_previous_batch(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    validation_only(metadata["split"])  # Before opening candidate statements.
    check(metadata["batch_version"] == BATCH_VERSION, "unsupported review batch version")
    payload = (directory / "review_batch.csv").read_bytes()
    check(sha256(payload) == metadata["batch_sha256"], "previous batch hash mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    check(tuple(reader.fieldnames or ()) == FIELDS, "unexpected previous batch fields")
    rows = list(reader)
    check(len(rows) == metadata["row_count"], "previous batch count mismatch")
    return rows, metadata


def build_review_batch(registry, audit_dir, *, previous=None, topics=None, entity_ids=None):
    validation_only(registry.split)  # Before any factual file access.
    scope = set(TOPICS) if topics is None else set(topics)
    check(bool(scope) and scope <= set(TOPICS), "invalid batch topic scope")
    if entity_ids is not None:
        allowed = {e.entity_id for topic in scope for e in registry.manifest.eligible(topic, "validation")}
        check(set(entity_ids) <= allowed, "batch entity scope contains non-validation or wrong-topic entities")
    queues = defaultdict(list)
    for row in registry.rows:
        check(row["split"] == "validation", "non-validation registry row")
        queues[row["entity_id"]].append(row)
    for queue in queues.values():
        queue.sort(key=lambda row: row["candidate_rank"])
        check([row["candidate_rank"] for row in queue] == list(range(1, len(queue) + 1)),
              "candidate ranks must be contiguous")
    prior = {}
    prior_hash = None
    if previous is not None:
        prior_rows, metadata = previous
        validation_only(metadata["split"])
        check(metadata["batch_version"] == BATCH_VERSION, "unsupported previous batch")
        check(all(metadata[k] == registry.provenance[k] for k in PROVENANCE_KEYS), "previous batch queue provenance changed")
        check(sha256(csv_bytes(prior_rows, FIELDS)) == metadata["batch_sha256"], "previous batch content mismatch")
        prior_hash = metadata["batch_sha256"]
        by_fact = {row["fact_id"]: row for row in registry.rows}
        for row in prior_rows:
            check(set(row) == set(FIELDS), "unexpected previous batch fields")
            identity = row["entity_id"]
            check(identity not in prior and row["fact_id"] in by_fact, "duplicate/unknown previous candidate")
            current = by_fact[row["fact_id"]]
            check(all(row[k] == current[k] for k in FIELDS if k not in
                      ("known_true_objects", "candidate_rank", "validation_status")), "previous candidate identity changed")
            check(int(row["candidate_rank"]) == current["candidate_rank"], "previous candidate rank changed")
            check(row["validation_status"] == "unverified", "previous exported candidate must have been unverified")
            prior[identity] = current
    selected, counts = [], []
    for topic in sorted(scope):
        entities = registry.manifest.eligible(topic, "validation")
        if entity_ids is not None:
            entities = [e for e in entities if e.entity_id in entity_ids]
        accepted = exported = exhausted = 0
        for entity in entities:
            queue = queues[entity.entity_id]
            if any(row["validation_status"] in ACCEPTED for row in queue):
                accepted += 1
                continue
            unverified = next((row for row in queue if row["validation_status"] == "unverified"), None)
            if entity.entity_id in prior:
                old = prior[entity.entity_id]
                status, rank = old["validation_status"], old["candidate_rank"]
                check(status in ("unverified", "rejected_true_or_ambiguous"), "invalid progression status")
                expected_rank = rank if status == "unverified" else rank + 1
                if expected_rank > len(queue):
                    check(unverified is None, "earlier unreviewed candidate would be skipped")
                else:
                    check(unverified is not None and unverified["candidate_rank"] == expected_rank,
                          "cannot skip candidates: next batch must stay on current rank or advance exactly one rank")
            if unverified is None:
                exhausted += 1
                continue
            selected.append(unverified)
            exported += 1
        counts.append({"topic": topic, "compound_usable_validation_entities": len(entities),
                       "entities_with_accepted_negative": accepted, "entities_without_accepted_negative": len(entities) - accepted,
                       "review_batch_candidates": exported, "entities_without_unverified_candidate": exhausted})

    # Read only the selected validation entities' already-recorded true objects.
    path = Path(audit_dir) / "entity_knowledge.csv"
    check(sha256(path.read_bytes()) == registry.provenance["source_audit_files"]["entity_knowledge.csv"],
          "source knowledge hash mismatch")
    wanted = {row["entity_id"]: row for row in selected}
    true_objects = {}
    for row in split_rows(path, "validation"):
        identity = row["entity_id"]
        if identity not in wanted:
            continue
        check(identity not in true_objects, "duplicate source knowledge")
        check((row["topic"], row["entity"]) == (wanted[identity]["topic"], wanted[identity]["entity"]),
              "source knowledge entity mismatch")
        values = json.loads(row["known_true_objects"])
        check(isinstance(values, list) and all(isinstance(v, str) for v in values), "invalid known-true objects")
        true_objects[identity] = json.dumps(sorted(set(values)), ensure_ascii=False)
    check(set(true_objects) == set(wanted), "missing source knowledge")
    rows = [{key: true_objects[row["entity_id"]] if key == "known_true_objects" else row[key] for key in FIELDS}
            for row in sorted(selected, key=lambda row: (row["topic"], row["entity_id"]))]
    payload = csv_bytes(rows, FIELDS)
    metadata = {"batch_version": BATCH_VERSION, "split": "validation", "row_count": len(rows),
                "batch_sha256": sha256(payload), "registry_sha256": registry.digest,
                "previous_batch_sha256": prior_hash,
                **{key: registry.provenance[key] for key in PROVENANCE_KEYS},
                "selection_rule": "one lowest-ranked unverified candidate per entity without any accepted negative",
                "next_batch_rule": "accepted entities disappear; unreviewed stays; rejection advances exactly one rank or exhausts queue",
                "summary": counts, "registry_modified": False, "factual_judgments_added": False,
                "test_candidates_exported": 0, "model_or_probe_outputs_used": False}
    if topics is not None or entity_ids is not None:
        metadata["topic_scope"] = sorted(scope)
        metadata["entity_scope"] = sorted(entity_ids) if entity_ids is not None else "all eligible"
    return {"review_batch.csv": payload, "summary.csv": csv_bytes(counts, tuple(counts[0])),
            "metadata.json": json_bytes(metadata)}
