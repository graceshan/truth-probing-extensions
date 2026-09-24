"""Source-only proposition and negative-candidate audit; no benchmark generation."""

import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.clean_compounds import EntityManifest
from src.entity_partitions import (
    SPLITS, TOPICS, TRUE_OBJECT_PATTERNS, canonical_json, csv_bytes, entity_id, sha256,
)

PROPOSITION_FIELDS = ("topic", "entity", "entity_id", "object_value", "proposition_key",
                      "source_statement", "source_label", "source_dataset", "source_file", "source_row")
CANDIDATE_FIELDS = ("topic", "split", "entity", "entity_id", "candidate_object", "proposition_key",
                    "classification", "source_proposition_rows", "generator_stratum_blocked")
CLASSES = ("invalid_known_true", "supported_false", "unverified_negative")


def proposition_key(topic, entity, value):
    """Exact tuple identity, independent of source row, label, or surface sentence."""
    return canonical_json([topic, entity, value]).decode("utf-8")


def json_text(value):
    return canonical_json(value).decode("utf-8")


def parse_sources(source_dir):
    propositions, failures, sources = [], [], []
    for topic in TOPICS:
        name = topic + ".csv"
        path = Path(source_dir) / name
        payload = path.read_bytes()
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
        if not {"statement", "label"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{name}: missing statement/label columns")
        count = 0
        for count, row in enumerate(reader, 1):
            statement, label = row.get("statement"), row.get("label")
            match = re.fullmatch(TRUE_OBJECT_PATTERNS[topic], statement or "")
            if match is None or label not in ("0", "1"):
                failures.append({"topic": topic, "source_file": name, "source_row": count,
                                 "source_statement": statement, "source_label": label,
                                 "reason": "parse_failure" if match is None else "invalid_label"})
                continue
            entity, value = match.groups()
            propositions.append(dict(zip(PROPOSITION_FIELDS, (
                topic, entity, entity_id(topic, entity), value, proposition_key(topic, entity, value),
                statement, int(label), topic, name, count,
            ))))
        sources.append({"file": name, "sha256": sha256(payload), "rows": count})
    return propositions, failures, sources


def index_propositions(propositions):
    knowledge, groups = {}, defaultdict(list)
    for row in propositions:
        identity = (row["topic"], row["entity"])
        sets = knowledge.setdefault(identity, {"known_true_objects": set(), "known_false_objects": set()})
        sets["known_true_objects" if row["source_label"] == 1 else "known_false_objects"].add(row["object_value"])
        groups[row["proposition_key"]].append(row)
    contradictions, duplicates, multiple_true = [], [], []
    for key, rows in sorted(groups.items()):
        record = {"proposition_key": key, "topic": rows[0]["topic"], "entity": rows[0]["entity"],
                  "object_value": rows[0]["object_value"], "labels": sorted({r["source_label"] for r in rows}),
                  "occurrences": len(rows), "source_refs": sorted(
                      f"{r['source_file']}:{r['source_row']}" for r in rows)}
        if record["labels"] == [0, 1]:
            contradictions.append(record)
        if len(rows) > 1:
            duplicates.append(record)
    for (topic, entity), sets in sorted(knowledge.items()):
        if len(sets["known_true_objects"]) > 1:
            multiple_true.append({"topic": topic, "entity": entity,
                                  "entity_id": entity_id(topic, entity),
                                  "known_true_objects": sorted(sets["known_true_objects"])})
    return knowledge, groups, {"contradictory_propositions": contradictions,
                               "duplicate_propositions": duplicates,
                               "entities_with_multiple_true_objects": multiple_true}


def classify_candidate(value, known_true_objects, known_false_objects):
    # A contradiction never becomes evidence of a valid negative. It is also
    # reported separately, retaining both source labels rather than resolving it.
    if value in known_true_objects:
        return "invalid_known_true"
    if value in known_false_objects:
        return "supported_false"
    return "unverified_negative"


def audit_candidates(manifest, knowledge, groups):
    candidates, strata = [], []
    empty = {"known_true_objects": set(), "known_false_objects": set()}
    for topic in TOPICS:
        for split in SPLITS:
            eligible = manifest.eligible(topic, split)
            true_sets = {e.entity: knowledge.get((topic, e.entity), empty)["known_true_objects"]
                         for e in eligible}
            pool = set().union(*true_sets.values()) if true_sets else set()
            ambiguous = any(len(values) != 1 for values in true_sets.values())
            no_alternative = any(not (pool - values) for values in true_sets.values())
            blocked = ambiguous or no_alternative
            strata.append({"topic": topic, "split": split, "usable_entities": len(eligible),
                           "object_pool_size": len(pool), "current_generator_blocked": blocked,
                           "nonunique_or_missing_true_mapping": ambiguous,
                           "no_alternative_object": no_alternative})
            for entity in sorted(eligible, key=lambda e: e.entity):
                sets = knowledge.get((topic, entity.entity), empty)
                for value in sorted(pool):
                    # Mirror the current generator's exclusion only when its
                    # unique-object assumption holds. Ambiguous strata remain
                    # explicitly blocked, with all their pool alternatives audited.
                    if sets["known_true_objects"] == {value}:
                        continue
                    key = proposition_key(topic, entity.entity, value)
                    candidates.append(dict(zip(CANDIDATE_FIELDS, (
                        topic, split, entity.entity, entity.entity_id, value, key,
                        classify_candidate(value, **sets),
                        json_text(sorted(f"{r['source_file']}:{r['source_row']}" for r in groups.get(key, []))),
                        blocked,
                    ))))
    return candidates, strata


def build_audit(source_dir, manifest_path, metadata_path):
    manifest = EntityManifest(manifest_path, metadata_path)
    propositions, failures, sources = parse_sources(source_dir)
    knowledge, groups, issues = index_propositions(propositions)
    issues["parse_failures"] = [f for f in failures if f["reason"] == "parse_failure"]
    issues["invalid_labels"] = [f for f in failures if f["reason"] == "invalid_label"]
    manifest_keys = {(e.topic, e.entity) for e in manifest.entities.values()}
    issues["source_entities_absent_from_manifest"] = sorted(set(knowledge) - manifest_keys)
    expected_hashes = {s["file"]: s["sha256"] for s in manifest.metadata["sources"]}
    issues["source_hash_mismatches"] = [s["file"] for s in sources
                                       if expected_hashes.get(s["file"]) != s["sha256"]]
    candidates, strata = audit_candidates(manifest, knowledge, groups)
    supported = Counter((r["topic"], r["entity"]) for r in candidates
                        if r["classification"] == "supported_false")
    entity_rows = []
    all_keys = sorted(manifest_keys | set(knowledge))
    manifest_lookup = {(e.topic, e.entity): e for e in manifest.entities.values()}
    for topic, entity in all_keys:
        record = manifest_lookup.get((topic, entity))
        sets = knowledge.get((topic, entity), {"known_true_objects": set(), "known_false_objects": set()})
        entity_rows.append({"topic": topic, "entity": entity, "entity_id": entity_id(topic, entity),
                            "split": record.split if record else "absent_from_manifest",
                            "compound_usable": record.compound_usable if record else False,
                            "known_true_objects": json_text(sorted(sets["known_true_objects"])),
                            "known_false_objects": json_text(sorted(sets["known_false_objects"])),
                            "supported_false_candidates": supported[(topic, entity)]})
    summaries = []
    for topic in TOPICS:
        topic_entities = [r for r in entity_rows if r["topic"] == topic]
        classes = Counter(r["classification"] for r in candidates if r["topic"] == topic)
        summary = {"topic": topic, "entities": len(topic_entities),
                   "compound_usable_entities": sum(r["compound_usable"] for r in topic_entities),
                   "potential_candidate_negatives": sum(classes.values()),
                   **{name: classes[name] for name in CLASSES},
                   "entities_zero_supported_false": sum(r["supported_false_candidates"] == 0 for r in topic_entities),
                   "usable_entities_zero_supported_false": sum(r["compound_usable"] and
                       r["supported_false_candidates"] == 0 for r in topic_entities),
                   "contradictory_propositions": sum(r["topic"] == topic for r in issues["contradictory_propositions"]),
                   "multiple_true_entities": sum(r["topic"] == topic for r in issues["entities_with_multiple_true_objects"]),
                   "duplicate_proposition_groups": sum(r["topic"] == topic for r in issues["duplicate_propositions"]),
                   "duplicate_extra_rows": sum(r["occurrences"] - 1 for r in issues["duplicate_propositions"] if r["topic"] == topic),
                   "parse_failures": sum(r["topic"] == topic for r in issues["parse_failures"])}
        summaries.append(summary)
    assert all(r["classification"] in CLASSES for r in candidates)
    assert len({r["proposition_key"] for r in candidates}) == len(candidates)
    assert all(r["candidate_object"] not in knowledge[(r["topic"], r["entity"])]["known_true_objects"]
               for r in candidates if r["classification"] == "supported_false")
    examples = []
    for classification in CLASSES:
        for topic in TOPICS:
            eligible = [r for r in candidates if r["classification"] == classification
                        and r["topic"] == topic and r["split"] == "train"]
            examples.extend(eligible[:2])
    metadata = {
        "audit_version": "source-negative-audit-v1", "entity_manifest_sha256": manifest.digest,
        "entity_manifest_version": manifest.version, "sources": sources,
        "source_scope": "Five affirmative atomic datasets only; no inferred negation labels or external facts",
        "proposition_identity": "Exact compact JSON [topic, entity, object]; no normalization",
        "candidate_scope": "All split-local usable-object pool alternatives, excluding an entity's unique true object",
        "blocked_strata_policy": "Retain all true objects; audit ambiguous pool alternatives with generator_stratum_blocked=True",
        "classification_precedence": list(CLASSES),
        "unverified_negative_is_established_false": False,
        "displayed_examples_scope": "train entities only; source proposition tuples, no compound rendering",
        "summaries": summaries, "strata": strata,
        "issue_counts": {key: len(values) for key, values in issues.items()},
        "audit_complete_without_parse_or_provenance_gaps": not failures and not issues["source_hash_mismatches"]
            and not issues["source_entities_absent_from_manifest"],
    }
    def json_bytes(value):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return {
        "propositions.csv": csv_bytes(propositions, PROPOSITION_FIELDS),
        "entity_knowledge.csv": csv_bytes(entity_rows, tuple(entity_rows[0])),
        "candidate_audit.csv": csv_bytes(candidates, CANDIDATE_FIELDS),
        "summary.csv": csv_bytes(summaries, tuple(summaries[0])),
        "train_examples.csv": csv_bytes(examples, CANDIDATE_FIELDS),
        "issues.json": json_bytes(issues), "metadata.json": json_bytes(metadata),
    }, metadata
