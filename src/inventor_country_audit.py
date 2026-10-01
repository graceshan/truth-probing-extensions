"""Development-only inventor semantics audit and deterministic registry migration."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from src.clean_compounds import ATOMIC_TEMPLATES, check, stable_id
from src.data import ENTITY_PATTERNS
from src.entity_partitions import TRUE_OBJECT_PATTERNS, canonical_json, csv_bytes, sha256
from src.inventor_country_semantics import (
    SEMANTICS_VERSION, classify_inventor_candidate, country_components, eligible_country_pool,
    eligible_false_country, true_country_components,
)
from src.negative_audit import proposition_key


def inventor_queue_rows(config, eligible, knowledge, old_candidates, source_records):
    from src.validated_negatives import FIELDS, object_rank
    entities = [e for e in eligible.values() if e.topic == "inventors"]
    raw_pool = {value for e in entities for value in knowledge[e.entity_id][0]}
    pool = eligible_country_pool(raw_pool)
    rows = []
    false_refs = defaultdict(list)
    for row in source_records:
        if row["source_label"] == 0:
            false_refs[(row["entity_id"], row["raw_object"])].append(f"{row['source_file']}:{row['source_row']}")
    for entity in sorted(entities, key=lambda e: e.entity_id):
        true, false = knowledge[entity.entity_id]
        values = [value for value in pool if eligible_false_country(value, true)]
        values.sort(key=lambda value: (object_rank(config["negative_candidate_seed"], "inventors", entity.entity_id, value), value))
        for rank, value in enumerate(values, 1):
            classification = classify_inventor_candidate(value, true, false)
            supported = classification == "supported_false"
            source = old_candidates.get((entity.entity_id, value))
            refs = json.loads(source["source_proposition_rows"]) if source else false_refs.get((entity.entity_id, value), [])
            check(not supported or refs, "exact false proposition is missing source evidence")
            statement = ATOMIC_TEMPLATES["inventors"].format(entity=entity.entity, object=value)
            rows.append(dict(zip(FIELDS, (
                "inventors", entity.split, entity.entity_id, entity.entity, value, rank,
                object_rank(config["negative_candidate_seed"], "inventors", entity.entity_id, value),
                stable_id("fact", ["inventors", entity.entity_id, statement, False]), statement,
                proposition_key("inventors", entity.entity, value), classification,
                "source_supported_false" if supported else "unverified",
                json.dumps(sorted(refs)) if supported else "",
                "Exact proposition explicitly labeled false in frozen source audit." if supported else "No established false evidence.",
                "source-negative-audit-v1" if supported else "not_reviewed", config["validation_version"],
            ))))
    return rows


def migrate_registry(old_registry, config, root):
    from src.validated_negatives import REVIEW_FIELDS, ValidatedNegativeRegistry, initialize
    check(old_registry.split == "validation", "migration is validation-only; test access forbidden")
    check(config.get("inventor_country_semantics") == SEMANTICS_VERSION, "semantic configuration required")
    manifest, baseline, provenance = initialize(config, root, "validation")
    check(manifest.digest == old_registry.manifest.digest, "migration changed shared manifest")
    check(provenance["negative_candidate_seed"] == old_registry.provenance["negative_candidate_seed"] and
          provenance["ranking_version"] == old_registry.provenance["ranking_version"], "ranking seed or algorithm changed")
    old = {row["fact_id"]: row for row in old_registry.rows}
    check(not any(row["topic"] == "inventors" and row["validation_status"] == "externally_validated_false"
                  for row in old.values()), "manual inventor reviews require an explicit migration audit")
    rows = []
    for expected in baseline:
        fid = expected["fact_id"]
        if expected["topic"] != "inventors":
            check(fid in old, "non-inventor candidate changed")
            rows.append(dict(old[fid]))
        else:
            row = dict(expected)
            if fid in old:
                check(row["source_classification"] == old[fid]["source_classification"], "surviving candidate classification changed")
                row.update({key: old[fid][key] for key in REVIEW_FIELDS})
            rows.append(row)
    migrated = ValidatedNegativeRegistry(manifest, rows, baseline, provenance)
    before = {r["fact_id"]: dict(r) for r in old_registry.rows if r["topic"] != "inventors"}
    after = {r["fact_id"]: dict(r) for r in migrated.rows if r["topic"] != "inventors"}
    check(before == after, "non-inventor registry data changed")
    return migrated


def inventor_source_records(manifest, source_path):
    """Parse all train/validation inventor objects, including unusable entities.

    Mixed-source bytes are hashed. Entity membership is checked first; no test
    object is parsed, classified or included in the returned records.
    """
    source_path = Path(source_path)
    expected = {s["file"]: s["sha256"] for s in manifest.metadata["sources"]}
    check(sha256(source_path.read_bytes()) == expected["inventors.csv"], "inventor source hash mismatch")
    lookup = {e.entity: e for e in manifest.entities.values() if e.topic == "inventors"}
    records = []
    with source_path.open(newline="", encoding="utf-8") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 1):
            entity_match = re.search(ENTITY_PATTERNS["inventors"], row["statement"])
            check(entity_match is not None and entity_match[1] in lookup, "unmapped inventor source entity")
            entity = lookup[entity_match[1]]
            if entity.split == "test":
                continue
            parsed = re.fullmatch(TRUE_OBJECT_PATTERNS["inventors"], row["statement"])
            check(parsed is not None and row["label"] in ("0", "1"), "invalid development inventor source row")
            raw = parsed[2]
            records.append({"topic": "inventors", "split": entity.split, "entity_id": entity.entity_id,
                            "entity": entity.entity, "raw_object": raw,
                            "country_components": json.dumps(country_components(raw), ensure_ascii=False),
                            "slash_valued": "/" in raw, "source_label": int(row["label"]),
                            "source_statement": row["statement"], "source_file": "inventors.csv", "source_row": row_number})
    return records


def audit_migration(old_registry, new_registry, source_path):
    from src.validated_negatives import ACCEPTED, json_bytes
    check(old_registry.split == new_registry.split == "validation", "audit registry scope must be validation")
    records = inventor_source_records(old_registry.manifest, source_path)
    true, false = defaultdict(set), defaultdict(set)
    for row in records:
        (true if row["source_label"] else false)[row["entity_id"]].add(row["raw_object"])
    old = {r["fact_id"]: r for r in old_registry.rows if r["topic"] == "inventors"}
    new = {r["fact_id"]: r for r in new_registry.rows if r["topic"] == "inventors"}
    audit, excluded, newly_invalid = [], [], []
    for fid, row in sorted(old.items()):
        raw, identity = row["candidate_object"], row["entity_id"]
        classification = classify_inventor_candidate(raw, true[identity], false[identity])
        overlap = sorted(set(country_components(raw)) & true_country_components(true[identity]))
        record = {"topic": "inventors", "split": "validation", "entity": row["entity"], "entity_id": identity,
                  "fact_id": fid, "raw_candidate_object": raw, "old_candidate_rank": row["candidate_rank"],
                  "old_source_classification": row["source_classification"], "old_validation_status": row["validation_status"],
                  "semantic_classification": classification,
                  "candidate_components": json.dumps(country_components(raw)),
                  "raw_known_true_objects": json.dumps(sorted(true[identity])),
                  "known_true_components": json.dumps(sorted(true_country_components(true[identity]))),
                  "overlap_components": json.dumps(overlap), "eligible_single_country_negative": eligible_false_country(raw, true[identity]),
                  "excluded": fid not in new,
                  "exclusion_reasons": json.dumps((["slash_valued"] if "/" in raw else []) + (["known_true_overlap"] if overlap else []))}
        audit.append(record)
        if fid not in new:
            check(not record["eligible_single_country_negative"], "eligible original candidate unexpectedly lost")
            excluded.append(record)
        if classification == "invalid_known_true" and row["source_classification"] != "invalid_known_true":
            newly_invalid.append(record)
    added = [dict(row) for fid, row in new.items() if fid not in old]
    def rank_one(rows):
        return {r["entity_id"]: r for r in rows if r["candidate_rank"] == 1}
    before, after = rank_one(old.values()), rank_one(new.values())
    rank_changes = [{"entity": row["entity"], "entity_id": eid,
                     "old_candidate": row["candidate_object"], "old_fact_id": row["fact_id"],
                     "new_candidate": after[eid]["candidate_object"], "new_fact_id": after[eid]["fact_id"]}
                    for eid, row in sorted(before.items()) if row["fact_id"] != after[eid]["fact_id"]]
    slash = defaultdict(lambda: {"true_source_rows": 0, "false_source_rows": 0, "validation_candidates": 0, "source_splits": set()})
    for row in records:
        if row["slash_valued"]:
            item = slash[row["raw_object"]]
            item["true_source_rows" if row["source_label"] else "false_source_rows"] += 1
            item["source_splits"].add(row["split"])
    for row in old.values():
        if "/" in row["candidate_object"]:
            slash[row["candidate_object"]]["validation_candidates"] += 1
    slash_rows = [{"raw_object": raw, **{k: json.dumps(sorted(v)) if isinstance(v, set) else v for k, v in item.items()}}
                  for raw, item in sorted(slash.items())]
    source_conflicts = [{**row, "overlap_components": json.dumps(sorted(set(country_components(row["raw_object"])) & true_country_components(true[row["entity_id"]])))}
                       for row in records if row["source_label"] == 0 and
                       set(country_components(row["raw_object"])) & true_country_components(true[row["entity_id"]])]
    old_available = {r["entity_id"] for r in old.values() if r["validation_status"] in ACCEPTED}
    new_available = {r["entity_id"] for r in new.values() if r["validation_status"] in ACCEPTED}
    newly_unavailable = [{"entity": old_registry.manifest.entities[eid].entity, "entity_id": eid} for eid in sorted(old_available-new_available)]
    checks = []
    for entity, value in (("Henry Ford", "Turkey/the U.S"), ("Hans von Ohain", "Turkey/the U.S"),
                          ("George Washington Carver", "Poland/Germany")):
        matches = [r for r in audit if r["entity"] == entity and r["raw_candidate_object"] == value]
        if not matches:
            continue  # Synthetic fixtures need not contain these names.
        check(len(matches) == 1 and matches[0]["excluded"] and not matches[0]["eligible_single_country_negative"], "required exclusion failed")
        checks.append({"entity": entity, "candidate": value, "excluded": True, "classification": matches[0]["semantic_classification"]})
    summary = {"semantic_version": SEMANTICS_VERSION, "scope": "train/validation atomic source values; validation v2 registry candidates; no test facts",
               "old_registry_sha256": old_registry.digest, "new_registry_sha256": new_registry.digest,
               "inventor_source_sha256": sha256(Path(source_path).read_bytes()),
               "old_candidate_count": len(old), "new_candidate_count": len(new), "excluded_candidates": len(excluded),
               "newly_invalid_known_true": len(newly_invalid), "added_single_country_candidates": len(added),
               "rank_one_changes": len(rank_changes), "slash_values": slash_rows,
               "newly_unavailable_entities": newly_unavailable,
               "removed_previously_accepted_candidates": [r for r in excluded if r["old_validation_status"] in ACCEPTED],
               "source_semantic_conflicts": len(source_conflicts), "specific_exclusions": checks,
               "test_candidate_objects_inspected": 0, "compounds_generated": False, "manual_inventor_judgments_imported": False}
    def table(rows, empty_fields):
        return csv_bytes(rows, tuple(rows[0]) if rows else empty_fields)
    files = {"source_objects.csv": table(records, ("raw_object",)), "slash_values.csv": table(slash_rows, ("raw_object",)),
             "old_candidate_semantic_audit.csv": table(audit, ("fact_id",)), "excluded_candidates.csv": table(excluded, ("fact_id",)),
             "newly_invalid_known_true.csv": table(newly_invalid, ("fact_id",)), "new_candidates.csv": table(added, ("fact_id",)),
             "new_candidate_audit.csv": table([dict(r) for r in new.values()], ("fact_id",)),
             "rank_one_changes.csv": table(rank_changes, ("entity_id",)), "source_semantic_conflicts.csv": table(source_conflicts, ("entity_id",)),
             "summary.json": json_bytes(summary)}
    return files, summary
