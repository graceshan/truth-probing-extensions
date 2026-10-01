"""Audit prior source support under truth-first inventor component semantics."""

import json
from collections import defaultdict

from src.clean_compounds import check
from src.entity_partitions import csv_bytes
from src.inventor_country_semantics import classify_inventor_candidate, country_components, eligible_false_country, true_country_components
from src.validated_negatives import development_only, json_bytes

AUDIT_VERSION = "inventor-source-consistency-v1"


def audit_source_support(registries, source_records):
    """Use the complete historical source-supported cohort; never edit its labels.

    Classification and generation eligibility are separate: non-overlapping slash
    propositions can retain exact source support yet remain ineligible to generate.
    """
    registries = tuple(registries)
    check(bool(registries), "at least one non-test registry is required")
    seen_splits, sources_by_entity = set(), defaultdict(list)
    for registry in registries:
        development_only(registry.split)  # Before iterating any candidate records.
        check(registry.split not in seen_splits, "duplicate registry split")
        seen_splits.add(registry.split)
    for source in source_records:
        development_only(source["split"])
        check(source["source_label"] in (0, 1), "nonbinary raw source label")
        sources_by_entity[source["entity_id"]].append(source)
    rows, summaries = [], []
    for registry in sorted(registries, key=lambda r: r.split):
        prior = [r for r in registry.rows if r["topic"] == "inventors" and r["validation_status"] == "source_supported_false"]
        current = []
        for row in prior:
            check(row["split"] == registry.split, "registry source-support split mismatch")
            sources = sources_by_entity[row["entity_id"]]
            check(all(r["split"] == registry.split and r["entity"] == row["entity"] for r in sources), "source identity mismatch")
            true_rows = [r for r in sources if r["source_label"] == 1]
            exact_false_rows = [r for r in sources if r["source_label"] == 0 and r["raw_object"] == row["candidate_object"]]
            check(true_rows and exact_false_rows, "prior source support lacks raw true/false evidence")
            true = {r["raw_object"] for r in true_rows}
            false = {r["raw_object"] for r in sources if r["source_label"] == 0}
            value = row["candidate_object"]
            overlap = sorted(set(country_components(value)) & true_country_components(true))
            classification = classify_inventor_candidate(value, true, false)
            check(classification == ("invalid_known_true" if overlap else "supported_false"), "truth-first precedence failed")
            eligible = eligible_false_country(value, true)
            def provenance(records):
                return json.dumps([{key: r[key] for key in ("raw_object", "source_label", "source_statement", "source_file", "source_row")}
                                   for r in records], ensure_ascii=False, sort_keys=True)
            record = {"topic": "inventors", "split": registry.split, "entity": row["entity"], "entity_id": row["entity_id"],
                      "fact_id": row["fact_id"], "candidate_rank": row["candidate_rank"], "candidate_object": value,
                      "previous_validation_status": row["validation_status"], "new_semantic_classification": classification,
                      "raw_known_true_objects": json.dumps(sorted(true), ensure_ascii=False),
                      "known_true_components": json.dumps(sorted(true_country_components(true)), ensure_ascii=False),
                      "candidate_components": json.dumps(country_components(value), ensure_ascii=False),
                      "overlap_components": json.dumps(overlap, ensure_ascii=False),
                      "source_label_inconsistency": bool(overlap),
                      "remains_source_supported_false_after_truth_check": not bool(overlap),
                      "eligible_single_country_false": eligible,
                      "retained_as_eligible_source_supported_false": not bool(overlap) and eligible,
                      "raw_true_source_rows": provenance(true_rows), "raw_false_source_rows": provenance(exact_false_rows)}
            rows.append(record)
            current.append(record)
        demotions = sum(r["source_label_inconsistency"] for r in current)
        slash_only = sum(not r["source_label_inconsistency"] and not r["eligible_single_country_false"] for r in current)
        summaries.append({"split": registry.split, "previously_source_supported_false": len(current),
                          "demoted_known_true_overlap": demotions,
                          "source_supported_false_after_truth_check": len(current) - demotions,
                          "additional_slash_only_exclusions": slash_only,
                          "eligible_source_supported_false_remaining": len(current) - demotions - slash_only})
    rows.sort(key=lambda r: (r["split"], r["entity"]))
    demoted = [r for r in rows if r["source_label_inconsistency"]]
    slash = [r for r in rows if not r["source_label_inconsistency"] and not r["eligible_single_country_false"]]
    totals = {key: sum(r[key] for r in summaries) for key in summaries[0] if key != "split"}
    metadata = {"audit_version": AUDIT_VERSION,
                "precedence": ["any_known_true_component_overlap", "exact_raw_false_source_proposition", "unverified"],
                "cohort": "all previously source_supported_false inventor candidates in supplied non-test registries",
                "registry_hashes": {r.split: r.digest for r in registries}, "summaries": summaries, "totals": totals,
                "demoted_entities": [{k: r[k] for k in ("split", "entity", "fact_id")} for r in demoted],
                "luther_is_only_non_test_case": len(demoted) == 1 and demoted[0]["entity"] == "Luther Simjian",
                "raw_source_labels_preserved": True, "test_candidate_queues_accessed": False,
                "compounds_generated": False, "manual_inventor_reviews_imported": False}
    def table(records):
        return csv_bytes(records, tuple(rows[0]) if rows else ("fact_id",))
    return {"all_previously_supported.csv": table(rows), "source_label_inconsistencies.csv": table(demoted),
            "nonoverlap_slash_exclusions.csv": table(slash), "counts.csv": csv_bytes(summaries, tuple(summaries[0])),
            "metadata.json": json_bytes(metadata)}, metadata
