"""Rebuild validation inventor queues, audit exclusions, export the original 13 for review."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean_compounds import check
from src.entity_partitions import save_outputs
from src.inventor_country_audit import audit_migration, migrate_registry
from src.negative_review_batches import build_review_batch, load_previous_batch, validation_only
from src.validated_negatives import json_bytes, load_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    args = parser.parse_args()
    validation_only(args.split)  # Before configuration, source or registry reads.
    base = ROOT / "data/clean_protocol/validated_negatives/v1"
    old_config = json.loads((ROOT / "config/clean_protocol/validated_negatives.json").read_text())
    config = json.loads((ROOT / "config/clean_protocol/validated_negatives_inventor_single_country.json").read_text())
    old = load_registry(old_config, ROOT, "validation", base / "validation_reviewed_batch_001_manual_external_review_v2")
    migrated = migrate_registry(old, config, ROOT)
    audit_files, summary = audit_migration(old, migrated, ROOT / config["inventor_source_file"])
    check(len(summary["specific_exclusions"]) == 3, "required named exclusions missing")
    batch, _ = load_previous_batch(base / "review_batches/validation_batch_001")
    targets = {row["entity_id"] for row in batch if row["topic"] == "inventors"}
    check(len(targets) == 13, "expected exactly 13 original unresolved inventor entities")
    old_rows, new_rows = ({r["fact_id"]: dict(r) for r in registry.rows} for registry in (old, migrated))
    reviewed = [row["fact_id"] for row in batch if row["topic"] != "inventors"]
    check(len(reviewed) == 74 and all(old_rows[fid] == new_rows[fid] and
          new_rows[fid]["validation_status"] == "externally_validated_false" for fid in reviewed), "74 manual judgments changed")
    review_files = build_review_batch(migrated, ROOT / config["source_audit_dir"], topics=("inventors",), entity_ids=targets)
    export = list(csv.DictReader(io.StringIO(review_files["review_batch.csv"].decode())))
    check(len(export) == 13 and {r["entity_id"] for r in export} == targets, "inventor review coverage changed")
    check(all(r["topic"] == "inventors" and r["validation_status"] == "unverified" and "/" not in r["candidate_object"]
              for r in export), "unsafe inventor review batch")
    report = {"prior_74_fact_ids_ranks_statuses_and_all_fields_unchanged": True,
              "all_noninventor_registry_rows_unchanged": True, "manual_inventor_judgments_imported": 0,
              "review_batch_entities": 13, "additional_newly_unavailable_entities": summary["newly_unavailable_entities"],
              "test_facts_accessed": False, "compounds_generated": False}
    output = base / "validation_reviewed_batch_001_inventor_single_country_v3"
    save_outputs(output, migrated.files(parent_sha256=old.digest))
    # Reload against the new semantic configuration, including immutable queue checks.
    reloaded = load_registry(config, ROOT, "validation", output)
    check(reloaded.digest == migrated.digest, "semantic registry reload mismatch")
    audit_dir = ROOT / "data/clean_protocol/audits/inventor_single_country_v1"
    save_outputs(audit_dir, {**audit_files, "verification.json": json_bytes(report)})
    batch_dir = base / "review_batches/validation_inventors_single_country_001"
    save_outputs(batch_dir, review_files)
    print(json.dumps({"registry": str(output), "audit": str(audit_dir), "review_batch": str(batch_dir),
                      "old_candidates": summary["old_candidate_count"], "new_candidates": summary["new_candidate_count"],
                      "excluded": summary["excluded_candidates"], "newly_invalid_known_true": summary["newly_invalid_known_true"],
                      "added": summary["added_single_country_candidates"], "rank_one_changes": summary["rank_one_changes"],
                      **report}, indent=2))


if __name__ == "__main__":
    main()
