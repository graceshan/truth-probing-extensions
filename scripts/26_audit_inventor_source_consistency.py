"""Audit both non-test source-support cohorts and export all unresolved validation inventors."""

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
from src.inventor_country_audit import inventor_source_records, migrate_registry
from src.inventor_source_consistency import audit_source_support
from src.negative_review_batches import build_review_batch, load_previous_batch
from src.validated_negatives import json_bytes, load_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="non-test", choices=("non-test", "test"))
    args = parser.parse_args()
    check(args.scope == "non-test", "test candidate queues are forbidden")
    base = ROOT / "data/clean_protocol/validated_negatives/v1"
    old_config = json.loads((ROOT / "config/clean_protocol/validated_negatives.json").read_text())
    config = json.loads((ROOT / "config/clean_protocol/validated_negatives_inventor_single_country.json").read_text())
    train = load_registry(old_config, ROOT, "train", base / "train")
    old_validation = load_registry(old_config, ROOT, "validation", base / "validation_reviewed_batch_001_manual_external_review_v2")
    v3_path = base / "validation_reviewed_batch_001_inventor_single_country_v3"
    v3 = load_registry(config, ROOT, "validation", v3_path)
    records = inventor_source_records(v3.manifest, ROOT / config["inventor_source_file"])
    files, audit = audit_source_support((train, old_validation), records)
    # V3 was already built with truth-first classification. Reconstruct independently
    # from the preserved v2 input to establish whether any further update is needed.
    expected = migrate_registry(old_validation, config, ROOT)
    check(v3.rows == expected.rows and v3.digest == expected.digest, "v3 differs from truth-consistent reconstruction; investigate before writing a new snapshot")
    by_fact = {r["fact_id"]: r for r in v3.rows}
    batch, _ = load_previous_batch(base / "review_batches/validation_batch_001")
    old_by_fact = {r["fact_id"]: r for r in old_validation.rows}
    accepted = [r for r in batch if r["topic"] != "inventors"]
    check(len(accepted) == 74 and all(by_fact[r["fact_id"]] == old_by_fact[r["fact_id"]] and
          by_fact[r["fact_id"]]["validation_status"] == "externally_validated_false" for r in accepted), "manual non-inventor judgments changed")
    for row in audit["demoted_entities"]:
        if row["split"] == "validation":
            check(row["fact_id"] not in by_fact, "demoted false fact still present in v3")
    review_files = build_review_batch(v3, ROOT / config["source_audit_dir"], topics=("inventors",))
    exported = list(csv.DictReader(io.StringIO(review_files["review_batch.csv"].decode())))
    unresolved = {e.entity_id for e in v3.manifest.eligible("inventors", "validation") if v3.selected(e.entity_id) is None}
    check({r["entity_id"] for r in exported} == unresolved, "unresolved inventor review coverage mismatch")
    check(len(exported) == 14 and any(r["entity"] == "Luther Simjian" for r in exported), "unexpected unresolved inventor count; report discrepancy")
    check(all(r["topic"] == "inventors" and r["validation_status"] == "unverified" and "/" not in r["candidate_object"] for r in exported), "invalid review export")
    verification = {"v3_truth_consistency_verified": True, "v3_registry_sha256": v3.digest,
                    "v4_needed": False, "registry_rows_changed": 0, "prior_74_manual_judgments_unchanged": True,
                    "unresolved_validation_inventors": len(exported), "luther_simjian_included": True,
                    "test_candidate_queues_accessed": False, "compounds_generated": False,
                    "manual_inventor_judgments_imported": False}
    audit_dir = ROOT / "data/clean_protocol/audits/inventor_source_consistency_v1"
    batch_dir = base / "review_batches/validation_inventors_source_consistency_001"
    save_outputs(audit_dir, {**files, "verification.json": json_bytes(verification)})
    save_outputs(batch_dir, review_files)
    print(json.dumps({"audit_dir": str(audit_dir), "review_batch": str(batch_dir),
                      "counts": audit["summaries"], "totals": audit["totals"], **verification}, indent=2))


if __name__ == "__main__":
    main()
