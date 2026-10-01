"""Synthetic-only sequential review export; real registries are never edited."""

import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.entity_partitions import csv_bytes, save_outputs, sha256
from src.negative_audit import build_audit
from src.negative_review_batches import FIELDS, build_review_batch, load_previous_batch
from src.validated_negatives import AUDIT_FILES, apply_reviews, build_registry

ROOT = Path(__file__).resolve().parents[1]


class NegativeReviewBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        _, records, sources = synthetic_fixture(self.root)
        # Larger manually assigned synthetic validation pool for sequential ranks.
        for record in records:
            if record["split"] == "train" and record["compound_usable"]:
                record["split"] = "validation"
        source = self.root / "sources/cities.csv"
        source.write_text(source.read_text() + 'The city of Synthetic-0 is in Object-1.,0\n')
        for item in sources:
            item["sha256"] = sha256((self.root / "sources" / item["file"]).read_bytes())
        write_manifest(self.root, records, sources)
        self.audit = self.root / "audit"
        files, _ = build_audit(self.root / "sources", self.root / "manifest.csv", self.root / "manifest_metadata.json")
        save_outputs(self.audit, files)
        self.config = json.loads((ROOT / "config/clean_protocol/validated_negatives.json").read_text())
        self.config.update(entity_manifest="manifest.csv", entity_manifest_metadata="manifest_metadata.json",
                           source_audit_dir="audit", source_audit_files={name: sha256(files[name]) for name in AUDIT_FILES})
        self.registry = build_registry(self.config, self.root, "validation")

    @staticmethod
    def rows(files):
        return list(csv.DictReader(io.StringIO(files["review_batch.csv"].decode())))

    def previous(self, files, name="batch"):
        save_outputs(self.root / name, files)
        return load_previous_batch(self.root / name)

    @staticmethod
    def decision(row, status):
        return {"fact_id": row["fact_id"], "validation_status": status,
                "evidence_source": "synthetic-only", "evidence_note": "Invented test evidence.",
                "reviewer_or_method": "unit-test", "validation_version": "synthetic-review-v1"}

    def test_first_batch_one_rank_one_unverified_per_unavailable_entity(self):
        files = build_review_batch(self.registry, self.audit)
        rows = self.rows(files)
        self.assertEqual(len(rows), 29)  # 30 synthetic usable validation entities, one already supported.
        self.assertEqual(tuple(rows[0]), FIELDS)
        self.assertEqual(len({r["entity_id"] for r in rows}), len(rows))
        self.assertEqual([(r["topic"], r["entity_id"]) for r in rows], sorted((r["topic"], r["entity_id"]) for r in rows))
        for row in rows:
            self.assertEqual(row["candidate_rank"], "1")
            self.assertEqual(row["validation_status"], "unverified")
            self.assertEqual(self.registry.manifest.entities[row["entity_id"]].split, "validation")
            self.assertIsNone(self.registry.selected(row["entity_id"]))
            self.assertNotIn(row["candidate_object"], json.loads(row["known_true_objects"]))
        summary = json.loads(files["metadata.json"])["summary"]
        self.assertEqual(sum(r["review_batch_candidates"] for r in summary), 29)

    def test_accepted_disappears_rejection_advances_one_unreviewed_stays(self):
        first = build_review_batch(self.registry, self.audit)
        rows, previous = self.rows(first), self.previous(first)
        accepted, rejected, unreviewed = rows[:3]
        updated = apply_reviews(self.registry, [self.decision(accepted, "externally_validated_false"),
                                               self.decision(rejected, "rejected_true_or_ambiguous")], "synthetic-review-v1")
        second = build_review_batch(updated, self.audit, previous=previous)
        by_entity = {r["entity_id"]: r for r in self.rows(second)}
        self.assertNotIn(accepted["entity_id"], by_entity)
        self.assertEqual(by_entity[rejected["entity_id"]]["candidate_rank"], "2")
        self.assertEqual(by_entity[unreviewed["entity_id"]], unreviewed)
        updated_again = apply_reviews(updated, [self.decision(by_entity[rejected["entity_id"]], "rejected_true_or_ambiguous")], "synthetic-review-v1")
        third = build_review_batch(updated_again, self.audit, previous=self.previous(second, "second"))
        third_by_entity = {r["entity_id"]: r for r in self.rows(third)}
        self.assertEqual(third_by_entity[rejected["entity_id"]]["candidate_rank"], "3")
        self.assertEqual(third_by_entity[unreviewed["entity_id"]], unreviewed)

    def test_no_review_keeps_identical_candidate_bytes_and_registry(self):
        before = self.registry.files()
        first = build_review_batch(self.registry, self.audit)
        second = build_review_batch(self.registry, self.audit, previous=self.previous(first))
        self.assertEqual(first["review_batch.csv"], second["review_batch.csv"])
        self.assertEqual(before, self.registry.files())
        self.assertEqual(second, build_review_batch(self.registry, self.audit, previous=self.previous(first)))

    def test_out_of_order_rejections_cannot_skip_next_rank(self):
        first = build_review_batch(self.registry, self.audit)
        entity = self.rows(first)[0]["entity_id"]
        queue = [r for r in self.registry.rows if r["entity_id"] == entity]
        updated = apply_reviews(self.registry, [self.decision(r, "rejected_true_or_ambiguous") for r in queue[:2]], "synthetic-review-v1")
        with self.assertRaisesRegex(ValueError, "cannot skip candidates"):
            build_review_batch(updated, self.audit, previous=self.previous(first))

    def test_accepted_at_any_rank_removes_entity(self):
        first = build_review_batch(self.registry, self.audit)
        entity = self.rows(first)[0]["entity_id"]
        last = [r for r in self.registry.rows if r["entity_id"] == entity][-1]
        updated = apply_reviews(self.registry, [self.decision(last, "externally_validated_false")], "synthetic-review-v1")
        second = self.rows(build_review_batch(updated, self.audit, previous=self.previous(first)))
        self.assertNotIn(entity, {r["entity_id"] for r in second})

    def test_exhaustion_is_reported_without_fallback(self):
        entity = self.rows(build_review_batch(self.registry, self.audit))[0]["entity_id"]
        queue = [r for r in self.registry.rows if r["entity_id"] == entity]
        prefix = apply_reviews(self.registry, [self.decision(r, "rejected_true_or_ambiguous") for r in queue[:-1]], "synthetic-review-v1")
        first = build_review_batch(prefix, self.audit)
        exhausted = apply_reviews(prefix, [self.decision(queue[-1], "rejected_true_or_ambiguous")], "synthetic-review-v1")
        second = build_review_batch(exhausted, self.audit, previous=self.previous(first))
        self.assertNotIn(entity, {r["entity_id"] for r in self.rows(second)})
        self.assertEqual(sum(r["entities_without_unverified_candidate"] for r in json.loads(second["metadata.json"])["summary"]), 1)

    def test_cli_rejects_test_and_train_before_reading_files(self):
        for split in ("train", "test"):
            result = subprocess.run(["python3", "-B", str(ROOT / "scripts/24_prepare_negative_review_batch.py"),
                                     "first", "--split", split, "--config", "/does-not-exist", "--output-dir", str(self.root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation-only", result.stderr)
            self.assertNotIn("FileNotFoundError", result.stderr)
        with patch.object(self.registry, "split", "test"), patch.object(Path, "read_bytes", side_effect=AssertionError("test read")):
            with self.assertRaisesRegex(ValueError, "validation-only"):
                build_review_batch(self.registry, self.audit)

    def test_nonvalidation_knowledge_is_not_interpreted(self):
        path = self.audit / "entity_knowledge.csv"
        reader = csv.DictReader(io.StringIO(path.read_text()))
        rows = list(reader)
        for row in rows:
            if row["split"] != "validation":
                row["known_true_objects"] = "DO-NOT-PARSE"
        path.write_bytes(csv_bytes(rows, reader.fieldnames))
        self.registry.provenance["source_audit_files"]["entity_knowledge.csv"] = sha256(path.read_bytes())
        with patch.object(self.registry.manifest, "eligible", wraps=self.registry.manifest.eligible) as eligible:
            build_review_batch(self.registry, self.audit)
            self.assertTrue(all(call.args[1] == "validation" for call in eligible.call_args_list))

    def test_previous_batch_hash_and_provenance_checked(self):
        first = build_review_batch(self.registry, self.audit)
        previous = self.previous(first)
        previous[1]["negative_candidate_seed"] += 1
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            build_review_batch(self.registry, self.audit, previous=previous)
        (self.root / "batch/review_batch.csv").write_bytes(first["review_batch.csv"] + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_previous_batch(self.root / "batch")

    def test_test_previous_batch_rejected_before_candidate_file_is_opened(self):
        directory = self.root / "test_batch"
        directory.mkdir()
        (directory / "metadata.json").write_text(json.dumps({"split": "test"}))
        # No candidate file exists: the split guard must fire first.
        with self.assertRaisesRegex(ValueError, "validation-only"):
            load_previous_batch(directory)


if __name__ == "__main__":
    unittest.main()
