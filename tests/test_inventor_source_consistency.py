"""Synthetic precedence/cohort checks, without compounds or test facts."""

import copy
import csv
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from src.inventor_country_semantics import classify_inventor_candidate
from src.inventor_source_consistency import audit_source_support

ROOT = Path(__file__).resolve().parents[1]


class InventorSourceConsistencyTests(unittest.TestCase):
    def fixture(self):
        records, rows = [], {"train": [], "validation": []}
        examples = [("train", "Train overlap", "Austria/the U.S", "the U.S"),
                    ("train", "Train consistent", "France", "Germany"),
                    ("train", "Slash consistent", "France", "Germany/Italy"),
                    ("validation", "Luther Simjian", "Turkey/the U.S", "the U.S"),
                    ("validation", "Validation consistent", "Poland/Germany", "France")]
        for i, (split, entity, true, candidate) in enumerate(examples):
            identity = "synthetic-entity-" + str(i)
            for label, value in ((1, true), (0, candidate)):
                records.append({"split": split, "entity_id": identity, "entity": entity,
                                "raw_object": value, "source_label": label,
                                "source_statement": f"{entity} lived in {value}.",
                                "source_file": "synthetic.csv", "source_row": len(records) + 1})
            rows[split].append({"topic": "inventors", "split": split, "entity": entity, "entity_id": identity,
                                "fact_id": "synthetic-fact-" + str(i), "candidate_rank": 1,
                                "candidate_object": candidate, "validation_status": "source_supported_false"})
        registries = tuple(SimpleNamespace(split=split, rows=values, digest="synthetic-" + split) for split, values in rows.items())
        return registries, records

    def test_truth_overlap_is_checked_before_exact_false_membership(self):
        class ForbiddenFalseLookup:
            def __contains__(self, value):
                raise AssertionError("false label consulted before known-truth consistency")
        self.assertEqual(classify_inventor_candidate("the U.S", {"Turkey/the U.S"}, ForbiddenFalseLookup()), "invalid_known_true")
        self.assertEqual(classify_inventor_candidate("Turkey/the U.S", {"the U.S"}, ForbiddenFalseLookup()), "invalid_known_true")
        self.assertEqual(classify_inventor_candidate("France", {"Poland/Germany"}, {"France"}), "supported_false")

    def test_all_non_test_supported_candidates_audited_and_raw_labels_preserved(self):
        registries, records = self.fixture()
        before = copy.deepcopy(records)
        files, metadata = audit_source_support(registries, records)
        self.assertEqual(metadata["totals"], {"previously_source_supported_false": 5, "demoted_known_true_overlap": 2,
                         "source_supported_false_after_truth_check": 3, "additional_slash_only_exclusions": 1,
                         "eligible_source_supported_false_remaining": 2})
        self.assertFalse(metadata["luther_is_only_non_test_case"])
        demoted = list(csv.DictReader(io.StringIO(files["source_label_inconsistencies.csv"].decode())))
        self.assertEqual({r["entity"] for r in demoted}, {"Train overlap", "Luther Simjian"})
        for row in demoted:
            self.assertEqual(row["new_semantic_classification"], "invalid_known_true")
            self.assertEqual(json.loads(row["raw_false_source_rows"])[0]["source_label"], 0)
        self.assertEqual(before, records)

    def test_deterministic_audit(self):
        registries, records = self.fixture()
        self.assertEqual(audit_source_support(registries, records), audit_source_support(registries[::-1], records))

    def test_test_guard_precedes_candidate_and_source_access(self):
        class ForbiddenRows:
            def __iter__(self):
                raise AssertionError("test candidates accessed")
        with self.assertRaisesRegex(ValueError, "test candidates are locked"):
            audit_source_support([SimpleNamespace(split="test", rows=ForbiddenRows())], [])
        registries, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "test candidates are locked"):
            audit_source_support(registries, [{"split": "test"}])
        result = subprocess.run(["python3", "-B", str(ROOT / "scripts/26_audit_inventor_source_consistency.py"), "--scope", "test"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test candidate queues are forbidden", result.stderr)


if __name__ == "__main__":
    unittest.main()
