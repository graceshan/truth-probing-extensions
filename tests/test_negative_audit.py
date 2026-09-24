"""Synthetic checks for source evidence, ambiguity, exact identity, and repeatability."""

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.entity_partitions import sha256
from src.negative_audit import (
    build_audit, classify_candidate, index_propositions, parse_sources, proposition_key,
)


class NegativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.records, self.sources = synthetic_fixture(self.root)

    def append_source(self, statement, label):
        path = self.root / "sources/cities.csv"
        with path.open("a", newline="") as handle:
            csv.writer(handle).writerow((statement, label))
        for source in self.sources:
            if source["file"] == "cities.csv":
                source["sha256"] = sha256(path.read_bytes())
                source["rows"] += 1
        write_manifest(self.root, self.records, self.sources)

    def test_known_true_never_supported_false_even_with_contradiction(self):
        for true in ({"A"}, {"A", "B"}):
            for false in (set(), {"A"}, {"A", "B", "C"}):
                for obj in true:
                    self.assertEqual(classify_candidate(obj, true, false), "invalid_known_true")
        self.assertEqual(classify_candidate("B", {"A"}, {"B"}), "supported_false")
        self.assertEqual(classify_candidate("C", {"A"}, {"B"}), "unverified_negative")

    def test_contradictions_duplicates_and_multiple_true_objects_preserved(self):
        self.append_source("The city of Synthetic-0 is in Object-0.", 0)
        self.append_source("The city of Synthetic-0 is in Object-1.", 1)
        rows, _, _ = parse_sources(self.root / "sources")
        knowledge, _, issues = index_propositions(rows)
        self.assertEqual(knowledge[("cities", "Synthetic-0")]["known_true_objects"],
                         {"Object-0", "Object-1"})
        self.assertEqual(knowledge[("cities", "Synthetic-0")]["known_false_objects"], {"Object-0"})
        self.assertEqual(len(issues["contradictory_propositions"]), 1)
        self.assertEqual(issues["contradictory_propositions"][0]["labels"], [0, 1])
        self.assertEqual(len(issues["entities_with_multiple_true_objects"]), 1)
        self.assertEqual(issues["duplicate_propositions"][0]["occurrences"], 2)
        files, metadata = self.audit()
        candidates = list(csv.DictReader(io.StringIO(files["candidate_audit.csv"].decode())))
        invalid = [r for r in candidates if r["entity"] == "Synthetic-0" and r["topic"] == "cities"
                   and r["candidate_object"] in {"Object-0", "Object-1"}]
        self.assertEqual(len(invalid), 2)
        self.assertTrue(all(r["classification"] == "invalid_known_true" for r in invalid))
        self.assertTrue(all(r["generator_stratum_blocked"] == "True" for r in invalid))
        self.assertEqual(metadata["issue_counts"]["contradictory_propositions"], 1)

    def test_proposition_identity_is_exact_and_label_independent(self):
        keys = [proposition_key("cities", "A", "X"), proposition_key("inventors", "A", "X"),
                proposition_key("cities", "a", "X"), proposition_key("cities", " A", "X"),
                proposition_key("cities", "A", "x"), proposition_key("cities", "A", "X "),
                proposition_key("cities", "A", "é"), proposition_key("cities", "A", "e\u0301")]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys[0], '["cities","A","X"]')

    def test_parse_failures_and_invalid_labels_are_reported_without_resolution(self):
        self.append_source("Unparseable fixture statement", 0)
        self.append_source("The city of Synthetic-0 is in Object-0.", "unknown")
        _, metadata = self.audit()
        self.assertEqual(metadata["issue_counts"]["parse_failures"], 1)
        self.assertEqual(metadata["issue_counts"]["invalid_labels"], 1)
        self.assertFalse(metadata["audit_complete_without_parse_or_provenance_gaps"])

    def audit(self):
        return build_audit(self.root / "sources", self.root / "manifest.csv",
                           self.root / "manifest_metadata.json")

    def test_supported_false_requires_exact_proposition_and_split_pool_membership(self):
        self.append_source("The city of Synthetic-0 is in Object-1.", 0)
        self.append_source("The city of Synthetic-0 is in Object-4.", 0)
        files, _ = self.audit()
        candidates = list(csv.DictReader(io.StringIO(files["candidate_audit.csv"].decode())))
        subset = {r["candidate_object"]: r["classification"] for r in candidates
                  if r["topic"] == "cities" and r["entity"] == "Synthetic-0"}
        self.assertEqual(subset, {"Object-1": "supported_false", "Object-2": "unverified_negative",
                                  "Object-3": "unverified_negative"})

    def test_full_audit_is_byte_deterministic_and_does_not_generate(self):
        with patch("src.clean_compounds.CompoundGenerator._load_facts", side_effect=AssertionError("generation")):
            files, _ = self.audit()
            repeated, _ = self.audit()
        self.assertEqual(files, repeated)
        examples = list(csv.DictReader(io.StringIO(files["train_examples.csv"].decode())))
        self.assertTrue(all(r["split"] == "train" for r in examples))
        self.assertNotIn("compounds.csv", files)


if __name__ == "__main__":
    unittest.main()
