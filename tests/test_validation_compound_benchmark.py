"""Synthetic validation-only tests; never generate test compounds."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.clean_compounds import EntityManifest
from src.entity_partitions import TOPICS, entity_id, save_outputs, sha256
from src.negative_audit import build_audit
from src.validated_negatives import AUDIT_FILES, apply_reviews, build_registry
from src.validation_compound_benchmark import build_validation_benchmark, degree_four_pairs

ROOT = Path(__file__).resolve().parents[1]


class ValidationCompoundTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        _, records, sources = synthetic_fixture(self.root)
        for row in records:
            if row["split"] == "train" and row["compound_usable"]:
                row["split"] = "validation"
        write_manifest(self.root, records, sources)
        audit, _ = build_audit(self.root / "sources", self.root / "manifest.csv",
                               self.root / "manifest_metadata.json")
        save_outputs(self.root / "audit", audit)
        negative = json.loads((ROOT / "config/clean_protocol/validated_negatives.json").read_text())
        negative.update(entity_manifest="manifest.csv", entity_manifest_metadata="manifest_metadata.json",
                        source_audit_dir="audit", source_audit_files={name: sha256(audit[name]) for name in AUDIT_FILES})
        self.registry = build_registry(negative, self.root, "validation")
        self.manifest = self.registry.manifest
        updates = []
        for topic in TOPICS:
            for entity in self.manifest.eligible(topic, "validation"):
                queue = [r for r in self.registry.rows if r["entity_id"] == entity.entity_id]
                # Reject rank 1, accept rank 2, also accept rank 3: rank 2 must win.
                for row, status in zip(queue[:3], ("rejected_true_or_ambiguous", "externally_validated_false",
                                                   "externally_validated_false")):
                    updates.append(dict(fact_id=row["fact_id"], validation_status=status,
                                        evidence_source="synthetic fixture", evidence_note="Invented test judgment only.",
                                        reviewer_or_method="unit-test", validation_version="synthetic-v1"))
        self.reviewed = apply_reviews(self.registry, updates, "synthetic-v1")
        save_outputs(self.root / "registry", self.reviewed.files())
        (self.root / "negative.json").write_text(json.dumps(negative))
        protocol = json.loads((ROOT / "config/clean_protocol/generalization_protocols.json").read_text())
        protocol.update(entity_manifest="manifest.csv", entity_manifest_metadata="manifest_metadata.json",
                        entity_manifest_sha256=self.manifest.digest)
        (self.root / "protocol.json").write_text(json.dumps(protocol))
        self.config = json.loads((ROOT / "config/clean_protocol/entity_disjoint_development_validation_v1.json").read_text())
        self.config.update(protocol_config="protocol.json", negative_config="negative.json", source_dir="sources",
                           registry_dir="registry", registry_sha256=self.reviewed.digest,
                           pairs_per_topic={t: 12 for t in TOPICS})

    def test_regular_graph_all_real_topic_sizes_and_seed_order_invariance(self):
        for n in (5, 6, 16, 18, 34, 45, 149):
            records = [{"topic": "cities", "entity": f"Synthetic-{i}", "entity_id": entity_id("cities", f"Synthetic-{i}"),
                        "compound_usable": True, "split": "validation"} for i in range(n)]
            write_manifest(self.root, records, [])
            m = EntityManifest(self.root / "manifest.csv", self.root / "manifest_metadata.json")
            pairs, degrees = degree_four_pairs(m, "cities", "validation", 2*n, 0)
            self.assertEqual(len(pairs), 2*n)
            self.assertEqual(len(set(pairs)), 2*n)
            self.assertEqual(set(degrees.values()), {4})
            self.assertEqual(len(degrees), n)
            self.assertTrue(all(a < b for a, b in pairs))
            original = m.eligible
            with patch.object(m, "eligible", side_effect=lambda *args: list(reversed(original(*args)))):
                self.assertEqual((pairs, degrees), degree_four_pairs(m, "cities", "validation", 2*n, 0))
            if n > 5:
                self.assertNotEqual(pairs, degree_four_pairs(m, "cities", "validation", 2*n, 1)[0])

    def test_end_to_end_determinism_labels_coverage_and_lowest_accepted(self):
        files = build_validation_benchmark(self.config, self.root)
        self.assertEqual(files, build_validation_benchmark(self.config, self.root))
        rows = list(csv.DictReader(io.StringIO(files["development_validation_compounds.csv"].decode())))
        metadata = json.loads(files["metadata.json"])
        self.assertEqual(len(rows), 960)
        self.assertEqual(metadata["entity_count"], 30)
        self.assertEqual(metadata["unique_false_fact_count"], 30)
        self.assertEqual(metadata["false_constituent_occurrences"], 960)
        for row in rows:
            self.assertEqual((row["split"], row["evaluation_phase"]), ("validation", "development"))
            for side in ("a", "b"):
                if row[f"canonical_truth_{side}"] == "False":
                    negative = self.reviewed.selected(row[f"entity_{side}_id"])
                    self.assertEqual(negative["candidate_rank"], 2)
                    self.assertEqual(row[f"fact_{side}_id"], negative["fact_id"])
            a, b = row["canonical_truth_a"] == "True", row["canonical_truth_b"] == "True"
            self.assertEqual(row["compound_label"] == "True", (a and b) if row["operator"] == "AND" else (a or b))
            self.assertEqual((row["surface_first_truth"], row["surface_second_truth"]),
                             (str(a), str(b)) if row["ordering"] == "AB" else (str(b), str(a)))

    def test_wrong_scope_fails_before_files_or_registry(self):
        for override in ({"split": "test"}, {"split": "train"}, {"evaluation_phase": "final"},
                         {"protocol": "new_pairings"}, {"protocol": "strict_leave_one_topic_out"}):
            with patch("src.validation_compound_benchmark.ProtocolCatalog", side_effect=AssertionError("file access")):
                with self.assertRaisesRegex(ValueError, "only entity-disjoint"):
                    build_validation_benchmark({**self.config, **override}, self.root)
        with patch.object(self.manifest, "eligible", side_effect=AssertionError("entity access")):
            with self.assertRaisesRegex(ValueError, "only development"):
                degree_four_pairs(self.manifest, "cities", "test", 12, 0)

    def test_capacity_and_incompatible_count_fail_before_factual_access(self):
        for count, message in ((16, "capacity"), (11, "exactly 2\\*n")):
            config = {**self.config, "pairs_per_topic": {**self.config["pairs_per_topic"], "cities": count}}
            with patch("src.validation_compound_benchmark.load_registry", side_effect=AssertionError("registry loaded")):
                with self.assertRaisesRegex(ValueError, message):
                    build_validation_benchmark(config, self.root)

    def test_unverified_entities_cannot_be_silently_dropped(self):
        save_outputs(self.root / "unreviewed", self.registry.files())
        config = {**self.config, "registry_dir": "unreviewed", "registry_sha256": self.registry.digest}
        with self.assertRaisesRegex(ValueError, "without accepted negative"):
            build_validation_benchmark(config, self.root)

    def test_registry_snapshot_is_pinned(self):
        with self.assertRaisesRegex(ValueError, "reviewed registry hash mismatch"):
            build_validation_benchmark({**self.config, "registry_sha256": "wrong"}, self.root)


if __name__ == "__main__":
    unittest.main()
