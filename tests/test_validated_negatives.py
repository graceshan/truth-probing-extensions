"""Synthetic evidence only. No models, external validation, or real test facts."""

import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.clean_compounds import CompoundGenerator, TEMPLATE_VERSION, build_generation
from src.entity_partitions import csv_bytes, save_outputs, sha256
from src.negative_audit import build_audit
from src.validated_negatives import (
    ACCEPTED, AUDIT_FILES, FIELDS, ValidatedNegativeRegistry, apply_reviews, build_registry,
    export_candidates, load_registry, object_rank,
)

ROOT = Path(__file__).resolve().parents[1]


class ValidatedNegativeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.generation, self.records, self.sources = synthetic_fixture(self.root)
        # Add a single explicit synthetic false proposition; do not review real data.
        path = self.root / "sources/cities.csv"
        path.write_text(path.read_text() + 'The city of Synthetic-0 is in Object-1.,0\n')
        self.refresh_audit()
        self.registry = build_registry(self.config, self.root, "train")

    def refresh_audit(self):
        for source in self.sources:
            source["sha256"] = sha256((self.root / "sources" / source["file"]).read_bytes())
        write_manifest(self.root, self.records, self.sources)
        files, _ = build_audit(self.root / "sources", self.root / "manifest.csv", self.root / "manifest_metadata.json")
        audit = self.root / "audit"
        audit.mkdir(exist_ok=True)
        for name, payload in files.items():
            (audit / name).write_bytes(payload)
        self.config = {"schema_version": 1, "negative_candidate_seed": 0,
                       "ranking_version": "negative-candidate-v1", "registry_version": "validated-negative-registry-v1",
                       "validation_version": "synthetic-v1", "entity_manifest": "manifest.csv",
                       "entity_manifest_metadata": "manifest_metadata.json", "source_audit_dir": "audit",
                       "source_audit_files": {name: sha256(files[name]) for name in AUDIT_FILES}}

    def review(self, rows, statuses):
        return [{"fact_id": row["fact_id"], "validation_status": status,
                 "evidence_source": "synthetic-fixture-only", "evidence_note": "Invented test evidence.",
                 "reviewer_or_method": "unit-test", "validation_version": "synthetic-review-v1"}
                for row, status in zip(rows, statuses)]

    def queue(self, identity=None):
        if identity is None:
            identity = next(e.entity_id for e in self.registry.manifest.eligible("cities", "train") if e.entity == "Synthetic-2")
        return [dict(row) for row in self.registry.rows if row["entity_id"] == identity]

    def test_deterministic_ranking_exact_hash_inputs_and_effective_seed(self):
        repeated = build_registry(self.config, self.root, "train")
        self.assertEqual(self.registry.files(), repeated.files())
        other = build_registry({**self.config, "negative_candidate_seed": 1}, self.root, "train")
        ranks = lambda registry: {r["fact_id"]: r["candidate_rank"] for r in registry.rows}
        self.assertEqual(set(ranks(self.registry)), set(ranks(other)))
        self.assertNotEqual(ranks(self.registry), ranks(other))
        for row in self.registry.rows:
            payload = json.dumps(["negative-candidate-v1", 0, row["topic"], row["entity_id"], row["candidate_object"]],
                                 ensure_ascii=False, separators=(",", ":")).encode()
            self.assertEqual(row["ranking_hash"], sha256(payload))

    def test_audit_row_shuffle_preserves_candidate_ranks(self):
        before = [dict(r) for r in self.registry.rows]
        for name in ("candidate_audit.csv", "entity_knowledge.csv"):
            path = self.root / "audit" / name
            reader = csv.DictReader(io.StringIO(path.read_text()))
            rows = list(reader)
            path.write_bytes(csv_bytes(list(reversed(rows)), reader.fieldnames))
            self.config["source_audit_files"][name] = sha256(path.read_bytes())
        after = build_registry(self.config, self.root, "train")
        self.assertEqual(before, [dict(r) for r in after.rows])

    def test_all_known_true_objects_and_contradictions_are_excluded(self):
        path = self.root / "sources/cities.csv"
        path.write_text(path.read_text() + 'The city of Synthetic-0 is in Object-1.,1\n')
        self.refresh_audit()
        registry = build_registry(self.config, self.root, "train")
        values = {r["candidate_object"] for r in registry.rows if r["topic"] == "cities" and r["entity"] == "Synthetic-0"}
        self.assertEqual(values, {"Object-2", "Object-3"})

    def test_source_supported_initialization_and_unverified_unavailability(self):
        supported = [r for r in self.registry.rows if r["validation_status"] == "source_supported_false"]
        self.assertEqual(len(supported), 1)
        self.assertEqual(supported[0]["candidate_object"], "Object-1")
        self.assertTrue(supported[0]["evidence_source"])
        self.assertTrue(all(r["validation_status"] == "unverified" for r in self.queue()))
        self.assertIsNone(self.registry.selected(self.queue()[0]["entity_id"]))

    def test_rejected_first_candidate_falls_through_to_lowest_accepted(self):
        queue = self.queue()
        reviews = self.review(queue, ["rejected_true_or_ambiguous", "externally_validated_false", "externally_validated_false"])
        registry = apply_reviews(self.registry, reviews, "synthetic-review-v1")
        selected = registry.selected(queue[0]["entity_id"])
        self.assertEqual(selected["candidate_rank"], 2)
        self.assertEqual(selected["fact_id"], queue[1]["fact_id"])
        self.assertEqual([dict(r) for r in registry.rows if r["entity_id"] == selected["entity_id"]][0]["candidate_rank"], 1)

    def test_lowest_accepted_ignores_review_order_and_skips_unverified(self):
        queue = self.queue()
        reviews = self.review(queue[1:], ["externally_validated_false"] * 2)
        a = apply_reviews(self.registry, reviews, "synthetic-review-v1")
        b = apply_reviews(self.registry, list(reversed(reviews)), "synthetic-review-v1")
        self.assertEqual(a.digest, b.digest)
        self.assertEqual(a.selected(queue[0]["entity_id"])["candidate_rank"], 2)

    def test_generator_true_facts_unchanged_and_unaccepted_false_fails(self):
        manifest = self.registry.manifest
        kwargs = dict(manifest=manifest, source_dir=self.root / "sources", split="train",
                      generation_seed=0, template_version=TEMPLATE_VERSION)
        original, validated = CompoundGenerator(**kwargs), CompoundGenerator(**kwargs, negative_registry=self.registry)
        old, new = original._load_facts("cities"), validated._load_facts("cities")
        for (identity, truth), fact in old.items():
            if truth:
                self.assertEqual(fact, new[(identity, truth)])
        ids = sorted({r["entity_id"] for r in self.registry.rows if r["topic"] == "cities" and r["entity"] in ("Synthetic-2", "Synthetic-3")})
        self.assertTrue(all(identity in validated.unavailable_entities for identity in ids))
        validated.example(*ids, True, True, "AND", "AB")
        with self.assertRaisesRegex(ValueError, "entity unavailable"):
            validated.example(*ids, True, False, "AND", "AB")

    def test_validated_smoke_generation_and_reduced_pair_capacity(self):
        # Accept invented candidates for exactly two train entities per topic.
        chosen, updates = set(), []
        for topic in self.registry.summary():
            for entity in self.registry.manifest.eligible(topic["topic"], "train")[:2]:
                row = self.queue(entity.entity_id)[0]
                chosen.add(entity.entity_id)
                updates += self.review([row], ["externally_validated_false"])
        registry = apply_reviews(self.registry, updates, "synthetic-review-v1")
        save_outputs(self.root / "registry", registry.files(parent_sha256=self.registry.digest))
        (self.root / "negative_config.json").write_text(json.dumps(self.config))
        config = {**self.generation, "validated_negatives": {"config": "negative_config.json", "registry_dir": "registry"}}
        output = build_generation(config, self.root)
        self.assertEqual(output, build_generation(config, self.root))
        rows = list(csv.DictReader(io.StringIO(output["compounds.csv"].decode())))
        self.assertEqual(len(rows), 80)
        for row in rows:
            for side in ("a", "b"):
                if row[f"canonical_truth_{side}"] == "False":
                    selected = registry.selected(row[f"entity_{side}_id"])
                    self.assertEqual(row[f"fact_{side}_id"], selected["fact_id"])
                    self.assertIn(selected["validation_status"], ACCEPTED)
        blocked = {**config, "pairs_per_topic": {"inventors": 2}}
        with patch.object(CompoundGenerator, "_load_facts", side_effect=AssertionError("facts read")):
            with self.assertRaisesRegex(ValueError, "capacity 1"):
                build_generation(blocked, self.root)

    def test_test_workflows_fail_before_file_access(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read protected bytes")), \
             patch.object(Path, "read_text", side_effect=AssertionError("read protected text")):
            with self.assertRaisesRegex(ValueError, "test candidates are locked"):
                build_registry(self.config, self.root, "test")
            with self.assertRaisesRegex(ValueError, "test candidates are locked"):
                load_registry(self.config, self.root, "test", self.root / "absent")
            with self.assertRaisesRegex(ValueError, "test candidates are locked"):
                export_candidates(self.registry, "test")
        for action in ("initialize", "export", "review"):
            result = subprocess.run(["python3", "-B", str(ROOT / "scripts/23_manage_negative_registry.py"),
                                     action, "--split", "test", "--config", "/does-not-exist"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test candidates are locked", result.stderr)
            self.assertNotIn("FileNotFoundError", result.stderr)

    def test_other_split_facts_are_not_interpreted(self):
        # Corrupt only synthetic test factual fields; development must never parse them.
        for name, field in (("entity_knowledge.csv", "known_true_objects"), ("candidate_audit.csv", "classification")):
            path = self.root / "audit" / name
            reader = csv.DictReader(io.StringIO(path.read_text()))
            rows = list(reader)
            for row in rows:
                if row["split"] == "test":
                    row[field] = "MUST-NOT-BE-INTERPRETED"
            path.write_bytes(csv_bytes(rows, reader.fieldnames))
            self.config["source_audit_files"][name] = sha256(path.read_bytes())
        self.assertEqual(self.registry.rows, build_registry(self.config, self.root, "train").rows)

    def test_generator_rejects_test_and_mismatched_registry(self):
        for split in ("test", "validation"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                CompoundGenerator(self.registry.manifest, self.root / "sources", split, 0,
                                  TEMPLATE_VERSION, negative_registry=self.registry)
        test_id = self.registry.manifest.eligible("cities", "test")[0].entity_id
        with self.assertRaisesRegex(ValueError, "test candidates are locked"):
            self.registry.selected(test_id)

    def test_review_interface_rejects_scores_models_and_false_source_promotion(self):
        row = self.queue()[0]
        for column in ("probe_score", "model", "model_output", "validation_auroc", "compound_score", "score"):
            reviews = self.review([row], ["externally_validated_false"])
            reviews[0][column] = "forbidden"
            with self.assertRaisesRegex(ValueError, "score/model columns"):
                apply_reviews(self.registry, reviews, "synthetic-review-v1")
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            apply_reviews(self.registry, self.review([row], ["source_supported_false"]), "synthetic-review-v1")
        reviews = self.review([row], ["externally_validated_false"])
        reviews[0]["evidence_source"] = ""
        with self.assertRaisesRegex(ValueError, "require evidence"):
            apply_reviews(self.registry, reviews, "synthetic-review-v1")

    def test_registry_roundtrip_tampering_and_split_checks(self):
        queue = self.queue()
        registry = apply_reviews(self.registry, self.review(queue[:1], ["externally_validated_false"]), "synthetic-review-v1")
        save_outputs(self.root / "registry", registry.files(parent_sha256=self.registry.digest))
        self.assertEqual(registry.digest, load_registry(self.config, self.root, "train", self.root / "registry").digest)
        for key, value in (("candidate_rank", 99), ("statement", "forged"), ("score", 1)):
            rows = [dict(r) for r in self.registry.rows]
            rows[0][key] = value
            with self.assertRaises(ValueError):
                ValidatedNegativeRegistry(self.registry.manifest, rows, self.registry.rows, self.registry.provenance)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            load_registry(self.config, self.root, "validation", self.root / "registry")

    def test_default_generation_is_byte_identical_to_prechange_golden(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = synthetic_fixture(directory)
            self.assertEqual(sha256(build_generation(config, directory)["compounds.csv"]),
                             "0245278c3458055a6a2e6089095682088177873605edff227d2ca25f19b5a8aa")
            self.assertEqual(sha256(build_generation(config, directory)["metadata.json"]),
                             "dea5773ba0742887099abfc73a9e694d46b4268cdfc97bd2839e898562f94bf3")

    def test_exploratory_files_match_git_snapshot(self):
        paths = [p for p in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
                 if (p.startswith(("data/", "results/")) and "clean_protocol" not in p)
                 or p == "scripts/01_generate_r1_r2_datasets.py"]
        result = subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
