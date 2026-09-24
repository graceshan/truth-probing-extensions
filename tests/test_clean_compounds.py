"""Synthetic-only correctness and leakage checks for clean generation."""

import csv
import io
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.clean_compounds import (
    FIELDS, TEMPLATE_VERSION, CompoundGenerator, EntityManifest, boolean_truth,
    build_generation, sample_pairs, unordered_pair_id, validate_r1,
)
from src.entity_partitions import TOPICS


class CleanCompoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.records, self.sources = synthetic_fixture(self.root)
        self.manifest = self.load_manifest()
        self.generator = self.generator_for("train")
        self.ids = [e.entity_id for e in self.manifest.eligible("cities", "train")]

    def load_manifest(self):
        return EntityManifest(self.root / "manifest.csv", self.root / "manifest_metadata.json")

    def generator_for(self, split):
        return CompoundGenerator(self.manifest, self.root / "sources", split, 0, TEMPLATE_VERSION)

    def test_boolean_helper_binary_and_nary(self):
        for a, b in product((False, True), repeat=2):
            self.assertEqual(boolean_truth("AND", [a, b]), a and b)
            self.assertEqual(boolean_truth("OR", [a, b]), a or b)
            self.assertEqual(boolean_truth("XOR", [a, b]), a != b)
        self.assertTrue(boolean_truth("XOR", [True, True, True]))
        self.assertFalse(boolean_truth("XOR", [True, False, True]))
        self.assertTrue(boolean_truth("OR", [False, False, True]))
        self.assertFalse(boolean_truth("AND", [True, True, False]))
        for values in ([], ["TF"], [1, 0]):
            with self.assertRaises(ValueError):
                boolean_truth("AND", values)
        with self.assertRaises(ValueError):
            boolean_truth("INVALID", [True])

    def test_each_split_only_uses_authorized_usable_entities(self):
        for split in ("train", "validation", "test"):
            for topic in TOPICS:
                gen = self.generator_for(split)
                pairs = sample_pairs(self.manifest, topic, split, 1, 0)
                rows = gen.standard_r1_pair(*pairs[0])
                validate_r1(rows, self.manifest, split, {topic: 1})
                for row in rows:
                    for key in ("entity_a_id", "entity_b_id"):
                        entity = self.manifest.entities[row[key]]
                        self.assertEqual(entity.split, split)
                        self.assertTrue(entity.compound_usable)

    def test_cross_split_wrong_split_self_and_unusable_pairs_fail_before_facts(self):
        other = self.manifest.eligible("cities", "validation")
        unusable = next(e for e in self.manifest.entities.values()
                        if e.topic == "cities" and not e.compound_usable)
        cross_topic = self.manifest.eligible("inventors", "train")[0]
        invalid = ((self.ids[0], other[0].entity_id),
                   (other[0].entity_id, other[1].entity_id),
                   (self.ids[0], self.ids[0]), (self.ids[0], unusable.entity_id),
                   (self.ids[0], cross_topic.entity_id), (self.ids[0], "absent"))
        with patch.object(self.generator, "_load_facts", side_effect=AssertionError("facts loaded")):
            for pair in invalid:
                with self.assertRaises(ValueError):
                    self.generator.standard_r1_pair(*pair)

    def test_duplicate_manifest_identity_across_splits_fails(self):
        duplicate = {**self.records[0], "split": "test"}
        write_manifest(self.root, self.records + [duplicate], self.sources)
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.load_manifest()

    def test_manifest_and_source_hashes_are_checked(self):
        path = self.root / "manifest.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load_manifest()
        path.write_bytes(original)
        source = self.root / "sources/cities.csv"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.generator.standard_r1_pair(*self.ids[:2])

    def test_false_objects_come_only_from_requested_split(self):
        rows = self.generator.standard_r1_pair(*self.ids[:2])
        for row in rows:
            for side in ("a", "b"):
                statement = row[f"fact_{side}_statement"]
                entity = self.manifest.entities[row[f"entity_{side}_id"]]
                own_object = "Object-" + entity.entity.split("-")[-1]
                used_object = statement.removesuffix(".").split(" is in ")[1]
                self.assertIn(used_object, {f"Object-{i}" for i in range(4)})
                self.assertEqual(used_object == own_object, row[f"canonical_truth_{side}"])

    def test_insufficient_true_object_diversity_fails(self):
        path = self.root / "sources/cities.csv"
        text = path.read_text()
        for i in range(4):
            text = text.replace(f"Object-{i}", "Same-object")
        path.write_text(text)
        from src.entity_partitions import sha256
        for source in self.sources:
            if source["file"] == "cities.csv":
                source["sha256"] = sha256(path.read_bytes())
        write_manifest(self.root, self.records, self.sources)
        self.manifest = self.load_manifest()
        with self.assertRaisesRegex(ValueError, "no distinct within-split wrong object"):
            self.generator_for("train").standard_r1_pair(*self.ids[:2])

    def test_ids_canonical_labels_and_surface_swap(self):
        a, b = self.ids[:2]
        ab = self.generator.example(a, b, True, False, "AND", "AB")
        ba = self.generator.example(a, b, True, False, "AND", "BA")
        reversed_input = self.generator.example(b, a, True, False, "AND", "AB")
        self.assertEqual(ab, reversed_input)
        self.assertEqual(ab["pair_id"], ba["pair_id"])
        self.assertEqual(unordered_pair_id(self.manifest, b, a, "train"), ab["pair_id"])
        self.assertNotEqual(ab["example_id"], ba["example_id"])
        for field in ("entity_a_id", "entity_b_id", "fact_a_id", "fact_b_id",
                      "canonical_truth_a", "canonical_truth_b"):
            self.assertEqual(ab[field], ba[field])
        self.assertEqual((ba["canonical_truth_a"], ba["canonical_truth_b"]), (True, False))
        self.assertEqual((ba["surface_first_truth"], ba["surface_second_truth"]), (False, True))
        self.assertEqual((ba["surface_first_entity_id"], ba["surface_second_entity_id"]), (b, a))
        xor = self.generator.example(a, b, True, False, "XOR", "BA")
        self.assertTrue(xor["compound_label"])
        self.assertEqual(ab["fact_a_id"], xor["fact_a_id"])
        self.assertEqual(ab["pair_id"], xor["pair_id"])

    def test_sixteen_variants_and_all_four_canonical_cells(self):
        rows = self.generator.standard_r1_pair(*self.ids[:2])
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["example_id"] for r in rows}), 16)
        self.assertEqual({(r["canonical_truth_a"], r["canonical_truth_b"]) for r in rows},
                         set(product((True, False), repeat=2)))
        for row in rows:
            self.assertEqual(row["compound_label"], boolean_truth(row["operator"],
                             [row["canonical_truth_a"], row["canonical_truth_b"]]))
        validate_r1(rows, self.manifest, "train", {"cities": 1})
        with self.assertRaises(ValueError):
            validate_r1(rows[:-1], self.manifest, "train", {"cities": 1})
        corrupted = [dict(r) for r in rows]
        corrupted[0]["surface_first_truth"] = not corrupted[0]["surface_first_truth"]
        with self.assertRaisesRegex(ValueError, "surface"):
            validate_r1(corrupted, self.manifest, "train", {"cities": 1})

    def test_capacity_is_checked_before_sampling_or_loading_facts(self):
        with patch("src.clean_compounds.combinations", side_effect=AssertionError("sampling started")):
            with self.assertRaisesRegex(ValueError, "requested 7 pairs, capacity 6"):
                sample_pairs(self.manifest, "cities", "train", 7, 0)
        bad_config = {**self.config, "pairs_per_topic": {"cities": 1, "inventors": 7}}
        with patch.object(CompoundGenerator, "_load_facts", side_effect=AssertionError("facts loaded")):
            with self.assertRaisesRegex(ValueError, "capacity"):
                build_generation(bad_config, self.root)
        for count in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                sample_pairs(self.manifest, "cities", "train", count, 0)

    def test_unique_unordered_pairs_allow_entity_reuse(self):
        pairs = sample_pairs(self.manifest, "cities", "train", 6, 0)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len({frozenset(pair) for pair in pairs}), 6)
        self.assertEqual(len({entity for pair in pairs for entity in pair}), 4)
        self.assertEqual(sample_pairs(self.manifest, "cities", "train", 0, 0), [])

    def test_byte_determinism_and_input_order_independence(self):
        files = build_generation(self.config, self.root)
        self.assertEqual(files, build_generation(self.config, self.root))
        rows = list(csv.DictReader(io.StringIO(files["compounds.csv"].decode())))
        self.assertEqual(len(rows), 80)
        self.assertEqual(tuple(rows[0]), FIELDS)
        metadata = json.loads(files["metadata.json"])
        self.assertEqual(metadata["output_row_count"], 80)
        self.assertEqual(metadata["requested_pairs_per_topic"], self.config["pairs_per_topic"])
        write_manifest(self.root, list(reversed(self.records)), self.sources)
        reordered = build_generation(self.config, self.root)
        self.assertEqual(files["compounds.csv"], reordered["compounds.csv"])

    def test_sampling_seed_is_effective_without_changing_pair_identity(self):
        p0 = sample_pairs(self.manifest, "cities", "train", 3, 0)
        p1 = sample_pairs(self.manifest, "cities", "train", 3, 1)
        self.assertNotEqual(p0, p1)
        self.assertEqual(p1, sample_pairs(self.manifest, "cities", "train", 3, 1))
        for a, b in set(p0) & set(p1):
            self.assertEqual(unordered_pair_id(self.manifest, a, b, "train"),
                             unordered_pair_id(self.manifest, b, a, "train"))


if __name__ == "__main__":
    unittest.main()
