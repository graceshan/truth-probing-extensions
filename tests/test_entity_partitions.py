"""Checks use synthetic entities; no held-out examples are displayed or scored."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.entity_partitions import (
    TOPICS, assign_splits, entity_id, read_entities, save_outputs,
)


class EntityPartitionTests(unittest.TestCase):
    def test_identity_preserves_topic_and_exact_string(self):
        identities = [entity_id("cities", "A"), entity_id("inventors", "A"),
                      entity_id("cities", "a"), entity_id("cities", " A"),
                      entity_id("cities", "é"), entity_id("cities", "e\u0301")]
        self.assertEqual(len(set(identities)), len(identities))
        self.assertEqual(entity_id("cities", "A"), identities[0])

    def test_stratification_seed_and_order(self):
        records = [{"topic": "cities", "entity": f"synthetic-{i}",
                    "entity_id": entity_id("cities", f"synthetic-{i}"),
                    "compound_usable": i < 25} for i in range(35)]
        result = assign_splits(records, 0)
        self.assertEqual(result, assign_splits(list(reversed(records)), 0))
        self.assertNotEqual(result, assign_splits(records, 1))
        self.assertTrue(all("split" not in r for r in records))
        for usable, expected in ((True, (15, 5, 5)), (False, (6, 2, 2))):
            actual = tuple(sum(r["compound_usable"] == usable and r["split"] == split
                               for r in result) for split in ("train", "validation", "test"))
            self.assertEqual(actual, expected)

    def test_atomic_union_keeps_false_only_and_negated_only_entities(self):
        templates = {
            "cities": ("The city of {e} is in X.", "The city of {e} is not in X."),
            "sp_en_trans": ("The Spanish word '{e}' means 'X'.",
                            "The Spanish word '{e}' does not mean 'X'."),
            "inventors": ("{e} lived in X.", "{e} did not live in X."),
            "element_symb": ("{e} has the symbol X.", "{e} does not have the symbol X."),
            "animal_class": ("The {e} is a X.", "The {e} is not a X."),
        }
        with tempfile.TemporaryDirectory() as directory:
            for topic in TOPICS:
                for index, name in enumerate((topic, "neg_" + topic)):
                    with (Path(directory) / (name + ".csv")).open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(("statement", "label"))
                        writer.writerow((templates[topic][index].format(e="known"), 1 - index))
                        writer.writerow((templates[topic][index].format(e="false-only"), index))
                        if index:
                            writer.writerow((templates[topic][index].format(e="negated-only"), 1))
            records, _ = read_entities(directory)
        self.assertEqual(len(records), 15)
        for topic in TOPICS:
            self.assertEqual({r["entity"]: r["compound_usable"]
                              for r in records if r["topic"] == topic},
                             {"known": True, "false-only": False, "negated-only": False})

    def test_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            save_outputs(directory, {"manifest.csv": b"original\n"})
            path = Path(directory) / "manifest.csv"
            modified = path.stat().st_mtime_ns
            save_outputs(directory, {"manifest.csv": b"original\n"})
            self.assertEqual(path.stat().st_mtime_ns, modified)
            with self.assertRaises(AssertionError):
                save_outputs(directory, {"manifest.csv": b"replacement\n", "new.csv": b"new"})
            self.assertEqual(path.read_bytes(), b"original\n")
            self.assertFalse((Path(directory) / "new.csv").exists())


if __name__ == "__main__":
    unittest.main()
