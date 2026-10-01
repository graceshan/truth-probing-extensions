"""Identity-only synthetic ledgers; no compound examples or test scoring."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from compound_fixtures import write_manifest
from src.clean_compounds import unordered_pair_id
from src.entity_partitions import TOPICS, entity_id, sha256
from src.generalization_protocols import ProtocolCatalog, ROLES

ROOT = Path(__file__).resolve().parents[1]


class GeneralizationProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.records = [
            {"topic": topic, "entity": f"Synthetic-{split}-{i}",
             "entity_id": entity_id(topic, f"Synthetic-{split}-{i}"),
             "split": split, "compound_usable": i != 4}
            for topic in TOPICS for split in ("train", "validation", "test") for i in range(5)
        ]
        write_manifest(self.root, self.records, [])
        self.config = json.loads((ROOT / "config/clean_protocol/generalization_protocols.json").read_text())
        self.config.update(entity_manifest="manifest.csv", entity_manifest_metadata="manifest_metadata.json",
                           entity_manifest_sha256=sha256((self.root / "manifest.csv").read_bytes()))
        self.catalog = self.load_catalog()

    def load_catalog(self):
        path = self.root / "protocols.json"
        path.write_text(json.dumps(self.config))
        return ProtocolCatalog(path, self.root)

    def atomic(self, topic, split, index=0):
        return {"topic": topic, "split": split, "entity_id": entity_id(topic, f"Synthetic-{split}-{index}")}

    def pair(self, topic, split, indices=(0, 1), example_id=None):
        ids = [self.atomic(topic, split, index)["entity_id"] for index in indices]
        row = {"topic": topic, "split": split, "entity_a_id": ids[0], "entity_b_id": ids[1],
               "pair_id": unordered_pair_id(self.catalog.manifest, *ids, split)}
        if example_id is not None:
            row["example_id"] = example_id
        return row

    def ledger(self, regime):
        scopes = self.catalog.regime(regime)["exposure"]
        topic = scopes["atomic_probe_fitting"]["topics"][0]
        evaluation = scopes["evaluation"]
        return {
            "atomic_probe_fitting": [self.atomic(topic, "train", i) for i in range(5)],
            "layer_C_selection": [self.atomic(topic, "validation")],
            "compound_fitting": [self.pair(topic, "train")],
            "calibration": [self.pair(topic, "train", (0, 2))],
            "evaluation": [self.pair(evaluation["topics"][0], evaluation["entity_splits"][0], (2, 3))],
        }

    def test_all_seven_regimes_accept_valid_development_ledgers(self):
        for name in self.config["regimes"]:
            with self.subTest(regime=name):
                result = self.catalog.validate_exposures(name, self.ledger(name))
                self.assertTrue(result["checks_passed"])
                self.assertFalse(result["authorizes_generation_or_scoring"])
                self.assertFalse(result["test_metrics_computed"])

    def test_new_pairings_allows_entity_overlap_but_claim_is_pair_disjoint(self):
        result = self.catalog.validate_exposures("new_pairings", self.ledger("new_pairings"))
        self.assertEqual(result["claim"], "pair-disjoint")
        self.assertEqual(result["evaluation_entity_overlap_with_atomic_fitting"], 2)

    def test_pair_reuse_and_variants_across_compound_roles_fail(self):
        for source, target in (("compound_fitting", "evaluation"), ("calibration", "evaluation"),
                               ("compound_fitting", "calibration")):
            with self.subTest(source=source, target=target):
                ledger = self.ledger("new_pairings")
                original = ledger[source][0]
                original["example_id"] = "synthetic-AB"
                swapped = {**original, "entity_a_id": original["entity_b_id"],
                           "entity_b_id": original["entity_a_id"], "example_id": "synthetic-BA"}
                ledger[target] = [swapped]
                with self.assertRaisesRegex(ValueError, "pair leakage"):
                    self.catalog.validate_exposures("new_pairings", ledger)

    def test_variants_stay_together_without_using_truth_or_surface_fields(self):
        ledger = self.ledger("new_pairings")
        pair = ledger["evaluation"][0]
        # Variant identities only; no statements, facts, or labels are generated.
        ledger["evaluation"] = [{**pair, "example_id": f"synthetic-variant-{i}"} for i in range(16)]
        result = self.catalog.validate_exposures("new_pairings", ledger)
        self.assertEqual(result["counts"]["evaluation"]["pairs"], 1)
        self.assertEqual(result["counts"]["evaluation"]["records"], 16)

    def test_entity_disjoint_rejects_evaluation_entities_in_all_fitting_roles(self):
        for role in ("atomic_probe_fitting", "compound_fitting", "calibration"):
            ledger = self.ledger("entity_disjoint")
            ledger[role] = ([self.atomic("cities", "validation", 2)] if role == "atomic_probe_fitting"
                            else [self.pair("cities", "validation", (2, 3))])
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "forbidden entity split"):
                self.catalog.validate_exposures("entity_disjoint", ledger)

    def test_entity_disjoint_rejects_train_entities_in_evaluation(self):
        ledger = self.ledger("entity_disjoint")
        ledger["evaluation"] = [self.pair("cities", "train", (2, 3))]
        with self.assertRaisesRegex(ValueError, "forbidden entity split"):
            self.catalog.validate_exposures("entity_disjoint", ledger)

    def test_each_strict_holdout_excludes_topic_from_every_development_role(self):
        for topic in TOPICS:
            name = "holdout_" + topic
            for role in ROLES[:-1]:
                ledger = self.ledger(name)
                split = "validation" if role == "layer_C_selection" else "train"
                ledger[role] = ([self.atomic(topic, split)] if role in ROLES[:2]
                                else [self.pair(topic, split)])
                with self.subTest(topic=topic, role=role), self.assertRaisesRegex(ValueError, "forbidden topic"):
                    self.catalog.validate_exposures(name, ledger)

    def test_strict_holdout_evaluates_only_heldout_topic(self):
        for topic in TOPICS:
            name = "holdout_" + topic
            ledger = self.ledger(name)
            other = next(t for t in TOPICS if t != topic)
            ledger["evaluation"] = [self.pair(other, "validation")]
            with self.assertRaisesRegex(ValueError, "forbidden topic"):
                self.catalog.validate_exposures(name, ledger)

    def test_compound_only_topic_holdout_cannot_claim_strict(self):
        ledger = self.ledger("holdout_cities")
        ledger["atomic_probe_fitting"].append(self.atomic("cities", "train"))
        with self.assertRaisesRegex(ValueError, "atomic_probe_fitting: forbidden topic"):
            self.catalog.validate_exposures("holdout_cities", ledger)

    def test_shared_identity_detects_forgery_cross_split_and_cross_topic(self):
        for mode in ("unknown", "wrong_split", "cross_split", "wrong_topic", "pair_id", "self_pair", "unusable"):
            ledger = self.ledger("new_pairings")
            row = ledger["evaluation"][0]
            if mode == "unknown":
                row["entity_a_id"] = "absent"
            elif mode in ("wrong_split", "cross_split"):
                row["entity_a_id"] = self.atomic("cities", "validation")["entity_id"]
                if mode == "wrong_split":
                    row["entity_b_id"] = self.atomic("cities", "validation", 1)["entity_id"]
            elif mode == "wrong_topic":
                row["entity_a_id"] = self.atomic("inventors", "train")["entity_id"]
            elif mode == "pair_id":
                row["pair_id"] = "forged"
            elif mode == "self_pair":
                row["entity_b_id"] = row["entity_a_id"]
            else:
                row["entity_a_id"] = self.atomic("cities", "train", 4)["entity_id"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.catalog.validate_exposures("new_pairings", ledger)

    def test_duplicate_entity_in_shared_manifest_fails(self):
        write_manifest(self.root, self.records + [{**self.records[0], "split": "validation"}], [])
        self.config["entity_manifest_sha256"] = sha256((self.root / "manifest.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.load_catalog()

    def test_exact_topic_scoped_ids_and_atomic_unusable_entities_are_preserved(self):
        first, second = self.atomic("cities", "train", 4), self.atomic("inventors", "train", 4)
        self.assertNotEqual(first["entity_id"], second["entity_id"])
        ledger = self.ledger("entity_disjoint")
        ledger["atomic_probe_fitting"].append(second)
        result = self.catalog.validate_exposures("entity_disjoint", ledger)
        self.assertEqual(result["counts"]["atomic_probe_fitting"]["entities"], 6)

    def test_labels_statements_scores_and_incomplete_ledgers_are_rejected(self):
        for field in ("label", "statement", "score", "compound_label"):
            ledger = self.ledger("new_pairings")
            ledger["evaluation"][0][field] = "forbidden"
            with self.assertRaisesRegex(ValueError, "identity metadata only"):
                self.catalog.validate_exposures("new_pairings", ledger)
        ledger = self.ledger("new_pairings")
        del ledger["calibration"]
        with self.assertRaisesRegex(ValueError, "all five roles"):
            self.catalog.validate_exposures("new_pairings", ledger)

    def test_final_scopes_are_defined_but_cannot_be_instantiated(self):
        for name in self.config["regimes"]:
            final = self.catalog.regime(name, "final")
            expected = ["train"] if name == "new_pairings" else ["test"]
            self.assertEqual(final["exposure"]["evaluation"]["entity_splits"], expected)
            self.assertFalse(final["evaluation_enabled"])
            with self.assertRaisesRegex(ValueError, "final evaluation exposure is disabled"):
                self.catalog.validate_exposures(name, self.ledger(name), phase="final")

    def test_catalog_rules_cannot_be_weakened(self):
        original = copy.deepcopy(self.config)
        self.config["protocols"]["strict_leave_one_topic_out"]["exposure"]["layer_C_selection"]["topics"] = "all"
        with self.assertRaisesRegex(ValueError, "forbidden exposure policy"):
            self.load_catalog()
        self.config = original
        self.config["final_evaluation_enabled"] = True
        with self.assertRaisesRegex(ValueError, "remain disabled"):
            self.load_catalog()

    def test_deterministic_output_and_ledger_order_independence(self):
        ledger = self.ledger("new_pairings")
        first = self.catalog.validate_exposures("new_pairings", ledger)
        reordered = {role: list(reversed(records)) for role, records in reversed(list(ledger.items()))}
        self.assertEqual(first, self.catalog.validate_exposures("new_pairings", reordered))
        self.assertEqual(self.catalog.summary(), self.load_catalog().summary())


if __name__ == "__main__":
    unittest.main()
