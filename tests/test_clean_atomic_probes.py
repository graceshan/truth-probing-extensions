"""Synthetic tests: fitting/selection never receive test labels or activations."""

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from src.clean_atomic_probes import (
    AtomicData, C_VALUES, PROBE_CONFIG, RETRY_MAX_ITER, SELECTION_RULE, evaluate_frozen_test,
    fit_and_save, fit_converged_probe, load_frozen_probe, select_probe, selection_key,
)
from src.clean_compounds import ATOMIC_TEMPLATES
from src.entity_partitions import FIELDS, TOPICS, csv_bytes, entity_id, sha256


def atomic_fixture(root):
    root = Path(root)
    for directory in ("source", "acts_a", "acts_b"):
        (root / directory).mkdir()
    manifest, sources = [], []
    for topic in TOPICS:
        for i, split in enumerate(("train", "train", "validation", "validation", "test", "test")):
            manifest.append({"topic": topic, "entity": f"Fixture-{i}",
                             "entity_id": entity_id(topic, f"Fixture-{i}"),
                             "compound_usable": i != 1, "split": split})
        for negated in (False, True):
            dataset = "neg_" + topic if negated else topic
            rows = []
            for i in range(6):
                for truth in (0, 1):
                    statement = ATOMIC_TEMPLATES[topic].format(entity=f"Fixture-{i}", object=f"Object-{truth}")
                    if negated:
                        statement = (statement.replace(" is in ", " is not in ")
                                     .replace(" means ", " does not mean ")
                                     .replace(" lived in ", " did not live in ")
                                     .replace(" has the symbol ", " does not have the symbol ")
                                     .replace(" is a ", " is not a "))
                    rows.append({"statement": statement, "label": 1 - truth if negated else truth})
            payload = csv_bytes(rows, ("statement", "label"))
            (root / "source" / (dataset + ".csv")).write_bytes(payload)
            sources.append({"file": dataset + ".csv", "sha256": sha256(payload)})
            acts = np.zeros((12, 2, 3), dtype=np.float16)
            for row, record in enumerate(rows):
                # Train/validation carry different markers to audit fit/scoring calls.
                acts[row, :, 0] = 2 * record["label"] - 1
                acts[row, :, 1] = 10 if row < 4 else 20
                acts[row, :, 2] = row % 4
            acts[8:] = np.nan  # Any accidental test-value read would fail selection.
            for directory in ("acts_a", "acts_b"):
                (root / directory / (dataset + ".csv")).write_bytes(payload)
                np.save(root / directory / (dataset + ".npy"), acts)
    payload = csv_bytes(manifest, FIELDS)
    (root / "manifest.csv").write_bytes(payload)
    (root / "manifest_metadata.json").write_text(json.dumps({
        "config": {"schema_version": 1}, "manifest_sha256": sha256(payload), "sources": sources}))
    return {"schema_version": 1, "C_values": C_VALUES, "source_dir": "source",
            "entity_manifest": "manifest.csv", "entity_manifest_metadata": "manifest_metadata.json",
            "models": {key: {"model": key, "activation_pattern": directory + "/{dataset}.npy",
                             "metadata_pattern": directory + "/{dataset}.csv", "n_layers": 2,
                             "hidden_size": 3, "dtype": "float16", "results_dir": "out/" + key}
                       for key, directory in (("a", "acts_a"), ("b", "acts_b"))}}


class CleanAtomicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = atomic_fixture(self.root)
        self.data = AtomicData(self.root, self.config, "a")

    def test_manifest_join_forms_models_and_unusable_entities(self):
        other = AtomicData(self.root, self.config, "b")
        self.assertEqual(self.data.structural["assignment_sha256"], other.structural["assignment_sha256"])
        self.assertEqual(self.data.counts, other.counts)
        for split in ("train", "validation", "test"):
            self.assertEqual(sum(r["entities"] for r in self.data.counts if r["split"] == split), 10)
            self.assertEqual(sum(r["rows"] for r in self.data.counts if r["split"] == split), 40)
        unusable = entity_id("cities", "Fixture-1")
        self.assertIn(unusable, self.data.partition("train").entity_ids)
        self.assertEqual(self.data.activation_rows_read, {"train": 0, "validation": 0, "test": 0})

    def test_default_drops_test_labels_and_rejects_test_access(self):
        self.assertTrue(all(set(b.labels) == {"train", "validation"} for b in self.data.blocks))
        with self.assertRaises(PermissionError):
            self.data.partition("test")
        with self.assertRaises(PermissionError):
            evaluate_frozen_test(self.data, self.root / "absent")

    def test_train_only_fit_validation_only_decision_calls_full_grid(self):
        original_fit, original_decision = LogisticRegression.fit, LogisticRegression.decision_function
        fit_calls, decision_calls = [], []
        def spy_fit(probe, X, y, **kwargs):
            self.assertEqual(X.shape, (40, 3))
            self.assertTrue(np.all(X[:, 1] == 10))
            self.assertTrue(np.array_equal(y, self.data.partition("train").labels))
            for key, expected in PROBE_CONFIG.items():
                self.assertEqual(probe.get_params()[key], expected)
            fit_calls.append(probe.C)
            return original_fit(probe, X, y, **kwargs)
        def spy_decision(probe, X):
            self.assertEqual(X.shape, (40, 3))
            self.assertTrue(np.all(X[:, 1] == 20))
            decision_calls.append(len(X))
            return original_decision(probe, X)
        with patch.object(LogisticRegression, "fit", spy_fit), patch.object(LogisticRegression, "decision_function", spy_decision):
            rows, best, _ = select_probe(self.data.partition("train"), self.data.partition("validation"))
        self.assertEqual(fit_calls, C_VALUES * 2)
        self.assertEqual(len(decision_calls), 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual((best["layer"], best["C"]), (0, 0.001))
        self.assertEqual(self.data.activation_rows_read["test"], 0)
        self.assertTrue(all("test" not in field for row in rows for field in row))
        for row in rows:
            self.assertEqual(row["initial_max_iter"], 2000)
            self.assertFalse(row["retry_needed"])
            self.assertTrue(row["final_converged"])
            self.assertEqual(row["final_n_iter"], row["n_iter"])
            self.assertEqual(row["validation_overall_auroc"], 1.0)
            self.assertEqual(row["validation_topic_macro_auroc"], 1.0)
            self.assertEqual(row["validation_affirmative_auroc"], 1.0)
            self.assertEqual(row["validation_negated_auroc"], 1.0)

    def test_retry_changes_only_iteration_budget_and_reuses_identical_train_arrays(self):
        train = self.data.partition("train")
        X, y = train.matrix(0), train.labels
        original_fit = LogisticRegression.fit
        calls = []
        def fit(probe, actual_X, actual_y):
            self.assertIs(actual_X, X)
            self.assertIs(actual_y, y)
            calls.append((probe, probe.get_params()))
            result = original_fit(probe, actual_X, actual_y)
            if probe.max_iter == 2000:
                warnings.warn("forced initial convergence warning", ConvergenceWarning)
            return result
        with patch.object(LogisticRegression, "fit", fit):
            probe, record = fit_converged_probe(X, y, 0.01, layer=0)
        self.assertEqual(len(calls), 2)
        self.assertIsNot(calls[0][0], calls[1][0])
        self.assertEqual({k for k in calls[0][1] if calls[0][1][k] != calls[1][1][k]}, {"max_iter"})
        self.assertEqual(probe.max_iter, RETRY_MAX_ITER)
        self.assertTrue(record["retry_needed"] and record["final_converged"])
        self.assertEqual(record["final_max_iter"], 10000)

    def test_selection_scores_only_the_converged_retry(self):
        original_fit, original_decision = LogisticRegression.fit, LogisticRegression.decision_function
        def fit(probe, X, y):
            result = original_fit(probe, X, y)
            if probe.C == 0.001 and probe.max_iter == 2000:
                warnings.warn("forced retry", ConvergenceWarning)
            return result
        def decision(probe, X):
            if probe.C == 0.001:
                self.assertEqual(probe.max_iter, 10000, "nonconverged fit reached validation")
                return original_decision(probe, X)
            return -original_decision(probe, X)
        with patch.object(LogisticRegression, "fit", fit), patch.object(LogisticRegression, "decision_function", decision):
            rows, best, selected = select_probe(self.data.partition("train"), self.data.partition("validation"))
        self.assertEqual((best["layer"], best["C"]), (0, 0.001))
        self.assertTrue(best["retry_needed"])
        self.assertEqual(selected.max_iter, 10000)
        self.assertEqual(sum(r["retry_needed"] for r in rows), 2)
        self.assertEqual(self.data.activation_rows_read["test"], 0)

    def test_warning_after_retry_fails_without_scoring_or_saving(self):
        original_fit = LogisticRegression.fit
        budgets = []
        def fit(probe, X, y):
            budgets.append(probe.max_iter)
            result = original_fit(probe, X, y)
            warnings.warn("forced failure", ConvergenceWarning)
            return result
        directory = self.root / "failed_run"
        with patch.object(LogisticRegression, "fit", fit), \
             patch.object(LogisticRegression, "decision_function", side_effect=AssertionError("scored failed fit")):
            with self.assertRaisesRegex(RuntimeError, "Layer 0, C=0.001.*max_iter=10000"):
                fit_and_save(self.data, directory)
        self.assertEqual(budgets, [2000, 10000])
        self.assertFalse(directory.exists())
        self.assertEqual(self.data.activation_rows_read["test"], 0)

    def test_only_convergence_warnings_trigger_retries(self):
        original_fit = LogisticRegression.fit
        calls = []
        def fit(probe, X, y):
            calls.append(probe.max_iter)
            result = original_fit(probe, X, y)
            probe.n_iter_ = np.array([2000])  # The iteration count alone is not the trigger.
            warnings.warn("unrelated warning", UserWarning)
            return result
        train = self.data.partition("train")
        with patch.object(LogisticRegression, "fit", fit), self.assertWarnsRegex(UserWarning, "unrelated"):
            _, record = fit_converged_probe(train.matrix(0), train.labels, 0.01, layer=0)
        self.assertEqual(calls, [2000])
        self.assertFalse(record["retry_needed"])
        self.assertEqual(record["final_n_iter"], 2000)

    def test_primary_tie_break_ignores_other_diagnostics(self):
        rows = [{"validation_overall_auroc": 0.9, "C": 1.0, "layer": 0, "diagnostic": 1.0},
                {"validation_overall_auroc": 0.9, "C": 0.001, "layer": 5, "diagnostic": 0.5},
                {"validation_overall_auroc": 0.9, "C": 0.001, "layer": 2, "diagnostic": 0.1},
                {"validation_overall_auroc": 0.8, "C": 0.001, "layer": 0, "diagnostic": 1.0}]
        self.assertIs(min(rows, key=selection_key), rows[2])

    def test_wrong_partition_cannot_enter_fitting(self):
        with patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit called")):
            with self.assertRaisesRegex(ValueError, "train and validation"):
                select_probe(self.data.partition("validation"), self.data.partition("train"))

    def test_frozen_artifact_roundtrip_without_test_scoring(self):
        directory = self.root / "out"
        result = fit_and_save(self.data, directory)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["test_decision_scores_computed"], 0)
        self.assertEqual(result["selection_rule"], SELECTION_RULE)
        self.assertFalse((directory / "final_test").exists())
        with patch.object(LogisticRegression, "fit", side_effect=AssertionError("refit")), \
             patch.object(LogisticRegression, "decision_function", side_effect=AssertionError("score")):
            frozen, probe = load_frozen_probe(self.data, directory)
        self.assertEqual(probe.C, result["selected_C"])
        self.assertEqual(frozen["selected_layer"], result["selected_layer"])
        (directory / "selected_probe.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "probe hash"):
            load_frozen_probe(self.data, directory)

    def test_grid_cannot_be_changed(self):
        changed = copy.deepcopy(self.config)
        changed["C_values"] = [0.1]
        with self.assertRaisesRegex(ValueError, "fixed C grid"):
            AtomicData(self.root, changed, "a")

    def test_affirmative_negated_coverage_mismatch_fails(self):
        source = self.root / "source/neg_cities.csv"
        payload = source.read_bytes().replace(b"Fixture-1", b"Fixture-0")
        source.write_bytes(payload)
        (self.root / "acts_a/neg_cities.csv").write_bytes(payload)
        metadata_path = self.root / "manifest_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        for row in metadata["sources"]:
            if row["file"] == "neg_cities.csv":
                row["sha256"] = sha256(payload)
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            AtomicData(self.root, self.config, "a")

    def test_missing_duplicate_manifest_entities_fail(self):
        import csv
        path = self.root / "manifest.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        def update(records):
            payload = csv_bytes(records, FIELDS)
            path.write_bytes(payload)
            metadata_path = self.root / "manifest_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["manifest_sha256"] = sha256(payload)
            metadata_path.write_text(json.dumps(metadata))
        update(rows[1:])
        with self.assertRaisesRegex(ValueError, "absent from manifest"):
            AtomicData(self.root, self.config, "a")
        update(rows + [{**rows[0], "split": "test"}])
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            AtomicData(self.root, self.config, "a")


if __name__ == "__main__":
    unittest.main()
