"""Synthetic repaired caches only; no real data or real 140-fit selection."""
import ast
import builtins
from contextlib import ExitStack
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src import clean_atomic_probes as shared
from src import pinned_atomic_probes as pinned
from src import repaired_atomic_cache as cache_validation


class PinnedAtomicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / pinned.CACHE
        self.cache.mkdir(parents=True)
        records = []
        for topic in pinned.TOPICS:
            for form in ("affirmative", "negated"):
                for split, offset in (("train", 0), ("validation", 2)):
                    for label in (0, 1):
                        records.append(dict(zip(pinned.FIELDS, (
                            ("neg_" if form == "negated" else "") + topic, offset + label,
                            f"Synthetic {topic}, {split}, {label}.\nUnicode é.",
                            f"entity_{topic}_{split}", topic, form, split, label))))
        self.parts = {s: [r for r in records if r["split"] == s] for s in ("train", "validation")}
        for name, value in (("COUNTS", {"train": 20, "validation": 20}), ("WIDTH", 3),
                            ("INPUT_DIGEST", pinned.digest(pinned.canonical(records)))):
            for module in (pinned, cache_validation):
                patcher = patch.object(module, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
        for split, rows in self.parts.items():
            (self.cache / split).mkdir()
            self.write_metadata(split, rows)
            values = np.zeros((20, 28, 3), dtype=np.float16)
            values[:, :, 0] = np.asarray([2 * r["label"] - 1 for r in rows])[:, None]
            values[:, :, 1] = 10 if split == "train" else 20
            np.save(self.cache / split / "activations.npy", values)
        self.manifest = {
            "schema_version": 1, "implementation": "qwen25-pinned-atomic-repair-v1",
            "contract": pinned.contract(), "activation_shapes": {s: [20, 28, 3] for s in self.parts},
            "resolved_model": {"model_resolved_revision": pinned.REVISION, "tokenizer_resolved_revision": pinned.REVISION},
            "runtime": {"torch": "2.11.0+cu128", "transformers": "5.12.1"},
            "input": {"allowed_rows_sha256": pinned.INPUT_DIGEST, "counts": pinned.COUNTS,
                      "columns": list(pinned.FIELDS), "partition_rows_sha256": {
                          s: pinned.digest(pinned.canonical(r)) for s, r in self.parts.items()},
                      "path": "data/tiu_datasets/forbidden.csv"},
            "smoke": {"passed": True, "batch_size": 1, "padding": False,
                      "policy": "unpadded-single-repeat-and-independent-forward-exact-v1"},
            # These links must never be followed; all needed evidence is in-cache.
            "smoke_report": {"path": "compound_data/forbidden.json"},
        }
        self.refresh_receipt()

    def write_metadata(self, split, rows, fields=None):
        with (self.cache / split / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or pinned.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def refresh_receipt(self):
        manifest_path = self.cache / "extraction_manifest.json"
        manifest_path.write_bytes(pinned.canonical(self.manifest))
        self.completion = {"complete": True, "counts": dict(pinned.COUNTS),
                           "manifest_sha256": pinned.file_hash(manifest_path), "outputs": {
                               name: {"sha256": pinned.file_hash(self.cache / name),
                                      "bytes": (self.cache / name).stat().st_size} for name in pinned.OUTPUT_FILES}}
        self.save_completion()

    def save_completion(self):
        (self.cache / "completion.json").write_text(json.dumps(self.completion))

    def io_guard(self, stack):
        """Only six cache files, new results, and Python implementation sources."""
        permitted = {self.cache / n for n in ("completion.json", "extraction_manifest.json", *pinned.OUTPUT_FILES)}
        opened = []
        def check(path):
            if isinstance(path, int):
                return
            path = Path(os.fsdecode(path)).absolute()
            if path.is_relative_to(self.root):
                self.assertTrue(path in permitted or path.is_relative_to(self.root / pinned.RESULTS),
                                f"forbidden data access: {path}")
                opened.append(path)
            elif path.is_relative_to(pinned.ROOT):
                self.assertIn(path.suffix, (".py", ".pyc"), f"forbidden real repo data access: {path}")
        for module in (builtins, io, os):
            original = module.open
            def guarded(path, *args, _original=original, **kwargs):
                check(path)
                return _original(path, *args, **kwargs)
            stack.enter_context(patch.object(module, "open", guarded))
        stack.enter_context(patch.object(shared, "AtomicData", side_effect=AssertionError("old loader called")))
        return opened

    def test_check_only_validates_without_fitting_or_loading_matrices(self):
        with ExitStack() as stack:
            opened = self.io_guard(stack)
            stack.enter_context(patch.object(np, "load", side_effect=AssertionError("matrix materialized")))
            stack.enter_context(patch.object(shared, "fit_and_save", side_effect=AssertionError("fitting")))
            result = pinned.run(check_only=True, root=self.root)
        self.assertEqual(set(opened), {self.cache / n for n in
                         ("completion.json", "extraction_manifest.json", *pinned.OUTPUT_FILES)})
        self.assertEqual(result["fits_performed"], 0)
        self.assertEqual(result["activation_rows_read"], {"train": 0, "validation": 0, "test": 0})
        self.assertFalse((self.root / pinned.RESULTS).exists())

    def test_partition_float64_exact_alignment_and_no_test_api(self):
        with ExitStack() as stack:
            self.io_guard(stack)
            data = pinned.RepairedAtomicData(self.root)
            self.assertIs(data._authorize_test, False)
            for split, marker in (("train", 10), ("validation", 20)):
                part = data.partition(split)
                matrix = part.matrix(4)
                self.assertEqual(matrix.dtype, np.float64)
                self.assertEqual(matrix.shape, (20, 3))
                np.testing.assert_array_equal(matrix[:, 0], 2 * part.labels - 1)
                self.assertTrue(np.all(matrix[:, 1] == marker))
                self.assertEqual(part.topics.tolist(), [r["topic"] for r in self.parts[split]])
            for split in ("test", "development", "../test"):
                with self.assertRaises(PermissionError):
                    data.partition(split)

    def test_actual_shared_140_fit_grid_and_tie_break_on_small_synthetic_cache(self):
        fit, decision = LogisticRegression.fit, LogisticRegression.decision_function
        calls = []
        def spy_fit(probe, X, y):
            self.assertTrue(np.all(X[:, 1] == 10))
            self.assertEqual(X.dtype, np.float64)
            for name, expected in shared.PROBE_CONFIG.items():
                self.assertEqual(probe.get_params()[name], expected)
            calls.append(probe.C)
            return fit(probe, X, y)
        def spy_decision(probe, X):
            self.assertTrue(np.all(X[:, 1] == 20))
            return decision(probe, X)
        with ExitStack() as stack:
            self.io_guard(stack)
            spies = {name: stack.enter_context(patch.object(shared, name, wraps=getattr(shared, name)))
                     for name in ("fit_and_save", "select_probe", "fit_converged_probe", "diagnostics", "selection_key")}
            stack.enter_context(patch.object(LogisticRegression, "fit", spy_fit))
            stack.enter_context(patch.object(LogisticRegression, "decision_function", spy_decision))
            # Silence the routine progress output for the synthetic grid.
            stack.enter_context(patch("sys.stdout", new=io.StringIO()))
            result = pinned.run(root=self.root)
        self.assertEqual(calls, shared.C_VALUES * 28)
        self.assertEqual(spies["fit_and_save"].call_count, 1)
        self.assertEqual(spies["select_probe"].call_count, 1)
        for name in ("fit_converged_probe", "diagnostics"):
            self.assertEqual(spies[name].call_count, 140)
        self.assertGreaterEqual(spies["selection_key"].call_count, 139)
        self.assertEqual((result["selected_layer"], result["selected_C"]), (0, 0.001))
        self.assertEqual(result["selection_rule"], shared.SELECTION_RULE)
        self.assertEqual(result["optimization_policy"], shared.OPTIMIZATION_POLICY)
        self.assertEqual(result["activation_rows_read"], {"train": 560, "validation": 560, "test": 0})
        self.assertFalse(result["test_evaluated"])
        saved = json.loads((self.root / pinned.RESULTS / "selection.json").read_text())
        for name, record in saved["structural"]["repaired_cache_files"].items():
            self.assertEqual(record["sha256"], pinned.file_hash(self.cache / name))
        self.assertEqual(set(saved["structural"]["adapter_code_sha256"]), set(pinned.ADAPTER_SOURCES))
        with self.assertRaisesRegex(ValueError, "existing pinned selection"):
            pinned.run(root=self.root)

    def test_selection_implementation_is_imported_not_duplicated(self):
        self.assertIs(pinned.selection, shared)
        tree = ast.parse(Path(pinned.__file__).read_text())
        definitions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        reused = {"diagnostics", "selection_key", "fit_converged_probe", "select_probe", "fit_and_save",
                  "C_VALUES", "PROBE_CONFIG", "RETRY_MAX_ITER", "OPTIMIZATION_POLICY", "SELECTION_RULE"}
        assignments = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        self.assertFalse(reused & (definitions | assignments))

    def test_shared_convergence_retry_with_repaired_partition(self):
        part = pinned.RepairedAtomicData(self.root).partition("train")
        X, y = part.matrix(0), part.labels
        original = LogisticRegression.fit
        for fail_retry in (False, True):
            with self.subTest(fail_retry=fail_retry):
                budgets = []
                def fit(probe, actual_X, actual_y):
                    self.assertIs(actual_X, X)
                    self.assertIs(actual_y, y)
                    budgets.append(probe.max_iter)
                    result = original(probe, actual_X, actual_y)
                    if probe.max_iter == shared.PROBE_CONFIG["max_iter"] or fail_retry:
                        warnings.warn("synthetic convergence warning", ConvergenceWarning)
                    return result
                with patch.object(LogisticRegression, "fit", fit):
                    if fail_retry:
                        with self.assertRaises(RuntimeError):
                            shared.fit_converged_probe(X, y, 0.001, layer=0)
                    else:
                        _, result = shared.fit_converged_probe(X, y, 0.001, layer=0)
                        self.assertTrue(result["retry_needed"] and result["final_converged"])
                self.assertEqual(budgets, [2000, shared.RETRY_MAX_ITER])

    def test_missing_completion_rejected(self):
        (self.cache / "completion.json").unlink()
        with self.assertRaises(FileNotFoundError):
            pinned.run(check_only=True, root=self.root)

    def test_completion_failures_stop_before_fit(self):
        original = json.loads(json.dumps(self.completion))
        for failure in ("incomplete", "counts", "manifest_hash", "output_hash", "output_size", "extra_path", "missing_path"):
            with self.subTest(failure=failure):
                self.completion = json.loads(json.dumps(original))
                if failure == "incomplete":
                    self.completion["complete"] = False
                elif failure == "counts":
                    self.completion["counts"]["train"] += 1
                elif failure == "manifest_hash":
                    self.completion["manifest_sha256"] = "0" * 64
                elif failure in ("output_hash", "output_size"):
                    record = self.completion["outputs"]["train/activations.npy"]
                    record["sha256" if failure == "output_hash" else "bytes"] = "0" * 64 if failure == "output_hash" else 0
                elif failure == "extra_path":
                    self.completion["outputs"]["../../../../data/tiu_datasets/forbidden.csv"] = {}
                else:
                    del self.completion["outputs"]["validation/metadata.csv"]
                self.save_completion()
                with ExitStack() as stack:
                    self.io_guard(stack)
                    fit = stack.enter_context(patch.object(shared, "fit_and_save", side_effect=AssertionError("fit")))
                    with self.assertRaises(ValueError):
                        pinned.run(root=self.root)
                    fit.assert_not_called()

    def test_manifest_convention_and_revision_failures(self):
        original = json.loads(json.dumps(self.manifest))
        mutations = [("contract", key, value) for key, value in (
            ("model_revision", "b" * 40), ("tokenizer_revision", "b" * 40),
            ("batch_size", 4), ("padding", True), ("compute_dtype", "float32"),
            ("attention", "eager"), ("use_cache", True), ("explicit_semantic_position_ids", False),
            ("saved_dtype", "float32"), ("selected_layer", 17))]
        mutations += [("resolved_model", "model_resolved_revision", "b" * 40),
                      ("resolved_model", "tokenizer_resolved_revision", "b" * 40),
                      ("input", "allowed_rows_sha256", "0" * 64), ("smoke", "passed", False),
                      ("runtime", "torch", "2.8.0+cu128"), ("runtime", "transformers", "5.0.0")]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                self.manifest = json.loads(json.dumps(original))
                self.manifest[section][key] = value
                self.refresh_receipt()
                with self.assertRaises(ValueError):
                    pinned.RepairedAtomicData(self.root)

    def test_bad_metadata_even_with_recomputed_output_hashes(self):
        original_manifest = json.loads(json.dumps(self.manifest))
        for failure in ("test_split", "wrong_split", "duplicate", "count", "label", "single_class", "overlap", "statement", "schema"):
            with self.subTest(failure=failure):
                self.manifest = json.loads(json.dumps(original_manifest))
                rows = [dict(r) for r in self.parts["train"]]
                fields = None
                if failure in ("test_split", "wrong_split"):
                    rows[0]["split"] = "test" if failure == "test_split" else "validation"
                elif failure == "duplicate":
                    rows[1] = dict(rows[0])
                elif failure == "count":
                    rows.pop()
                elif failure == "label":
                    rows[0]["label"] = 2
                elif failure == "single_class":
                    for row in rows:
                        row["label"] = 0
                elif failure == "overlap":
                    rows[0]["entity_id"] = self.parts["validation"][0]["entity_id"]
                elif failure == "statement":
                    rows[0]["statement"] += " changed"
                else:
                    fields = list(pinned.FIELDS) + ["unexpected"]
                # Bind changed metadata to manifest, so semantic/digest checks must catch it.
                self.manifest["input"]["partition_rows_sha256"]["train"] = pinned.digest(pinned.canonical(rows))
                self.write_metadata("train", rows, fields)
                self.refresh_receipt()
                with self.assertRaises(ValueError):
                    pinned.RepairedAtomicData(self.root)

    def test_shape_dtype_and_trailing_bytes_rejected(self):
        path = self.cache / "train/activations.npy"
        for failure in ("rows", "layers", "width", "dtype", "trailing"):
            with self.subTest(failure=failure):
                shape = {"rows": (19, 28, 3), "layers": (20, 27, 3), "width": (20, 28, 4)}.get(failure, (20, 28, 3))
                np.save(path, np.zeros(shape, dtype=np.float32 if failure == "dtype" else np.float16))
                if failure == "trailing":
                    with path.open("ab") as handle:
                        handle.write(b"extra")
                self.refresh_receipt()
                with self.assertRaises(ValueError):
                    pinned.RepairedAtomicData(self.root)

    def test_symlink_cannot_redirect_to_historical_or_test_file(self):
        for name in ("completion.json", "extraction_manifest.json", *pinned.OUTPUT_FILES):
            with self.subTest(name=name):
                path = self.cache / name
                payload = path.read_bytes()
                path.unlink()
                path.symlink_to(self.root / "acts/historical_or_test.npy")
                with self.assertRaisesRegex(ValueError, "symlink"):
                    pinned.RepairedAtomicData(self.root)
                path.unlink()
                path.write_bytes(payload)

    def test_pinned_cli_has_no_test_or_path_override_option(self):
        cli = pinned.ROOT / "scripts/34_select_pinned_atomic_probe.py"
        for option in ("--evaluate-test", "--config", "--model"):
            with self.subTest(option=option):
                result = subprocess.run([sys.executable, str(cli), option], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
