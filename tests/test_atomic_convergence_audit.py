"""Optimization-audit isolation and replacement-rule tests on synthetic data."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.atomic_convergence_audit import AUDIT_MAX_ITER, refit_warned, replacement_selection
from src.clean_atomic_probes import PROBE_CONFIG
from src.entity_partitions import TOPICS


class ConvergenceAuditTests(unittest.TestCase):
    def test_only_warned_refit_and_only_iteration_budget_changes(self):
        labels = np.tile([0, 1], 10)
        topics = np.repeat(TOPICS, 4)
        forms = np.tile(["affirmative", "affirmative", "negated", "negated"], 5)
        def partition(split, marker):
            return SimpleNamespace(split=split, entity_ids={split}, labels=labels,
                                   topics=topics, forms=forms,
                                   matrix=lambda layer: np.column_stack((2 * labels - 1, np.full(20, marker))))
        rows = [{"layer": 0, "C": 10.0, "n_iter": 2000, "validation_overall_auroc": 0.7,
                 "convergence_warning": True},
                {"layer": 1, "C": 0.01, "n_iter": 20, "validation_overall_auroc": 1.0,
                 "convergence_warning": False}]
        original_fit, original_decision = LogisticRegression.fit, LogisticRegression.decision_function
        calls = []
        def fit(probe, X, y):
            self.assertEqual(probe.max_iter, AUDIT_MAX_ITER)
            for name, value in PROBE_CONFIG.items():
                if name != "max_iter":
                    self.assertEqual(probe.get_params()[name], value)
            self.assertEqual(probe.C, 10.0)
            self.assertTrue(np.all(X[:, 1] == 10))
            calls.append(1)
            return original_fit(probe, X, y)
        def decision(probe, X):
            self.assertTrue(np.all(X[:, 1] == 20))
            return original_decision(probe, X)
        with patch.object(LogisticRegression, "fit", fit), patch.object(LogisticRegression, "decision_function", decision):
            audits, _ = refit_warned(partition("train", 10), partition("validation", 20), rows, PROBE_CONFIG)
        self.assertEqual(len(calls), 1)
        self.assertTrue(audits[0]["refit_converged"])
        self.assertEqual(audits[0]["refit_validation_auroc"], 1.0)

    def test_replacement_preserves_unresolved_and_uses_original_tie_rule(self):
        original = [{"layer": 0, "C": 1.0, "validation_overall_auroc": 0.8},
                    {"layer": 1, "C": 0.01, "validation_overall_auroc": 0.9},
                    {"layer": 2, "C": 10.0, "validation_overall_auroc": 0.7}]
        refits = [{"layer": 0, "C": 1.0, "refit_validation_auroc": 0.9, "refit_converged": True},
                  {"layer": 2, "C": 10.0, "refit_validation_auroc": 1.0, "refit_converged": False}]
        result = replacement_selection(original, refits)
        self.assertFalse(result["selected_configuration_changed"])
        self.assertFalse(result["all_warned_fits_converged"])
        self.assertEqual(result["unresolved_configurations"], [{"layer": 2, "C": 10.0}])
        self.assertEqual(original[0]["validation_overall_auroc"], 0.8)
        refits[1]["refit_converged"] = True
        self.assertTrue(replacement_selection(original, refits)["selected_configuration_changed"])

    def test_test_partition_is_rejected_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "train/validation"):
            refit_warned(SimpleNamespace(split="train"), SimpleNamespace(split="test"), [], PROBE_CONFIG)


if __name__ == "__main__":
    unittest.main()
