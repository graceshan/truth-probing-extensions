"""Synthetic formulas, orientation, strict split access and selection tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.atomic_probe_methods import (METHODS, FittedMethod, balanced_burger_indices, covariance_mass_mean,
                                     difference_of_means, fit_ttpd, learn_truth_directions)
from src.atomic_method_suite import choose_layers
from src.clean_atomic_method_data import MethodData
from test_clean_atomic_probes import atomic_fixture


def truth_fixture():
    tg, tp = np.array([2., 0., 1.]), np.array([0., 3., 1.])
    X, labels, p, datasets = [], [], [], []
    for topic in range(2):
        for polarity in (-1, 1):
            mu = np.array([100 * topic, 500 * polarity, -30.])
            for tau in (-1, 1):
                X.append(mu + tau * tg + tau * polarity * tp)
                labels.append((tau + 1) // 2)
                p.append(polarity)
                datasets.append(f"{topic}_{polarity}")
    return np.array(X), np.array(labels), np.array(p), np.array(datasets), tg, tp


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_difference_means_formula_and_label_sign(self):
        X = np.array([[0., 0.], [2., 0.], [1., 1.], [3., 3.]])
        y = np.array([0, 0, 1, 1])
        direction = np.array([1., 2.]) / np.sqrt(5)
        fit = difference_of_means(X, y)
        np.testing.assert_allclose(fit.parameters["coef"], direction)
        np.testing.assert_allclose(difference_of_means(X, 1-y).parameters["coef"], -direction)

    def test_invalid_and_degenerate_training_fails(self):
        with self.assertRaises(ValueError):
            difference_of_means(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
        with self.assertRaises(ValueError):
            difference_of_means(np.ones((4, 2)), np.zeros(4))

    def test_l2_readout_matches_sklearn_true_positive_logit(self):
        rng = np.random.RandomState(82)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
        lr = LogisticRegression(C=.01, penalty="l2", max_iter=2000).fit(X, y)
        fit = FittedMethod("l2_logistic", {"coef": lr.coef_[0], "intercept": lr.intercept_[0]}, {})
        np.testing.assert_array_equal(lr.classes_, [0, 1])
        np.testing.assert_allclose(fit.decision_function(X), lr.decision_function(X))
        np.testing.assert_allclose(1 / (1 + np.exp(-fit.decision_function(X))), lr.predict_proba(X)[:, 1])

    def test_mass_mean_full_correlated_covariance_not_diagonal(self):
        rng = np.random.RandomState(84)
        residuals = rng.normal(size=(30, 3)) @ np.array([[2., 1., .2], [.3, 1., .7], [.1, .4, 2.]])
        X = np.concatenate((residuals, residuals + [1., 2., 3.]))
        y = np.repeat([0, 1], 30)
        cov = np.cov(residuals.T, bias=True)
        expected = np.linalg.solve(cov, np.array([1., 2., 3.]))
        fit = covariance_mass_mean(X, y)
        np.testing.assert_allclose(fit.parameters["coef"], expected, atol=1e-12)
        self.assertFalse(np.allclose(expected, np.array([1., 2., 3.]) / np.diag(cov)))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = covariance_mass_mean(X @ rotation, y)
        np.testing.assert_allclose(rotated.decision_function(X @ rotation), fit.decision_function(X), atol=1e-10)

    def test_mass_mean_within_class_covariance_formula(self):
        residuals = np.array([[-2., 0.], [2., 0.], [0., -1.], [0., 1.]])
        X = np.concatenate((residuals, residuals + [1., 1.]))
        y = np.repeat([0, 1], 4)
        fit = covariance_mass_mean(X, y)
        np.testing.assert_allclose(fit.parameters["coef"], [.5, 2.])
        # Total covariance would include the between-class term and is incorrect.
        total_covariance = np.cov(X.T, bias=True)
        self.assertFalse(np.allclose(fit.parameters["coef"], np.linalg.solve(total_covariance, [1., 1.])))
        np.testing.assert_allclose(covariance_mass_mean(X, 1-y).parameters["coef"], [-.5, -2.])
        np.testing.assert_allclose(covariance_mass_mean(X + 1000, y).parameters["coef"], [.5, 2.])

    def test_mass_mean_absolute_cutoff_and_singular_covariance(self):
        residuals = np.array([[-1., -.01, 0.], [-1., .01, 0.], [1., -.01, 0.], [1., .01, 0.]])
        X = np.concatenate((residuals, residuals + 1))
        fit = covariance_mass_mean(X, np.repeat([0, 1], 4))
        np.testing.assert_allclose(fit.parameters["coef"], [1., 0., 0.], atol=1e-12)
        self.assertEqual(fit.details["covariance_rank_retained"], 1)
        with self.assertRaises(ValueError):
            covariance_mass_mean(X, np.repeat([0, 1], 4), atol=0.1)

    def test_truth_directions_ols_dataset_centering_and_sign(self):
        X, y, p, datasets, tg, tp = truth_fixture()
        fit = learn_truth_directions(X, y, p, datasets)
        np.testing.assert_allclose(fit.parameters["t_g"], tg)
        np.testing.assert_allclose(fit.parameters["t_p"], tp)
        np.testing.assert_allclose(fit.decision_function(X), X @ tg)
        neg = learn_truth_directions(X, 1-y, p, datasets)
        np.testing.assert_allclose(neg.parameters["t_g"], -tg)
        np.testing.assert_allclose(neg.parameters["t_p"], -tp)

    def test_truth_directions_unbalanced_labels_use_full_gram(self):
        X, y, p, ds, _, _ = truth_fixture()
        indices = [0, 0, 1, 2, 3, 4, 5, 6, 7]
        X, y, p, ds = X[indices], y[indices], p[indices], ds[indices]
        centered = X.copy()
        for d in np.unique(ds):
            centered[ds == d] -= centered[ds == d].mean(0)
        design = np.column_stack((2*y-1, (2*y-1)*p))
        expected = np.linalg.inv(design.T @ design) @ design.T @ centered
        actual = learn_truth_directions(X, y, p, ds)
        np.testing.assert_allclose(actual.parameters["t_g"], expected[0])
        np.testing.assert_allclose(actual.parameters["t_p"], expected[1])

    def test_polarity_reversal_preserves_general_direction(self):
        X, y, p, ds, tg, tp = truth_fixture()
        actual = learn_truth_directions(X, y, -p, ds)
        np.testing.assert_allclose(actual.parameters["t_g"], tg)
        np.testing.assert_allclose(actual.parameters["t_p"], -tp)

    def test_ttpd_matches_original_two_stage_classifiers(self):
        rng = np.random.RandomState(45)
        X = rng.normal(size=(240, 4))
        p = np.where(X[:, 1] + rng.normal(size=240) > 0, 1, -1)
        y = (X[:, 0] + .5*p + rng.normal(size=240) > 0).astype(int)
        datasets = np.where(p == 1, "affirmative", "negated")
        fit = fit_ttpd(X, y, p, datasets)
        polarity = LogisticRegression(penalty=None, fit_intercept=True).fit(X, (p == 1).astype(int))
        truth = learn_truth_directions(X, y, p, datasets)
        features = np.column_stack((X @ truth.parameters["t_g"], X @ polarity.coef_[0]))
        head = LogisticRegression(penalty=None, fit_intercept=True).fit(features, y)
        np.testing.assert_allclose(fit.decision_function(X), head.decision_function(features), atol=1e-10)
        np.testing.assert_allclose(fit.decision_function(X), X @ fit.parameters["coef"] + fit.parameters["intercept"], atol=1e-10)
        self.assertFalse(np.allclose(fit.parameters["polarity_coef"], fit.parameters["t_p"]))
        np.testing.assert_array_equal(fit.parameters["classes"], [0, 1])
        np.testing.assert_array_equal(fit.parameters["polarity_classes"], [0, 1])
        np.testing.assert_allclose(fit_ttpd(X, 1-y, p, datasets).decision_function(X), -fit.decision_function(X), atol=1e-8)

    def test_parameters_roundtrip_without_pickle(self):
        X, y, p, ds, _, _ = truth_fixture()
        fit = fit_ttpd(X, y, p, ds)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parameters.npz"
            np.savez_compressed(path, **fit.parameters)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(fit.parameters))
                for key in saved.files:
                    np.testing.assert_array_equal(saved[key], fit.parameters[key])

    def test_pandas_dataset_names_are_safe_unicode(self):
        X, y, p, ds, _, _ = truth_fixture()
        # Real pandas string columns arrive as object arrays, unlike the simple
        # NumPy Unicode fixture used by the original round-trip test.
        ds = pd.Series(ds).to_numpy(dtype=object)
        fit = learn_truth_directions(X, y, p, ds)
        self.assertIn(fit.parameters["dataset_names"].dtype.kind, "US")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parameters.npz"
            np.savez_compressed(path, **fit.parameters)
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    self.assertFalse(archive[name].dtype.hasobject)

    def test_selection_fixed_primary_and_independent_secondary(self):
        rows = [{"method": m, "layer": layer, "validation_overall_auroc": score,
                 "validation_topic_macro_auroc": 1-score}
                for m in METHODS for layer, score in [(0, .9), (1, .7), (2, .9)]]
        chosen = choose_layers(rows, 1)
        for method in METHODS:
            self.assertEqual(chosen["primary_fixed_model_layer"][method]["layer"], 1)
            self.assertEqual(chosen["secondary_method_selected_layer"][method]["layer"], 0)


class StrictAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = atomic_fixture(self.root)
        # Corrupt forbidden labels. A reader that validates/parses these as
        # numeric labels must fail; the strict loader must never load them.
        for directory in ("source", "acts_a", "acts_b"):
            for path in (self.root / directory).glob("*.csv"):
                frame = pd.read_csv(path, dtype=str)
                frame.loc[8:, "label"] = "FORBIDDEN_TEST_LABEL"
                frame.to_csv(path, index=False)

    def test_test_labels_skipped_before_load_and_values_never_sliced(self):
        real_read, real_load = pd.read_csv, np.load
        calls = []
        def read(*args, **kwargs):
            if "label" in kwargs.get("usecols", []):
                self.assertIn("skiprows", kwargs)
                self.assertTrue(all(kwargs["skiprows"](line) for line in range(9, 13)))
                calls.append(args[0])
            return real_read(*args, **kwargs)
        def load(*args, **kwargs):
            self.assertEqual(kwargs, {"mmap_mode": "r", "allow_pickle": False})
            arr = real_load(*args, **kwargs)
            class Guard:
                shape, dtype = arr.shape, arr.dtype
                def __getitem__(self, key):
                    self_outer.assertTrue(np.all(np.asarray(key[0]) < 8))
                    return arr[key]
            self_outer = self
            return Guard()
        with patch("pandas.read_csv", side_effect=read), patch("numpy.load", side_effect=load):
            data = MethodData(self.root, self.config, "a")
            for split in ("train", "validation"):
                for layer in range(2):
                    X, digest = data.matrix(split, layer)
                    self.assertTrue(np.isfinite(X).all())
                    self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "TEST"):
                data.matrix("test", 0)
        self.assertEqual(len(calls), 20)
        self.assertEqual(data.provenance["test_labels_loaded"], 0)

    def test_sampling_balanced_paired_deterministic_train_only(self):
        data = MethodData(self.root, self.config, "a")
        train = data.partitions["train"]
        selected = balanced_burger_indices(train)
        np.testing.assert_array_equal(selected, balanced_burger_indices(train))
        sampled = train.iloc[selected]
        self.assertEqual(sampled.groupby("dataset").size().nunique(), 1)
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            balanced_burger_indices(data.partitions["validation"])
        for topic, group in sampled.groupby("topic"):
            a, n = group[group.form == "affirmative"], group[group.form == "negated"]
            np.testing.assert_array_equal(a.row_index, n.row_index)
            np.testing.assert_array_equal(a.label, 1-n.label)

    def test_training_directions_unchanged_by_validation_changes(self):
        data = MethodData(self.root, self.config, "a")
        train_X, _ = data.matrix("train", 0)
        y = data.partitions["train"].label.to_numpy()
        original = difference_of_means(train_X, y).parameters["coef"]
        for path in (self.root / "acts_a").glob("*.npy"):
            arr = np.load(path, mmap_mode="r+")
            arr[4:8] = 999
            arr.flush()
        new_X, _ = data.matrix("train", 0)
        np.testing.assert_array_equal(original, difference_of_means(new_X, y).parameters["coef"])


if __name__ == "__main__":
    unittest.main()
