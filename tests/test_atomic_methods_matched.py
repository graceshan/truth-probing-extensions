"""Matched row identity, fitting inputs and parameter persistence controls."""
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import atomic_methods_matched as matched
from src.atomic_probe_methods import FittedMethod, METHODS


def fixture():
    rng = np.random.RandomState(52)
    n = 80
    p = np.tile([-1, 1], n // 2)
    y = rng.randint(0, 2, n)
    X = rng.normal(size=(n, 5))
    X[:, 0] += .7 * (2*y-1)
    X[:, 1] += .8 * p
    rows = pd.DataFrame({"dataset": np.where(p == 1, "cities", "neg_cities"),
                         "row_index": np.arange(n), "statement": [f"row {i}" for i in range(n)],
                         "entity_id": [f"entity {i}" for i in range(n)], "topic": "cities",
                         "form": np.where(p == 1, "affirmative", "negated"), "split": "train", "label": y})
    return X, rows


class MatchedControlTests(unittest.TestCase):
    def test_exact_saved_order_resolved_without_resampling(self):
        _, train = fixture()
        saved = train.iloc[[31, 8, 54, 2]].copy()
        indices = matched.resolve_training_subset(train, saved, matched.rows_hash(saved), 4)
        np.testing.assert_array_equal(indices, [31, 8, 54, 2])
        self.assertEqual(matched.row_ids(train.iloc[indices]), matched.row_ids(saved))

    def test_altered_labels_order_and_duplicates_rejected(self):
        _, train = fixture()
        saved = train.iloc[:10].copy()
        digest = matched.rows_hash(saved)
        changed = saved.copy()
        changed.loc[0, "label"] = 1 - changed.loc[0, "label"]
        for rows in (changed, saved.iloc[::-1], pd.concat([saved.iloc[:9], saved.iloc[:1]])):
            with self.assertRaises(ValueError):
                matched.resolve_training_subset(train, rows, digest, 10)
        changed_train = train.copy()
        changed_train.loc[0, "statement"] = "changed source statement"
        with self.assertRaisesRegex(ValueError, "contents"):
            matched.resolve_training_subset(changed_train, saved, digest, 10)

    def test_nontrain_and_unknown_rows_rejected(self):
        _, train = fixture()
        for split in ("validation", "test"):
            saved = train.iloc[:10].copy()
            saved["split"] = split
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                matched.resolve_training_subset(train, saved, matched.rows_hash(saved), 10)
        saved = train.iloc[:10].copy()
        saved.loc[0, "row_index"] = 9000
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            matched.resolve_training_subset(train, saved, matched.rows_hash(saved), 10)

    def test_all_five_methods_receive_identical_training_arrays(self):
        X, rows = fixture()
        functions = ["fit_converged_probe", "difference_of_means", "covariance_mass_mean",
                     "learn_truth_directions", "fit_ttpd"]
        from contextlib import ExitStack
        with threadpool_limits(limits=1), ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(matched, name, wraps=getattr(matched, name))) for name in functions]
            fits = matched.fit_matched_methods(X, rows, layer=17, C=.01, covariance_atol=.001)
        self.assertEqual(tuple(fit.method for fit in fits), METHODS)
        for call in mocks:
            self.assertEqual(call.call_count, 1)
            self.assertIs(call.call_args.args[0], X)
            np.testing.assert_array_equal(call.call_args.args[1], rows.label.to_numpy())
        self.assertEqual(mocks[0].call_args.args[2], .01)
        self.assertEqual(mocks[0].call_args.kwargs["layer"], 17)
        for fit in fits:
            self.assertEqual(fit.details["fit_rows"], len(rows))
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **fit.parameters)
            with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as archive:
                restored = FittedMethod(fit.method, {key: archive[key] for key in archive.files}, {})
            np.testing.assert_array_equal(restored.decision_function(X), fit.decision_function(X))

    def test_fit_refuses_validation_rows(self):
        X, rows = fixture()
        rows["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            matched.fit_matched_methods(X, rows, layer=17, C=.01, covariance_atol=.001)


if __name__ == "__main__":
    unittest.main()
