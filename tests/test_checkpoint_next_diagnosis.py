import unittest
import numpy as np
from src.checkpoint_next_score_diagnosis import cross_auc, decompose, confusion
from src.checkpoint_r2_selection_v1 import TOPICS


class EdgeCases(unittest.TestCase):
    def test_ties_and_zero_class_mass(self):
        score = np.array([0., 1., 0., 0.])
        weights = np.array([[1, 1, 1, 1], [0, 0, 1, 2]])
        v, _, _ = cross_auc(score, [0, 1], [2, 3], weights)
        self.assertEqual(v[0], .75)
        self.assertTrue(np.isnan(v[1]))

    def test_zero_topic_mass_keeps_pooled_but_invalidates_fixed_macro(self):
        rows = [dict(topic=t, label=y, operator='OR', truth_a=y, truth_b=0) for t in TOPICS for y in (0, 1)]
        score = np.tile([0., 1.], 5)
        weights = np.ones((2, 10)); weights[0, :2] = 0; weights[1, 1::2] = 0
        result = decompose(rows, score, weights)
        self.assertEqual(result['values'][1, 0], 1.)
        self.assertTrue(np.isnan(result['values'][1, 1]))
        self.assertTrue(np.isnan(result['matrix'][1, 0, 0]))
        self.assertEqual(result['contribution'][1, 0, 0], 0.)
        self.assertTrue(np.isnan(result['values'][2, 0]))

    def test_single_class_balanced_accuracy_undefined(self):
        self.assertIsNone(confusion(np.array([1, 1]), np.array([True, False]))['balanced_accuracy'])

    def test_positive_scaling_ranking_signs_and_relative_scale_composition(self):
        score = np.array([[1., 3.], [2., 1.]])
        scaled = score * np.array([4., 1.])
        np.testing.assert_array_equal(score >= 0, scaled >= 0)
        for h in (0, 1): np.testing.assert_array_equal(np.argsort(score[:, h]), np.argsort(scaled[:, h]))
        self.assertFalse(np.array_equal(np.argsort(score.max(axis=1)), np.argsort(scaled.max(axis=1))))


if __name__ == '__main__': unittest.main()
