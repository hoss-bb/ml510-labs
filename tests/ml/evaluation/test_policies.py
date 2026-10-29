import unittest
import numpy as np
from ml_lib.evaluation.policies import select_threshold, review_policy
from ml_lib.evaluation.classification import decision_metrics


class PolicyTests(unittest.TestCase):
    def test_threshold_constraint(self):
        chosen = select_threshold([1, 0, 1, 0], [.9, .8, .7, .6], max_false_positive_rate=0.)
        self.assertEqual(chosen["threshold"], .9)
        self.assertEqual(chosen["validation_recall"], .5)

    def test_threshold_ties(self):
        chosen = select_threshold([1, 0, 0], [.9, .8, .7], max_false_positive_rate=1.)
        self.assertEqual(chosen["threshold"], .9)

    def test_no_feasible_positive(self):
        chosen = select_threshold([0, 1], [.9, .2], max_false_positive_rate=0.)
        self.assertGreater(chosen["threshold"], 1.)

    def test_capacity_ties(self):
        rows, counts = review_policy([.9, .9, .9, .1], case_ids=[30, 10, 20, 40], threshold=.5, capacity_fraction=.5)
        self.assertEqual(rows.loc[rows.reviewed, "case_id"].tolist(), [10, 20])
        self.assertEqual(counts["over_capacity"], 1)

    def test_score_floor(self):
        rows, counts = review_policy([.9, .1, .1, .1], case_ids=range(4), threshold=.5, capacity_fraction=.5)
        self.assertEqual(counts["reviews"], 1)
        self.assertEqual(decision_metrics([1, 0, 0, 1], rows.reviewed)["fn"], 1)

    def test_empty_capacity(self):
        rows, counts = review_policy([.8, .9], case_ids=[1, 2], threshold=.5, capacity_fraction=0)
        self.assertEqual(counts["over_capacity"], 2)
        self.assertFalse(rows.reviewed.any())

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            review_policy([.1, .2], case_ids=[1, 1], threshold=.5)
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [.1, .2], max_false_positive_rate=np.nan)
