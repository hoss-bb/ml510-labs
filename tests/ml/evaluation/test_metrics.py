import unittest
import numpy as np
from ml_lib.evaluation.metrics import regression_metrics


class MetricTests(unittest.TestCase):
    def test_known_metrics_and_direction(self):
        m=regression_metrics(np.array([0.,2.,4.]),np.array([1.,2.,2.]))
        self.assertEqual(m["mae"],1.)
        self.assertAlmostEqual(m["rmse"],np.sqrt(5/3))
        self.assertAlmostEqual(m["mean_residual"],1/3)

    def test_shape_broadcasting_rejected(self):
        with self.assertRaises(ValueError): regression_metrics([1,2],[[1],[2]])
        with self.assertRaises(ValueError): regression_metrics([[1],[2]],[[1],[2]])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError): regression_metrics([],[])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): regression_metrics([1,np.nan],[1,2])

    def test_undefined_scores_are_none(self):
        m=regression_metrics([0,0],[1,1])
        self.assertIsNone(m["r2"]); self.assertIsNone(m["wape"])

    def test_negative_predictions_visible(self):
        self.assertEqual(regression_metrics([1,2],[-1,2])["negative_prediction_rate"],.5)
