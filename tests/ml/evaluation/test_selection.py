"""Selection behavior needs no ML backend or real dataset."""
import unittest
from ml_lib.evaluation.selection import select_model


class SelectionTests(unittest.TestCase):
    def test_error_metrics_minimize(self):
        for metric in ("mae", "rmse", "wape"):
            with self.subTest(metric=metric):
                result = select_model({metric: {"weak": 3.0, "strong": 1.0}}, metric)
                self.assertEqual(result["selected_model"], "strong")
                self.assertEqual(result["direction"], "minimize")
                self.assertEqual(result["validation_score"], 1.0)
                self.assertEqual(result["metric"], metric)

    def test_r2_maximizes_including_negative_scores(self):
        result = select_model({"r2": {"weak": -4.0, "strong": -0.2}}, "r2")
        self.assertEqual(result["selected_model"], "strong")
        self.assertEqual(result["direction"], "maximize")

    def test_ties_keep_first_candidate(self):
        for metric in ("mae", "r2"):
            self.assertEqual(select_model({metric: {"first": 1, "second": 1}}, metric)["selected_model"], "first")

    def test_default_is_minimum_mae(self):
        self.assertEqual(select_model({"mae": {"first": 4, "second": 2}})["selected_model"], "second")

    def test_invalid_metric(self):
        with self.assertRaisesRegex(ValueError, "Unsupported selection metric"):
            select_model({}, "mean_residual")

    def test_missing_metric(self):
        with self.assertRaisesRegex(ValueError, "do not include"):
            select_model({}, "mae")

    def test_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "No validation candidates"):
            select_model({"mae": {}})

    def test_undefined_candidate_is_not_silently_dropped(self):
        for value in (None, float("nan"), float("inf"), -float("inf"), "undefined"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Undefined.*bad"):
                select_model({"mae": {"good": 1.0, "bad": value}})
