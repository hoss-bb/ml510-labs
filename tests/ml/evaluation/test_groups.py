"""Aligned group comparisons using arbitrary observations and populations."""
import unittest
import pandas as pd
from ml_lib.evaluation.groups import regression_group_comparisons


class GroupComparisonTests(unittest.TestCase):
    def inputs(self):
        actual = pd.Series([10., 20., 30.], index=["x", "y", "z"])
        predictions = {"reference": pd.Series([8., 18., 28.], index=actual.index),
                       "alternative": pd.Series([30., 10., 20.], index=["z", "x", "y"])}
        groups = {"all": pd.Series(True, index=actual.index),
                  "segment": pd.Series([True, False, True], index=actual.index),
                  "absent": pd.Series(False, index=actual.index)}
        return actual, predictions, groups

    def test_alignment_overlap_counts_and_empty_group(self):
        actual, predictions, groups = self.inputs()
        entities = pd.Series(["c2", "c1", "c1"], index=["z", "x", "y"])
        report = regression_group_comparisons(actual, predictions, baseline="reference",
                                             groups=groups, entity_ids=entities)
        all_rows = report[report.group == "all"].set_index("model")
        self.assertEqual(all_rows.loc["alternative", "mae"], 0)
        self.assertEqual(all_rows.loc["alternative", "mae_reduction_percent"], 100)
        self.assertEqual(all_rows.loc["reference", "entities"], 2)
        self.assertEqual(report[report.group == "segment"].n.tolist(), [2, 2])
        self.assertTrue(report[report.group == "absent"].mae.isna().all())
        self.assertEqual(report[report.group == "absent"].n.tolist(), [0, 0])

    def test_zero_baseline_has_no_relative_improvement(self):
        actual, predictions, groups = self.inputs()
        predictions["reference"] = actual.copy()
        result = regression_group_comparisons(actual, predictions, baseline="reference", groups=groups)
        self.assertTrue(result.mae_reduction_percent.isna().all())

    def test_invalid_identities_values_and_masks(self):
        for bad in [pd.Series([1, 2, 3], index=["x", "x", "z"]),
                    pd.Series([1, 2, 3], index=["x", "y", "other"]),
                    pd.Series([1, float("inf"), 3], index=["x", "y", "z"])]:
            actual, predictions, groups = self.inputs()
            predictions["alternative"] = bad
            with self.assertRaises(ValueError):
                regression_group_comparisons(actual, predictions, baseline="reference", groups=groups)
        actual, predictions, groups = self.inputs()
        groups["invalid"] = pd.Series([1, 0, 1], index=actual.index)
        with self.assertRaises(ValueError):
            regression_group_comparisons(actual, predictions, baseline="reference", groups=groups)
