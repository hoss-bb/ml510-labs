import unittest
import numpy as np
import pandas as pd
from dataclasses import replace
from ml_lib.problems.contracts import PredictionContract
from ml_lib.evaluation.reports import evaluate_predictions, comparison_table, reference_comparisons


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.contract=PredictionContract("test","regression","case","target","units","given context","observed context","decision")
        self.times=pd.date_range("2011-01-01",periods=3,freq="h")

    def report(self,name="a",contract=None,times=None,actual=None,fingerprint="same"):
        return evaluate_predictions(np.array(actual if actual is not None else [1,2,3]),np.array([1,2,2]),
            target_times=self.times if times is None else times,contract=contract or self.contract,
            model_name=name,split="validation",dataset_fingerprint=fingerprint)

    def test_valid_comparison(self):
        self.assertEqual(len(comparison_table([self.report(),self.report("b")])),2)

    def test_different_contract_rejected(self):
        with self.assertRaisesRegex(ValueError,"Incompatible"):
            comparison_table([self.report(),self.report("b",contract=replace(self.contract,name="different"))])

    def test_different_population_rejected(self):
        with self.assertRaises(ValueError): comparison_table([self.report(),self.report("b",times=self.times+pd.Timedelta(hours=1))])

    def test_different_labels_rejected(self):
        with self.assertRaises(ValueError): comparison_table([self.report(),self.report("b",actual=[2,3,4])])

    def test_different_source_rejected(self):
        with self.assertRaises(ValueError): comparison_table([self.report(),self.report("b",fingerprint="other")])

    def test_slice_counts(self):
        table=self.report().by("hour")
        self.assertEqual(table.n.sum(),3)

    def test_forecast_timestamps_enforced(self):
        contract=replace(self.contract,task="forecasting")
        with self.assertRaisesRegex(ValueError,"Forecast targets"):
            evaluate_predictions([1,2,3],[1,2,3],target_times=self.times,origins=self.times,
                contract=contract,model_name="bad",split="test",dataset_fingerprint="x")

    def test_reference_error_reduction(self):
        reference = self.report("reference")
        candidate = self.report("candidate")
        candidate.predictions["prediction"] = candidate.predictions.actual
        for metric in ("mae", "rmse", "wape"):
            with self.subTest(metric=metric):
                table = reference_comparisons([reference, candidate], {"simple": "reference"}, metric)
                row = table.set_index("model").loc["candidate"]
                self.assertAlmostEqual(row.improvement, reference.metrics[metric])
                self.assertAlmostEqual(row.relative_error_reduction, 1.0)

    def test_reference_r2_difference(self):
        reference = self.report("reference")
        candidate = self.report("candidate")
        candidate.predictions["prediction"] = candidate.predictions.actual
        row = reference_comparisons([reference, candidate], {"simple": "reference"}, "r2").iloc[1]
        self.assertAlmostEqual(row.improvement, 1.0 - reference.metrics["r2"])
        self.assertIsNone(row.relative_error_reduction)

    def test_zero_reference_error(self):
        report = self.report()
        report.predictions["prediction"] = report.predictions.actual
        row = reference_comparisons([report], {"simple": "a"}, "mae").iloc[0]
        self.assertEqual(row.improvement, 0)
        self.assertIsNone(row.relative_error_reduction)

    def test_reference_population_guard(self):
        for other in (self.report("b", times=self.times + pd.Timedelta(hours=1)),
                      self.report("b", actual=[2, 3, 4]),
                      self.report("b", fingerprint="other")):
            with self.subTest(other=other.model_name), self.assertRaises(ValueError):
                reference_comparisons([self.report(), other], {"simple": "a"}, "mae")

    def test_reference_metric_guard(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            reference_comparisons([self.report()], {"simple": "a"}, "mape")
        with self.assertRaisesRegex(ValueError, "Undefined"):
            reference_comparisons([self.report(actual=[0, 0, 0])], {"simple": "a"}, "wape")

    def test_missing_reference(self):
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            reference_comparisons([self.report()], {"simple": "missing"}, "mae")
