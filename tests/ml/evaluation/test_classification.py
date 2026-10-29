"""Probability reports retain their case identities."""
import unittest
from io import StringIO
import numpy as np
import pandas as pd
from ml_lib.problems.classification import ClassificationProblem
from ml_lib.problems.contracts import PredictionContract
from ml_lib.evaluation.classification import classification_metrics, decision_metrics, evaluate_classification, classification_comparison, reliability_table
from ml_lib.evaluation.selection import select_model


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.contract = PredictionContract("example", "classification", "case", "label", "0 or 1", "before action", "observed inputs", "review")
        self.y = [0, 0, 1, 1]
        self.p = [.1, .2, .7, .8]

    def report(self, name="model", **kwargs):
        values = dict(y_true=self.y, probability=self.p, case_ids=[10, 11, 12, 13], contract=self.contract,
                      model_name=name, split="validation", dataset_fingerprint="abc")
        values.update(kwargs)
        return evaluate_classification(**values)

    def test_probability_metrics(self):
        m = classification_metrics(self.y, self.p)
        self.assertEqual(m["average_precision"], 1.)
        self.assertEqual(m["roc_auc"], 1.)
        self.assertEqual(m["positives"], 2)
        for p in [[0.2], [0, 0, 1, np.nan], [0, 0, 1, 2], [[.1], [.2], [.7], [.8]]]:
            with self.assertRaises(ValueError):
                classification_metrics(self.y, p)

    def test_undefined_metrics(self):
        m = classification_metrics([0, 0], [.1, .2])
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["average_precision"])
        self.assertIsNone(decision_metrics([0, 0], [0, 0])["precision"])

    def test_selection_direction(self):
        table = pd.DataFrame({"average_precision": [.2, .8], "log_loss": [.8, .2]}, index=["a", "b"])
        for metric in table:
            self.assertEqual(select_model(table, metric)["selected_model"], "b")

    def test_matching_cases(self):
        first = self.report()
        self.assertEqual(len(classification_comparison([first, self.report("other")])), 2)
        restored = self.report("restored")
        restored.predictions = pd.read_csv(StringIO(restored.predictions.to_csv(index=False)))
        restored.predictions.index += 100
        self.assertEqual(len(classification_comparison([first, restored])), 2)
        for change in [{"case_ids": [11, 10, 12, 13]}, {"y_true": [0, 1, 1, 1]}, {"dataset_fingerprint": "other"}]:
            with self.assertRaises(ValueError):
                classification_comparison([first, self.report("other", **change)])

    def test_case_identity(self):
        with self.assertRaises(ValueError):
            self.report(case_ids=[10, 10, 12, 13])
        with self.assertRaises(ValueError):
            self.report(metadata=pd.DataFrame({"slice": [1]*4}, index=[13, 12, 11, 10]))
        with self.assertRaisesRegex(ValueError, "Observed labels"):
            self.report(y_true=pd.Series(self.y, index=[13, 12, 11, 10]))

    def test_problem_alignment(self):
        X = pd.DataFrame({"x": range(4)}, index=[10, 11, 12, 13])
        metadata = pd.DataFrame(index=X.index)
        y = pd.Series(self.y, index=X.index)
        ClassificationProblem(X, y, metadata, self.contract)
        with self.assertRaisesRegex(ValueError, "identical ordered case IDs"):
            ClassificationProblem(X, y.iloc[::-1], metadata, self.contract)

    def test_reliability_counts(self):
        table = reliability_table(self.y, [0, .2, .7, 1.], bins=5)
        self.assertEqual(table.n.sum(), 4)
        self.assertEqual(table.positives.sum(), 2)

    def test_constrained_recall(self):
        report = self.report(probability=[.8, .1, .9, .7], max_false_positive_rate=0.)
        score = report.metrics
        self.assertEqual(score["recall_at_fpr"], .5)
        self.assertEqual(score["threshold"], .9)
        self.assertEqual(score["achieved_fpr"], 0.)
        self.assertEqual(score["precision"], 1.)
        self.assertEqual(score["reviews"], 1)
        with self.assertRaisesRegex(ValueError, "validation only"):
            self.report(split="test", max_false_positive_rate=.05)
        with self.assertRaises(ValueError):
            _ = self.report(y_true=[0, 0, 0, 0], max_false_positive_rate=.05).metrics

    def test_comparison_limits(self):
        with self.assertRaisesRegex(ValueError, "same false-positive-rate limit"):
            classification_comparison([self.report(max_false_positive_rate=.01),
                                      self.report("other", max_false_positive_rate=.05)])
        with self.assertRaisesRegex(ValueError, "same false-positive-rate limit"):
            classification_comparison([self.report(), self.report("other", max_false_positive_rate=.05)])

    def test_constrained_selection(self):
        table = pd.DataFrame({"recall_at_fpr": [.2, .3, .3],
                              "max_false_positive_rate": [.05]*3}, index=["a", "b", "c"])
        selected = select_model(table, "recall_at_fpr")
        self.assertEqual(selected["selected_model"], "b")
        self.assertEqual(selected["direction"], "maximize")
        self.assertEqual(selected["max_false_positive_rate"], .05)
        table.loc["c", "max_false_positive_rate"] = .01
        with self.assertRaisesRegex(ValueError, "same finite"):
            select_model(table, "recall_at_fpr")
