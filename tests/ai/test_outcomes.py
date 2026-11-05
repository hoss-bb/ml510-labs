"""Paired judgments are evidence, not an application replacement decision."""
import unittest
import pandas as pd
from ai_lib.evaluation.reports import paired_outcomes


class PairedOutcomeTests(unittest.TestCase):
    def table(self, left, right):
        return pd.DataFrame([{"case_id": case, "method": method, "passed": verdict, "error": None}
                             for method, values in [("A", left), ("B", right)]
                             for case, verdict in zip(["c1", "c2"], values)])

    def compare(self, table):
        return paired_outcomes(table, expected_cases=["c1", "c2"], baseline="A", candidate="B")

    def test_improvements_regressions_and_ties(self):
        result = self.compare(self.table([True, False], [False, True]).iloc[::-1])
        self.assertEqual(result["improvements"], 1)
        self.assertEqual(result["regressions"], 1)
        self.assertEqual(result["passed_cases"], {"A": 1, "B": 1})
        self.assertTrue(result["complete"])
        self.assertNotIn("selected", result)
        tied = self.compare(self.table([True, False], [True, False]))
        self.assertEqual((tied["improvements"], tied["regressions"]), (0, 0))

    def test_errors_and_missing_judgments_are_incomplete(self):
        table = self.table([True, False], [True, True])
        table.loc[2, "error"] = "timeout"
        result = self.compare(table)
        self.assertFalse(result["complete"])
        self.assertEqual(result["paired_completed"], 1)
        self.assertEqual(result["incomplete_cases"]["B"], ["c1"])
        self.assertEqual(result["passed_cases"]["B"], 1)
        self.assertEqual(result["regressions"], 0)
        self.assertFalse(self.compare(self.table([None, False], [True, True]))["complete"])

    def test_roster_and_boolean_contracts(self):
        table = self.table([True, False], [True, True])
        for bad in [table.iloc[:-1], pd.concat([table, table.iloc[:1]]),
                    table.assign(case_id=["c1", "c2", "c1", "other"]),
                    table.assign(passed=[1, 0, 1, 1])]:
            with self.assertRaises(ValueError):
                self.compare(bad)
