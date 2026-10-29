import unittest
from tests.ml.fixtures import numeric_hourly
from ml_lib.evaluation.diagnostics import lag_correlations


class DiagnosticTests(unittest.TestCase):
    def test_lag_diagnostic_uses_clock_time(self):
        series=numeric_hourly(20).y.drop(numeric_hourly(20).index[10])
        result=lag_correlations(series,lags=[1])
        self.assertEqual(result.loc[0,"paired_observations"],17)
        self.assertAlmostEqual(result.loc[0,"correlation"],1.)
