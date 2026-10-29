import unittest
import numpy as np
import pandas as pd
from tests.ml.fixtures import numeric_hourly
from ml_lib.features.temporal import lag_features, make_sequences


class TemporalTests(unittest.TestCase):
    def test_irregular_positional_lag_rejected(self):
        frame = numeric_hourly(20).drop(numeric_hourly(20).index[4])
        with self.assertRaisesRegex(ValueError, "complete hourly"):
            lag_features(frame, "y", [1])

    def test_shift_is_target_relative(self):
        frame = numeric_hourly(20)
        out = lag_features(frame, "y", [1, 3], [3])
        self.assertEqual(out.iloc[8].y_lag_1, 7)
        self.assertEqual(out.iloc[8].y_lag_3, 5)
        self.assertEqual(out.iloc[8].y_mean_3, 6)

    def test_same_target_value_cannot_change_its_input(self):
        frame = numeric_hourly(20)
        before = lag_features(frame, "y", [1, 3], [3]).iloc[10]
        frame.iloc[10, frame.columns.get_loc("y")] = 999999
        after = lag_features(frame, "y", [1, 3], [3]).iloc[10]
        pd.testing.assert_series_equal(before, after)

    def test_next_hour_alignment(self):
        d = make_sequences(numeric_hourly(10), history_columns=["y", "weather"], future_columns=["calendar"], target="y", lookback=3)
        self.assertEqual(d.history.shape, (7, 3, 2))
        np.testing.assert_array_equal(d.history[0, :, 0], [0, 1, 2])
        self.assertEqual(d.y[0], 3)
        self.assertEqual(d.future.shape, (7, 1))
        self.assertEqual(d.target_times[0], numeric_hourly(10).index[3].to_datetime64())

    def test_incomplete_windows_are_not_imputed(self):
        f = numeric_hourly(10)
        f.loc[f.index[4], "y"] = np.nan
        d = make_sequences(f, history_columns=["y"], target="y", lookback=3)
        self.assertEqual(d.coverage["candidate_windows"], 7)
        self.assertEqual(d.coverage["retained_windows"], 3)
        self.assertTrue(np.isfinite(d.y).all())

    def test_future_target_forbidden(self):
        with self.assertRaisesRegex(ValueError, "future target"):
            make_sequences(numeric_hourly(), history_columns=["weather"], future_columns=["y"], target="y")

    def test_empty_windows_have_valid_shapes(self):
        d = make_sequences(numeric_hourly(5), history_columns=["y"], target="y", lookback=20)
        self.assertEqual(d.history.shape, (0, 20, 1))
        self.assertEqual(d.y.shape, (0,))
