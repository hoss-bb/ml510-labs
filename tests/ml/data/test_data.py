import unittest
import pandas as pd
from tests.ml.fixtures import numeric_hourly
from ml_lib.data.validation import require_time_index, hourly_grid


class DataTests(unittest.TestCase):
    def test_hourly_grid_preserves_missing_observations(self):
        frame = numeric_hourly(6)
        missing_time = frame.index[2]
        grid = hourly_grid(frame.drop(index=missing_time))
        pd.testing.assert_index_equal(grid.index, frame.index)
        self.assertTrue(grid.loc[missing_time].isna().all())

    def test_unsorted_rejected(self):
        with self.assertRaises(ValueError): require_time_index(numeric_hourly().iloc[::-1])
