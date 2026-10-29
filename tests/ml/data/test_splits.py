import unittest
import numpy as np
from tests.ml.fixtures import numeric_hourly
from ml_lib.data.splitting import TimeSplit
from ml_lib.features.temporal import make_sequences


class SplitTests(unittest.TestCase):
    def test_target_split_isolation(self):
        f = numeric_hourly(20)
        d = make_sequences(f, history_columns=["y"], target="y", lookback=3)
        policy = TimeSplit(str(f.index[8]), str(f.index[14]))
        masks = policy.masks(d.target_times, origins=d.origins)
        self.assertTrue((d.target_times[masks["train"]] < f.index[8].to_datetime64()).all())
        self.assertTrue((d.target_times[masks["validation"]] >= f.index[8].to_datetime64()).all())
        self.assertEqual(sum(x.sum() for x in masks.values()), len(d.y))

    def test_first_validation_can_use_train_history(self):
        f = numeric_hourly(20)
        d = make_sequences(f, history_columns=["y"], target="y", lookback=4)
        masks = TimeSplit(str(f.index[8]), str(f.index[14])).masks(d.target_times, origins=d.origins)
        valid = d.subset(masks["validation"])
        self.assertEqual(valid.origins[0], f.index[7])
        np.testing.assert_array_equal(valid.history[0,:,0], [4,5,6,7])

    def test_invalid_time_boundaries(self):
        with self.assertRaises(ValueError): TimeSplit("2012-01-01", "2011-01-01")

    def test_missing_boundary(self):
        for missing in (None, "NaT", ""):
            for boundaries in ((missing, "2012-07-01"), ("2012-04-01", missing)):
                with self.subTest(boundaries=boundaries), self.assertRaisesRegex(ValueError, "valid timestamps"):
                    TimeSplit(*boundaries)

    def test_malformed_boundary(self):
        with self.assertRaisesRegex(ValueError, "valid timestamps"):
            TimeSplit("not-a-date", "2012-07-01")
