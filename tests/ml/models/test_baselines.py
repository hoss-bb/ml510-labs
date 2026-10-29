import unittest
import numpy as np
from tests.ml.fixtures import numeric_hourly
from ml_lib.models.baselines import naive_forecast, matched_baseline_splits


class BaselineTests(unittest.TestCase):
    def test_next_hour_references(self):
        f = numeric_hourly(100)
        origin = f.index[[47]]
        np.testing.assert_array_equal(naive_forecast(f.y, origin), [47])
        np.testing.assert_array_equal(naive_forecast(f.y, origin, period=24), [24])

    def test_naive_cannot_use_future(self):
        with self.assertRaisesRegex(ValueError, "period"):
            naive_forecast(numeric_hourly().y, [numeric_hourly().index[50]], period=0)


class BaselineCoverageTests(unittest.TestCase):
    def test_common_mask_excludes_missing_reference_and_records_counts(self):
        frame = numeric_hourly(16)
        frame.loc[frame.index[3], "y"] = np.nan
        origins = frame.index[[6, 8, 10, 11]]
        masks = {"train": np.array([True, True, False, False]),
                 "validation": np.array([False, False, True, True])}
        predictions, retained, counts = matched_baseline_splits(
            frame.y, origins, masks, periods={"last": None, "seasonal": 4})
        # Origin 6's next-hour target uses seasonal source hour 3, which is unknown.
        np.testing.assert_array_equal(retained["train"], [False, True, False, False])
        np.testing.assert_array_equal(retained["validation"], masks["validation"])
        for mask in retained.values():
            for values in predictions.values():
                self.assertTrue(np.isfinite(values[mask]).all())
                self.assertEqual(values[mask].shape, (int(mask.sum()),))
        self.assertEqual(counts.set_index("split").loc["train", "dropped_missing_baselines"], 1)
        self.assertTrue((counts.before_baselines == counts.retained + counts.dropped_missing_baselines).all())

    def test_complete_references_preserve_population(self):
        frame = numeric_hourly(12)
        origins = frame.index[[5, 8]]
        original = {"train": np.array([True, False]), "validation": np.array([False, True])}
        predictions, retained, counts = matched_baseline_splits(
            frame.y, origins, original, periods={"last": None, "seasonal": 4})
        np.testing.assert_array_equal(predictions["last"], [5, 8])
        np.testing.assert_array_equal(predictions["seasonal"], [2, 5])
        for split in original:
            np.testing.assert_array_equal(retained[split], original[split])
        self.assertEqual(int(counts.dropped_missing_baselines.sum()), 0)

    def test_empty_common_split_fails_clearly(self):
        frame = numeric_hourly(6)
        with self.assertRaisesRegex(ValueError, "Not enough usable samples"):
            matched_baseline_splits(frame.y, frame.index[[1, 2]],
                {"train": np.array([True, True])}, periods={"seasonal": 4})
