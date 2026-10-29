import unittest
import numpy as np
from ml_lib.evaluation.resampling import paired_bootstrap


class ResamplingTests(unittest.TestCase):
    def test_pairing(self):
        result = paired_bootstrap([1, 4, 9], [3, 6, 11], n_resamples=50)
        np.testing.assert_allclose(result['draws'], 2)

    def test_reproducibility(self):
        args = ([1, 2, 9], [4, 1, 6])
        np.testing.assert_array_equal(paired_bootstrap(*args)['draws'],
                                      paired_bootstrap(*args)['draws'])

    def test_groups(self):
        result = paired_bootstrap([0, 0, 0], [1, 1, 4], groups=['a', 'a', 'b'], n_resamples=100)
        self.assertEqual(result['units'], 2)
        self.assertEqual(result['difference'], 2)
        self.assertTrue(set(result['draws']).issubset({1., 2., 4.}))

    def test_invalid_inputs(self):
        for kwargs in [{'groups': [1, 1]}, {'groups': [1, None]},
                       {'confidence_level': 1}, {'n_resamples': 2.5}]:
            with self.assertRaises(ValueError):
                paired_bootstrap([1, 2], [2, 3], **kwargs)
        with self.assertRaises(ValueError):
            paired_bootstrap([1, np.nan], [2, 3])
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [3])

    def test_undefined_statistic(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [2, 3], statistic=lambda x: np.nan)
