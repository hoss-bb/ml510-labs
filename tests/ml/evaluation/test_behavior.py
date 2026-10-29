"""Aligned prediction comparisons used for explicit behavioural obligations."""
import unittest
import numpy as np
import pandas as pd
from ml_lib.evaluation.behavior import compare_predictions


class BehaviorTests(unittest.TestCase):
    def test_alignment(self):
        a = pd.Series([2., 5.], index=['a', 'b'])
        result = compare_predictions(a, a.iloc[::-1], atol=0, rtol=0)
        self.assertEqual(result.index.tolist(), ['a', 'b'])
        self.assertFalse(result.violated.any())

    def test_tolerance(self):
        a = pd.Series([10., 0., -10.], index=['a', 'b', 'c'])
        b = pd.Series([11., .25, -12.], index=a.index)
        result = compare_predictions(a, b, atol=.25, rtol=.1)
        self.assertEqual(result.index[result.violated].tolist(), ['c'])
        self.assertEqual(result.loc['c', 'difference'], -2.)

    def test_invalid_cases(self):
        a = pd.Series([1., 2.], index=['a', 'b'])
        for b in [pd.Series(dtype=float), pd.Series([1., 2.], index=['a', 'a']),
                  pd.Series([1., 2.], index=['a', None]), pd.Series([1.], index=['a']),
                  pd.Series([1., 2.], index=['a', 'c']), pd.Series([1., np.inf], index=a.index)]:
            with self.subTest(candidate=b):
                with self.assertRaises(ValueError):
                    compare_predictions(a, b, atol=0, rtol=0)

    def test_invalid_tolerances(self):
        a = pd.Series([1.])
        for atol, rtol in [(-1, 0), (0, np.inf), (np.nan, 0), ('small', 0), (None, 0)]:
            with self.assertRaises(ValueError):
                compare_predictions(a, a, atol=atol, rtol=rtol)

    def test_multiindex(self):
        index = pd.MultiIndex.from_tuples([('x', 1), ('x', 2)], names=['origin', 'lead'])
        a = pd.Series([1., 2.], index=index)
        self.assertFalse(compare_predictions(a, a.iloc[::-1], atol=0, rtol=0).violated.any())
