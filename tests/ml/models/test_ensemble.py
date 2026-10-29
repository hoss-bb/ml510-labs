"""An ensemble must average compatible fitted models."""
import unittest
from types import SimpleNamespace
import numpy as np
from ml_lib.models.ensemble import SequenceEnsemble


class EnsembleTests(unittest.TestCase):
    def test_member_count(self):
        with self.assertRaises(ValueError): SequenceEnsemble([])

    def test_unfitted_member(self):
        member = SimpleNamespace(model=None, signature=None)
        with self.assertRaises(ValueError): SequenceEnsemble([member, member])

    def test_feature_signature(self):
        a = SimpleNamespace(model=object(), signature={"history_names": ["x"]})
        b = SimpleNamespace(model=object(), signature={"history_names": ["y"]})
        with self.assertRaises(ValueError): SequenceEnsemble([a, b])

    def test_equal_average(self):
        a = SimpleNamespace(model=object(), signature={}, predict=lambda _: np.array([2., 10.]))
        b = SimpleNamespace(model=object(), signature={}, predict=lambda _: np.array([6., 4.]))
        np.testing.assert_array_equal(SequenceEnsemble([a, b]).predict(None), [4., 7.])
