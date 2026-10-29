import unittest
import numpy as np
import pandas as pd
from ml_lib.features.windows import make_windows, partition_windows


class WindowTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'a':np.arange(40.)}, index=pd.date_range('2020-01-01', periods=40, freq='min'))

    def test_alignment(self):
        w = make_windows(self.frame(), frequency='1min', steps=10)
        self.assertEqual(w.values.shape, (4,10,1))
        np.testing.assert_array_equal(w.values[0,:,0], np.arange(10))
        self.assertEqual(w.ends[0], pd.Timestamp('2020-01-01 00:10'))

    def test_gaps(self):
        f = self.frame().drop(pd.Timestamp('2020-01-01 00:05'))
        w = make_windows(f, frequency='1min', steps=10)
        self.assertEqual(w.coverage['dropped_windows'],1)
        self.assertEqual(w.values[0,0,0],10)

    def test_partitions(self):
        w = make_windows(self.frame(), frequency='1min', steps=10)
        m = partition_windows(w, {'train':('2020-01-01','2020-01-01 00:20'), 'test':('2020-01-01 00:20','2020-01-02')})
        self.assertEqual(m['train'].sum(),2)
        with self.assertRaises(ValueError):
            partition_windows(w, {'train':('2020-01-01','2020-01-01 00:25'), 'test':('2020-01-01 00:25','2020-01-02')})

    def test_invalid_times(self):
        f = self.frame().iloc[::-1]
        with self.assertRaises(ValueError):
            make_windows(f, frequency='1min', steps=10)
