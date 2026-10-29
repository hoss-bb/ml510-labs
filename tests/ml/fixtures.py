"""Small numeric data for shared-library tests."""
import numpy as np
import pandas as pd


def numeric_hourly(n=240):
    idx = pd.date_range("2011-01-01", periods=n, freq="h", name="timestamp")
    return pd.DataFrame({"y": np.arange(n, dtype=float), "weather": np.arange(n, dtype=float) / 10,
                         "calendar": np.sin(np.arange(n) * 2 * np.pi / 24)}, index=idx)
