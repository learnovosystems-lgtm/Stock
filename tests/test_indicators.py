import numpy as np
import pandas as pd

from ati.indicators.technical import atr, enrich, rsi


def _synth(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = close + rng.normal(0, 0.2, n)
    vol = rng.integers(100000, 500000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_enrich_reproducible():
    df = _synth()
    a = enrich(df)
    b = enrich(df)
    pd.testing.assert_frame_equal(a, b)
    assert a["atr_14"].notna().sum() > 50
    assert ((a["rsi_14"].dropna() >= 0) & (a["rsi_14"].dropna() <= 100)).all()


def test_atr_positive():
    df = _synth()
    a = atr(df).dropna()
    assert (a > 0).all()
