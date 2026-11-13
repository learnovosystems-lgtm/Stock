import numpy as np
import pandas as pd

from ati.research.walkforward import run_rolling_walkforward


def _synth_frames(n_sym=8, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    frames = {}
    for i in range(n_sym):
        rets = rng.normal(0.0004, 0.015, n)
        close = 100 * np.exp(np.cumsum(rets))
        high = close * 1.01
        low = close * 0.99
        open_ = close
        vol = rng.integers(200_000, 800_000, n)
        frames[f"S{i}.NS"] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )
    nifty = frames["S0.NS"].copy()
    sectors = {f"S{i}.NS": ("IT" if i % 2 == 0 else "AUTO") for i in range(n_sym)}
    return frames, nifty, sectors


def test_walkforward_runs_and_returns_report():
    frames, nifty, sectors = _synth_frames()
    report = run_rolling_walkforward(frames, nifty, sectors, n_folds=3, vix_cash_threshold=None)
    assert report.metrics.strategy_id == "dual_momentum_nifty_gated"
    assert isinstance(report.promotable, bool)
    assert report.reason
