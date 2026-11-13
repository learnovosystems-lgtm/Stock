import numpy as np
import pandas as pd

from ati.research.gates import (
    drawdown_from_peak,
    india_vix_elevated,
    nifty_above_sma,
    sector_neutral_scores,
    universe_breadth,
)


def test_nifty_above_sma_true_and_false():
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    # Rising series ending above SMA
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    df = pd.DataFrame({"Close": close})
    assert nifty_above_sma(df, window=200) is True

    # Falling series ending below SMA
    close2 = pd.Series(np.linspace(200, 100, len(idx)), index=idx)
    df2 = pd.DataFrame({"Close": close2})
    assert nifty_above_sma(df2, window=200) is False


def test_universe_breadth_and_drawdown():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    panel = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 10),
            "B": np.linspace(100, 90, 10),
            "C": np.linspace(100, 105, 10),
        },
        index=idx,
    )
    b = universe_breadth(panel, 9, lookback=5)
    assert 0.0 <= b <= 1.0
    assert b > 0.5  # A and C up
    # Incremental peak tracking (as used in the live backtest loop)
    peak = 100.0
    for eq in (100.0, 110.0, 99.0):
        dd, peak = drawdown_from_peak([eq], peak)
    assert peak == 110.0
    assert abs(dd - (99 / 110 - 1)) < 1e-9


def test_india_vix_elevated():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    low = pd.DataFrame({"Close": [12, 13, 14, 15, 16]}, index=idx)
    high = pd.DataFrame({"Close": [12, 13, 14, 15, 19]}, index=idx)
    assert india_vix_elevated(low, threshold=18.0) is False
    assert india_vix_elevated(high, threshold=18.0) is True


def test_sector_neutral_scores_subtract_median():
    raw = {"A": 0.20, "B": 0.10, "C": 0.30, "D": 0.05}
    sectors = {"A": "IT", "B": "IT", "C": "AUTO", "D": "AUTO"}
    out = sector_neutral_scores(raw, sectors)
    # IT median = 0.15 → A=0.05, B=-0.05
    assert abs(out["A"] - 0.05) < 1e-9
    assert abs(out["B"] + 0.05) < 1e-9
    # AUTO median = 0.175 → C=0.125, D=-0.125
    assert abs(out["C"] - 0.125) < 1e-9
    assert abs(out["D"] + 0.125) < 1e-9
