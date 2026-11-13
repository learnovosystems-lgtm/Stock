import numpy as np
import pandas as pd

from ati.research.gates import nifty_above_sma, sector_neutral_scores


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
