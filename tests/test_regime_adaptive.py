import numpy as np
import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.research.adaptive import run_mean_reversion_baseline, run_regime_adaptive_book
from ati.research.regime_series import book_for_regime, classify_regime_at


def _synthetic_frames(n: int = 400, n_syms: int = 8) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(42)
    frames = {}
    for i in range(n_syms):
        # Mild drift + noise; a couple of names mean-revert-ish
        drift = 0.0004 if i < 4 else -0.0001
        rets = drift + rng.normal(0, 0.012, size=n)
        close = 100 * np.cumprod(1 + rets)
        frames[f"S{i}.NS"] = pd.DataFrame(
            {
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": rng.integers(200_000, 800_000, size=n),
            },
            index=idx,
        )
    # Rising NIFTY for bull classification most of the time
    n_close = 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.008, size=n))
    nifty = pd.DataFrame(
        {
            "Open": n_close * 0.99,
            "High": n_close * 1.01,
            "Low": n_close * 0.98,
            "Close": n_close,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    return frames, nifty


def test_book_for_regime_mapping():
    assert book_for_regime(MarketRegime.BULL) == "dual_momentum"
    assert book_for_regime(MarketRegime.SIDEWAYS) == "mean_reversion"
    assert book_for_regime(MarketRegime.SIDEWAYS, nifty_above_trend=True) == "mean_reversion"
    assert book_for_regime(MarketRegime.SIDEWAYS, nifty_above_trend=False) == "cash"
    assert book_for_regime(MarketRegime.BEAR) == "cash"
    assert book_for_regime(MarketRegime.HIGH_VOL) == "cash"
    assert book_for_regime(MarketRegime.UNCERTAIN) == "cash"


def test_classify_regime_at_runs():
    _, nifty = _synthetic_frames()
    close = nifty["Close"]
    regime = classify_regime_at(close, i=250)
    assert isinstance(regime, MarketRegime)


def test_mean_reversion_and_adaptive_smoke():
    frames, nifty = _synthetic_frames()
    mr = run_mean_reversion_baseline(frames, rebalance_every=5)
    assert mr.strategy_id == "mean_reversion_baseline"
    result, diag = run_regime_adaptive_book(
        frames,
        nifty,
        lookback=60,
        bull_rebalance=21,
        sideways_rebalance=5,
        nifty_sma_gate=50,
        min_breadth=None,
        max_portfolio_dd=None,
        vix_cash_threshold=None,
    )
    assert result.strategy_id == "regime_adaptive_multi"
    assert isinstance(diag.switches, int)
    assert diag.switches >= 0
