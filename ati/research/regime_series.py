"""Point-in-time regime classification for backtests (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ati.indicators.technical import adx, sma
from ati.models.opportunity import MarketRegime
from ati.research.gates import universe_breadth


def classify_regime_at(
    nifty_close: pd.Series,
    *,
    i: int,
    panel: pd.DataFrame | None = None,
    vix_close: pd.Series | None = None,
    lookback_ret: int = 63,
    breadth_lookback: int = 21,
) -> MarketRegime:
    """Classify market regime using only data available at bar i."""
    if i < max(lookback_ret, 60) or nifty_close is None or len(nifty_close) <= i:
        return MarketRegime.UNCERTAIN

    close = nifty_close.iloc[: i + 1].astype(float)
    px = float(close.iloc[-1])
    past = float(close.iloc[-1 - lookback_ret])
    if past <= 0 or pd.isna(px) or pd.isna(past):
        return MarketRegime.UNCERTAIN
    nifty_trend = px / past - 1.0

    # Realized vol (20d)
    rets = close.pct_change().dropna()
    vol = float(rets.tail(20).std() * np.sqrt(252)) if len(rets) >= 20 else 0.0

    # ADX on synthetic OHLC from close (index proxy when only Close is aligned)
    try:
        ohlc = pd.DataFrame(
            {"High": close * 1.002, "Low": close * 0.998, "Close": close},
            index=close.index,
        )
        adx_df = adx(ohlc, 14)
        col = "adx" if "adx" in adx_df.columns else adx_df.columns[-1]
        adx_v = float(adx_df[col].iloc[-1]) if len(adx_df) and not pd.isna(adx_df[col].iloc[-1]) else 0.0
    except Exception:  # noqa: BLE001
        adx_v = 0.0

    sma50 = float(sma(close, 50).iloc[-1]) if len(close) >= 50 else float("nan")
    sma200 = float(sma(close, 200).iloc[-1]) if len(close) >= 200 else float("nan")
    ma_structure = 0.0
    if not pd.isna(sma50) and not pd.isna(sma200):
        if px > sma50 > sma200:
            ma_structure = 1.0
        elif px < sma50 < sma200:
            ma_structure = -1.0

    breadth = 0.0
    if panel is not None and i >= breadth_lookback:
        # Map share-up [0,1] → [-1,1] to match live detector scale
        share = universe_breadth(panel, i, breadth_lookback)
        breadth = share * 2.0 - 1.0

    vix_last = 15.0
    if vix_close is not None and len(vix_close) > i:
        v = float(vix_close.iloc[i])
        if not pd.isna(v):
            vix_last = v

    if vix_last >= 22 and vol >= 0.22:
        return MarketRegime.HIGH_VOL
    if nifty_trend > 0.02 and adx_v >= 20 and breadth >= 0.1:
        return MarketRegime.BULL
    if nifty_trend < -0.02 and adx_v >= 20 and breadth <= -0.1:
        return MarketRegime.BEAR
    if adx_v < 18 and abs(nifty_trend) < 0.04:
        return MarketRegime.SIDEWAYS
    if ma_structure > 0 and nifty_trend > 0 and breadth >= 0:
        return MarketRegime.BULL
    if ma_structure < 0 and nifty_trend < 0 and breadth <= 0:
        return MarketRegime.BEAR
    return MarketRegime.UNCERTAIN


def book_for_regime(regime: MarketRegime) -> str:
    """Map regime → active research book. Cash when edge is undefined."""
    if regime == MarketRegime.BULL:
        return "dual_momentum"
    if regime == MarketRegime.SIDEWAYS:
        return "mean_reversion"
    if regime == MarketRegime.HIGH_VOL:
        return "cash"  # defensive: no new risk until vol compresses
    if regime == MarketRegime.BEAR:
        return "cash"  # long-only system stays flat in confirmed bears
    return "cash"
