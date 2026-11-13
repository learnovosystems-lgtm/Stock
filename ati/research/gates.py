"""Shared market gates used by backtests and live scanners."""

from __future__ import annotations

import pandas as pd

from ati.indicators.technical import sma


def nifty_above_sma(nifty: pd.DataFrame | None, window: int = 200, at: pd.Timestamp | None = None) -> bool:
    """True when NIFTY close is above its SMA — risk-on gate."""
    if nifty is None or nifty.empty or "Close" not in nifty.columns:
        return False
    close = nifty["Close"].astype(float)
    if at is not None:
        close = close.loc[:at]
    if len(close) < window:
        return False
    ma = sma(close, window)
    last_close = float(close.iloc[-1])
    last_ma = float(ma.iloc[-1])
    if pd.isna(last_ma):
        return False
    return last_close > last_ma


def sector_neutral_scores(
    raw_scores: dict[str, float],
    sectors: dict[str, str] | None,
) -> dict[str, float]:
    """Subtract sector median momentum so rankings are not pure sector bets."""
    if not sectors or not raw_scores:
        return raw_scores
    by_sector: dict[str, list[float]] = {}
    for sym, score in raw_scores.items():
        sec = sectors.get(sym, "UNKNOWN")
        by_sector.setdefault(sec, []).append(score)
    medians = {sec: float(pd.Series(vals).median()) for sec, vals in by_sector.items()}
    return {
        sym: score - medians.get(sectors.get(sym, "UNKNOWN"), 0.0)
        for sym, score in raw_scores.items()
    }
