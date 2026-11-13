"""Shared market gates used by backtests and live scanners."""

from __future__ import annotations

import pandas as pd

from ati.indicators.technical import sma


def india_vix_elevated(vix: pd.DataFrame | None, threshold: float = 18.0, at: pd.Timestamp | None = None) -> bool:
    """True when India VIX is at/above threshold — risk-off overlay."""
    if vix is None or vix.empty or "Close" not in vix.columns:
        return False
    close = vix["Close"].astype(float)
    if at is not None:
        close = close.loc[:at]
    if close.empty:
        return False
    last = float(close.iloc[-1])
    return not pd.isna(last) and last >= threshold


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


def universe_breadth(
    panel: pd.DataFrame,
    i: int,
    lookback: int,
) -> float:
    """Share of names with positive lookback return at bar i. Returns 0..1."""
    if i < lookback or panel.empty:
        return 0.0
    ups = 0
    total = 0
    row_now = panel.iloc[i]
    row_past = panel.iloc[i - lookback]
    for sym in panel.columns:
        now = row_now[sym]
        past = row_past[sym]
        if pd.isna(now) or pd.isna(past) or past <= 0:
            continue
        total += 1
        if float(now / past - 1.0) > 0:
            ups += 1
    return ups / total if total else 0.0


def drawdown_from_peak(equity_curve: list[float], peak: float) -> tuple[float, float]:
    """Return (current_drawdown, updated_peak). Drawdown <= 0."""
    if not equity_curve:
        return 0.0, peak
    eq = float(equity_curve[-1])
    peak = max(peak, eq)
    if peak <= 0:
        return 0.0, peak
    return eq / peak - 1.0, peak
