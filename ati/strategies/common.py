"""Shared entry/exit geometry helpers for strategy families."""

from __future__ import annotations

import pandas as pd

from ati.indicators.technical import enrich, relative_strength, swing_high, swing_low


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "atr_14" in df.columns:
        return df
    return enrich(df)


def rs_vs_nifty(daily: pd.DataFrame, nifty: pd.DataFrame, window: int = 63) -> float:
    if daily.empty or nifty.empty:
        return 0.0
    series = relative_strength(daily["Close"], nifty["Close"], window=window).dropna()
    return float(series.iloc[-1]) if len(series) else 0.0


def weekly_bias(weekly: pd.DataFrame) -> float:
    """+1 bullish, -1 bearish, 0 neutral from weekly structure."""
    if weekly is None or len(weekly) < 30:
        return 0.0
    w = enrich(weekly)
    last = w.iloc[-1]
    score = 0.0
    if last["Close"] > last["sma_20"]:
        score += 0.5
    if last["Close"] > last["sma_50"]:
        score += 0.5
    if last["Close"] < last["sma_20"]:
        score -= 0.5
    if last["Close"] < last["sma_50"]:
        score -= 0.5
    return max(-1.0, min(1.0, score))


def lower_tf_confirmation(hourly: pd.DataFrame, m15: pd.DataFrame, direction: str) -> float:
    """Soft confirmation 0..1 — does not require all TFs to agree."""
    score = 0.5
    for frame, weight in ((hourly, 0.3), (m15, 0.2)):
        if frame is None or len(frame) < 30:
            continue
        f = enrich(frame)
        last = f.iloc[-1]
        bullish = last["Close"] > last["ema_21"] and last["rsi_14"] >= 50
        bearish = last["Close"] < last["ema_21"] and last["rsi_14"] <= 50
        if direction == "LONG" and bullish:
            score += weight
        elif direction == "SHORT" and bearish:
            score += weight
        elif direction == "LONG" and bearish:
            score -= weight * 0.5
        elif direction == "SHORT" and bullish:
            score -= weight * 0.5
    return float(max(0.0, min(1.0, score)))


def long_targets(entry: float, stop: float, atr: float) -> tuple[float, float, float]:
    risk = entry - stop
    t1 = entry + max(2 * risk, 1.5 * atr)
    t2 = entry + max(3 * risk, 2.5 * atr)
    t3 = entry + max(4 * risk, 4.0 * atr)
    return t1, t2, t3


def short_targets(entry: float, stop: float, atr: float) -> tuple[float, float, float]:
    risk = stop - entry
    t1 = entry - max(2 * risk, 1.5 * atr)
    t2 = entry - max(3 * risk, 2.5 * atr)
    t3 = entry - max(4 * risk, 4.0 * atr)
    return t1, t2, t3


def atr_stop_long(df: pd.DataFrame, mult: float = 2.0) -> float:
    last = df.iloc[-1]
    swing = swing_low(df, 10)
    atr_stop = float(last["Close"] - mult * last["atr_14"])
    return max(swing, atr_stop) if swing < last["Close"] else atr_stop


def atr_stop_short(df: pd.DataFrame, mult: float = 2.0) -> float:
    last = df.iloc[-1]
    swing = swing_high(df, 10)
    atr_stop = float(last["Close"] + mult * last["atr_14"])
    return min(swing, atr_stop) if swing > last["Close"] else atr_stop


def is_overextended(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    if pd.isna(last.get("atr_14")) or last["atr_14"] <= 0:
        return False
    dist = (last["Close"] - last["sma_20"]) / last["atr_14"]
    return dist > 2.5 or float(last.get("rsi_14", 50)) > 78
