"""Reproducible technical indicators — pure pandas/numpy, no opaque vendor TA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, window=1)  # true range raw via atr helper with window1 ewm — replace
    prev_close = close.shift(1)
    tr_raw = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_n = tr_raw.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean() / atr_n
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_v = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return pd.DataFrame({"adx": adx_v, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + n_std * std,
            "bb_lower": mid - n_std * std,
            "bb_width": (2 * n_std * std) / mid.replace(0, np.nan),
        }
    )


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dc_high": df["High"].rolling(window, min_periods=window).max(),
            "dc_low": df["Low"].rolling(window, min_periods=window).min(),
        },
        index=df.index,
    )


def relative_strength(stock_close: pd.Series, bench_close: pd.Series, window: int = 63) -> pd.Series:
    aligned = pd.concat([stock_close.rename("s"), bench_close.rename("b")], axis=1).dropna()
    rs = aligned["s"] / aligned["b"].replace(0, np.nan)
    return (rs / rs.shift(window) - 1.0).reindex(stock_close.index)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    rets = np.log(close / close.shift(1))
    return rets.rolling(window, min_periods=window).std() * np.sqrt(252)


def swing_low(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["Low"].iloc[-lookback:].min())


def swing_high(df: pd.DataFrame, lookback: int = 10) -> float:
    return float(df["High"].iloc[-lookback:].max())


def volume_surge(volume: pd.Series, window: int = 20) -> pd.Series:
    base = volume.rolling(window, min_periods=window).mean()
    return volume / base.replace(0, np.nan)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard feature set used across strategies."""
    out = df.copy()
    out["sma_20"] = sma(out["Close"], 20)
    out["sma_50"] = sma(out["Close"], 50)
    out["sma_200"] = sma(out["Close"], 200)
    out["ema_9"] = ema(out["Close"], 9)
    out["ema_21"] = ema(out["Close"], 21)
    out["rsi_14"] = rsi(out["Close"], 14)
    out["atr_14"] = atr(out, 14)
    adx_df = adx(out, 14)
    out = out.join(adx_df)
    out = out.join(bollinger(out["Close"]))
    out = out.join(macd(out["Close"]))
    out = out.join(donchian(out, 20))
    out["vol_surge"] = volume_surge(out["Volume"], 20)
    out["rvol_20"] = realized_vol(out["Close"], 20)
    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_21"] = out["Close"].pct_change(21)
    out["ret_63"] = out["Close"].pct_change(63)
    return out
