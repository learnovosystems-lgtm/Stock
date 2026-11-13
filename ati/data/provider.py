"""Market data provider — Yahoo Finance for NSE with local cache.

All strategy calculations must run on these frames so signals are reproducible
from cached Parquet when offline.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf

from ati.config import DEFAULT, AppConfig
from ati.data.cache import load_cached, save_cached


OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = _flatten_columns(df)
    keep = [c for c in OHLCV_COLS if c in df.columns]
    out = df[keep].copy()
    for c in OHLCV_COLS:
        if c not in out.columns:
            out[c] = 0.0 if c == "Volume" else pd.NA
    out = out[OHLCV_COLS].dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


class MarketDataProvider:
    """Fetch and cache multi-timeframe OHLCV for Indian equities and indices."""

    def __init__(self, config: AppConfig | None = None, use_cache: bool = True):
        self.config = config or DEFAULT
        self.use_cache = use_cache

    def download(
        self,
        symbol: str,
        interval: str,
        period: str | None = None,
        start: str | None = None,
        end: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        if self.use_cache and not force:
            cached = load_cached(symbol, interval)
            if cached is not None and len(cached) > 0:
                # Refresh if last bar older than 1 trading day for daily+
                age_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - cached.index.max()).total_seconds() / 3600
                max_age = 6 if interval.endswith("m") or interval.endswith("h") else 30
                if age_hours < max_age:
                    return cached

        kwargs: dict = {
            "tickers": symbol,
            "interval": interval,
            "auto_adjust": True,
            "progress": False,
            "threads": False,
        }
        if start:
            kwargs["start"] = start
            if end:
                kwargs["end"] = end
        else:
            kwargs["period"] = period or self._default_period(interval)

        # Retry lightly for transient Yahoo failures
        last_err: Exception | None = None
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
            if last_err and not self.use_cache:
                raise last_err
            cached = load_cached(symbol, interval)
            return cached if cached is not None else pd.DataFrame(columns=OHLCV_COLS)

        df = _normalize(raw)
        if self.use_cache and not df.empty:
            save_cached(symbol, interval, df)
        return df

    def _default_period(self, interval: str) -> str:
        if interval in ("1m", "2m", "5m"):
            return "5d"
        if interval in ("15m", "30m", "60m", "1h", "90m"):
            return f"{self.config.lookback_days_intraday}d"
        if interval in ("1wk", "1w"):
            return "5y"
        return "2y"

    def get_daily(self, symbol: str, force: bool = False) -> pd.DataFrame:
        return self.download(symbol, self.config.market.daily_interval, force=force)

    def get_weekly(self, symbol: str, force: bool = False) -> pd.DataFrame:
        return self.download(symbol, self.config.market.weekly_interval, force=force)

    def get_hourly(self, symbol: str, force: bool = False) -> pd.DataFrame:
        return self.download(symbol, self.config.market.hour_interval, force=force)

    def get_30m(self, symbol: str, force: bool = False) -> pd.DataFrame:
        return self.download(symbol, self.config.market.confirm_interval, force=force)

    def get_15m(self, symbol: str, force: bool = False) -> pd.DataFrame:
        return self.download(symbol, self.config.market.refine_interval, force=force)

    def get_multi_tf(self, symbol: str, force: bool = False) -> dict[str, pd.DataFrame]:
        return {
            "weekly": self.get_weekly(symbol, force=force),
            "daily": self.get_daily(symbol, force=force),
            "1h": self.get_hourly(symbol, force=force),
            "30m": self.get_30m(symbol, force=force),
            "15m": self.get_15m(symbol, force=force),
        }

    def get_many_daily(self, symbols: Iterable[str], force: bool = False) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.get_daily(sym, force=force)
            except Exception:  # noqa: BLE001
                out[sym] = pd.DataFrame(columns=OHLCV_COLS)
            time.sleep(0.05)  # be polite to Yahoo
        return out

    def nifty(self, force: bool = False) -> pd.DataFrame:
        return self.get_daily(self.config.market.nifty_symbol, force=force)

    def india_vix(self, force: bool = False) -> pd.DataFrame:
        return self.get_daily(self.config.market.india_vix_symbol, force=force)
