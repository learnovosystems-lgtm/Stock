"""Parquet OHLCV cache for reproducible calculations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ati.config import CACHE_DIR, ensure_dirs


def _path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("/", "_")
    return CACHE_DIR / f"{safe}__{interval}.parquet"


def load_cached(symbol: str, interval: str) -> pd.DataFrame | None:
    ensure_dirs()
    path = _path(symbol, interval)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
    df = df.sort_index()
    return df


def save_cached(symbol: str, interval: str, df: pd.DataFrame) -> Path:
    ensure_dirs()
    path = _path(symbol, interval)
    out = df.copy()
    if out.index.name is None:
        out.index.name = "Date"
    out.to_parquet(path)
    return path


def clear_cache() -> int:
    ensure_dirs()
    n = 0
    for p in CACHE_DIR.glob("*.parquet"):
        p.unlink()
        n += 1
    return n
