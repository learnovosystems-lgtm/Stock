"""Higher-timeframe supply/demand zones projected to lower-TF entry context.

Demand = rally-base-rally (or drop-base-rally) swing base.
Supply = drop-base-drop (or rally-base-drop) swing base.
Longs prefer price in/near HTF demand and not pressing into HTF supply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Zone:
    kind: str  # demand | supply
    timeframe: str
    low: float
    high: float
    mid: float
    strength: float
    formed_at: str

    def contains(self, price: float, pad: float = 0.002) -> bool:
        return self.low * (1 - pad) <= price <= self.high * (1 + pad)

    def distance_pct(self, price: float) -> float:
        if self.contains(price):
            return 0.0
        if price < self.low:
            return price / self.low - 1.0
        return price / self.high - 1.0


@dataclass
class ZoneContext:
    weekly_demand: list[Zone]
    weekly_supply: list[Zone]
    daily_demand: list[Zone]
    daily_supply: list[Zone]
    entry_bias: str  # long_ok | wait_pullback | avoid
    why: str
    nearest_demand: Zone | None
    nearest_supply: Zone | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_bias": self.entry_bias,
            "why": self.why,
            "nearest_demand": asdict(self.nearest_demand) if self.nearest_demand else None,
            "nearest_supply": asdict(self.nearest_supply) if self.nearest_supply else None,
            "weekly_demand": [asdict(z) for z in self.weekly_demand[:3]],
            "weekly_supply": [asdict(z) for z in self.weekly_supply[:3]],
            "daily_demand": [asdict(z) for z in self.daily_demand[:3]],
            "daily_supply": [asdict(z) for z in self.daily_supply[:3]],
        }


def _bases(df: pd.DataFrame, timeframe: str, lookback: int = 80) -> tuple[list[Zone], list[Zone]]:
    if df is None or len(df) < 30:
        return [], []
    d = df.tail(lookback).copy()
    demand: list[Zone] = []
    supply: list[Zone] = []
    highs = d["High"].astype(float).values
    lows = d["Low"].astype(float).values
    closes = d["Close"].astype(float).values
    idx = d.index
    # 3-bar fractal bases
    for i in range(2, len(d) - 2):
        # Demand: local low with reclaim
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1] and closes[i + 2] > closes[i]:
            lo = float(min(lows[i - 1 : i + 2]))
            hi = float(max(highs[i - 1 : i + 2]))
            rng = hi - lo
            if rng <= 0 or rng / closes[i] > 0.08:
                continue
            strength = min(1.5, (closes[i + 2] / closes[i] - 1.0) * 10)
            demand.append(
                Zone(
                    kind="demand",
                    timeframe=timeframe,
                    low=round(lo, 2),
                    high=round(hi, 2),
                    mid=round((lo + hi) / 2, 2),
                    strength=float(max(0.2, strength)),
                    formed_at=str(pd.Timestamp(idx[i]).date()),
                )
            )
        # Supply: local high with rejection
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1] and closes[i + 2] < closes[i]:
            lo = float(min(lows[i - 1 : i + 2]))
            hi = float(max(highs[i - 1 : i + 2]))
            rng = hi - lo
            if rng <= 0 or rng / closes[i] > 0.08:
                continue
            strength = min(1.5, (closes[i] / max(closes[i + 2], 1e-9) - 1.0) * 10)
            supply.append(
                Zone(
                    kind="supply",
                    timeframe=timeframe,
                    low=round(lo, 2),
                    high=round(hi, 2),
                    mid=round((lo + hi) / 2, 2),
                    strength=float(max(0.2, strength)),
                    formed_at=str(pd.Timestamp(idx[i]).date()),
                )
            )
    demand = sorted(demand, key=lambda z: z.formed_at, reverse=True)[:5]
    supply = sorted(supply, key=lambda z: z.formed_at, reverse=True)[:5]
    return demand, supply


def build_zone_context(
    daily: pd.DataFrame,
    weekly: pd.DataFrame | None,
    price: float,
) -> ZoneContext:
    wd, ws = _bases(weekly, "1W") if weekly is not None and len(weekly) else ([], [])
    dd, ds = _bases(daily, "1D")
    # Prefer nearer demand below / supply above
    demands = wd + dd
    supplies = ws + ds
    nearest_demand = None
    nearest_supply = None
    if demands:
        below = [z for z in demands if z.high <= price * 1.01]
        nearest_demand = min(below or demands, key=lambda z: abs(z.mid - price))
    if supplies:
        above = [z for z in supplies if z.low >= price * 0.99]
        nearest_supply = min(above or supplies, key=lambda z: abs(z.mid - price))

    in_weekly_demand = any(z.contains(price) for z in wd)
    in_daily_demand = any(z.contains(price) for z in dd)
    into_supply = any(z.contains(price) or (0 < z.distance_pct(price) < 0.01) for z in (ws + ds))

    if into_supply and not (in_weekly_demand or in_daily_demand):
        bias = "avoid"
        why = "Price pressing HTF/LTF supply — wait for acceptance or pullback to demand"
    elif in_weekly_demand or in_daily_demand:
        bias = "long_ok"
        tf = "weekly" if in_weekly_demand else "daily"
        why = f"Price in {tf} demand zone — higher-TF location supports long entry"
    elif nearest_demand and nearest_demand.distance_pct(price) > -0.03:
        bias = "wait_pullback"
        why = (
            f"Wait pullback into demand {nearest_demand.low:.0f}-{nearest_demand.high:.0f} "
            f"({nearest_demand.timeframe})"
        )
    else:
        bias = "wait_pullback"
        why = "No clear demand support nearby — defer aggressive entry"

    return ZoneContext(
        weekly_demand=wd,
        weekly_supply=ws,
        daily_demand=dd,
        daily_supply=ds,
        entry_bias=bias,
        why=why,
        nearest_demand=nearest_demand,
        nearest_supply=nearest_supply,
    )


def refine_entry_ranges(
    *,
    buy: float,
    stop: float,
    sell: float,
    zones: ZoneContext,
) -> dict[str, Any]:
    """Produce entry/exit/SL ranges (not single ticks) for safer execution."""
    entry_low = buy * 0.992
    entry_high = buy * 1.008
    if zones.nearest_demand and zones.entry_bias in ("long_ok", "wait_pullback"):
        entry_low = min(entry_low, zones.nearest_demand.low)
        entry_high = min(entry_high, max(buy, zones.nearest_demand.high))
    sl_low = stop * 0.995
    sl_high = stop * 1.005
    if zones.nearest_demand:
        sl_low = min(sl_low, zones.nearest_demand.low * 0.995)
    exit_low = sell * 0.99
    exit_high = sell * 1.02
    if zones.nearest_supply:
        exit_high = min(exit_high, zones.nearest_supply.low)
        exit_low = min(exit_low, zones.nearest_supply.low * 0.995)
    return {
        "entry_range": [round(entry_low, 2), round(entry_high, 2)],
        "exit_range": [round(exit_low, 2), round(exit_high, 2)],
        "sl_range": [round(sl_low, 2), round(sl_high, 2)],
        "ideal_buy": round(buy, 2),
        "ideal_sell": round(sell, 2),
        "ideal_stop": round(stop, 2),
    }
