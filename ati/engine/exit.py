"""Dynamic exit engine — never widens stops to avoid realizing a loss."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ati.indicators.technical import atr, enrich, swing_low


class ExitReason(str, Enum):
    NONE = "NONE"
    STOP = "STOP"
    TRAIL = "TRAIL"
    TARGET_1 = "TARGET_1"
    TARGET_2 = "TARGET_2"
    TARGET_3 = "TARGET_3"
    TREND_REVERSAL = "TREND_REVERSAL"
    MOMENTUM_FADE = "MOMENTUM_FADE"
    TIME = "TIME"
    INVALIDATION = "INVALIDATION"


@dataclass
class ExitPlan:
    initial_stop: float
    current_stop: float
    targets: list[float]
    partial_fractions: list[float]
    max_hold_days: int
    notes: str


@dataclass
class ExitAction:
    reason: ExitReason
    price: float
    fraction: float  # 0..1 of position
    new_stop: float | None
    message: str


def build_exit_plan(entry: float, stop: float, t1: float, t2: float, t3: float) -> ExitPlan:
    if stop >= entry:
        raise ValueError("Long stop must be below entry")
    return ExitPlan(
        initial_stop=stop,
        current_stop=stop,
        targets=[t1, t2, t3],
        partial_fractions=[0.4, 0.3, 0.3],
        max_hold_days=30,
        notes="ATR/swing hybrid stop with partial profit booking; stop never widens",
    )


def tighten_stop(current_stop: float, candidate: float, entry: float) -> float:
    """Stops may only move in the protective direction."""
    # For longs, only raise stop
    return max(current_stop, min(candidate, entry * 1.5))


def evaluate_exit(
    *,
    daily: pd.DataFrame,
    entry: float,
    plan: ExitPlan,
    bars_held: int,
    remaining_fraction: float,
    targets_hit: set[int],
) -> ExitAction:
    d = enrich(daily) if "atr_14" not in daily.columns else daily
    last = d.iloc[-1]
    price = float(last["Close"])
    atr_v = float(last["atr_14"]) if not pd.isna(last.get("atr_14")) else price * 0.02

    # Hard stop
    if price <= plan.current_stop:
        return ExitAction(ExitReason.STOP, price, remaining_fraction, plan.current_stop, "Stop-loss hit")

    # Targets
    for i, tgt in enumerate(plan.targets):
        if i in targets_hit:
            continue
        if price >= tgt:
            frac = plan.partial_fractions[i] if i < len(plan.partial_fractions) else remaining_fraction
            frac = min(frac, remaining_fraction)
            # Move stop to breakeven / prior target after T1
            new_stop = plan.current_stop
            if i == 0:
                new_stop = tighten_stop(plan.current_stop, entry, entry)
            elif i >= 1:
                new_stop = tighten_stop(plan.current_stop, plan.targets[i - 1], entry)
            reason = [ExitReason.TARGET_1, ExitReason.TARGET_2, ExitReason.TARGET_3][i]
            return ExitAction(reason, price, frac, new_stop, f"Target {i+1} reached")

    # Trailing ATR stop (only tighten)
    trail = price - 2.2 * atr_v
    swing = swing_low(d, 8)
    candidate = max(trail, swing)
    if candidate > plan.current_stop + 1e-9:
        # Update trail passively; only exit if next bar violates — here we return TRAIL update via fraction 0
        return ExitAction(ExitReason.TRAIL, price, 0.0, candidate, "Trailing stop tightened")

    # Momentum deterioration
    if not pd.isna(last.get("rsi_14")) and last["rsi_14"] < 45 and last["Close"] < last["ema_21"]:
        if bars_held >= 3 and price > entry:
            return ExitAction(
                ExitReason.MOMENTUM_FADE,
                price,
                remaining_fraction,
                plan.current_stop,
                "Momentum deterioration with price under EMA21",
            )

    # Trend reversal
    if not pd.isna(last.get("sma_50")) and last["Close"] < last["sma_50"] and last.get("adx", 0) >= 18:
        if bars_held >= 5:
            return ExitAction(
                ExitReason.TREND_REVERSAL,
                price,
                remaining_fraction,
                plan.current_stop,
                "Close below SMA50 in trending regime",
            )

    if bars_held >= plan.max_hold_days:
        return ExitAction(ExitReason.TIME, price, remaining_fraction, plan.current_stop, "Time-based exit")

    return ExitAction(ExitReason.NONE, price, 0.0, None, "Hold")
