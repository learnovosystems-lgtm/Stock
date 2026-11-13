from __future__ import annotations

from typing import Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.strategies.base import RawSignal, Strategy, StrategyContext
from ati.strategies.common import (
    atr_stop_long,
    long_targets,
    lower_tf_confirmation,
    prepare_daily,
    weekly_bias,
)


class TrendFollowingStrategy(Strategy):
    id = "trend_following_v1"
    family = "trend_following"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.BEAR}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 220:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("sma_50", "sma_200", "adx", "atr_14", "ema_21")):
            return None
        wb = weekly_bias(ctx.weekly)
        bullish_stack = last["Close"] > last["sma_50"] > last["sma_200"] and last["adx"] >= 20
        if not bullish_stack or wb <= 0:
            return None
        # Pullback into rising EMA21 without breaking structure
        near_ema = abs(last["Close"] - last["ema_21"]) <= 0.75 * last["atr_14"]
        if not near_ema:
            return None

        stop = atr_stop_long(d, 2.2)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = 0.4 * min(float(last["adx"]) / 40, 1.2) + 0.3 * (1 if bullish_stack else 0) + 0.3 * conf

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=float(last["ema_21"]),
            entry_low=float(last["ema_21"] - 0.3 * last["atr_14"]),
            entry_high=entry * 1.004,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(d["High"].iloc[-20:].max()),
            max_entry=entry * 1.012,
            invalidation=stop,
            holding_period="10-40 trading days",
            entry_justification=f"SMA50>SMA200 stack, ADX {last['adx']:.1f}, pullback to EMA21",
            exit_justification="Trail with ATR; exit on close below SMA50 or ADX collapse with MA break",
            invalidation_condition=f"Close below {stop:.2f} or loss of SMA50/200 stack",
            why_good="Trend pullbacks offer asymmetric entries inside established higher-timeframe direction.",
            what_could_go_wrong="Trend exhaustion / distribution at resistance after shallow pullback.",
            tags={"adx": float(last["adx"]), "ltf_conf": conf},
        )
