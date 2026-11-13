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


class PullbackStrategy(Strategy):
    id = "pullback_v1"
    family = "pullback"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 80:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("sma_20", "sma_50", "rsi_14", "atr_14", "ret_63")):
            return None
        if weekly_bias(ctx.weekly) < 0:
            return None
        uptrend = last["sma_20"] > last["sma_50"] and last["ret_63"] > 0.03
        pullback = last["Close"] <= last["sma_20"] * 1.01 and last["rsi_14"] < 48
        reclaim = last["Close"] >= last["ema_21"] * 0.995
        if not (uptrend and pullback and reclaim):
            return None

        stop = atr_stop_long(d, 2.0)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = 0.4 + 0.3 * (1 - float(last["rsi_14"]) / 100) + 0.3 * conf

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=float(min(entry, last["sma_20"])),
            entry_low=float(last["sma_20"] - 0.4 * last["atr_14"]),
            entry_high=entry * 1.006,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(d["High"].iloc[-15:].max()),
            max_entry=entry * 1.01,
            invalidation=stop,
            holding_period="5-25 trading days",
            entry_justification=(
                f"Uptrend pullback to SMA20 with RSI {last['rsi_14']:.1f} and EMA21 reclaim"
            ),
            exit_justification="ATR stop under swing; targets at prior high / extension; exit if swing low breaks",
            invalidation_condition=f"Break of pullback swing low at {stop:.2f}",
            why_good="Buying strength on controlled dips improves entry quality vs chasing breakouts.",
            what_could_go_wrong="Pullback becomes trend reversal; support fails with rising volume.",
            tags={"rsi": float(last["rsi_14"]), "ltf_conf": conf},
        )
