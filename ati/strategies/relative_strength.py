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
    rs_vs_nifty,
    weekly_bias,
)


class RelativeStrengthStrategy(Strategy):
    id = "relative_strength_v1"
    family = "relative_strength"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOL}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 80:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("atr_14", "sma_50", "vol_surge")):
            return None
        rs = rs_vs_nifty(d, ctx.nifty_daily, 63)
        if rs < 0.05:
            return None
        if ctx.sector_strength < -0.02 and rs < 0.12:
            return None
        if weekly_bias(ctx.weekly) < -0.5:
            return None
        if last["Close"] < last["sma_50"]:
            return None

        stop = atr_stop_long(d, 2.0)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.45 * min(rs / 0.2, 1.5)
            + 0.25 * max(0.0, min(1.0, (ctx.sector_strength + 0.05) / 0.1))
            + 0.3 * conf
        )

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry * 0.997,
            entry_low=entry * 0.988,
            entry_high=entry * 1.006,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(d["High"].iloc[-20:].max()),
            max_entry=entry * 1.012,
            invalidation=stop,
            holding_period="5-30 trading days",
            entry_justification=(
                f"RS vs NIFTY {rs:.1%}, sector strength {ctx.sector_strength:.1%}, above SMA50"
            ),
            exit_justification="Exit when RS percentile fades or ATR trail hit",
            invalidation_condition=f"RS turns negative or stop {stop:.2f} hit",
            why_good="Leaders relative to NIFTY and strong sectors historically concentrate upside in Indian swings.",
            what_could_go_wrong="Factor reversal / index catch-up; RS leaders can correct sharply.",
            tags={"rs_nifty": rs, "sector_strength": ctx.sector_strength, "ltf_conf": conf},
        )
