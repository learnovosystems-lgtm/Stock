from __future__ import annotations

from typing import Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.strategies.base import RawSignal, Strategy, StrategyContext
from ati.strategies.common import (
    atr_stop_long,
    is_overextended,
    long_targets,
    lower_tf_confirmation,
    prepare_daily,
    rs_vs_nifty,
    weekly_bias,
)


class MomentumStrategy(Strategy):
    id = "momentum_v1"
    family = "momentum"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 80:
            return None
        last = d.iloc[-1]
        needed = ("ret_21", "ret_63", "atr_14", "vol_surge", "sma_50")
        if any(pd.isna(last[c]) for c in needed):
            return None

        rs = rs_vs_nifty(d, ctx.nifty_daily)
        wb = weekly_bias(ctx.weekly)
        if last["ret_63"] <= 0.05 or last["ret_21"] <= 0 or rs <= 0:
            return None
        if last["Close"] < last["sma_50"] or wb < 0:
            return None
        if is_overextended(d):
            return None

        stop = atr_stop_long(d, 2.0)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.35 * min(float(last["ret_63"]) / 0.2, 1.5)
            + 0.25 * min(max(rs, 0) / 0.15, 1.5)
            + 0.2 * min(float(last["vol_surge"]) / 1.5, 1.2)
            + 0.2 * conf
        )
        if float(last["vol_surge"]) < 0.9:
            score *= 0.85

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry * 0.998,
            entry_low=entry * 0.99,
            entry_high=entry * 1.005,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(last["dc_high"]) if not pd.isna(last.get("dc_high")) else entry,
            max_entry=entry * 1.01,
            invalidation=stop,
            holding_period="5-20 trading days",
            entry_justification=(
                f"63d momentum {last['ret_63']:.1%}, RS vs NIFTY {rs:.1%}, "
                f"price above SMA50, weekly bias {wb:.1f}"
            ),
            exit_justification=(
                "ATR/swing stop; scale out at T1/T2; exit on momentum deterioration "
                "(RSI cross below 50 / close under EMA21)"
            ),
            invalidation_condition=f"Daily close below {stop:.2f} or RS flips negative vs NIFTY",
            why_good=(
                "Intermediate-term winners with relative strength tend to persist in "
                "bullish Indian regimes when volume participates."
            ),
            what_could_go_wrong="Sudden regime shift, gap through ATR stop, or sector rotation against the name.",
            tags={"rs_nifty": rs, "weekly_bias": wb, "ltf_conf": conf},
        )
