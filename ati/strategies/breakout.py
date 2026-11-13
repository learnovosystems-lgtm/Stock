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


class BreakoutStrategy(Strategy):
    id = "breakout_v1"
    family = "breakout"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 60:
            return None
        last = d.iloc[-1]
        prev = d.iloc[-2]
        if any(pd.isna(last[c]) for c in ("dc_high", "atr_14", "vol_surge")):
            return None
        wb = weekly_bias(ctx.weekly)
        if wb < 0:
            return None

        # Volume-confirmed Donchian breakout
        broke = last["Close"] > prev["dc_high"] and last["vol_surge"] >= 1.3
        near = last["Close"] >= 0.985 * last["dc_high"] and last["Close"] <= 1.02 * last["dc_high"]
        if not (broke or near):
            return None

        stop = atr_stop_long(d, 1.8)
        # Prefer swing low under the base
        stop = max(stop, float(d["Low"].iloc[-20:].min()))
        if stop >= last["Close"]:
            stop = float(last["Close"] - 1.5 * last["atr_14"])
        entry = float(last["Close"])
        confirm = float(last["dc_high"])
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.35 * (1.1 if broke else 0.7)
            + 0.35 * min(float(last["vol_surge"]) / 2.0, 1.3)
            + 0.3 * conf
        )

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=confirm,
            entry_low=confirm * 0.995,
            entry_high=confirm * 1.01,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=confirm,
            max_entry=confirm * 1.015,
            invalidation=stop,
            holding_period="3-15 trading days",
            entry_justification=(
                f"Donchian 20 break {'confirmed' if broke else 'approaching'} with "
                f"volume surge {last['vol_surge']:.2f}x"
            ),
            exit_justification="Stop under base; book partial at 1.5-2R; trail remainder",
            invalidation_condition=f"Failed breakout close back inside range / below {stop:.2f}",
            why_good="Volume-confirmed range expansion often initiates swing trends in liquid NSE names.",
            what_could_go_wrong="Bull trap / false breakout common in high-VIX chop.",
            tags={"vol_surge": float(last["vol_surge"]), "broke": broke, "ltf_conf": conf},
        )
