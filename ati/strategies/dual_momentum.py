"""Dual-momentum swing signal — absolute + relative strength, low-chase entries."""

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


class DualMomentumStrategy(Strategy):
    id = "dual_momentum_v1"
    family = "relative_strength"
    version = "1.1.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOL}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 140:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("ret_63", "atr_14", "sma_50", "sma_200")):
            return None

        # Absolute momentum (~6m)
        abs_mom = float(last["Close"] / d["Close"].iloc[-127] - 1.0) if len(d) >= 127 else float(last["ret_63"])
        if abs_mom <= 0:
            return None

        rs = rs_vs_nifty(d, ctx.nifty_daily, 126) if len(d) >= 130 else rs_vs_nifty(d, ctx.nifty_daily, 63)
        if rs < 0.02:
            return None
        if weekly_bias(ctx.weekly) < 0:
            return None
        if last["Close"] < last["sma_50"]:
            return None
        if is_overextended(d):
            return None

        # Prefer pullback-quality entry inside the RS trend (cost-aware)
        near_value = float(last["Close"]) <= float(last["sma_20"]) * 1.03 if not pd.isna(last.get("sma_20")) else True
        if not near_value and float(last.get("rsi_14", 50) or 50) > 65:
            return None

        stop = atr_stop_long(d, 2.4)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.35 * min(abs_mom / 0.25, 1.4)
            + 0.35 * min(max(rs, 0) / 0.15, 1.4)
            + 0.15 * (1.0 if last["sma_50"] > last["sma_200"] else 0.5)
            + 0.15 * conf
        )

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=float(min(entry, last.get("sma_20", entry) or entry)),
            entry_low=entry * 0.985,
            entry_high=entry * 1.008,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(d["High"].iloc[-20:].max()),
            max_entry=entry * 1.012,
            invalidation=stop,
            holding_period="15-45 trading days",
            entry_justification=(
                f"Dual momentum: abs 6m {abs_mom:.1%}, RS vs NIFTY {rs:.1%}, "
                f"SMA50 filter, non-chase entry"
            ),
            exit_justification="Wider ATR trail; time box 45d; exit if absolute momentum flips negative",
            invalidation_condition=f"Stop {stop:.2f} or absolute momentum <= 0 on weekly close",
            why_good=(
                "Absolute+relative momentum with lower turnover historically survives Indian "
                "transaction costs better than daily breakout chasing."
            ),
            what_could_go_wrong="Momentum crashes / factor reversals; cash gate may miss sharp bottoms.",
            tags={"abs_mom": abs_mom, "rs_nifty": rs, "ltf_conf": conf},
        )
