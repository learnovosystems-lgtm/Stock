"""Dual-momentum swing signal — absolute + relative strength, NIFTY-gated."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.research.gates import india_vix_elevated, nifty_above_sma
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
    id = "dual_momentum_v2"
    family = "relative_strength"
    version = "2.1.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BULL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 140:
            return None
        # Hard index trend gate — do not take new risk when NIFTY < SMA200
        if not nifty_above_sma(ctx.nifty_daily, window=200):
            return None
        # India VIX overlay — cash bias when fear elevated
        vix = ctx.extras.get("india_vix")
        if vix is not None and india_vix_elevated(vix, threshold=18.0):
            return None

        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("ret_63", "atr_14", "sma_50", "sma_200")):
            return None

        abs_mom = float(last["Close"] / d["Close"].iloc[-85] - 1.0) if len(d) >= 85 else float(last["ret_63"])
        if abs_mom <= 0:
            return None

        rs = rs_vs_nifty(d, ctx.nifty_daily, 84)
        if rs < 0.02:
            return None
        if weekly_bias(ctx.weekly) < 0:
            return None
        if last["Close"] < last["sma_50"]:
            return None
        if is_overextended(d):
            return None

        near_value = (
            float(last["Close"]) <= float(last["sma_20"]) * 1.03
            if not pd.isna(last.get("sma_20"))
            else True
        )
        if not near_value and float(last.get("rsi_14", 50) or 50) > 65:
            return None

        entry = float(last["Close"])
        stop = atr_stop_long(d, 2.4)
        # Enforce minimum stop distance so risk sizing stays meaningful (~1.5%)
        min_stop = entry * 0.985
        stop = min(stop, min_stop)
        if entry - stop <= 0 or (entry - stop) / entry < 0.01:
            stop = entry * 0.975
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.30 * min(abs_mom / 0.20, 1.4)
            + 0.35 * min(max(rs, 0) / 0.12, 1.4)
            + 0.20 * (1.0 if last["sma_50"] > last["sma_200"] else 0.4)
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
            holding_period="21-63 trading days",
            entry_justification=(
                f"NIFTY>SMA200 gate ON; dual mom abs~84d {abs_mom:.1%}, RS {rs:.1%}, "
                f"non-chase entry"
            ),
            exit_justification="ATR trail; exit if NIFTY loses SMA200 or absolute momentum flips",
            invalidation_condition=f"Stop {stop:.2f}; or NIFTY closes below SMA200",
            why_good=(
                "Index trend gate plus absolute/relative momentum aims to stay in cash "
                "during broad bear phases that previously destroyed active books."
            ),
            what_could_go_wrong="Whipsaw around SMA200; late re-entry after sharp bottoms.",
            tags={"abs_mom": abs_mom, "rs_nifty": rs, "ltf_conf": conf, "nifty_sma_gate": True},
        )
