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


class MultiFactorStrategy(Strategy):
    """Combines momentum, RS, quality trend, and volume into a scored composite."""

    id = "multi_factor_v1"
    family = "multi_factor"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return set(MarketRegime) - {MarketRegime.UNCERTAIN}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 80:
            return None
        last = d.iloc[-1]
        needed = ("ret_21", "ret_63", "atr_14", "vol_surge", "rsi_14", "adx", "sma_50")
        if any(pd.isna(last[c]) for c in needed):
            return None
        if is_overextended(d):
            return None

        rs = rs_vs_nifty(d, ctx.nifty_daily)
        wb = weekly_bias(ctx.weekly)
        factors = {
            "momentum": min(max(float(last["ret_63"]) / 0.15, -1), 1.5),
            "rs": min(max(rs / 0.1, -1), 1.5),
            "trend": 1.0 if last["Close"] > last["sma_50"] else -0.5,
            "volume": min(float(last["vol_surge"]) / 1.5, 1.3) - 0.5,
            "sector": min(max(ctx.sector_strength / 0.05, -1), 1.2),
            "weekly": wb,
            "not_overbought": 0.5 if last["rsi_14"] < 70 else -0.8,
        }
        weights = {
            "momentum": 0.2,
            "rs": 0.2,
            "trend": 0.15,
            "volume": 0.1,
            "sector": 0.15,
            "weekly": 0.1,
            "not_overbought": 0.1,
        }
        score = sum(factors[k] * weights[k] for k in weights)
        if score < 0.55:
            return None

        stop = atr_stop_long(d, 2.0)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = 0.75 * score + 0.25 * conf

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry * 0.998,
            entry_low=entry * 0.99,
            entry_high=entry * 1.006,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(d["High"].iloc[-20:].max()),
            max_entry=entry * 1.012,
            invalidation=stop,
            holding_period="5-25 trading days",
            entry_justification="Multi-factor composite passed threshold: " + ", ".join(
                f"{k}={v:.2f}" for k, v in factors.items()
            ),
            exit_justification="Factor deterioration or ATR trail; partial profits at R multiples",
            invalidation_condition=f"Composite breakdown / stop {stop:.2f}",
            why_good="Diversified factor confirmation reduces reliance on any single brittle pattern.",
            what_could_go_wrong="Factor crowdedness; correlated crash across momentum/RS books.",
            tags={"factors": factors, "ltf_conf": conf},
        )
