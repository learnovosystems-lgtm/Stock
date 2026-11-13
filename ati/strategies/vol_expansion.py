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
)


class VolatilityExpansionStrategy(Strategy):
    id = "volatility_expansion_v1"
    family = "volatility_expansion"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.HIGH_VOL, MarketRegime.BULL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 40:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("bb_width", "atr_14", "vol_surge", "dc_high")):
            return None
        # Compression then expansion
        width = d["bb_width"].dropna()
        if len(width) < 30:
            return None
        compress = float(width.iloc[-1]) <= float(width.iloc[-30:].quantile(0.25))
        expanding = float(last["atr_14"]) > float(d["atr_14"].iloc[-10:-1].mean())
        upside = last["Close"] > last["sma_20"] and last["vol_surge"] >= 1.2
        if not ((compress and upside) or (expanding and upside and last["Close"] >= last["dc_high"] * 0.99)):
            return None

        stop = atr_stop_long(d, 1.7)
        entry = float(last["Close"])
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = 0.5 * (1.0 if compress else 0.7) + 0.2 * min(float(last["vol_surge"]) / 2, 1.2) + 0.3 * conf

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry,
            entry_low=entry * 0.992,
            entry_high=entry * 1.008,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(last["dc_high"]),
            max_entry=entry * 1.015,
            invalidation=stop,
            holding_period="2-12 trading days",
            entry_justification=(
                f"BB width compression/expansion with volume {last['vol_surge']:.2f}x and upside close"
            ),
            exit_justification="Tight ATR stop; time-box if expansion fails to trend",
            invalidation_condition=f"Volatility expansion fails and price closes below {stop:.2f}",
            why_good="Low-volatility regimes often precede directional swings; expansion with volume is actionable.",
            what_could_go_wrong="Two-sided expansion / whipsaw in event-driven weeks.",
            tags={"bb_width": float(last["bb_width"]), "ltf_conf": conf},
        )
