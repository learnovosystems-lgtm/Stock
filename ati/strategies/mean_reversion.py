from __future__ import annotations

from typing import Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.strategies.base import RawSignal, Strategy, StrategyContext
from ati.strategies.common import long_targets, lower_tf_confirmation, prepare_daily


class MeanReversionStrategy(Strategy):
    id = "mean_reversion_v1"
    family = "mean_reversion"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOL, MarketRegime.BEAR}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 40:
            return None
        last = d.iloc[-1]
        if any(pd.isna(last[c]) for c in ("bb_lower", "bb_mid", "rsi_14", "atr_14", "adx")):
            return None
        # Prefer non-trending or mild trend
        if last["adx"] > 28 and last["Close"] < last["sma_50"]:
            return None
        stretched = last["Close"] <= last["bb_lower"] or last["rsi_14"] <= 32
        if not stretched:
            return None

        entry = float(last["Close"])
        stop = float(min(d["Low"].iloc[-5:].min(), entry - 1.2 * last["atr_14"]))
        if entry - stop <= 0:
            return None
        # Mean reversion targets towards mid/upper band — still enforce >= 2R via long_targets floor
        t1 = float(last["bb_mid"])
        risk = entry - stop
        if (t1 - entry) / risk < 2.0:
            t1 = entry + 2.0 * risk
        t2 = entry + 3.0 * risk
        t3 = float(last["bb_upper"]) if not pd.isna(last["bb_upper"]) else entry + 4.0 * risk
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = (
            0.45 * (1 - float(last["rsi_14"]) / 50)
            + 0.25 * (1 if last["Close"] <= last["bb_lower"] else 0.5)
            + 0.3 * conf
        )

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(max(0.1, score)),
            ideal_entry=entry,
            entry_low=stop + 0.2 * risk,
            entry_high=entry * 1.004,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=None,
            max_entry=entry * 1.008,
            invalidation=stop,
            holding_period="2-10 trading days",
            entry_justification=f"Stretch to BB lower / RSI {last['rsi_14']:.1f} with ADX {last['adx']:.1f}",
            exit_justification="Exit at mean (BB mid) / time stop 10d; hard stop on further extension",
            invalidation_condition=f"Close below {stop:.2f} (continuation dump)",
            why_good="In sideways/high-vol regimes, extreme deviations often mean-revert in liquid names.",
            what_could_go_wrong="Catching a falling knife in a new bear impulse.",
            tags={"rsi": float(last["rsi_14"]), "adx": float(last["adx"])},
        )
