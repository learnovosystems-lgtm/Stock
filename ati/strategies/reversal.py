from __future__ import annotations

from typing import Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.strategies.base import RawSignal, Strategy, StrategyContext
from ati.strategies.common import long_targets, lower_tf_confirmation, prepare_daily


class ReversalStrategy(Strategy):
    id = "reversal_v1"
    family = "reversal"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.BEAR, MarketRegime.HIGH_VOL, MarketRegime.SIDEWAYS}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        if len(d) < 40:
            return None
        last = d.iloc[-1]
        prev = d.iloc[-2]
        if any(pd.isna(last[c]) for c in ("rsi_14", "macd_hist", "atr_14", "sma_20")):
            return None
        # Bullish reversal seed: oversold + MACD hist turning up + hammer-ish close
        oversold = last["rsi_14"] < 35 and prev["rsi_14"] <= last["rsi_14"]
        macd_turn = last["macd_hist"] > prev["macd_hist"] and last["macd_hist"] < 0
        reclaim = last["Close"] > last["Open"] and last["Close"] > prev["Close"]
        if not (oversold and macd_turn and reclaim):
            return None

        entry = float(last["Close"])
        stop = float(min(d["Low"].iloc[-8:].min(), entry - 1.4 * last["atr_14"]))
        if entry - stop <= 0:
            return None
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        # Cap first target near SMA20 for reversals
        t1 = min(t1, float(last["sma_20"]) if last["sma_20"] > entry else t1)
        if (t1 - entry) / (entry - stop) < 2.0:
            t1 = entry + 2.0 * (entry - stop)
        conf = lower_tf_confirmation(ctx.hourly, ctx.m15, "LONG")
        score = 0.4 + 0.3 * (35 - float(last["rsi_14"])) / 35 + 0.3 * conf

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry,
            entry_low=entry * 0.99,
            entry_high=entry * 1.005,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=float(last["sma_20"]),
            max_entry=entry * 1.01,
            invalidation=stop,
            holding_period="2-12 trading days",
            entry_justification=(
                f"Oversold RSI {last['rsi_14']:.1f} with MACD histogram turn and bullish reclaim candle"
            ),
            exit_justification="Quick scale to SMA20; hard stop — never widen",
            invalidation_condition=f"Reversal failure below {stop:.2f}",
            why_good="Exhaustion + momentum turn can offer tactical mean-reversion entries after washouts.",
            what_could_go_wrong="Failed reversal in strong downtrends; news gaps.",
            tags={"rsi": float(last["rsi_14"]), "ltf_conf": conf},
        )
