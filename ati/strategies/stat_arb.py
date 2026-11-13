from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.strategies.base import RawSignal, Strategy, StrategyContext
from ati.strategies.common import long_targets, prepare_daily


class StatisticalArbitrageStrategy(Strategy):
    """Residual mean-reversion vs NIFTY (single-name beta-neutral style signal).

    Not classic pairs until a peer series is supplied in ctx.extras['peer_daily'].
    """

    id = "statistical_arbitrage_v1"
    family = "statistical_arbitrage"
    version = "1.0.0"

    def suitable_regimes(self) -> set[MarketRegime]:
        return {MarketRegime.SIDEWAYS, MarketRegime.BULL, MarketRegime.BEAR}

    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        d = prepare_daily(ctx.daily)
        n = ctx.nifty_daily
        if len(d) < 80 or n is None or len(n) < 80:
            return None
        peer = ctx.extras.get("peer_daily")
        if peer is not None and len(peer) >= 80:
            base = peer["Close"]
        else:
            base = n["Close"]

        aligned = pd.concat([d["Close"].rename("s"), base.rename("b")], axis=1).dropna()
        if len(aligned) < 80:
            return None
        # Rolling beta hedge residual z-score
        window = 60
        s_ret = np.log(aligned["s"]).diff()
        b_ret = np.log(aligned["b"]).diff()
        cov = s_ret.rolling(window).cov(b_ret)
        var = b_ret.rolling(window).var()
        beta = (cov / var.replace(0, np.nan)).iloc[-1]
        if pd.isna(beta):
            return None
        residual = s_ret - float(beta) * b_ret
        mu = residual.rolling(window).mean()
        sd = residual.rolling(window).std()
        z = ((residual - mu) / sd.replace(0, np.nan)).iloc[-1]
        if pd.isna(z) or z > -1.5:
            return None  # only long residual cheapness for paper long-only book

        last = d.iloc[-1]
        if pd.isna(last.get("atr_14")):
            return None
        entry = float(last["Close"])
        stop = float(entry - max(1.5 * last["atr_14"], entry * 0.03))
        t1, t2, t3 = long_targets(entry, stop, float(last["atr_14"]))
        score = min(1.5, abs(float(z)) / 2.0)

        return RawSignal(
            strategy_id=self.id,
            family=self.family,
            direction="LONG",
            score=float(score),
            ideal_entry=entry,
            entry_low=entry * 0.99,
            entry_high=entry * 1.004,
            stop=stop,
            target_1=t1,
            target_2=t2,
            target_3=t3,
            breakout_confirmation=None,
            max_entry=entry * 1.008,
            invalidation=stop,
            holding_period="3-15 trading days",
            entry_justification=f"Residual z-score {float(z):.2f} vs hedge beta {float(beta):.2f}",
            exit_justification="Exit as residual z mean-reverts toward 0; stop on further dislocation",
            invalidation_condition=f"Z-score continues lower and price breaks {stop:.2f}",
            why_good="Temporary idiosyncratic discount vs market/peer can revert without needing index direction.",
            what_could_go_wrong="Structural de-rating; residual is not guaranteed stationary.",
            tags={"z": float(z), "beta": float(beta)},
        )
