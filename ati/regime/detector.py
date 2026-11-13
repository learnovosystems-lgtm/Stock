"""Market regime intelligence from NIFTY, India VIX, ADX, MA, breadth, sectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ati.data.provider import MarketDataProvider
from ati.indicators.technical import adx, enrich, sma
from ati.models.opportunity import MarketRegime


@dataclass
class RegimeState:
    regime: MarketRegime
    nifty_trend: float
    india_vix: float
    adx: float
    ma_structure: float  # +1 bullish stack, -1 bearish, 0 mixed
    volatility: float
    breadth: float  # advance/decline of scanned universe [-1, 1]
    sector_rotation_score: float
    confidence: float
    notes: str

    def allows_longs(self) -> bool:
        return self.regime in (MarketRegime.BULL, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOL)

    def allows_shorts(self) -> bool:
        return self.regime in (MarketRegime.BEAR, MarketRegime.HIGH_VOL)

    def preferred_families(self) -> list[str]:
        mapping = {
            MarketRegime.BULL: [
                "momentum",
                "relative_strength",
                "trend_following",
                "breakout",
                "pullback",
                "multi_factor",
            ],
            MarketRegime.BEAR: [
                "mean_reversion",
                "reversal",
                "relative_strength",
                "volatility_expansion",
                "multi_factor",
            ],
            MarketRegime.SIDEWAYS: [
                "mean_reversion",
                "pullback",
                "statistical_arbitrage",
                "multi_factor",
            ],
            MarketRegime.HIGH_VOL: [
                "volatility_expansion",
                "breakout",
                "mean_reversion",
                "multi_factor",
            ],
            MarketRegime.UNCERTAIN: [],
        }
        return mapping[self.regime]


class RegimeDetector:
    def __init__(self, provider: MarketDataProvider | None = None):
        self.provider = provider or MarketDataProvider()

    def detect(
        self,
        universe_daily: Mapping[str, pd.DataFrame] | None = None,
        sector_returns: Mapping[str, float] | None = None,
    ) -> RegimeState:
        nifty = enrich(self.provider.nifty())
        vix = self.provider.india_vix()
        if nifty.empty or len(nifty) < 60:
            return RegimeState(
                regime=MarketRegime.UNCERTAIN,
                nifty_trend=0.0,
                india_vix=float("nan"),
                adx=0.0,
                ma_structure=0.0,
                volatility=0.0,
                breadth=0.0,
                sector_rotation_score=0.0,
                confidence=0.0,
                notes="Insufficient NIFTY history",
            )

        last = nifty.iloc[-1]
        nifty_trend = float(last.get("ret_63", 0) or 0)
        adx_v = float(last.get("adx", 0) or 0)
        vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else 15.0
        vol = float(last.get("rvol_20", 0) or 0)

        ma_structure = 0.0
        if last["Close"] > last["sma_50"] > last["sma_200"]:
            ma_structure = 1.0
        elif last["Close"] < last["sma_50"] < last["sma_200"]:
            ma_structure = -1.0

        breadth = 0.0
        if universe_daily:
            ups = downs = 0
            for df in universe_daily.values():
                if df is None or len(df) < 22:
                    continue
                r = float(df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1)
                if r > 0:
                    ups += 1
                else:
                    downs += 1
            total = ups + downs
            breadth = (ups - downs) / total if total else 0.0

        sector_rotation_score = 0.0
        if sector_returns:
            vals = np.array(list(sector_returns.values()), dtype=float)
            if len(vals):
                sector_rotation_score = float(np.nanmean(vals))

        # Classification rules (transparent, testable)
        # Trend impulse can lead MA stack — use ADX+breadth+return as primary,
        # MA structure as confirmation rather than a hard veto.
        notes: list[str] = []
        regime = MarketRegime.UNCERTAIN
        if vix_last >= 22 and vol >= 0.22:
            regime = MarketRegime.HIGH_VOL
            notes.append("Elevated India VIX and realized vol")
        elif nifty_trend > 0.02 and adx_v >= 20 and breadth >= 0.1:
            regime = MarketRegime.BULL
            notes.append("Positive 63d NIFTY with strong ADX and breadth")
            if ma_structure <= 0:
                notes.append("MA stack not fully aligned — reduced confidence")
        elif nifty_trend < -0.02 and adx_v >= 20 and breadth <= -0.1:
            regime = MarketRegime.BEAR
            notes.append("Negative 63d NIFTY with strong ADX and weak breadth")
        elif adx_v < 18 and abs(nifty_trend) < 0.04:
            regime = MarketRegime.SIDEWAYS
            notes.append("Low ADX / muted trend → sideways")
        elif ma_structure > 0 and nifty_trend > 0 and breadth >= 0:
            regime = MarketRegime.BULL
            notes.append("Bullish MA stack with non-negative breadth")
        elif ma_structure < 0 and nifty_trend < 0 and breadth <= 0:
            regime = MarketRegime.BEAR
            notes.append("Bearish MA stack with non-positive breadth")
        else:
            regime = MarketRegime.UNCERTAIN
            notes.append("Mixed evidence → NO TRADE bias")

        # Confidence blends agreement of factors
        # Optional HMM/GMM state as an extra vote (never sole authority)
        try:
            from ati.ml.regime_hmm import fit_predict_regime

            hmm = fit_predict_regime(nifty, vix)
            notes.append(f"HMM state={hmm.state}:{hmm.state_labels.get(hmm.state, '?')} conf={hmm.confidence:.2f}")
            if hmm.regime == MarketRegime.BULL and regime in (MarketRegime.BULL, MarketRegime.UNCERTAIN):
                if regime == MarketRegime.UNCERTAIN and hmm.confidence >= 0.55:
                    regime = MarketRegime.BULL
                    notes.append("HMM upgraded uncertain→BULL")
            elif hmm.regime == MarketRegime.BEAR and regime in (MarketRegime.BEAR, MarketRegime.UNCERTAIN):
                if regime == MarketRegime.UNCERTAIN and hmm.confidence >= 0.55:
                    regime = MarketRegime.BEAR
                    notes.append("HMM upgraded uncertain→BEAR")
        except Exception:  # noqa: BLE001
            hmm = None

        bullish_votes = sum(
            [
                ma_structure > 0,
                nifty_trend > 0,
                breadth > 0,
                adx_v >= 18 and ma_structure > 0,
                vix_last < 18,
                bool(hmm and hmm.regime == MarketRegime.BULL),
            ]
        )
        bearish_votes = sum(
            [
                ma_structure < 0,
                nifty_trend < 0,
                breadth < 0,
                adx_v >= 18 and ma_structure < 0,
                vix_last >= 20,
                bool(hmm and hmm.regime == MarketRegime.BEAR),
            ]
        )
        agreement = max(bullish_votes, bearish_votes) / 6.0
        if regime == MarketRegime.UNCERTAIN:
            confidence = 0.35 * agreement
        else:
            confidence = 0.45 + 0.55 * agreement
            if regime == MarketRegime.BULL and ma_structure <= 0:
                confidence *= 0.85
            if regime == MarketRegime.BEAR and ma_structure >= 0:
                confidence *= 0.85

        return RegimeState(
            regime=regime,
            nifty_trend=nifty_trend,
            india_vix=vix_last,
            adx=adx_v,
            ma_structure=ma_structure,
            volatility=vol,
            breadth=breadth,
            sector_rotation_score=sector_rotation_score,
            confidence=float(confidence),
            notes="; ".join(notes),
        )


def sector_relative_returns(
    provider: MarketDataProvider,
    sector_symbols: Mapping[str, str],
    lookback: int = 21,
) -> dict[str, float]:
    out: dict[str, float] = {}
    nifty = provider.nifty()
    if nifty.empty or len(nifty) < lookback + 1:
        return out
    nifty_ret = float(nifty["Close"].iloc[-1] / nifty["Close"].iloc[-lookback - 1] - 1)
    for sector, sym in sector_symbols.items():
        df = provider.get_daily(sym)
        if df.empty or len(df) < lookback + 1:
            continue
        sec_ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-lookback - 1] - 1)
        out[sector] = sec_ret - nifty_ret
    return out
