"""Dynamic strategy selection with stability penalties and ensemble agreement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ati.models.opportunity import MarketRegime, StrategyMetrics
from ati.regime.detector import RegimeState
from ati.strategies.base import RawSignal


@dataclass
class SelectionResult:
    selected: list[RawSignal]
    ensemble: bool
    reason: str
    family_weights: dict[str, float]


def _stability_adjusted_score(m: StrategyMetrics) -> float:
    """Prefer OOS risk-adjusted return with drawdown/stability penalties.

    Do NOT pick solely on highest recent return.
    """
    ret_component = m.expectancy
    sharpe_component = max(m.sharpe, -1.0)
    dd_penalty = abs(m.max_drawdown) * 1.5
    stability = m.stability_score
    # Unstable high-return strategies get crushed
    return (0.35 * sharpe_component + 0.35 * ret_component + 0.4 * stability) - dd_penalty


class StrategySelector:
    def __init__(self, metrics: list[StrategyMetrics] | None = None):
        self.metrics = metrics or []

    def family_weight(self, family: str, regime: MarketRegime, cap: str | None = None) -> float:
        preferred = {
            MarketRegime.BULL: {
                "momentum",
                "relative_strength",
                "trend_following",
                "breakout",
                "pullback",
                "multi_factor",
            },
            MarketRegime.BEAR: {"mean_reversion", "reversal", "relative_strength", "multi_factor"},
            MarketRegime.SIDEWAYS: {
                "mean_reversion",
                "pullback",
                "statistical_arbitrage",
                "multi_factor",
            },
            MarketRegime.HIGH_VOL: {
                "volatility_expansion",
                "breakout",
                "mean_reversion",
                "multi_factor",
            },
            MarketRegime.UNCERTAIN: set(),
        }[regime]
        base = 1.0 if family in preferred else 0.25
        if regime == MarketRegime.UNCERTAIN:
            return 0.0

        relevant = [
            m
            for m in self.metrics
            if m.strategy_id.startswith(family) or m.strategy_id.startswith(family.split("_")[0])
            or family in m.strategy_id
            or m.strategy_id.replace("_v1", "") == family
            or True  # soft match via family field encoded in strategy_id prefixes
        ]
        # Better match: metrics store strategy_id like momentum_v1
        relevant = [m for m in self.metrics if family in m.strategy_id or m.strategy_id.startswith(family)]
        if cap:
            cap_metrics = [m for m in relevant if m.cap_bucket in (cap, "ALL")]
            if cap_metrics:
                relevant = cap_metrics
        regime_metrics = [m for m in relevant if m.regime in (regime.value, "ALL")]
        pool = regime_metrics or relevant
        if not pool:
            return base
        adj = sum(_stability_adjusted_score(m) for m in pool) / len(pool)
        return max(0.0, base * (0.6 + 0.4 * _sigmoid(adj)))

    def select(
        self,
        signals: Iterable[RawSignal],
        regime: RegimeState,
        *,
        cap: str = "ALL",
        min_score: float = 0.55,
        ensemble_min_agree: int = 2,
    ) -> SelectionResult:
        if regime.regime == MarketRegime.UNCERTAIN or regime.confidence < 0.4:
            return SelectionResult([], False, "UNCERTAIN regime / low confidence → NO TRADE", {})

        weighted: list[tuple[float, RawSignal]] = []
        family_weights: dict[str, float] = {}
        for sig in signals:
            w = self.family_weight(sig.family, regime.regime, cap=cap)
            family_weights[sig.family] = w
            if w <= 0:
                continue
            final = sig.score * w * (0.7 + 0.3 * regime.confidence)
            if final >= min_score and sig.risk_reward >= 2.0:
                weighted.append((final, sig))

        if not weighted:
            return SelectionResult([], False, "No strategy cleared stability-weighted threshold", family_weights)

        weighted.sort(key=lambda x: x[0], reverse=True)
        # Ensemble if top independent families agree on LONG
        longs = [s for _, s in weighted if s.direction == "LONG"]
        families = {s.family for s in longs}
        ensemble = len(families) >= ensemble_min_agree
        if ensemble:
            # Keep top signal from each agreeing family (max 3)
            chosen: list[RawSignal] = []
            seen: set[str] = set()
            for _, s in weighted:
                if s.family in seen:
                    continue
                seen.add(s.family)
                chosen.append(s)
                if len(chosen) >= 3:
                    break
            return SelectionResult(
                chosen,
                True,
                f"Ensemble agreement across {sorted(seen)}",
                family_weights,
            )

        best_score, best = weighted[0]
        return SelectionResult([best], False, f"Selected {best.family} @ weighted {best_score:.2f}", family_weights)


def _sigmoid(x: float) -> float:
    import math

    return 1 / (1 + math.exp(-x))
