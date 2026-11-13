"""Dynamic strategy selection with stability penalties and ensemble agreement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ati.models.opportunity import MarketRegime, StrategyMetrics
from ati.regime.detector import RegimeState
from ati.strategies.base import RawSignal


# Map strategy_id prefixes / baseline ids → family for metric lookup
_FAMILY_ALIASES = {
    "breakout_momentum_baseline": "breakout",
    "pullback_trend_baseline": "pullback",
    "dual_momentum_baseline": "relative_strength",
    "dual_momentum_nifty_gated": "relative_strength",
    "breakout_param_search": "breakout",
    "mean_reversion_baseline": "mean_reversion",
    "regime_adaptive_multi": "multi_factor",
}


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
    sharpe_component = max(min(m.sharpe, 3.0), -2.0)
    dd_penalty = abs(m.max_drawdown) * 1.5
    stability = m.stability_score
    return (0.35 * sharpe_component + 0.35 * ret_component + 0.4 * stability) - dd_penalty


def _family_of_metric(strategy_id: str) -> str:
    if strategy_id in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[strategy_id]
    for fam in (
        "relative_strength",
        "trend_following",
        "mean_reversion",
        "statistical_arbitrage",
        "volatility_expansion",
        "multi_factor",
        "momentum",
        "breakout",
        "pullback",
        "reversal",
    ):
        if strategy_id.startswith(fam) or fam in strategy_id:
            return fam
    return strategy_id.split("_")[0]


class StrategySelector:
    def __init__(self, metrics: list[StrategyMetrics] | None = None):
        self.metrics = metrics or []

    def _metrics_for_family(self, family: str, regime: MarketRegime, cap: str | None) -> list[StrategyMetrics]:
        relevant = [m for m in self.metrics if _family_of_metric(m.strategy_id) == family]
        if cap:
            capped = [m for m in relevant if m.cap_bucket in (cap, "ALL")]
            if capped:
                relevant = capped
        regime_metrics = [m for m in relevant if m.regime in (regime.value, "ALL")]
        return regime_metrics or relevant

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
        if regime == MarketRegime.UNCERTAIN:
            return 0.0

        # Measured improvement bias: after negative daily breakout/pullback results,
        # prefer relative_strength / trend in bull; mean_reversion in sideways/bear.
        prior_boost = 0.0
        if regime == MarketRegime.BULL and family in ("relative_strength", "trend_following", "momentum"):
            prior_boost = 0.15
        if regime == MarketRegime.SIDEWAYS and family in ("mean_reversion", "multi_factor"):
            prior_boost = 0.15
        if regime == MarketRegime.BEAR and family in ("mean_reversion", "reversal"):
            prior_boost = 0.10
        if family in ("breakout", "pullback"):
            prior_boost = -0.1  # demote until metrics recover

        base = (1.0 if family in preferred else 0.25) + prior_boost
        pool = self._metrics_for_family(family, regime, cap)
        if not pool:
            return max(0.0, base)

        # Hard penalty: if latest measured expectancy and sharpe are both negative, crush weight
        latest = pool[-1]
        if latest.expectancy < 0 and latest.sharpe < 0:
            return max(0.0, base * 0.15)

        adj = sum(_stability_adjusted_score(m) for m in pool) / len(pool)
        return max(0.0, base * (0.55 + 0.45 * _sigmoid(adj)))

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
            if w <= 0.05:
                continue
            final = sig.score * w * (0.7 + 0.3 * regime.confidence)
            if final >= min_score and sig.risk_reward >= 2.0:
                weighted.append((final, sig))

        if not weighted:
            return SelectionResult([], False, "No strategy cleared stability-weighted threshold", family_weights)

        weighted.sort(key=lambda x: x[0], reverse=True)
        longs = [s for _, s in weighted if s.direction == "LONG"]
        families = {s.family for s in longs}
        ensemble = len(families) >= ensemble_min_agree
        if ensemble:
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
