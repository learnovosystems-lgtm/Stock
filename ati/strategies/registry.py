from __future__ import annotations

from ati.strategies.base import Strategy
from ati.strategies.breakout import BreakoutStrategy
from ati.strategies.mean_reversion import MeanReversionStrategy
from ati.strategies.momentum import MomentumStrategy
from ati.strategies.multi_factor import MultiFactorStrategy
from ati.strategies.pullback import PullbackStrategy
from ati.strategies.relative_strength import RelativeStrengthStrategy
from ati.strategies.reversal import ReversalStrategy
from ati.strategies.stat_arb import StatisticalArbitrageStrategy
from ati.strategies.trend import TrendFollowingStrategy
from ati.strategies.vol_expansion import VolatilityExpansionStrategy


def all_strategies() -> list[Strategy]:
    return [
        MomentumStrategy(),
        TrendFollowingStrategy(),
        BreakoutStrategy(),
        PullbackStrategy(),
        MeanReversionStrategy(),
        StatisticalArbitrageStrategy(),
        RelativeStrengthStrategy(),
        VolatilityExpansionStrategy(),
        ReversalStrategy(),
        MultiFactorStrategy(),
    ]


def strategies_by_family() -> dict[str, Strategy]:
    return {s.family: s for s in all_strategies()}
