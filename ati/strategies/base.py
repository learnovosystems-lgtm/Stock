"""Strategy plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ati.models.opportunity import MarketRegime
from ati.regime.detector import RegimeState


@dataclass
class StrategyContext:
    symbol: str
    sector: str
    cap: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    hourly: pd.DataFrame
    m30: pd.DataFrame
    m15: pd.DataFrame
    nifty_daily: pd.DataFrame
    regime: RegimeState
    sector_strength: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawSignal:
    strategy_id: str
    family: str
    direction: str  # LONG / SHORT / FLAT
    score: float
    ideal_entry: float
    entry_low: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float
    target_3: float
    breakout_confirmation: float | None
    max_entry: float | None
    invalidation: float
    holding_period: str
    entry_justification: str
    exit_justification: str
    invalidation_condition: str
    why_good: str
    what_could_go_wrong: str
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        risk = abs(self.ideal_entry - self.stop)
        if risk <= 0:
            return 0.0
        reward = abs(self.target_1 - self.ideal_entry)
        return reward / risk


class Strategy(ABC):
    id: str = "base"
    family: str = "base"
    version: str = "1.0.0"

    @abstractmethod
    def generate(self, ctx: StrategyContext) -> Optional[RawSignal]:
        raise NotImplementedError

    def suitable_regimes(self) -> set[MarketRegime]:
        return set(MarketRegime)
