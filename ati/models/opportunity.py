"""Canonical opportunity / signal schemas (directive section K)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    WAIT = "WAIT"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    UNCERTAIN = "UNCERTAIN"


class CapBucket(str, Enum):
    LARGE = "LARGE"
    MID = "MID"
    SMALL = "SMALL"


class Opportunity(BaseModel):
    stock: str
    exchange: str = "NSE"
    signal: SignalType
    current_price: float
    ideal_entry: float
    entry_range_low: float
    entry_range_high: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward: float
    position_quantity: int
    capital_required: float
    max_rupee_risk: float
    expected_holding_period: str
    opportunity_score: float
    strategy_score: float
    market_regime_suitability: float
    entry_quality_score: float
    risk_score: float
    model_confidence: float
    est_prob_target_before_stop: float
    expected_return: float
    strategy_selected: str
    market_regime: MarketRegime
    sector: str
    sector_strength: float
    relative_strength_vs_nifty: float
    entry_justification: str
    exit_justification: str
    invalidation_condition: str
    why_good: str
    what_could_go_wrong: str
    data_timestamp: datetime
    breakout_confirmation_price: Optional[float] = None
    max_acceptable_entry: Optional[float] = None
    invalidation_price: Optional[float] = None
    extras: dict[str, Any] = Field(default_factory=dict)

    def format_report(self) -> str:
        er = f"{self.entry_range_low:.2f} - {self.entry_range_high:.2f}"
        lines = [
            f"Stock: {self.stock}",
            f"Exchange: {self.exchange}",
            f"Signal: {self.signal.value}",
            f"Current price: {self.current_price:.2f}",
            f"Ideal entry: {self.ideal_entry:.2f}",
            f"Entry range: {er}",
            f"Stop-loss: {self.stop_loss:.2f}",
            f"Target 1: {self.target_1:.2f}",
            f"Target 2: {self.target_2:.2f}",
            f"Target 3: {self.target_3:.2f}",
            f"Risk/reward: {self.risk_reward:.2f}",
            f"Position quantity: {self.position_quantity}",
            f"Capital required: {self.capital_required:.2f}",
            f"Maximum rupee risk: {self.max_rupee_risk:.2f}",
            f"Expected holding period: {self.expected_holding_period}",
            f"Opportunity score: {self.opportunity_score:.2f}",
            f"Model confidence: {self.model_confidence:.2f}",
            f"Estimated probability of target before stop: {self.est_prob_target_before_stop:.2%}",
            f"Strategy selected: {self.strategy_selected}",
            f"Market regime: {self.market_regime.value}",
            f"Sector strength: {self.sector_strength:.2f}",
            f"Entry justification: {self.entry_justification}",
            f"Exit justification: {self.exit_justification}",
            f"Invalidation condition: {self.invalidation_condition}",
            f"Data timestamp: {self.data_timestamp.isoformat()}",
            "",
            f"Why this is a good opportunity: {self.why_good}",
            f"What could make this wrong: {self.what_could_go_wrong}",
        ]
        return "\n".join(lines)


class StrategyMetrics(BaseModel):
    strategy_id: str
    version: str
    sample_count: int
    win_rate: float
    expectancy: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    avg_holding_days: float
    stability_score: float
    total_return: float
    regime: str = "ALL"
    cap_bucket: str = "ALL"
    oos: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
