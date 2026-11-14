"""Hard risk gates and position sizing for Indian paper trading."""

from __future__ import annotations

from dataclasses import dataclass

from ati.config import RiskConfig, DEFAULT
from ati.engine.charges import ChargeEngine, default_charge_engine
from ati.models.charges import Segment, Side


@dataclass
class PortfolioRiskState:
    equity: float
    cash: float
    open_risk_rupees: float
    open_positions: int
    day_pnl: float


@dataclass
class PositionSizeResult:
    quantity: int
    capital_required: float
    max_rupee_risk: float
    allowed: bool
    reason: str


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig | None = None,
        *,
        charges: ChargeEngine | None = None,
    ):
        self.config = config or DEFAULT.risk
        self.charges = charges or default_charge_engine(slippage_pct=self.config.slippage_pct)

    def risk_budget_per_trade(self, equity: float) -> float:
        return equity * self.config.risk_per_trade_pct

    def size_long(
        self,
        *,
        equity: float,
        cash: float,
        entry: float,
        stop: float,
        state: PortfolioRiskState,
        avg_volume: float | None = None,
    ) -> PositionSizeResult:
        if entry <= 0 or stop <= 0 or stop >= entry:
            return PositionSizeResult(0, 0.0, 0.0, False, "Invalid entry/stop for long")

        if state.open_positions >= self.config.max_positions:
            return PositionSizeResult(0, 0.0, 0.0, False, "Max concurrent positions reached")

        if state.day_pnl <= -equity * self.config.max_daily_loss_pct:
            return PositionSizeResult(0, 0.0, 0.0, False, "Daily loss limit breached")

        stop_dist = entry - stop
        risk_rupees = self.risk_budget_per_trade(equity)
        remaining_open_risk = equity * self.config.max_open_risk_pct - state.open_risk_rupees
        if remaining_open_risk <= 0:
            return PositionSizeResult(0, 0.0, 0.0, False, "Combined open risk limit reached")

        risk_rupees = min(risk_rupees, remaining_open_risk)
        qty = int(risk_rupees // stop_dist)
        if qty < 1:
            return PositionSizeResult(0, 0.0, 0.0, False, "Stop too wide for 1% risk budget")

        # Liquidity: do not take more than ~2% of 20d avg volume notionally
        if avg_volume and avg_volume > 0:
            qty = min(qty, max(1, int(0.02 * avg_volume)))

        capital = qty * entry
        max_exposure = equity * self.config.max_portfolio_exposure_pct
        if capital > cash:
            qty = int(cash // entry)
            capital = qty * entry
        if capital > max_exposure:
            qty = int(max_exposure // entry)
            capital = qty * entry

        if qty < 1:
            return PositionSizeResult(0, 0.0, 0.0, False, "Insufficient cash/exposure room")

        actual_risk = qty * stop_dist
        return PositionSizeResult(qty, capital, actual_risk, True, "OK")

    def round_trip_cost_pct(
        self,
        *,
        segment: Segment | str | None = None,
        price: float = 1000.0,
        qty: float = 100.0,
    ) -> float:
        seg = segment or DEFAULT.default_segment
        return self.charges.round_trip_cost_pct(price=price, qty=qty, segment=seg, include_slippage=True)

    def apply_buy_slippage(self, price: float) -> float:
        """Slippage-only buy price (fees via ChargeEngine.economics_for_fill)."""
        return self.charges.slipped_price(price, Side.BUY)

    def apply_sell_slippage(self, price: float) -> float:
        """Slippage-only sell price (fees via ChargeEngine.economics_for_fill)."""
        return self.charges.slipped_price(price, Side.SELL)
