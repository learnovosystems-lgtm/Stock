"""Charge breakdown models for broker-configurable fee engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BrokerId(str, Enum):
    FYERS = "fyers"
    ZERODHA = "zerodha"


class Segment(str, Enum):
    EQUITY_DELIVERY = "equity_delivery"
    EQUITY_INTRADAY = "equity_intraday"
    FUTURES = "futures"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class ChargeBreakdown:
    """Itemized statutory + brokerage fees for one executed fill.

    Slippage is intentionally excluded — apply it on price separately so fees
    and slippage are never double-counted.
    """

    broker: str
    schedule_id: str
    effective_date: str
    segment: str
    side: str
    qty: float
    price: float
    turnover: float
    brokerage: float
    stt: float
    exchange_txn: float
    clearing: float
    sebi: float
    stamp: float
    ipft: float
    gst: float
    dp: float
    total_fees: float
    source_urls: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_urls"] = list(self.source_urls)
        return d


@dataclass(frozen=True)
class FillEconomics:
    """Cash impact of one fill: slipped execution price + fees."""

    raw_price: float
    exec_price: float
    slippage_rupees: float
    charges: ChargeBreakdown
    notional: float
    total_fees: float
    cash_delta: float  # negative on buy, positive on sell (fees already netted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_price": self.raw_price,
            "exec_price": self.exec_price,
            "slippage_rupees": self.slippage_rupees,
            "notional": self.notional,
            "total_fees": self.total_fees,
            "cash_delta": self.cash_delta,
            "charges": self.charges.to_dict(),
        }


@dataclass(frozen=True)
class RoundTripEconomics:
    """Reconcile buy + one or more sell fills into net P&L."""

    buy: FillEconomics
    sells: tuple[FillEconomics, ...]
    gross_pnl: float
    total_fees: float
    total_slippage: float
    net_pnl: float
    qty_bought: float
    qty_sold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "buy": self.buy.to_dict(),
            "sells": [s.to_dict() for s in self.sells],
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "net_pnl": self.net_pnl,
            "qty_bought": self.qty_bought,
            "qty_sold": self.qty_sold,
        }
