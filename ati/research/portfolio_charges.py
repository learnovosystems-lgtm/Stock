"""Fill-level ChargeEngine bookkeeping for portfolio (dual-mom / adaptive) backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ati.config import DEFAULT, RiskConfig
from ati.engine.charges import ChargeEngine, default_charge_engine
from ati.models.charges import Segment, Side

PORTFOLIO_BT_COST_MODEL = "charge_engine_fill_level"


@dataclass
class _Lot:
    shares: float
    entry_raw: float
    entry_exec: float
    entry_fees: float
    entry_i: int
    book: str = ""


@dataclass
class PortfolioChargeBook:
    """Cash + lots with per-fill ChargeEngine economics (delivery segment)."""

    cash: float
    engine: ChargeEngine
    session: Any
    segment: Segment = Segment.EQUITY_DELIVERY
    holdings: dict[str, _Lot] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    cost_model: str = PORTFOLIO_BT_COST_MODEL

    @classmethod
    def create(
        cls,
        *,
        initial_capital: float,
        risk: RiskConfig | None = None,
        dp_ledger_path: Path | None = None,
        slippage_pct: float | None = None,
    ) -> "PortfolioChargeBook":
        risk = risk or DEFAULT.risk
        slip = risk.slippage_pct if slippage_pct is None else slippage_pct
        eng = default_charge_engine(slippage_pct=slip, dp_ledger_path=dp_ledger_path)
        return cls(cash=float(initial_capital), engine=eng, session=eng.new_session())

    def portfolio_value(self, panel: pd.DataFrame, i: int) -> float:
        value = self.cash
        row = panel.iloc[i]
        for sym, lot in self.holdings.items():
            value += lot.shares * float(row[sym])
        return value

    def buy(
        self,
        sym: str,
        shares: float,
        raw_px: float,
        *,
        trade_date,
        bar_i: int,
        order_id: str,
        book: str = "",
    ) -> bool:
        if shares <= 1e-12 or raw_px <= 0:
            return False
        # Non-mutating size probe so we do not pollute order brokerage aggregation
        probe = self.engine.estimate_economics_for_fill(
            side=Side.BUY,
            qty=shares,
            raw_price=raw_px,
            segment=self.segment,
            symbol=sym,
            trade_date=trade_date,
            assume_dp_charged=True,
        )
        if -probe.cash_delta > self.cash + 1e-9:
            unit = -probe.cash_delta / shares
            shares = self.cash / max(unit, 1e-9)
            if shares <= 1e-12:
                return False
        eco = self.engine.economics_for_fill(
            side=Side.BUY,
            qty=shares,
            raw_price=raw_px,
            segment=self.segment,
            session=self.session,
            order_id=order_id,
            symbol=sym,
            trade_date=trade_date,
        )
        if -eco.cash_delta > self.cash + 1e-9:
            return False
        self.cash += eco.cash_delta
        if sym in self.holdings:
            lot = self.holdings[sym]
            total_sh = lot.shares + shares
            lot.entry_raw = (lot.entry_raw * lot.shares + raw_px * shares) / total_sh
            lot.entry_exec = (lot.entry_exec * lot.shares + eco.exec_price * shares) / total_sh
            lot.entry_fees += eco.total_fees
            lot.shares = total_sh
            if book:
                lot.book = book
        else:
            self.holdings[sym] = _Lot(
                shares=shares,
                entry_raw=raw_px,
                entry_exec=eco.exec_price,
                entry_fees=eco.total_fees,
                entry_i=bar_i,
                book=book,
            )
        return True

    def sell(
        self,
        sym: str,
        shares: float,
        raw_px: float,
        *,
        trade_date,
        bar_i: int,
        order_id: str,
        reason: str,
        log_trade: bool = True,
    ) -> dict | None:
        lot = self.holdings.get(sym)
        if lot is None or shares <= 1e-12:
            return None
        shares = min(shares, lot.shares)
        eco = self.engine.economics_for_fill(
            side=Side.SELL,
            qty=shares,
            raw_price=raw_px,
            segment=self.segment,
            session=self.session,
            order_id=order_id,
            symbol=sym,
            trade_date=trade_date,
        )
        frac = shares / lot.shares
        entry_fee_alloc = lot.entry_fees * frac
        gross = (eco.exec_price - lot.entry_exec) * shares
        exit_fees = eco.total_fees
        total_fees = exit_fees + entry_fee_alloc
        net = gross - total_fees
        self.cash += eco.cash_delta
        trade = {
            "symbol": sym,
            "entry": lot.entry_exec,
            "exit": eco.exec_price,
            "entry_raw": lot.entry_raw,
            "exit_raw": raw_px,
            "qty": shares,
            "pnl": net,
            "gross_pnl": gross,
            "fees": total_fees,
            "exit_fees": exit_fees,
            "entry_fees": entry_fee_alloc,
            "brokerage": eco.charges.brokerage,
            "stt": eco.charges.stt,
            "exchange_txn": eco.charges.exchange_txn,
            "sebi": eco.charges.sebi,
            "ipft": eco.charges.ipft,
            "gst": eco.charges.gst,
            "stamp": eco.charges.stamp,
            "dp": eco.charges.dp,
            "ret": net / max(lot.entry_exec * shares, 1e-9),
            "bars": max(1, bar_i - lot.entry_i),
            "reason": reason,
            "book": lot.book,
            "cost_model": self.cost_model,
        }
        rem = lot.shares - shares
        if rem <= 1e-12:
            self.holdings.pop(sym)
        else:
            lot.shares = rem
            lot.entry_fees = max(0.0, lot.entry_fees - entry_fee_alloc)
        if log_trade:
            self.trades.append(trade)
        return trade

    def close_position(self, sym: str, panel: pd.DataFrame, i: int, reason: str, trade_date) -> None:
        lot = self.holdings.get(sym)
        if lot is None:
            return
        px = float(panel.iloc[i][sym])
        self.sell(
            sym,
            lot.shares,
            px,
            trade_date=trade_date,
            bar_i=i,
            order_id=f"bt-close-{sym}-{reason}-{trade_date}",
            reason=reason,
        )

    def flatten(self, panel: pd.DataFrame, i: int, reason: str, trade_date) -> None:
        for sym in list(self.holdings):
            self.close_position(sym, panel, i, reason, trade_date)
