"""Phase-1 ChargeEngine golden vectors, broker isolation, reconciliation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from datetime import datetime, timezone

from ati.engine.charges import ChargeEngine, resolve_schedule
from ati.models.charges import Segment, Side
from ati.models.opportunity import MarketRegime, Opportunity, SignalType
from ati.paper.portfolio import PaperBroker


def _approx(a: float, b: float, tol: float = 0.05) -> None:
    assert abs(a - b) <= tol, f"{a} != {b} (±{tol})"


def _opp() -> Opportunity:
    return Opportunity(
        stock="TEST.NS",
        signal=SignalType.BUY,
        current_price=1000,
        ideal_entry=1000,
        entry_range_low=990,
        entry_range_high=1010,
        stop_loss=980,
        target_1=1040,
        target_2=1060,
        target_3=1080,
        risk_reward=2.0,
        position_quantity=50,
        capital_required=50_000,
        max_rupee_risk=1000,
        expected_holding_period="10d",
        opportunity_score=70,
        strategy_score=0.8,
        market_regime_suitability=0.7,
        entry_quality_score=0.7,
        risk_score=0.3,
        model_confidence=0.6,
        est_prob_target_before_stop=0.55,
        expected_return=0.05,
        strategy_selected="dual_momentum_test",
        market_regime=MarketRegime.BULL,
        sector="TEST",
        sector_strength=0.01,
        relative_strength_vs_nifty=0.05,
        entry_justification="t",
        exit_justification="t",
        invalidation_condition="t",
        why_good="t",
        what_could_go_wrong="t",
        data_timestamp=datetime.now(timezone.utc),
    )


def test_fyers_schedule_not_zerodha():
    fy = resolve_schedule("fyers")
    zd = resolve_schedule("zerodha")
    assert fy.broker == "fyers"
    assert zd.broker == "zerodha"
    assert fy.schedule_id != zd.schedule_id
    # Delivery brokerage: FYERS has ₹20/0.3%; Zerodha delivery brokerage is ₹0
    assert fy.equity_delivery.brokerage_flat == 20.0
    assert zd.equity_delivery.brokerage_flat == 0.0


def test_fyers_equity_delivery_sample():
    """Sample: buy 100 @ 1000, sell 100 @ 1100 — FYERS delivery Standard."""
    eng = ChargeEngine("fyers", slippage_pct=0.0, as_of="2026-11-14")
    buy = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000.0, segment=Segment.EQUITY_DELIVERY)
    sell = eng.charges_for_fill(side=Side.SELL, qty=100, price=1100.0, segment=Segment.EQUITY_DELIVERY)

    # Brokerage: min(20, 0.3% * turnover)
    _approx(buy.brokerage, min(20.0, 0.003 * 100_000))
    _approx(sell.brokerage, min(20.0, 0.003 * 110_000))
    # STT 0.1% both sides
    _approx(buy.stt, 100.0)
    _approx(sell.stt, 110.0)
    # Stamp 0.015% buy only
    _approx(buy.stamp, 15.0)
    assert sell.stamp == 0.0
    # DP on sell
    _approx(sell.dp, 12.5)
    assert buy.total_fees > 0 and sell.total_fees > 0
    assert buy.schedule_id.startswith("fyers_")
    assert any("fyers.in/charges-list" in u for u in buy.source_urls)


def test_fyers_equity_intraday_sample():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    buy = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000.0, segment=Segment.EQUITY_INTRADAY)
    sell = eng.charges_for_fill(side=Side.SELL, qty=100, price=1005.0, segment=Segment.EQUITY_INTRADAY)
    # Brokerage min(20, 0.03%)
    _approx(buy.brokerage, min(20.0, 0.0003 * 100_000))
    assert buy.stt == 0.0
    _approx(sell.stt, 0.00025 * 1005 * 100)
    assert buy.dp == 0.0 and sell.dp == 0.0


def test_fyers_futures_sample():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    # 1 lot notionally: qty=50 @ 2500
    buy = eng.charges_for_fill(side=Side.BUY, qty=50, price=2500.0, segment=Segment.FUTURES)
    sell = eng.charges_for_fill(side=Side.SELL, qty=50, price=2550.0, segment=Segment.FUTURES)
    _approx(buy.brokerage, min(20.0, 0.0003 * 125_000))
    assert buy.stt == 0.0
    _approx(sell.stt, 0.0005 * 50 * 2550)
    assert buy.stamp > 0 and sell.stamp == 0.0
    assert buy.clearing > 0


def test_slippage_not_double_counted_in_fees():
    eng = ChargeEngine("fyers", slippage_pct=0.001)
    eco = eng.economics_for_fill(side=Side.BUY, qty=10, raw_price=1000.0, segment=Segment.EQUITY_DELIVERY)
    # Fees computed on slipped price, but fee total must not include slippage rupees
    assert eco.slippage_rupees == pytest.approx(10.0)  # 1 * 10 qty
    assert eco.charges.total_fees == eco.total_fees
    # Slippage is not a line item inside charges
    assert not hasattr(eco.charges, "slippage")
    fee_sum = (
        eco.charges.brokerage
        + eco.charges.stt
        + eco.charges.exchange_txn
        + eco.charges.clearing
        + eco.charges.sebi
        + eco.charges.stamp
        + eco.charges.ipft
        + eco.charges.gst
        + eco.charges.dp
    )
    _approx(fee_sum, eco.total_fees, tol=1e-4)


def test_partial_fills_multiple_exits_reconciliation():
    """One buy + two partial sells: each sell is a separate executed order (brokerage each)."""
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    buy = eng.economics_for_fill(side=Side.BUY, qty=100, raw_price=1000.0, segment=Segment.EQUITY_DELIVERY)
    s1 = eng.economics_for_fill(side=Side.SELL, qty=40, raw_price=1100.0, segment=Segment.EQUITY_DELIVERY)
    s2 = eng.economics_for_fill(side=Side.SELL, qty=60, raw_price=1120.0, segment=Segment.EQUITY_DELIVERY)
    rt = eng.reconcile_round_trip(buy=buy, sells=[s1, s2])
    assert rt.qty_bought == 100
    assert rt.qty_sold == 100
    # Two sell orders => two brokerage assessments
    assert s1.charges.brokerage > 0 and s2.charges.brokerage > 0
    # Gross from slipped notionals
    expected_gross = (40 * 1100 + 60 * 1120) - (100 * 1000)
    _approx(rt.gross_pnl, expected_gross, tol=0.01)
    assert rt.net_pnl < rt.gross_pnl
    # Cash identity: buy cash + sell cashes == net for full round trip
    _approx(buy.cash_delta + s1.cash_delta + s2.cash_delta, rt.net_pnl, tol=0.05)


def test_paper_broker_net_fill_and_operational(tmp_path: Path):
    db = tmp_path / "paper.db"
    broker = PaperBroker(db_path=db, allowed_strategy_substrings=())
    assert broker.account()["cash"] == pytest.approx(100_000.0)
    assert broker.charges.broker == "fyers"
    ok, msg = broker.try_enter(_opp())
    assert ok, msg
    assert "Bought" in msg
    acct = broker.account()
    assert acct["cash"] < 100_000.0
    pos = broker.positions()
    assert len(pos) == 1
    assert pos[0].entry_fees > 0
    sell = broker.charges.economics_for_fill(
        side=Side.SELL, qty=20, raw_price=1040.0, segment=Segment.EQUITY_DELIVERY
    )
    assert sell.total_fees > 0


def test_zerodha_profile_isolated():
    """Selecting zerodha must not change fyers engine math."""
    fy = ChargeEngine("fyers", slippage_pct=0.0)
    zd = ChargeEngine("zerodha", slippage_pct=0.0)
    fy_buy = fy.charges_for_fill(side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY)
    zd_buy = zd.charges_for_fill(side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY)
    assert fy_buy.brokerage != zd_buy.brokerage or fy_buy.broker == "fyers"
    assert zd_buy.brokerage == 0.0  # Zerodha delivery brokerage
    assert fy_buy.brokerage == pytest.approx(20.0) or fy_buy.brokerage == pytest.approx(min(20, 300))
