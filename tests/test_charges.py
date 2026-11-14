"""Phase-1 ChargeEngine: FYERS Standard corrections, order brokerage, DP, reconciliation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ati.engine.charges import ChargeEngine, resolve_schedule
from ati.models.charges import Segment, Side
from ati.models.opportunity import MarketRegime, Opportunity, SignalType
from ati.paper.portfolio import PaperBroker


def _approx(a: float, b: float, tol: float = 0.02) -> None:
    assert abs(a - b) <= tol, f"{a} != {b} (±{tol})"


def _opp(stock: str = "TEST.NS") -> Opportunity:
    return Opportunity(
        stock=stock,
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


def test_fyers_not_zerodha_and_ipft_from_charges_list():
    fy = resolve_schedule("fyers")
    zd = resolve_schedule("zerodha")
    assert fy.broker == "fyers" and zd.broker == "zerodha"
    assert fy.equity_delivery.ipft_per_crore == pytest.approx(0.01)
    assert fy.futures.ipft_per_crore == pytest.approx(0.01)
    assert fy.equity_options.ipft_per_crore == pytest.approx(50.0)
    assert "0.01" in fy.notes and "10/crore" in fy.notes


def test_equity_delivery_charges(tmp_path: Path):
    eng = ChargeEngine("fyers", slippage_pct=0.0, as_of="2026-11-14", dp_ledger_path=tmp_path / "dp.db")
    buy = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=1000.0, segment=Segment.EQUITY_DELIVERY, symbol="AAA"
    )
    sell = eng.charges_for_fill(
        side=Side.SELL, qty=100, price=1100.0, segment=Segment.EQUITY_DELIVERY, symbol="AAA"
    )
    _approx(buy.brokerage, 20.0)
    _approx(sell.brokerage, 20.0)
    _approx(buy.stt, 100.0)
    _approx(sell.stt, 110.0)
    _approx(buy.stamp, 15.0)
    assert sell.stamp == 0.0
    _approx(sell.dp, 12.5)
    # IPFT ₹0.01/crore on ₹1L = ₹0.0001 → rounds to ₹0.00 at 2dp (documented)
    assert buy.ipft == 0.0
    # GST excludes STT/stamp (FYERS charges-list base)
    gst_base = buy.brokerage + buy.exchange_txn + buy.clearing + buy.sebi + buy.ipft
    _approx(buy.gst, gst_base * 0.18, tol=0.02)


def test_equity_intraday_charges():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    buy = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000.0, segment=Segment.EQUITY_INTRADAY)
    sell = eng.charges_for_fill(side=Side.SELL, qty=100, price=1005.0, segment=Segment.EQUITY_INTRADAY)
    _approx(buy.brokerage, min(20.0, 0.0003 * 100_000))
    assert buy.stt == 0.0
    _approx(sell.stt, 0.00025 * 1005 * 100)
    assert buy.dp == 0.0 and sell.dp == 0.0


def test_equity_futures_charges():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    buy = eng.charges_for_fill(side=Side.BUY, qty=50, price=2500.0, segment=Segment.FUTURES)
    sell = eng.charges_for_fill(side=Side.SELL, qty=50, price=2550.0, segment=Segment.FUTURES)
    _approx(buy.brokerage, min(20.0, 0.0003 * 125_000))
    assert buy.stt == 0.0
    _approx(sell.stt, 0.0005 * 50 * 2550)
    assert buy.dp == 0 and sell.dp == 0
    # Visible IPFT needs ~₹1cr turnover under ₹0.01/crore schedule
    big = eng.charges_for_fill(side=Side.BUY, qty=10_000, price=1000.0, segment=Segment.FUTURES)
    _approx(big.ipft, 0.01)  # 1 crore * 0.01 / crore


def test_equity_options_charges():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    # Premium ₹100, qty 50
    buy = eng.charges_for_fill(side=Side.BUY, qty=50, price=100.0, segment=Segment.EQUITY_OPTIONS)
    sell = eng.charges_for_fill(side=Side.SELL, qty=50, price=120.0, segment=Segment.EQUITY_OPTIONS)
    _approx(buy.brokerage, 20.0)  # flat
    assert buy.stt == 0.0
    _approx(sell.stt, 0.0015 * 50 * 120)
    _approx(buy.ipft, 5000 * 50.0 / 10_000_000)


def test_brokerage_cap_boundaries():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    # Delivery: 0.3% of 1000 = 3 < 20 → pct wins
    small = eng.charges_for_fill(side=Side.BUY, qty=1, price=1000, segment=Segment.EQUITY_DELIVERY)
    _approx(small.brokerage, 3.0)
    # 0.3% of 100000 = 300 → capped at 20
    big = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY)
    _approx(big.brokerage, 20.0)
    # Intraday 0.03% of 100000 = 30 → 20
    intra = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_INTRADAY)
    _approx(intra.brokerage, 20.0)
    # Intraday small: 0.03% of 10000 = 3
    intra_s = eng.charges_for_fill(side=Side.BUY, qty=10, price=1000, segment=Segment.EQUITY_INTRADAY)
    _approx(intra_s.brokerage, 3.0)


def test_brokerage_aggregation_across_partial_fills():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    sess = eng.new_session()
    # Same order: two fills totaling 100 @ 1000 → brokerage once = 20
    f1 = eng.charges_for_fill(
        side=Side.BUY, qty=40, price=1000, segment=Segment.EQUITY_DELIVERY,
        order_id="O1", session=sess, symbol="AAA", trade_date="2026-11-14",
    )
    f2 = eng.charges_for_fill(
        side=Side.BUY, qty=60, price=1000, segment=Segment.EQUITY_DELIVERY,
        order_id="O1", session=sess, symbol="AAA", trade_date="2026-11-14",
    )
    _approx(f1.brokerage + f2.brokerage, 20.0)
    _approx(f2.brokerage_order_total, 20.0)
    # Separate orders → two brokerages
    sess2 = eng.new_session()
    a = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY,
        order_id="A", session=sess2, symbol="AAA", trade_date="2026-11-14",
    )
    b = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY,
        order_id="B", session=sess2, symbol="AAA", trade_date="2026-11-14",
    )
    _approx(a.brokerage + b.brokerage, 40.0)


def test_dp_once_per_isin_per_day_and_different_isins(tmp_path: Path):
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp.db")
    sess = eng.new_session()
    s1 = eng.charges_for_fill(
        side=Side.SELL, qty=40, price=1100, segment=Segment.EQUITY_DELIVERY,
        order_id="S1", session=sess, symbol="RELIANCE", trade_date="2026-11-14",
    )
    s2 = eng.charges_for_fill(
        side=Side.SELL, qty=60, price=1120, segment=Segment.EQUITY_DELIVERY,
        order_id="S2", session=sess, symbol="RELIANCE", trade_date="2026-11-14",
    )
    _approx(s1.dp, 12.5)
    assert s2.dp == 0.0  # same ISIN same day
    other = eng.charges_for_fill(
        side=Side.SELL, qty=10, price=500, segment=Segment.EQUITY_DELIVERY,
        order_id="S3", session=sess, symbol="TCS", trade_date="2026-11-14",
    )
    _approx(other.dp, 12.5)
    next_day = eng.charges_for_fill(
        side=Side.SELL, qty=10, price=1100, segment=Segment.EQUITY_DELIVERY,
        order_id="S4", session=sess, symbol="RELIANCE", trade_date="2026-11-15",
    )
    _approx(next_day.dp, 12.5)


def test_no_dp_on_intraday_or_futures():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    sess = eng.new_session()
    i = eng.charges_for_fill(
        side=Side.SELL, qty=10, price=100, segment=Segment.EQUITY_INTRADAY,
        session=sess, symbol="X", trade_date="2026-11-14", order_id="i1",
    )
    f = eng.charges_for_fill(
        side=Side.SELL, qty=10, price=100, segment=Segment.FUTURES,
        session=sess, symbol="X", trade_date="2026-11-14", order_id="f1",
    )
    assert i.dp == 0 and f.dp == 0


def test_gst_rounding_and_ipft():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    c = eng.charges_for_fill(side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY)
    assert c.gst == round(c.gst, 2)
    assert c.total_fees == round(c.total_fees, 2)
    # ₹1 crore delivery: IPFT = ₹0.01
    big = eng.charges_for_fill(side=Side.BUY, qty=10_000, price=1000, segment=Segment.EQUITY_DELIVERY)
    assert big.ipft == pytest.approx(0.01)


def test_partial_and_complete_exit_reconciliation(tmp_path: Path):
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp.db")
    sess = eng.new_session()
    buy = eng.economics_for_fill(
        side=Side.BUY, qty=100, raw_price=1000, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="B", symbol="AAA", trade_date="2026-11-14",
    )
    s1 = eng.economics_for_fill(
        side=Side.SELL, qty=40, raw_price=1100, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="S1", symbol="AAA", trade_date="2026-11-14",
    )
    s2 = eng.economics_for_fill(
        side=Side.SELL, qty=60, raw_price=1120, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="S2", symbol="AAA", trade_date="2026-11-14",
    )
    rt = eng.reconcile_round_trip(buy=buy, sells=[s1, s2])
    assert rt.qty_sold == 100
    assert s1.charges.dp > 0 and s2.charges.dp == 0
    _approx(buy.cash_delta + s1.cash_delta + s2.cash_delta, rt.net_pnl)
    assert rt.net_pnl < rt.gross_pnl
    # Separate brokerage on two sell orders
    assert s1.charges.brokerage > 0 and s2.charges.brokerage > 0


def test_reject_oversell_and_segment_mismatch(tmp_path: Path):
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp.db")
    buy = eng.economics_for_fill(
        side=Side.BUY, qty=10, raw_price=100, segment=Segment.EQUITY_DELIVERY, symbol="AAA"
    )
    sell = eng.economics_for_fill(
        side=Side.SELL, qty=11, raw_price=110, segment=Segment.EQUITY_DELIVERY, symbol="AAA"
    )
    with pytest.raises(ValueError, match="exceeds"):
        eng.reconcile_round_trip(buy=buy, sells=[sell])
    sell_i = eng.economics_for_fill(
        side=Side.SELL, qty=10, raw_price=110, segment=Segment.EQUITY_INTRADAY, symbol="AAA"
    )
    with pytest.raises(ValueError, match="segment"):
        eng.reconcile_round_trip(buy=buy, sells=[sell_i])


def test_missing_isin_raises_on_delivery_sell():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    with pytest.raises(ValueError, match="ISIN or canonical"):
        eng.charges_for_fill(side=Side.SELL, qty=10, price=100, segment=Segment.EQUITY_DELIVERY)


def test_dp_persists_across_engine_restarts(tmp_path: Path):
    ledger = tmp_path / "dp.db"
    eng1 = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=ledger)
    s1 = eng1.charges_for_fill(
        side=Side.SELL, qty=10, price=100, segment=Segment.EQUITY_DELIVERY,
        symbol="RELIANCE", trade_date="2026-11-14", order_id="a",
    )
    assert s1.dp == 12.5
    eng2 = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=ledger)
    s2 = eng2.charges_for_fill(
        side=Side.SELL, qty=5, price=100, segment=Segment.EQUITY_DELIVERY,
        symbol="RELIANCE", trade_date="2026-11-14", order_id="b",
    )
    assert s2.dp == 0.0


def test_mtm_estimate_does_not_mutate_session_or_dp(tmp_path: Path):
    ledger = tmp_path / "dp2.db"
    eng = ChargeEngine("fyers", slippage_pct=0.001, dp_ledger_path=ledger)
    sess = eng.new_session()
    # Real sell charges DP
    real = eng.economics_for_fill(
        side=Side.SELL, qty=10, raw_price=1000, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="REAL", symbol="AAA", trade_date="2026-11-14",
    )
    assert real.charges.dp == 12.5
    orders_before = dict(sess._orders)
    dp_before = set(sess._dp_charged)
    est = eng.estimate_economics_for_fill(
        side=Side.SELL, qty=10, raw_price=1000, segment=Segment.EQUITY_DELIVERY,
        symbol="AAA", trade_date="2026-11-14",
    )
    assert est.charges.dp == 0.0  # already charged — estimate peeks ledger
    assert sess._orders == orders_before
    assert sess._dp_charged == dp_before
    assert est.slippage_rupees > 0


def test_historical_schedule_boundaries():
    with pytest.raises(ValueError, match="2026-04-01|No verified"):
        ChargeEngine("fyers", as_of="2026-03-31")
    ok = ChargeEngine("fyers", as_of="2026-04-01")
    assert ok.schedule.effective_date.isoformat() == "2026-04-01"
    assumed = ChargeEngine("fyers", as_of="2025-01-01", allow_assumed_schedule=True)
    assert assumed.assumed_schedule is True


def test_exchange_selection_and_rejection():
    nse = ChargeEngine("fyers", slippage_pct=0.0, exchange="NSE")
    bse = ChargeEngine("fyers", slippage_pct=0.0, exchange="BSE")
    n = nse.charges_for_fill(
        side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY, symbol="X"
    )
    b = bse.charges_for_fill(
        side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY, symbol="X"
    )
    assert b.exchange_txn != n.exchange_txn
    with pytest.raises(ValueError, match="Unsupported exchange"):
        ChargeEngine("fyers", exchange="MCX")
    with pytest.raises(ValueError, match="BSE F&O"):
        bse.charges_for_fill(side=Side.BUY, qty=1, price=100, segment=Segment.FUTURES, symbol="X")


def test_gst_base_excludes_stt_and_stamp():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    c = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=1000, segment=Segment.EQUITY_DELIVERY, symbol="X"
    )
    info = eng.schedule_info()
    assert "stt" in info["gst_excludes"] and "stamp" in info["gst_excludes"]
    expected = (c.brokerage + c.exchange_txn + c.clearing + c.sebi + c.ipft) * 0.18
    _approx(c.gst, expected, tol=0.02)
    # STT and stamp are positive but not in GST base
    assert c.stt > 0 and c.stamp > 0


def test_portfolio_bt_cost_model_flagged():
    from ati.research.backtest import EVENT_BASELINE_COST_MODEL, PORTFOLIO_BT_COST_MODEL

    assert EVENT_BASELINE_COST_MODEL == "charge_engine_fill_level"
    assert "approximate" in PORTFOLIO_BT_COST_MODEL


def test_paper_partial_exit_fee_alloc(tmp_path: Path):
    ledger = tmp_path / "dp3.db"
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=ledger)
    broker = PaperBroker(
        db_path=tmp_path / "paper.db",
        allowed_strategy_substrings=(),
        charges=eng,
    )
    ok, msg = broker.try_enter(_opp())
    assert ok, msg
    pos = broker.positions()[0]
    entry_fees = pos.entry_fees
    # Simulate first partial via charges session (manage_exits needs bars)
    sell = eng.economics_for_fill(
        side=Side.SELL, qty=20, raw_price=1040, segment=Segment.EQUITY_DELIVERY,
        session=broker.charge_session, order_id="px1", symbol="TEST.NS", trade_date="2026-11-14",
    )
    assert sell.charges.dp == 12.5
    sell2 = eng.economics_for_fill(
        side=Side.SELL, qty=20, raw_price=1040, segment=Segment.EQUITY_DELIVERY,
        session=broker.charge_session, order_id="px2", symbol="TEST.NS", trade_date="2026-11-14",
    )
    assert sell2.charges.dp == 0.0
    assert entry_fees > 0


def test_slippage_not_in_fee_lines():
    eng = ChargeEngine("fyers", slippage_pct=0.001)
    eco = eng.economics_for_fill(side=Side.BUY, qty=10, raw_price=1000, segment=Segment.EQUITY_DELIVERY)
    assert eco.slippage_rupees == pytest.approx(10.0)
    fee_sum = (
        eco.charges.brokerage + eco.charges.stt + eco.charges.exchange_txn + eco.charges.clearing
        + eco.charges.sebi + eco.charges.stamp + eco.charges.ipft + eco.charges.gst
        + eco.charges.dp + eco.charges.other
    )
    _approx(fee_sum, eco.total_fees)


def test_historical_schedule_selection():
    # Before 2026-04-01: only schedule starts 2026-04-01 for fyers — should still resolve
    # (single open-ended schedule). as_of after effective works.
    s = resolve_schedule("fyers", as_of="2026-06-01")
    assert s.effective_date == date(2026, 4, 1)
    with pytest.raises(ValueError):
        resolve_schedule("fyers", as_of="2020-01-01")


def test_event_charges_not_invented_unless_requested():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    normal = eng.charges_for_fill(side=Side.BUY, qty=10, price=100, segment=Segment.EQUITY_DELIVERY)
    assert normal.other == 0.0
    cat = eng.charges_for_fill(
        side=Side.BUY, qty=10, price=100, segment=Segment.EQUITY_DELIVERY, event="call_and_trade"
    )
    _approx(cat.other, 50.0)


def test_paper_cash_ledger_reconciles(tmp_path: Path):
    broker = PaperBroker(db_path=tmp_path / "p.db", allowed_strategy_substrings=())
    start = broker.account()["cash"]
    ok, msg = broker.try_enter(_opp())
    assert ok, msg
    after = broker.account()["cash"]
    assert after < start
    pos = broker.positions()[0]
    assert pos.entry_fees > 0
    # Cash drop equals notional+fees from last buy fill
    import sqlite3

    with sqlite3.connect(tmp_path / "p.db") as cx:
        row = cx.execute(
            "SELECT price, qty, fees FROM fills WHERE side='BUY' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    expected = start - (row[0] * row[1] + row[2])
    _approx(after, expected, tol=0.05)


def test_mutual_fund_zero_brokerage():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    c = eng.charges_for_fill(side=Side.BUY, qty=10, price=100, segment=Segment.MUTUAL_FUND)
    assert c.brokerage == 0 and c.total_fees == 0
