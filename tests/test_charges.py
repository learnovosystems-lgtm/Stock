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
    # Options IPFT ₹0.01/crore (charges-list + NSE/FA/73061); NOT ₹50/crore KB remnant
    assert fy.equity_options.ipft_per_crore == pytest.approx(0.01)
    assert fy.equity_options.ipft_per_crore != pytest.approx(50.0)
    assert "0.01" in fy.notes and "50/crore" in fy.notes
    assert "NSE/FA/73061" in fy.notes or any("73061" in u for u in fy.source_urls)
    assert fy.effective_date >= date(2026, 4, 1)  # after options IPFT revision 2026-03-01


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
    # Small premium TO → IPFT rounds to 0 at 2dp under ₹0.01/crore
    assert buy.ipft == 0.0
    # Visible options IPFT: ₹1 crore premium * ₹0.01/crore = ₹0.01
    big = eng.charges_for_fill(
        side=Side.BUY, qty=100_000, price=100.0, segment=Segment.EQUITY_OPTIONS
    )
    _approx(big.ipft, 0.01)
    # Must NOT use legacy ₹50/crore (would be ₹50 on ₹1cr)
    assert big.ipft != pytest.approx(50.0)


def test_options_ipft_effective_date_and_rate():
    """Options IPFT ₹0.01/crore for schedule effective after NSE/FA/73061 (2026-03-01)."""
    fy = resolve_schedule("fyers", as_of="2026-04-01")
    assert fy.effective_date == date(2026, 4, 1)
    assert fy.equity_options.ipft_per_crore == pytest.approx(0.01)
    info = ChargeEngine("fyers", as_of="2026-04-01").schedule_info()
    assert "0.01" in info["ipft_policy"]
    assert "50/crore" in info["ipft_policy"]  # discrepancy documented, not applied
    assert "73061" in info["ipft_policy"] or any("73061" in u for u in info["source_urls"])


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


def test_concurrent_dp_exactly_once(tmp_path: Path):
    """Simultaneous delivery sells for same ISIN/date → exactly one DP debit."""
    import concurrent.futures

    from ati.engine.dp_ledger import try_record_dp_charge

    ledger = tmp_path / "dp_conc.db"
    results: list[bool] = []

    def claim() -> bool:
        return try_record_dp_charge(
            broker="fyers",
            instrument_id="INE002A01018",
            trade_date="2026-11-14",
            path=ledger,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(claim) for _ in range(32)]
        results = [f.result() for f in futs]
    assert sum(1 for r in results if r) == 1
    assert sum(1 for r in results if not r) == 31

    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp_eng.db")
    dps: list[float] = []

    def sell_once(i: int) -> float:
        c = eng.charges_for_fill(
            side=Side.SELL,
            qty=10,
            price=100,
            segment=Segment.EQUITY_DELIVERY,
            isin="INE002A01018",
            trade_date="2026-11-14",
            order_id=f"conc-{i}",
        )
        return c.dp

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(sell_once, i) for i in range(24)]
        dps = [f.result() for f in futs]
    assert sum(1 for d in dps if d == 12.5) == 1
    assert sum(1 for d in dps if d == 0.0) == 23


def test_dp_failed_fill_releases_reservation(tmp_path: Path):
    """Failed sell must not permanently consume DP; retry can charge once."""
    from ati.engine.dp_ledger import dp_already_charged

    ledger = tmp_path / "dp_fail.db"
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=ledger)
    fill_id = "sell-attempt-1"
    c = eng.charges_for_fill(
        side=Side.SELL,
        qty=10,
        price=100,
        segment=Segment.EQUITY_DELIVERY,
        symbol="RELIANCE",
        trade_date="2026-11-14",
        defer_dp_confirm=True,
        dp_fill_id=fill_id,
        order_id=fill_id,
    )
    assert c.dp == 12.5
    assert not dp_already_charged(
        broker="fyers", instrument_id="RELIANCE", trade_date="2026-11-14", path=ledger
    )
    # Simulate failed fill commit
    assert eng.release_dp_for_fill(
        instrument_id="RELIANCE", trade_date="2026-11-14", fill_id=fill_id
    )
    # Retry succeeds and can confirm
    fill2 = "sell-attempt-2"
    c2 = eng.charges_for_fill(
        side=Side.SELL,
        qty=10,
        price=100,
        segment=Segment.EQUITY_DELIVERY,
        symbol="RELIANCE",
        trade_date="2026-11-14",
        defer_dp_confirm=True,
        dp_fill_id=fill2,
        order_id=fill2,
    )
    assert c2.dp == 12.5
    assert eng.confirm_dp_for_fill(
        instrument_id="RELIANCE", trade_date="2026-11-14", fill_id=fill2
    )
    assert dp_already_charged(
        broker="fyers", instrument_id="RELIANCE", trade_date="2026-11-14", path=ledger
    )
    # Third attempt: no DP
    c3 = eng.charges_for_fill(
        side=Side.SELL,
        qty=5,
        price=100,
        segment=Segment.EQUITY_DELIVERY,
        symbol="RELIANCE",
        trade_date="2026-11-14",
        order_id="sell-3",
    )
    assert c3.dp == 0.0


def test_realistic_segment_examples_and_partial_exit(tmp_path: Path):
    eng = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp.db")
    # Delivery
    d_buy = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=2500, segment=Segment.EQUITY_DELIVERY, symbol="INFY"
    )
    d_sell = eng.charges_for_fill(
        side=Side.SELL, qty=100, price=2600, segment=Segment.EQUITY_DELIVERY, symbol="INFY",
        trade_date="2026-11-14",
    )
    assert d_buy.brokerage > 0 and d_sell.dp == 12.5 and d_sell.stt > 0
    # Intraday
    i_buy = eng.charges_for_fill(
        side=Side.BUY, qty=100, price=2500, segment=Segment.EQUITY_INTRADAY, symbol="INFY"
    )
    i_sell = eng.charges_for_fill(
        side=Side.SELL, qty=100, price=2510, segment=Segment.EQUITY_INTRADAY, symbol="INFY"
    )
    assert i_buy.stt == 0 and i_sell.dp == 0 and i_sell.stt > 0
    # Futures
    f_buy = eng.charges_for_fill(
        side=Side.BUY, qty=50, price=25000, segment=Segment.FUTURES, symbol="NIFTY"
    )
    assert f_buy.clearing > 0 and f_buy.dp == 0
    # Partial exits — DP once
    eng2 = ChargeEngine("fyers", slippage_pct=0.0, dp_ledger_path=tmp_path / "dp2.db")
    sess = eng2.new_session()
    buy = eng2.economics_for_fill(
        side=Side.BUY, qty=100, raw_price=1000, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="b1", symbol="TCS",
    )
    s1 = eng2.economics_for_fill(
        side=Side.SELL, qty=40, raw_price=1100, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="s1", symbol="TCS", trade_date="2026-11-14",
    )
    s2 = eng2.economics_for_fill(
        side=Side.SELL, qty=60, raw_price=1120, segment=Segment.EQUITY_DELIVERY,
        session=sess, order_id="s2", symbol="TCS", trade_date="2026-11-14",
    )
    assert s1.charges.dp == 12.5 and s2.charges.dp == 0.0
    rt = eng2.reconcile_round_trip(buy=buy, sells=[s1, s2])
    assert abs(rt.net_pnl - (rt.gross_pnl - rt.total_fees - rt.total_slippage)) < 0.05


def test_portfolio_book_cash_reconciliation(tmp_path: Path):
    from ati.research.portfolio_charges import PortfolioChargeBook

    book = PortfolioChargeBook.create(
        initial_capital=100_000, dp_ledger_path=tmp_path / "dp.db", slippage_pct=0.0
    )
    start = book.cash
    assert book.buy("AAA", 50, 1000, trade_date="2026-11-10", bar_i=0, order_id="b")
    after_buy = book.cash
    assert after_buy < start
    buy_fees = start - after_buy - 50 * 1000  # fees portion of cash drop at raw=exec (slip 0)
    t = book.sell(
        "AAA", 50, 1100, trade_date="2026-11-14", bar_i=5, order_id="s", reason="exit"
    )
    assert t is not None
    assert abs(t["pnl"] - (t["gross_pnl"] - t["fees"])) < 0.05
    # Final cash = start - buy_cost + sell_proceeds_net_of_exit_fees
    # With slip=0: start - 50*1000 - entry_fees + 50*1100 - exit_fees
    expected = start - 50 * 1000 - t["entry_fees"] + 50 * 1100 - t["exit_fees"]
    _approx(book.cash, expected, tol=0.05)
    _approx(book.cash, start + t["gross_pnl"] - t["fees"], tol=0.05)
    assert t["dp"] == 12.5
    assert t["cost_model"] == "charge_engine_fill_level"
    assert buy_fees == pytest.approx(t["entry_fees"], abs=0.05)


def test_gst_charges_list_base_documented():
    eng = ChargeEngine("fyers", slippage_pct=0.0)
    info = eng.schedule_info()
    assert info["gst_base_components"] == [
        "brokerage",
        "exchange_txn",
        "clearing",
        "sebi",
        "ipft",
        "dp",
    ]
    assert "stt" in info["gst_excludes"] and "stamp" in info["gst_excludes"]
    assert "charges-list" in info["gst_source"].lower()
    assert "statutory kb" in info["gst_source"].lower()
    # Large turnover so IPFT is visible in GST base
    c = eng.charges_for_fill(
        side=Side.BUY, qty=10_000, price=1000.0, segment=Segment.EQUITY_DELIVERY, symbol="GST"
    )
    expected = (c.brokerage + c.exchange_txn + c.clearing + c.sebi + c.ipft) * 0.18
    _approx(c.gst, expected, tol=0.02)
    assert c.stt > 0 and c.stamp > 0


def test_portfolio_bt_fill_level_charge_engine():
    from ati.research.adaptive import PORTFOLIO_BT_COST_MODEL as ADAPTIVE_FLAG
    from ati.research.backtest import (
        EVENT_BASELINE_COST_MODEL,
        PORTFOLIO_BT_COST_MODEL,
        W4_OUTSTANDING_PORTFOLIO_BT,
    )
    from ati.research.portfolio_charges import PortfolioChargeBook

    assert EVENT_BASELINE_COST_MODEL == "charge_engine_fill_level"
    assert PORTFOLIO_BT_COST_MODEL == "charge_engine_fill_level"
    assert ADAPTIVE_FLAG == PORTFOLIO_BT_COST_MODEL
    assert W4_OUTSTANDING_PORTFOLIO_BT is False
    book = PortfolioChargeBook.create(initial_capital=100_000)
    assert book.cost_model == "charge_engine_fill_level"
    info = ChargeEngine("fyers").schedule_info()
    assert info.get("w4_outstanding") is None
    assert info["portfolio_bt_cost_model"] == "charge_engine_fill_level"
