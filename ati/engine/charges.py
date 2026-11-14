"""Broker-configurable ChargeEngine — FYERS Standard (default) / Zerodha profiles.

Rules (Phase-1 corrections):
- Brokerage is assessed at the **executed-order** level (partial fills of the same
  order_id share one brokerage = min(cap, pct × order turnover)).
- IPFT for NSE equity/intraday/futures/options (on premium): ₹0.01/crore from
  fyers.in/charges-list; options aligned with NSE Circular NSE/FA/73061
  (27 Feb 2026; eff. 1 Mar 2026). Statutory KB still cites ₹10/crore (equity/FUT)
  and ₹50/crore (options) — charges-list + NSE circular win; KB not used.
- GST taxable base per charges-list: brokerage + exchange txn + clearing + SEBI
  turnover + NSE IPFT (+ DP GST when charged). STT and stamp excluded. Statutory
  KB describes GST differently — charges-list treatment used for this schedule.
- DP ₹12.50+GST once per ISIN per trading day on delivery sells only (atomic ledger).
- Slippage is price-only — never folded into fee lines.
- Optional event charges (call-and-trade, square-off, pledge, ledger interest)
  exist as hooks; not applied unless explicitly requested.

Sources:
- https://fyers.in/charges-list
- https://fyers.in/pricing
- https://support.fyers.in/portal/en/kb/articles/what-are-the-statutory-charges-on-trades-at-fyers-stt-gst-sebi-turnover-fee-ipft-etc
- NSE Circular NSE/FA/73061 (2026-02-27)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from ati.engine.dp_ledger import dp_already_charged, try_record_dp_charge

from ati.models.charges import (
    BrokerId,
    ChargeBreakdown,
    FillEconomics,
    RoundTripEconomics,
    Segment,
    Side,
)


CRORE = 10_000_000.0
IST = ZoneInfo("Asia/Kolkata")
MONEY_QUANT = Decimal("0.01")


def _money(x: float | Decimal) -> float:
    return float(Decimal(str(x)).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class _SegmentRates:
    brokerage_flat: float
    brokerage_pct: float  # 0 => flat-only (or zero)
    stt_pct: float
    stt_on_buy: bool
    stt_on_sell: bool
    exchange_txn_pct: float
    clearing_pct: float
    sebi_per_crore: float
    stamp_pct_buy: float
    ipft_per_crore: float
    gst_rate: float
    dp_flat: float
    dp_gst: bool
    # Turnover base for STT/exch: "notional" (qty*price) or "premium" (options)
    turnover_basis: str = "notional"


@dataclass(frozen=True)
class FeeSchedule:
    broker: str
    schedule_id: str
    effective_date: date
    end_date: date | None
    source_urls: tuple[str, ...]
    equity_delivery: _SegmentRates
    equity_intraday: _SegmentRates
    futures: _SegmentRates
    equity_options: _SegmentRates
    mutual_fund: _SegmentRates
    # Segment rates.exchange_txn_pct are NSE defaults. BSE equity override from charges-list.
    supported_exchanges: tuple[str, ...] = ("NSE", "BSE")
    bse_equity_exchange_txn_pct: float = 0.00375 / 100.0  # fyers.in/charges-list
    # Optional event rates (applied only when event requested)
    call_and_trade: float = 50.0
    admin_square_off: float = 50.0
    pledge_unpledge: float = 0.0
    ledger_interest_daily_pct: float = 0.00041  # 0.041%/day FYERS
    account_opening: float = 0.0
    amc: float = 0.0
    notes: str = ""
    # GST taxable base (official FYERS charges-list): brokerage + exchange txn +
    # clearing + SEBI + IPFT (+ DP when charged). NOT STT. NOT stamp duty.
    gst_base_components: tuple[str, ...] = (
        "brokerage",
        "exchange_txn",
        "clearing",
        "sebi",
        "ipft",
        "dp",
    )


def _fyers_standard_2026_04_01() -> FeeSchedule:
    """FYERS Standard plan — effective 2026-04-01 (STT F&O revision date).

    IPFT (all NSE equity segments including options on premium): ₹0.01/Crore
    from fyers.in/charges-list, aligned with NSE Circular NSE/FA/73061
    (27 Feb 2026) revising equity options IPFT to ₹0.01/crore effective
    1 Mar 2026. This schedule begins 2026-04-01, so options use ₹0.01/crore.

    Discrepancies with FYERS statutory help article (still cites equity/FUT
    ₹10/crore and options ₹50/crore in places): charges-list + NSE circular
    win; KB values are NOT used.
    """
    gst = 0.18
    sebi = 10.0
    # ₹ per crore — fyers.in/charges-list + NSE/FA/73061 (options eff. 2026-03-01)
    ipft_eq = 0.01
    ipft_opt = 0.01
    eq_exch = 0.0030699 / 100.0
    fut_exch = 0.0018299 / 100.0
    opt_exch = 0.0355299 / 100.0  # on premium
    delivery = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.003,
        stt_pct=0.001,
        stt_on_buy=True,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00015,
        ipft_per_crore=ipft_eq,
        gst_rate=gst,
        dp_flat=12.5,
        dp_gst=True,
    )
    intraday = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,
        stt_pct=0.00025,
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00003,
        ipft_per_crore=ipft_eq,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    futures = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,
        stt_pct=0.0005,  # 0.05% sell — revised 2026-04-01
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=fut_exch,
        clearing_pct=0.0005 / 100.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00002,
        ipft_per_crore=ipft_eq,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    options = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0,  # flat ₹20
        stt_pct=0.0015,  # 0.15% on premium, sell
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=opt_exch,
        clearing_pct=0.009 / 100.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00003,
        ipft_per_crore=ipft_opt,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
        turnover_basis="premium",
    )
    mf = _SegmentRates(
        brokerage_flat=0.0,
        brokerage_pct=0.0,
        stt_pct=0.0,
        stt_on_buy=False,
        stt_on_sell=False,
        exchange_txn_pct=0.0,
        clearing_pct=0.0,
        sebi_per_crore=0.0,
        stamp_pct_buy=0.0,
        ipft_per_crore=0.0,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    return FeeSchedule(
        broker=BrokerId.FYERS.value,
        schedule_id="fyers_standard_2026_04_01",
        effective_date=date(2026, 4, 1),
        end_date=None,
        source_urls=(
            "https://fyers.in/charges-list",
            "https://fyers.in/pricing",
            "https://support.fyers.in/portal/en/kb/articles/what-are-the-statutory-charges-on-trades-at-fyers-stt-gst-sebi-turnover-fee-ipft-etc",
            "NSE Circular NSE/FA/73061 (2026-02-27; equity options IPFT ₹0.01/crore eff. 2026-03-01)",
        ),
        equity_delivery=delivery,
        equity_intraday=intraday,
        futures=futures,
        equity_options=options,
        mutual_fund=mf,
        notes=(
            "FYERS Standard. Account opening/AMC ₹0. Prime not modeled. "
            "IPFT NSE (delivery/intraday/futures/options on premium) = ₹0.01/crore "
            "from fyers.in/charges-list; options aligned with NSE/FA/73061 "
            "(eff. 2026-03-01). Statutory KB still cites ₹10/crore (equity/FUT) "
            "and ₹50/crore (options) — discrepancy documented; charges-list used. "
            "GST base per charges-list: brokerage + exchange txn + clearing + SEBI + "
            "NSE IPFT (+ DP GST when charged); STT and stamp excluded. Statutory KB "
            "describes GST differently — charges-list treatment used. "
            "F&O STT rates effective 2026-04-01."
        ),
    )


def _zerodha_equity_2024_10_01() -> FeeSchedule:
    """Zerodha alternate profile — never selected when charges_broker=fyers."""
    gst = 0.18
    sebi = 10.0
    ipft = 10.0
    eq_exch = 0.00297 / 100.0
    fut_exch = 0.00173 / 100.0
    delivery = _SegmentRates(
        brokerage_flat=0.0,
        brokerage_pct=0.0,
        stt_pct=0.001,
        stt_on_buy=True,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00015,
        ipft_per_crore=ipft,
        gst_rate=gst,
        dp_flat=13.5,
        dp_gst=True,
    )
    intraday = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,
        stt_pct=0.00025,
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00003,
        ipft_per_crore=ipft,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    futures = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,
        stt_pct=0.0005,
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=fut_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00002,
        ipft_per_crore=ipft,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    options = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0,
        stt_pct=0.0015,
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=0.03503 / 100.0,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00003,
        ipft_per_crore=50.0,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
        turnover_basis="premium",
    )
    mf = _SegmentRates(
        brokerage_flat=0.0,
        brokerage_pct=0.0,
        stt_pct=0.0,
        stt_on_buy=False,
        stt_on_sell=False,
        exchange_txn_pct=0.0,
        clearing_pct=0.0,
        sebi_per_crore=0.0,
        stamp_pct_buy=0.0,
        ipft_per_crore=0.0,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    return FeeSchedule(
        broker=BrokerId.ZERODHA.value,
        schedule_id="zerodha_equity_2024_10_01",
        effective_date=date(2024, 10, 1),
        end_date=None,
        source_urls=("https://zerodha.com/charges",),
        equity_delivery=delivery,
        equity_intraday=intraday,
        futures=futures,
        equity_options=options,
        mutual_fund=mf,
        notes="Alternate broker profile only. Never applied when charges_broker=fyers.",
    )


_SCHEDULES: tuple[FeeSchedule, ...] = (
    _fyers_standard_2026_04_01(),
    _zerodha_equity_2024_10_01(),
)


def list_schedules() -> list[FeeSchedule]:
    return list(_SCHEDULES)


def _parse_as_of(as_of: date | datetime | str | None) -> date:
    if as_of is None:
        return datetime.now(IST).date()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    return date.fromisoformat(str(as_of)[:10])


def resolve_schedule(
    broker: str,
    as_of: date | datetime | str | None = None,
    *,
    allow_assumed: bool = False,
) -> FeeSchedule:
    """Resolve versioned fee schedule.

    FYERS Standard is pinned from 2026-04-01. Dates before that raise unless
    allow_assumed=True (explicitly flags an assumed/unsupported historical use).
    """
    broker_l = (broker or BrokerId.FYERS.value).strip().lower()
    d = _parse_as_of(as_of)
    matches = [
        s
        for s in _SCHEDULES
        if s.broker == broker_l and s.effective_date <= d and (s.end_date is None or d < s.end_date)
    ]
    if not matches:
        if allow_assumed:
            # Caller opted in — use earliest schedule and mark via notes (caller must flag)
            candidates = [s for s in _SCHEDULES if s.broker == broker_l]
            if not candidates:
                raise ValueError(f"No fee schedule for broker={broker_l!r}")
            candidates.sort(key=lambda s: s.effective_date)
            return candidates[0]
        raise ValueError(
            f"No verified fee schedule for broker={broker_l!r} as_of={d.isoformat()}. "
            f"FYERS Standard begins 2026-04-01; pass allow_assumed=True only when the "
            f"backtest explicitly flags an assumed schedule."
        )
    matches.sort(key=lambda s: s.effective_date)
    return matches[-1]


def _rates_for(schedule: FeeSchedule, segment: Segment | str) -> _SegmentRates:
    seg = Segment(segment) if not isinstance(segment, Segment) else segment
    mapping = {
        Segment.EQUITY_DELIVERY: schedule.equity_delivery,
        Segment.EQUITY_INTRADAY: schedule.equity_intraday,
        Segment.FUTURES: schedule.futures,
        Segment.EQUITY_OPTIONS: schedule.equity_options,
        Segment.MUTUAL_FUND: schedule.mutual_fund,
    }
    if seg not in mapping:
        raise ValueError(f"Unsupported segment: {segment}")
    return mapping[seg]


def _brokerage_for_turnover(rates: _SegmentRates, turnover: float) -> float:
    if rates.brokerage_pct <= 0 and rates.brokerage_flat <= 0:
        return 0.0
    if rates.brokerage_pct <= 0:
        return _money(rates.brokerage_flat)
    pct_amt = turnover * rates.brokerage_pct
    if rates.brokerage_flat <= 0:
        return _money(pct_amt)
    return _money(min(rates.brokerage_flat, pct_amt))


def canonical_instrument_id(isin: str | None = None, symbol: str | None = None) -> str | None:
    """Prefer ISIN; else canonical symbol (exchange suffix stripped)."""
    if isin and str(isin).strip():
        return str(isin).strip().upper()
    if symbol and str(symbol).strip():
        return str(symbol).replace(".NS", "").replace(".BO", "").strip().upper()
    return None


def _normalize_isin(isin: str | None, symbol: str | None = None) -> str | None:
    return canonical_instrument_id(isin, symbol)


def _parse_trade_date(trade_date: date | datetime | str | None) -> date:
    if trade_date is None:
        return datetime.now(IST).date()
    if isinstance(trade_date, datetime):
        return trade_date.astimezone(IST).date() if trade_date.tzinfo else trade_date.date()
    if isinstance(trade_date, date):
        return trade_date
    return date.fromisoformat(str(trade_date)[:10])


@dataclass
class _OrderState:
    side: Side
    segment: Segment
    turnover: float = 0.0
    brokerage_allocated: float = 0.0


@dataclass
class ChargeSession:
    """Tracks per-order brokerage aggregation and per-ISIN/day DP."""

    engine: "ChargeEngine"
    _orders: dict[str, _OrderState] = field(default_factory=dict)
    _dp_charged: set[tuple[str, date]] = field(default_factory=set)

    def fill(
        self,
        *,
        side: Side | str,
        qty: float,
        price: float,
        segment: Segment | str,
        order_id: str | None = None,
        isin: str | None = None,
        symbol: str | None = None,
        trade_date: date | datetime | str | None = None,
        event: str | None = None,
    ) -> ChargeBreakdown:
        return self.engine.charges_for_fill(
            side=side,
            qty=qty,
            price=price,
            segment=segment,
            order_id=order_id,
            isin=isin,
            symbol=symbol,
            trade_date=trade_date,
            session=self,
            event=event,
        )

    def economics_for_fill(self, **kwargs) -> FillEconomics:
        return self.engine.economics_for_fill(session=self, **kwargs)


class ChargeEngine:
    """Single calculation path for backtest, paper, journal, and dashboard."""

    def __init__(
        self,
        broker: str = BrokerId.FYERS.value,
        *,
        as_of: date | datetime | str | None = None,
        slippage_pct: float = 0.0005,
        exchange: str = "NSE",
        allow_assumed_schedule: bool = False,
        dp_ledger_path: Path | None = None,
    ):
        self.broker = (broker or BrokerId.FYERS.value).strip().lower()
        self.as_of = _parse_as_of(as_of)
        self.slippage_pct = float(slippage_pct)
        self.exchange = (exchange or "NSE").strip().upper()
        self.allow_assumed_schedule = allow_assumed_schedule
        self.dp_ledger_path = dp_ledger_path
        self.schedule = resolve_schedule(
            self.broker, self.as_of, allow_assumed=allow_assumed_schedule
        )
        if self.exchange not in self.schedule.supported_exchanges:
            raise ValueError(
                f"Unsupported exchange {self.exchange!r} for broker={self.broker}. "
                f"Supported: {self.schedule.supported_exchanges}"
            )
        self.assumed_schedule = allow_assumed_schedule and self.as_of < self.schedule.effective_date

    def new_session(self) -> ChargeSession:
        return ChargeSession(self)

    def with_as_of(self, as_of: date | datetime | str | None) -> "ChargeEngine":
        return ChargeEngine(
            self.broker,
            as_of=as_of,
            slippage_pct=self.slippage_pct,
            exchange=self.exchange,
            allow_assumed_schedule=self.allow_assumed_schedule,
            dp_ledger_path=self.dp_ledger_path,
        )

    def slipped_price(self, price: float, side: Side | str) -> float:
        s = Side(side) if not isinstance(side, Side) else side
        if price <= 0:
            return price
        if s == Side.BUY:
            return price * (1.0 + self.slippage_pct)
        return price * (1.0 - self.slippage_pct)

    def charges_for_fill(
        self,
        *,
        side: Side | str,
        qty: float,
        price: float,
        segment: Segment | str = Segment.EQUITY_DELIVERY,
        order_id: str | None = None,
        isin: str | None = None,
        symbol: str | None = None,
        trade_date: date | datetime | str | None = None,
        session: ChargeSession | None = None,
        apply_dp: bool | None = None,
        event: str | None = None,
    ) -> ChargeBreakdown:
        s = Side(side) if not isinstance(side, Side) else side
        seg = Segment(segment) if not isinstance(segment, Segment) else segment
        rates = _rates_for(self.schedule, seg)
        qty = float(qty)
        price = float(price)
        if qty <= 0 or price < 0 or (price == 0 and seg != Segment.MUTUAL_FUND):
            raise ValueError("qty must be positive and price non-negative")

        turnover = qty * price
        td = _parse_trade_date(trade_date)
        instrument_id = canonical_instrument_id(isin, symbol)

        # --- Brokerage (order-level) ---
        oid = order_id or f"__anon_{id(object())}_{s.value}_{seg.value}_{turnover}"
        if session is None:
            brokerage = _brokerage_for_turnover(rates, turnover)
            brokerage_order_total = brokerage
        else:
            st = session._orders.get(oid)
            if st is None:
                st = _OrderState(side=s, segment=seg)
                session._orders[oid] = st
            else:
                if st.side != s:
                    raise ValueError(f"Order {oid} side mismatch: {st.side} vs {s}")
                if st.segment != seg:
                    raise ValueError(f"Order {oid} segment mismatch: {st.segment} vs {seg}")
            st.turnover += turnover
            order_brokerage = _brokerage_for_turnover(rates, st.turnover)
            brokerage = _money(max(0.0, order_brokerage - st.brokerage_allocated))
            st.brokerage_allocated = order_brokerage
            brokerage_order_total = order_brokerage

        # --- Statutory on this fill ---
        stt = 0.0
        if s == Side.BUY and rates.stt_on_buy:
            stt = turnover * rates.stt_pct
        elif s == Side.SELL and rates.stt_on_sell:
            stt = turnover * rates.stt_pct

        exchange_txn_pct = self._exchange_txn_pct(seg, rates)
        exchange_txn = turnover * exchange_txn_pct
        clearing = turnover * rates.clearing_pct
        sebi = turnover * (rates.sebi_per_crore / CRORE)
        # IPFT applies to NSE only (FYERS charges-list / statutory KB)
        ipft = turnover * (rates.ipft_per_crore / CRORE) if self.exchange == "NSE" else 0.0
        stamp = turnover * rates.stamp_pct_buy if s == Side.BUY else 0.0

        # GST base per FYERS charges-list: brokerage + exch + clearing + SEBI + IPFT
        # (not STT, not stamp). DP GST added separately when DP applies.
        gst_base = brokerage + exchange_txn + clearing + sebi + ipft
        gst = gst_base * rates.gst_rate

        # --- DP once per instrument/day (delivery sell only); atomic ledger claim ---
        dp = 0.0
        charge_dp = False
        if seg == Segment.EQUITY_DELIVERY and s == Side.SELL and rates.dp_flat > 0:
            if apply_dp is False:
                charge_dp = False
            else:
                if not instrument_id:
                    raise ValueError(
                        "Delivery sell requires ISIN or canonical symbol for DP "
                        "deduplication; refusing to bypass DP aggregation"
                    )
                # Session fast-path: already claimed in this process
                if session is not None and (instrument_id, td) in session._dp_charged:
                    charge_dp = False
                else:
                    # Atomic insert-or-ignore — only the winner debits DP
                    won = try_record_dp_charge(
                        broker=self.broker,
                        instrument_id=instrument_id,
                        trade_date=td,
                        path=self.dp_ledger_path,
                    )
                    charge_dp = won
                    if won and session is not None:
                        session._dp_charged.add((instrument_id, td))
            if charge_dp:
                dp = rates.dp_flat
                if rates.dp_gst:
                    gst += rates.dp_flat * rates.gst_rate

        # --- Optional event charges (not invented for normal paper) ---
        other = 0.0
        if event == "call_and_trade":
            other = self.schedule.call_and_trade
            gst += other * rates.gst_rate
        elif event == "admin_square_off":
            other = self.schedule.admin_square_off
            gst += other * rates.gst_rate
        elif event in ("pledge", "unpledge") and self.schedule.pledge_unpledge > 0:
            other = self.schedule.pledge_unpledge
            gst += other * rates.gst_rate

        brokerage = _money(brokerage)
        stt = _money(stt)
        exchange_txn = _money(exchange_txn)
        clearing = _money(clearing)
        sebi = _money(sebi)
        stamp = _money(stamp)
        ipft = _money(ipft)
        gst = _money(gst)
        dp = _money(dp)
        other = _money(other)
        total = _money(brokerage + stt + exchange_txn + clearing + sebi + stamp + ipft + gst + dp + other)

        return ChargeBreakdown(
            broker=self.schedule.broker,
            schedule_id=self.schedule.schedule_id,
            effective_date=self.schedule.effective_date.isoformat(),
            segment=seg.value,
            side=s.value,
            qty=qty,
            price=price,
            turnover=_money(turnover),
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            clearing=clearing,
            sebi=sebi,
            stamp=stamp,
            ipft=ipft,
            gst=gst,
            dp=dp,
            other=other,
            total_fees=total,
            order_id=order_id,
            isin=instrument_id,
            trade_date=td.isoformat(),
            brokerage_order_total=_money(brokerage_order_total),
            source_urls=self.schedule.source_urls,
            notes=self.schedule.notes,
        )

    def _exchange_txn_pct(self, seg: Segment, rates: _SegmentRates) -> float:
        if self.exchange == "NSE":
            return rates.exchange_txn_pct
        if self.exchange == "BSE":
            if seg in (Segment.FUTURES, Segment.EQUITY_OPTIONS):
                raise ValueError(
                    "BSE F&O exchange transaction rates are not configured in this schedule; "
                    "refusing to apply NSE rates to BSE derivatives"
                )
            if seg in (Segment.EQUITY_DELIVERY, Segment.EQUITY_INTRADAY):
                return self.schedule.bse_equity_exchange_txn_pct
            raise ValueError(f"Unsupported BSE segment {seg}")
        raise ValueError(f"Unsupported exchange {self.exchange}")

    def estimate_economics_for_fill(
        self,
        *,
        side: Side | str,
        qty: float,
        raw_price: float,
        segment: Segment | str = Segment.EQUITY_DELIVERY,
        apply_slippage: bool = True,
        isin: str | None = None,
        symbol: str | None = None,
        trade_date: date | datetime | str | None = None,
        assume_dp_charged: bool | None = None,
    ) -> FillEconomics:
        """Non-mutating estimate for MTM / dashboard — never touches order session or DP ledger writes.

        assume_dp_charged:
          True  → do not include DP in estimate
          False → include DP in estimate
          None  → peek persisted ledger (read-only); include DP only if not yet charged
        """
        s = Side(side) if not isinstance(side, Side) else side
        seg = Segment(segment) if not isinstance(segment, Segment) else segment
        exec_px = self.slipped_price(raw_price, s) if apply_slippage else float(raw_price)
        apply_dp: bool | None
        if seg == Segment.EQUITY_DELIVERY and s == Side.SELL:
            instrument_id = canonical_instrument_id(isin, symbol)
            if not instrument_id:
                raise ValueError("MTM delivery exit estimate requires ISIN or symbol")
            if assume_dp_charged is True:
                apply_dp = False
            elif assume_dp_charged is False:
                apply_dp = True
            else:
                td = _parse_trade_date(trade_date)
                already = dp_already_charged(
                    broker=self.broker,
                    instrument_id=instrument_id,
                    trade_date=td,
                    path=self.dp_ledger_path,
                )
                apply_dp = not already
        else:
            apply_dp = False
        # Isolated session so brokerage aggregation is not shared with live paper session
        iso = self.new_session()
        # For estimate DP we pass apply_dp explicitly; charges_for_fill still persists when True.
        # Use a dry path: compute via internal helper that skips ledger write.
        return self._economics_estimate(
            side=s,
            qty=qty,
            exec_px=exec_px,
            raw_price=float(raw_price),
            segment=seg,
            isin=isin,
            symbol=symbol,
            trade_date=trade_date,
            include_dp=bool(apply_dp),
            session=iso,
        )

    def _economics_estimate(
        self,
        *,
        side: Side,
        qty: float,
        exec_px: float,
        raw_price: float,
        segment: Segment,
        isin: str | None,
        symbol: str | None,
        trade_date: date | datetime | str | None,
        include_dp: bool,
        session: ChargeSession,
    ) -> FillEconomics:
        """Fill economics without persisting DP (estimate-only)."""
        rates = _rates_for(self.schedule, segment)
        turnover = float(qty) * exec_px
        td = _parse_trade_date(trade_date)
        instrument_id = canonical_instrument_id(isin, symbol)
        brokerage = _brokerage_for_turnover(rates, turnover)
        stt = 0.0
        if side == Side.BUY and rates.stt_on_buy:
            stt = turnover * rates.stt_pct
        elif side == Side.SELL and rates.stt_on_sell:
            stt = turnover * rates.stt_pct
        exchange_txn = turnover * self._exchange_txn_pct(segment, rates)
        clearing = turnover * rates.clearing_pct
        sebi = turnover * (rates.sebi_per_crore / CRORE)
        ipft = turnover * (rates.ipft_per_crore / CRORE) if self.exchange == "NSE" else 0.0
        stamp = turnover * rates.stamp_pct_buy if side == Side.BUY else 0.0
        gst_base = brokerage + exchange_txn + clearing + sebi + ipft
        gst = gst_base * rates.gst_rate
        dp = 0.0
        if include_dp and segment == Segment.EQUITY_DELIVERY and side == Side.SELL and rates.dp_flat > 0:
            dp = rates.dp_flat
            if rates.dp_gst:
                gst += rates.dp_flat * rates.gst_rate
        brokerage = _money(brokerage)
        stt = _money(stt)
        exchange_txn = _money(exchange_txn)
        clearing = _money(clearing)
        sebi = _money(sebi)
        stamp = _money(stamp)
        ipft = _money(ipft)
        gst = _money(gst)
        dp = _money(dp)
        total = _money(brokerage + stt + exchange_txn + clearing + sebi + stamp + ipft + gst + dp)
        charges = ChargeBreakdown(
            broker=self.schedule.broker,
            schedule_id=self.schedule.schedule_id,
            effective_date=self.schedule.effective_date.isoformat(),
            segment=segment.value,
            side=side.value,
            qty=float(qty),
            price=exec_px,
            turnover=_money(turnover),
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn,
            clearing=clearing,
            sebi=sebi,
            stamp=stamp,
            ipft=ipft,
            gst=gst,
            dp=dp,
            other=0.0,
            total_fees=total,
            order_id=None,
            isin=instrument_id,
            trade_date=td.isoformat(),
            brokerage_order_total=brokerage,
            source_urls=self.schedule.source_urls,
            notes="estimate_only; DP/order state not mutated",
        )
        notional = exec_px * float(qty)
        slip_rupees = abs(exec_px - raw_price) * float(qty)
        cash_delta = -(notional + total) if side == Side.BUY else (notional - total)
        return FillEconomics(
            raw_price=raw_price,
            exec_price=exec_px,
            slippage_rupees=_money(slip_rupees),
            charges=charges,
            notional=_money(notional),
            total_fees=total,
            cash_delta=_money(cash_delta),
        )

    def economics_for_fill(
        self,
        *,
        side: Side | str,
        qty: float,
        raw_price: float,
        segment: Segment | str = Segment.EQUITY_DELIVERY,
        apply_slippage: bool = True,
        session: ChargeSession | None = None,
        order_id: str | None = None,
        isin: str | None = None,
        symbol: str | None = None,
        trade_date: date | datetime | str | None = None,
        apply_dp: bool | None = None,
        event: str | None = None,
    ) -> FillEconomics:
        s = Side(side) if not isinstance(side, Side) else side
        exec_px = self.slipped_price(raw_price, s) if apply_slippage else float(raw_price)
        charges = self.charges_for_fill(
            side=s,
            qty=qty,
            price=exec_px,
            segment=segment,
            order_id=order_id,
            isin=isin,
            symbol=symbol,
            trade_date=trade_date,
            session=session,
            apply_dp=apply_dp,
            event=event,
        )
        notional = exec_px * float(qty)
        slip_rupees = abs(exec_px - float(raw_price)) * float(qty)
        if s == Side.BUY:
            cash_delta = -(notional + charges.total_fees)
        else:
            cash_delta = notional - charges.total_fees
        return FillEconomics(
            raw_price=float(raw_price),
            exec_price=exec_px,
            slippage_rupees=_money(slip_rupees),
            charges=charges,
            notional=_money(notional),
            total_fees=charges.total_fees,
            cash_delta=_money(cash_delta),
        )

    def reconcile_round_trip(
        self,
        *,
        buy: FillEconomics,
        sells: Sequence[FillEconomics],
    ) -> RoundTripEconomics:
        if buy.charges.side != Side.BUY.value:
            raise ValueError("buy fill must be BUY")
        if not sells:
            raise ValueError("at least one sell fill required")
        buy_seg = buy.charges.segment
        for s in sells:
            if s.charges.side != Side.SELL.value:
                raise ValueError("sell fills must be SELL")
            if s.charges.segment != buy_seg:
                raise ValueError(
                    f"segment mismatch: buy={buy_seg} sell={s.charges.segment}"
                )
        qty_bought = buy.charges.qty
        qty_sold = sum(s.charges.qty for s in sells)
        if qty_sold > qty_bought + 1e-9:
            raise ValueError(f"sold qty {qty_sold} exceeds bought qty {qty_bought}")

        sell_notional = sum(s.notional for s in sells)
        buy_notional_sold = buy.exec_price * qty_sold
        gross = sell_notional - buy_notional_sold
        fees = buy.total_fees + sum(s.total_fees for s in sells)
        slip = buy.slippage_rupees + sum(s.slippage_rupees for s in sells)
        brokerage = buy.charges.brokerage + sum(s.charges.brokerage for s in sells)
        dp = buy.charges.dp + sum(s.charges.dp for s in sells)
        statutory = fees - brokerage - dp

        buy_fee_alloc = buy.total_fees * (qty_sold / qty_bought) if qty_bought else 0.0
        buy_cash_for_sold = -(buy.exec_price * qty_sold + buy_fee_alloc)
        sell_cash = sum(s.cash_delta for s in sells)
        net = buy_cash_for_sold + sell_cash
        return RoundTripEconomics(
            buy=buy,
            sells=tuple(sells),
            gross_pnl=_money(gross),
            total_fees=_money(fees),
            total_slippage=_money(slip),
            brokerage=_money(brokerage),
            statutory=_money(statutory),
            dp=_money(dp),
            net_pnl=_money(net),
            qty_bought=qty_bought,
            qty_sold=qty_sold,
        )

    def round_trip_cost_pct(
        self,
        *,
        price: float = 1000.0,
        qty: float = 100.0,
        segment: Segment | str = Segment.EQUITY_DELIVERY,
        include_slippage: bool = True,
    ) -> float:
        session = self.new_session()
        buy = self.economics_for_fill(
            side=Side.BUY,
            qty=qty,
            raw_price=price,
            segment=segment,
            apply_slippage=include_slippage,
            session=session,
            order_id="rt_buy",
            symbol="SAMPLE",
            trade_date=self.as_of,
        )
        sell = self.economics_for_fill(
            side=Side.SELL,
            qty=qty,
            raw_price=price,
            segment=segment,
            apply_slippage=include_slippage,
            session=session,
            order_id="rt_sell",
            symbol="SAMPLE",
            trade_date=self.as_of,
        )
        rt = self.reconcile_round_trip(buy=buy, sells=[sell])
        base = price * qty
        return abs(rt.net_pnl) / base if base else 0.0

    def schedule_info(self) -> dict:
        s = self.schedule
        return {
            "broker": s.broker,
            "schedule_id": s.schedule_id,
            "effective_date": s.effective_date.isoformat(),
            "end_date": s.end_date.isoformat() if s.end_date else None,
            "source_urls": list(s.source_urls),
            "notes": s.notes,
            "as_of": self.as_of.isoformat(),
            "slippage_pct": self.slippage_pct,
            "exchange": self.exchange,
            "supported_exchanges": list(s.supported_exchanges),
            "assumed_schedule": self.assumed_schedule,
            "account_opening": s.account_opening,
            "amc": s.amc,
            "gst_base_components": list(s.gst_base_components),
            "gst_excludes": ["stt", "stamp"],
            "gst_source": (
                "fyers.in/charges-list — brokerage + exchange txn + clearing + SEBI + "
                "NSE IPFT (+ DP GST). Statutory KB describes GST differently; "
                "charges-list treatment used for this schedule."
            ),
            "ipft_policy": (
                "NSE delivery/intraday/futures/options(on premium) ₹0.01/crore from "
                "fyers.in/charges-list; options aligned with NSE/FA/73061 (eff. 2026-03-01). "
                "Statutory KB still cites ₹10/crore (equity/FUT) and ₹50/crore (options) — "
                "not used. BSE IPFT=0."
            ),
            "w4_outstanding": (
                "Portfolio dual-momentum/adaptive backtests still use approximate "
                "blended RT %% (PORTFOLIO_BT_COST_MODEL); not fill-level ChargeEngine. "
                "W4 is not fully complete while those paths remain approximate."
            ),
        }


def default_charge_engine(
    broker: str | None = None,
    *,
    slippage_pct: float | None = None,
    as_of: date | datetime | str | None = None,
    exchange: str = "NSE",
    allow_assumed_schedule: bool = False,
    dp_ledger_path: Path | None = None,
) -> ChargeEngine:
    from ati.config import DEFAULT

    return ChargeEngine(
        broker or DEFAULT.charges_broker,
        as_of=as_of,
        slippage_pct=DEFAULT.risk.slippage_pct if slippage_pct is None else slippage_pct,
        exchange=exchange,
        allow_assumed_schedule=allow_assumed_schedule,
        dp_ledger_path=dp_ledger_path,
    )
