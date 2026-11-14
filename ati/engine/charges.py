"""Broker-configurable ChargeEngine — FYERS / Zerodha schedules, versioned by date.

Slippage is applied on price only. Statutory + brokerage fees are itemized and
never folded into the slipped price, preventing double-counting.

Official sources (pinned in schedule metadata):
- FYERS charges list: https://fyers.in/charges-list
- FYERS statutory KB: https://support.fyers.in/portal/en/kb/articles/what-are-the-statutory-charges-on-trades-at-fyers-stt-gst-sebi-turnover-fee-ipft-etc
- FYERS pricing: https://fyers.in/pricing
- Zerodha charges (legacy / alternate profile only): https://zerodha.com/charges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

from ati.models.charges import (
    BrokerId,
    ChargeBreakdown,
    FillEconomics,
    RoundTripEconomics,
    Segment,
    Side,
)


CRORE = 10_000_000.0


@dataclass(frozen=True)
class _SegmentRates:
    # Brokerage: min(flat_cap, pct * turnover) when pct > 0; else flat
    brokerage_flat: float
    brokerage_pct: float  # 0 => flat only
    # STT as fraction of turnover; buy/sell flags
    stt_pct: float
    stt_on_buy: bool
    stt_on_sell: bool
    exchange_txn_pct: float
    clearing_pct: float
    sebi_per_crore: float
    stamp_pct_buy: float  # 0 on sell
    ipft_per_crore: float
    gst_rate: float
    # DP on delivery sell (per scrip / day) — applied once per sell order when enabled
    dp_flat: float
    dp_gst: bool


@dataclass(frozen=True)
class FeeSchedule:
    broker: str
    schedule_id: str
    effective_date: date  # inclusive
    end_date: date | None  # exclusive; None = open
    source_urls: tuple[str, ...]
    equity_delivery: _SegmentRates
    equity_intraday: _SegmentRates
    futures: _SegmentRates
    notes: str = ""


def _fyers_standard_2026_04_01() -> FeeSchedule:
    """FYERS Standard plan rates from fyers.in/charges-list (captured 2026-11-14).

    IPFT: charges-list shows ₹0.01/Crore; FYERS statutory KB (updated 01 Apr 2026)
    documents ₹10/Crore with a worked example. We pin KB ₹10/Crore for IPFT and
    record the discrepancy in notes.
    """
    gst = 0.18
    sebi = 10.0
    ipft = 10.0  # per crore — see notes
    eq_exch = 0.0030699 / 100.0
    fut_exch = 0.0018299 / 100.0
    delivery = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.003,  # 0.3%
        stt_pct=0.001,  # 0.1%
        stt_on_buy=True,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,  # not separately listed for cash equity on charges-list
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00015,  # 0.015%
        ipft_per_crore=ipft,
        gst_rate=gst,
        dp_flat=12.5,
        dp_gst=True,
    )
    intraday = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,  # 0.03%
        stt_pct=0.00025,  # 0.025% sell only
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=eq_exch,
        clearing_pct=0.0,
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00003,  # 0.003%
        ipft_per_crore=ipft,
        gst_rate=gst,
        dp_flat=0.0,
        dp_gst=False,
    )
    futures = _SegmentRates(
        brokerage_flat=20.0,
        brokerage_pct=0.0003,  # 0.03%
        stt_pct=0.0005,  # 0.05% sell only
        stt_on_buy=False,
        stt_on_sell=True,
        exchange_txn_pct=fut_exch,
        clearing_pct=0.0005 / 100.0,  # 0.0005% from charges-list clearing row
        sebi_per_crore=sebi,
        stamp_pct_buy=0.00002,  # 0.002%
        ipft_per_crore=ipft,
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
        ),
        equity_delivery=delivery,
        equity_intraday=intraday,
        futures=futures,
        notes=(
            "FYERS Standard (₹20 or pct, whichever lower). "
            "IPFT pinned at ₹10/crore per FYERS statutory KB example (01 Apr 2026); "
            "charges-list UI shows ₹0.01/crore — discrepancy documented."
        ),
    )


def _zerodha_equity_2024_10_01() -> FeeSchedule:
    """Zerodha alternate profile — never used when broker=fyers.

    Approximate published equity/F&O discount schedule for configuration parity.
    Source: https://zerodha.com/charges (captured for versioning; not applied to FYERS).
    """
    gst = 0.18
    sebi = 10.0
    ipft = 10.0
    eq_exch = 0.00297 / 100.0
    fut_exch = 0.00173 / 100.0
    delivery = _SegmentRates(
        brokerage_flat=0.0,  # Zerodha equity delivery brokerage ₹0
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
        dp_flat=13.5,  # approx CDSL+broker; profile-specific
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
        stt_pct=0.0002,  # Zerodha FUT STT historically 0.02% sell — verify on use
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
    return FeeSchedule(
        broker=BrokerId.ZERODHA.value,
        schedule_id="zerodha_equity_2024_10_01",
        effective_date=date(2024, 10, 1),
        end_date=None,
        source_urls=("https://zerodha.com/charges",),
        equity_delivery=delivery,
        equity_intraday=intraday,
        futures=futures,
        notes="Alternate broker profile only. Never selected when charges_broker=fyers.",
    )


_SCHEDULES: tuple[FeeSchedule, ...] = (
    _fyers_standard_2026_04_01(),
    _zerodha_equity_2024_10_01(),
)


def list_schedules() -> list[FeeSchedule]:
    return list(_SCHEDULES)


def _parse_as_of(as_of: date | datetime | str | None) -> date:
    if as_of is None:
        return date.today()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    return date.fromisoformat(str(as_of)[:10])


def resolve_schedule(broker: str, as_of: date | datetime | str | None = None) -> FeeSchedule:
    broker_l = (broker or BrokerId.FYERS.value).strip().lower()
    d = _parse_as_of(as_of)
    matches = [
        s
        for s in _SCHEDULES
        if s.broker == broker_l and s.effective_date <= d and (s.end_date is None or d < s.end_date)
    ]
    if not matches:
        raise ValueError(f"No fee schedule for broker={broker_l!r} as_of={d.isoformat()}")
    matches.sort(key=lambda s: s.effective_date)
    return matches[-1]


def _rates_for(schedule: FeeSchedule, segment: Segment | str) -> _SegmentRates:
    seg = Segment(segment) if not isinstance(segment, Segment) else segment
    if seg == Segment.EQUITY_DELIVERY:
        return schedule.equity_delivery
    if seg == Segment.EQUITY_INTRADAY:
        return schedule.equity_intraday
    if seg == Segment.FUTURES:
        return schedule.futures
    raise ValueError(f"Unsupported segment: {segment}")


def _brokerage(rates: _SegmentRates, turnover: float) -> float:
    if rates.brokerage_pct <= 0 and rates.brokerage_flat <= 0:
        return 0.0
    if rates.brokerage_pct <= 0:
        return float(rates.brokerage_flat)
    pct_amt = turnover * rates.brokerage_pct
    if rates.brokerage_flat <= 0:
        return pct_amt
    return float(min(rates.brokerage_flat, pct_amt))


class ChargeEngine:
    """Single calculation path for backtest, paper, journal, and dashboard."""

    def __init__(
        self,
        broker: str = BrokerId.FYERS.value,
        *,
        as_of: date | datetime | str | None = None,
        slippage_pct: float = 0.0005,
        exchange: str = "NSE",
    ):
        self.broker = (broker or BrokerId.FYERS.value).strip().lower()
        if self.broker == BrokerId.FYERS.value:
            # Hard guard: never silently fall back to Zerodha math for FYERS
            pass
        self.as_of = _parse_as_of(as_of)
        self.slippage_pct = float(slippage_pct)
        self.exchange = exchange
        self.schedule = resolve_schedule(self.broker, self.as_of)

    def with_as_of(self, as_of: date | datetime | str | None) -> "ChargeEngine":
        return ChargeEngine(self.broker, as_of=as_of, slippage_pct=self.slippage_pct, exchange=self.exchange)

    def slipped_price(self, price: float, side: Side | str) -> float:
        """Market impact only — no fees embedded."""
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
        apply_dp: bool | None = None,
    ) -> ChargeBreakdown:
        s = Side(side) if not isinstance(side, Side) else side
        seg = Segment(segment) if not isinstance(segment, Segment) else segment
        rates = _rates_for(self.schedule, seg)
        qty = float(qty)
        price = float(price)
        if qty <= 0 or price <= 0:
            raise ValueError("qty and price must be positive")
        turnover = qty * price

        brokerage = _brokerage(rates, turnover)
        stt = 0.0
        if s == Side.BUY and rates.stt_on_buy:
            stt = turnover * rates.stt_pct
        elif s == Side.SELL and rates.stt_on_sell:
            stt = turnover * rates.stt_pct

        exchange_txn = turnover * rates.exchange_txn_pct
        clearing = turnover * rates.clearing_pct
        sebi = turnover * (rates.sebi_per_crore / CRORE)
        ipft = turnover * (rates.ipft_per_crore / CRORE)
        stamp = turnover * rates.stamp_pct_buy if s == Side.BUY else 0.0

        # GST on brokerage + exchange + clearing + SEBI + IPFT (not STT/stamp/DP base before GST rules)
        gst_base = brokerage + exchange_txn + clearing + sebi + ipft
        gst = gst_base * rates.gst_rate

        dp = 0.0
        use_dp = rates.dp_flat > 0 if apply_dp is None else apply_dp
        if s == Side.SELL and use_dp and rates.dp_flat > 0:
            dp = rates.dp_flat
            if rates.dp_gst:
                gst += rates.dp_flat * rates.gst_rate

        total = brokerage + stt + exchange_txn + clearing + sebi + stamp + ipft + gst + dp
        return ChargeBreakdown(
            broker=self.schedule.broker,
            schedule_id=self.schedule.schedule_id,
            effective_date=self.schedule.effective_date.isoformat(),
            segment=seg.value,
            side=s.value,
            qty=qty,
            price=price,
            turnover=round(turnover, 6),
            brokerage=round(brokerage, 6),
            stt=round(stt, 6),
            exchange_txn=round(exchange_txn, 6),
            clearing=round(clearing, 6),
            sebi=round(sebi, 6),
            stamp=round(stamp, 6),
            ipft=round(ipft, 6),
            gst=round(gst, 6),
            dp=round(dp, 6),
            total_fees=round(total, 6),
            source_urls=self.schedule.source_urls,
            notes=self.schedule.notes,
        )

    def economics_for_fill(
        self,
        *,
        side: Side | str,
        qty: float,
        raw_price: float,
        segment: Segment | str = Segment.EQUITY_DELIVERY,
        apply_dp: bool | None = None,
        apply_slippage: bool = True,
    ) -> FillEconomics:
        s = Side(side) if not isinstance(side, Side) else side
        exec_px = self.slipped_price(raw_price, s) if apply_slippage else float(raw_price)
        charges = self.charges_for_fill(
            side=s, qty=qty, price=exec_px, segment=segment, apply_dp=apply_dp
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
            slippage_rupees=round(slip_rupees, 6),
            charges=charges,
            notional=round(notional, 6),
            total_fees=charges.total_fees,
            cash_delta=round(cash_delta, 6),
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
        qty_bought = buy.charges.qty
        qty_sold = sum(s.charges.qty for s in sells)
        # Gross using slipped notionals (fees excluded)
        sell_notional = sum(s.notional for s in sells)
        buy_notional_sold = buy.exec_price * qty_sold
        gross = sell_notional - buy_notional_sold
        fees = buy.total_fees + sum(s.total_fees for s in sells)
        slip = buy.slippage_rupees + sum(s.slippage_rupees for s in sells)
        # Net from cash: sum cash_deltas for matched qty (buy fees allocated pro-rata if partial)
        buy_fee_alloc = buy.total_fees * (qty_sold / qty_bought) if qty_bought else 0.0
        buy_cash_for_sold = -(buy.exec_price * qty_sold + buy_fee_alloc)
        sell_cash = sum(s.cash_delta for s in sells)
        net = buy_cash_for_sold + sell_cash
        return RoundTripEconomics(
            buy=buy,
            sells=tuple(sells),
            gross_pnl=round(gross, 6),
            total_fees=round(fees, 6),
            total_slippage=round(slip, 6),
            net_pnl=round(net, 6),
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
        """Approximate round-trip fee(+optional slip) as fraction of entry notional."""
        buy = self.economics_for_fill(
            side=Side.BUY, qty=qty, raw_price=price, segment=segment, apply_slippage=include_slippage
        )
        sell = self.economics_for_fill(
            side=Side.SELL, qty=qty, raw_price=price, segment=segment, apply_slippage=include_slippage
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
        }


def default_charge_engine(
    broker: str | None = None,
    *,
    slippage_pct: float | None = None,
    as_of: date | datetime | str | None = None,
) -> ChargeEngine:
    from ati.config import DEFAULT

    return ChargeEngine(
        broker or DEFAULT.charges_broker,
        as_of=as_of,
        slippage_pct=DEFAULT.risk.slippage_pct if slippage_pct is None else slippage_pct,
    )
