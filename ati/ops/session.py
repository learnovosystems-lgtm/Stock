"""Continuous operation helpers — market-hours monitor and EOD reconciliation."""

from __future__ import annotations

import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

from ati.config import DEFAULT, REPORTS_DIR, ensure_dirs
from ati.data.provider import MarketDataProvider
from ati.data.universe import load_universe
from ati.paper.portfolio import PaperBroker
from ati.research.registry import StrategyRegistry
from ati.scanner.market_scanner import MarketScanner


IST = ZoneInfo("Asia/Kolkata")


def is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return time(9, 15) <= t <= time(15, 30)


def _default_broker() -> PaperBroker:
    uni = load_universe()
    sectors = {i.symbol: i.sector for i in uni}
    sectors.update({i.symbol.replace(".NS", ""): i.sector for i in uni})
    return PaperBroker(
        allowed_strategy_substrings=DEFAULT.paper_allowed_strategy_substrings,
        symbol_sectors=sectors,
    )


def _default_scanner() -> MarketScanner:
    return MarketScanner(metrics=StrategyRegistry().latest_metrics())


def _is_gated_candidate(strategy_selected: str) -> bool:
    s = strategy_selected.lower()
    return "dual_momentum" in s or "relative_strength" in s


def run_intrasession_once(scanner: MarketScanner | None = None, broker: PaperBroker | None = None) -> dict:
    scanner = scanner or _default_scanner()
    broker = broker or _default_broker()
    acct = broker.account()
    result = scanner.scan(equity=acct["equity"], cash=acct["cash"])
    entered = []
    for opp in result.opportunities:
        if opp.signal.value != "BUY":
            continue
        if not _is_gated_candidate(opp.strategy_selected):
            entered.append({"stock": opp.stock, "ok": False, "msg": "Skipped non-gated strategy"})
            continue
        ok, msg = broker.try_enter(opp)
        entered.append({"stock": opp.stock, "ok": ok, "msg": msg, "strategy": opp.strategy_selected})
        if len([e for e in entered if e["ok"]]) >= 1:
            break

    provider = scanner.provider
    bars = {}
    for p in broker.positions():
        sym = p.symbol if p.symbol.endswith(".NS") else p.symbol + ".NS"
        bars[sym] = provider.get_daily(sym)
    exit_msgs = broker.manage_exits(bars)
    return {
        "phase": "intrasession",
        "market_open": is_market_open(),
        "regime": result.regime,
        "regime_notes": result.regime_notes,
        "opportunities": len(result.opportunities),
        "buy_signals": sum(1 for o in result.opportunities if o.signal.value == "BUY"),
        "entries": entered,
        "exits": exit_msgs,
        "account": broker.account(),
    }


def run_eod(scanner: MarketScanner | None = None, broker: PaperBroker | None = None) -> dict:
    ensure_dirs()
    scanner = scanner or _default_scanner()
    broker = broker or _default_broker()
    provider = MarketDataProvider()
    bars = {}
    for p in broker.positions():
        sym = p.symbol if p.symbol.endswith(".NS") else p.symbol + ".NS"
        bars[sym] = provider.get_daily(sym, force=True)
    exit_msgs = broker.manage_exits(bars)
    broker.roll_day()
    acct = broker.account()
    scan = scanner.scan(equity=acct["equity"], cash=acct["cash"], force_refresh=True)

    # Paper entries from EOD watchlist (gated only)
    entered = []
    for opp in scan.opportunities:
        if opp.signal.value != "BUY" or not _is_gated_candidate(opp.strategy_selected):
            continue
        ok, msg = broker.try_enter(opp)
        entered.append({"stock": opp.stock, "ok": ok, "msg": msg, "strategy": opp.strategy_selected})
        if ok:
            break

    watchlist = [
        {
            "stock": o.stock,
            "signal": o.signal.value,
            "score": o.opportunity_score,
            "strategy": o.strategy_selected,
        }
        for o in scan.opportunities[:20]
    ]
    report = {
        "phase": "eod",
        "regime": scan.regime,
        "regime_notes": scan.regime_notes,
        "exits": exit_msgs,
        "entries": entered,
        "watchlist": watchlist,
        "account": broker.snapshot(),
        "generated_at": datetime.now(IST).isoformat(),
    }
    path = REPORTS_DIR / f"eod_{datetime.now(IST).strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return report
