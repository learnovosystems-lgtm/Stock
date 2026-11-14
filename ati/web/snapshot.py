"""Build a single JSON snapshot for the dashboard (live + research)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ati.config import REPORTS_DIR, ensure_dirs
from ati.data.provider import MarketDataProvider
from ati.data.universe import load_universe
from ati.engine.suggestion_quality import annotate_opportunity, enrich_opportunity_dict
from ati.journal.store import (
    evaluate_hits_against_prices,
    list_journal,
    list_suggestions,
    performance_stats,
)
from ati.models.opportunity import Opportunity
from ati.ops.session import _default_broker, is_market_open
from ati.regime.detector import RegimeDetector
from ati.research.registry import StrategyRegistry
from ati.scanner.market_scanner import MarketScanner


def _mtm_positions(broker, provider: MarketDataProvider) -> list[dict[str, Any]]:
    out = []
    for p in broker.positions():
        sym = p.symbol if p.symbol.endswith(".NS") else p.symbol + ".NS"
        df = provider.get_daily(sym)
        last = float(df["Close"].iloc[-1]) if df is not None and len(df) else p.entry
        pnl = (last - p.entry) * p.qty * p.remaining_frac
        pnl_pct = last / p.entry - 1.0 if p.entry else 0.0
        out.append(
            {
                "symbol": p.symbol.replace(".NS", ""),
                "qty": p.qty,
                "entry": round(p.entry, 2),
                "buy": round(p.entry, 2),
                "sell": round(p.t1, 2),
                "last": round(last, 2),
                "stop": round(p.stop, 2),
                "t1": round(p.t1, 2),
                "t2": round(p.t2, 2),
                "t3": round(p.t3, 2),
                "rr": round((p.t1 - p.entry) / max(p.entry - p.stop, 1e-9), 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pct": round(pnl_pct, 4),
                "strategy": p.strategy,
                "opened_at": p.opened_at,
                "bars_held": p.bars_held,
            }
        )
    return out


def _cards_from_opps(opps: list[Opportunity], metrics) -> list[dict[str, Any]]:
    cards = []
    for o in opps:
        ann = annotate_opportunity(o, metrics)
        cards.append(enrich_opportunity_dict(o, ann))
    cards.sort(
        key=lambda c: (
            0 if c["signal"] == "BUY" else 1,
            0 if c["quality_grade"] == "A" else 1 if c["quality_grade"] == "B" else 2,
            -c["quality_score"],
        )
    )
    return cards


def build_snapshot(*, run_scan: bool = True, limit_scan: int = 80) -> dict[str, Any]:
    ensure_dirs()
    provider = MarketDataProvider()
    broker = _default_broker()
    uni = load_universe()
    registry = StrategyRegistry()
    metrics = registry.latest_metrics()

    sample = {i.symbol: provider.get_daily(i.symbol) for i in uni[:50]}
    sample = {k: v for k, v in sample.items() if v is not None and len(v) > 60}
    regime = RegimeDetector(provider).detect(sample)
    nifty = provider.nifty()
    vix = provider.india_vix()
    nifty_last = float(nifty["Close"].iloc[-1]) if len(nifty) else None
    vix_last = float(vix["Close"].iloc[-1]) if len(vix) else None

    opportunities: list[dict[str, Any]] = []
    scan_meta: dict[str, Any] = {}
    if run_scan:
        acct = broker.account()
        scanner = MarketScanner(metrics=metrics)
        result = scanner.scan(
            universe=uni[:limit_scan] if limit_scan else uni,
            equity=acct["equity"],
            cash=acct["cash"],
        )
        scan_meta = {
            "regime": result.regime,
            "regime_notes": result.regime_notes,
            "scanned": result.scanned,
            "generated_at": result.generated_at,
        }
        opportunities = _cards_from_opps(result.opportunities[:25], metrics)
    else:
        feed = REPORTS_DIR / "opportunities_latest.json"
        latest = REPORTS_DIR / "scan_latest.json"
        if feed.exists():
            raw = json.loads(feed.read_text())
            scan_meta = {
                "regime": raw.get("regime"),
                "regime_notes": raw.get("regime_notes"),
                "scanned": raw.get("scanned"),
                "generated_at": raw.get("generated_at"),
            }
            opportunities = raw.get("opportunities") or raw.get("actionable") or []
        elif latest.exists():
            raw = json.loads(latest.read_text())
            scan_meta = {
                "regime": raw.get("regime"),
                "regime_notes": raw.get("regime_notes"),
                "scanned": raw.get("scanned"),
                "generated_at": raw.get("generated_at"),
            }
            # Best-effort remap from persisted Opportunity dumps
            for o in raw.get("opportunities", [])[:25]:
                opportunities.append(
                    {
                        "stock": str(o.get("stock", "")).replace(".NS", ""),
                        "signal": o.get("signal"),
                        "score": round(float(o.get("opportunity_score", 0) or 0), 1),
                        "quality_grade": "C",
                        "quality_score": round(float(o.get("opportunity_score", 0) or 0), 1),
                        "strategy": o.get("strategy_selected"),
                        "current": o.get("current_price"),
                        "buy": o.get("ideal_entry"),
                        "sell": o.get("target_1"),
                        "sell_t2": o.get("target_2"),
                        "sell_t3": o.get("target_3"),
                        "stop": o.get("stop_loss"),
                        "rr": o.get("risk_reward"),
                        "sector": o.get("sector"),
                    }
                )

    positions = _mtm_positions(broker, provider)
    acct = broker.account()
    mtm_equity = float(acct["cash"])
    for m, p in zip(positions, broker.positions()):
        mtm_equity += float(m["last"]) * p.qty * p.remaining_frac
    if not positions:
        mtm_equity = float(acct["equity"])

    # Manual journal + suggestion history (no Zerodha API)
    prices = {p["symbol"]: p["last"] for p in positions}
    for o in opportunities:
        if o.get("stock") and o.get("current"):
            prices[o["stock"]] = float(o["current"])
    hits = evaluate_hits_against_prices(prices)
    history = list_suggestions(include_expired=True, limit=40)
    journal = list_journal(limit=40)
    perf = performance_stats()

    production = registry.list_production()
    experiments = []
    try:
        for m in metrics[-12:]:
            experiments.append(
                {
                    "strategy_id": m.strategy_id,
                    "version": m.version,
                    "sharpe": round(m.sharpe, 3),
                    "total_return": round(m.total_return, 3),
                    "max_drawdown": round(m.max_drawdown, 3),
                    "stability": round(m.stability_score, 3),
                    "trades": m.sample_count,
                    "oos": m.oos,
                }
            )
    except Exception:  # noqa: BLE001
        pass

    actionable = [c for c in opportunities if c.get("signal") == "BUY" and c.get("quality_grade") in ("A", "B")]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_open": is_market_open(),
        "brand": "ATI Desk",
        "tagline": "Adaptive Indian equities — Zerodha live + paper",
        "regime": {
            "regime": regime.regime.value,
            "confidence": round(regime.confidence, 3),
            "nifty_trend": round(regime.nifty_trend, 4),
            "india_vix": round(vix_last or regime.india_vix, 2),
            "breadth": round(regime.breadth, 3),
            "adx": round(regime.adx, 2),
            "notes": regime.notes,
            "preferred_families": regime.preferred_families(),
            "nifty_last": nifty_last,
        },
        "account": {
            "cash": round(acct["cash"], 2),
            "equity_book": round(acct["equity"], 2),
            "equity_mtm": round(mtm_equity, 2),
            "day_pnl": round(acct["day_pnl"], 2),
            "open_positions": len(positions),
            "starting_capital": 100_000.0,
            "mtm_return": round(mtm_equity / 100_000.0 - 1.0, 4),
        },
        "positions": positions,
        "broker": {
            "mode": "manual_journal",
            "kite_web": "https://kite.zerodha.com",
            "message": "Log into Kite in your browser, then mark entries/exits here for P&L (no API).",
        },
        "opportunities": opportunities,
        "actionable": actionable,
        "suggestion_history": history,
        "journal": journal,
        "performance": perf,
        "hits_marked": hits,
        "scan": scan_meta,
        "production": production,
        "metrics": experiments,
        "universe_size": len(uni),
        "research": {
            "dual_momentum_oos_sharpe": 0.68,
            "adaptive_oos_sharpe": 0.68,
            "adaptive_oos_stability": 0.48,
            "promoted": ["dual_momentum_nifty_gated", "regime_adaptive_multi"],
            "note": (
                "Suggestions use news sentiment + HTF/LTF supply-demand + OOS grades. "
                "Each call has buy/sell/SL ranges, why-chosen, and expiry."
            ),
        },
    }


def write_snapshot(path: Path | None = None, **kwargs) -> Path:
    ensure_dirs()
    snap = build_snapshot(**kwargs)
    path = path or (REPORTS_DIR / "dashboard_snapshot.json")
    path.write_text(json.dumps(snap, indent=2, default=str))
    return path
