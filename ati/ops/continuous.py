"""Continuous opportunity analysis loop."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ati.config import REPORTS_DIR, ensure_dirs
from ati.data.provider import MarketDataProvider
from ati.data.universe import load_universe
from ati.engine.suggestion_quality import annotate_opportunity
from ati.journal.store import evaluate_hits_against_prices, expire_stale, performance_stats
from ati.news.sentiment import analyze_market_sentiment
from ati.ops.enrich_cards import enrich_live_card, persist_cards
from ati.ops.session import _default_broker, _default_scanner, is_market_open
from ati.research.registry import StrategyRegistry
from ati.web.snapshot import build_snapshot, write_snapshot


def run_analysis_pass(
    *,
    limit: int = 80,
    min_rr: float = 2.0,
    enrich_top: int = 12,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Scan → quality → news/zones enrich → persist history with expiry."""
    ensure_dirs()
    expire_stale()
    registry = StrategyRegistry()
    metrics = registry.latest_metrics()
    broker = _default_broker()
    scanner = _default_scanner()
    provider = MarketDataProvider()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    acct = broker.account()
    if progress:
        progress(f"Scanning {len(uni)} names…")
    result = scanner.scan(universe=uni, equity=acct["equity"], cash=acct["cash"])

    if progress:
        progress("Market news sentiment…")
    market_sentiment = analyze_market_sentiment()

    # Pre-sort by raw score then enrich top N (news/zones are slower)
    ranked = sorted(result.opportunities, key=lambda o: o.opportunity_score, reverse=True)
    cards = []
    for i, opp in enumerate(ranked[: max(enrich_top, 8)]):
        ann = annotate_opportunity(opp, metrics, min_rr=min_rr)
        if progress:
            progress(f"Enrich {opp.stock} ({i+1}/{min(len(ranked), enrich_top)})…")
        card = enrich_live_card(opp, ann, provider=provider, market_sentiment=market_sentiment)
        cards.append(card)

    cards.sort(
        key=lambda c: (
            0 if c["signal"] == "BUY" else 1 if c["signal"] == "WAIT" else 2,
            0 if c["quality_grade"] == "A" else 1 if c["quality_grade"] == "B" else 2,
            -c["quality_score"],
        )
    )
    persist_cards(cards)

    # Mark target/SL hits vs last closes for open suggestions
    prices = {}
    for c in cards:
        prices[c["stock"]] = float(c.get("current") or 0)
    # also refresh prices for historical live suggestions
    hits = evaluate_hits_against_prices(prices)

    feed = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_open": is_market_open(),
        "regime": result.regime,
        "regime_notes": result.regime_notes,
        "market_sentiment": market_sentiment.to_dict(),
        "scanned": result.scanned,
        "opportunities": cards,
        "actionable": [c for c in cards if c["signal"] == "BUY" and c["quality_grade"] in ("A", "B")],
        "counts": {
            "total": len(cards),
            "buy": sum(1 for c in cards if c["signal"] == "BUY"),
            "grade_a": sum(1 for c in cards if c["quality_grade"] == "A"),
            "grade_b": sum(1 for c in cards if c["quality_grade"] == "B"),
        },
        "hits_marked": hits,
        "performance": performance_stats(),
    }
    path = REPORTS_DIR / "opportunities_live.json"
    path.write_text(json.dumps(feed, indent=2, default=str))
    (REPORTS_DIR / "opportunities_latest.json").write_text(path.read_text())
    if progress:
        progress(
            f"Feed: {feed['counts']['buy']} BUY · A={feed['counts']['grade_a']} "
            f"B={feed['counts']['grade_b']} · hits={len(hits)}"
        )
    return feed


def run_continuous(
    *,
    interval_sec: int = 300,
    limit: int = 80,
    cycles: int = 0,
    refresh_desk: bool = True,
    progress: Callable[[str], None] | None = None,
) -> None:
    n = 0
    while True:
        n += 1
        if progress:
            progress(f"—— cycle {n} ——")
        try:
            run_analysis_pass(limit=limit, progress=progress)
            if refresh_desk:
                write_snapshot(run_scan=False)
                dash = Path("dashboard/data/snapshot.json")
                dash.parent.mkdir(parents=True, exist_ok=True)
                snap = build_snapshot(run_scan=False)
                dash.write_text(json.dumps(snap, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(f"cycle error: {exc}")
        if cycles and n >= cycles:
            break
        time.sleep(max(30, interval_sec))
