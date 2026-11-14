"""Continuous opportunity analysis loop."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ati.config import REPORTS_DIR, ensure_dirs
from ati.data.universe import load_universe
from ati.engine.suggestion_quality import annotate_opportunity, enrich_opportunity_dict
from ati.ops.session import _default_broker, _default_scanner, is_market_open
from ati.research.registry import StrategyRegistry
from ati.web.snapshot import write_snapshot


def run_analysis_pass(
    *,
    limit: int = 80,
    min_rr: float = 2.0,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """One continuous-analysis cycle: scan → quality filter → persist feed."""
    ensure_dirs()
    registry = StrategyRegistry()
    metrics = registry.latest_metrics()
    broker = _default_broker()
    scanner = _default_scanner()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    acct = broker.account()
    if progress:
        progress(f"Scanning {len(uni)} names…")
    result = scanner.scan(universe=uni, equity=acct["equity"], cash=acct["cash"])

    cards = []
    for opp in result.opportunities:
        ann = annotate_opportunity(opp, metrics, min_rr=min_rr)
        card = enrich_opportunity_dict(opp, ann)
        cards.append(card)

    # Prefer A/B BUY first for actionable feed
    cards.sort(
        key=lambda c: (
            0 if c["signal"] == "BUY" else 1 if c["signal"] == "WAIT" else 2,
            0 if c["quality_grade"] == "A" else 1 if c["quality_grade"] == "B" else 2,
            -c["quality_score"],
        )
    )

    feed = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_open": is_market_open(),
        "regime": result.regime,
        "regime_notes": result.regime_notes,
        "scanned": result.scanned,
        "opportunities": cards,
        "actionable": [c for c in cards if c["signal"] == "BUY" and c["quality_grade"] in ("A", "B")],
        "counts": {
            "total": len(cards),
            "buy": sum(1 for c in cards if c["signal"] == "BUY"),
            "grade_a": sum(1 for c in cards if c["quality_grade"] == "A"),
            "grade_b": sum(1 for c in cards if c["quality_grade"] == "B"),
        },
    }
    path = REPORTS_DIR / "opportunities_live.json"
    path.write_text(json.dumps(feed, indent=2, default=str))
    latest = REPORTS_DIR / "opportunities_latest.json"
    latest.write_text(path.read_text())
    if progress:
        progress(
            f"Feed written: {feed['counts']['buy']} BUY · "
            f"A={feed['counts']['grade_a']} B={feed['counts']['grade_b']}"
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
    """Loop analysis forever (cycles=0) or for N cycles."""
    n = 0
    while True:
        n += 1
        if progress:
            progress(f"—— cycle {n} ——")
        try:
            run_analysis_pass(limit=limit, progress=progress)
            if refresh_desk:
                write_snapshot(run_scan=False)
                # also sync dashboard static copy
                dash = Path("dashboard/data/snapshot.json")
                src = REPORTS_DIR / "dashboard_snapshot.json"
                if src.exists():
                    dash.parent.mkdir(parents=True, exist_ok=True)
                    # rebuild desk with scan=false still pulls latest scan file
                    from ati.web.snapshot import build_snapshot

                    snap = build_snapshot(run_scan=False)
                    dash.write_text(json.dumps(snap, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(f"cycle error: {exc}")
        if cycles and n >= cycles:
            break
        time.sleep(max(30, interval_sec))
