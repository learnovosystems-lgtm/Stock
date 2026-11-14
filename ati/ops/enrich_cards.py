"""Enrich opportunity cards with news, zones, why, ranges, expiry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ati.data.provider import MarketDataProvider
from ati.engine.suggestion_quality import QualityAnnotation, enrich_opportunity_dict
from ati.journal.store import save_suggestions
from ati.models.opportunity import Opportunity
from ati.news.sentiment import SentimentReport, analyze_market_sentiment, analyze_symbol_sentiment
from ati.structure.supply_demand import build_zone_context, refine_entry_ranges


def _horizon_hours(opp: Opportunity) -> float:
    # Swing ideas last longer; mean-rev shorter
    fam = str(opp.extras.get("selection_family") or "")
    if "mean" in fam:
        return 18.0
    if "breakout" in fam:
        return 24.0
    return 36.0


def build_why(
    opp: Opportunity,
    ann: QualityAnnotation,
    sentiment: SentimentReport,
    zones_why: str,
) -> str:
    bits = [
        f"Strategy {opp.strategy_selected} score {opp.opportunity_score:.0f}",
        f"grade {ann.grade} (OOS evid sharpe={ann.evidence_sharpe})",
        f"RR {opp.risk_reward:.2f}",
        f"regime {opp.market_regime.value}",
        f"news {sentiment.label}/{sentiment.gate}",
        zones_why,
    ]
    return " · ".join(bits)


def enrich_live_card(
    opp: Opportunity,
    ann: QualityAnnotation,
    *,
    provider: MarketDataProvider,
    market_sentiment: SentimentReport,
) -> dict[str, Any]:
    card = enrich_opportunity_dict(opp, ann)
    sym = opp.stock
    sentiment = analyze_symbol_sentiment(sym)
    # Market caution softens stock gate
    if market_sentiment.gate == "caution" and sentiment.gate == "allow":
        sentiment.gate = "caution"
        sentiment.reason = f"{sentiment.reason}; market tape cautious"

    daily = provider.get_daily(sym if sym.endswith(".NS") else sym + ".NS")
    weekly = provider.get_weekly(sym if sym.endswith(".NS") else sym + ".NS") if hasattr(provider, "get_weekly") else None
    if weekly is None:
        try:
            weekly = provider.download(sym if sym.endswith(".NS") else sym + ".NS", "1wk", period="5y")
        except Exception:  # noqa: BLE001
            weekly = None

    price = float(opp.current_price)
    zones = build_zone_context(daily, weekly, price)
    ranges = refine_entry_ranges(buy=float(opp.ideal_entry), stop=float(opp.stop_loss), sell=float(opp.target_1), zones=zones)

    # Gate: news block or zone avoid → demote signal
    status = "live"
    signal = card["signal"]
    if signal == "BUY" and sentiment.gate == "block":
        signal = "WAIT"
        card["quality_grade"] = "WATCH"
        card["quality_score"] = min(card["quality_score"], 40)
    if signal == "BUY" and zones.entry_bias == "avoid":
        signal = "WAIT"
        card["quality_grade"] = "WATCH" if card["quality_grade"] == "A" else card["quality_grade"]
    if signal == "BUY" and zones.entry_bias == "wait_pullback" and card["quality_grade"] == "A":
        card["quality_grade"] = "B"

    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=_horizon_hours(opp))
    why = build_why(opp, ann, sentiment, zones.why)

    card.update(
        {
            "signal": signal,
            "status": status,
            "why_chosen": why,
            "news": sentiment.to_dict(),
            "zones": zones.to_dict(),
            "entry_range": ranges["entry_range"],
            "exit_range": ranges["exit_range"],
            "sl_range": ranges["sl_range"],
            "buy": ranges["ideal_buy"],
            "sell": ranges["ideal_sell"],
            "stop": ranges["ideal_stop"],
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "expires_in_hours": round((expires - now).total_seconds() / 3600, 1),
            "market_news": market_sentiment.label,
        }
    )
    return card


def persist_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    save_suggestions(cards)
    return cards
