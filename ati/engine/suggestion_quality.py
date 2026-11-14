"""Suggestion quality overlays — blend live score with OOS evidence."""

from __future__ import annotations

from dataclasses import dataclass

from ati.engine.selection import StrategySelector, _family_of_metric
from ati.models.opportunity import Opportunity, StrategyMetrics


@dataclass
class QualityAnnotation:
    grade: str  # A / B / C / WATCH
    quality_score: float  # 0-100
    evidence_sharpe: float | None
    evidence_stability: float | None
    reasons: list[str]


def annotate_opportunity(
    opp: Opportunity,
    metrics: list[StrategyMetrics],
    *,
    min_rr: float = 2.0,
    min_score: float = 55.0,
) -> QualityAnnotation:
    """Attach a conservative quality grade so weak suggestions are demoted."""
    reasons: list[str] = []
    family = str(opp.extras.get("selection_family") or _family_of_metric(opp.strategy_selected))
    sel = StrategySelector(metrics)
    fam_w = sel.family_weight(family, opp.market_regime)

    evid = [m for m in metrics if _family_of_metric(m.strategy_id) == family]
    evid_oos = [m for m in evid if m.oos] or evid
    latest = evid_oos[-1] if evid_oos else None

    base = float(opp.opportunity_score)
    q = base

    if opp.risk_reward < min_rr:
        q *= 0.55
        reasons.append(f"RR {opp.risk_reward:.2f} < {min_rr:.1f}")
    else:
        reasons.append(f"RR {opp.risk_reward:.2f} OK")

    if base < min_score:
        q *= 0.7
        reasons.append(f"score {base:.0f} below {min_score:.0f}")

    if latest is not None:
        if latest.sharpe < 0:
            q *= 0.45
            reasons.append(f"family OOS Sharpe {latest.sharpe:.2f} negative")
        elif latest.sharpe < 0.25:
            q *= 0.75
            reasons.append(f"family OOS Sharpe weak {latest.sharpe:.2f}")
        else:
            q += min(12.0, latest.sharpe * 6)
            reasons.append(f"family OOS Sharpe {latest.sharpe:.2f}")
        if latest.stability_score < 0.35:
            q *= 0.85
            reasons.append(f"stability {latest.stability_score:.2f}")
        else:
            q += latest.stability_score * 8
    else:
        reasons.append("no family OOS evidence yet")
        q *= 0.9

    if fam_w < 0.3:
        q *= 0.7
        reasons.append("regime weight low for family")

    if opp.signal.value == "BUY" and opp.model_confidence < 0.45:
        q *= 0.85
        reasons.append("model confidence soft")

    q = float(max(0.0, min(100.0, q)))
    if q >= 72 and opp.risk_reward >= min_rr and opp.signal.value in ("BUY", "SHORT"):
        grade = "A"
    elif q >= 58 and opp.signal.value in ("BUY", "SHORT", "WAIT"):
        grade = "B"
    elif q >= 45:
        grade = "C"
    else:
        grade = "WATCH"

    return QualityAnnotation(
        grade=grade,
        quality_score=round(q, 1),
        evidence_sharpe=round(latest.sharpe, 3) if latest else None,
        evidence_stability=round(latest.stability_score, 3) if latest else None,
        reasons=reasons,
    )


def enrich_opportunity_dict(opp: Opportunity, ann: QualityAnnotation) -> dict:
    """Full trade card fields for desk / continuous feed."""
    sell = float(opp.target_1)
    buy = float(opp.ideal_entry)
    stop = float(opp.stop_loss)
    risk = abs(buy - stop)
    reward = abs(sell - buy)
    rr = reward / risk if risk > 0 else float(opp.risk_reward)
    return {
        "stock": opp.stock,
        "signal": opp.signal.value,
        "score": round(opp.opportunity_score, 1),
        "quality_grade": ann.grade,
        "quality_score": ann.quality_score,
        "quality_reasons": ann.reasons,
        "evidence_sharpe": ann.evidence_sharpe,
        "evidence_stability": ann.evidence_stability,
        "strategy": opp.strategy_selected,
        "sector": opp.sector,
        "current": round(opp.current_price, 2),
        "buy": round(buy, 2),
        "buy_low": round(opp.entry_range_low, 2),
        "buy_high": round(opp.entry_range_high, 2),
        "sell": round(sell, 2),
        "sell_t2": round(opp.target_2, 2),
        "sell_t3": round(opp.target_3, 2),
        "stop": round(stop, 2),
        "rr": round(rr, 2),
        "qty": opp.position_quantity,
        "capital": round(opp.capital_required, 2),
        "risk_rs": round(opp.max_rupee_risk, 2),
        "holding": opp.expected_holding_period,
        "why": opp.why_good,
        "risk_note": opp.what_could_go_wrong,
        "invalidation": opp.invalidation_condition,
        "entry_justification": opp.entry_justification,
        "exit_justification": opp.exit_justification,
        "timestamp": opp.data_timestamp.isoformat(),
    }
