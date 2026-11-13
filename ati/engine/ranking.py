"""Signal ranking across the universe."""

from __future__ import annotations

from datetime import datetime, timezone

from ati.engine.entry import EntryDecision
from ati.engine.risk import PositionSizeResult
from ati.models.opportunity import MarketRegime, Opportunity, SignalType
from ati.regime.detector import RegimeState
from ati.strategies.base import RawSignal


def opportunity_score(
    *,
    strategy_score: float,
    regime_suitability: float,
    entry_quality: float,
    risk_score: float,
    model_confidence: float,
    est_prob: float,
) -> float:
    return float(
        100
        * (
            0.25 * strategy_score
            + 0.15 * regime_suitability
            + 0.20 * entry_quality
            + 0.10 * (1 - risk_score)
            + 0.15 * model_confidence
            + 0.15 * est_prob
        )
    )


def build_opportunity(
    *,
    stock: str,
    sector: str,
    raw: RawSignal,
    regime: RegimeState,
    entry: EntryDecision,
    sizing: PositionSizeResult,
    current_price: float,
    sector_strength: float,
    rs_nifty: float,
    model_confidence: float,
    est_prob: float,
    expected_return: float,
    strategy_label: str | None = None,
) -> Opportunity:
    risk = abs(entry.ideal_entry - raw.stop)
    reward = abs(raw.target_1 - entry.ideal_entry)
    rr = reward / risk if risk > 0 else 0.0
    regime_suit = min(1.0, regime.confidence + (0.2 if raw.family in regime.preferred_families() else 0.0))
    risk_score = min(1.0, (sizing.max_rupee_risk / 1000.0) if sizing.max_rupee_risk else 1.0)

    if entry.action == "BUY" and sizing.allowed:
        signal = SignalType.BUY if raw.direction == "LONG" else SignalType.SHORT
    elif entry.action == "WAIT":
        signal = SignalType.WAIT
    else:
        signal = SignalType.NO_TRADE

    strat_score = min(1.0, raw.score)
    opp = opportunity_score(
        strategy_score=strat_score,
        regime_suitability=regime_suit,
        entry_quality=entry.entry_quality,
        risk_score=risk_score,
        model_confidence=model_confidence,
        est_prob=est_prob,
    )

    return Opportunity(
        stock=stock.replace(".NS", ""),
        exchange="NSE",
        signal=signal,
        current_price=current_price,
        ideal_entry=entry.ideal_entry,
        entry_range_low=entry.entry_low,
        entry_range_high=entry.entry_high,
        stop_loss=raw.stop,
        target_1=raw.target_1,
        target_2=raw.target_2,
        target_3=raw.target_3,
        risk_reward=rr,
        position_quantity=sizing.quantity if signal == SignalType.BUY else 0,
        capital_required=sizing.capital_required if signal == SignalType.BUY else 0.0,
        max_rupee_risk=sizing.max_rupee_risk if signal == SignalType.BUY else 0.0,
        expected_holding_period=raw.holding_period,
        opportunity_score=opp,
        strategy_score=strat_score,
        market_regime_suitability=regime_suit,
        entry_quality_score=entry.entry_quality,
        risk_score=risk_score,
        model_confidence=model_confidence,
        est_prob_target_before_stop=est_prob,
        expected_return=expected_return,
        strategy_selected=strategy_label or raw.strategy_id,
        market_regime=regime.regime,
        sector=sector,
        sector_strength=sector_strength,
        relative_strength_vs_nifty=rs_nifty,
        entry_justification=f"{raw.entry_justification} | {entry.reason}",
        exit_justification=raw.exit_justification,
        invalidation_condition=raw.invalidation_condition,
        why_good=raw.why_good,
        what_could_go_wrong=raw.what_could_go_wrong,
        data_timestamp=datetime.now(timezone.utc),
        breakout_confirmation_price=entry.breakout_confirmation,
        max_acceptable_entry=entry.max_acceptable_entry,
        invalidation_price=entry.invalidation_price,
        extras={"selection_family": raw.family},
    )


def rank_opportunities(opps: list[Opportunity]) -> list[Opportunity]:
    # BUY first by score, then WAIT, never invent fillers
    def key(o: Opportunity):
        pri = {"BUY": 0, "SHORT": 0, "WAIT": 1, "EXIT": 2, "SELL": 2, "NO_TRADE": 3}
        return (pri.get(o.signal.value, 9), -o.opportunity_score)

    return sorted(opps, key=key)
