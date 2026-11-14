from datetime import datetime, timezone

from ati.brokers.zerodha import credentials_present, fetch_live_snapshot
from ati.engine.suggestion_quality import annotate_opportunity, enrich_opportunity_dict
from ati.models.opportunity import MarketRegime, Opportunity, SignalType, StrategyMetrics


def test_zerodha_sample_without_secrets():
    snap = fetch_live_snapshot()
    assert snap.source in ("sample", "zerodha")
    if not credentials_present():
        assert snap.connected is False
        assert snap.positions == []


def test_quality_annotation_demotes_weak_rr():
    opp = Opportunity(
        stock="AAA",
        signal=SignalType.BUY,
        current_price=100,
        ideal_entry=100,
        entry_range_low=99,
        entry_range_high=101,
        stop_loss=98,
        target_1=102,
        target_2=103,
        target_3=104,
        risk_reward=1.0,
        position_quantity=10,
        capital_required=1000,
        max_rupee_risk=20,
        expected_holding_period="10d",
        opportunity_score=70,
        strategy_score=0.8,
        market_regime_suitability=0.7,
        entry_quality_score=0.7,
        risk_score=0.2,
        model_confidence=0.6,
        est_prob_target_before_stop=0.55,
        expected_return=0.05,
        strategy_selected="dual_momentum_v2",
        market_regime=MarketRegime.BULL,
        sector="IT",
        sector_strength=0.01,
        relative_strength_vs_nifty=0.05,
        entry_justification="t",
        exit_justification="t",
        invalidation_condition="t",
        why_good="t",
        what_could_go_wrong="t",
        data_timestamp=datetime.now(timezone.utc),
        extras={"selection_family": "relative_strength"},
    )
    metrics = [
        StrategyMetrics(
            strategy_id="dual_momentum_nifty_gated",
            version="wf",
            sample_count=30,
            win_rate=0.5,
            expectancy=0.02,
            sharpe=0.68,
            sortino=0.6,
            max_drawdown=-0.2,
            profit_factor=1.5,
            avg_holding_days=40,
            stability_score=0.48,
            total_return=0.3,
            oos=True,
        )
    ]
    ann = annotate_opportunity(opp, metrics, min_rr=2.0)
    card = enrich_opportunity_dict(opp, ann)
    assert "buy" in card and "sell" in card and "stop" in card and "rr" in card
    assert ann.grade in ("A", "B", "C", "WATCH")
    assert card["rr"] < 2.0
