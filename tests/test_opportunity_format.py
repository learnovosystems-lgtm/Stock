from datetime import datetime, timezone

from ati.models.opportunity import MarketRegime, Opportunity, SignalType


def test_opportunity_report_contains_required_fields():
    opp = Opportunity(
        stock="RELIANCE",
        signal=SignalType.WAIT,
        current_price=1000,
        ideal_entry=995,
        entry_range_low=990,
        entry_range_high=1005,
        stop_loss=970,
        target_1=1045,
        target_2=1070,
        target_3=1095,
        risk_reward=2.0,
        position_quantity=0,
        capital_required=0,
        max_rupee_risk=0,
        expected_holding_period="5-20 trading days",
        opportunity_score=55,
        strategy_score=0.7,
        market_regime_suitability=0.6,
        entry_quality_score=0.5,
        risk_score=0.4,
        model_confidence=0.55,
        est_prob_target_before_stop=0.52,
        expected_return=0.02,
        strategy_selected="momentum_v1",
        market_regime=MarketRegime.BULL,
        sector="ENERGY",
        sector_strength=0.01,
        relative_strength_vs_nifty=0.04,
        entry_justification="test",
        exit_justification="test",
        invalidation_condition="test",
        why_good="test",
        what_could_go_wrong="test",
        data_timestamp=datetime.now(timezone.utc),
    )
    text = opp.format_report()
    for key in [
        "Stock:",
        "Exchange:",
        "Signal:",
        "Ideal entry:",
        "Stop-loss:",
        "Target 1:",
        "Risk/reward:",
        "Opportunity score:",
        "Model confidence:",
        "Strategy selected:",
        "Market regime:",
        "Data timestamp:",
    ]:
        assert key in text
