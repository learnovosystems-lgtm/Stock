from datetime import datetime, timezone

from ati.models.opportunity import MarketRegime, Opportunity, SignalType
from ati.paper.portfolio import PaperBroker


def _opp(stock: str, strategy: str, sector: str = "IT", stop: float = 95.0) -> Opportunity:
    return Opportunity(
        stock=stock,
        signal=SignalType.BUY,
        current_price=100,
        ideal_entry=100,
        entry_range_low=99,
        entry_range_high=101,
        stop_loss=stop,
        target_1=110,
        target_2=115,
        target_3=120,
        risk_reward=2.0,
        position_quantity=10,
        capital_required=1000,
        max_rupee_risk=50,
        expected_holding_period="10d",
        opportunity_score=70,
        strategy_score=0.8,
        market_regime_suitability=0.7,
        entry_quality_score=0.7,
        risk_score=0.3,
        model_confidence=0.6,
        est_prob_target_before_stop=0.55,
        expected_return=0.05,
        strategy_selected=strategy,
        market_regime=MarketRegime.BULL,
        sector=sector,
        sector_strength=0.01,
        relative_strength_vs_nifty=0.05,
        entry_justification="t",
        exit_justification="t",
        invalidation_condition="t",
        why_good="t",
        what_could_go_wrong="t",
        data_timestamp=datetime.now(timezone.utc),
    )


def test_rejects_disallowed_strategy(tmp_path):
    broker = PaperBroker(
        db_path=tmp_path / "p.db",
        allowed_strategy_substrings=("dual_momentum",),
        symbol_sectors={"AAA": "IT", "BBB": "AUTO"},
    )
    ok, msg = broker.try_enter(_opp("AAA", "breakout_v1"))
    assert not ok
    assert "not allowed" in msg.lower()


def test_sector_cap_and_allowed_entry(tmp_path):
    broker = PaperBroker(
        db_path=tmp_path / "p2.db",
        allowed_strategy_substrings=("dual_momentum", "relative_strength"),
        symbol_sectors={"AAA.NS": "IT", "CCC.NS": "IT", "BBB.NS": "AUTO"},
    )
    ok1, _ = broker.try_enter(_opp("AAA", "dual_momentum_v2", sector="IT", stop=90))
    assert ok1
    ok2, msg2 = broker.try_enter(_opp("CCC", "dual_momentum_v2", sector="IT", stop=90))
    assert not ok2
    assert "sector" in msg2.lower()
    ok3, _ = broker.try_enter(_opp("BBB", "ensemble:relative_strength+trend_following", sector="AUTO", stop=90))
    assert ok3


def test_rejects_tiny_stop(tmp_path):
    broker = PaperBroker(
        db_path=tmp_path / "p3.db",
        allowed_strategy_substrings=("dual_momentum",),
    )
    ok, msg = broker.try_enter(_opp("AAA", "dual_momentum_v2", stop=99.5))
    assert not ok
    assert "tight" in msg.lower()
