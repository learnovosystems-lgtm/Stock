from ati.engine.selection import StrategySelector
from ati.models.opportunity import MarketRegime
from ati.regime.detector import RegimeState
from ati.strategies.base import RawSignal


def _sig(family: str, score: float = 0.9) -> RawSignal:
    return RawSignal(
        strategy_id=f"{family}_v1",
        family=family,
        direction="LONG",
        score=score,
        ideal_entry=100,
        entry_low=99,
        entry_high=101,
        stop=95,
        target_1=110,
        target_2=115,
        target_3=120,
        breakout_confirmation=102,
        max_entry=101.5,
        invalidation=95,
        holding_period="5d",
        entry_justification="j",
        exit_justification="j",
        invalidation_condition="j",
        why_good="j",
        what_could_go_wrong="j",
    )


def test_uncertain_regime_returns_no_trade():
    sel = StrategySelector()
    regime = RegimeState(
        regime=MarketRegime.UNCERTAIN,
        nifty_trend=0,
        india_vix=15,
        adx=10,
        ma_structure=0,
        volatility=0.1,
        breadth=0,
        sector_rotation_score=0,
        confidence=0.2,
        notes="mixed",
    )
    out = sel.select([_sig("momentum")], regime)
    assert out.selected == []


def test_ensemble_when_multiple_families_agree():
    sel = StrategySelector()
    regime = RegimeState(
        regime=MarketRegime.BULL,
        nifty_trend=0.05,
        india_vix=14,
        adx=25,
        ma_structure=1,
        volatility=0.15,
        breadth=0.2,
        sector_rotation_score=0.01,
        confidence=0.8,
        notes="bull",
    )
    out = sel.select(
        [_sig("momentum"), _sig("relative_strength"), _sig("breakout")],
        regime,
        min_score=0.4,
    )
    assert out.ensemble
    assert len(out.selected) >= 2


def test_negative_measured_metrics_crush_family_weight():
    from datetime import datetime, timezone

    from ati.models.opportunity import StrategyMetrics

    metrics = [
        StrategyMetrics(
            strategy_id="breakout_momentum_baseline",
            version="1.0.0",
            sample_count=37,
            win_rate=0.4,
            expectancy=-0.005,
            sharpe=-0.45,
            sortino=-0.4,
            max_drawdown=-0.12,
            profit_factor=0.7,
            avg_holding_days=15,
            stability_score=0.3,
            total_return=-0.05,
            created_at=datetime.now(timezone.utc),
        )
    ]
    sel = StrategySelector(metrics)
    w_break = sel.family_weight("breakout", MarketRegime.BULL)
    w_rs = sel.family_weight("relative_strength", MarketRegime.BULL)
    assert w_break < w_rs
    assert w_break < 0.3
