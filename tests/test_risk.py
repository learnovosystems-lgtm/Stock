from ati.engine.risk import PortfolioRiskState, RiskEngine


def test_position_sizing_respects_1pct_and_stop():
    eng = RiskEngine()
    state = PortfolioRiskState(equity=100_000, cash=100_000, open_risk_rupees=0, open_positions=0, day_pnl=0)
    res = eng.size_long(equity=100_000, cash=100_000, entry=1000, stop=980, state=state)
    assert res.allowed
    assert res.max_rupee_risk <= 1000 + 1e-6
    assert res.quantity == 50  # 1000 / 20


def test_rejects_when_max_positions():
    eng = RiskEngine()
    state = PortfolioRiskState(equity=100_000, cash=100_000, open_risk_rupees=0, open_positions=3, day_pnl=0)
    res = eng.size_long(equity=100_000, cash=100_000, entry=1000, stop=980, state=state)
    assert not res.allowed


def test_never_sizes_invalid_stop():
    eng = RiskEngine()
    state = PortfolioRiskState(equity=100_000, cash=100_000, open_risk_rupees=0, open_positions=0, day_pnl=0)
    res = eng.size_long(equity=100_000, cash=100_000, entry=1000, stop=1005, state=state)
    assert not res.allowed
