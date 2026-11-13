from ati.engine.exit import build_exit_plan, tighten_stop


def test_stop_never_widens_for_long():
    assert tighten_stop(100, 90, 110) == 100
    assert tighten_stop(100, 105, 110) == 105


def test_exit_plan_requires_stop_below_entry():
    plan = build_exit_plan(100, 95, 110, 115, 120)
    assert plan.current_stop == 95
    try:
        build_exit_plan(100, 105, 110, 115, 120)
        assert False, "expected ValueError"
    except ValueError:
        pass
