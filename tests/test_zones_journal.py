import pandas as pd
import numpy as np

from ati.journal.store import (
    init_db,
    journal_close,
    journal_open,
    list_suggestions,
    performance_stats,
    save_suggestions,
)
from ati.structure.supply_demand import build_zone_context, refine_entry_ranges


def test_zones_and_ranges():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    close = np.linspace(100, 120, 80) + np.sin(np.linspace(0, 8, 80)) * 2
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 100000,
        },
        index=idx,
    )
    z = build_zone_context(df, df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}), float(close[-1]))
    assert z.entry_bias in ("long_ok", "wait_pullback", "avoid")
    ranges = refine_entry_ranges(buy=110, stop=105, sell=120, zones=z)
    assert len(ranges["entry_range"]) == 2
    assert ranges["entry_range"][0] <= ranges["entry_range"][1]


def test_journal_and_suggestion_history(tmp_path, monkeypatch):
    monkeypatch.setenv("ATI_STATE_DIR", str(tmp_path))
    # force db under tmp via monkeypatch of path function
    import ati.journal.store as store

    monkeypatch.setattr(store, "_db_path", lambda: tmp_path / "journal.db")
    init_db()
    save_suggestions(
        [
            {
                "stock": "AAA",
                "signal": "BUY",
                "buy": 100,
                "sell": 110,
                "stop": 95,
                "why_chosen": "test why",
                "quality_grade": "A",
            }
        ]
    )
    hist = list_suggestions()
    assert len(hist) >= 1
    assert "why_chosen" in hist[0]
    j = journal_open(stock="AAA", qty=2, entry_price=100, stop=95, target=110, suggestion_id=hist[0]["suggestion_id"])
    closed = journal_close(j["id"], exit_price=110)
    assert closed["pnl"] > 0
    stats = performance_stats()
    assert stats["journal_closed"] >= 1
