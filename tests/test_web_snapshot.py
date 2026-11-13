import json
from pathlib import Path

from ati.web.snapshot import build_snapshot, write_snapshot


def test_build_snapshot_fast_no_scan(tmp_path):
    snap = build_snapshot(run_scan=False)
    assert "account" in snap
    assert "regime" in snap
    assert "research" in snap
    assert snap["research"]["adaptive_oos_sharpe"] > 0
    path = write_snapshot(tmp_path / "snap.json", run_scan=False)
    data = json.loads(Path(path).read_text())
    assert data["brand"] == "ATI Desk"
