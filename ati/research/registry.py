"""Strategy version registry — production promotion requires validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ati.config import REGISTRY_DIR, ensure_dirs
from ati.models.opportunity import StrategyMetrics


class StrategyRegistry:
    def __init__(self, path: Path | None = None):
        ensure_dirs()
        self.path = path or (REGISTRY_DIR / "strategies.json")
        if not self.path.exists():
            self._write({"strategies": [], "experiments": []})

    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str))

    def record_metrics(self, metrics: StrategyMetrics, *, production: bool = False) -> None:
        data = self._read()
        entry = metrics.model_dump(mode="json")
        entry["production"] = production
        data["strategies"].append(entry)
        self._write(data)

    def record_experiment(self, name: str, params: dict, result: dict, success: bool) -> None:
        data = self._read()
        data["experiments"].append(
            {
                "name": name,
                "params": params,
                "result": result,
                "success": success,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(data)

    def list_production(self) -> list[dict]:
        data = self._read()
        return [s for s in data["strategies"] if s.get("production")]

    def promote(
        self,
        strategy_id: str,
        version: str,
        *,
        min_trades: int = 20,
        min_stability: float = 0.45,
        max_dd: float = -0.25,
    ) -> bool:
        """Promote only if latest OOS metrics clear gates."""
        data = self._read()
        candidates = [
            s
            for s in data["strategies"]
            if s.get("strategy_id") == strategy_id and s.get("version") == version and s.get("oos", True)
        ]
        if not candidates:
            return False
        latest = candidates[-1]
        if latest.get("sample_count", 0) < min_trades:
            return False
        if latest.get("stability_score", 0) < min_stability:
            return False
        if latest.get("max_drawdown", -1) < max_dd:
            return False
        # Demote others of same id
        for s in data["strategies"]:
            if s.get("strategy_id") == strategy_id:
                s["production"] = False
        latest["production"] = True
        self._write(data)
        return True

    def latest_metrics(self) -> list[StrategyMetrics]:
        data = self._read()
        out = []
        for s in data["strategies"][-50:]:
            try:
                out.append(StrategyMetrics.model_validate({k: v for k, v in s.items() if k != "production"}))
            except Exception:  # noqa: BLE001
                continue
        return out
