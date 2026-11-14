"""Zerodha Kite Connect adapter — live positions and P&L.

Credentials (never hardcode; set via Cursor secrets / env):
  KITE_API_KEY
  KITE_ACCESS_TOKEN
Optional:
  KITE_API_SECRET  (login/session refresh flows)

When credentials are missing, returns a clearly labelled sample payload so the
desk UI still works offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


KITE_BASE = "https://api.kite.trade"


@dataclass
class LivePosition:
    symbol: str
    exchange: str
    product: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_pct: float
    day_pnl: float
    buy_value: float
    sell_value: float
    side: str  # LONG / SHORT
    source: str = "zerodha"


@dataclass
class BrokerSnapshot:
    connected: bool
    source: str
    message: str
    equity: float | None
    positions: list[LivePosition]
    raw_orders: list[dict[str, Any]]
    fetched_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "source": self.source,
            "message": self.message,
            "equity": self.equity,
            "positions": [asdict(p) for p in self.positions],
            "orders": self.raw_orders,
            "fetched_at": self.fetched_at,
            "total_pnl": round(sum(p.pnl for p in self.positions), 2),
            "total_day_pnl": round(sum(p.day_pnl for p in self.positions), 2),
        }


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


def credentials_present() -> bool:
    return bool(_env("KITE_API_KEY") and _env("KITE_ACCESS_TOKEN"))


def _kite_get(path: str) -> dict[str, Any]:
    api_key = _env("KITE_API_KEY")
    token = _env("KITE_ACCESS_TOKEN")
    if not api_key or not token:
        raise RuntimeError("KITE_API_KEY / KITE_ACCESS_TOKEN not set")
    req = urllib.request.Request(
        f"{KITE_BASE}{path}",
        headers={
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{token}",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("message") or "Kite API error")
    return payload.get("data") or {}


def _parse_positions(data: dict[str, Any]) -> list[LivePosition]:
    # Kite returns net + day; prefer net for open book
    rows = data.get("net") or data.get("day") or []
    out: list[LivePosition] = []
    for r in rows:
        qty = int(r.get("quantity") or 0)
        if qty == 0:
            continue
        avg = float(r.get("average_price") or 0)
        last = float(r.get("last_price") or avg)
        pnl = float(r.get("pnl") or (last - avg) * qty)
        day_pnl = float(r.get("m2m") or r.get("unrealised") or 0)
        buy_value = float(r.get("buy_value") or abs(avg * qty))
        sell_value = float(r.get("sell_value") or 0)
        pnl_pct = (last / avg - 1.0) if avg > 0 else 0.0
        tradingsymbol = str(r.get("tradingsymbol") or "")
        out.append(
            LivePosition(
                symbol=tradingsymbol,
                exchange=str(r.get("exchange") or "NSE"),
                product=str(r.get("product") or ""),
                quantity=qty,
                average_price=round(avg, 2),
                last_price=round(last, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 4),
                day_pnl=round(day_pnl, 2),
                buy_value=round(buy_value, 2),
                sell_value=round(sell_value, 2),
                side="LONG" if qty > 0 else "SHORT",
            )
        )
    return out


def _sample_snapshot() -> BrokerSnapshot:
    """Offline demo so the desk renders without secrets."""
    return BrokerSnapshot(
        connected=False,
        source="sample",
        message="Zerodha credentials not bound — showing sample layout. Bind KITE_API_KEY + KITE_ACCESS_TOKEN.",
        equity=None,
        positions=[
            LivePosition(
                symbol="SAMPLE",
                exchange="NSE",
                product="CNC",
                quantity=0,
                average_price=0.0,
                last_price=0.0,
                pnl=0.0,
                pnl_pct=0.0,
                day_pnl=0.0,
                buy_value=0.0,
                sell_value=0.0,
                side="LONG",
                source="sample",
            )
        ],
        raw_orders=[],
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def fetch_live_snapshot() -> BrokerSnapshot:
    """Fetch live Zerodha net positions + order book summary."""
    now = datetime.now(timezone.utc).isoformat()
    if not credentials_present():
        snap = _sample_snapshot()
        # Don't show fake SAMPLE row as a real position in UI aggregates
        snap.positions = []
        return snap

    try:
        pos_data = _kite_get("/portfolio/positions")
        positions = _parse_positions(pos_data)
        orders: list[dict[str, Any]] = []
        try:
            od = _kite_get("/orders")
            if isinstance(od, list):
                orders = od[-20:]
        except Exception:  # noqa: BLE001
            orders = []

        margins = None
        equity = None
        try:
            margins = _kite_get("/user/margins")
            equity = float((margins.get("equity") or {}).get("net") or 0) or None
        except Exception:  # noqa: BLE001
            pass

        return BrokerSnapshot(
            connected=True,
            source="zerodha",
            message="Live Zerodha positions",
            equity=equity,
            positions=positions,
            raw_orders=orders,
            fetched_at=now,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="ignore")[:300]
        return BrokerSnapshot(
            connected=False,
            source="zerodha",
            message=f"Kite HTTP {exc.code}: {body}",
            equity=None,
            positions=[],
            raw_orders=[],
            fetched_at=now,
        )
    except Exception as exc:  # noqa: BLE001
        return BrokerSnapshot(
            connected=False,
            source="zerodha",
            message=f"Kite error: {exc}",
            equity=None,
            positions=[],
            raw_orders=[],
            fetched_at=now,
        )
