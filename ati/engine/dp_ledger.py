"""Persistent DP charge ledger — once per broker / instrument / trading date.

Survives process restarts so delivery sell debits are not double-charged.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

from ati.config import STATE_DIR, ensure_dirs


def _db_path(path: Path | None = None) -> Path:
    ensure_dirs()
    return path or (STATE_DIR / "dp_charges.db")


def init_dp_ledger(path: Path | None = None) -> Path:
    db = _db_path(path)
    with sqlite3.connect(db) as cx:
        cx.execute(
            """
            CREATE TABLE IF NOT EXISTS dp_charges (
                broker TEXT NOT NULL,
                instrument_id TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                charged_at TEXT NOT NULL,
                PRIMARY KEY (broker, instrument_id, trade_date)
            )
            """
        )
    return db


def dp_already_charged(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    init_dp_ledger(path)
    td = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    with sqlite3.connect(_db_path(path)) as cx:
        row = cx.execute(
            "SELECT 1 FROM dp_charges WHERE broker=? AND instrument_id=? AND trade_date=?",
            (broker.lower(), instrument_id.upper(), td),
        ).fetchone()
    return row is not None


def record_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    """Record DP charge. Returns True if newly recorded, False if already present."""
    init_dp_ledger(path)
    td = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    now = datetime.now(tz=None).isoformat()  # naive local stamp for ledger audit
    with sqlite3.connect(_db_path(path)) as cx:
        try:
            cx.execute(
                "INSERT INTO dp_charges(broker, instrument_id, trade_date, charged_at) VALUES (?,?,?,?)",
                (broker.lower(), instrument_id.upper(), td, now),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def clear_dp_ledger(path: Path | None = None) -> None:
    """Test helper — wipe ledger."""
    init_dp_ledger(path)
    with sqlite3.connect(_db_path(path)) as cx:
        cx.execute("DELETE FROM dp_charges")
