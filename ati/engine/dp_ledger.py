"""Persistent DP charge ledger — once per broker / instrument / trading date.

Survives process restarts so delivery sell debits are not double-charged.
Recording is atomic (INSERT OR IGNORE under BEGIN IMMEDIATE) so concurrent
sellers for the same key yield exactly one successful charge.
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
    with sqlite3.connect(db, timeout=30.0) as cx:
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
    """Read-only peek (for MTM estimates). Prefer try_record_dp_charge for debiting."""
    init_dp_ledger(path)
    td = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        row = cx.execute(
            "SELECT 1 FROM dp_charges WHERE broker=? AND instrument_id=? AND trade_date=?",
            (broker.lower(), instrument_id.upper(), td),
        ).fetchone()
    return row is not None


def try_record_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    """Atomically claim the DP charge for (broker, instrument, date).

    Returns True iff this caller successfully recorded the charge (and therefore
    should debit DP). Returns False if another request already owns the key.
    Concurrent callers are serialized via BEGIN IMMEDIATE + INSERT OR IGNORE.
    """
    init_dp_ledger(path)
    td = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    now = datetime.now().isoformat()
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cx.execute("BEGIN IMMEDIATE")
        cur = cx.execute(
            """
            INSERT OR IGNORE INTO dp_charges(broker, instrument_id, trade_date, charged_at)
            VALUES (?,?,?,?)
            """,
            (broker.lower(), instrument_id.upper(), td, now),
        )
        cx.execute("COMMIT")
        return cur.rowcount == 1


def record_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    """Alias for try_record_dp_charge (backward compatible)."""
    return try_record_dp_charge(
        broker=broker,
        instrument_id=instrument_id,
        trade_date=trade_date,
        path=path,
    )


def clear_dp_ledger(path: Path | None = None) -> None:
    """Test helper — wipe ledger."""
    init_dp_ledger(path)
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cx.execute("DELETE FROM dp_charges")
