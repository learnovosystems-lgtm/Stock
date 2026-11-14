"""Persistent DP charge ledger — once per broker / instrument / trading date.

Two-phase protocol (crash-safe with paper fills):
  1. reserve_dp_charge(fill_id) → True if this attempt may debit DP
  2. confirm_dp_charge(fill_id) after the sell fill is committed
  3. release_dp_charge(fill_id) if the sell fails / rolls back

Only *confirmed* rows permanently consume the once-per-day slot.
Stale *pending* rows can be released so a failed sell cannot block retries.
Concurrent reserves for the same key yield exactly one winner.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

from ati.config import STATE_DIR, ensure_dirs

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"


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
                status TEXT NOT NULL,
                fill_id TEXT,
                charged_at TEXT NOT NULL,
                PRIMARY KEY (broker, instrument_id, trade_date)
            )
            """
        )
        # Migrate older schemas that lacked status/fill_id
        cols = {r[1] for r in cx.execute("PRAGMA table_info(dp_charges)").fetchall()}
        if "status" not in cols:
            cx.execute(
                f"ALTER TABLE dp_charges ADD COLUMN status TEXT NOT NULL DEFAULT '{STATUS_CONFIRMED}'"
            )
        if "fill_id" not in cols:
            cx.execute("ALTER TABLE dp_charges ADD COLUMN fill_id TEXT")
    return db


def _key(broker: str, instrument_id: str, trade_date: date | str) -> tuple[str, str, str]:
    td = trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date)[:10]
    return broker.lower(), instrument_id.upper(), td


def dp_already_charged(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    """True if a *confirmed* DP debit exists (pending does not count for MTM peeks)."""
    init_dp_ledger(path)
    b, inst, td = _key(broker, instrument_id, trade_date)
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        row = cx.execute(
            "SELECT 1 FROM dp_charges WHERE broker=? AND instrument_id=? AND trade_date=? AND status=?",
            (b, inst, td, STATUS_CONFIRMED),
        ).fetchone()
    return row is not None


def reserve_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    fill_id: str,
    path: Path | None = None,
) -> bool:
    """Atomically reserve the DP slot for this fill_id.

    Returns True iff this caller may debit DP (new reserve or same fill_id retry).
    Returns False if another fill already reserved/confirmed the key.
    """
    init_dp_ledger(path)
    b, inst, td = _key(broker, instrument_id, trade_date)
    now = datetime.now().isoformat()
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cx.execute("BEGIN IMMEDIATE")
        row = cx.execute(
            "SELECT status, fill_id FROM dp_charges WHERE broker=? AND instrument_id=? AND trade_date=?",
            (b, inst, td),
        ).fetchone()
        if row is None:
            cx.execute(
                """
                INSERT INTO dp_charges(broker, instrument_id, trade_date, status, fill_id, charged_at)
                VALUES (?,?,?,?,?,?)
                """,
                (b, inst, td, STATUS_PENDING, fill_id, now),
            )
            cx.execute("COMMIT")
            return True
        status, existing_fill = row
        if status == STATUS_CONFIRMED:
            cx.execute("COMMIT")
            return False
        if status == STATUS_PENDING and existing_fill == fill_id:
            cx.execute("COMMIT")
            return True  # idempotent retry of same fill
        cx.execute("COMMIT")
        return False


def confirm_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    fill_id: str,
    path: Path | None = None,
) -> bool:
    """Promote pending → confirmed for this fill_id. Returns True if confirmed."""
    init_dp_ledger(path)
    b, inst, td = _key(broker, instrument_id, trade_date)
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cur = cx.execute(
            """
            UPDATE dp_charges SET status=?, charged_at=?
            WHERE broker=? AND instrument_id=? AND trade_date=? AND fill_id=? AND status=?
            """,
            (STATUS_CONFIRMED, datetime.now().isoformat(), b, inst, td, fill_id, STATUS_PENDING),
        )
        return cur.rowcount == 1


def release_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    fill_id: str,
    path: Path | None = None,
) -> bool:
    """Drop a pending reservation so a failed sell does not consume the day slot."""
    init_dp_ledger(path)
    b, inst, td = _key(broker, instrument_id, trade_date)
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cur = cx.execute(
            """
            DELETE FROM dp_charges
            WHERE broker=? AND instrument_id=? AND trade_date=? AND fill_id=? AND status=?
            """,
            (b, inst, td, fill_id, STATUS_PENDING),
        )
        return cur.rowcount == 1


def try_record_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
    fill_id: str | None = None,
) -> bool:
    """Reserve+confirm in one step (backtests / callers without a separate fill commit)."""
    fid = fill_id or f"auto-{datetime.now().isoformat()}"
    if not reserve_dp_charge(
        broker=broker, instrument_id=instrument_id, trade_date=trade_date, fill_id=fid, path=path
    ):
        return False
    confirm_dp_charge(
        broker=broker, instrument_id=instrument_id, trade_date=trade_date, fill_id=fid, path=path
    )
    return True


def record_dp_charge(
    *,
    broker: str,
    instrument_id: str,
    trade_date: date | str,
    path: Path | None = None,
) -> bool:
    return try_record_dp_charge(
        broker=broker, instrument_id=instrument_id, trade_date=trade_date, path=path
    )


def clear_dp_ledger(path: Path | None = None) -> None:
    init_dp_ledger(path)
    with sqlite3.connect(_db_path(path), timeout=30.0) as cx:
        cx.execute("DELETE FROM dp_charges")
