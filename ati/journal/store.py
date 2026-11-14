"""Persistent suggestions + manual trade journal (no broker API).

User marks actual entry/exit prices for P&L. Suggestions are kept for review
with expiry timestamps. Target/SL hits are recorded for success/fail stats.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ati.config import STATE_DIR, ensure_dirs


def _db_path() -> Path:
    ensure_dirs()
    return STATE_DIR / "journal.db"


def _conn() -> sqlite3.Connection:
    cx = sqlite3.connect(_db_path())
    cx.row_factory = sqlite3.Row
    return cx


def init_db() -> None:
    with _conn() as cx:
        cx.executescript(
            """
            CREATE TABLE IF NOT EXISTS suggestions (
                id TEXT PRIMARY KEY,
                stock TEXT NOT NULL,
                signal TEXT NOT NULL,
                payload TEXT NOT NULL,
                why TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL,
                outcome TEXT,
                outcome_at TEXT,
                outcome_note TEXT
            );
            CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY,
                suggestion_id TEXT,
                stock TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_at TEXT NOT NULL,
                exit_price REAL,
                exit_at TEXT,
                stop REAL,
                target REAL,
                pnl REAL,
                pnl_pct REAL,
                notes TEXT,
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sug_status ON suggestions(status);
            CREATE INDEX IF NOT EXISTS idx_sug_stock ON suggestions(stock);
            """
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def save_suggestions(cards: list[dict[str, Any]], *, default_horizon_hours: float = 36.0) -> list[str]:
    """Persist new suggestion cards; keep history forever (status evolves)."""
    init_db()
    ids: list[str] = []
    now = _now()
    with _conn() as cx:
        for c in cards:
            sid = c.get("suggestion_id") or str(uuid.uuid4())
            expires = c.get("expires_at") or (now + timedelta(hours=default_horizon_hours)).isoformat()
            why = c.get("why_chosen") or c.get("why") or ""
            # Upsert by stock+signal+day bucket to avoid exact dup spam, but keep history rows
            cx.execute(
                """
                INSERT INTO suggestions(id, stock, signal, payload, why, created_at, expires_at, status)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    sid,
                    c.get("stock"),
                    c.get("signal"),
                    json.dumps(c, default=str),
                    why,
                    c.get("created_at") or now.isoformat(),
                    expires,
                    c.get("status") or "live",
                ),
            )
            ids.append(sid)
            c["suggestion_id"] = sid
            c["expires_at"] = expires
    return ids


def list_suggestions(
    *,
    include_expired: bool = True,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    init_db()
    now = _now().isoformat()
    q = "SELECT * FROM suggestions"
    args: list[Any] = []
    clauses = []
    if status:
        clauses.append("status = ?")
        args.append(status)
    if not include_expired:
        clauses.append("expires_at >= ?")
        args.append(now)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    out = []
    with _conn() as cx:
        for row in cx.execute(q, args):
            payload = json.loads(row["payload"])
            payload.update(
                {
                    "suggestion_id": row["id"],
                    "why_chosen": row["why"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "status": row["status"],
                    "outcome": row["outcome"],
                    "outcome_at": row["outcome_at"],
                    "outcome_note": row["outcome_note"],
                    "expired": row["expires_at"] < now and row["status"] == "live",
                }
            )
            out.append(payload)
    return out


def expire_stale() -> int:
    init_db()
    now = _now().isoformat()
    with _conn() as cx:
        cur = cx.execute(
            "UPDATE suggestions SET status='expired' WHERE status='live' AND expires_at < ?",
            (now,),
        )
        return cur.rowcount


def mark_suggestion_outcome(suggestion_id: str, outcome: str, note: str = "") -> None:
    init_db()
    with _conn() as cx:
        cx.execute(
            """
            UPDATE suggestions
            SET status=?, outcome=?, outcome_at=?, outcome_note=?
            WHERE id=?
            """,
            (outcome, outcome, _now().isoformat(), note, suggestion_id),
        )


def journal_open(
    *,
    stock: str,
    qty: float,
    entry_price: float,
    stop: float | None = None,
    target: float | None = None,
    suggestion_id: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    init_db()
    jid = str(uuid.uuid4())
    now = _now().isoformat()
    with _conn() as cx:
        cx.execute(
            """
            INSERT INTO journal(id, suggestion_id, stock, side, qty, entry_price, entry_at,
                                exit_price, exit_at, stop, target, pnl, pnl_pct, notes, status)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                jid,
                suggestion_id,
                stock.replace(".NS", ""),
                "LONG",
                qty,
                entry_price,
                now,
                None,
                None,
                stop,
                target,
                None,
                None,
                notes,
                "open",
            ),
        )
    return {"id": jid, "stock": stock, "qty": qty, "entry_price": entry_price, "status": "open"}


def journal_update_entry(journal_id: str, entry_price: float) -> None:
    init_db()
    with _conn() as cx:
        cx.execute("UPDATE journal SET entry_price=? WHERE id=? AND status='open'", (entry_price, journal_id))


def journal_close(journal_id: str, exit_price: float, notes: str = "") -> dict[str, Any]:
    init_db()
    with _conn() as cx:
        row = cx.execute("SELECT * FROM journal WHERE id=?", (journal_id,)).fetchone()
        if not row:
            raise ValueError("journal id not found")
        pnl = (exit_price - row["entry_price"]) * row["qty"]
        pnl_pct = exit_price / row["entry_price"] - 1.0 if row["entry_price"] else 0.0
        cx.execute(
            """
            UPDATE journal SET exit_price=?, exit_at=?, pnl=?, pnl_pct=?, notes=COALESCE(?, notes), status='closed'
            WHERE id=?
            """,
            (exit_price, _now().isoformat(), pnl, pnl_pct, notes or None, journal_id),
        )
        # Link suggestion outcome if present
        if row["suggestion_id"]:
            outcome = "hit_target" if pnl > 0 else "hit_sl"
            cx.execute(
                """
                UPDATE suggestions SET status=?, outcome=?, outcome_at=?, outcome_note=?
                WHERE id=?
                """,
                (outcome, outcome, _now().isoformat(), f"journal close pnl={pnl:.2f}", row["suggestion_id"]),
            )
        return {"id": journal_id, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 4), "status": "closed"}


def list_journal(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    with _conn() as cx:
        rows = cx.execute("SELECT * FROM journal ORDER BY entry_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def evaluate_hits_against_prices(prices: dict[str, float]) -> list[dict[str, Any]]:
    """Mark live suggestions that already hit target or SL vs last prices."""
    init_db()
    expire_stale()
    updates = []
    with _conn() as cx:
        rows = cx.execute("SELECT * FROM suggestions WHERE status='live'").fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            stock = row["stock"]
            px = prices.get(stock) or prices.get(stock + ".NS")
            if px is None:
                continue
            stop = float(payload.get("stop") or payload.get("ideal_stop") or 0)
            sell = float(payload.get("sell") or payload.get("ideal_sell") or 0)
            signal = row["signal"]
            if signal != "BUY":
                continue
            if stop and px <= stop:
                cx.execute(
                    """
                    UPDATE suggestions SET status='hit_sl', outcome='hit_sl', outcome_at=?, outcome_note=?
                    WHERE id=?
                    """,
                    (_now().isoformat(), f"last={px} <= stop={stop}", row["id"]),
                )
                updates.append({"id": row["id"], "stock": stock, "outcome": "hit_sl", "price": px})
            elif sell and px >= sell:
                cx.execute(
                    """
                    UPDATE suggestions SET status='hit_target', outcome='hit_target', outcome_at=?, outcome_note=?
                    WHERE id=?
                    """,
                    (_now().isoformat(), f"last={px} >= target={sell}", row["id"]),
                )
                updates.append({"id": row["id"], "stock": stock, "outcome": "hit_target", "price": px})
    return updates


def performance_stats() -> dict[str, Any]:
    init_db()
    expire_stale()
    with _conn() as cx:
        def count(status: str) -> int:
            return int(cx.execute("SELECT COUNT(*) c FROM suggestions WHERE status=?", (status,)).fetchone()["c"])

        live = count("live")
        hit_t = count("hit_target")
        hit_sl = count("hit_sl")
        expired = count("expired")
        closed_j = cx.execute("SELECT COUNT(*) c FROM journal WHERE status='closed'").fetchone()["c"]
        open_j = cx.execute("SELECT COUNT(*) c FROM journal WHERE status='open'").fetchone()["c"]
        pnl = cx.execute("SELECT COALESCE(SUM(pnl),0) s FROM journal WHERE status='closed'").fetchone()["s"]
        wins = cx.execute("SELECT COUNT(*) c FROM journal WHERE status='closed' AND pnl>0").fetchone()["c"]
        losses = cx.execute("SELECT COUNT(*) c FROM journal WHERE status='closed' AND pnl<=0").fetchone()["c"]
    decided = hit_t + hit_sl
    win_rate = (hit_t / decided) if decided else None
    return {
        "suggestions_live": live,
        "suggestions_hit_target": hit_t,
        "suggestions_hit_sl": hit_sl,
        "suggestions_expired": expired,
        "suggestion_win_rate": round(win_rate, 3) if win_rate is not None else None,
        "journal_open": open_j,
        "journal_closed": closed_j,
        "journal_wins": wins,
        "journal_losses": losses,
        "journal_realized_pnl": round(float(pnl), 2),
        "successful": hit_t + wins,
        "failed": hit_sl + losses,
        "live": live + open_j,
    }
