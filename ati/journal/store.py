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

from ati.config import STATE_DIR, DEFAULT, ensure_dirs
from ati.engine.charges import default_charge_engine
from ati.models.charges import Segment, Side


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
        cols = {r[1] for r in cx.execute("PRAGMA table_info(journal)").fetchall()}
        for col, decl in (
            ("gross_pnl", "REAL"),
            ("fees", "REAL"),
            ("net_pnl", "REAL"),
            ("segment", "TEXT"),
            ("schedule_id", "TEXT"),
            ("charges_json", "TEXT"),
        ):
            if col not in cols:
                cx.execute(f"ALTER TABLE journal ADD COLUMN {col} {decl}")


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


def journal_close(
    journal_id: str,
    exit_price: float,
    notes: str = "",
    *,
    segment: str | None = None,
    apply_slippage: bool = False,
) -> dict[str, Any]:
    """Close a journal lot with ChargeEngine net P&L.

    Manual journal prices are treated as already-executed (apply_slippage=False)
    unless the caller opts in. Fees always apply for the configured broker.
    """
    init_db()
    seg = Segment(segment or DEFAULT.default_segment)
    engine = default_charge_engine()
    session = engine.new_session()
    with _conn() as cx:
        row = cx.execute("SELECT * FROM journal WHERE id=?", (journal_id,)).fetchone()
        if not row:
            raise ValueError("journal id not found")
        qty = float(row["qty"])
        entry = float(row["entry_price"])
        stock = str(row["stock"])
        buy = engine.economics_for_fill(
            side=Side.BUY,
            qty=qty,
            raw_price=entry,
            segment=seg,
            apply_slippage=apply_slippage,
            session=session,
            order_id=f"journal-buy-{journal_id}",
            symbol=stock,
            trade_date=row["entry_at"],
        )
        sell = engine.economics_for_fill(
            side=Side.SELL,
            qty=qty,
            raw_price=float(exit_price),
            segment=seg,
            apply_slippage=apply_slippage,
            session=session,
            order_id=f"journal-sell-{journal_id}",
            symbol=stock,
            trade_date=_now().isoformat(),
        )
        rt = engine.reconcile_round_trip(buy=buy, sells=[sell])
        gross = (float(exit_price) - entry) * qty
        fees = rt.total_fees
        net = rt.net_pnl
        pnl_pct = net / (entry * qty) if entry and qty else 0.0
        charges_blob = json.dumps({"buy": buy.charges.to_dict(), "sell": sell.charges.to_dict()})
        cx.execute(
            """
            UPDATE journal SET exit_price=?, exit_at=?, pnl=?, pnl_pct=?, notes=COALESCE(?, notes),
                   status='closed', gross_pnl=?, fees=?, net_pnl=?, segment=?, schedule_id=?, charges_json=?
            WHERE id=?
            """,
            (
                exit_price,
                _now().isoformat(),
                net,
                pnl_pct,
                notes or None,
                gross,
                fees,
                net,
                seg.value,
                buy.charges.schedule_id,
                charges_blob,
                journal_id,
            ),
        )
        if row["suggestion_id"]:
            outcome = "hit_target" if net > 0 else "hit_sl"
            cx.execute(
                """
                UPDATE suggestions SET status=?, outcome=?, outcome_at=?, outcome_note=?
                WHERE id=?
                """,
                (outcome, outcome, _now().isoformat(), f"journal close net={net:.2f}", row["suggestion_id"]),
            )
        return {
            "id": journal_id,
            "pnl": round(net, 2),
            "gross_pnl": round(gross, 2),
            "fees": round(fees, 2),
            "net_pnl": round(net, 2),
            "pnl_pct": round(pnl_pct, 4),
            "status": "closed",
            "schedule_id": buy.charges.schedule_id,
        }


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
        pnl = cx.execute(
            "SELECT COALESCE(SUM(COALESCE(net_pnl, pnl)),0) s FROM journal WHERE status='closed'"
        ).fetchone()["s"]
        fees = cx.execute(
            "SELECT COALESCE(SUM(fees),0) s FROM journal WHERE status='closed'"
        ).fetchone()["s"]
        gross = cx.execute(
            "SELECT COALESCE(SUM(COALESCE(gross_pnl, pnl)),0) s FROM journal WHERE status='closed'"
        ).fetchone()["s"]
        wins = cx.execute(
            "SELECT COUNT(*) c FROM journal WHERE status='closed' AND COALESCE(net_pnl, pnl)>0"
        ).fetchone()["c"]
        losses = cx.execute(
            "SELECT COUNT(*) c FROM journal WHERE status='closed' AND COALESCE(net_pnl, pnl)<=0"
        ).fetchone()["c"]
    decided = hit_t + hit_sl
    win_rate = (hit_t / decided) if decided else None
    from ati.engine.charges import default_charge_engine

    info = default_charge_engine().schedule_info()
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
        "journal_realized_gross": round(float(gross), 2),
        "journal_realized_fees": round(float(fees), 2),
        "journal_realized_net": round(float(pnl), 2),
        "charges": info,
        "successful": hit_t + wins,
        "failed": hit_sl + losses,
        "live": live + open_j,
    }
