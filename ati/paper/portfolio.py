"""Paper-trading portfolio with Indian cost model and hard risk gates."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ati.config import STATE_DIR, DEFAULT, RiskConfig, ensure_dirs
from ati.engine.exit import ExitPlan, ExitReason, build_exit_plan, evaluate_exit, tighten_stop
from ati.engine.risk import PortfolioRiskState, RiskEngine
from ati.models.opportunity import Opportunity


@dataclass
class Position:
    symbol: str
    qty: int
    entry: float
    stop: float
    t1: float
    t2: float
    t3: float
    opened_at: str
    strategy: str
    bars_held: int = 0
    remaining_frac: float = 1.0
    targets_hit: str = ""  # comma ids
    realized_pnl: float = 0.0


class PaperBroker:
    def __init__(self, db_path: Path | None = None, risk: RiskConfig | None = None):
        ensure_dirs()
        self.db_path = db_path or (STATE_DIR / "paper.db")
        self.risk_cfg = risk or DEFAULT.risk
        self.risk = RiskEngine(self.risk_cfg)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as cx:
            cx.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    day_pnl REAL NOT NULL,
                    day_anchor REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    qty INTEGER,
                    entry REAL,
                    stop REAL,
                    t1 REAL,
                    t2 REAL,
                    t3 REAL,
                    opened_at TEXT,
                    strategy TEXT,
                    bars_held INTEGER,
                    remaining_frac REAL,
                    targets_hit TEXT,
                    realized_pnl REAL
                );
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    symbol TEXT,
                    side TEXT,
                    qty REAL,
                    price REAL,
                    pnl REAL,
                    note TEXT
                );
                """
            )
            cur = cx.execute("SELECT COUNT(*) FROM account")
            if cur.fetchone()[0] == 0:
                now = datetime.now(timezone.utc).isoformat()
                cap = self.risk_cfg.starting_capital
                cx.execute(
                    "INSERT INTO account(id, cash, equity, day_pnl, day_anchor, updated_at) VALUES (1,?,?,?,?,?)",
                    (cap, cap, 0.0, cap, now),
                )

    def account(self) -> dict:
        with self._conn() as cx:
            row = cx.execute(
                "SELECT cash, equity, day_pnl, day_anchor, updated_at FROM account WHERE id=1"
            ).fetchone()
        return {
            "cash": row[0],
            "equity": row[1],
            "day_pnl": row[2],
            "day_anchor": row[3],
            "updated_at": row[4],
        }

    def positions(self) -> list[Position]:
        with self._conn() as cx:
            rows = cx.execute("SELECT * FROM positions").fetchall()
        cols = [
            "symbol",
            "qty",
            "entry",
            "stop",
            "t1",
            "t2",
            "t3",
            "opened_at",
            "strategy",
            "bars_held",
            "remaining_frac",
            "targets_hit",
            "realized_pnl",
        ]
        return [Position(**dict(zip(cols, r))) for r in rows]

    def risk_state(self) -> PortfolioRiskState:
        acct = self.account()
        positions = self.positions()
        open_risk = 0.0
        for p in positions:
            open_risk += max(0.0, (p.entry - p.stop) * p.qty * p.remaining_frac)
        return PortfolioRiskState(
            equity=acct["equity"],
            cash=acct["cash"],
            open_risk_rupees=open_risk,
            open_positions=len(positions),
            day_pnl=acct["day_pnl"],
        )

    def try_enter(self, opp: Opportunity) -> tuple[bool, str]:
        if opp.signal.value != "BUY" or opp.position_quantity < 1:
            return False, "Signal not actionable BUY"
        state = self.risk_state()
        if any(p.symbol == opp.stock for p in self.positions()):
            return False, "Already in position"
        sizing = self.risk.size_long(
            equity=state.equity,
            cash=state.cash,
            entry=opp.ideal_entry,
            stop=opp.stop_loss,
            state=state,
        )
        if not sizing.allowed:
            return False, sizing.reason
        qty = min(sizing.quantity, opp.position_quantity)
        px = self.risk.apply_buy_slippage(opp.ideal_entry)
        cost = px * qty
        acct = self.account()
        if cost > acct["cash"]:
            return False, "Insufficient cash"
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as cx:
            cx.execute(
                "UPDATE account SET cash=?, updated_at=? WHERE id=1",
                (acct["cash"] - cost, now),
            )
            cx.execute(
                """
                INSERT INTO positions(symbol, qty, entry, stop, t1, t2, t3, opened_at, strategy,
                                      bars_held, remaining_frac, targets_hit, realized_pnl)
                VALUES (?,?,?,?,?,?,?,?,?,0,1.0,'',0)
                """,
                (
                    opp.stock,
                    qty,
                    px,
                    opp.stop_loss,
                    opp.target_1,
                    opp.target_2,
                    opp.target_3,
                    now,
                    opp.strategy_selected,
                ),
            )
            cx.execute(
                "INSERT INTO fills(ts, symbol, side, qty, price, pnl, note) VALUES (?,?,?,?,?,?,?)",
                (now, opp.stock, "BUY", qty, px, 0.0, opp.strategy_selected),
            )
        self.mark_to_market({opp.stock: px})
        return True, f"Bought {qty} {opp.stock} @ {px:.2f}"

    def mark_to_market(self, prices: dict[str, float]) -> None:
        acct = self.account()
        positions = self.positions()
        mtm = acct["cash"]
        for p in positions:
            px = prices.get(p.symbol) or prices.get(p.symbol + ".NS") or p.entry
            mtm += px * p.qty * p.remaining_frac
        day_pnl = mtm - acct["day_anchor"]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as cx:
            cx.execute(
                "UPDATE account SET equity=?, day_pnl=?, updated_at=? WHERE id=1",
                (mtm, day_pnl, now),
            )

    def roll_day(self) -> None:
        acct = self.account()
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as cx:
            cx.execute(
                "UPDATE account SET day_anchor=?, day_pnl=0, updated_at=? WHERE id=1",
                (acct["equity"], now),
            )

    def manage_exits(self, daily_bars: dict[str, "pd.DataFrame"]) -> list[str]:  # noqa: F821
        import pandas as pd

        messages = []
        for p in self.positions():
            sym_ns = p.symbol if p.symbol.endswith(".NS") else p.symbol + ".NS"
            df = daily_bars.get(sym_ns) or daily_bars.get(p.symbol)
            if df is None or df.empty:
                continue
            plan = ExitPlan(
                initial_stop=p.stop,
                current_stop=p.stop,
                targets=[p.t1, p.t2, p.t3],
                partial_fractions=[0.4, 0.3, 0.3],
                max_hold_days=30,
                notes="",
            )
            hit = set(int(x) for x in p.targets_hit.split(",") if x != "")
            action = evaluate_exit(
                daily=df,
                entry=p.entry,
                plan=plan,
                bars_held=p.bars_held + 1,
                remaining_fraction=p.remaining_frac,
                targets_hit=hit,
            )
            with self._conn() as cx:
                cx.execute(
                    "UPDATE positions SET bars_held=? WHERE symbol=?",
                    (p.bars_held + 1, p.symbol),
                )
            if action.reason == ExitReason.NONE:
                continue
            if action.reason == ExitReason.TRAIL and action.new_stop is not None:
                new_stop = tighten_stop(p.stop, action.new_stop, p.entry)
                with self._conn() as cx:
                    cx.execute("UPDATE positions SET stop=? WHERE symbol=?", (new_stop, p.symbol))
                messages.append(f"{p.symbol}: trail stop -> {new_stop:.2f}")
                continue
            # Execute exit fraction
            qty = p.qty * action.fraction
            if qty <= 0:
                continue
            px = self.risk.apply_sell_slippage(action.price)
            pnl = (px - p.entry) * qty
            acct = self.account()
            now = datetime.now(timezone.utc).isoformat()
            new_frac = max(0.0, p.remaining_frac - action.fraction)
            with self._conn() as cx:
                cx.execute(
                    "UPDATE account SET cash=?, updated_at=? WHERE id=1",
                    (acct["cash"] + px * qty, now),
                )
                cx.execute(
                    "INSERT INTO fills(ts, symbol, side, qty, price, pnl, note) VALUES (?,?,?,?,?,?,?)",
                    (now, p.symbol, "SELL", qty, px, pnl, action.reason.value),
                )
                if new_frac <= 1e-9:
                    cx.execute("DELETE FROM positions WHERE symbol=?", (p.symbol,))
                else:
                    hits = p.targets_hit
                    if action.reason.value.startswith("TARGET"):
                        idx = int(action.reason.value.split("_")[1]) - 1
                        hits = ",".join([x for x in hits.split(",") if x] + [str(idx)])
                    new_stop = action.new_stop if action.new_stop is not None else p.stop
                    new_stop = tighten_stop(p.stop, new_stop, p.entry)
                    cx.execute(
                        "UPDATE positions SET remaining_frac=?, stop=?, targets_hit=?, realized_pnl=? WHERE symbol=?",
                        (new_frac, new_stop, hits, p.realized_pnl + pnl, p.symbol),
                    )
            messages.append(f"{p.symbol}: {action.reason.value} qty={qty:.2f} px={px:.2f} pnl={pnl:.2f}")
        # refresh equity
        last_prices = {}
        for p in self.positions():
            sym_ns = p.symbol if p.symbol.endswith(".NS") else p.symbol + ".NS"
            df = daily_bars.get(sym_ns) or daily_bars.get(p.symbol)
            if df is not None and not df.empty:
                last_prices[p.symbol] = float(df["Close"].iloc[-1])
        self.mark_to_market(last_prices)
        return messages

    def snapshot(self) -> dict:
        return {
            "account": self.account(),
            "positions": [p.__dict__ for p in self.positions()],
            "risk_state": self.risk_state().__dict__,
        }
