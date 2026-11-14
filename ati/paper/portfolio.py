"""Paper-trading portfolio with ChargeEngine net costs and hard risk gates."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ati.config import STATE_DIR, DEFAULT, RiskConfig, ensure_dirs
from ati.engine.charges import ChargeEngine, default_charge_engine
from ati.engine.exit import ExitPlan, ExitReason, build_exit_plan, evaluate_exit, tighten_stop
from ati.engine.risk import PortfolioRiskState, RiskEngine
from ati.models.charges import Segment, Side
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
    entry_fees: float = 0.0
    realized_fees: float = 0.0


class PaperBroker:
    def __init__(
        self,
        db_path: Path | None = None,
        risk: RiskConfig | None = None,
        *,
        allowed_strategy_substrings: tuple[str, ...] | None = None,
        symbol_sectors: dict[str, str] | None = None,
        charges: ChargeEngine | None = None,
        segment: Segment | str | None = None,
    ):
        ensure_dirs()
        self.db_path = db_path or (STATE_DIR / "paper.db")
        self.risk_cfg = risk or DEFAULT.risk
        self.charges = charges or default_charge_engine(slippage_pct=self.risk_cfg.slippage_pct)
        self.risk = RiskEngine(self.risk_cfg, charges=self.charges)
        self.segment = Segment(segment or DEFAULT.default_segment)
        self.allowed_strategy_substrings = (
            allowed_strategy_substrings
            if allowed_strategy_substrings is not None
            else DEFAULT.paper_allowed_strategy_substrings
        )
        self.symbol_sectors = symbol_sectors or {}
        self._position_sectors: dict[str, str] = {}
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
            self._migrate(cx)
            cur = cx.execute("SELECT COUNT(*) FROM account")
            if cur.fetchone()[0] == 0:
                now = datetime.now(timezone.utc).isoformat()
                cap = self.risk_cfg.starting_capital
                cx.execute(
                    "INSERT INTO account(id, cash, equity, day_pnl, day_anchor, updated_at) VALUES (1,?,?,?,?,?)",
                    (cap, cap, 0.0, cap, now),
                )

    def _migrate(self, cx: sqlite3.Connection) -> None:
        pos_cols = {r[1] for r in cx.execute("PRAGMA table_info(positions)").fetchall()}
        if "entry_fees" not in pos_cols:
            cx.execute("ALTER TABLE positions ADD COLUMN entry_fees REAL DEFAULT 0")
        if "realized_fees" not in pos_cols:
            cx.execute("ALTER TABLE positions ADD COLUMN realized_fees REAL DEFAULT 0")
        fill_cols = {r[1] for r in cx.execute("PRAGMA table_info(fills)").fetchall()}
        for col, decl in (
            ("fees", "REAL DEFAULT 0"),
            ("gross_pnl", "REAL DEFAULT 0"),
            ("net_pnl", "REAL DEFAULT 0"),
            ("raw_price", "REAL"),
            ("schedule_id", "TEXT"),
            ("segment", "TEXT"),
            ("charges_json", "TEXT"),
        ):
            if col not in fill_cols:
                cx.execute(f"ALTER TABLE fills ADD COLUMN {col} {decl}")

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
            "charges_broker": self.charges.broker,
            "charges_schedule": self.charges.schedule.schedule_id,
        }

    def positions(self) -> list[Position]:
        with self._conn() as cx:
            cur = cx.execute("SELECT * FROM positions")
            colnames = [d[0] for d in cur.description]
            rows = cur.fetchall()
        out = []
        for r in rows:
            data = dict(zip(colnames, r))
            out.append(
                Position(
                    symbol=data["symbol"],
                    qty=int(data["qty"]),
                    entry=float(data["entry"]),
                    stop=float(data["stop"]),
                    t1=float(data["t1"]),
                    t2=float(data["t2"]),
                    t3=float(data["t3"]),
                    opened_at=data["opened_at"],
                    strategy=data["strategy"],
                    bars_held=int(data.get("bars_held") or 0),
                    remaining_frac=float(data.get("remaining_frac") or 1.0),
                    targets_hit=data.get("targets_hit") or "",
                    realized_pnl=float(data.get("realized_pnl") or 0.0),
                    entry_fees=float(data.get("entry_fees") or 0.0),
                    realized_fees=float(data.get("realized_fees") or 0.0),
                )
            )
        return out

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

    def _strategy_allowed(self, strategy_selected: str) -> bool:
        if not self.allowed_strategy_substrings:
            return True
        s = strategy_selected.lower()
        return any(frag.lower() in s for frag in self.allowed_strategy_substrings)

    def _sector_for(self, stock: str, fallback: str | None = None) -> str:
        for key in (stock, stock + ".NS", stock.replace(".NS", "")):
            if key in self.symbol_sectors:
                return self.symbol_sectors[key]
            if key in self._position_sectors:
                return self._position_sectors[key]
        return fallback or "UNKNOWN"

    def try_enter(self, opp: Opportunity) -> tuple[bool, str]:
        if opp.signal.value != "BUY" or opp.position_quantity < 1:
            return False, "Signal not actionable BUY"
        if not self._strategy_allowed(opp.strategy_selected):
            return False, f"Strategy not allowed for paper: {opp.strategy_selected}"
        state = self.risk_state()
        if any(p.symbol == opp.stock for p in self.positions()):
            return False, "Already in position"

        sector = opp.sector or self._sector_for(opp.stock)
        open_sectors = [self._sector_for(p.symbol) for p in self.positions()]
        if open_sectors.count(sector) >= self.risk_cfg.max_positions_per_sector:
            return False, f"Max positions for sector {sector} reached"

        stop_dist_pct = abs(opp.ideal_entry - opp.stop_loss) / max(opp.ideal_entry, 1e-9)
        if stop_dist_pct < 0.01:
            return False, "Stop too tight (<1%); skipped"

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
        eco = self.charges.economics_for_fill(
            side=Side.BUY,
            qty=qty,
            raw_price=opp.ideal_entry,
            segment=self.segment,
        )
        cash_needed = -eco.cash_delta
        acct = self.account()
        if cash_needed > acct["cash"]:
            return False, "Insufficient cash"
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as cx:
            cx.execute(
                "UPDATE account SET cash=?, updated_at=? WHERE id=1",
                (acct["cash"] + eco.cash_delta, now),
            )
            cx.execute(
                """
                INSERT INTO positions(symbol, qty, entry, stop, t1, t2, t3, opened_at, strategy,
                                      bars_held, remaining_frac, targets_hit, realized_pnl,
                                      entry_fees, realized_fees)
                VALUES (?,?,?,?,?,?,?,?,?,0,1.0,'',0,?,0)
                """,
                (
                    opp.stock,
                    qty,
                    eco.exec_price,
                    opp.stop_loss,
                    opp.target_1,
                    opp.target_2,
                    opp.target_3,
                    now,
                    opp.strategy_selected,
                    eco.total_fees,
                ),
            )
            cx.execute(
                """
                INSERT INTO fills(ts, symbol, side, qty, price, pnl, note, fees, gross_pnl, net_pnl,
                                  raw_price, schedule_id, segment, charges_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now,
                    opp.stock,
                    "BUY",
                    qty,
                    eco.exec_price,
                    0.0,
                    opp.strategy_selected,
                    eco.total_fees,
                    0.0,
                    0.0,
                    eco.raw_price,
                    eco.charges.schedule_id,
                    eco.charges.segment,
                    json.dumps(eco.charges.to_dict()),
                ),
            )
        self._position_sectors[opp.stock] = sector
        self.mark_to_market({opp.stock: eco.exec_price})
        return True, (
            f"Bought {qty} {opp.stock} @ {eco.exec_price:.2f} "
            f"fees={eco.total_fees:.2f} [{opp.strategy_selected}]"
        )

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
            qty = p.qty * action.fraction
            if qty <= 0:
                continue
            # DP once per delivery sell order (each partial exit is a separate executed order)
            eco = self.charges.economics_for_fill(
                side=Side.SELL,
                qty=qty,
                raw_price=action.price,
                segment=self.segment,
                apply_dp=(self.segment == Segment.EQUITY_DELIVERY),
            )
            gross = (eco.exec_price - p.entry) * qty
            entry_fee_alloc = p.entry_fees * (qty / max(p.qty, 1e-9))
            net = gross - eco.total_fees - entry_fee_alloc
            acct = self.account()
            now = datetime.now(timezone.utc).isoformat()
            new_frac = max(0.0, p.remaining_frac - action.fraction)
            with self._conn() as cx:
                cx.execute(
                    "UPDATE account SET cash=?, updated_at=? WHERE id=1",
                    (acct["cash"] + eco.cash_delta, now),
                )
                cx.execute(
                    """
                    INSERT INTO fills(ts, symbol, side, qty, price, pnl, note, fees, gross_pnl, net_pnl,
                                      raw_price, schedule_id, segment, charges_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        now,
                        p.symbol,
                        "SELL",
                        qty,
                        eco.exec_price,
                        net,
                        action.reason.value,
                        eco.total_fees,
                        gross,
                        net,
                        eco.raw_price,
                        eco.charges.schedule_id,
                        eco.charges.segment,
                        json.dumps(eco.charges.to_dict()),
                    ),
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
                    remaining_entry_fees = max(0.0, p.entry_fees - entry_fee_alloc)
                    cx.execute(
                        """
                        UPDATE positions SET remaining_frac=?, stop=?, targets_hit=?,
                               realized_pnl=?, entry_fees=?, realized_fees=? WHERE symbol=?
                        """,
                        (
                            new_frac,
                            new_stop,
                            hits,
                            p.realized_pnl + net,
                            remaining_entry_fees,
                            p.realized_fees + eco.total_fees,
                            p.symbol,
                        ),
                    )
            messages.append(
                f"{p.symbol}: {action.reason.value} qty={qty:.2f} px={eco.exec_price:.2f} "
                f"gross={gross:.2f} fees={eco.total_fees:.2f} net={net:.2f}"
            )
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
            "charges": self.charges.schedule_info(),
        }
