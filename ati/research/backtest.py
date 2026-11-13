"""Event-driven swing backtest with Indian cost model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ati.config import RiskConfig, DEFAULT
from ati.engine.risk import RiskEngine
from ati.indicators.technical import enrich


@dataclass
class BacktestResult:
    strategy_id: str
    trades: int
    win_rate: float
    expectancy: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    avg_holding_days: float
    total_return: float
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    stability_score: float


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _stability(trade_returns: np.ndarray) -> float:
    if len(trade_returns) < 6:
        return 0.3
    # Split into thirds — penalize sign flips / large variance of means
    thirds = np.array_split(trade_returns, 3)
    means = np.array([t.mean() if len(t) else 0.0 for t in thirds])
    if np.allclose(means.std(), 0):
        return 0.8
    same_sign = np.all(means >= 0) or np.all(means <= 0)
    base = 1 / (1 + means.std() / (abs(means.mean()) + 1e-9))
    return float(np.clip(base * (1.1 if same_sign else 0.7), 0, 1))


def run_breakout_momentum_baseline(
    frames: dict[str, pd.DataFrame],
    *,
    risk: RiskConfig | None = None,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Strong baseline: volume-confirmed Donchian breakout + ATR stop, 1:2 R:R.

    Used as Phase-0 measurement before adaptive selection.
    """
    risk = risk or DEFAULT.risk
    engine = RiskEngine(risk)
    cash = initial_capital
    equity = initial_capital
    open_pos: dict[str, dict] = {}
    curve = []
    trades = []

    # Align on calendar union
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values() if len(df)]))
    enriched = {s: enrich(df) for s, df in frames.items() if len(df) >= 80}

    for dt in all_dates:
        # Mark-to-market + exits
        to_close = []
        for sym, pos in open_pos.items():
            df = enriched.get(sym)
            if df is None or dt not in df.index:
                continue
            bar = df.loc[dt]
            pos["bars"] += 1
            # stop / target
            if bar["Low"] <= pos["stop"]:
                px = engine.apply_sell_slippage(pos["stop"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "stop",
                    }
                )
                to_close.append(sym)
            elif bar["High"] >= pos["target"]:
                px = engine.apply_sell_slippage(pos["target"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "target",
                    }
                )
                to_close.append(sym)
            elif pos["bars"] >= 20:
                px = engine.apply_sell_slippage(float(bar["Close"]))
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "time",
                    }
                )
                to_close.append(sym)
        for sym in to_close:
            open_pos.pop(sym, None)

        # Entries
        if len(open_pos) < risk.max_positions:
            for sym, df in enriched.items():
                if sym in open_pos or dt not in df.index:
                    continue
                i = df.index.get_loc(dt)
                if isinstance(i, slice) or not isinstance(i, (int, np.integer)):
                    continue
                if i < 40:
                    continue
                row = df.iloc[i]
                prev = df.iloc[i - 1]
                if any(pd.isna(row[c]) for c in ("dc_high", "atr_14", "vol_surge", "sma_50")):
                    continue
                # Measured improvement over naive 1.3x breakout chase:
                # require stronger volume, established trend (SMA50>SMA200),
                # and avoid RSI overextension at breakout.
                if pd.isna(row.get("sma_200")) or pd.isna(row.get("rsi_14")):
                    continue
                if not (
                    row["Close"] > prev["dc_high"]
                    and row["vol_surge"] >= 1.5
                    and row["Close"] > row["sma_50"] > row["sma_200"]
                    and row["rsi_14"] < 72
                ):
                    continue
                entry = engine.apply_buy_slippage(float(row["Close"]))
                stop = float(row["Close"] - 2.2 * row["atr_14"])
                # Prefer structural stop under recent swing if tighter risk still >= 0.5%
                swing = float(df["Low"].iloc[max(0, i - 10) : i + 1].min())
                if swing < entry and (entry - swing) / entry <= 0.08:
                    stop = max(stop, swing)
                if stop >= entry:
                    continue
                risk_amt = equity * risk.risk_per_trade_pct
                qty = int(risk_amt // (entry - stop))
                if qty < 1 or qty * entry > cash:
                    continue
                target = entry + 2.5 * (entry - stop)
                cash -= entry * qty
                open_pos[sym] = {"entry": entry, "stop": stop, "target": target, "qty": qty, "bars": 0}
                if len(open_pos) >= risk.max_positions:
                    break

        mtm = cash
        for sym, pos in open_pos.items():
            df = enriched.get(sym)
            if df is not None and dt in df.index:
                mtm += float(df.loc[dt, "Close"]) * pos["qty"]
            else:
                mtm += pos["entry"] * pos["qty"]
        equity = mtm
        curve.append((dt, equity))

    # Force flatten
    if open_pos and curve:
        last_dt = curve[-1][0]
        for sym, pos in list(open_pos.items()):
            df = enriched.get(sym)
            px = float(df.loc[last_dt, "Close"]) if df is not None and last_dt in df.index else pos["entry"]
            px = engine.apply_sell_slippage(px)
            pnl = (px - pos["entry"]) * pos["qty"]
            trades.append(
                {
                    "symbol": sym,
                    "entry": pos["entry"],
                    "exit": px,
                    "pnl": pnl,
                    "ret": pnl / (pos["entry"] * pos["qty"]),
                    "bars": pos["bars"],
                    "reason": "eod_flatten",
                }
            )

    trade_log = pd.DataFrame(trades)
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    if trade_log.empty:
        return BacktestResult(
            strategy_id="breakout_momentum_baseline",
            trades=0,
            win_rate=0.0,
            expectancy=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            avg_holding_days=0.0,
            total_return=0.0,
            equity_curve=eq,
            trade_log=trade_log,
            stability_score=0.0,
        )

    rets = trade_log["ret"].values
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    win_rate = float((rets > 0).mean())
    expectancy = float(rets.mean())
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else float("inf")
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else 0.0

    return BacktestResult(
        strategy_id="breakout_momentum_baseline",
        trades=len(trade_log),
        win_rate=win_rate,
        expectancy=expectancy,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_max_drawdown(eq),
        profit_factor=pf if np.isfinite(pf) else 99.0,
        avg_holding_days=float(trade_log["bars"].mean()),
        total_return=total_return,
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=_stability(rets),
    )


def run_pullback_trend_baseline(
    frames: dict[str, pd.DataFrame],
    *,
    risk: RiskConfig | None = None,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Alternative baseline: buy pullbacks to rising SMA20 in SMA50>SMA200 trends."""
    risk = risk or DEFAULT.risk
    engine = RiskEngine(risk)
    cash = initial_capital
    equity = initial_capital
    open_pos: dict[str, dict] = {}
    curve = []
    trades = []
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values() if len(df)]))
    enriched = {s: enrich(df) for s, df in frames.items() if len(df) >= 120}

    for dt in all_dates:
        to_close = []
        for sym, pos in open_pos.items():
            df = enriched.get(sym)
            if df is None or dt not in df.index:
                continue
            bar = df.loc[dt]
            pos["bars"] += 1
            if bar["Low"] <= pos["stop"]:
                px = engine.apply_sell_slippage(pos["stop"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "stop",
                    }
                )
                to_close.append(sym)
            elif bar["High"] >= pos["target"]:
                px = engine.apply_sell_slippage(pos["target"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "target",
                    }
                )
                to_close.append(sym)
            elif pos["bars"] >= 25:
                px = engine.apply_sell_slippage(float(bar["Close"]))
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append(
                    {
                        "symbol": sym,
                        "entry": pos["entry"],
                        "exit": px,
                        "pnl": pnl,
                        "ret": pnl / (pos["entry"] * pos["qty"]),
                        "bars": pos["bars"],
                        "reason": "time",
                    }
                )
                to_close.append(sym)
        for sym in to_close:
            open_pos.pop(sym, None)

        if len(open_pos) < risk.max_positions:
            for sym, df in enriched.items():
                if sym in open_pos or dt not in df.index:
                    continue
                i = df.index.get_loc(dt)
                if not isinstance(i, (int, np.integer)) or i < 60:
                    continue
                row = df.iloc[i]
                if any(pd.isna(row[c]) for c in ("sma_20", "sma_50", "sma_200", "atr_14", "rsi_14")):
                    continue
                if not (
                    row["sma_50"] > row["sma_200"]
                    and row["Close"] > row["sma_50"]
                    and row["Close"] <= row["sma_20"] * 1.01
                    and 35 <= row["rsi_14"] <= 55
                ):
                    continue
                entry = engine.apply_buy_slippage(float(row["Close"]))
                stop = float(min(df["Low"].iloc[i - 10 : i + 1].min(), entry - 2.0 * row["atr_14"]))
                if stop >= entry:
                    continue
                qty = int((equity * risk.risk_per_trade_pct) // (entry - stop))
                if qty < 1 or qty * entry > cash:
                    continue
                target = entry + 2.5 * (entry - stop)
                cash -= entry * qty
                open_pos[sym] = {"entry": entry, "stop": stop, "target": target, "qty": qty, "bars": 0}
                if len(open_pos) >= risk.max_positions:
                    break

        mtm = cash
        for sym, pos in open_pos.items():
            df = enriched.get(sym)
            if df is not None and dt in df.index:
                mtm += float(df.loc[dt, "Close"]) * pos["qty"]
            else:
                mtm += pos["entry"] * pos["qty"]
        equity = mtm
        curve.append((dt, equity))

    trade_log = pd.DataFrame(trades)
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    if trade_log.empty:
        return BacktestResult(
            strategy_id="pullback_trend_baseline",
            trades=0,
            win_rate=0.0,
            expectancy=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            avg_holding_days=0.0,
            total_return=0.0,
            equity_curve=eq,
            trade_log=trade_log,
            stability_score=0.0,
        )
    rets = trade_log["ret"].values
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 99.0
    return BacktestResult(
        strategy_id="pullback_trend_baseline",
        trades=len(trade_log),
        win_rate=float((rets > 0).mean()),
        expectancy=float(rets.mean()),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_max_drawdown(eq),
        profit_factor=pf,
        avg_holding_days=float(trade_log["bars"].mean()),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else 0.0,
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=_stability(rets),
    )
