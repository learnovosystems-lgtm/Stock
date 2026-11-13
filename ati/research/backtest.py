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


def run_dual_momentum_baseline(
    frames: dict[str, pd.DataFrame],
    nifty: pd.DataFrame | None = None,
    *,
    risk: RiskConfig | None = None,
    initial_capital: float = 100_000.0,
    lookback: int = 84,
    top_n: int = 3,
    rebalance_every: int = 42,
    abs_mom_floor: float = 0.0,
    nifty_sma_gate: int = 200,
    sectors: dict[str, str] | None = None,
    sector_neutral: bool = True,
    strategy_id: str = "dual_momentum_baseline",
) -> BacktestResult:
    """Low-turnover dual momentum with optional NIFTY SMA cash gate + sector neutrality.

    Defaults (84/42/3) from Phase-1 grid. Phase-2 adds SMA200 risk-on gate and
    sector-median residual ranking to cut concentrated sector bets.
    """
    from ati.research.gates import nifty_above_sma, sector_neutral_scores

    risk = risk or DEFAULT.risk
    cost_rt = RiskEngine(risk).round_trip_cost_pct()

    closes = {
        sym: df["Close"].astype(float)
        for sym, df in frames.items()
        if df is not None and len(df) >= lookback + 40
    }
    empty = BacktestResult(
        strategy_id=strategy_id,
        trades=0,
        win_rate=0.0,
        expectancy=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        avg_holding_days=0.0,
        total_return=0.0,
        equity_curve=pd.Series(dtype=float),
        trade_log=pd.DataFrame(),
        stability_score=0.0,
    )
    if len(closes) < top_n + 2:
        return empty

    panel = pd.DataFrame(closes).sort_index().ffill()
    nifty_aligned = None
    if nifty is not None and len(nifty) > lookback:
        nifty_aligned = nifty.reindex(panel.index).ffill()
        nifty_close = nifty_aligned["Close"].astype(float)
    else:
        nifty_close = None

    dates = list(panel.index)
    cash = initial_capital
    holdings: dict[str, float] = {}
    entry_px: dict[str, float] = {}
    entry_i: dict[str, int] = {}
    trades: list[dict] = []
    curve: list[tuple] = []
    last_reb = -10_000

    def portfolio_value(i: int) -> float:
        value = cash
        for sym, sh in holdings.items():
            value += sh * float(panel.iloc[i][sym])
        return value

    def close_position(sym: str, i: int, reason: str) -> None:
        nonlocal cash
        sh = holdings.pop(sym)
        px = float(panel.iloc[i][sym])
        sell = px * (1 - cost_rt / 2)
        cash += sh * sell
        e_px = entry_px.pop(sym)
        e_i = entry_i.pop(sym)
        buy = e_px * (1 + cost_rt / 2)
        trades.append(
            {
                "symbol": sym,
                "entry": buy,
                "exit": sell,
                "pnl": sh * (sell - buy),
                "ret": sell / buy - 1.0,
                "bars": max(1, i - e_i),
                "reason": reason,
            }
        )

    for i, dt in enumerate(dates):
        equity = portfolio_value(i)
        curve.append((dt, equity))
        if i < max(lookback, nifty_sma_gate or 0):
            continue
        if i - last_reb < rebalance_every:
            continue

        # Hard cash gate: if NIFTY below SMA, flatten and stay in cash
        risk_on = True
        if nifty_sma_gate and nifty_aligned is not None:
            risk_on = nifty_above_sma(nifty_aligned.iloc[: i + 1], window=nifty_sma_gate)

        if not risk_on:
            for sym in list(holdings):
                close_position(sym, i, "nifty_gate_cash")
            last_reb = i
            curve[-1] = (dt, portfolio_value(i))
            continue

        scores: dict[str, float] = {}
        for sym in panel.columns:
            px_now = float(panel.iloc[i][sym])
            px_past = float(panel.iloc[i - lookback][sym])
            if px_past <= 0 or pd.isna(px_now) or pd.isna(px_past):
                continue
            abs_mom = px_now / px_past - 1.0
            if abs_mom <= abs_mom_floor:
                continue
            score = abs_mom
            if nifty_close is not None:
                n_now = float(nifty_close.iloc[i])
                n_past = float(nifty_close.iloc[i - lookback])
                if n_past > 0 and not pd.isna(n_now):
                    score = abs_mom - (n_now / n_past - 1.0)
            scores[sym] = score

        if sector_neutral and sectors:
            scores = sector_neutral_scores(scores, sectors)

        target = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
        target_set = set(target)

        for sym in list(holdings):
            if sym not in target_set:
                close_position(sym, i, "rebalance" if target else "cash")

        equity = portfolio_value(i)
        if not target:
            last_reb = i
            curve[-1] = (dt, equity)
            continue

        n = len(target)
        target_value = equity / n
        for sym in list(holdings):
            if sym not in target_set:
                continue
            px = float(panel.iloc[i][sym])
            cur = holdings[sym] * px
            if cur > target_value * 1.05:
                sell_sh = (cur - target_value) / px
                sell = px * (1 - cost_rt / 2)
                cash += sell_sh * sell
                holdings[sym] -= sell_sh

        equity = portfolio_value(i)
        target_value = equity / n
        for sym in target:
            px = float(panel.iloc[i][sym])
            buy = px * (1 + cost_rt / 2)
            desired_sh = target_value / buy
            have = holdings.get(sym, 0.0)
            if desired_sh > have + 1e-9:
                need = desired_sh - have
                cost = need * buy
                if cost > cash:
                    need = cash / buy
                    cost = need * buy
                if need <= 0:
                    continue
                cash -= cost
                holdings[sym] = have + need
                if sym not in entry_px:
                    entry_px[sym] = px
                    entry_i[sym] = i

        last_reb = i
        curve[-1] = (dt, portfolio_value(i))

    if holdings:
        i = len(dates) - 1
        for sym in list(holdings):
            close_position(sym, i, "eod_flatten")
        curve[-1] = (dates[-1], cash)

    trade_log = pd.DataFrame(trades)
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    if trade_log.empty or eq.empty:
        return empty

    rets = trade_log["ret"].values
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else 99.0
    return BacktestResult(
        strategy_id=strategy_id,
        trades=len(trade_log),
        win_rate=float((rets > 0).mean()),
        expectancy=float(rets.mean()),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_max_drawdown(eq),
        profit_factor=pf if np.isfinite(pf) else 99.0,
        avg_holding_days=float(trade_log["bars"].mean()),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else 0.0,
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=_stability(rets),
    )


def run_nifty_buyhold_benchmark(
    nifty: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Passive NIFTY total-path benchmark (adjusted close proxy)."""
    if nifty is None or len(nifty) < 50:
        return BacktestResult(
            strategy_id="nifty_buyhold_benchmark",
            trades=0,
            win_rate=0.0,
            expectancy=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            avg_holding_days=0.0,
            total_return=0.0,
            equity_curve=pd.Series(dtype=float),
            trade_log=pd.DataFrame(),
            stability_score=0.0,
        )
    close = nifty["Close"].astype(float).dropna()
    eq = initial_capital * (close / float(close.iloc[0]))
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std() * np.sqrt(252)) if len(downside) and downside.std() > 0 else 0.0
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    trade_log = pd.DataFrame(
        [
            {
                "symbol": "^NSEI",
                "entry": float(close.iloc[0]),
                "exit": float(close.iloc[-1]),
                "pnl": float(eq.iloc[-1] - initial_capital),
                "ret": ret,
                "bars": len(close),
                "reason": "benchmark",
            }
        ]
    )
    return BacktestResult(
        strategy_id="nifty_buyhold_benchmark",
        trades=1,
        win_rate=1.0 if ret > 0 else 0.0,
        expectancy=ret,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=_max_drawdown(eq),
        profit_factor=99.0 if ret > 0 else 0.0,
        avg_holding_days=float(len(close)),
        total_return=ret,
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=0.8 if ret > 0 else 0.2,
    )
