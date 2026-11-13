"""Regime-adaptive multi-strategy book — auto-switches by market situation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ati.config import DEFAULT, RiskConfig
from ati.engine.risk import RiskEngine
from ati.models.opportunity import MarketRegime
from ati.research.backtest import BacktestResult
from ati.research.gates import (
    drawdown_from_peak,
    india_vix_elevated,
    nifty_above_sma,
    sector_neutral_scores,
    universe_breadth,
)
from ati.research.regime_series import book_for_regime, classify_regime_at


@dataclass
class AdaptiveDiagnostics:
    regime_counts: dict[str, int]
    book_counts: dict[str, int]
    switches: int


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _finalize(strategy_id: str, trades: list[dict], curve: list[tuple]) -> BacktestResult:
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
    # stability via thirds of trade returns
    if len(rets) < 6:
        stability = 0.3
    else:
        thirds = np.array_split(rets, 3)
        means = np.array([t.mean() if len(t) else 0.0 for t in thirds])
        if np.allclose(means.std(), 0):
            stability = 0.8
        else:
            same_sign = np.all(means >= 0) or np.all(means <= 0)
            base = 1 / (1 + means.std() / (abs(means.mean()) + 1e-9))
            stability = float(np.clip(base * (1.1 if same_sign else 0.7), 0, 1))

    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min()) if len(eq) else 0.0
    return BacktestResult(
        strategy_id=strategy_id,
        trades=len(trade_log),
        win_rate=float((rets > 0).mean()),
        expectancy=float(rets.mean()),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=dd,
        profit_factor=pf if np.isfinite(pf) else 99.0,
        avg_holding_days=float(trade_log["bars"].mean()),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else 0.0,
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=stability,
    )


def run_mean_reversion_baseline(
    frames: dict[str, pd.DataFrame],
    *,
    risk: RiskConfig | None = None,
    initial_capital: float = 100_000.0,
    lookback: int = 20,
    top_n: int = 3,
    rebalance_every: int = 10,
    rsi_max: float = 35.0,
    trade_start=None,
    strategy_id: str = "mean_reversion_baseline",
) -> BacktestResult:
    """Standalone mean-reversion book for SIDEWAYS regimes (stretch → mean)."""
    risk = risk or DEFAULT.risk
    cost_rt = RiskEngine(risk).round_trip_cost_pct()
    closes = {
        sym: df["Close"].astype(float)
        for sym, df in frames.items()
        if df is not None and len(df) >= lookback + 40
    }
    if len(closes) < top_n + 2:
        return _finalize(strategy_id, [], [])

    panel = pd.DataFrame(closes).sort_index().ffill()
    dates = list(panel.index)
    cash = initial_capital
    holdings: dict[str, float] = {}
    entry_px: dict[str, float] = {}
    entry_i: dict[str, int] = {}
    trades: list[dict] = []
    curve: list[tuple] = []
    last_reb = -10_000
    trade_start_ts = pd.Timestamp(trade_start) if trade_start is not None else None

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
                "book": "mean_reversion",
            }
        )

    for i, dt in enumerate(dates):
        equity = portfolio_value(i)
        curve.append((dt, equity))
        if i < lookback + 5:
            continue
        if trade_start_ts is not None and pd.Timestamp(dt) < trade_start_ts:
            continue
        if i - last_reb < rebalance_every:
            continue

        scores: dict[str, float] = {}
        for sym in panel.columns:
            series = panel[sym].iloc[: i + 1]
            if series.isna().iloc[-1] or len(series) < lookback + 5:
                continue
            rsi = float(_rsi(series, 14).iloc[-1])
            if pd.isna(rsi) or rsi > rsi_max:
                continue
            # Prefer deeper stretches (lower RSI → higher score)
            scores[sym] = 50.0 - rsi

        target = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
        target_set = set(target)
        for sym in list(holdings):
            if sym not in target_set:
                close_position(sym, i, "mr_rebalance" if target else "mr_cash")

        equity = portfolio_value(i)
        if not target:
            last_reb = i
            curve[-1] = (dt, equity)
            continue

        n = len(target)
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
            elif have > desired_sh * 1.05:
                sell_sh = have - desired_sh
                sell = px * (1 - cost_rt / 2)
                cash += sell_sh * sell
                holdings[sym] = desired_sh

        last_reb = i
        curve[-1] = (dt, portfolio_value(i))

    if holdings:
        i = len(dates) - 1
        for sym in list(holdings):
            close_position(sym, i, "eod_flatten")
        curve[-1] = (dates[-1], cash)

    return _finalize(strategy_id, trades, curve)


def run_regime_adaptive_book(
    frames: dict[str, pd.DataFrame],
    nifty: pd.DataFrame | None = None,
    *,
    risk: RiskConfig | None = None,
    initial_capital: float = 100_000.0,
    lookback: int = 84,
    top_n: int = 3,
    bull_rebalance: int = 42,
    sideways_rebalance: int = 10,
    nifty_sma_gate: int = 200,
    sectors: dict[str, str] | None = None,
    sector_neutral: bool = True,
    india_vix: pd.DataFrame | None = None,
    vix_cash_threshold: float | None = 18.0,
    min_breadth: float | None = 0.40,
    max_portfolio_dd: float | None = -0.20,
    regime_persist: int = 5,
    trade_start=None,
    strategy_id: str = "regime_adaptive_multi",
) -> tuple[BacktestResult, AdaptiveDiagnostics]:
    """Auto-switch books by regime: BULL→dual-mom, SIDEWAYS→mean-rev, else→cash.

    Bull-book overlays (NIFTY SMA, breadth); global VIX + DD kill-switch.
    Regime must persist `regime_persist` bars before a book switch (hysteresis).
    """
    risk = risk or DEFAULT.risk
    cost_rt = RiskEngine(risk).round_trip_cost_pct()
    closes = {
        sym: df["Close"].astype(float)
        for sym, df in frames.items()
        if df is not None and len(df) >= lookback + 40
    }
    empty_diag = AdaptiveDiagnostics({}, {}, 0)
    if len(closes) < top_n + 2 or nifty is None or len(nifty) < lookback:
        return _finalize(strategy_id, [], []), empty_diag

    panel = pd.DataFrame(closes).sort_index().ffill()
    nifty_aligned = nifty.reindex(panel.index).ffill()
    nifty_close = nifty_aligned["Close"].astype(float)
    vix_aligned = None
    vix_close = None
    if india_vix is not None and len(india_vix) and vix_cash_threshold is not None:
        vix_aligned = india_vix.reindex(panel.index).ffill()
        vix_close = vix_aligned["Close"].astype(float)

    dates = list(panel.index)
    cash = initial_capital
    holdings: dict[str, float] = {}
    entry_px: dict[str, float] = {}
    entry_i: dict[str, int] = {}
    entry_book: dict[str, str] = {}
    trades: list[dict] = []
    curve: list[tuple] = []
    last_reb = -10_000
    peak_equity = initial_capital
    dd_halt = False
    active_book = "cash"
    switches = 0
    regime_counts: dict[str, int] = {}
    book_counts: dict[str, int] = {}
    pending_book = "cash"
    pending_count = 0
    trade_start_ts = pd.Timestamp(trade_start) if trade_start is not None else None

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
        book = entry_book.pop(sym, "unknown")
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
                "book": book,
            }
        )

    def flatten_all(i: int, reason: str) -> None:
        for sym in list(holdings):
            close_position(sym, i, reason)

    def dual_mom_targets(i: int) -> list[str]:
        scores: dict[str, float] = {}
        for sym in panel.columns:
            px_now = float(panel.iloc[i][sym])
            px_past = float(panel.iloc[i - lookback][sym])
            if px_past <= 0 or pd.isna(px_now) or pd.isna(px_past):
                continue
            abs_mom = px_now / px_past - 1.0
            if abs_mom <= 0:
                continue
            score = abs_mom
            n_now = float(nifty_close.iloc[i])
            n_past = float(nifty_close.iloc[i - lookback])
            if n_past > 0 and not pd.isna(n_now):
                score = abs_mom - (n_now / n_past - 1.0)
            scores[sym] = score
        if sector_neutral and sectors:
            scores = sector_neutral_scores(scores, sectors)
        return [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

    def mean_rev_targets(i: int) -> list[str]:
        """Quality mean-rev: RSI stretch only while price holds SMA100 (no knives)."""
        scores: dict[str, float] = {}
        for sym in panel.columns:
            series = panel[sym].iloc[: i + 1]
            if series.isna().iloc[-1] or len(series) < 120:
                continue
            px = float(series.iloc[-1])
            sma100 = float(series.tail(100).mean())
            if px < sma100:
                continue
            rsi = float(_rsi(series, 14).iloc[-1])
            if pd.isna(rsi) or rsi > 35.0:
                continue
            scores[sym] = 50.0 - rsi
        return [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

    def allocate(i: int, target: list[str], book: str) -> None:
        nonlocal cash
        target_set = set(target)
        for sym in list(holdings):
            if sym not in target_set:
                close_position(sym, i, f"{book}_rebalance" if target else f"{book}_cash")
        if not target:
            return
        equity = portfolio_value(i)
        n = len(target)
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
                    entry_book[sym] = book
            elif have > desired_sh * 1.05:
                sell_sh = have - desired_sh
                sell = px * (1 - cost_rt / 2)
                cash += sell_sh * sell
                holdings[sym] = desired_sh

    for i, dt in enumerate(dates):
        equity = portfolio_value(i)
        curve.append((dt, equity))
        dd, peak_equity = drawdown_from_peak([equity], peak_equity)

        in_trade_window = trade_start_ts is None or pd.Timestamp(dt) >= trade_start_ts
        if max_portfolio_dd is not None and in_trade_window and dd <= max_portfolio_dd and not dd_halt:
            if holdings:
                flatten_all(i, "dd_killswitch")
                curve[-1] = (dt, portfolio_value(i))
                last_reb = i
            dd_halt = True
            peak_equity = portfolio_value(i)
            continue
        if dd_halt and i - last_reb >= bull_rebalance:
            dd_halt = False

        if i < max(lookback, nifty_sma_gate or 0, 60):
            continue
        if trade_start_ts is not None and pd.Timestamp(dt) < trade_start_ts:
            continue

        regime = classify_regime_at(
            nifty_close,
            i=i,
            panel=panel,
            vix_close=vix_close,
            lookback_ret=63,
            breadth_lookback=21,
        )
        regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        nifty_trend_ok = True
        if nifty_sma_gate:
            nifty_trend_ok = nifty_above_sma(
                nifty_aligned.iloc[: i + 1], window=nifty_sma_gate
            )
        desired_book = book_for_regime(regime, nifty_above_trend=nifty_trend_ok)

        # Book-aware overlays: SMA200/breadth gate bull book; VIX/DD are global.
        # SIDEWAYS mean-rev already requires nifty_above_trend inside book_for_regime.
        cash_reason = "overlay_cash"
        if desired_book == "dual_momentum":
            if not nifty_trend_ok:
                desired_book = "cash"
                cash_reason = "nifty_gate_cash"
            elif min_breadth is not None and universe_breadth(panel, i, lookback) < min_breadth:
                desired_book = "cash"
                cash_reason = "breadth_gate_cash"
        if desired_book != "cash" and vix_aligned is not None and vix_cash_threshold is not None:
            if india_vix_elevated(vix_aligned.iloc[: i + 1], threshold=vix_cash_threshold):
                desired_book = "cash"
                cash_reason = "vix_gate_cash"
        if desired_book != "cash" and dd_halt:
            desired_book = "cash"
            cash_reason = "dd_halt_cash"
        risk_on = desired_book != "cash"

        # Hysteresis: require persistent regime signal before switching books
        if desired_book == pending_book:
            pending_count += 1
        else:
            pending_book = desired_book
            pending_count = 1
        target_book = active_book
        if pending_book != active_book and pending_count >= regime_persist:
            target_book = pending_book

        # Book-specific rebalance cadence
        cadence = sideways_rebalance if target_book == "mean_reversion" else bull_rebalance
        if target_book == "cash":
            cadence = min(bull_rebalance, sideways_rebalance)
        book_switch = target_book != active_book
        if not book_switch and i - last_reb < cadence:
            continue

        if book_switch:
            switches += 1
            if holdings:
                flatten_all(i, f"switch_{active_book}_to_{target_book}")
            active_book = target_book

        book_counts[active_book] = book_counts.get(active_book, 0) + 1

        if active_book == "cash":
            flatten_all(i, cash_reason if not risk_on else "regime_cash")
            last_reb = i
            curve[-1] = (dt, portfolio_value(i))
            continue

        if active_book == "dual_momentum":
            allocate(i, dual_mom_targets(i), "dual_momentum")
        elif active_book == "mean_reversion":
            allocate(i, mean_rev_targets(i), "mean_reversion")
        else:
            flatten_all(i, "unknown_book_cash")

        last_reb = i
        curve[-1] = (dt, portfolio_value(i))

    if holdings:
        i = len(dates) - 1
        flatten_all(i, "eod_flatten")
        curve[-1] = (dates[-1], cash)

    result = _finalize(strategy_id, trades, curve)
    diag = AdaptiveDiagnostics(regime_counts=regime_counts, book_counts=book_counts, switches=switches)
    return result, diag
