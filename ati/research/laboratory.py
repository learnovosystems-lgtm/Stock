"""Autonomous strategy research laboratory with walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from ati.engine.risk import RiskEngine
from ati.indicators.technical import enrich
from ati.models.opportunity import StrategyMetrics
from ati.research.backtest import BacktestResult, _max_drawdown, _stability, run_dual_momentum_baseline
from ati.research.registry import StrategyRegistry


@dataclass
class ExperimentResult:
    name: str
    params: dict
    metrics: StrategyMetrics
    promoted_candidate: bool


def _run_paramized_breakout(
    frames: dict[str, pd.DataFrame],
    *,
    donchian: int,
    vol_surge: float,
    atr_mult: float,
    rr: float,
    max_hold: int,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    risk = RiskEngine().config
    cash = initial_capital
    equity = initial_capital
    open_pos: dict[str, dict] = {}
    curve = []
    trades = []
    enriched = {}
    for s, df in frames.items():
        if len(df) < 80:
            continue
        e = enrich(df)
        e["dc_high_n"] = e["High"].rolling(donchian).max()
        enriched[s] = e
    all_dates = sorted(set().union(*[set(df.index) for df in enriched.values()])) if enriched else []

    for dt in all_dates:
        to_close = []
        for sym, pos in open_pos.items():
            df = enriched[sym]
            if dt not in df.index:
                continue
            bar = df.loc[dt]
            pos["bars"] += 1
            if bar["Low"] <= pos["stop"]:
                px = float(pos["stop"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append({"ret": pnl / (pos["entry"] * pos["qty"]), "bars": pos["bars"], "pnl": pnl})
                to_close.append(sym)
            elif bar["High"] >= pos["target"]:
                px = float(pos["target"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append({"ret": pnl / (pos["entry"] * pos["qty"]), "bars": pos["bars"], "pnl": pnl})
                to_close.append(sym)
            elif pos["bars"] >= max_hold:
                px = float(bar["Close"])
                pnl = (px - pos["entry"]) * pos["qty"]
                cash += px * pos["qty"]
                trades.append({"ret": pnl / (pos["entry"] * pos["qty"]), "bars": pos["bars"], "pnl": pnl})
                to_close.append(sym)
        for sym in to_close:
            open_pos.pop(sym, None)

        if len(open_pos) < risk.max_positions:
            for sym, df in enriched.items():
                if sym in open_pos or dt not in df.index:
                    continue
                i = df.index.get_loc(dt)
                if not isinstance(i, (int, np.integer)) or i < donchian + 5:
                    continue
                row = df.iloc[i]
                prev = df.iloc[i - 1]
                if pd.isna(row.get("dc_high_n")) or pd.isna(row.get("atr_14")):
                    continue
                if row["Close"] > prev["dc_high_n"] and row["vol_surge"] >= vol_surge and row["Close"] > row["sma_50"]:
                    entry = float(row["Close"])
                    stop = entry - atr_mult * float(row["atr_14"])
                    if stop >= entry:
                        continue
                    qty = int((equity * risk.risk_per_trade_pct) // (entry - stop))
                    if qty < 1 or qty * entry > cash:
                        continue
                    target = entry + rr * (entry - stop)
                    cash -= entry * qty
                    open_pos[sym] = {"entry": entry, "stop": stop, "target": target, "qty": qty, "bars": 0}
                    if len(open_pos) >= risk.max_positions:
                        break

        mtm = cash + sum(
            float(enriched[s].loc[dt, "Close"]) * p["qty"]
            if dt in enriched[s].index
            else p["entry"] * p["qty"]
            for s, p in open_pos.items()
        )
        equity = mtm
        curve.append((dt, equity))

    trade_log = pd.DataFrame(trades)
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    if trade_log.empty or eq.empty:
        return BacktestResult(
            strategy_id="param_breakout",
            trades=0,
            win_rate=0,
            expectancy=0,
            sharpe=0,
            sortino=0,
            max_drawdown=0,
            profit_factor=0,
            avg_holding_days=0,
            total_return=0,
            equity_curve=eq,
            trade_log=trade_log,
            stability_score=0,
        )
    rets = trade_log["ret"].values
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    wins = rets[rets > 0].sum()
    losses = rets[rets <= 0].sum()
    return BacktestResult(
        strategy_id="param_breakout",
        trades=len(trade_log),
        win_rate=float((rets > 0).mean()),
        expectancy=float(rets.mean()),
        sharpe=sharpe,
        sortino=0.0,
        max_drawdown=_max_drawdown(eq),
        profit_factor=float(wins / abs(losses)) if losses != 0 else 99.0,
        avg_holding_days=float(trade_log["bars"].mean()),
        total_return=float(eq.iloc[-1] / eq.iloc[0] - 1),
        equity_curve=eq,
        trade_log=trade_log,
        stability_score=_stability(rets),
    )


class ResearchLaboratory:
    def __init__(self, registry: StrategyRegistry | None = None):
        self.registry = registry or StrategyRegistry()

    def walk_forward_split(self, frames: dict[str, pd.DataFrame], train_frac: float = 0.7):
        # Split each frame by time
        train, test = {}, {}
        for s, df in frames.items():
            if len(df) < 100:
                continue
            cut = int(len(df) * train_frac)
            train[s] = df.iloc[:cut]
            test[s] = df.iloc[cut:]
        return train, test

    def search_breakout_params(self, frames: dict[str, pd.DataFrame]) -> list[ExperimentResult]:
        train, test = self.walk_forward_split(frames)
        grid = {
            "donchian": [15, 20, 25],
            "vol_surge": [1.2, 1.4],
            "atr_mult": [1.8, 2.2],
            "rr": [2.0, 2.5],
            "max_hold": [15, 25],
        }
        keys = list(grid.keys())
        results: list[ExperimentResult] = []
        for values in product(*[grid[k] for k in keys]):
            params = dict(zip(keys, values))
            # Fit selection on train (measure), evaluate OOS on test
            train_bt = _run_paramized_breakout(train, **params)
            test_bt = _run_paramized_breakout(test, **params)
            metrics = StrategyMetrics(
                strategy_id="breakout_param_search",
                version=f"d{params['donchian']}_v{params['vol_surge']}_a{params['atr_mult']}_r{params['rr']}_h{params['max_hold']}",
                sample_count=test_bt.trades,
                win_rate=test_bt.win_rate,
                expectancy=test_bt.expectancy,
                sharpe=test_bt.sharpe,
                sortino=test_bt.sortino,
                max_drawdown=test_bt.max_drawdown,
                profit_factor=test_bt.profit_factor,
                avg_holding_days=test_bt.avg_holding_days,
                stability_score=test_bt.stability_score,
                total_return=test_bt.total_return,
                oos=True,
            )
            # Candidate if OOS sharpe>0, stability ok, not just best train
            promoted = (
                test_bt.trades >= 8
                and test_bt.sharpe > 0
                and test_bt.stability_score >= 0.4
                and test_bt.max_drawdown > -0.25
                and train_bt.sharpe > -0.2
            )
            self.registry.record_experiment(
                name="breakout_param_search",
                params=params,
                result=metrics.model_dump(mode="json"),
                success=promoted,
            )
            if promoted:
                self.registry.record_metrics(metrics, production=False)
            results.append(ExperimentResult("breakout_param_search", params, metrics, promoted))
        results.sort(key=lambda r: (r.promoted_candidate, r.metrics.sharpe, r.metrics.stability_score), reverse=True)
        return results

    def search_dual_momentum_params(
        self,
        frames: dict[str, pd.DataFrame],
        nifty: pd.DataFrame | None = None,
    ) -> list[ExperimentResult]:
        train, test = self.walk_forward_split(frames)
        # Split nifty the same way when available
        nifty_train = nifty_test = None
        if nifty is not None and len(nifty) > 100:
            cut = int(len(nifty) * 0.7)
            nifty_train, nifty_test = nifty.iloc[:cut], nifty.iloc[cut:]

        grid = {
            "lookback": [63, 84, 105],
            "top_n": [2, 3],
            "rebalance_every": [21, 42, 63],
            "abs_mom_floor": [0.0, 0.03],
        }
        keys = list(grid.keys())
        results: list[ExperimentResult] = []
        for values in product(*[grid[k] for k in keys]):
            params = dict(zip(keys, values))
            train_bt = run_dual_momentum_baseline(
                train, nifty_train, nifty_sma_gate=200, sector_neutral=False, **params
            )
            test_bt = run_dual_momentum_baseline(
                test, nifty_test, nifty_sma_gate=200, sector_neutral=False, **params
            )
            metrics = StrategyMetrics(
                strategy_id="dual_momentum_baseline",
                version=(
                    f"lb{params['lookback']}_n{params['top_n']}_"
                    f"rb{params['rebalance_every']}_f{params['abs_mom_floor']}"
                ),
                sample_count=test_bt.trades,
                win_rate=test_bt.win_rate,
                expectancy=test_bt.expectancy,
                sharpe=test_bt.sharpe,
                sortino=test_bt.sortino,
                max_drawdown=test_bt.max_drawdown,
                profit_factor=test_bt.profit_factor,
                avg_holding_days=test_bt.avg_holding_days,
                stability_score=test_bt.stability_score,
                total_return=test_bt.total_return,
                oos=True,
            )
            promoted = (
                test_bt.trades >= 6
                and test_bt.sharpe > 0.2
                and test_bt.stability_score >= 0.35
                and test_bt.max_drawdown > -0.22
                and test_bt.total_return > 0
                and train_bt.sharpe > -0.3
            )
            self.registry.record_experiment(
                name="dual_momentum_search",
                params=params,
                result=metrics.model_dump(mode="json"),
                success=promoted,
            )
            if promoted:
                self.registry.record_metrics(metrics, production=False)
            results.append(ExperimentResult("dual_momentum_search", params, metrics, promoted))
        results.sort(
            key=lambda r: (r.promoted_candidate, r.metrics.sharpe, r.metrics.stability_score),
            reverse=True,
        )
        return results
