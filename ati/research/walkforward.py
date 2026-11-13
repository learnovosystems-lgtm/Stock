"""Rolling walk-forward evaluation for NIFTY-gated dual momentum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ati.models.opportunity import StrategyMetrics
from ati.research.backtest import BacktestResult, run_dual_momentum_baseline, run_nifty_buyhold_benchmark
from ati.research.registry import StrategyRegistry


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    strategy: BacktestResult
    benchmark: BacktestResult
    beats_benchmark_sharpe: bool


@dataclass
class WalkForwardReport:
    strategy_id: str
    folds: list[FoldResult]
    oos_sharpe_mean: float
    oos_sharpe_std: float
    oos_return_mean: float
    oos_max_dd_worst: float
    oos_trades: int
    fold_stability: float
    beats_benchmark_fold_share: float
    promotable: bool
    reason: str
    metrics: StrategyMetrics


def _slice_frames(frames: dict[str, pd.DataFrame], start, end) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in frames.items():
        part = df.loc[(df.index >= start) & (df.index <= end)]
        if len(part) >= 80:
            out[sym] = part
    return out


def _fold_stability(sharpes: list[float]) -> float:
    if len(sharpes) < 2:
        return 0.3
    arr = np.array(sharpes, dtype=float)
    positive_share = float((arr > 0).mean())
    dispersion = float(arr.std())
    mean = float(arr.mean())
    base = 1 / (1 + dispersion / (abs(mean) + 1e-9))
    return float(np.clip(base * (0.5 + 0.5 * positive_share), 0, 1))


def run_rolling_walkforward(
    frames: dict[str, pd.DataFrame],
    nifty: pd.DataFrame,
    sectors: dict[str, str] | None = None,
    *,
    india_vix: pd.DataFrame | None = None,
    n_folds: int = 4,
    train_frac: float = 0.6,
    lookback: int = 84,
    top_n: int = 3,
    rebalance_every: int = 42,
    nifty_sma_gate: int = 200,
    vix_cash_threshold: float | None = 18.0,
    registry: StrategyRegistry | None = None,
) -> WalkForwardReport:
    """Expanding/rolling OOS folds on the gated dual-momentum book."""
    registry = registry or StrategyRegistry()
    # Align on calendar intersection
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values() if len(df)]))
    if nifty is not None and len(nifty):
        all_dates = [d for d in all_dates if d in nifty.index]
    if len(all_dates) < 400:
        empty_m = StrategyMetrics(
            strategy_id="dual_momentum_nifty_gated",
            version="wf_insufficient",
            sample_count=0,
            win_rate=0,
            expectancy=0,
            sharpe=0,
            sortino=0,
            max_drawdown=0,
            profit_factor=0,
            avg_holding_days=0,
            stability_score=0,
            total_return=0,
            oos=True,
        )
        return WalkForwardReport(
            strategy_id="dual_momentum_nifty_gated",
            folds=[],
            oos_sharpe_mean=0,
            oos_sharpe_std=0,
            oos_return_mean=0,
            oos_max_dd_worst=0,
            oos_trades=0,
            fold_stability=0,
            beats_benchmark_fold_share=0,
            promotable=False,
            reason="Insufficient history for walk-forward",
            metrics=empty_m,
        )

    dates = pd.DatetimeIndex(all_dates)
    # Use expanding train + contiguous OOS test blocks at the end of each fold window
    fold_results: list[FoldResult] = []
    # Split the timeline into n_folds sequential test segments after a warm-up
    warmup = int(len(dates) * 0.35)
    test_region = dates[warmup:]
    if len(test_region) < n_folds * 40:
        n_folds = max(2, len(test_region) // 40)
    edges = np.linspace(0, len(test_region), n_folds + 1, dtype=int)

    for f in range(n_folds):
        test_slice = test_region[edges[f] : edges[f + 1]]
        if len(test_slice) < 30:
            continue
        test_start, test_end = test_slice[0], test_slice[-1]
        # Train = all history before test_start (expanding)
        train_dates = dates[dates < test_start]
        if len(train_dates) < 120:
            continue
        # Keep last train_frac of available pre-test history as active train window
        cut = int(len(train_dates) * (1 - train_frac))
        train_window = train_dates[cut:]
        train_start, train_end = train_window[0], train_window[-1]

        train_frames = _slice_frames(frames, train_start, train_end)
        # Warmup history before test so SMA200/lookback are defined; trades only after test_start
        warmup_bars = max(lookback, nifty_sma_gate) + 5
        warmup_start_idx = max(0, list(dates).index(test_start) - warmup_bars)
        warmup_start = dates[warmup_start_idx]
        test_frames = _slice_frames(frames, warmup_start, test_end)
        nifty_train = nifty.loc[(nifty.index >= train_start) & (nifty.index <= train_end)]
        nifty_test = nifty.loc[(nifty.index >= warmup_start) & (nifty.index <= test_end)]
        nifty_bench = nifty.loc[(nifty.index >= test_start) & (nifty.index <= test_end)]
        vix_test = None
        if india_vix is not None and len(india_vix):
            vix_test = india_vix.loc[(india_vix.index >= warmup_start) & (india_vix.index <= test_end)]

        # Fit-free strategy: params fixed; train used only for presence/liquidity filter
        liquid = {s for s, df in train_frames.items() if len(df) >= lookback + 20}
        test_frames = {s: df for s, df in test_frames.items() if s in liquid}
        if len(test_frames) < top_n + 2:
            continue

        strat = run_dual_momentum_baseline(
            test_frames,
            nifty_test,
            lookback=lookback,
            top_n=top_n,
            rebalance_every=rebalance_every,
            nifty_sma_gate=nifty_sma_gate,
            sectors=sectors,
            sector_neutral=True,
            strategy_id="dual_momentum_nifty_gated",
            trade_start=test_start,
            india_vix=vix_test,
            vix_cash_threshold=vix_cash_threshold,
        )
        bench = run_nifty_buyhold_benchmark(nifty_bench)
        fold_results.append(
            FoldResult(
                fold=f + 1,
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                strategy=strat,
                benchmark=bench,
                beats_benchmark_sharpe=strat.sharpe > bench.sharpe and strat.trades >= 2,
            )
        )

    if not fold_results:
        metrics = StrategyMetrics(
            strategy_id="dual_momentum_nifty_gated",
            version="wf_no_folds",
            sample_count=0,
            win_rate=0,
            expectancy=0,
            sharpe=0,
            sortino=0,
            max_drawdown=0,
            profit_factor=0,
            avg_holding_days=0,
            stability_score=0,
            total_return=0,
            oos=True,
        )
        return WalkForwardReport(
            strategy_id="dual_momentum_nifty_gated",
            folds=[],
            oos_sharpe_mean=0,
            oos_sharpe_std=0,
            oos_return_mean=0,
            oos_max_dd_worst=0,
            oos_trades=0,
            fold_stability=0,
            beats_benchmark_fold_share=0,
            promotable=False,
            reason="No valid OOS folds",
            metrics=metrics,
        )

    sharpes = [f.strategy.sharpe for f in fold_results]
    rets = [f.strategy.total_return for f in fold_results]
    dds = [f.strategy.max_drawdown for f in fold_results]
    trades = sum(f.strategy.trades for f in fold_results)
    win_rates = [f.strategy.win_rate for f in fold_results if f.strategy.trades > 0]
    expectancies = [f.strategy.expectancy for f in fold_results if f.strategy.trades > 0]
    holds = [f.strategy.avg_holding_days for f in fold_results if f.strategy.trades > 0]
    pfs = [f.strategy.profit_factor for f in fold_results if f.strategy.trades > 0]
    beat_share = float(np.mean([1.0 if f.beats_benchmark_sharpe else 0.0 for f in fold_results]))
    stability = _fold_stability(sharpes)

    metrics = StrategyMetrics(
        strategy_id="dual_momentum_nifty_gated",
        version=f"wf_{n_folds}fold_lb{lookback}_rb{rebalance_every}",
        sample_count=trades,
        win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        expectancy=float(np.mean(expectancies)) if expectancies else 0.0,
        sharpe=float(np.mean(sharpes)),
        sortino=float(np.mean([f.strategy.sortino for f in fold_results])),
        max_drawdown=float(min(dds)) if dds else 0.0,
        profit_factor=float(np.mean(pfs)) if pfs else 0.0,
        avg_holding_days=float(np.mean(holds)) if holds else 0.0,
        stability_score=stability,
        total_return=float(np.mean(rets)),
        oos=True,
    )

    # Promotion gates: OOS edge + stability + not catastrophic DD + enough trades
    promotable = (
        metrics.sample_count >= 12
        and metrics.sharpe >= 0.25
        and metrics.stability_score >= 0.35
        and metrics.max_drawdown > -0.40
        and beat_share >= 0.5
    )
    reason = "Clears rolling OOS gates" if promotable else (
        f"Blocked: trades={metrics.sample_count}, sharpe={metrics.sharpe:.2f}, "
        f"stability={metrics.stability_score:.2f}, dd={metrics.max_drawdown:.2f}, "
        f"beat_bh_share={beat_share:.0%}"
    )

    registry.record_metrics(metrics, production=False)
    registry.record_experiment(
        name="rolling_walkforward_gated",
        params={
            "n_folds": n_folds,
            "lookback": lookback,
            "rebalance_every": rebalance_every,
            "top_n": top_n,
            "nifty_sma_gate": nifty_sma_gate,
            "vix_cash_threshold": vix_cash_threshold,
        },
        result={
            "metrics": metrics.model_dump(mode="json"),
            "beat_share": beat_share,
            "folds": [
                {
                    "fold": f.fold,
                    "test": f"{f.test_start}->{f.test_end}",
                    "sharpe": f.strategy.sharpe,
                    "bench_sharpe": f.benchmark.sharpe,
                    "ret": f.strategy.total_return,
                    "trades": f.strategy.trades,
                }
                for f in fold_results
            ],
        },
        success=promotable,
    )
    if promotable:
        registry.promote(
            metrics.strategy_id,
            metrics.version,
            min_trades=12,
            min_stability=0.35,
            max_dd=-0.40,
        )

    return WalkForwardReport(
        strategy_id=metrics.strategy_id,
        folds=fold_results,
        oos_sharpe_mean=float(np.mean(sharpes)),
        oos_sharpe_std=float(np.std(sharpes)),
        oos_return_mean=float(np.mean(rets)),
        oos_max_dd_worst=float(min(dds)),
        oos_trades=trades,
        fold_stability=stability,
        beats_benchmark_fold_share=beat_share,
        promotable=promotable,
        reason=reason,
        metrics=metrics,
    )
