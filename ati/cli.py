"""ATI command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ati.config import ensure_dirs
from ati.data.provider import MarketDataProvider
from ati.data.universe import load_universe
from ati.ml.train import train_from_frames
from ati.models.opportunity import StrategyMetrics
from ati.ops.session import _default_broker, is_market_open, run_eod, run_intrasession_once
from ati.paper.portfolio import PaperBroker
from ati.research.adaptive import run_mean_reversion_baseline, run_regime_adaptive_book
from ati.research.adaptive_walkforward import run_adaptive_walkforward
from ati.research.backtest import (
    run_breakout_momentum_baseline,
    run_dual_momentum_baseline,
    run_nifty_buyhold_benchmark,
    run_pullback_trend_baseline,
)
from ati.research.laboratory import ResearchLaboratory
from ati.research.registry import StrategyRegistry
from ati.research.walkforward import run_rolling_walkforward
from ati.scanner.market_scanner import MarketScanner

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Autonomous Adaptive Trading Intelligence (India)")
console = Console()


def _progress(msg: str) -> None:
    console.print(f"[cyan]{msg}[/cyan]")


@app.command("fetch-data")
def fetch_data(
    limit: int = typer.Option(0, help="Limit universe size (0=all)"),
    force: bool = typer.Option(False, help="Force refresh cache"),
):
    """Download and cache OHLCV for the universe + NIFTY/VIX."""
    ensure_dirs()
    provider = MarketDataProvider()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    console.print(f"Fetching {len(uni)} symbols + indices…")
    provider.nifty(force=force)
    provider.india_vix(force=force)
    for i, inst in enumerate(uni, 1):
        provider.get_daily(inst.symbol, force=force)
        if i % 10 == 0:
            console.print(f"  {i}/{len(uni)}")
    console.print("[green]Done.[/green]")


@app.command("scan")
def scan(
    top: int = typer.Option(10, help="Show top N opportunities"),
    limit: int = typer.Option(0, help="Limit universe (0=all)"),
    force: bool = typer.Option(False, help="Force data refresh"),
    paper_enter: bool = typer.Option(False, help="Auto-enter top BUY into paper book"),
):
    """Run market scanner and print canonical opportunity reports."""
    ensure_dirs()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    registry = StrategyRegistry()
    scanner = MarketScanner(metrics=registry.latest_metrics())
    broker = PaperBroker()
    acct = broker.account()
    result = scanner.scan(
        uni,
        equity=acct["equity"],
        cash=acct["cash"],
        force_refresh=force,
        progress=_progress,
    )
    console.print(f"\n[bold]Regime:[/bold] {result.regime} — {result.regime_notes}")
    console.print(f"Scanned {result.scanned} names → {len(result.opportunities)} ranked opportunities\n")
    if not result.opportunities:
        console.print("[yellow]Zero opportunities qualify. NO TRADE.[/yellow]")
        return

    shown = 0
    for opp in result.opportunities:
        if shown >= top:
            break
        console.print(opp.format_report())
        console.print("-" * 72)
        shown += 1
        if paper_enter and opp.signal.value == "BUY":
            ok, msg = broker.try_enter(opp)
            console.print(f"[paper] {msg}")
            if ok:
                paper_enter = False  # one entry per scan unless expanded later


@app.command("backtest-baseline")
def backtest_baseline(
    limit: int = typer.Option(0, help="Symbols to include (0=full universe)"),
    strategy: str = typer.Option(
        "all",
        help="breakout|pullback|dual|dual-gated|mean-rev|adaptive|benchmark|all",
    ),
):
    """Measure baselines / adaptive multi-book and record metrics (honest, including losses)."""
    ensure_dirs()
    provider = MarketDataProvider()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    frames = provider.get_many_daily([i.symbol for i in uni])
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 200}
    nifty = provider.nifty()
    sectors = {i.symbol: i.sector for i in uni}
    registry = StrategyRegistry()

    def _record(result, version: str = "1.1.0"):
        payload = {
            "strategy_id": result.strategy_id,
            "trades": result.trades,
            "win_rate": round(result.win_rate, 4),
            "expectancy": round(result.expectancy, 4),
            "sharpe": round(result.sharpe, 4),
            "sortino": round(result.sortino, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "profit_factor": round(result.profit_factor, 4),
            "avg_holding_days": round(result.avg_holding_days, 2),
            "total_return": round(result.total_return, 4),
            "stability_score": round(result.stability_score, 4),
            "symbols": len(frames),
            "bars_nifty": len(nifty),
        }
        console.print(json.dumps(payload, indent=2))
        registry.record_metrics(
            StrategyMetrics(
                strategy_id=result.strategy_id,
                version=version,
                sample_count=result.trades,
                win_rate=result.win_rate,
                expectancy=result.expectancy,
                sharpe=result.sharpe,
                sortino=result.sortino,
                max_drawdown=result.max_drawdown,
                profit_factor=result.profit_factor,
                avg_holding_days=result.avg_holding_days,
                stability_score=result.stability_score,
                total_return=result.total_return,
                oos=False,
            )
        )

    if strategy in ("breakout", "all"):
        _record(run_breakout_momentum_baseline(frames))
    if strategy in ("pullback", "all"):
        _record(run_pullback_trend_baseline(frames))
    if strategy in ("dual", "all"):
        _record(
            run_dual_momentum_baseline(
                frames,
                nifty,
                nifty_sma_gate=0,
                sector_neutral=False,
                strategy_id="dual_momentum_baseline",
            ),
            version="1.1.0-lb84-rb42",
        )
    if strategy in ("dual-gated", "all"):
        vix = provider.india_vix()
        _record(
            run_dual_momentum_baseline(
                frames,
                nifty,
                nifty_sma_gate=200,
                sectors=sectors,
                sector_neutral=True,
                strategy_id="dual_momentum_nifty_gated",
                india_vix=vix,
                vix_cash_threshold=18.0,
                min_breadth=0.40,
                max_portfolio_dd=-0.20,
            ),
            version="2.2.0-breadth-ddkill",
        )
    if strategy in ("mean-rev", "all"):
        _record(run_mean_reversion_baseline(frames), version="1.0.0-rsi35-rb10")
    if strategy in ("adaptive", "all"):
        vix = provider.india_vix()
        result, diag = run_regime_adaptive_book(
            frames,
            nifty,
            sectors=sectors,
            india_vix=vix,
            vix_cash_threshold=18.0,
            min_breadth=0.40,
            max_portfolio_dd=-0.20,
        )
        _record(result, version="1.0.0-regime-switch")
        console.print_json(
            data={
                "adaptive_regime_days": diag.regime_counts,
                "adaptive_book_rebalances": diag.book_counts,
                "adaptive_switches": diag.switches,
            }
        )
    if strategy in ("benchmark", "all"):
        _record(run_nifty_buyhold_benchmark(nifty), version="1.0.0")


@app.command("research")
def research(
    limit: int = typer.Option(25, help="Symbols for lab search"),
    family: str = typer.Option("dual", help="breakout|dual|all"),
):
    """Run research laboratory parameter search with walk-forward OOS validation."""
    ensure_dirs()
    provider = MarketDataProvider()
    uni = load_universe()[:limit]
    frames = provider.get_many_daily([i.symbol for i in uni])
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 120}
    nifty = provider.nifty()
    lab = ResearchLaboratory()
    batches: list[tuple[str, list]] = []
    if family in ("breakout", "all"):
        batches.append(("breakout_param_search", lab.search_breakout_params(frames)))
    if family in ("dual", "all"):
        batches.append(("dual_momentum_baseline", lab.search_dual_momentum_params(frames, nifty)))

    for strategy_id, results in batches:
        table = Table(title=f"Research lab — {strategy_id} OOS")
        table.add_column("params")
        table.add_column("trades")
        table.add_column("sharpe")
        table.add_column("stability")
        table.add_column("ret")
        table.add_column("candidate")
        for r in results[:12]:
            table.add_row(
                str(r.params),
                str(r.metrics.sample_count),
                f"{r.metrics.sharpe:.3f}",
                f"{r.metrics.stability_score:.3f}",
                f"{r.metrics.total_return:.3f}",
                "yes" if r.promoted_candidate else "no",
            )
        console.print(table)
        for r in results:
            if r.promoted_candidate:
                ok = lab.registry.promote(strategy_id, r.metrics.version)
                console.print(f"Promote {strategy_id} {r.metrics.version}: {'OK' if ok else 'rejected by gates'}")
                break
        else:
            console.print(f"[yellow]No {strategy_id} candidate cleared promotion gates.[/yellow]")


@app.command("train-ml")
def train_ml(limit: int = typer.Option(40, help="Symbols for training")):
    """Train XGBoost/GBM signal reliability model on historical path labels."""
    ensure_dirs()
    provider = MarketDataProvider()
    uni = load_universe()[:limit]
    frames = provider.get_many_daily([i.symbol for i in uni])
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 120}
    report = train_from_frames(frames, provider.nifty())
    console.print(report)


@app.command("walkforward")
def walkforward(
    limit: int = typer.Option(0, help="Symbols (0=full universe)"),
    folds: int = typer.Option(4, help="Number of OOS folds"),
    mode: str = typer.Option(
        "adaptive",
        help="dual = single gated dual-mom | adaptive = regime multi-book switch",
    ),
):
    """Rolling walk-forward; default is regime-adaptive multi-strategy auto-switch."""
    ensure_dirs()
    provider = MarketDataProvider()
    uni = load_universe()
    if limit > 0:
        uni = uni[:limit]
    frames = provider.get_many_daily([i.symbol for i in uni])
    frames = {k: v for k, v in frames.items() if v is not None and len(v) > 200}
    nifty = provider.nifty()
    sectors = {i.symbol: i.sector for i in uni}
    vix = provider.india_vix()
    if mode == "dual":
        report = run_rolling_walkforward(
            frames,
            nifty,
            sectors,
            india_vix=vix,
            n_folds=folds,
            vix_cash_threshold=18.0,
            min_breadth=0.40,
            max_portfolio_dd=-0.20,
        )
        title = "Walk-forward OOS — dual-momentum gated"
    else:
        report = run_adaptive_walkforward(
            frames,
            nifty,
            sectors,
            india_vix=vix,
            n_folds=folds,
            vix_cash_threshold=18.0,
            min_breadth=0.40,
            max_portfolio_dd=-0.20,
        )
        title = "Walk-forward OOS — regime-adaptive multi-book"
    table = Table(title=title)
    table.add_column("fold")
    table.add_column("test")
    table.add_column("trades")
    table.add_column("sharpe")
    table.add_column("bh_sharpe")
    table.add_column("ret")
    table.add_column("beats_bh")
    table.add_column("books")
    for f in report.folds:
        books = ""
        diag = getattr(f, "diagnostics", None)
        if diag is not None:
            books = ",".join(f"{k}:{v}" for k, v in sorted(diag.book_counts.items()))
        table.add_row(
            str(f.fold),
            f"{f.test_start}→{f.test_end}",
            str(f.strategy.trades),
            f"{f.strategy.sharpe:.3f}",
            f"{f.benchmark.sharpe:.3f}",
            f"{f.strategy.total_return:.3f}",
            "yes" if f.beats_benchmark_sharpe else "no",
            books or "—",
        )
    console.print(table)
    console.print_json(
        data={
            "mode": mode,
            "strategy_id": report.strategy_id,
            "oos_sharpe_mean": round(report.oos_sharpe_mean, 4),
            "oos_sharpe_std": round(report.oos_sharpe_std, 4),
            "oos_return_mean": round(report.oos_return_mean, 4),
            "oos_max_dd_worst": round(report.oos_max_dd_worst, 4),
            "oos_trades": report.oos_trades,
            "fold_stability": round(report.fold_stability, 4),
            "beats_benchmark_fold_share": round(report.beats_benchmark_fold_share, 4),
            "promotable": report.promotable,
            "reason": report.reason,
            "version": report.metrics.version,
            "production": [p.get("version") for p in StrategyRegistry().list_production()],
        }
    )


@app.command("paper-status")
def paper_status():
    """Show paper account, positions, and risk state."""
    broker = _default_broker()
    snap = broker.snapshot()
    snap["allowed_strategies"] = list(broker.allowed_strategy_substrings)
    console.print_json(data=snap)


@app.command("session")
def session(mode: str = typer.Argument("auto", help="auto|intra|eod")):
    """Run continuous-operation pass (gated paper entries only)."""
    ensure_dirs()
    if mode == "auto":
        mode = "intra" if is_market_open() else "eod"
    if mode == "intra":
        out = run_intrasession_once()
    else:
        out = run_eod()
    console.print_json(data=out)


@app.command("regime")
def regime_cmd():
    """Print current market regime assessment."""
    from ati.regime.detector import RegimeDetector

    det = RegimeDetector()
    provider = MarketDataProvider()
    uni = load_universe()[:40]
    daily = provider.get_many_daily([i.symbol for i in uni])
    state = det.detect(daily)
    console.print_json(
        data={
            "regime": state.regime.value,
            "nifty_trend": state.nifty_trend,
            "india_vix": state.india_vix,
            "adx": state.adx,
            "ma_structure": state.ma_structure,
            "volatility": state.volatility,
            "breadth": state.breadth,
            "confidence": state.confidence,
            "preferred_families": state.preferred_families(),
            "notes": state.notes,
        }
    )


@app.command("analyze")
def analyze(
    limit: int = typer.Option(80, help="Universe limit"),
):
    """One-shot continuous analysis: graded opportunities with buy/sell/RR/SL."""
    from ati.ops.continuous import run_analysis_pass

    feed = run_analysis_pass(limit=limit, progress=_progress)
    console.print_json(
        data={
            "generated_at": feed["generated_at"],
            "counts": feed["counts"],
            "actionable": feed["actionable"][:10],
        }
    )


@app.command("watch")
def watch(
    interval: int = typer.Option(300, help="Seconds between analysis cycles"),
    limit: int = typer.Option(80, help="Universe limit"),
    cycles: int = typer.Option(0, help="Stop after N cycles (0=forever)"),
):
    """Continuous opportunity analysis loop (buy/sell/RR/SL feed)."""
    from ati.ops.continuous import run_continuous

    run_continuous(
        interval_sec=interval,
        limit=limit,
        cycles=cycles,
        progress=_progress,
    )


@app.command("zerodha-status")
def zerodha_status():
    """Fetch live Zerodha positions / P&L (requires KITE_* env secrets)."""
    from ati.brokers.zerodha import fetch_live_snapshot

    console.print_json(data=fetch_live_snapshot().to_dict())


@app.command("desk-snapshot")
def desk_snapshot(
    scan: bool = typer.Option(True, help="Run live scanner while building snapshot"),
    limit: int = typer.Option(80, help="Universe limit for scan latency"),
):
    """Write dashboard JSON snapshot (paper MTM + regime + opportunities + research)."""
    from ati.web.snapshot import write_snapshot
    from pathlib import Path

    ensure_dirs()
    path = write_snapshot(run_scan=scan, limit_scan=limit)
    dash = Path("dashboard/data/snapshot.json")
    dash.parent.mkdir(parents=True, exist_ok=True)
    dash.write_text(path.read_text())
    console.print(f"Wrote {path} and {dash}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8787, help="Bind port"),
    seed: bool = typer.Option(True, help="Seed snapshot before listen"),
):
    """Host ATI Desk dashboard with live paper/scan APIs."""
    from ati.web.server import serve as serve_desk
    from ati.web.snapshot import write_snapshot
    from pathlib import Path

    if seed:
        console.print("Seeding live desk snapshot…")
        path = write_snapshot(run_scan=True, limit_scan=60)
        dash = Path("dashboard/data/snapshot.json")
        dash.parent.mkdir(parents=True, exist_ok=True)
        dash.write_text(path.read_text())
        console.print(f"Snapshot ready: {dash}")
    console.print(f"Opening ATI Desk on http://{host}:{port}")
    serve_desk(host=host, port=port)


if __name__ == "__main__":
    app()
