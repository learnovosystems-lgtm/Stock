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
from ati.ops.session import is_market_open, run_eod, run_intrasession_once
from ati.paper.portfolio import PaperBroker
from ati.research.backtest import (
    run_breakout_momentum_baseline,
    run_dual_momentum_baseline,
    run_nifty_buyhold_benchmark,
    run_pullback_trend_baseline,
)
from ati.research.laboratory import ResearchLaboratory
from ati.research.registry import StrategyRegistry
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
        help="breakout|pullback|dual|dual-gated|benchmark|all",
    ),
):
    """Measure baselines and record metrics (honest, including losses)."""
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
        _record(
            run_dual_momentum_baseline(
                frames,
                nifty,
                nifty_sma_gate=200,
                sectors=sectors,
                sector_neutral=True,
                strategy_id="dual_momentum_nifty_gated",
            ),
            version="2.0.0-sma200-sectneu",
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


@app.command("paper-status")
def paper_status():
    """Show paper account, positions, and risk state."""
    broker = PaperBroker()
    console.print_json(data=broker.snapshot())


@app.command("session")
def session(mode: str = typer.Argument("auto", help="auto|intra|eod")):
    """Run continuous-operation pass (intrasession or EOD)."""
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


if __name__ == "__main__":
    app()
