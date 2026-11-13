# ATI — Autonomous Adaptive Trading Intelligence (Indian Equities)

Evidence-driven swing trading research and paper-trading engine for **NSE**.  
Not a fake AI dashboard: every signal is computed from cached market data with reproducible formulas.

## What it does

1. **Market scanner** — ranks the liquid NSE universe with multi-timeframe context  
2. **Adaptive strategy selection** — 10 strategy families, regime-aware, ensemble when they agree  
3. **Entry / exit engines** — WAIT if chasing; stops never widen  
4. **Risk engine** — ₹1,00,000 paper book, 1% risk/trade, 3% open risk, 3 positions, min 1:2 R:R  
5. **Research laboratory** — parameter search + walk-forward OOS validation + version registry  
6. **ML layer** — XGBoost/GBM estimates P(target before stop); LLMs are not used for prices  
7. **Paper broker** — SQLite ledger with Indian-style costs/slippage  

Read the full decision record: [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# Cache market data
python3 -m ati.cli fetch-data --limit 40

# Regime
python3 -m ati.cli regime

# Measure baselines (incl. NIFTY-gated dual-momentum on full universe)
python3 -m ati.cli backtest-baseline --strategy dual-gated
python3 -m ati.cli backtest-baseline --strategy benchmark
python3 -m ati.cli backtest-baseline --strategy all

# Train ML reliability model
python3 -m ati.cli train-ml --limit 40

# Research lab (walk-forward dual-momentum by default)
python3 -m ati.cli research --limit 30 --family dual

# Scan for opportunities (real data)
python3 -m ati.cli scan --limit 40 --top 5

# Paper session (auto intra/EOD)
python3 -m ati.cli session auto
python3 -m ati.cli paper-status
```

## Opportunity output

Each opportunity uses the canonical report format (Stock, Signal, Ideal entry, Stop-loss, Targets, R:R, scores, regime, justifications, timestamp). The system will happily report **zero** opportunities.

## Architecture (selected)

| Layer | Choice | Why |
|-------|--------|-----|
| Data | Yahoo Finance + Parquet cache | Works for `.NS`, NIFTY, India VIX, 15m/1h without API keys |
| Baseline | Volume-confirmed Donchian breakout + ATR risk | Strong, testable swing baseline before adaptation |
| Selection | Regime router + stability-penalized OOS metrics + ensemble | Avoids chasing highest recent return |
| Execution | Local paper broker first | Prove edge before live brokerage adapters |
| ML | Trees / logistic / HMM-ready / IsolationForest | Measurable; deep nets only if they beat this |

## Disclaimer

For research and paper trading only. Markets involve loss of capital. Past backtests do not guarantee future results.
