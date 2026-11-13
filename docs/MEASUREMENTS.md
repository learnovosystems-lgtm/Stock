# Measured Results Log

All figures use real Yahoo NSE OHLCV, ₹ costs/slippage model, ₹1,00,000 paper capital, 1% risk/trade, max 3 positions.

## 2026-11-13 — Phase 0 baselines (≈2y daily, ~35 liquid names)

### Breakout momentum (tightened filters)

| Metric | Value |
|--------|-------|
| Trades | 37–42 |
| Win rate | ~40–43% |
| Expectancy | ≈ −0.5%/trade |
| Sharpe | ≈ −0.42 to −0.46 |
| Max DD | ≈ −12% |
| Decision | **Not promoted** |

### Pullback trend

| Metric | Value |
|--------|-------|
| Trades | ~57–60 |
| Sharpe | ≈ −0.02 to −0.59 (sample-sensitive) |
| Decision | **Not promoted** — demoted in selector priors |

## 2026-11-13 — Phase 1 improvement cycle (post-merge)

### Hypothesis
Daily high-turnover breakouts are cost-destroyed. Switch to dual momentum (absolute + relative), lower rebalance frequency, longer exits, and metric-weighted selection that crushes families with negative measured expectancy/Sharpe.

### Dual momentum (tuned defaults lookback=84, rebalance=42, top_n=3)

| Metric | Value |
|--------|-------|
| Trades | 22 |
| Win rate | ~23–41% depending on window |
| Full-sample Sharpe | ≈ **+0.10** |
| Full-sample return | ≈ **+0.4%** |
| Max DD | ≈ −24.5% |
| Avg hold | ~56 trading days |
| vs prior 126/21 defaults | Sharpe −0.91 / ret −26% — **rejected** |

### Walk-forward dual-momentum grid
Best OOS cells were mild positive Sharpe (~0.17) with **low stability** — **no production promotion**.

### NIFTY buy & hold benchmark
Recorded via `ati.cli backtest-baseline --strategy benchmark` for opportunity-cost context.

### Selector behaviour after metrics
- Families with latest expectancy&lt;0 and Sharpe&lt;0 get weight crushed (~0.15×)
- Bull prior boosts `relative_strength` / `trend_following` / `momentum`
- `breakout` / `pullback` prior demoted until metrics recover

### Live scan discipline
Continues to emit **WAIT / zero BUY** when entries are overextended. Paper capital preserved.

## Decisions locked by evidence

1. Do **not** promote daily breakout or naive dual-momentum 126/21.
2. Keep dual-momentum 84/42 as **research default**, not production.
3. Prefer NO TRADE over forcing signals while expectancy remains weak after costs.
4. Next levers: longer history / larger universe, sector-neutral residuals, explicit NIFTY SMA200 gate on entries, paid NSE feed for cleaner survivorship.

## Commands to reproduce

```bash
python3 -m ati.cli fetch-data --limit 40
python3 -m ati.cli backtest-baseline --limit 40 --strategy all
python3 -m ati.cli research --limit 30 --family dual
python3 -m ati.cli scan --limit 40 --top 5
```
