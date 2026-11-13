# Measured Results Log

All figures use real Yahoo NSE OHLCV, ₹ costs/slippage model, ₹1,00,000 paper capital, 1% risk/trade, max 3 positions.

## 2026-11-13 — Phase 0 baselines (~2y, ~35 names)

Daily breakout / pullback: **negative expectancy after costs — not promoted.**

## 2026-11-13 — Phase 1 (post #1)

Dual-momentum 84/42 on small sample: mild positive / unstable — research default only.  
NIFTY buy-hold still ahead on that window.

## 2026-11-13 — Phase 2 (NIFTY SMA200 gate + expanded universe)

### Setup
- Universe expanded to **~120 liquid NSE names**
- Daily history **~5y** (~1235 NIFTY bars)
- Dual momentum with **NIFTY > SMA200 cash gate** + **sector-neutral residual ranking**

### Full-sample comparison

| Strategy | Trades | Sharpe | Total return | Max DD | Profit factor | Stability |
|----------|--------|--------|--------------|--------|---------------|-----------|
| **dual_momentum_nifty_gated** | 44 | **0.75** | **+99%** | −30.6% | **2.35** | 0.29 |
| dual_momentum (no gate) | 64 | 0.63 | +102% | **−55.0%** | 1.95 | 0.21 |
| **nifty_buyhold_benchmark** | 1 | 0.55 | +39% | −16.5% | — | 0.80 |

### Decisions
1. **NIFTY SMA200 gate is a measured win** — better Sharpe than ungated dual-mom and beats NIFTY BH Sharpe/return on this sample, with far lower DD than ungated (−31% vs −55%).
2. Still **not production-promoted**: walk-forward cells remain low-stability / thin trade counts. Gate is enabled for research + live **cash bias**, promotion still requires OOS stability.
3. Scanner enforces the same gate (risk-on families disabled when NIFTY < SMA200).

### Live scan (same day)
- Gate: **ON**
- Ranked opportunities appeared (e.g. TECHM ensemble RS+trend) under the gated dual-momentum path.

## Reproduce

```bash
python3 -m ati.cli fetch-data
python3 -m ati.cli backtest-baseline --strategy dual-gated
python3 -m ati.cli backtest-baseline --strategy benchmark
python3 -m ati.cli research --limit 60 --family dual
python3 -m ati.cli scan --limit 60 --top 5
```
