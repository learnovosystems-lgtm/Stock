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

## 2026-11-13 — Phase 3 (rolling walk-forward + gated paper)

### Goals
1. Harden OOS evaluation with rolling folds (+ warmup so SMA200 is defined)
2. Restrict paper auto-entries to dual_momentum / relative_strength only
3. Enforce max 1 position per sector + min 1% stop distance

### Rolling walk-forward (4 folds, gated dual-mom)

| Fold | OOS Sharpe | BH Sharpe | Beats BH | Trades |
|------|------------|-----------|----------|--------|
| 1 | 2.10 | 1.51 | yes | 10 |
| 2 | 0.63 | 0.15 | yes | 8 |
| 3 | 0.17 | 0.95 | no | 9 |
| 4 | −0.45 | 0.04 | no | 3 |

Aggregate: mean Sharpe **0.61**, trades **30**, DD worst **−21%**, beat-BH share **50%**, fold stability **0.34**.

**Promotion: blocked** (stability 0.34 &lt; 0.35 gate — near miss; recent fold weak).  
Paper trading of gated BUYs is enabled anyway for live measurement vs backtest.

### Paper session smoke
EOD session entered **COFORGE** via `dual_momentum_v2` (allowed list enforced).

### Commands

```bash
python3 -m ati.cli walkforward --folds 4
python3 -m ati.cli session eod
python3 -m ati.cli paper-status
```

## 2026-11-13 — Phase 4 (India VIX cash overlay)

### Change
When India VIX ≥ 18, flatten to cash (in addition to NIFTY SMA200 gate).

### Full-sample (gated + VIX18)
Sharpe **0.63**, return **+71%**, DD **−31%**, stability **0.60** (vs Phase-2 stability 0.29).  
Still beats NIFTY BH Sharpe (0.55).

### Walk-forward impact
Threshold 18 does **not** move OOS fold stats materially (fold 4 remains weak). Lower thresholds (15–16) hurt fold 2. Keep **18** as live defensive overlay; promotion still blocked on OOS stability.

## Reproduce

```bash
python3 -m ati.cli fetch-data
python3 -m ati.cli backtest-baseline --strategy dual-gated
python3 -m ati.cli backtest-baseline --strategy benchmark
python3 -m ati.cli walkforward --folds 4
python3 -m ati.cli research --limit 60 --family dual
python3 -m ati.cli scan --limit 60 --top 5
python3 -m ati.cli session eod
```
