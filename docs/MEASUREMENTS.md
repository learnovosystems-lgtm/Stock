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

## 2026-11-13 — Phase 5/6 (breadth + DD kill-switch + multi-book regime switch)

### Why not only dual-momentum?
Live scanner already ranks **11 strategy families** and `StrategySelector` weights them by regime. Research/paper had collapsed onto dual-mom after breakout/pullback failed measurement. Phase 6 restores **auto-switch**:

| Regime | Active book |
|--------|-------------|
| BULL | dual_momentum (NIFTY SMA200 + breadth gates) |
| SIDEWAYS | mean_reversion (RSI stretch) |
| BEAR / HIGH_VOL / UNCERTAIN | cash |
| Any | India VIX≥18 → cash; portfolio DD ≤ −20% → flatten + cool-off |

Paper allowlist expanded to: dual_momentum, relative_strength, trend_following, mean_reversion, multi_factor, momentum, ensemble.

### Full-sample (~5y)

| Strategy | Trades | Sharpe | Return | Max DD | Stability | Notes |
|----------|--------|--------|--------|--------|-----------|-------|
| **dual_momentum_nifty_gated** (+breadth0.40, DD−20%) | 33 | **0.65** | **+66%** | −27.5% | 0.34 | Still beats BH Sharpe 0.55 |
| mean_reversion_baseline | 232 | 0.32 | +26% | −39% | 0.21 | Unstable alone — SIDEWAYS sleeve only |
| regime_adaptive_multi | 11 | −0.56 | −19% | −22% | 0.32 | Full-sample hurt by early DD cool-offs |
| nifty_buyhold | 1 | 0.55 | +39% | −16.5% | 0.80 | Benchmark |

### Walk-forward OOS (4 folds)

**Dual-momentum gated (promoted):**

| Fold | Sharpe | BH | Beats | Trades |
|------|--------|-----|-------|--------|
| 1 | 2.10 | 1.51 | yes | 10 |
| 2 | 0.46 | 0.15 | yes | 8 |
| 3 | 0.17 | 0.95 | no | 9 |
| 4 | 0.00 | 0.04 | no | 0 (cash — gates) |

Aggregate: mean Sharpe **0.68**, stability **0.39**, trades 27, beat-BH 50%, DD worst −21%.  
**Promotion: YES** → `dual_momentum_nifty_gated` / `wf_4fold_lb84_rb42`.

**Regime-adaptive multi-book (not yet promoted):**

| Fold | Sharpe | BH | Beats | Trades |
|------|--------|-----|-------|--------|
| 1 | 0.66 | 1.51 | no | 16 |
| 2 | 0.54 | 0.15 | yes | 11 |
| 3 | 1.26 | 0.95 | yes | 36 |
| 4 | −0.74 | 0.04 | no | 18 |

Aggregate: mean Sharpe **0.43**, stability **0.33** (gate 0.35 — near miss), trades 81.  
Mean-reversion sleeve helps fold 3; fold 4 still weak. Keep measuring live; do not displace promoted dual-mom yet.

### Decisions
1. **Primary production sleeve:** gated dual-momentum (promoted on OOS).
2. **Live auto-switch:** scanner + selector rotate families by regime; paper accepts multiple measured families.
3. **Adaptive research book:** continues as challenger until fold stability ≥ 0.35.
4. Breakout/pullback remain demoted (negative measured expectancy).

## Reproduce

```bash
python3 -m ati.cli fetch-data
python3 -m ati.cli backtest-baseline --strategy dual-gated
python3 -m ati.cli backtest-baseline --strategy mean-rev
python3 -m ati.cli backtest-baseline --strategy adaptive
python3 -m ati.cli backtest-baseline --strategy benchmark
python3 -m ati.cli walkforward --mode dual --folds 4
python3 -m ati.cli walkforward --mode adaptive --folds 4
python3 -m ati.cli research --limit 60 --family dual
python3 -m ati.cli scan --limit 60 --top 5
python3 -m ati.cli session eod
```
