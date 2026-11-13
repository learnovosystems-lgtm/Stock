# Measured Results Log

All figures use real Yahoo NSE OHLCV, ₹ costs/slippage model, ₹1,00,000 paper capital, 1% risk/trade, max 3 positions.

## 2026-11-13 — Phase 0 baselines (≈2y daily, 35 liquid names)

### Breakout momentum (tightened filters)

| Metric | Value |
|--------|-------|
| Trades | 37 |
| Win rate | 43.2% |
| Expectancy | -0.55%/trade |
| Sharpe | -0.46 |
| Max DD | -12.0% |
| Profit factor | 0.78 |
| Stability | 0.29 |

### Pullback trend

| Metric | Value |
|--------|-------|
| Trades | 60 |
| Win rate | 31.7% |
| Expectancy | -0.55%/trade |
| Sharpe | -0.59 |
| Max DD | -22.3% |
| Profit factor | 0.80 |
| Stability | 0.29 |

**Decision:** Neither baseline is promoted to production. Scanner remains selective (WAIT / NO TRADE bias). Research lab OOS grid showed high short-window Sharpe but **failed stability gates** — correctly not promoted.

## ML

- Path-label classifier samples: 14,765
- Time-series CV AUC: ~0.59
- Inference uses blended calibrated probability (model + heuristic prior)

## Live scan (same day)

- Regime: **BULL** (ADX strong, breadth strong, MA stack incomplete → confidence 0.66)
- Qualifying actionable BUY: **0**
- Ranked WAIT examples: KOTAKBANK, ANGELONE (overextended — do not chase)
- Paper equity: ₹1,00,000 flat (discipline preserved)

## Next improvement levers

1. Sector-neutral residual / RS portfolio construction (weekly rebalance)
2. Cost-aware longer holding periods (reduce turnover)
3. Expand walk-forward windows before any production promotion
4. Optional paid NSE feed for deeper intraday research
