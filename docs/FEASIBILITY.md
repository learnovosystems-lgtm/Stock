# Feasibility Assessment — Autonomous Adaptive Trading Engine (Indian Equities)

## 1. Objective

Build a self-directed quantitative research and opportunity-discovery engine for **Indian equities (NSE)** that:

- Discovers opportunities with **reproducible calculations on real market data**
- Adapts strategy selection by **rolling out-of-sample evidence**, not fixed dogma
- Enforces hard risk limits in **paper trading** (₹1,00,000 starting capital)
- Never uses an LLM as a numerical price oracle

## 2. Market & Data Reality (Quantitative)

| Factor | Finding | Implication |
|--------|---------|-------------|
| NSE OHLCV (daily) | Available via Yahoo Finance (`SYMBOL.NS`) | Primary free data path viable |
| NIFTY (`^NSEI`), India VIX (`^INDIAVIX`) | Available | Regime inputs feasible |
| Sector indices | Partial (`^CNXIT`, `^CNXAUTO`, …); some aliases missing | Use working proxies + sector ETF/index map |
| Intraday 15m / 1h | ~5 trading days history via Yahoo | Enough for **entry refinement**, not deep intraday research |
| Weekly bars | Available | Directional context OK |
| Level-2 / tick | Not free/reliable without paid feed | Paper fills use OHLC + slippage model |
| Corporate actions | Yahoo `auto_adjust=True` | Use adjusted prices; track splits in cache metadata |
| Survivorship | Free feeds omit delisted names | Bias mitigation: freeze universe snapshots; prefer liquid large/mid caps first |

**Verdict:** Feasible for swing research + paper trading on liquid NSE names. Not feasible (with free data) for HFT, precise VWAP, or full historical small-cap universe without survivorship correction.

## 3. Architecture Alternatives Compared

### 3.1 Data Provider

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Yahoo Finance (`yfinance`)** | No key, NSE + VIX + NIFTY, multi-TF | Rate limits, delayed, incomplete sectors | **Selected primary** |
| NSE official / unofficial scrapers | Native India | Fragile, ToS risk, brittle HTML/JSON | Fallback only |
| Polygon / Bloomberg / TrueData | Quality | Paid, keys required | Optional later adapter |
| Alpaca | Paper broker | US-centric | Rejected for India primary |

**Selected:** `yfinance` + local Parquet/SQLite cache for **reproducibility**. Provider interface allows paid NSE feeds later without rewriting strategies.

### 3.2 Strategy Framework

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Single fixed strategy | Simple | Violates directive | Rejected |
| Indicator soup / LLM picks | Looks “smart” | Non-reproducible | Rejected |
| **Plugin strategy registry + regime router + ensemble** | Matches directive; testable | More code | **Selected** |
| Pure RL from day one | Adaptive in theory | Sample-inefficient, opaque | Deferred until beats baselines |

**Selected:** Ten strategy **families** as independent plugins; regime engine weights/selects; ensemble when independent signals agree; **NO TRADE** when evidence insufficient.

### 3.3 Baseline Strategy (Strongest Evidence Start)

Empirical literature and retail-executable Indian swing practice favor **momentum + relative strength + volume-confirmed breakouts with ATR risk**, gated by market regime — not pure mean reversion in strong trends.

**Phase-0 baseline:** Relative-strength momentum with volume breakout / pullback entries, ATR stops, 1:2+ R:R filter.

Then measure walk-forward; only then promote competitors / ensembles.

### 3.4 ML Stack

| Option | Role | Decision |
|--------|------|----------|
| Logistic Regression / RF / XGBoost / LightGBM | P(target before stop), expected move, signal reliability | **Selected now** |
| HMM / Gaussian HMM | Regime classification | **Selected** |
| Isolation Forest | Anomaly / unreliable tape | **Selected** |
| LSTM / RL / deep nets | Sequence alpha | **Only if** measurable lift vs tree baselines |
| LLM numerical forecasts | — | **Forbidden** |

### 3.5 Execution

| Option | Decision |
|--------|----------|
| Live brokerage API first | Rejected until paper edge proven |
| **Local paper broker** (₹ costs: brokerage/STT/slippage) | **Selected** |
| Zerodha/Angel later | Adapter interface reserved |

### 3.6 UI

| Option | Decision |
|--------|----------|
| Fake AI dashboard | Rejected |
| **CLI + structured opportunity reports + SQLite state** | Selected |
| Optional read-only status JSON | Fine for ops |

## 4. Quantitative Risk Envelope (Initial Paper)

| Rule | Value |
|------|-------|
| Capital | ₹1,00,000 |
| Risk / trade | 1% (₹1,000) |
| Combined open risk | ≤ 3% |
| Daily loss halt | 2% |
| Max positions | 3 |
| Min R:R | 1:2 |
| Leverage | None |

Position size = `floor(risk_rupees / stop_distance)`, then capped by cash, liquidity, and exposure limits.

## 5. Feasibility by Directive Section

| Section | Feasible now? | Notes |
|---------|---------------|-------|
| A Adaptive swing + multi-TF | Yes | Daily/weekly primary; 15m/1h for entry polish (short history) |
| B Dynamic selection (10 families) | Yes | Walk-forward registry; start with liquid universe |
| C Entry optimization | Yes | Deterministic scoring; WAIT if chase |
| D Dynamic exits | Yes | Never widen stops |
| E Regime intelligence | Yes | NIFTY + VIX + ADX + MA + breadth proxy |
| F ML probability layer | Yes | Train on labeled historical setups |
| G Research laboratory | Yes | Parameter grid + walk-forward; version registry |
| H Capital/risk | Yes | Hard-coded engine gates |
| I Signal ranking | Yes | Multi-score Opportunity object |
| J Continuous ops | Partial | Scheduler/CLI loops; full always-on needs host process |
| K Output format | Yes | Canonical Opportunity serializer |

## 6. Phased Delivery

1. **Data + universe + costs + regime**
2. **Scanner + baseline momentum/RS/breakout + risk + opportunity format**
3. **Full strategy families + selection/ensemble + exits**
4. **Backtest lab + walk-forward + strategy registry**
5. **ML probability/regime models**
6. **Paper trading + session ops (intraday/EOD)**
7. **Measure → adapt → promote only validated versions**

## 7. Known Limits (Honest)

- Free intraday depth is shallow → swing bias, not scalping.
- Breadth uses liquid-universe advance/decline, not full NSE tape.
- Statistical arbitrage limited to liquid pairs / sector residuals.
- Past performance ≠ future results; paper ≠ live fills.

## 8. Success Criteria (Measured)

- End-to-end scan produces **zero or more** fully populated Opportunity records from **cached real OHLCV**.
- Baseline walk-forward metrics persisted (Sharpe, max DD, hit rate, expectancy).
- Paper book respects risk gates with unit tests.
- Strategy registry refuses unvalidated “promotion” to production flag.
