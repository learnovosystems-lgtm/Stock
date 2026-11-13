"""Market scanner — ranks real opportunities across the NSE universe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from ati.config import REPORTS_DIR, DEFAULT, AppConfig, ensure_dirs
from ati.data.provider import MarketDataProvider
from ati.data.universe import SECTOR_INDEX, Instrument, load_universe
from ati.engine.entry import evaluate_entry
from ati.engine.ranking import build_opportunity, rank_opportunities
from ati.engine.risk import PortfolioRiskState, RiskEngine
from ati.engine.selection import StrategySelector
from ati.ml.inference import SignalModel
from ati.models.opportunity import Opportunity, StrategyMetrics
from ati.regime.detector import RegimeDetector, sector_relative_returns
from ati.research.gates import nifty_above_sma
from ati.strategies.base import StrategyContext
from ati.strategies.common import is_overextended, prepare_daily, rs_vs_nifty, weekly_bias
from ati.strategies.registry import all_strategies


@dataclass
class ScanResult:
    regime_notes: str
    regime: str
    opportunities: list[Opportunity]
    scanned: int
    generated_at: str


class MarketScanner:
    def __init__(
        self,
        config: AppConfig | None = None,
        provider: MarketDataProvider | None = None,
        metrics: list[StrategyMetrics] | None = None,
        model: SignalModel | None = None,
    ):
        self.config = config or DEFAULT
        self.provider = provider or MarketDataProvider(self.config)
        self.risk = RiskEngine(self.config.risk)
        self.regime_detector = RegimeDetector(self.provider)
        self.selector = StrategySelector(metrics or [])
        self.strategies = all_strategies()
        self.model = model or SignalModel.load_default()

    def scan(
        self,
        universe: list[Instrument] | None = None,
        *,
        equity: float | None = None,
        cash: float | None = None,
        force_refresh: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> ScanResult:
        ensure_dirs()
        universe = universe or load_universe()
        equity = equity if equity is not None else self.config.risk.starting_capital
        cash = cash if cash is not None else equity

        if progress:
            progress("Loading universe daily bars…")
        daily_map = self.provider.get_many_daily([i.symbol for i in universe], force=force_refresh)
        # Liquidity filter
        filtered: list[Instrument] = []
        for inst in universe:
            df = daily_map.get(inst.symbol)
            if df is None or len(df) < 80:
                continue
            avg_vol = float(df["Volume"].tail(20).mean())
            px = float(df["Close"].iloc[-1])
            if avg_vol < self.config.min_avg_volume or px < self.config.min_price:
                continue
            filtered.append(inst)

        sector_rets = sector_relative_returns(self.provider, SECTOR_INDEX)
        regime = self.regime_detector.detect(daily_map, sector_rets)
        nifty = self.provider.nifty()
        india_vix = self.provider.india_vix()
        from ati.research.gates import india_vix_elevated

        nifty_on = nifty_above_sma(nifty, window=self.config.nifty_trend_sma)
        vix_off = india_vix_elevated(india_vix, threshold=18.0)
        # regime.breadth is in [-1,1]; convert to positive-share proxy in [0,1]
        breadth_share = (regime.breadth + 1.0) / 2.0
        risk_on = nifty_on and not vix_off and breadth_share >= 0.40

        if progress:
            progress(
                f"Regime={regime.regime.value} conf={regime.confidence:.2f} "
                f"risk_on={'ON' if risk_on else 'OFF'} "
                f"(nifty={nifty_on}, vix_ok={not vix_off}, breadth={breadth_share:.2f}) "
                f"— scanning {len(filtered)} names"
            )

        opportunities: list[Opportunity] = []
        state = PortfolioRiskState(equity=equity, cash=cash, open_risk_rupees=0.0, open_positions=0, day_pnl=0.0)

        # When index trend gate is off, still allow mean-reversion/reversal families only
        gated_off_families = {
            "momentum",
            "relative_strength",
            "trend_following",
            "breakout",
            "pullback",
            "multi_factor",
            "volatility_expansion",
        }

        for inst in filtered:
            daily = prepare_daily(daily_map[inst.symbol])
            if daily.empty:
                continue
            # Stage 1: daily + weekly only (fast). Intraday fetched only for survivors.
            weekly = self.provider.get_weekly(inst.symbol)
            empty = pd.DataFrame()
            ctx = StrategyContext(
                symbol=inst.symbol,
                sector=inst.sector,
                cap=inst.cap.value,
                daily=daily,
                weekly=weekly,
                hourly=empty,
                m30=empty,
                m15=empty,
                nifty_daily=nifty,
                regime=regime,
                sector_strength=float(sector_rets.get(inst.sector, 0.0)),
                extras={"india_vix": india_vix},
            )

            raw_signals = []
            for strat in self.strategies:
                if regime.regime not in strat.suitable_regimes():
                    continue
                if not risk_on and strat.family in gated_off_families:
                    continue
                try:
                    sig = strat.generate(ctx)
                except Exception:  # noqa: BLE001
                    continue
                if sig is not None:
                    raw_signals.append(sig)

            selection = self.selector.select(raw_signals, regime, cap=inst.cap.value)
            if not selection.selected:
                continue

            # Stage 2: refine survivors with 1h / 15m confirmation
            hourly = self.provider.get_hourly(inst.symbol)
            m30 = self.provider.get_30m(inst.symbol)
            m15 = self.provider.get_15m(inst.symbol)
            ctx.hourly = hourly
            ctx.m30 = m30
            ctx.m15 = m15
            refined = []
            for strat in self.strategies:
                if regime.regime not in strat.suitable_regimes():
                    continue
                if not risk_on and strat.family in gated_off_families:
                    continue
                try:
                    sig = strat.generate(ctx)
                except Exception:  # noqa: BLE001
                    continue
                if sig is not None:
                    refined.append(sig)
            selection = self.selector.select(refined, regime, cap=inst.cap.value)
            if not selection.selected:
                continue

            # Ensemble: blend scores if multiple agree; use top for levels
            primary = selection.selected[0]
            if selection.ensemble and len(selection.selected) > 1:
                primary.score = sum(s.score for s in selection.selected) / len(selection.selected)
                label = "ensemble:" + "+".join(s.family for s in selection.selected)
            else:
                label = primary.strategy_id

            last = daily.iloc[-1]
            price = float(last["Close"])
            rs = rs_vs_nifty(daily, nifty)
            wb = weekly_bias(weekly)
            overext = is_overextended(daily)
            vol_ok = float(last.get("vol_surge", 1.0) or 1.0) >= 1.0
            in_trend = bool(last.get("Close", 0) > last.get("sma_50", 0)) if not pd.isna(last.get("sma_50")) else False
            approaching = bool(
                not pd.isna(last.get("dc_high")) and price >= 0.98 * float(last["dc_high"])
            )
            upside = 0.0
            if not pd.isna(last.get("dc_high")) and primary.stop < price:
                upside = (float(last["dc_high"]) - price) / max(price - primary.stop, 1e-9)

            stop_dist_pct = abs(price - primary.stop) / price
            stop_practical = 0.005 <= stop_dist_pct <= 0.08

            entry = evaluate_entry(
                primary,
                price,
                market_bullish=regime.ma_structure >= 0 and regime.nifty_trend >= 0,
                stronger_than_nifty=rs > 0,
                sector_outperforming=ctx.sector_strength >= 0,
                in_trend=in_trend or wb > 0,
                volume_confirming=vol_ok,
                approaching_breakout=approaching,
                pullback_available=price <= float(last.get("sma_20", price)) * 1.02
                if not pd.isna(last.get("sma_20"))
                else False,
                overextended=overext,
                upside_to_resistance=max(upside, primary.risk_reward),
                stop_practical=stop_practical,
            )

            avg_vol = float(daily["Volume"].tail(20).mean())
            sizing = self.risk.size_long(
                equity=state.equity,
                cash=state.cash,
                entry=entry.ideal_entry,
                stop=primary.stop,
                state=state,
                avg_volume=avg_vol,
            )

            ml = self.model.predict(daily, nifty, primary)
            opp = build_opportunity(
                stock=inst.symbol,
                sector=inst.sector,
                raw=primary,
                regime=regime,
                entry=entry,
                sizing=sizing,
                current_price=price,
                sector_strength=ctx.sector_strength,
                rs_nifty=rs,
                model_confidence=ml.confidence,
                est_prob=ml.prob_target_before_stop,
                expected_return=ml.expected_return,
                strategy_label=label,
            )
            opportunities.append(opp)

        ranked = rank_opportunities(opportunities)
        # Keep actionable + high-quality WAIT; drop NO_TRADE clutter unless debugging
        ranked = [o for o in ranked if o.signal.value in ("BUY", "WAIT", "SHORT")]
        gate_note = (
            f"NIFTY>SMA{self.config.nifty_trend_sma}={'ON' if nifty_on else 'OFF'}; "
            f"VIX<18={'OFF' if vix_off else 'ON'}; "
            f"breadth={breadth_share:.2f}; risk_on={'ON' if risk_on else 'OFF'}"
        )
        result = ScanResult(
            regime_notes=f"{regime.notes}; {gate_note}",
            regime=regime.regime.value,
            opportunities=ranked,
            scanned=len(filtered),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._persist(result)
        return result

    def _persist(self, result: ScanResult) -> Path:
        ensure_dirs()
        path = REPORTS_DIR / f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "regime": result.regime,
            "regime_notes": result.regime_notes,
            "scanned": result.scanned,
            "generated_at": result.generated_at,
            "opportunities": [o.model_dump(mode="json") for o in result.opportunities],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        latest = REPORTS_DIR / "scan_latest.json"
        latest.write_text(path.read_text())
        return path
