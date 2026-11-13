"""Central configuration for paper-trading risk and market defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STATE_DIR = DATA_DIR / "state"
REPORTS_DIR = DATA_DIR / "reports"
REGISTRY_DIR = DATA_DIR / "registry"
MODELS_DIR = DATA_DIR / "models"


@dataclass(frozen=True)
class RiskConfig:
    starting_capital: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    max_open_risk_pct: float = 0.03
    max_daily_loss_pct: float = 0.02
    max_positions: int = 3
    min_reward_risk: float = 2.0
    max_portfolio_exposure_pct: float = 0.95
    # Approximate NSE cash equity costs (delivery-oriented paper model)
    brokerage_pct: float = 0.0003
    stt_sell_pct: float = 0.001
    slippage_pct: float = 0.0005
    other_charges_pct: float = 0.0002


@dataclass(frozen=True)
class MarketConfig:
    exchange: str = "NSE"
    currency: str = "INR"
    nifty_symbol: str = "^NSEI"
    india_vix_symbol: str = "^INDIAVIX"
    timezone: str = "Asia/Kolkata"
    session_open: str = "09:15"
    session_close: str = "15:30"
    # Yahoo intervals used by the engine
    weekly_interval: str = "1wk"
    daily_interval: str = "1d"
    hour_interval: str = "60m"
    confirm_interval: str = "30m"
    refine_interval: str = "15m"


@dataclass
class AppConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    lookback_days_daily: int = 1300  # ~5y trading days for expanded research
    lookback_days_intraday: int = 10
    min_avg_volume: float = 150_000.0
    min_price: float = 50.0
    max_price: float = 50_000.0
    production_strategy_flag: str = "production"
    # Require NIFTY above this SMA before new risk-on entries
    nifty_trend_sma: int = 200


DEFAULT = AppConfig()


def ensure_dirs() -> None:
    for path in (CACHE_DIR, STATE_DIR, REPORTS_DIR, REGISTRY_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
