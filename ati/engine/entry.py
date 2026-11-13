"""Intelligent entry optimization — WAIT when price chases beyond max entry."""

from __future__ import annotations

from dataclasses import dataclass

from ati.strategies.base import RawSignal


@dataclass
class EntryDecision:
    action: str  # BUY / WAIT / REJECT
    ideal_entry: float
    entry_low: float
    entry_high: float
    breakout_confirmation: float | None
    max_acceptable_entry: float
    invalidation_price: float
    entry_quality: float
    reason: str


def evaluate_entry(
    signal: RawSignal,
    current_price: float,
    *,
    market_bullish: bool,
    stronger_than_nifty: bool,
    sector_outperforming: bool,
    in_trend: bool,
    volume_confirming: bool,
    approaching_breakout: bool,
    pullback_available: bool,
    overextended: bool,
    upside_to_resistance: float,
    stop_practical: bool,
) -> EntryDecision:
    checks = {
        "market_bullish": market_bullish,
        "stronger_than_nifty": stronger_than_nifty,
        "sector_outperforming": sector_outperforming,
        "in_trend": in_trend,
        "volume_confirming": volume_confirming,
        "approaching_breakout": approaching_breakout,
        "pullback_available": pullback_available,
        "not_overextended": not overextended,
        "upside_ok": upside_to_resistance >= 1.5,
        "stop_practical": stop_practical,
    }
    # Weighted quality — not all must pass
    weights = {
        "market_bullish": 0.12,
        "stronger_than_nifty": 0.14,
        "sector_outperforming": 0.10,
        "in_trend": 0.12,
        "volume_confirming": 0.12,
        "approaching_breakout": 0.08,
        "pullback_available": 0.08,
        "not_overextended": 0.10,
        "upside_ok": 0.08,
        "stop_practical": 0.06,
    }
    quality = sum(weights[k] * (1.0 if v else 0.0) for k, v in checks.items())

    max_entry = signal.max_entry if signal.max_entry is not None else signal.entry_high
    invalidation = signal.invalidation

    if not stop_practical or signal.risk_reward < 2.0:
        return EntryDecision(
            action="REJECT",
            ideal_entry=signal.ideal_entry,
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            breakout_confirmation=signal.breakout_confirmation,
            max_acceptable_entry=max_entry,
            invalidation_price=invalidation,
            entry_quality=quality,
            reason="Stop impractical or R:R below 1:2",
        )

    if overextended and not pullback_available:
        return EntryDecision(
            action="WAIT",
            ideal_entry=signal.ideal_entry,
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            breakout_confirmation=signal.breakout_confirmation,
            max_acceptable_entry=max_entry,
            invalidation_price=invalidation,
            entry_quality=quality,
            reason="Overextended — wait for pullback toward ideal entry",
        )

    if current_price > max_entry:
        return EntryDecision(
            action="WAIT",
            ideal_entry=signal.ideal_entry,
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            breakout_confirmation=signal.breakout_confirmation,
            max_acceptable_entry=max_entry,
            invalidation_price=invalidation,
            entry_quality=quality,
            reason=f"Price {current_price:.2f} above max acceptable entry {max_entry:.2f} — do not chase",
        )

    if quality < 0.45:
        return EntryDecision(
            action="WAIT",
            ideal_entry=signal.ideal_entry,
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            breakout_confirmation=signal.breakout_confirmation,
            max_acceptable_entry=max_entry,
            invalidation_price=invalidation,
            entry_quality=quality,
            reason="Entry checklist quality below threshold",
        )

    in_range = signal.entry_low <= current_price <= max_entry
    action = "BUY" if in_range and signal.direction == "LONG" else "WAIT"
    return EntryDecision(
        action=action,
        ideal_entry=signal.ideal_entry,
        entry_low=signal.entry_low,
        entry_high=signal.entry_high,
        breakout_confirmation=signal.breakout_confirmation,
        max_acceptable_entry=max_entry,
        invalidation_price=invalidation,
        entry_quality=quality,
        reason="Entry checklist passed" if action == "BUY" else "Await price in entry range",
    )
