"""Live news + lightweight sentiment for recommendation gating.

Uses Yahoo Finance ticker news (free) and keyword polarity — no paid API.
Negative stock/market news demotes or blocks new BUY suggestions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

_POS = {
    "surge",
    "rally",
    "gain",
    "profit",
    "beat",
    "upgrade",
    "record",
    "strong",
    "growth",
    "win",
    "order",
    "deal",
    "approval",
    "bullish",
    "outperform",
}
_NEG = {
    "fall",
    "falls",
    "drop",
    "crash",
    "loss",
    "miss",
    "downgrade",
    "fraud",
    "probe",
    "ban",
    "weak",
    "cut",
    "layoff",
    "warning",
    "bearish",
    "default",
    "penalty",
    "raid",
    "scandal",
    "sebi",
    "fine",
}


@dataclass
class NewsItem:
    title: str
    publisher: str
    link: str
    published: str | None
    polarity: float  # -1..+1


@dataclass
class SentimentReport:
    symbol: str
    score: float  # -1..+1
    label: str  # bullish / neutral / bearish
    gate: str  # allow / caution / block
    reason: str
    headlines: list[NewsItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 3),
            "label": self.label,
            "gate": self.gate,
            "reason": self.reason,
            "headlines": [
                {
                    "title": h.title,
                    "publisher": h.publisher,
                    "link": h.link,
                    "published": h.published,
                    "polarity": h.polarity,
                }
                for h in self.headlines[:5]
            ],
        }


def _polarity(text: str) -> float:
    words = set(re.findall(r"[a-zA-Z]+", (text or "").lower()))
    if not words:
        return 0.0
    p = len(words & _POS)
    n = len(words & _NEG)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)


def _fetch_ticker_news(symbol: str, limit: int = 8) -> list[NewsItem]:
    sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    items: list[NewsItem] = []
    try:
        t = yf.Ticker(sym)
        raw = getattr(t, "news", None) or []
        for n in raw[:limit]:
            # yfinance shapes vary
            content = n.get("content") if isinstance(n.get("content"), dict) else n
            title = (
                (content or {}).get("title")
                or n.get("title")
                or ""
            )
            if not title:
                continue
            publisher = (
                ((content or {}).get("provider") or {}).get("displayName")
                if isinstance((content or {}).get("provider"), dict)
                else n.get("publisher") or ""
            )
            link = (
                ((content or {}).get("clickThroughUrl") or {}).get("url")
                if isinstance((content or {}).get("clickThroughUrl"), dict)
                else n.get("link") or ""
            )
            pub = None
            ts = n.get("providerPublishTime") or (content or {}).get("pubDate")
            if isinstance(ts, (int, float)):
                pub = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            elif isinstance(ts, str):
                pub = ts
            items.append(
                NewsItem(
                    title=str(title),
                    publisher=str(publisher or ""),
                    link=str(link or ""),
                    published=pub,
                    polarity=_polarity(str(title)),
                )
            )
    except Exception:  # noqa: BLE001
        return []
    return items


def _fetch_index_news(limit: int = 6) -> list[NewsItem]:
    # NIFTY proxy via ^NSEI rarely has news; use a liquid large-cap basket headline mix
    items: list[NewsItem] = []
    for sym in ("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"):
        items.extend(_fetch_ticker_news(sym.replace(".NS", ""), limit=2))
        if len(items) >= limit:
            break
    return items[:limit]


def analyze_symbol_sentiment(symbol: str) -> SentimentReport:
    headlines = _fetch_ticker_news(symbol)
    if not headlines:
        return SentimentReport(
            symbol=symbol.replace(".NS", ""),
            score=0.0,
            label="neutral",
            headlines=[],
            gate="allow",
            reason="No fresh headlines — neutral",
        )
    score = sum(h.polarity for h in headlines) / len(headlines)
    if score <= -0.35:
        label, gate = "bearish", "block"
        reason = "Negative headlines dominate — block new BUY"
    elif score <= -0.1:
        label, gate = "cautious", "caution"
        reason = "Mildly negative news — reduce conviction"
    elif score >= 0.25:
        label, gate = "bullish", "allow"
        reason = "Supportive headlines"
    else:
        label, gate = "neutral", "allow"
        reason = "Mixed/neutral news tape"
    return SentimentReport(
        symbol=symbol.replace(".NS", ""),
        score=float(score),
        label=label,
        headlines=headlines,
        gate=gate,
        reason=reason,
    )


def analyze_market_sentiment() -> SentimentReport:
    headlines = _fetch_index_news()
    if not headlines:
        return SentimentReport(
            symbol="MARKET",
            score=0.0,
            label="neutral",
            headlines=[],
            gate="allow",
            reason="No market headlines",
        )
    score = sum(h.polarity for h in headlines) / len(headlines)
    if score <= -0.3:
        gate, label = "caution", "bearish"
        reason = "Broad negative tape — prefer cash / tighter filters"
    elif score >= 0.25:
        gate, label = "allow", "bullish"
        reason = "Broad supportive tape"
    else:
        gate, label = "allow", "neutral"
        reason = "Neutral market news"
    return SentimentReport(
        symbol="MARKET",
        score=float(score),
        label=label,
        headlines=headlines,
        gate=gate,
        reason=reason,
    )
