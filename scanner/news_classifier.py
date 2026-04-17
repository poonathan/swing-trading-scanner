import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import requests

from data.fetcher import TickerData
from scanner.models import NewsResult

logger = logging.getLogger(__name__)

_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
_TIMEOUT = 5


def classify(data: TickerData, config: dict) -> NewsResult:
    cfg = config.get("news", {})
    result = NewsResult(symbol=data.symbol, score=50.0)

    fund_kws = [k.lower() for k in cfg.get("fundamental_keywords", [])]
    macro_kws = [k.lower() for k in cfg.get("macro_keywords", [])]
    sector_kws = [k.lower() for k in cfg.get("sector_keywords", [])]
    window_days = cfg.get("sentiment_window_days", 14)
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    headlines = _collect_headlines(data, cutoff)
    result.headlines_analyzed = len(headlines)
    result.top_headlines = [h["title"] for h in headlines[:5]]

    fund_score = 0.0
    macro_score = 0.0

    for item in headlines:
        text = (item["title"] + " " + item.get("summary", "")).lower()
        age_days = max(0.1, item.get("age_days", 7))
        # Recency weight: last 3 days = 2x, 4-7 days = 1.5x, older = 1x
        weight = 2.0 if age_days <= 3 else (1.5 if age_days <= 7 else 1.0)

        for kw in fund_kws:
            if kw in text:
                fund_score += weight
        for kw in macro_kws:
            if kw in text:
                macro_score += weight
        for kw in sector_kws:
            if kw in text:
                macro_score += weight * 0.5  # sector = partial macro credit

    result.fundamental_signal_count = round(fund_score, 2)
    result.macro_signal_count = round(macro_score, 2)

    # Classify event type
    if fund_score >= 3.0:
        event_type = "fundamental"
    elif fund_score >= 1.5 and macro_score >= 2.0:
        event_type = "mixed"
    elif macro_score >= 2.0:
        event_type = "macro"
    elif fund_score == 0 and macro_score == 0 and len(headlines) >= 3:
        event_type = "unrelated"
    elif len(headlines) == 0:
        event_type = "unknown"
    else:
        event_type = "macro" if macro_score > fund_score else "unknown"

    result.event_type = event_type

    # Check upgrades/downgrades in last 21 days
    downgrade_count = _count_recent_downgrades(data, window_days=21)
    result.downgrade_count = downgrade_count

    # Check pre-earnings proximity
    result.pre_earnings = _is_pre_earnings(data, days_ahead=10)

    # Score
    score_map = {
        "macro": 85,
        "sector": 75,
        "unrelated": 90,
        "unknown": 50,
        "mixed": 30,
        "fundamental": 5,
    }
    base_score = score_map.get(event_type, 50)

    # Downgrade penalty (15 pts per downgrade, capped at -30)
    downgrade_penalty = min(30, downgrade_count * 15)
    base_score = max(0, base_score - downgrade_penalty)

    # Pre-earnings bonus for non-fundamental events
    if result.pre_earnings and event_type not in ("fundamental", "mixed"):
        base_score = min(100, base_score + 8)

    result.score = float(base_score)
    result.is_safe = event_type not in ("fundamental",)
    result.confidence = _confidence(len(headlines), fund_score, macro_score)
    return result


def _collect_headlines(data: TickerData, cutoff: datetime) -> list:
    items = []

    # yfinance news
    for item in (data.news or []):
        try:
            title = item.get("title", "")
            if not title:
                continue
            pub = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
            if pub:
                pub_dt = datetime.fromtimestamp(int(pub), tz=timezone.utc) if isinstance(pub, (int, float)) else _parse_dt(pub)
                if pub_dt and pub_dt < cutoff:
                    continue
                age_days = (datetime.now(timezone.utc) - pub_dt).days if pub_dt else 7
            else:
                age_days = 7
            items.append({"title": title, "summary": item.get("summary", ""), "age_days": age_days})
        except Exception:
            continue

    # Yahoo Finance RSS fallback
    try:
        url = _RSS_TEMPLATE.format(symbol=data.symbol)
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        for entry in feed.entries[:20]:
            pub_dt = _parse_dt(getattr(entry, "published", ""))
            if pub_dt and pub_dt < cutoff:
                continue
            age_days = (datetime.now(timezone.utc) - pub_dt).days if pub_dt else 7
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            if title and title not in {h["title"] for h in items}:
                items.append({"title": title, "summary": summary, "age_days": age_days})
    except Exception as e:
        logger.debug(f"{data.symbol}: RSS fetch failed - {e}")

    return items


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _count_recent_downgrades(data: TickerData, window_days: int = 21) -> int:
    try:
        ud = data.upgrades_downgrades
        if ud is None or ud.empty:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        if hasattr(ud.index, "tz_localize"):
            try:
                if ud.index.tz is None:
                    ud.index = ud.index.tz_localize("UTC")
            except Exception:
                pass
        recent = ud[ud.index >= cutoff]
        if "Action" in recent.columns:
            return int((recent["Action"].str.lower().isin(["down", "downgrade"])).sum())
        if "ToGrade" in recent.columns and "FromGrade" in recent.columns:
            # Heuristic: downgrade if from high grade to lower
            pass
        return 0
    except Exception:
        return 0


def _is_pre_earnings(data: TickerData, days_ahead: int = 10) -> bool:
    try:
        cal = data.calendar
        if not cal:
            return False
        for key in ["Earnings Date", "earnings_date", "earningsDate"]:
            val = cal.get(key)
            if val:
                dates = val if isinstance(val, list) else [val]
                for d in dates:
                    if isinstance(d, str):
                        d = datetime.strptime(d[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    elif hasattr(d, "to_pydatetime"):
                        d = d.to_pydatetime()
                        if d.tzinfo is None:
                            d = d.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    if now <= d <= now + timedelta(days=days_ahead):
                        return True
        return False
    except Exception:
        return False


def _confidence(n_headlines: int, fund_score: float, macro_score: float) -> float:
    if n_headlines == 0:
        return 0.2
    signal_strength = fund_score + macro_score
    base = min(0.9, 0.3 + n_headlines * 0.05 + signal_strength * 0.05)
    return round(base, 2)
