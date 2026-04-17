import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

from data import cache

logger = logging.getLogger(__name__)


@dataclass
class TickerData:
    symbol: str
    info: dict = field(default_factory=dict)
    fast_info: dict = field(default_factory=dict)
    history: Optional[pd.DataFrame] = None
    income_stmt: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None
    news: list = field(default_factory=list)
    upgrades_downgrades: Optional[pd.DataFrame] = None
    calendar: dict = field(default_factory=dict)
    recommendations_summary: Optional[pd.DataFrame] = None
    institutional_holders: Optional[pd.DataFrame] = None


def fetch(symbol: str, use_cache: bool = True) -> Optional[TickerData]:
    data = TickerData(symbol=symbol)
    try:
        ticker = yf.Ticker(symbol)

        # --- price history (1 year) ---
        hist = cache.get(symbol, "history") if use_cache else None
        if hist is None:
            hist = ticker.history(period="1y", auto_adjust=True)
            if hist is not None and not hist.empty and use_cache:
                cache.set(symbol, "history", hist)
        data.history = hist

        if hist is None or hist.empty or len(hist) < 30:
            logger.debug(f"{symbol}: insufficient price history")
            return None

        # --- fast_info (price/volume, no deprecated keys) ---
        fi_cached = cache.get(symbol, "fast_info") if use_cache else None
        if fi_cached is None:
            fi = ticker.fast_info
            fi_dict = {
                "market_cap": _safe(fi, "market_cap"),
                "last_price": _safe(fi, "last_price"),
                "last_volume": _safe(fi, "last_volume"),
                "year_high": _safe(fi, "year_high"),
                "year_low": _safe(fi, "year_low"),
                "fifty_day_average": _safe(fi, "fifty_day_average"),
                "two_hundred_day_average": _safe(fi, "two_hundred_day_average"),
                "three_month_average_volume": _safe(fi, "three_month_average_volume"),
                "ten_day_average_volume": _safe(fi, "ten_day_average_volume"),
                "quote_type": _safe(fi, "quote_type"),
            }
            if use_cache:
                cache.set(symbol, "fast_info", fi_dict)
            data.fast_info = fi_dict
        else:
            data.fast_info = fi_cached

        # --- ticker.info (fundamentals) ---
        info_cached = cache.get(symbol, "info", ttl=cache.TTL_FUNDAMENTAL) if use_cache else None
        if info_cached is None:
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            if use_cache:
                cache.set(symbol, "info", info)
            data.info = info
        else:
            data.info = info_cached

        # --- income statement (quarterly) ---
        inc_cached = cache.get(symbol, "income_stmt", ttl=cache.TTL_FUNDAMENTAL) if use_cache else None
        if inc_cached is None:
            try:
                inc = ticker.quarterly_income_stmt
            except Exception:
                inc = None
            if inc is not None and not inc.empty and use_cache:
                cache.set(symbol, "income_stmt", inc)
            data.income_stmt = inc
        else:
            data.income_stmt = inc_cached

        # --- balance sheet (quarterly) ---
        bs_cached = cache.get(symbol, "balance_sheet", ttl=cache.TTL_FUNDAMENTAL) if use_cache else None
        if bs_cached is None:
            try:
                bs = ticker.quarterly_balance_sheet
            except Exception:
                bs = None
            if bs is not None and not bs.empty and use_cache:
                cache.set(symbol, "balance_sheet", bs)
            data.balance_sheet = bs
        else:
            data.balance_sheet = bs_cached

        # --- cash flow (quarterly) ---
        cf_cached = cache.get(symbol, "cash_flow", ttl=cache.TTL_FUNDAMENTAL) if use_cache else None
        if cf_cached is None:
            try:
                cf = ticker.quarterly_cashflow
            except Exception:
                cf = None
            if cf is not None and not cf.empty and use_cache:
                cache.set(symbol, "cash_flow", cf)
            data.cash_flow = cf
        else:
            data.cash_flow = cf_cached

        # --- news ---
        news_cached = cache.get(symbol, "news") if use_cache else None
        if news_cached is None:
            try:
                news = ticker.news or []
            except Exception:
                news = []
            if use_cache:
                cache.set(symbol, "news", news)
            data.news = news
        else:
            data.news = news_cached

        # --- upgrades/downgrades ---
        try:
            ud = ticker.upgrades_downgrades
            data.upgrades_downgrades = ud
        except Exception:
            data.upgrades_downgrades = None

        # --- calendar ---
        try:
            cal = ticker.calendar or {}
            if isinstance(cal, pd.DataFrame):
                cal = cal.to_dict()
            data.calendar = cal
        except Exception:
            data.calendar = {}

        # --- recommendations summary ---
        try:
            rs = ticker.recommendations_summary
            data.recommendations_summary = rs if rs is not None and not rs.empty else None
        except Exception:
            data.recommendations_summary = None

        # --- institutional holders ---
        try:
            ih = ticker.institutional_holders
            data.institutional_holders = ih if ih is not None and not ih.empty else None
        except Exception:
            data.institutional_holders = None

        return data

    except Exception as e:
        logger.warning(f"{symbol}: fetch failed - {e}")
        return None


def _safe(obj, attr):
    try:
        val = getattr(obj, attr)
        return val if val == val else None  # filter NaN
    except Exception:
        return None
