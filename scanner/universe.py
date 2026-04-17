import logging
from typing import List

import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

_POPULAR_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "JPM", "V", "MA", "UNH", "JNJ", "XOM", "PG", "HD", "COST",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT",
    "LRCX", "KLAC", "MRVL", "ON", "SMCI", "ARM",
    "PYPL", "SQ", "SHOP", "SNOW", "DDOG", "ZS", "CRWD", "NET",
    "NOW", "PANW", "OKTA", "MDB", "TEAM",
    "LLY", "ABBV", "BMY", "AMGN", "GILD", "REGN", "MRNA", "BIIB",
    "BA", "CAT", "HON", "GE", "RTX", "LMT",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    "WMT", "TGT", "NKE", "SBUX", "MCD",
]


def load_symbols(
    symbols: str = None,
    watchlist_file: str = None,
    sp500: bool = False,
    nasdaq100: bool = False,
    default: bool = False,
) -> List[str]:
    result = []

    if symbols:
        result = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    elif watchlist_file:
        try:
            with open(watchlist_file) as f:
                result = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
        except FileNotFoundError:
            logger.error(f"Watchlist file not found: {watchlist_file}")

    elif sp500:
        result = _scrape_sp500()

    elif nasdaq100:
        result = _scrape_nasdaq100()

    else:
        result = list(_POPULAR_WATCHLIST)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for s in result:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    logger.info(f"Universe: {len(deduped)} symbols loaded")
    return deduped


def pre_filter(symbols: List[str], config: dict) -> List[str]:
    """Quick filter using fast_info to drop illiquid/non-equity tickers."""
    cfg = config.get("fundamental", {})
    min_cap = cfg.get("min_market_cap", 500_000_000)
    passed = []

    for sym in symbols:
        try:
            fi = yf.Ticker(sym).fast_info
            qt = getattr(fi, "quote_type", None)
            if qt and qt.upper() not in ("EQUITY", ""):
                logger.debug(f"{sym}: skipped (quote_type={qt})")
                continue
            cap = getattr(fi, "market_cap", None)
            if cap and cap < min_cap:
                logger.debug(f"{sym}: skipped (market_cap too small)")
                continue
            vol = getattr(fi, "three_month_average_volume", None)
            if vol and vol < 100_000:
                logger.debug(f"{sym}: skipped (low volume)")
                continue
            passed.append(sym)
        except Exception as e:
            logger.debug(f"{sym}: pre-filter error - {e}")
            passed.append(sym)  # Give benefit of doubt on errors

    logger.info(f"Pre-filter: {len(passed)}/{len(symbols)} passed")
    return passed


def _scrape_sp500() -> List[str]:
    try:
        resp = requests.get(_SP500_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            table = soup.find("table", class_="wikitable")
        if table is None:
            logger.error("Could not find S&P 500 table on Wikipedia")
            return list(_POPULAR_WATCHLIST)
        symbols = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                sym = cols[0].get_text(strip=True).replace(".", "-")
                symbols.append(sym)
        logger.info(f"S&P 500: scraped {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.error(f"S&P 500 scrape failed: {e}")
        return list(_POPULAR_WATCHLIST)


def _scrape_nasdaq100() -> List[str]:
    try:
        resp = requests.get(_NDX100_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        # Wikipedia Nasdaq-100 page has a table with id "constituents"
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            # Fallback: find wikitable with "Ticker" in header
            for t in soup.find_all("table", class_="wikitable"):
                headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
                if any("ticker" in h or "symbol" in h for h in headers):
                    table = t
                    break
        if table is None:
            logger.error("Could not find Nasdaq-100 table on Wikipedia")
            return list(_POPULAR_WATCHLIST)
        symbols = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                # Ticker is usually first or second column depending on page layout
                for col in cols[:2]:
                    text = col.get_text(strip=True).replace(".", "-")
                    # Valid ticker: 1-5 uppercase letters/digits/hyphen
                    if text and 1 <= len(text) <= 6 and text.replace("-", "").isalpha():
                        symbols.append(text.upper())
                        break
        symbols = list(dict.fromkeys(symbols))  # deduplicate preserving order
        logger.info(f"Nasdaq-100: scraped {len(symbols)} symbols")
        return symbols if len(symbols) >= 50 else list(_POPULAR_WATCHLIST)
    except Exception as e:
        logger.error(f"Nasdaq-100 scrape failed: {e}")
        return list(_POPULAR_WATCHLIST)
