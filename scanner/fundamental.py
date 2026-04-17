import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.fetcher import TickerData
from scanner.models import FundamentalResult

logger = logging.getLogger(__name__)


def score(data: TickerData, config: dict) -> FundamentalResult:
    cfg = config.get("fundamental", {})
    min_score = config.get("scoring", {}).get("min_fundamental_score", 45)
    info = data.info
    fi = data.fast_info

    result = FundamentalResult(symbol=data.symbol, score=0.0)

    # Liquidity gate
    market_cap = fi.get("market_cap")
    if market_cap and market_cap < cfg.get("min_market_cap", 500_000_000):
        result.fail_reason = f"Market cap too small: ${market_cap:,.0f}"
        return result
    result.market_cap = market_cap

    pts = 0.0

    # --- ROE (20 pts) ---
    roe = _get(info, "returnOnEquity")
    result.return_on_equity = roe
    if roe is not None:
        if roe >= 0.25:
            pts += 20
        elif roe >= 0.15:
            pts += 15
        elif roe >= 0.08:
            pts += 10
        elif roe >= 0.0:
            pts += 4

    # --- Debt/Equity + Current Ratio (20 pts) ---
    de = _get(info, "debtToEquity")
    cr = _get(info, "currentRatio")
    result.debt_to_equity = de
    result.current_ratio = cr
    max_de = cfg.get("max_debt_to_equity", 3.0)
    if de is not None:
        de_norm = de / 100.0 if de > 10 else de  # yfinance sometimes returns as percentage
        if de_norm < 0.5:
            pts += 15
        elif de_norm < 1.0:
            pts += 11
        elif de_norm < max_de:
            pts += 6
    else:
        pts += 8  # unknown debt: give partial credit
    if cr is not None:
        if cr >= 2.0:
            pts += 5
        elif cr >= 1.0:
            pts += 3

    # --- Revenue trend from quarterly income stmt (20 pts) ---
    rev_pts, rev_growth, rev_trend = _revenue_score(data.income_stmt)
    pts += rev_pts
    result.revenue_growth_yoy = rev_growth
    result.revenue_trend = rev_trend

    # --- Free Cash Flow (15 pts) ---
    fcf = _get(info, "freeCashflow")
    ocf = _get(info, "operatingCashflow")
    result.free_cash_flow = fcf
    result.operating_cashflow = ocf
    if fcf is not None:
        if fcf > 0:
            pts += 10
            rev_val = _get(info, "totalRevenue") or 1
            if rev_val and fcf / rev_val > 0.05:
                pts += 5
    elif ocf is not None and ocf > 0:
        pts += 7

    # --- Earnings quality (15 pts) ---
    eg = _get(info, "earningsGrowth")
    result.earnings_growth = eg
    if eg is not None:
        if eg >= 0.20:
            pts += 15
        elif eg >= 0.10:
            pts += 10
        elif eg >= 0.0:
            pts += 5

    # --- Valuation sanity (10 pts) ---
    pe = _get(info, "trailingPE")
    fpe = _get(info, "forwardPE")
    peg = _get(info, "trailingPegRatio")
    result.pe_ratio = pe
    result.forward_pe = fpe
    result.peg_ratio = peg
    max_pe = cfg.get("max_pe_ratio", 60)
    if pe is not None and pe > 0:
        if pe <= 20:
            pts += 10
        elif pe <= 35:
            pts += 7
        elif pe <= max_pe:
            pts += 3
    elif fpe is not None and fpe > 0:
        if fpe <= 20:
            pts += 8
        elif fpe <= 35:
            pts += 5

    result.score = min(100.0, round(pts, 1))
    result.pass_gate = result.score >= min_score
    if not result.pass_gate and not result.fail_reason:
        result.fail_reason = f"Score {result.score:.0f} below gate {min_score}"
    return result


def _revenue_score(income_stmt: Optional[pd.DataFrame]):
    if income_stmt is None or income_stmt.empty:
        return 8, None, "unknown"
    try:
        row = None
        for label in ["Total Revenue", "TotalRevenue", "Revenue"]:
            if label in income_stmt.index:
                row = income_stmt.loc[label]
                break
        if row is None or len(row) < 2:
            return 8, None, "unknown"

        row = row.dropna()
        if len(row) < 2:
            return 8, None, "unknown"

        # YoY: compare most recent quarter to same quarter 1 year ago
        yoy = None
        if len(row) >= 5:
            v_now = float(row.iloc[0])
            v_year_ago = float(row.iloc[4])
            if v_year_ago and v_year_ago != 0:
                yoy = (v_now - v_year_ago) / abs(v_year_ago)

        # QoQ trend: 3 consecutive quarters
        trend_pts = 0
        growth_quarters = sum(
            1 for i in range(min(3, len(row) - 1))
            if float(row.iloc[i]) > float(row.iloc[i + 1])
        )
        if growth_quarters == 3:
            trend_pts = 20
            trend = "growing"
        elif growth_quarters == 2:
            trend_pts = 13
            trend = "growing"
        elif growth_quarters == 1:
            trend_pts = 7
            trend = "stable"
        else:
            trend_pts = 2
            trend = "declining"

        # Blend with YoY if available
        if yoy is not None:
            if yoy >= 0.15:
                trend_pts = max(trend_pts, 18)
            elif yoy >= 0.05:
                trend_pts = max(trend_pts, 13)
            elif yoy >= -0.05:
                trend_pts = max(trend_pts, 8)

        return trend_pts, yoy, trend
    except Exception:
        return 8, None, "unknown"


def _get(d: dict, key: str):
    val = d.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None
