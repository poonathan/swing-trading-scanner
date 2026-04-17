import csv
import json
import os
from datetime import datetime
from typing import List

from scanner.models import CompositeResult


def export_csv(results: List[CompositeResult], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "symbol", "signal", "composite_score",
        "fund_score", "tech_score", "corr_score", "news_score",
        "current_price", "year_high", "pct_from_high",
        "correction_pct", "volume_ratio", "obv_trend", "rsi_at_bottom",
        "event_type", "spy_correlation", "is_macro_correlated",
        "roe", "debt_to_equity", "revenue_growth", "earnings_growth",
        "was_in_uptrend", "adx", "near_support",
        "sector", "industry", "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(_flatten(r))
    return path


def export_json(results: List[CompositeResult], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "generated_at": datetime.now().isoformat(),
        "count": len(results),
        "results": [_flatten(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def _flatten(r: CompositeResult) -> dict:
    return {
        "symbol": r.symbol,
        "signal": r.signal,
        "composite_score": r.composite_score,
        "fund_score": r.fundamental.score if r.fundamental else None,
        "tech_score": r.technical.score if r.technical else None,
        "corr_score": r.correction.score if r.correction else None,
        "news_score": r.news.score if r.news else None,
        "current_price": r.current_price,
        "year_high": r.year_high,
        "pct_from_high": round(r.pct_from_high * 100, 1) if r.pct_from_high else None,
        "correction_pct": round(r.correction.correction_pct * 100, 1) if r.correction else None,
        "volume_ratio": r.correction.volume_ratio if r.correction else None,
        "obv_trend": r.correction.obv_trend if r.correction else None,
        "rsi_at_bottom": r.correction.rsi_at_bottom if r.correction else None,
        "event_type": r.news.event_type if r.news else None,
        "spy_correlation": r.correction.spy_correlation if r.correction else None,
        "is_macro_correlated": r.correction.is_macro_correlated if r.correction else None,
        "roe": round(r.fundamental.return_on_equity * 100, 1) if r.fundamental and r.fundamental.return_on_equity else None,
        "debt_to_equity": r.fundamental.debt_to_equity if r.fundamental else None,
        "revenue_growth": round(r.fundamental.revenue_growth_yoy * 100, 1) if r.fundamental and r.fundamental.revenue_growth_yoy else None,
        "earnings_growth": round(r.fundamental.earnings_growth * 100, 1) if r.fundamental and r.fundamental.earnings_growth else None,
        "was_in_uptrend": r.technical.was_in_uptrend if r.technical else None,
        "adx": r.technical.adx_value if r.technical else None,
        "near_support": r.correction.is_near_support if r.correction else None,
        "sector": r.sector,
        "industry": r.industry,
        "reason": r.reason,
    }
