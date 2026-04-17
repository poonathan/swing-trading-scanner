import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SRLevel:
    price: float
    level_type: str       # "support" | "resistance" | "both"
    strength: float       # 0–100; higher = stronger / more confluent
    touches: int
    fib_label: Optional[str] = None   # e.g. "61.8%" when from Fibonacci


def find_sr_levels(
    hist: pd.DataFrame,
    n_levels: int = 6,
    tolerance_pct: float = 0.02,
    pivot_order: int = 5,
) -> List[SRLevel]:
    """Return up to n_levels support/resistance levels from pivot clustering + Fibonacci."""
    if hist is None or len(hist) < 30:
        return []

    close = hist["Close"].astype(float)
    high  = hist["High"].astype(float)
    low   = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    pivot_highs = _find_pivots(high, order=pivot_order, mode="high")
    pivot_lows  = _find_pivots(low,  order=pivot_order, mode="low")

    raw: List[Tuple[float, str]] = (
        [(p, "resistance") for p in pivot_highs] +
        [(p, "support")    for p in pivot_lows]
    )
    levels = _cluster_levels(raw, close, high, low, volume, tolerance_pct)

    lookback = min(len(high), 90)
    recent_high = float(high.iloc[-lookback:].max())
    recent_low  = float(low.iloc[-lookback:].min())
    levels += _fibonacci_levels(recent_low, recent_high, close, tolerance_pct)

    levels = _deduplicate(levels, tolerance_pct)
    levels.sort(key=lambda x: -x.strength)
    return levels[:n_levels]


def nearest_support(price: float, levels: List[SRLevel], max_pct: float = 0.05) -> Optional[SRLevel]:
    candidates = [
        lv for lv in levels
        if lv.level_type in ("support", "both")
        and lv.price < price
        and (price - lv.price) / price <= max_pct
    ]
    return max(candidates, key=lambda x: x.price) if candidates else None


def nearest_resistance(price: float, levels: List[SRLevel], max_pct: float = 0.10) -> Optional[SRLevel]:
    candidates = [
        lv for lv in levels
        if lv.level_type in ("resistance", "both")
        and lv.price > price
        and (lv.price - price) / price <= max_pct
    ]
    return min(candidates, key=lambda x: x.price) if candidates else None


# ── internals ────────────────────────────────────────────────────────────────

def _find_pivots(series: pd.Series, order: int, mode: str) -> List[float]:
    arr = series.values.astype(float)
    n = len(arr)
    out = []
    for i in range(order, n - order):
        window = arr[i - order: i + order + 1]
        val = arr[i]
        if mode == "high" and val == window.max():
            out.append(float(val))
        elif mode == "low" and val == window.min():
            out.append(float(val))
    return out


def _cluster_levels(
    raw: List[Tuple[float, str]],
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    tol: float,
) -> List[SRLevel]:
    if not raw:
        return []

    current_price = float(close.iloc[-1])
    avg_vol = float(volume.mean())
    clusters: List[List[Tuple[float, str]]] = []

    for price, ltype in sorted(raw, key=lambda x: x[0]):
        placed = False
        for cluster in clusters:
            ref = cluster[0][0]
            if ref > 0 and abs(price - ref) / ref <= tol:
                cluster.append((price, ltype))
                placed = True
                break
        if not placed:
            clusters.append([(price, ltype)])

    result = []
    all_prices = pd.concat([high, low])

    for cluster in clusters:
        prices = [p for p, _ in cluster]
        types  = [t for _, t in cluster]
        avg_price = float(np.mean(prices))
        if avg_price <= 0:
            continue

        # Classify by position relative to current price, not pivot direction.
        # A past resistance pivot below the current price now acts as support,
        # and a past support pivot above acts as resistance.
        margin = 0.01  # within 1% of current price = "at price" → both
        if avg_price < current_price * (1 - margin):
            ltype = "support"
        elif avg_price > current_price * (1 + margin):
            ltype = "resistance"
        else:
            ltype = "both"

        touch_count = int(((all_prices - avg_price).abs() / avg_price < tol).sum())

        # Recency bonus: level touched within last 30 bars
        recent_h = high.iloc[-30:] if len(high) >= 30 else high
        recent_l = low.iloc[-30:]  if len(low)  >= 30 else low
        recency_bonus = 20.0 if (
            ((recent_h - avg_price).abs() / avg_price < tol).any() or
            ((recent_l - avg_price).abs() / avg_price < tol).any()
        ) else 0.0

        # Volume at level
        mask = ((high - avg_price).abs() / avg_price < tol) | ((low - avg_price).abs() / avg_price < tol)
        vol_at_level = float(volume[mask].mean()) if mask.any() else avg_vol
        vol_score = min(30.0, (vol_at_level / avg_vol) * 15.0) if avg_vol > 0 and not np.isnan(vol_at_level) else 0.0

        pct_away = abs(avg_price - current_price) / current_price
        proximity_score = max(0.0, 20.0 - pct_away * 200.0)

        strength = min(100.0, touch_count * 8.0 + recency_bonus + vol_score + proximity_score)
        result.append(SRLevel(
            price=round(avg_price, 2),
            level_type=ltype,
            strength=round(strength, 1),
            touches=touch_count,
        ))
    return result


def _fibonacci_levels(low: float, high: float, close: pd.Series, tol: float) -> List[SRLevel]:
    if high <= low:
        return []
    current_price = float(close.iloc[-1])
    diff = high - low
    ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    labels = ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]
    result = []
    for ratio, label in zip(ratios, labels):
        fib_price = high - diff * ratio
        pct_away = abs(fib_price - current_price) / current_price
        if pct_away > 0.20:
            continue
        base = 55.0 if ratio in (0.382, 0.500, 0.618) else 35.0
        proximity_bonus = max(0.0, 20.0 - pct_away * 150.0)
        strength = min(100.0, base + proximity_bonus)
        ltype = "support" if fib_price < current_price else "resistance"
        result.append(SRLevel(
            price=round(fib_price, 2),
            level_type=ltype,
            strength=round(strength, 1),
            touches=1,
            fib_label=label,
        ))
    return result


def _deduplicate(levels: List[SRLevel], tol: float) -> List[SRLevel]:
    result: List[SRLevel] = []
    for lv in sorted(levels, key=lambda x: -x.strength):
        if not any(
            existing.price > 0 and abs(lv.price - existing.price) / existing.price < tol
            for existing in result
        ):
            result.append(lv)
    return result
