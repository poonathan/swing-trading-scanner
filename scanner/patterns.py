"""
Chart pattern detection and scoring.

Entry point: score(data, config) → PatternResult
Each sub-detector returns a PatternMatch or None.
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from data.fetcher import TickerData
from scanner.support_resistance import SRLevel, find_sr_levels, nearest_resistance, nearest_support

logger = logging.getLogger(__name__)


# ── public types (imported into models.py) ───────────────────────────────────

class PatternMatch:
    __slots__ = ("name", "signal", "confidence", "description", "key_price")

    def __init__(self, name: str, signal: str, confidence: float, description: str, key_price: Optional[float] = None):
        self.name = name
        self.signal = signal          # "bullish" | "bearish"
        self.confidence = confidence  # 0–1
        self.description = description
        self.key_price = key_price    # breakout/breakdown level

    def __repr__(self):
        return f"PatternMatch({self.name}, {self.signal}, conf={self.confidence:.2f})"


# ── main entry ────────────────────────────────────────────────────────────────

def score(data: TickerData, config: dict) -> "PatternResult":
    from scanner.models import PatternResult  # late import to avoid circular

    cfg = config.get("patterns", {})
    result = PatternResult(symbol=data.symbol, score=50.0)

    hist = data.history
    if hist is None or len(hist) < 40:
        return result

    # --- S/R levels ---
    sr_levels = find_sr_levels(
        hist,
        n_levels=cfg.get("sr_n_levels", 6),
        tolerance_pct=cfg.get("sr_tolerance_pct", 0.02),
        pivot_order=cfg.get("pivot_order", 5),
    )
    result.sr_levels = sr_levels

    current_price = float(hist["Close"].iloc[-1])
    sup = nearest_support(current_price, sr_levels, max_pct=0.05)
    res = nearest_resistance(current_price, sr_levels, max_pct=0.10)
    result.nearest_support_price    = sup.price    if sup else None
    result.nearest_support_strength = sup.strength if sup else None
    result.nearest_resistance_price = res.price    if res else None

    # --- Pattern detection ---
    close  = hist["Close"].astype(float)
    high   = hist["High"].astype(float)
    low    = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    detectors = [
        _detect_double_bottom,
        _detect_bull_flag,
        _detect_falling_wedge,
        _detect_ascending_triangle,
        _detect_cup_handle,
        _detect_inverse_head_shoulders,
        _detect_head_shoulders,      # bearish — penalizes score
        _detect_descending_triangle, # bearish
    ]

    patterns: List[PatternMatch] = []
    for fn in detectors:
        try:
            m = fn(close, high, low, volume)
            if m is not None:
                patterns.append(m)
        except Exception as e:
            logger.debug(f"{data.symbol}: {fn.__name__} failed: {e}")

    result.patterns_detected = patterns
    result.pattern_count = len(patterns)

    # --- Composite pattern score (0–100, 50 = neutral) ---
    bullish_pts = sum(p.confidence * 100 for p in patterns if p.signal == "bullish")
    bearish_pts = sum(p.confidence * 100 for p in patterns if p.signal == "bearish")
    # bearish patterns penalise at 60% weight (swing scanner looks for dips to buy)
    net = bullish_pts - bearish_pts * 0.6
    raw_score = 50.0 + net * 0.45
    result.score = max(0.0, min(100.0, round(raw_score, 1)))

    # S/R bonus: near strong support = +8, near resistance close = −5
    if sup and sup.strength >= 60:
        result.score = min(100.0, result.score + 8.0)
    if res and (res.price - current_price) / current_price < 0.03:
        result.score = max(0.0, result.score - 5.0)

    # Top (highest-confidence bullish) pattern
    bullish = [p for p in patterns if p.signal == "bullish"]
    if bullish:
        top = max(bullish, key=lambda p: p.confidence)
        result.has_bullish_pattern       = True
        result.top_pattern_name          = top.name
        result.top_pattern_confidence    = top.confidence
        result.top_pattern_key_price     = top.key_price

    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_local_min_idx(series: pd.Series, order: int = 5) -> List[int]:
    arr = series.values.astype(float)
    n = len(arr)
    return [
        i for i in range(order, n - order)
        if arr[i] == min(arr[i - order: i + order + 1])
    ]


def _find_local_max_idx(series: pd.Series, order: int = 5) -> List[int]:
    arr = series.values.astype(float)
    n = len(arr)
    return [
        i for i in range(order, n - order)
        if arr[i] == max(arr[i - order: i + order + 1])
    ]


# ── bullish patterns ──────────────────────────────────────────────────────────

def _detect_double_bottom(close, high, low, volume) -> Optional[PatternMatch]:
    """Two roughly equal lows separated by a rally. Bullish reversal (W-pattern)."""
    lookback = min(len(close), 90)
    c = close.iloc[-lookback:].reset_index(drop=True)
    l = low.iloc[-lookback:].reset_index(drop=True)

    idxs = _find_local_min_idx(l, order=5)
    if len(idxs) < 2:
        return None

    i1, i2 = int(idxs[-2]), int(idxs[-1])
    p1, p2 = float(l.iloc[i1]), float(l.iloc[i2])

    if i2 - i1 < 10:
        return None

    avg_low = (p1 + p2) / 2
    if avg_low <= 0 or abs(p1 - p2) / avg_low > 0.03:
        return None

    peak_between = float(c.iloc[i1:i2].max())
    rally_pct = (peak_between - avg_low) / avg_low
    if rally_pct < 0.03:
        return None

    neckline = peak_between
    current_price = float(c.iloc[-1])
    pct_to_neckline = max(0.0, (neckline - current_price) / neckline)

    symmetry = 1.0 - (abs(p1 - p2) / avg_low) / 0.03
    rally_sc = min(1.0, rally_pct / 0.08)
    prox_sc  = max(0.0, 1.0 - pct_to_neckline / 0.05) if pct_to_neckline > 0 else 0.8

    confidence = round(symmetry * 0.35 + rally_sc * 0.35 + prox_sc * 0.30, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Double Bottom",
        signal="bullish",
        confidence=confidence,
        description=f"W-pattern: lows ~${avg_low:.2f}, neckline ${neckline:.2f} ({pct_to_neckline:.1%} to breakout)",
        key_price=round(neckline, 2),
    )


def _detect_bull_flag(close, high, low, volume) -> Optional[PatternMatch]:
    """Strong up-move (pole) followed by tight consolidation (flag)."""
    n = len(close)
    pole_w, flag_w = 20, 15
    if n < pole_w + flag_w:
        return None

    pole_start = float(close.iloc[-(pole_w + flag_w)])
    pole_end   = float(close.iloc[-flag_w])
    if pole_start <= 0:
        return None
    pole_gain = (pole_end - pole_start) / pole_start
    if pole_gain < 0.07:
        return None

    flag = close.iloc[-flag_w:].astype(float)
    flag_high = float(flag.max())
    flag_low  = float(flag.min())
    if flag_high <= 0:
        return None
    flag_range = (flag_high - flag_low) / flag_high
    if flag_range > 0.08:
        return None

    # Flag should drift flat or slightly down (not continue up)
    slope = float(np.polyfit(range(flag_w), flag.values, 1)[0])
    drift = slope * flag_w / float(flag.iloc[0]) if float(flag.iloc[0]) > 0 else 0
    if drift > 0.02:
        return None

    vol_pole = float(volume.iloc[-(pole_w + flag_w):-flag_w].mean())
    vol_flag = float(volume.iloc[-flag_w:].mean())
    vol_contraction = vol_pole > 0 and vol_flag < vol_pole * 0.85

    base_sc    = min(1.0, pole_gain / 0.15)
    channel_sc = max(0.0, 1.0 - flag_range / 0.08)
    vol_bonus  = 0.15 if vol_contraction else 0.0

    confidence = round(base_sc * 0.50 + channel_sc * 0.35 + vol_bonus, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Bull Flag",
        signal="bullish",
        confidence=confidence,
        description=f"Pole +{pole_gain:.1%}, flag range {flag_range:.1%}, breakout above ${flag_high:.2f}",
        key_price=round(flag_high, 2),
    )


def _detect_falling_wedge(close, high, low, volume) -> Optional[PatternMatch]:
    """Two converging downward trendlines. Bullish reversal or continuation."""
    lookback = min(len(close), 60)
    if lookback < 20:
        return None

    h = high.iloc[-lookback:].astype(float).values
    l = low.iloc[-lookback:].astype(float).values
    c = close.iloc[-lookback:].astype(float)

    x = np.arange(lookback, dtype=float)
    h_slope = float(np.polyfit(x, h, 1)[0])
    l_slope = float(np.polyfit(x, l, 1)[0])

    if h_slope >= 0 or l_slope >= 0:
        return None
    # Lows must fall slower than highs (converging upward)
    if l_slope >= h_slope:
        return None

    h_mean = float(np.mean(h))
    if h_mean <= 0:
        return None
    convergence = (h_slope - l_slope) / abs(h_slope)
    if convergence < 0.05:
        return None

    price_range = float(np.mean(h) - np.mean(l))
    current_price = float(c.iloc[-1])
    l_line_end = float(np.polyval(np.polyfit(x, l, 1), lookback - 1))
    h_line_end = float(np.polyval(np.polyfit(x, h, 1), lookback - 1))

    prox = abs(current_price - l_line_end) / price_range if price_range > 0 else 1.0
    total_drop = (float(h[-1]) - float(h[0])) / float(h[0]) if float(h[0]) > 0 else 0

    base_sc  = min(1.0, convergence * 3.0)
    prox_bon = max(0.0, 0.30 - prox * 0.30)
    drop_sc  = min(0.30, abs(total_drop) * 2.0)

    confidence = round(base_sc * 0.50 + prox_bon + drop_sc, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Falling Wedge",
        signal="bullish",
        confidence=confidence,
        description=f"Converging downtrend lines, breakout above ${h_line_end:.2f}",
        key_price=round(h_line_end, 2),
    )


def _detect_ascending_triangle(close, high, low, volume) -> Optional[PatternMatch]:
    """Flat resistance ceiling + rising support lows = bullish breakout setup."""
    lookback = min(len(close), 60)
    if lookback < 20:
        return None

    h = high.iloc[-lookback:].astype(float).values
    l = low.iloc[-lookback:].astype(float).values
    x = np.arange(lookback, dtype=float)

    h_slope = float(np.polyfit(x, h, 1)[0])
    l_slope = float(np.polyfit(x, l, 1)[0])
    h_mean  = float(np.mean(h))
    l_mean  = float(np.mean(l))

    if h_mean <= 0 or l_mean <= 0:
        return None

    h_slope_norm = h_slope / h_mean * lookback
    l_slope_norm = l_slope / l_mean * lookback

    if abs(h_slope_norm) > 0.04:  # highs not flat enough
        return None
    if l_slope_norm < 0.03:       # lows not rising
        return None

    resistance = float(np.mean(h))
    current_price = float(close.iloc[-1])
    pct_to_break = (resistance - current_price) / resistance
    if pct_to_break < 0:
        return None

    flat_sc   = max(0.0, 1.0 - abs(h_slope_norm) / 0.04)
    rising_sc = min(1.0, l_slope_norm / 0.06)
    prox_sc   = max(0.0, 1.0 - pct_to_break / 0.05)

    confidence = round(flat_sc * 0.35 + rising_sc * 0.35 + prox_sc * 0.30, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Ascending Triangle",
        signal="bullish",
        confidence=confidence,
        description=f"Flat resistance ~${resistance:.2f}, rising lows. Breakout {pct_to_break:.1%} away",
        key_price=round(resistance, 2),
    )


def _detect_cup_handle(close, high, low, volume) -> Optional[PatternMatch]:
    """Rounded U-base followed by a small handle pullback. Bullish continuation."""
    n = len(close)
    lookback = min(n, 120)
    if lookback < 50:
        return None

    c = close.iloc[-lookback:].astype(float).reset_index(drop=True)
    t = lookback // 3

    left   = float(c.iloc[:t].mean())
    bottom = float(c.iloc[t:2*t].mean())
    right  = float(c.iloc[2*t:3*t].mean())

    if min(left, right) <= 0:
        return None

    cup_depth    = (min(left, right) - bottom) / min(left, right)
    rim_symmetry = abs(left - right) / min(left, right)

    if cup_depth < 0.05 or rim_symmetry > 0.10:
        return None

    handle_len = max(5, min(15, lookback // 5))
    handle = c.iloc[-handle_len:]
    h_max  = float(handle.max())
    if h_max <= 0:
        return None
    handle_drop = (h_max - float(handle.min())) / h_max
    if handle_drop > 0.06:
        return None

    rim_price = max(left, right)
    current_price = float(c.iloc[-1])
    pct_to_rim = (rim_price - current_price) / rim_price
    if pct_to_rim > 0.05:
        return None

    depth_sc  = min(1.0, cup_depth / 0.15)
    sym_sc    = max(0.0, 1.0 - rim_symmetry / 0.10)
    prox_sc   = max(0.0, 1.0 - pct_to_rim / 0.05)
    handle_sc = max(0.0, 1.0 - handle_drop / 0.06)

    confidence = round((depth_sc + sym_sc + prox_sc + handle_sc) / 4.0, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Cup & Handle",
        signal="bullish",
        confidence=confidence,
        description=f"U-base depth {cup_depth:.1%}, handle {handle_drop:.1%} pullback, rim breakout ${rim_price:.2f}",
        key_price=round(rim_price, 2),
    )


def _detect_inverse_head_shoulders(close, high, low, volume) -> Optional[PatternMatch]:
    """Three lows: shoulders equal, head lowest. Neckline break = bullish reversal."""
    lookback = min(len(close), 90)
    if lookback < 40:
        return None

    l = low.iloc[-lookback:].astype(float).reset_index(drop=True)
    c = close.iloc[-lookback:].astype(float).reset_index(drop=True)

    idxs = _find_local_min_idx(l, order=5)
    if len(idxs) < 3:
        return None

    i_ls, i_h, i_rs = int(idxs[-3]), int(idxs[-2]), int(idxs[-1])
    ls, head, rs = float(l.iloc[i_ls]), float(l.iloc[i_h]), float(l.iloc[i_rs])

    if not (head < ls and head < rs):
        return None

    shoulder_avg = (ls + rs) / 2
    if shoulder_avg <= 0 or abs(ls - rs) / shoulder_avg > 0.04:
        return None

    head_depth = (shoulder_avg - head) / shoulder_avg
    if head_depth < 0.03:
        return None

    left_peak  = float(c.iloc[i_ls:i_h].max()) if i_h > i_ls + 1 else shoulder_avg
    right_peak = float(c.iloc[i_h:i_rs].max()) if i_rs > i_h + 1 else shoulder_avg
    neckline   = (left_peak + right_peak) / 2

    current_price   = float(c.iloc[-1])
    pct_to_neckline = max(0.0, (neckline - current_price) / neckline)

    sym_sc   = 1.0 - (abs(ls - rs) / shoulder_avg) / 0.04
    depth_sc = min(1.0, head_depth / 0.08)
    prox_sc  = max(0.0, 1.0 - pct_to_neckline / 0.05) if pct_to_neckline > 0 else 0.7

    confidence = round(sym_sc * 0.35 + depth_sc * 0.35 + prox_sc * 0.30, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Inv. Head & Shoulders",
        signal="bullish",
        confidence=confidence,
        description=f"Shoulders ~${shoulder_avg:.2f}, head ${head:.2f} (depth {head_depth:.1%}), neckline ${neckline:.2f}",
        key_price=round(neckline, 2),
    )


# ── bearish patterns ──────────────────────────────────────────────────────────

def _detect_head_shoulders(close, high, low, volume) -> Optional[PatternMatch]:
    """Three highs: shoulders equal, head highest. Bearish reversal."""
    lookback = min(len(close), 90)
    if lookback < 40:
        return None

    h = high.iloc[-lookback:].astype(float).reset_index(drop=True)
    c = close.iloc[-lookback:].astype(float).reset_index(drop=True)

    idxs = _find_local_max_idx(h, order=5)
    if len(idxs) < 3:
        return None

    i_ls, i_h, i_rs = int(idxs[-3]), int(idxs[-2]), int(idxs[-1])
    ls, head, rs = float(h.iloc[i_ls]), float(h.iloc[i_h]), float(h.iloc[i_rs])

    if not (head > ls and head > rs):
        return None

    shoulder_avg = (ls + rs) / 2
    if shoulder_avg <= 0 or abs(ls - rs) / shoulder_avg > 0.04:
        return None

    head_height = (head - shoulder_avg) / shoulder_avg
    if head_height < 0.03:
        return None

    left_trough  = float(c.iloc[i_ls:i_h].min()) if i_h > i_ls + 1 else shoulder_avg
    right_trough = float(c.iloc[i_h:i_rs].min()) if i_rs > i_h + 1 else shoulder_avg
    neckline = (left_trough + right_trough) / 2

    current_price = float(c.iloc[-1])
    pct_below_neckline = (neckline - current_price) / neckline

    sym_sc    = 1.0 - (abs(ls - rs) / shoulder_avg) / 0.04
    height_sc = min(1.0, head_height / 0.08)
    # More bearish confidence if price already near/below neckline
    break_sc  = min(1.0, max(0.0, pct_below_neckline * 20 + 0.5))

    confidence = round(sym_sc * 0.35 + height_sc * 0.35 + break_sc * 0.30, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Head & Shoulders",
        signal="bearish",
        confidence=confidence,
        description=f"Shoulders ~${shoulder_avg:.2f}, head ${head:.2f}, neckline ${neckline:.2f}",
        key_price=round(neckline, 2),
    )


def _detect_descending_triangle(close, high, low, volume) -> Optional[PatternMatch]:
    """Flat support floor + declining resistance highs. Bearish breakdown setup."""
    lookback = min(len(close), 60)
    if lookback < 20:
        return None

    h = high.iloc[-lookback:].astype(float).values
    l = low.iloc[-lookback:].astype(float).values
    x = np.arange(lookback, dtype=float)

    h_slope = float(np.polyfit(x, h, 1)[0])
    l_slope = float(np.polyfit(x, l, 1)[0])
    l_mean  = float(np.mean(l))

    if l_mean <= 0:
        return None
    l_slope_norm = l_slope / l_mean * lookback

    if abs(l_slope_norm) > 0.03:  # lows not flat
        return None
    h_mean = float(np.mean(h))
    if h_mean <= 0:
        return None
    h_slope_norm = h_slope / h_mean * lookback
    if h_slope_norm > -0.02:      # highs not declining
        return None

    support = float(np.mean(l))
    current_price = float(close.iloc[-1])
    pct_to_support = (current_price - support) / current_price if current_price > 0 else 1.0

    flat_sc     = max(0.0, 1.0 - abs(l_slope_norm) / 0.03)
    decline_sc  = min(1.0, abs(h_slope_norm) / 0.06)
    prox_sc     = max(0.0, 1.0 - pct_to_support / 0.05)

    confidence = round(flat_sc * 0.35 + decline_sc * 0.35 + prox_sc * 0.30, 2)
    confidence = max(0.0, min(1.0, confidence))
    if confidence < 0.35:
        return None

    return PatternMatch(
        name="Descending Triangle",
        signal="bearish",
        confidence=confidence,
        description=f"Flat support ~${support:.2f}, declining highs. Watch breakdown below ${support:.2f}",
        key_price=round(support, 2),
    )
