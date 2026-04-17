import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from data.fetcher import TickerData
from scanner.models import CorrectionResult

logger = logging.getLogger(__name__)

_spy_cache: Optional[pd.Series] = None


def detect(data: TickerData, config: dict) -> Tuple[CorrectionResult, int]:
    """Returns (CorrectionResult, correction_days_estimate) for use by technical scorer."""
    cfg = config.get("correction", {})
    result = CorrectionResult(symbol=data.symbol, score=0.0)

    hist = data.history
    if hist is None or len(hist) < 30:
        return result, 0

    close = hist["Close"]
    volume = hist["Volume"]

    min_drop = cfg.get("min_drop_pct", 0.04)
    max_drop = cfg.get("max_drop_pct", 0.25)
    window = cfg.get("window_days", 30)

    # Use the last `window` trading days
    recent_close = close.iloc[-window:]
    recent_vol = volume.iloc[-window:]

    if len(recent_close) < 5:
        return result, 0

    # Find local peak within window
    peak_idx = recent_close.idxmax()
    peak_price = float(recent_close[peak_idx])
    current_price = float(close.iloc[-1])
    correction_pct = (peak_price - current_price) / peak_price

    result.correction_pct = round(correction_pct, 4)
    result.correction_start_date = str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx)

    if correction_pct < min_drop or correction_pct > max_drop:
        # Still compute partial score but flag as outside ideal range
        if correction_pct <= 0:
            return result, 0

    # Days from peak to now
    peak_pos = recent_close.index.get_loc(peak_idx)
    if isinstance(peak_pos, slice):
        peak_pos = peak_pos.stop - 1
    days_declining = len(recent_close) - peak_pos - 1
    result.days_declining = days_declining

    # Volume analysis during the decline
    post_peak_vol = recent_vol.iloc[peak_pos + 1:] if peak_pos + 1 < len(recent_vol) else pd.Series(dtype=float)
    prior_20d_vol = volume.iloc[-window - 20:-window] if len(volume) > window + 20 else volume.iloc[:20]

    avg_drop_vol = float(post_peak_vol.mean()) if not post_peak_vol.empty else float(volume.mean())
    avg_prior_vol = float(prior_20d_vol.mean()) if not prior_20d_vol.empty else float(volume.mean())

    result.avg_volume_during_drop = avg_drop_vol
    result.avg_volume_prior_20d = avg_prior_vol
    vol_ratio = avg_drop_vol / avg_prior_vol if avg_prior_vol > 0 else 1.0
    result.volume_ratio = round(vol_ratio, 3)

    # OBV trend during decline
    if len(post_peak_vol) >= 3:
        direction = close.iloc[-(len(post_peak_vol) + 1):].diff().iloc[1:]
        dir_sign = direction.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv_decline = (post_peak_vol * dir_sign.values).cumsum()
        x = np.arange(len(obv_decline))
        if len(x) >= 2:
            slope = np.polyfit(x, obv_decline.values.astype(float), 1)[0]
            result.obv_trend = "rising" if slope > 0 else ("declining" if slope < -abs(avg_prior_vol) * 0.05 else "stable")
        else:
            result.obv_trend = "stable"
    else:
        result.obv_trend = "stable"

    # RSI at current (correction bottom)
    rsi_series = _compute_rsi(close, 14)
    rsi_val = rsi_series.dropna().iloc[-1] if not rsi_series.dropna().empty else None
    result.rsi_at_bottom = float(rsi_val) if rsi_val is not None else None

    # Support level detection
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    swing_low = float(close.iloc[:-window].min()) if len(close) > window else None

    support_candidates = [v for v in [sma50, sma200, swing_low] if v and not np.isnan(v)]
    near = False
    best_support = None
    for s in support_candidates:
        if abs(current_price - s) / s <= 0.03:
            near = True
            best_support = s
            break
    result.is_near_support = near
    result.support_level = best_support

    # SPY correlation check
    spy_corr = _spy_correlation(hist, window=days_declining or 5)
    result.spy_correlation = spy_corr
    threshold = cfg.get("spy_correlation_macro_threshold", 0.65)
    result.is_macro_correlated = (spy_corr is not None and spy_corr >= threshold)

    # --- Scoring ---
    pts = 0.0

    # Correction depth (25 pts)
    if 0.08 <= correction_pct <= 0.15:
        pts += 25
    elif 0.05 <= correction_pct < 0.08:
        pts += 16
    elif 0.15 < correction_pct <= 0.20:
        pts += 14
    elif min_drop <= correction_pct < min_drop + 0.01:
        pts += 8
    elif correction_pct > 0.20:
        pts += 6

    # Volume declining during drop (30 pts) — key quality signal
    if vol_ratio < 0.70:
        pts += 30
    elif vol_ratio < 0.85:
        pts += 22
    elif vol_ratio < 1.00:
        pts += 14
    elif vol_ratio < 1.15:
        pts += 6
    # vol_ratio >= 1.15 = heavy selling = 0 pts

    # OBV trend (20 pts)
    if result.obv_trend == "rising":
        pts += 20
    elif result.obv_trend == "stable":
        pts += 12

    # RSI oversold (15 pts)
    if rsi_val is not None:
        if rsi_val <= 30:
            pts += 15
        elif rsi_val <= 40:
            pts += 11
        elif rsi_val <= 50:
            pts += 6

    # Near support (10 pts)
    if near:
        pts += 10

    result.score = min(100.0, round(pts, 1))
    return result, max(days_declining, 3)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _spy_correlation(hist: pd.DataFrame, window: int = 10) -> Optional[float]:
    global _spy_cache
    try:
        if _spy_cache is None or len(_spy_cache) < 30:
            spy = yf.download("SPY", period="3mo", progress=False, auto_adjust=True)
            if spy is not None and not spy.empty:
                _spy_cache = spy["Close"].pct_change().dropna()

        if _spy_cache is None or _spy_cache.empty:
            return None

        stock_ret = hist["Close"].pct_change().dropna().iloc[-window:]
        spy_ret = _spy_cache.reindex(stock_ret.index).dropna()
        aligned = stock_ret.reindex(spy_ret.index).dropna()

        if len(aligned) < 5:
            return None

        corr = float(aligned.corr(spy_ret.reindex(aligned.index)))
        return None if np.isnan(corr) else round(corr, 3)
    except Exception:
        return None
