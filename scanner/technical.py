import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from data.fetcher import TickerData
from scanner.models import TechnicalResult

logger = logging.getLogger(__name__)


def score(data: TickerData, config: dict, correction_days: int = 10) -> TechnicalResult:
    cfg = config.get("technical", {})
    result = TechnicalResult(symbol=data.symbol, score=0.0)

    hist = data.history
    if hist is None or len(hist) < 60:
        return result

    # Work on the pre-correction window
    pre = hist.iloc[:-correction_days] if correction_days > 0 and len(hist) > correction_days + 20 else hist
    if len(pre) < 50:
        return result

    close = pre["Close"]
    high = pre["High"]
    low = pre["Low"]
    volume = pre["Volume"]

    pts = 0.0

    # --- SMA checks at the start of correction ---
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([None] * len(close), index=close.index)

    last_close = float(close.iloc[-1])
    last_sma50 = _last_valid(sma50)
    last_sma200 = _last_valid(sma200)

    if last_sma200 is not None and last_close > last_sma200:
        pts += 20
        result.price_above_200sma = True

    if last_sma50 is not None and last_close > last_sma50:
        pts += 15
        result.price_above_50sma_before_drop = True

    # Golden cross bonus
    if last_sma50 is not None and last_sma200 is not None and last_sma50 > last_sma200:
        pts += 5

    # --- ADX (trend strength, 15 pts) ---
    adx_val = _compute_adx(high, low, close, period=14)
    result.adx_value = adx_val
    min_adx = cfg.get("min_adx", 18)
    if adx_val is not None:
        if adx_val >= 25:
            pts += 15
        elif adx_val >= min_adx:
            pts += 9
        elif adx_val >= 15:
            pts += 4

    # --- RSI at correction start (15 pts) ---
    rsi_series = _compute_rsi(close, period=14)
    rsi_val = _last_valid(rsi_series)
    result.rsi_at_peak = rsi_val
    floor = cfg.get("rsi_uptrend_floor", 45)
    if rsi_val is not None:
        if 55 <= rsi_val <= 70:
            pts += 15
        elif floor <= rsi_val < 55:
            pts += 9
        elif rsi_val > 70:
            pts += 6  # overbought before correction is common
        elif rsi_val < floor:
            pts += 2

    # --- MACD bullish (15 pts) ---
    macd_bullish = _macd_bullish(close)
    result.macd_bullish_before_drop = macd_bullish
    if macd_bullish:
        pts += 15

    # --- Higher highs / higher lows (10 pts) ---
    hhhl = _higher_highs_lows(close, n_swings=4)
    result.higher_highs_higher_lows = hhhl
    if hhhl:
        pts += 10

    # --- OBV uptrend (10 pts) ---
    obv_up = _obv_uptrend(close, volume, lookback=20)
    result.obv_uptrend = obv_up
    if obv_up:
        pts += 10

    result.score = min(100.0, round(pts, 1))
    result.was_in_uptrend = pts >= 40
    result.momentum_score = pts
    return result


def _last_valid(series: pd.Series) -> Optional[float]:
    dropped = series.dropna()
    if dropped.empty:
        return None
    val = float(dropped.iloc[-1])
    return None if np.isnan(val) or np.isinf(val) else val


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
    try:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        atr = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=period - 1, min_periods=period).mean()
        val = float(adx.dropna().iloc[-1]) if not adx.dropna().empty else None
        return None if val is None or np.isnan(val) else val
    except Exception:
        return None


def _macd_bullish(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
    try:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    except Exception:
        return False


def _higher_highs_lows(close: pd.Series, n_swings: int = 4) -> bool:
    try:
        if len(close) < 30:
            return False
        # Simple: divide into equal windows and check if peaks and troughs rise
        chunk = len(close) // n_swings
        peaks = [float(close.iloc[i * chunk:(i + 1) * chunk].max()) for i in range(n_swings)]
        troughs = [float(close.iloc[i * chunk:(i + 1) * chunk].min()) for i in range(n_swings)]
        hh = all(peaks[i] > peaks[i - 1] for i in range(1, len(peaks)))
        hl = all(troughs[i] > troughs[i - 1] for i in range(1, len(troughs)))
        return hh and hl
    except Exception:
        return False


def _obv_uptrend(close: pd.Series, volume: pd.Series, lookback: int = 20) -> bool:
    try:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()
        recent = obv.iloc[-lookback:]
        if len(recent) < 5:
            return False
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent.values.astype(float), 1)[0]
        return slope > 0
    except Exception:
        return False
