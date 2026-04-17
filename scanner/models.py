from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FundamentalResult:
    symbol: str
    score: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    return_on_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    gross_margins: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cashflow: Optional[float] = None
    revenue_trend: Optional[str] = None  # "growing" | "stable" | "declining"
    pass_gate: bool = False
    fail_reason: str = ""


@dataclass
class TechnicalResult:
    symbol: str
    score: float
    was_in_uptrend: bool = False
    price_above_200sma: bool = False
    price_above_50sma_before_drop: bool = False
    adx_value: Optional[float] = None
    rsi_at_peak: Optional[float] = None
    macd_bullish_before_drop: bool = False
    higher_highs_higher_lows: bool = False
    obv_uptrend: bool = False
    momentum_score: float = 0.0


@dataclass
class CorrectionResult:
    symbol: str
    score: float
    correction_pct: float = 0.0
    correction_start_date: Optional[str] = None
    days_declining: int = 0
    avg_volume_during_drop: float = 0.0
    avg_volume_prior_20d: float = 0.0
    volume_ratio: float = 1.0
    obv_trend: str = "unknown"  # "declining" | "stable" | "rising"
    rsi_at_bottom: Optional[float] = None
    is_near_support: bool = False
    support_level: Optional[float] = None
    spy_correlation: Optional[float] = None
    is_macro_correlated: bool = False


@dataclass
class NewsResult:
    symbol: str
    score: float
    event_type: str = "unknown"  # "macro"|"sector"|"unrelated"|"fundamental"|"mixed"|"unknown"
    fundamental_signal_count: float = 0.0
    macro_signal_count: float = 0.0
    headlines_analyzed: int = 0
    top_headlines: list = field(default_factory=list)
    confidence: float = 0.5
    is_safe: bool = True
    downgrade_count: int = 0
    pre_earnings: bool = False


@dataclass
class CompositeResult:
    symbol: str
    composite_score: float
    fundamental: Optional[FundamentalResult] = None
    technical: Optional[TechnicalResult] = None
    correction: Optional[CorrectionResult] = None
    news: Optional[NewsResult] = None
    signal: str = "AVOID"  # "BUY_DIP" | "WATCH" | "AVOID" | "INSUFFICIENT_DATA"
    reason: str = ""
    current_price: Optional[float] = None
    year_high: Optional[float] = None
    pct_from_high: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
