from scanner.models import (
    CompositeResult,
    CorrectionResult,
    FundamentalResult,
    NewsResult,
    PatternResult,
    TechnicalResult,
)

_DEFAULT_WEIGHTS = {
    "fundamental": 0.30,
    "technical":   0.25,
    "correction":  0.22,
    "news":        0.08,
    "pattern":     0.15,
}


def compute(
    symbol: str,
    fund: FundamentalResult,
    tech: TechnicalResult,
    corr: CorrectionResult,
    news: NewsResult,
    config: dict,
    pattern: PatternResult = None,
    current_price: float = None,
    year_high: float = None,
    sector: str = None,
    industry: str = None,
) -> CompositeResult:
    weights = config.get("scoring", {}).get("weights", _DEFAULT_WEIGHTS)

    # Hard gate 1: fundamental breakdown
    if not fund.pass_gate:
        return CompositeResult(
            symbol=symbol,
            composite_score=fund.score,
            fundamental=fund, technical=tech, correction=corr, news=news, pattern=pattern,
            signal="AVOID",
            reason=f"Fundamental gate failed: {fund.fail_reason}",
            current_price=current_price, year_high=year_high,
            sector=sector, industry=industry,
        )

    # Hard gate 2: fundamental deterioration in news
    if not news.is_safe and news.confidence >= 0.5:
        return CompositeResult(
            symbol=symbol,
            composite_score=fund.score * weights.get("fundamental", 0.30),
            fundamental=fund, technical=tech, correction=corr, news=news, pattern=pattern,
            signal="AVOID",
            reason=f"Fundamental deterioration in news ({news.event_type}, conf={news.confidence:.0%})",
            current_price=current_price, year_high=year_high,
            sector=sector, industry=industry,
        )

    # Hard gate 3: no meaningful correction
    if corr.correction_pct < config.get("correction", {}).get("min_drop_pct", 0.04):
        return CompositeResult(
            symbol=symbol,
            composite_score=0.0,
            fundamental=fund, technical=tech, correction=corr, news=news, pattern=pattern,
            signal="AVOID",
            reason=f"No correction detected ({corr.correction_pct:.1%} drop, minimum required)",
            current_price=current_price, year_high=year_high,
            sector=sector, industry=industry,
        )

    # Normalise weights so they always sum to 1.0 regardless of config
    w_fund  = weights.get("fundamental", 0.30)
    w_tech  = weights.get("technical",   0.25)
    w_corr  = weights.get("correction",  0.22)
    w_news  = weights.get("news",        0.08)
    w_pat   = weights.get("pattern",     0.15)
    total_w = w_fund + w_tech + w_corr + w_news + w_pat
    if total_w > 0:
        w_fund /= total_w; w_tech /= total_w; w_corr /= total_w
        w_news /= total_w; w_pat  /= total_w

    pat_score = pattern.score if pattern is not None else 50.0  # neutral fallback

    composite = (
        fund.score  * w_fund +
        tech.score  * w_tech +
        corr.score  * w_corr +
        news.score  * w_news +
        pat_score   * w_pat
    )
    composite = round(composite, 1)

    pct_from_high = None
    if current_price and year_high and year_high > 0:
        pct_from_high = (current_price - year_high) / year_high

    if composite >= 72 and corr.correction_pct >= 0.05:
        signal = "BUY_DIP"
    elif composite >= 58:
        signal = "WATCH"
    else:
        signal = "AVOID"

    reason = _build_reason(fund, tech, corr, news, pattern, composite, signal)

    return CompositeResult(
        symbol=symbol,
        composite_score=composite,
        fundamental=fund, technical=tech, correction=corr, news=news, pattern=pattern,
        signal=signal,
        reason=reason,
        current_price=current_price,
        year_high=year_high,
        pct_from_high=pct_from_high,
        sector=sector,
        industry=industry,
    )


def _build_reason(fund, tech, corr, news, pattern, composite, signal) -> str:
    parts = []
    if fund.score >= 70:
        parts.append("strong fundamentals")
    elif fund.score >= 50:
        parts.append("decent fundamentals")
    if tech.was_in_uptrend:
        parts.append("was in uptrend")
    if corr.correction_pct >= 0.05:
        parts.append(f"{corr.correction_pct:.0%} correction")
    if corr.volume_ratio < 0.85:
        parts.append("low-volume selloff")
    if news.event_type in ("macro", "sector", "unrelated"):
        parts.append(f"{news.event_type} event (not fundamental)")
    if corr.is_macro_correlated:
        parts.append("correlated with SPY drop")
    if corr.is_near_support:
        parts.append("near support")
    if pattern is not None:
        if pattern.has_bullish_pattern and pattern.top_pattern_name:
            parts.append(f"{pattern.top_pattern_name} pattern")
        if pattern.nearest_support_price:
            parts.append(f"S/R support ${pattern.nearest_support_price:.2f}")
    return "; ".join(parts) if parts else f"composite={composite}"
