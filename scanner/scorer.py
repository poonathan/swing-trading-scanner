from scanner.models import (
    CompositeResult,
    CorrectionResult,
    FundamentalResult,
    NewsResult,
    TechnicalResult,
)


def compute(
    symbol: str,
    fund: FundamentalResult,
    tech: TechnicalResult,
    corr: CorrectionResult,
    news: NewsResult,
    config: dict,
    current_price: float = None,
    year_high: float = None,
    sector: str = None,
    industry: str = None,
) -> CompositeResult:
    weights = config.get("scoring", {}).get("weights", {
        "fundamental": 0.35,
        "technical": 0.30,
        "correction": 0.25,
        "news": 0.10,
    })
    min_composite = config.get("scoring", {}).get("min_composite_score", 60)

    # Hard gate 1: fundamental breakdown
    if not fund.pass_gate:
        return CompositeResult(
            symbol=symbol,
            composite_score=fund.score,
            fundamental=fund, technical=tech, correction=corr, news=news,
            signal="AVOID",
            reason=f"Fundamental gate failed: {fund.fail_reason}",
            current_price=current_price, year_high=year_high,
            sector=sector, industry=industry,
        )

    # Hard gate 2: fundamental deterioration in news
    if not news.is_safe and news.confidence >= 0.5:
        return CompositeResult(
            symbol=symbol,
            composite_score=fund.score * weights["fundamental"],
            fundamental=fund, technical=tech, correction=corr, news=news,
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
            fundamental=fund, technical=tech, correction=corr, news=news,
            signal="AVOID",
            reason=f"No correction detected ({corr.correction_pct:.1%} drop, minimum required)",
            current_price=current_price, year_high=year_high,
            sector=sector, industry=industry,
        )

    composite = (
        fund.score * weights.get("fundamental", 0.35)
        + tech.score * weights.get("technical", 0.30)
        + corr.score * weights.get("correction", 0.25)
        + news.score * weights.get("news", 0.10)
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

    reason = _build_reason(fund, tech, corr, news, composite, signal)

    return CompositeResult(
        symbol=symbol,
        composite_score=composite,
        fundamental=fund, technical=tech, correction=corr, news=news,
        signal=signal,
        reason=reason,
        current_price=current_price,
        year_high=year_high,
        pct_from_high=pct_from_high,
        sector=sector,
        industry=industry,
    )


def _build_reason(fund, tech, corr, news, composite, signal) -> str:
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
    return "; ".join(parts) if parts else f"composite={composite}"
