import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from data import fetcher
from scanner import correction, fundamental, news_classifier, scorer, technical
from scanner.models import CompositeResult

logger = logging.getLogger(__name__)


def scan_ticker(symbol: str, config: dict, use_cache: bool = True) -> Optional[CompositeResult]:
    try:
        data = fetcher.fetch(symbol, use_cache=use_cache)
        if data is None:
            return None

        # Fast-info gates
        fi = data.fast_info
        min_cap = config.get("fundamental", {}).get("min_market_cap", 500_000_000)
        cap = fi.get("market_cap")
        if cap and cap < min_cap:
            logger.debug(f"{symbol}: market cap gate failed")
            return None

        fund = fundamental.score(data, config)
        if not fund.pass_gate:
            logger.debug(f"{symbol}: fundamental gate failed - {fund.fail_reason}")
            # Still return so CLI can show filtered results if verbose
            return CompositeResult(
                symbol=symbol,
                composite_score=fund.score,
                fundamental=fund,
                signal="AVOID",
                reason=f"Fundamental gate: {fund.fail_reason}",
                current_price=fi.get("last_price"),
                year_high=fi.get("year_high"),
                sector=data.info.get("sector"),
                industry=data.info.get("industry"),
            )

        corr_result, corr_days = correction.detect(data, config)
        min_drop = config.get("correction", {}).get("min_drop_pct", 0.04)
        if corr_result.correction_pct < min_drop:
            logger.debug(f"{symbol}: correction too small ({corr_result.correction_pct:.1%})")
            return None

        tech = technical.score(data, config, correction_days=corr_days)
        news = news_classifier.classify(data, config)

        return scorer.compute(
            symbol=symbol,
            fund=fund,
            tech=tech,
            corr=corr_result,
            news=news,
            config=config,
            current_price=fi.get("last_price"),
            year_high=fi.get("year_high"),
            sector=data.info.get("sector"),
            industry=data.info.get("industry"),
        )

    except Exception as e:
        logger.warning(f"{symbol}: pipeline error - {e}", exc_info=True)
        return None


def scan_universe(
    symbols: List[str],
    config: dict,
    workers: int = 5,
    use_cache: bool = True,
    progress_cb=None,
) -> List[CompositeResult]:
    results = []
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_ticker, sym, config, use_cache): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                logger.warning(f"{sym}: future error - {e}")
            if progress_cb:
                progress_cb(done, total, sym)
            # Polite rate limiting between batch completions
            if done % workers == 0:
                time.sleep(0.3)

    # Sort: BUY_DIP first, then WATCH, then by score desc
    order = {"BUY_DIP": 0, "WATCH": 1, "AVOID": 2, "INSUFFICIENT_DATA": 3}
    results.sort(key=lambda r: (order.get(r.signal, 9), -r.composite_score))
    return results
