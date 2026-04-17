# Swing Trading Scanner

## Purpose
Scans for fundamentally strong stocks that corrected on macro/unrelated events, not fundamental deterioration. Goal: buy the dip, profit from bounce.

## Running
```bash
# Analyze a single stock in detail
python main.py score AAPL

# Scan a custom list
python main.py scan --symbols "AAPL,NVDA,MSFT,TSLA" --min-score 60

# Scan from a watchlist file (one symbol per line)
python main.py scan --watchlist my_stocks.txt --verbose

# Scan S&P 500 (takes 30-40 min first run, cached after)
python main.py scan --sp500 --min-score 65 --export results/sp500.csv

# Cache management
python main.py cache-info
python main.py clear-cache
```

## Key Files
- `config.yaml` — scoring weights, thresholds, keyword lists
- `scanner/fundamental.py` — ROE, D/E, revenue growth, FCF scoring
- `scanner/technical.py` — SMA, ADX, RSI, MACD, OBV uptrend detection
- `scanner/correction.py` — correction depth, volume character, SPY correlation
- `scanner/news_classifier.py` — keyword-based fundamental vs macro event classification
- `scanner/scorer.py` — composite score + BUY_DIP/WATCH/AVOID signal
- `scanner/pipeline.py` — parallel orchestration with ThreadPoolExecutor
- `data/fetcher.py` — yfinance wrapper (uses fast_info for price, not deprecated ticker.info price keys)

## Signals
- **BUY_DIP** (score ≥ 72): Strong setup — fundamentals intact, uptrend, 5%+ correction, low volume selloff, unrelated event
- **WATCH** (score ≥ 58): Developing setup — monitor for further confirmation
- **AVOID**: Fundamental gate failed, news shows deterioration, or no meaningful correction

## Scoring Weights
| Dimension | Weight | Key factors |
|-----------|--------|-------------|
| Fundamental | 35% | ROE, D/E, revenue trend, FCF, earnings growth |
| Technical | 30% | 200/50 SMA, ADX, RSI, MACD, OBV, higher highs/lows |
| Correction | 25% | Drop depth 5-15%, volume ratio <1, near support, RSI oversold |
| News | 10% | Macro/sector/unrelated = good; fundamental = AVOID |

## Hard Gates (instant AVOID regardless of score)
1. Fundamental score < 45 (broken business)
2. News classified as "fundamental" with ≥50% confidence (earnings miss, guidance cut, etc.)
3. Correction < minimum threshold (no dip to buy)

## How Unrelated Correction is Detected
- News keyword classification: macro/sector keywords vs fundamental breakdown keywords
- SPY correlation: if stock dropped same day SPY dropped ≥1%, it's macro-driven
- Volume character: low volume on down days = no informed selling = healthy correction
- Fundamental score high despite price drop = financials not deteriorating
