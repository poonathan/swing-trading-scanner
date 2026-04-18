# Clear the data cache

Delete all cached price and fundamental data so the next scan/score fetches fresh data from yfinance.

```bash
python main.py clear-cache
```

**Usage:** `/clear-cache`

Cache lives at `data/.cache/`. TTLs:
- Price history: 4 hours
- Fundamentals (income stmt, balance sheet, info): 24 hours

Use this when you suspect stale data or after market close to force a fresh fetch.
