# Run a scan

Scan a universe of stocks and show signals.

Common invocations — pick based on what the user asks:

```bash
# Scan specific symbols
python main.py scan --symbols "$ARGUMENTS" --verbose

# Scan Nasdaq-100
python main.py scan --nasdaq100 --verbose

# Scan S&P 500
python main.py scan --sp500 --verbose

# Scan with minimum correction filter
python main.py scan --nasdaq100 --min-correction 0.05 --verbose

# Export results to CSV
python main.py scan --nasdaq100 --export results.csv
```

**Usage examples:**
- `/scan AAPL,NVDA,MSFT` — scan specific symbols
- `/scan nasdaq100` — scan full Nasdaq-100
- `/scan sp500` — scan full S&P 500

Run the appropriate command, then summarise the BUY_DIP and WATCH signals from the output.
If no argument given, default to: `python main.py scan --nasdaq100 --verbose`
