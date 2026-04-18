# Launch the GUI

Start the Streamlit GUI app.

```bash
streamlit run gui/app.py
```

Or via the launcher script which also checks dependencies:

```bash
cmd /c launch.bat
```

**Usage:** `/launch`

After running, the app opens at http://localhost:8501
- Dashboard: latest scan results and KPIs
- Scanner: run new scans
- Deep Dive: single-ticker TradingView chart + full analysis
- Trend Tracker: score history
- History: full log with filters
- Watchlist: tracked symbols
