# Add a feature to the Swing Trading Scanner

Before implementing any feature, read these files to understand the current state:

1. `gui/app.py` — all GUI logic (Streamlit multi-page app)
2. `scanner/models.py` — all dataclasses (FundamentalResult, TechnicalResult, CorrectionResult, NewsResult, PatternResult, CompositeResult)
3. `config.yaml` — all thresholds and scoring weights
4. `gui/db.py` — SQLite schema and queries

## Key rules to follow

**Streamlit widget state:** Never write `st.session_state["widget_key"] = value` after the widget renders. Use an intermediate `_xxx_request` key applied at the TOP of the function before the widget renders. This has burned us twice — once for `nav_page`, once for `dd_select`.

**No backwards-compat shims:** If removing a field, delete it. Don't add `_old` aliases.

**No speculative features:** Only implement exactly what was asked. No "while I'm here" additions.

**DB migrations:** Add new columns via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `init_db()` inside a try/except so existing DBs auto-migrate safely.

**Scoring weights:** Always sum to 1.0. Defined in `config.yaml` under `scoring.weights` — do not hardcode in Python.

## Architecture reminder

```
Data flow: yfinance → fetcher.py → TickerData
           TickerData → scanner/* → *Result dataclasses
           *Result → scorer.py → CompositeResult (signal + composite_score)
           CompositeResult → gui/app.py (display) + gui/db.py (persist)
```

Now read the relevant files and implement: $ARGUMENTS
