"""Swing Trading Scanner — Streamlit GUI."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from gui import db
from scanner.pipeline import scan_ticker
from scanner.universe import load_symbols

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_BUY_COLOR = "#28a745"
_WATCH_COLOR = "#ffc107"
_AVOID_COLOR = "#dc3545"
_BUY_BG = "#d4edda"
_WATCH_BG = "#fff3cd"
_BUY_THRESH = 72
_WATCH_THRESH = 58


@st.cache_data(ttl=300)
def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


db.init_db()
config = load_config()

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Swing Trading Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LIGHT_CSS = """
<style>
    div[data-testid="metric-container"] {
        border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 12px 16px; background: #fafafa;
    }
    .stProgress > div > div > div { background-color: #28a745; }
    .block-container { padding-top: 1.5rem; }
    thead tr th { background: #f0f2f6 !important; font-weight: 600 !important; }
</style>
"""

_DARK_CSS = """
<style>
    .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #0d1117 !important; color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important; border-right: 1px solid #30363d !important;
    }
    div[data-testid="metric-container"] {
        background: #21262d !important; border-color: #30363d !important;
        border-radius: 10px; padding: 12px 16px;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    .stCaption, label { color: #adbac7 !important; }
    h1, h2, h3, h4 { color: #f0f6fc !important; }
    .stTextInput input, .stNumberInput input,
    [data-baseweb="select"] > div {
        background-color: #21262d !important; color: #e6edf3 !important;
        border-color: #444c56 !important;
    }
    [data-testid="stExpander"] { background-color: #161b22 !important; border-color: #30363d !important; }
    button[kind="secondary"] { background-color: #21262d !important; color: #e6edf3 !important; border-color: #444c56 !important; }
    thead tr th { background: #21262d !important; color: #e6edf3 !important; font-weight: 600 !important; }
    .block-container { padding-top: 1.5rem; }
    hr { border-color: #30363d !important; }
</style>
"""

_dark = st.session_state.get("dark_mode", False)
st.markdown(_DARK_CSS if _dark else _LIGHT_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

_NAV_OPTIONS = ["🏠  Dashboard", "🔍  Scanner", "🔎  Deep Dive", "📊  Trend Tracker", "📋  History", "⭐  Watchlist"]

# Apply any pending navigation BEFORE the radio widget is instantiated.
# Setting st.session_state[key] after a widget with that key renders throws an error.
if "_nav_request" in st.session_state:
    pending = st.session_state.pop("_nav_request")
    if pending in _NAV_OPTIONS:
        st.session_state["nav_page"] = pending

with st.sidebar:
    st.markdown("## 📈 Swing Scanner")
    st.divider()

    if st.session_state.get("nav_page") not in _NAV_OPTIONS:
        st.session_state["nav_page"] = _NAV_OPTIONS[0]

    page = st.radio(
        "nav",
        _NAV_OPTIONS,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.divider()
    stats = db.get_run_stats()
    lr = stats.get("latest_run")
    if lr:
        st.caption(f"**Last scan:** {lr['run_at'][:16].replace('T', ' ')}")
        st.caption(f"**Universe:** {lr.get('universe', '—')}")
        c1, c2 = st.columns(2)
        c1.metric("🟢 BUY DIP", lr.get("buy_dip_count", 0))
        c2.metric("🟡 WATCH", lr.get("watch_count", 0))
    else:
        st.info("No scans yet.\nRun a scan to start.")
    st.divider()
    st.caption(f"Total scans: **{stats['total_runs']}**")
    st.caption(f"Symbols tracked: **{stats['unique_symbols']}**")
    st.divider()
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode")

# ── Helpers ───────────────────────────────────────────────────────────────────

_SIG_EMOJI = {"BUY_DIP": "🟢", "WATCH": "🟡", "AVOID": "🔴"}
_SIG_COLOR = {"BUY_DIP": _BUY_COLOR, "WATCH": _WATCH_COLOR, "AVOID": _AVOID_COLOR}


def _fmt_cap(val) -> str:
    if not val:
        return "—"
    if val >= 1e12:  return f"${val/1e12:.2f}T"
    if val >= 1e9:   return f"${val/1e9:.1f}B"
    if val >= 1e6:   return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"


def _fmt_vol(val) -> str:
    if not val:
        return "—"
    if val >= 1e6: return f"{val/1e6:.1f}M"
    if val >= 1e3: return f"{val/1e3:.0f}K"
    return str(int(val))


def _display_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for display tables."""
    cols = {
        "symbol": "Symbol", "signal": "Signal",
        "composite_score": "Score", "fund_score": "Fund",
        "tech_score": "Tech", "corr_score": "Corr", "news_score": "News",
        "pat_score": "Pat", "top_pattern": "Pattern",
        "correction_pct": "Drop%", "volume_ratio": "VolRt",
        "event_type": "Event", "current_price": "Price",
        "sector": "Sector", "reason": "Reason",
    }
    existing = {k: v for k, v in cols.items() if k in df.columns}
    out = df[list(existing.keys())].rename(columns=existing).copy()
    if "Signal" in out:
        out["Signal"] = out["Signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}")
    if "Score" in out:
        out["Score"] = out["Score"].round(1)
    if "Drop%" in out:
        out["Drop%"] = out["Drop%"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    if "VolRt" in out:
        out["VolRt"] = out["VolRt"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    if "Price" in out:
        out["Price"] = out["Price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
    for c in ("Fund", "Tech", "Corr", "News", "Pat"):
        if c in out:
            out[c] = out[c].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    if "Pattern" in out:
        out["Pattern"] = out["Pattern"].apply(lambda x: x if pd.notna(x) and x else "—")
    return out


def _style_table(df: pd.DataFrame) -> "pd.Styler":
    def row_bg(row):
        sig = str(row.get("Signal", ""))
        if "BUY_DIP" in sig:
            return [f"background-color: {_BUY_BG}"] * len(row)
        if "WATCH" in sig:
            return [f"background-color: {_WATCH_BG}"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_bg, axis=1).hide(axis="index")


def _score_bar(df: pd.DataFrame) -> "pd.Styler":
    """Color the Score column as a gradient bar."""
    def score_color(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= _BUY_THRESH:
            return f"background: linear-gradient(90deg, {_BUY_BG} {v}%, white {v}%); color: #155724; font-weight:600"
        if v >= _WATCH_THRESH:
            return f"background: linear-gradient(90deg, {_WATCH_BG} {v}%, white {v}%); color: #856404; font-weight:600"
        return "color: #666"
    s = df.style.hide(axis="index")
    if "Score" in df.columns:
        s = s.applymap(score_color, subset=["Score"])
    return s


# ── Column guide tooltip ─────────────────────────────────────────────────────

def _col_guide():
    with st.expander("📖 Column guide — what do these columns mean?", expanded=False):
        st.markdown("""
| Column | What it means |
|--------|--------------|
| **Score** | Composite 0–100. **BUY DIP ≥ 72**, WATCH ≥ 58, below = Avoid |
| **Fund** | Fundamental score (ROE, D/E ratio, revenue growth, free cash flow, earnings) — 30% weight |
| **Tech** | Technical score (SMA 50/200, ADX trend strength, RSI, MACD, OBV, higher-highs/lows) — 25% weight |
| **Corr** | Correction quality (drop depth 5–15%, low-volume selloff, near S/R support, RSI oversold) — 22% weight |
| **News** | News event type — *macro/sector/unrelated* = good (not the company's fault), *fundamental* = avoid — 8% weight |
| **Pat** | Chart pattern score: 50 = neutral, > 50 = bullish setups found, < 50 = bearish patterns present — 15% weight |
| **Pattern** | Top detected chart pattern (▲ bullish reversal/continuation, ▼ bearish) |
| **Drop%** | How far the price has fallen from its recent high — sweet spot is 5–15% |
| **VolRt** | Volume during the correction vs 20-day average. **< 0.85x = healthy** (low-volume selloff, no panic selling) |
| **Event** | What triggered the drop: *macro* (Fed/rates), *sector* (industry news), *unrelated*, or *fundamental* (company-specific bad news) |
| **Price** | Latest market price |
        """)


# ── Signal strength gauge ─────────────────────────────────────────────────────

def _signal_gauge(score: float, title: str = "Composite Score") -> go.Figure:
    """Speedometer gauge showing composite score vs BUY DIP / WATCH thresholds."""
    if score >= _BUY_THRESH:
        bar_color = _BUY_COLOR
    elif score >= _WATCH_THRESH:
        bar_color = _WATCH_COLOR
    else:
        bar_color = _AVOID_COLOR

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 13}},
        number={"font": {"size": 32, "color": bar_color}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#bbb",
                     "tickvals": [0, _WATCH_THRESH, _BUY_THRESH, 100],
                     "ticktext": ["0", f"WATCH\n{_WATCH_THRESH}", f"BUY\n{_BUY_THRESH}", "100"]},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#e0e0e0",
            "steps": [
                {"range": [0, _WATCH_THRESH],  "color": "#ffeaea"},
                {"range": [_WATCH_THRESH, _BUY_THRESH], "color": "#fff8e1"},
                {"range": [_BUY_THRESH, 100],  "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": _BUY_COLOR, "width": 3},
                "thickness": 0.75,
                "value": _BUY_THRESH,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20), paper_bgcolor="white")
    return fig


# ── Dashboard ─────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("📈 Swing Trading Dashboard")
    latest = db.get_latest_results()
    upgrades = db.get_upgrades()
    deltas = db.get_score_deltas()

    if latest.empty:
        st.info("No scan results yet. Go to **🔍 Scanner** to run your first scan.")
        return

    buy_dip = latest[latest["signal"] == "BUY_DIP"]
    watch = latest[latest["signal"] == "WATCH"]

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🟢 BUY DIP", len(buy_dip),
              help=f"Stocks with composite score ≥ {_BUY_THRESH} AND correction ≥ 5% — strong buy-the-dip setup")
    c2.metric("🟡 WATCH", len(watch),
              help=f"Stocks with composite score ≥ {_WATCH_THRESH} — setup developing, worth monitoring")
    c3.metric("⬆️ Upgrades", len(upgrades),
              help="Stocks that moved from WATCH or AVOID → BUY DIP since the previous scan")
    c4.metric("📊 Scanned", len(latest),
              help="Total symbols processed in the latest scan run")
    lr = db.get_latest_run()
    c5.metric("🕐 Scan Age", _scan_age(lr["run_at"]) if lr else "—",
              help="Time since the last scan was run. Data may be stale if > 4 hours old")

    st.divider()

    # BUY DIP table
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader(f"🟢 BUY DIP Opportunities ({len(buy_dip)})")
        _col_guide()
        if not buy_dip.empty:
            disp = _display_cols(buy_dip)
            _selectable_table(disp, height=280, key="buydip")
        else:
            st.info("No BUY DIP candidates in the latest scan.")

    with col_right:
        st.subheader(f"⬆️ Newly Qualified ({len(upgrades)})")
        if not upgrades.empty:
            u = upgrades[["symbol", "composite_score", "prev_score", "correction_pct", "event_type"]].copy()
            u["delta"] = (u["composite_score"] - u["prev_score"]).round(1)
            u = u.rename(columns={
                "symbol": "Symbol", "composite_score": "Score",
                "prev_score": "Prev", "correction_pct": "Drop%",
                "event_type": "Event", "delta": "Δ Score"
            })
            u["Drop%"] = u["Drop%"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
            u["Δ Score"] = u["Δ Score"].apply(lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
            st.dataframe(u.style.hide(axis="index"), use_container_width=True, height=280)
        else:
            st.info("No new upgrades since last scan.")

    st.divider()

    # Score delta movers
    if not deltas.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📈 Most Improved")
            top = deltas[deltas["delta"] > 0].head(8)[["symbol", "score", "prev_score", "delta", "signal"]]
            top = top.rename(columns={"symbol": "Symbol", "score": "Score", "prev_score": "Prev", "delta": "Δ", "signal": "Signal"})
            top["Δ"] = top["Δ"].apply(lambda x: f"+{x:.1f}")
            top["Signal"] = top["Signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}")
            st.dataframe(top.style.hide(axis="index"), use_container_width=True, height=260)
        with col_b:
            st.subheader("📉 Most Declined")
            bot = deltas[deltas["delta"] < 0].tail(8)[["symbol", "score", "prev_score", "delta", "signal"]]
            bot = bot.sort_values("delta").rename(columns={"symbol": "Symbol", "score": "Score", "prev_score": "Prev", "delta": "Δ", "signal": "Signal"})
            bot["Δ"] = bot["Δ"].apply(lambda x: f"{x:.1f}")
            bot["Signal"] = bot["Signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}")
            st.dataframe(bot.style.hide(axis="index"), use_container_width=True, height=260)

    st.divider()

    # WATCH table
    st.subheader(f"🟡 WATCH Candidates ({len(watch)})")
    if not watch.empty:
        disp = _display_cols(watch)
        _selectable_table(disp, height=320, key="watch")
        _col_guide()


def _scan_age(run_at: str) -> str:
    try:
        dt = datetime.fromisoformat(run_at)
        delta = datetime.now() - dt
        h = int(delta.total_seconds() // 3600)
        m = int((delta.total_seconds() % 3600) // 60)
        return f"{h}h {m}m ago" if h > 0 else f"{m}m ago"
    except Exception:
        return "—"


# ── Scanner ───────────────────────────────────────────────────────────────────

def page_scanner():
    st.title("🔍 Run a Scan")

    with st.form("scan_form"):
        st.subheader("Universe")
        universe_choice = st.radio(
            "universe",
            ["Nasdaq-100", "S&P 500", "Watchlist", "Custom Symbols"],
            horizontal=True,
            label_visibility="collapsed",
        )
        custom_syms = st.text_input(
            "Custom symbols (comma-separated)",
            placeholder="AAPL, NVDA, MSFT, AMD",
            disabled=(universe_choice != "Custom Symbols"),
        )

        st.subheader("Parameters")
        col1, col2, col3, col4 = st.columns(4)
        min_score = col1.slider("Min Composite Score", 40, 85, 58, 1,
            help=f"Only show results at or above this score. WATCH threshold = {_WATCH_THRESH}, BUY DIP = {_BUY_THRESH}")
        min_correction = col2.slider("Min Correction %", 2, 20, 4, 1,
            help="Only include stocks that have dropped at least this much from their recent high. Sweet spot is 5–15%")
        workers = col3.slider("Parallel Workers", 1, 10, 5, 1,
            help="Number of stocks fetched in parallel. Higher = faster scan but more network load")
        use_cache = col4.checkbox("Use Cache", value=True,
            help="Use locally cached data (up to 4h old for prices, 24h for fundamentals) to speed up the scan")

        submitted = st.form_submit_button("🔍 Run Scan", use_container_width=True, type="primary")

    if not submitted:
        _show_last_scan_preview()
        return

    # Load symbols
    if universe_choice == "Custom Symbols":
        if not custom_syms.strip():
            st.error("Please enter at least one symbol."); return
        syms = load_symbols(symbols=custom_syms)
        univ_label = f"custom ({len(syms)})"
    elif universe_choice == "Nasdaq-100":
        with st.spinner("Loading Nasdaq-100 components..."):
            syms = load_symbols(nasdaq100=True)
        univ_label = "nasdaq100"
    elif universe_choice == "S&P 500":
        with st.spinner("Loading S&P 500 components..."):
            syms = load_symbols(sp500=True)
        univ_label = "sp500"
    else:  # Watchlist
        syms = db.get_watchlist_symbols()
        if not syms:
            st.warning("Watchlist is empty. Add symbols in the ⭐ Watchlist page."); return
        univ_label = "watchlist"

    st.info(f"Scanning **{len(syms)}** symbols · min score **{min_score}** · min correction **{min_correction}%**")

    scan_config = dict(config)
    scan_config["correction"] = dict(config.get("correction", {}))
    scan_config["correction"]["min_drop_pct"] = min_correction / 100

    # ── Run with live progress ────────────────────────────────────────────────
    progress_bar = st.progress(0.0, text="Starting scan…")
    status_box = st.empty()
    results = []
    total = len(syms)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_ticker, sym, scan_config, use_cache): sym for sym in syms}
        done = 0
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception:
                pass
            pct = done / total
            progress_bar.progress(pct, text=f"Scanned **{sym}** ({done}/{total})")

    progress_bar.empty()
    status_box.empty()

    # Filter for display (show qualifying + near-qualifying)
    display = [r for r in results if r.signal in ("BUY_DIP", "WATCH")]
    avoid = [r for r in results if r.signal == "AVOID"]

    # Save to DB
    all_to_save = [r for r in results if r.composite_score > 0]
    if all_to_save:
        run_id = db.save_scan(all_to_save, univ_label, scan_config)
        st.success(f"✅ Saved {len(all_to_save)} results (run #{run_id})")

    # Summary metrics
    buy_dip_ct = sum(1 for r in display if r.signal == "BUY_DIP")
    watch_ct = sum(1 for r in display if r.signal == "WATCH")
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 BUY DIP", buy_dip_ct)
    c2.metric("🟡 WATCH", watch_ct)
    c3.metric("🔴 AVOID / no correction", len(avoid))

    if not display:
        st.warning("No qualifying opportunities found. Try lowering the min score or correction threshold.")
        return

    # Results table
    def _top_pattern_label(r):
        p = r.pattern
        if p and p.has_bullish_pattern and p.top_pattern_name:
            return f"▲ {p.top_pattern_name}"
        if p and p.patterns_detected:
            bearish = [pm for pm in p.patterns_detected if pm.signal == "bearish"]
            if bearish:
                return f"▼ {bearish[0].name}"
        return None

    df = pd.DataFrame([{
        "symbol": r.symbol,
        "signal": r.signal,
        "composite_score": r.composite_score,
        "fund_score": r.fundamental.score if r.fundamental else None,
        "tech_score": r.technical.score if r.technical else None,
        "corr_score": r.correction.score if r.correction else None,
        "news_score": r.news.score if r.news else None,
        "pat_score": round(r.pattern.score, 1) if r.pattern else None,
        "top_pattern": _top_pattern_label(r),
        "correction_pct": round(r.correction.correction_pct * 100, 1) if r.correction else None,
        "volume_ratio": r.correction.volume_ratio if r.correction else None,
        "event_type": r.news.event_type if r.news else None,
        "current_price": r.current_price,
        "sector": r.sector,
        "reason": r.reason,
    } for r in sorted(display, key=lambda x: (x.signal != "BUY_DIP", -x.composite_score))])

    st.subheader("📊 Results")
    _col_guide()
    disp = _display_cols(df)
    _selectable_table(disp, height=420, key="scanner")

    # Quick add to watchlist
    st.divider()
    st.caption("Add any of these to your watchlist for ongoing tracking:")
    wl_syms = st.multiselect("Add to Watchlist", [r.symbol for r in display])
    if wl_syms and st.button("⭐ Add Selected to Watchlist"):
        for sym in wl_syms:
            db.add_to_watchlist(sym)
        st.success(f"Added {', '.join(wl_syms)} to watchlist")


def _show_last_scan_preview():
    latest = db.get_latest_results(min_score=50)
    if latest.empty:
        return
    with st.expander("📋 Latest scan results (click to expand)", expanded=False):
        disp = _display_cols(latest.head(20))
        st.dataframe(_style_table(disp), use_container_width=True, height=340)


# ── Trend Tracker ─────────────────────────────────────────────────────────────

def page_tracker():
    st.title("📊 Trend Tracker")
    st.caption("Track how a stock's setup evolves across multiple scans over time.")

    all_syms = db.get_all_tracked_symbols()
    if not all_syms:
        st.info("No scan history yet. Run a scan first."); return

    wl_syms = db.get_watchlist_symbols()
    default_sym = wl_syms[0] if wl_syms else all_syms[0]

    col_sel, col_add = st.columns([4, 1])
    symbol = col_sel.selectbox("Select Symbol", all_syms, index=all_syms.index(default_sym) if default_sym in all_syms else 0)

    # Add / remove watchlist
    in_wl = symbol in wl_syms
    btn_label = "⭐ Remove from Watchlist" if in_wl else "⭐ Add to Watchlist"
    if col_add.button(btn_label, use_container_width=True):
        if in_wl:
            db.remove_from_watchlist(symbol)
            st.rerun()
        else:
            db.add_to_watchlist(symbol)
            st.rerun()

    hist = db.get_symbol_history(symbol)
    if hist.empty:
        st.warning(f"No history for {symbol}."); return

    hist["scan_date"] = pd.to_datetime(hist["scan_date"])
    n_scans = len(hist)
    latest_row = hist.iloc[-1]
    first_row = hist.iloc[0]
    score_delta = latest_row["composite_score"] - first_row["composite_score"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scans recorded", n_scans)
    c2.metric("Latest Score", f"{latest_row['composite_score']:.1f}",
              delta=f"{score_delta:+.1f} since first scan")
    c3.metric("Latest Signal", f"{_SIG_EMOJI.get(latest_row['signal'],'')} {latest_row['signal']}")
    c4.metric("Latest Price", f"${latest_row['current_price']:.2f}" if latest_row["current_price"] else "—",
              delta=f"Drop: {latest_row['correction_pct']:.1f}%" if latest_row["correction_pct"] else None,
              delta_color="inverse")

    st.divider()

    # ── Score trend chart ─────────────────────────────────────────────────────
    st.subheader("📈 Composite Score Over Time")

    fig = go.Figure()

    # Threshold bands
    fig.add_hrect(y0=_BUY_THRESH, y1=100, fillcolor=_BUY_BG, opacity=0.25, line_width=0,
                  annotation_text="BUY DIP zone", annotation_position="top right")
    fig.add_hrect(y0=_WATCH_THRESH, y1=_BUY_THRESH, fillcolor=_WATCH_BG, opacity=0.30, line_width=0,
                  annotation_text="WATCH zone", annotation_position="top right")
    fig.add_hline(y=_BUY_THRESH, line_dash="dash", line_color=_BUY_COLOR, line_width=1.5,
                  annotation_text=f"BUY DIP ({_BUY_THRESH})")
    fig.add_hline(y=_WATCH_THRESH, line_dash="dash", line_color=_WATCH_COLOR, line_width=1.5,
                  annotation_text=f"WATCH ({_WATCH_THRESH})")

    # Score line
    fig.add_trace(go.Scatter(
        x=hist["scan_date"], y=hist["composite_score"],
        mode="lines+markers+text",
        name="Composite Score",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(
            size=10,
            color=[_SIG_COLOR.get(s, "#999") for s in hist["signal"]],
            line=dict(color="white", width=2),
        ),
        text=hist["composite_score"].round(1),
        textposition="top center",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}<br><extra></extra>",
    ))

    fig.update_layout(
        yaxis=dict(range=[0, 105], title="Score"),
        xaxis_title="Scan Date",
        height=360,
        margin=dict(t=30, b=10),
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Dimension scores breakdown ────────────────────────────────────────────
    st.subheader("📊 Dimension Scores Breakdown")

    dim_cols = ["fund_score", "tech_score", "corr_score", "news_score", "pat_score"]
    dim_labels = {"fund_score": "Fundamental", "tech_score": "Technical",
                  "corr_score": "Correction", "news_score": "News", "pat_score": "Pattern"}
    dim_colors = {"Fundamental": "#2196F3", "Technical": "#9C27B0",
                  "Correction": "#FF9800", "News": "#4CAF50", "Pattern": "#E91E63"}

    fig2 = go.Figure()
    for col in dim_cols:
        if col in hist.columns:
            label = dim_labels[col]
            fig2.add_trace(go.Bar(
                name=label,
                x=hist["scan_date"].dt.strftime("%Y-%m-%d"),
                y=hist[col],
                marker_color=dim_colors[label],
                opacity=0.85,
                hovertemplate=f"<b>{label}</b>: %{{y:.0f}}<br><extra></extra>",
            ))

    fig2.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 105], title="Score"),
        xaxis_title="Scan Date",
        height=320,
        margin=dict(t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Correction & Volume chart ─────────────────────────────────────────────
    if "correction_pct" in hist.columns and hist["correction_pct"].notna().any():
        st.subheader("📉 Correction Depth & Volume Ratio")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Drop %", x=hist["scan_date"].dt.strftime("%Y-%m-%d"),
            y=hist["correction_pct"], marker_color="#ef5350", opacity=0.7,
            yaxis="y",
        ))
        if "volume_ratio" in hist.columns:
            fig3.add_trace(go.Scatter(
                name="Vol Ratio", x=hist["scan_date"].dt.strftime("%Y-%m-%d"),
                y=hist["volume_ratio"], mode="lines+markers",
                line=dict(color="#1565C0", width=2), marker_size=7,
                yaxis="y2",
                hovertemplate="Vol ratio: %{y:.2f}x<extra></extra>",
            ))
        fig3.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1,
                       annotation_text="Vol = market avg", yref="y2")
        fig3.update_layout(
            yaxis=dict(title="Drop %", ticksuffix="%"),
            yaxis2=dict(title="Volume Ratio", overlaying="y", side="right", range=[0, 2.5]),
            height=280, margin=dict(t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── History table ─────────────────────────────────────────────────────────
    st.subheader("📋 All Scans for This Symbol")
    cols_show = ["scan_date", "signal", "composite_score", "fund_score", "tech_score",
                 "corr_score", "news_score", "pat_score", "top_pattern",
                 "correction_pct", "volume_ratio", "event_type", "current_price", "reason"]
    cols_show = [c for c in cols_show if c in hist.columns]
    hist_disp = hist[cols_show].sort_values("scan_date", ascending=False).copy()
    hist_disp["scan_date"] = hist_disp["scan_date"].dt.strftime("%Y-%m-%d")
    hist_disp["signal"] = hist_disp["signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}")
    hist_disp["composite_score"] = hist_disp["composite_score"].round(1)
    st.dataframe(hist_disp.style.hide(axis="index"), use_container_width=True, height=320)


# ── History ───────────────────────────────────────────────────────────────────

def page_history():
    st.title("📋 Scan History")

    from datetime import date, timedelta

    # Filters
    with st.expander("🔎 Filters", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 2])
        date_from = fc1.date_input("From", value=date.today() - timedelta(days=90))
        date_to = fc2.date_input("To", value=date.today())
        sig_choice = fc3.selectbox("Signal", ["All", "BUY_DIP", "WATCH", "AVOID"])
        sectors = ["All"] + db.get_all_sectors()
        sector_choice = fc4.selectbox("Sector", sectors)
        sym_search = fc5.text_input("Symbol contains", placeholder="NVDA")
        min_s = st.slider("Min Score", 0, 100, 0)

    df = db.get_history_df(
        date_from=date_from, date_to=date_to,
        signal=sig_choice if sig_choice != "All" else None,
        sector=sector_choice if sector_choice != "All" else None,
        min_score=min_s, symbol_search=sym_search or None,
    )

    st.caption(f"**{len(df)}** records found")

    if df.empty:
        st.info("No records match the current filters.")
        return

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "swing_history.csv", "text/csv", use_container_width=False)

    # Display
    disp_cols = ["scan_date", "symbol", "signal", "composite_score",
                 "fund_score", "tech_score", "corr_score", "news_score",
                 "correction_pct", "volume_ratio", "event_type",
                 "current_price", "sector", "reason"]
    disp_cols = [c for c in disp_cols if c in df.columns]
    display = df[disp_cols].copy()
    display["signal"] = display["signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}")
    display["composite_score"] = display["composite_score"].round(1)
    if "correction_pct" in display.columns:
        display["correction_pct"] = display["correction_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    if "volume_ratio" in display.columns:
        display["volume_ratio"] = display["volume_ratio"].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    if "current_price" in display.columns:
        display["current_price"] = display["current_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")

    st.dataframe(display.style.hide(axis="index"), use_container_width=True, height=500)

    # Signal distribution chart
    if not df.empty:
        st.divider()
        ch1, ch2 = st.columns(2)
        with ch1:
            sig_counts = df["signal"].value_counts().reset_index()
            sig_counts.columns = ["Signal", "Count"]
            color_map = {"BUY_DIP": _BUY_COLOR, "WATCH": _WATCH_COLOR, "AVOID": _AVOID_COLOR}
            fig = px.pie(sig_counts, names="Signal", values="Count",
                         color="Signal", color_discrete_map=color_map,
                         title="Signal Distribution")
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            if "sector" in df.columns:
                sec_counts = df[df["signal"].isin(["BUY_DIP", "WATCH"])]["sector"].value_counts().head(10).reset_index()
                sec_counts.columns = ["Sector", "Count"]
                fig2 = px.bar(sec_counts, x="Count", y="Sector", orientation="h",
                              title="Opportunities by Sector", color_discrete_sequence=["#2196F3"])
                fig2.update_layout(height=300, margin=dict(t=40, b=0), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2, use_container_width=True)


# ── Watchlist ─────────────────────────────────────────────────────────────────

def page_watchlist():
    st.title("⭐ Watchlist")
    st.caption("Track specific stocks across scans. Use the Scanner to rescan your watchlist anytime.")

    wl = db.get_watchlist()
    all_tracked = db.get_all_tracked_symbols()

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.subheader("Current Watchlist")
        if wl.empty:
            st.info("Watchlist is empty. Add symbols below.")
        else:
            # Get latest scores for watchlist symbols
            latest = db.get_latest_results()
            if not latest.empty:
                scores = latest[latest["symbol"].isin(wl["symbol"].tolist())]
                if not scores.empty:
                    merged = wl.merge(
                        scores[["symbol", "signal", "composite_score", "correction_pct", "current_price"]],
                        on="symbol", how="left"
                    )
                    merged["signal"] = merged["signal"].map(lambda s: f"{_SIG_EMOJI.get(s,'')} {s}" if pd.notna(s) else "—")
                    merged["composite_score"] = merged["composite_score"].round(1)
                    merged["correction_pct"] = merged["correction_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                    merged["current_price"] = merged["current_price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
                    merged["added_at"] = merged["added_at"].str[:10]
                    merged = merged.rename(columns={
                        "symbol": "Symbol", "added_at": "Added", "notes": "Notes",
                        "signal": "Signal", "composite_score": "Score",
                        "correction_pct": "Drop%", "current_price": "Price",
                    })
                    st.dataframe(merged.style.hide(axis="index"), use_container_width=True, height=380)
                else:
                    _wl_simple(wl)
            else:
                _wl_simple(wl)

            # Remove symbol
            remove_sym = st.selectbox("Remove symbol", ["—"] + wl["symbol"].tolist())
            if remove_sym != "—" and st.button("🗑️ Remove from Watchlist"):
                db.remove_from_watchlist(remove_sym)
                st.rerun()

    with col_b:
        st.subheader("Add Symbols")
        new_sym = st.text_input("Symbol", placeholder="NVDA").upper().strip()
        new_notes = st.text_input("Notes (optional)", placeholder="Watching for bounce")
        if st.button("⭐ Add to Watchlist", use_container_width=True, type="primary"):
            if new_sym:
                db.add_to_watchlist(new_sym, new_notes)
                st.success(f"Added {new_sym}")
                st.rerun()
            else:
                st.error("Enter a symbol.")

        st.divider()
        st.subheader("Add from Tracked Symbols")
        not_in_wl = [s for s in all_tracked if s not in (wl["symbol"].tolist() if not wl.empty else [])]
        bulk = st.multiselect("Pick symbols to add", not_in_wl[:100])
        if bulk and st.button("⭐ Add Selected"):
            for s in bulk:
                db.add_to_watchlist(s)
            st.success(f"Added {len(bulk)} symbols")
            st.rerun()

        st.divider()
        st.info("💡 Run a quick rescan of your watchlist from the **🔍 Scanner** page by selecting **Watchlist** as the universe.")


def _nav_to_deep_dive(symbol: str):
    """Navigate to Deep Dive for the given symbol, triggering auto-analysis."""
    st.session_state["deep_dive_nav_symbol"] = symbol
    st.session_state["dd_auto_analyze"] = True
    # Use _nav_request so the transfer to nav_page happens before the radio renders
    st.session_state["_nav_request"] = "🔎  Deep Dive"
    st.rerun()


def _selectable_table(disp: pd.DataFrame, height: int = 320, key: str = "") -> None:
    """Render a styled dataframe with a Quick Deep Dive selector below it."""
    st.dataframe(_style_table(disp), use_container_width=True, height=height)
    if not disp.empty and "Symbol" in disp.columns:
        symbols = disp["Symbol"].tolist()
        sel_c, btn_c = st.columns([3, 1])
        chosen = sel_c.selectbox(
            "Select ticker for Deep Dive",
            symbols,
            key=f"qd_sel_{key}",
            label_visibility="collapsed",
        )
        if btn_c.button("🔎 Deep Dive", use_container_width=True, key=f"qd_btn_{key}", type="primary"):
            _nav_to_deep_dive(chosen)


def _wl_simple(wl: pd.DataFrame):
    display = wl.copy()
    display["added_at"] = display["added_at"].str[:10]
    st.dataframe(display.rename(columns={"symbol": "Symbol", "added_at": "Added", "notes": "Notes"}).style.hide(axis="index"),
                 use_container_width=True, height=340)


# ── Deep Dive ─────────────────────────────────────────────────────────────────

def _render_dd_content(symbol: str, payload: dict) -> None:
    """Render all Deep Dive sections from a cached analysis payload."""
    result      = payload["result"]
    fund        = payload["fund"]
    tech        = payload["tech"]
    corr_result = payload["corr_result"]
    news        = payload["news"]
    pattern     = payload["pattern"]
    data        = payload["data"]
    dark        = st.session_state.get("dark_mode", False)
    bg_color    = "#0d1117" if dark else "white"
    grid_color  = "#2d333b" if dark else "#f0f0f0"
    spike_color = "#6e7681" if dark else "#999"

    # ── Signal banner ─────────────────────────────────────────────────────────
    sig_emoji = _SIG_EMOJI.get(result.signal, "⚪")
    sig_color = {"BUY_DIP": _BUY_COLOR, "WATCH": _WATCH_COLOR, "AVOID": _AVOID_COLOR}.get(result.signal, "#999")
    st.markdown(
        f'<div style="border-left:6px solid {sig_color};padding:10px 16px;border-radius:6px;'
        f'background:#fafafa;margin-bottom:8px">'
        f'<span style="font-size:1.4rem;font-weight:700">{sig_emoji} {result.signal}</span>'
        f'&nbsp;&nbsp;<span style="font-size:1.2rem">Score: {result.composite_score:.1f}/100</span><br>'
        f'<span style="color:#555;font-size:0.9rem">{result.reason}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Signal gauge + KPI row ────────────────────────────────────────────────
    g_col, k_col = st.columns([1, 3], gap="large")
    with g_col:
        st.plotly_chart(_signal_gauge(result.composite_score, f"{symbol} — Composite Score"),
                        use_container_width=True)
    with k_col:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📊 Fundamental", f"{fund.score:.0f}/100",
                  help="Scores ROE, debt/equity, revenue growth, free cash flow, and earnings growth. Weight: 30%")
        c2.metric("📈 Technical", f"{tech.score:.0f}/100",
                  help="Scores SMA 50/200 position, ADX trend strength, RSI momentum, MACD cross, OBV, higher-highs/lows. Weight: 25%")
        c3.metric("📉 Correction", f"{corr_result.score:.0f}/100",
                  delta=f"{corr_result.correction_pct:.1%} drop", delta_color="inverse",
                  help="Scores correction depth (5–15% ideal), volume ratio during drop (<0.85x = healthy), proximity to support, RSI oversold. Weight: 22%")
        c4.metric("📰 News", f"{news.score:.0f}/100",
                  help="Classifies news as macro/sector/unrelated (good) vs fundamental deterioration (bad). Penalises analyst downgrades. Weight: 8%")
        c5.metric("🔷 Pattern", f"{pattern.score:.0f}/100",
                  help="Detects 8 chart patterns (Double Bottom, Bull Flag, Cup & Handle etc.). 50 = neutral, >50 = bullish setups, <50 = bearish. Weight: 15%")

    st.divider()

    # ── Price chart + S/R overlay ─────────────────────────────────────────────
    col_chart, col_pat = st.columns([3, 1], gap="large")

    with col_chart:
        tf_options = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        tf_key = f"dd_tf_{symbol}"
        if tf_key not in st.session_state:
            st.session_state[tf_key] = "3M"

        tf_row = st.columns([1, 1, 1, 1, 6])
        for i, tf_label in enumerate(tf_options):
            btn_type = "primary" if st.session_state[tf_key] == tf_label else "secondary"
            if tf_row[i].button(tf_label, use_container_width=True, type=btn_type, key=f"tf_{symbol}_{tf_label}"):
                st.session_state[tf_key] = tf_label
                st.rerun()

        tf_bars = tf_options[st.session_state[tf_key]]
        st.subheader(f"📊 Price + S/R  ·  {st.session_state[tf_key]} view",
                     help="Green dashed = support levels · Red dashed = resistance · Purple solid = pattern key level · Orange/purple lines = SMA 50/200")

        hist = data.history
        if hist is not None and len(hist) >= 10:
            n_bars = min(tf_bars, len(hist))
            window = hist.iloc[-n_bars:]
            close_full = hist["Close"].astype(float)
            sma50  = close_full.rolling(50).mean().reindex(window.index)
            sma200 = close_full.rolling(200).mean().reindex(window.index)

            fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                                vertical_spacing=0.03, shared_xaxes=True)

            fig.add_trace(go.Candlestick(
                x=window.index,
                open=window["Open"], high=window["High"],
                low=window["Low"],   close=window["Close"],
                name="Price",
                increasing_line_color="#27ae60", increasing_fillcolor="#27ae60",
                decreasing_line_color="#e74c3c", decreasing_fillcolor="#e74c3c",
                showlegend=False,
            ), row=1, col=1)

            sma50_clean = sma50.dropna()
            if not sma50_clean.empty:
                fig.add_trace(go.Scatter(
                    x=sma50_clean.index, y=sma50_clean, mode="lines", name="SMA 50",
                    line=dict(color="#FF9800", width=1.5),
                    hovertemplate="SMA50: $%{y:.2f}<extra></extra>",
                ), row=1, col=1)

            sma200_clean = sma200.dropna()
            if not sma200_clean.empty:
                fig.add_trace(go.Scatter(
                    x=sma200_clean.index, y=sma200_clean, mode="lines", name="SMA 200",
                    line=dict(color="#9C27B0", width=1.5),
                    hovertemplate="SMA200: $%{y:.2f}<extra></extra>",
                ), row=1, col=1)

            for lv in pattern.sr_levels:
                if lv.level_type == "support":
                    color, dash, prefix = "#27ae60", "dot", "S"
                elif lv.level_type == "resistance":
                    color, dash, prefix = "#e74c3c", "dot", "R"
                else:
                    color, dash, prefix = "#f39c12", "dot", "S/R"
                label = f"{prefix} ${lv.price:.2f}"
                if lv.fib_label:
                    label += f" [{lv.fib_label}]"
                    dash = "dashdot"
                fig.add_hline(y=lv.price, line_dash=dash, line_color=color, line_width=1.5,
                              annotation_text=label, annotation_position="right",
                              annotation_font_size=10, annotation_font_color=color,
                              row=1, col=1)

            # Row 1 — Pattern key levels: shaded band + bold line + triangle markers
            for pm in pattern.patterns_detected:
                if not pm.key_price:
                    continue
                kcolor = "#8e44ad" if pm.signal == "bullish" else "#c0392b"
                sig_icon = "▲" if pm.signal == "bullish" else "▼"
                # Highlight band ±0.8% around key level
                fig.add_hrect(
                    y0=pm.key_price * 0.992, y1=pm.key_price * 1.008,
                    fillcolor=kcolor, opacity=0.12, line_width=0,
                    row=1, col=1,
                )
                # Bold line with coloured annotation box
                fig.add_hline(
                    y=pm.key_price, line_dash="solid", line_color=kcolor, line_width=2.5,
                    annotation_text=f" {sig_icon} {pm.name[:14]} ({pm.confidence:.0%}) ",
                    annotation_position="left",
                    annotation_bgcolor=kcolor,
                    annotation_font_size=10, annotation_font_color="white",
                    row=1, col=1,
                )
                # Triangle markers at bars touching the level
                ptol = 0.015
                if pm.signal == "bullish":
                    mask = (window["Low"] - pm.key_price).abs() / max(pm.key_price, 0.01) < ptol
                    y_mark = window.loc[mask, "Low"] * 0.993
                    msym = "triangle-up"
                else:
                    mask = (window["High"] - pm.key_price).abs() / max(pm.key_price, 0.01) < ptol
                    y_mark = window.loc[mask, "High"] * 1.007
                    msym = "triangle-down"
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=window.index[mask], y=y_mark, mode="markers",
                        name=f"{sig_icon} {pm.name}",
                        marker=dict(symbol=msym, size=13, color=kcolor,
                                    line=dict(color="white", width=1.5)),
                        showlegend=True,
                    ), row=1, col=1)

            vol_colors = [
                "#27ae60" if float(c) >= float(o) else "#e74c3c"
                for c, o in zip(window["Close"], window["Open"])
            ]
            fig.add_trace(go.Bar(
                x=window.index, y=window["Volume"],
                marker_color=vol_colors, name="Volume",
                opacity=0.7, showlegend=False,
                hovertemplate="Vol: %{y:,.0f}<extra></extra>",
            ), row=2, col=1)

            fig.update_layout(
                height=560, margin=dict(t=10, b=10, l=10, r=150),
                hovermode="x unified",
                spikedistance=200, hoverdistance=50,
                plot_bgcolor=bg_color, paper_bgcolor=bg_color,
                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                            font=dict(size=11)),
                xaxis_rangeslider_visible=False,
            )
            # Crosshair: spike lines on both axes
            fig.update_xaxes(showgrid=True, gridcolor=grid_color,
                             showspikes=True, spikemode="across", spikesnap="cursor",
                             spikecolor=spike_color, spikethickness=1, spikedash="dot",
                             row=1, col=1)
            fig.update_xaxes(showgrid=True, gridcolor=grid_color,
                             showspikes=True, spikemode="across", spikesnap="cursor",
                             spikecolor=spike_color, spikethickness=1, spikedash="dot",
                             row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", showgrid=True, gridcolor=grid_color,
                             showspikes=True, spikecolor=spike_color,
                             spikethickness=1, spikedash="dot",
                             row=1, col=1)
            fig.update_yaxes(title_text="Volume", showgrid=False, row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    with col_pat:
        st.subheader("📐 Detected Patterns",
                     help="▲ = bullish (buy signal), ▼ = bearish (caution). Key level = price to watch for breakout/breakdown. Confidence = how cleanly the pattern fits.")
        if pattern.patterns_detected:
            for pm in sorted(pattern.patterns_detected, key=lambda x: -x.confidence):
                icon  = "▲" if pm.signal == "bullish" else "▼"
                color = "green" if pm.signal == "bullish" else "red"
                st.markdown(f"**:{color}[{icon} {pm.name}]**  `{pm.confidence:.0%}`")
                if pm.key_price:
                    st.caption(f"Key level: **${pm.key_price:.2f}**")
                st.caption(pm.description)
                st.divider()
        else:
            st.info("No patterns detected in current data window.")

        st.subheader("🔵 S/R Levels",
                     help="Support (green) = price floor the stock tends to bounce from. Resistance (red) = ceiling it tends to stall at. Fibonacci levels are natural retracement zones (38.2%, 50%, 61.8%). Strength = touch count + volume + recency.")
        for lv in pattern.sr_levels[:6]:
            icon = "🟢" if lv.level_type == "support" else ("🔴" if lv.level_type == "resistance" else "🟡")
            fib  = f" `{lv.fib_label}`" if lv.fib_label else ""
            st.markdown(f"{icon} **${lv.price:.2f}**{fib}  str={lv.strength:.0f}")

    st.divider()

    # ── Risk / Reward calculator ───────────────────────────────────────────────
    st.subheader("⚖️ Risk / Reward Calculator",
                 help="Auto-populated from nearest S/R levels. Adjust entry, stop, and target to calculate your trade's risk:reward ratio.")
    current_px = result.current_price or 0.0
    nearest_sup = pattern.nearest_support_price
    nearest_res = pattern.nearest_resistance_price
    stop_default   = float(nearest_sup)  if nearest_sup else round(current_px * 0.95, 2)
    target_default = float(nearest_res) if nearest_res else round(current_px * 1.10, 2)

    rr_c1, rr_c2, rr_c3 = st.columns(3)
    rr_entry  = rr_c1.number_input("Entry Price ($)", value=float(current_px), min_value=0.0, step=0.01, format="%.2f")
    rr_stop   = rr_c2.number_input("Stop Loss ($)",   value=stop_default,      min_value=0.0, step=0.01, format="%.2f",
                                   help="Suggested: nearest S/R support below current price")
    rr_target = rr_c3.number_input("Profit Target ($)", value=target_default,  min_value=0.0, step=0.01, format="%.2f",
                                   help="Suggested: nearest S/R resistance above current price")

    if rr_entry > 0 and rr_stop > 0 and rr_target > rr_entry > rr_stop:
        risk       = rr_entry - rr_stop
        reward     = rr_target - rr_entry
        rr_ratio   = reward / risk
        risk_pct   = risk   / rr_entry * 100
        reward_pct = reward / rr_entry * 100

        out1, out2, out3, out4 = st.columns(4)
        out1.metric("Risk per Share",   f"${risk:.2f}",   delta=f"-{risk_pct:.1f}% from entry",   delta_color="inverse")
        out2.metric("Reward per Share", f"${reward:.2f}", delta=f"+{reward_pct:.1f}% from entry")
        out3.metric("R:R Ratio", f"{rr_ratio:.1f} : 1",
                    help="For every $1 risked, expected gain. Aim for 2:1 or better.")
        out4.metric("Position Risk", f"{risk_pct:.1f}%",
                    help="Percentage of entry price at risk if stop is hit.")

        if rr_ratio >= 2.0:
            st.success(
                f"Favorable setup — R:R is {rr_ratio:.1f}:1. "
                f"For every dollar you risk, the potential reward is {rr_ratio:.1f} dollars. "
                f"Entry at {rr_entry:.2f}, stop at {rr_stop:.2f}, target at {rr_target:.2f}."
            )
        elif rr_ratio >= 1.0:
            st.warning(
                f"Marginal setup — R:R is {rr_ratio:.1f}:1, which is below the recommended 2:1 minimum. "
                f"Consider tightening your stop loss or raising the profit target to improve the ratio."
            )
        else:
            st.error(
                f"Poor risk/reward — R:R is {rr_ratio:.1f}:1. "
                f"You are risking {risk_pct:.1f}% of your entry to gain only {reward_pct:.1f}%. "
                f"This setup does not meet minimum trade criteria. Reconsider entry or targets."
            )
    elif rr_target > 0 and rr_stop > 0 and not (rr_target > rr_entry > rr_stop):
        st.caption("Tip: Stop Loss must be below Entry, and Profit Target must be above Entry, to calculate R:R.")

    st.divider()

    # ── Dimension score radar ─────────────────────────────────────────────────
    col_radar, col_detail = st.columns([1, 2], gap="large")

    with col_radar:
        st.subheader("🕸 Score Radar")
        categories = ["Fundamental", "Technical", "Correction", "News", "Pattern"]
        values     = [fund.score, tech.score, corr_result.score, news.score, pattern.score]
        fig_r = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(33,150,243,0.15)",
            line=dict(color="#2196F3", width=2), marker=dict(size=6),
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=320, margin=dict(t=20, b=10, l=20, r=20), showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_detail:
        st.subheader("📋 Analysis Detail")
        tab_overview, tab_fund, tab_tech, tab_corr, tab_news, tab_val, tab_analyst, tab_own = st.tabs(
            ["Overview", "Fundamental", "Technical", "Correction", "News", "Valuation", "Analyst", "Ownership"]
        )

        with tab_overview:
            info = data.info
            fi   = data.fast_info
            # Company description
            summary = info.get("longBusinessSummary", "")
            if summary:
                st.markdown(f"**{info.get('longName', symbol)}** — {info.get('sector','')}, {info.get('industry','')}")
                st.caption(summary[:500] + ("…" if len(summary) > 500 else ""))
                st.divider()
            # Key company stats
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Market Cap",   _fmt_cap(info.get("marketCap") or fi.get("market_cap")))
            mc2.metric("Employees",    f"{info['fullTimeEmployees']:,}" if info.get("fullTimeEmployees") else "—")
            mc3.metric("Avg Vol (3M)", _fmt_vol(info.get("averageVolume")),
                       help="Average daily trading volume over the past 3 months")
            mc4.metric("Exchange",     info.get("exchange", "—"))
            mc5, mc6, mc7, mc8 = st.columns(4)
            mc5.metric("Country",  info.get("country", "—"))
            mc6.metric("Currency", info.get("currency", "USD"))
            mc7.metric("Avg Vol (10d)", _fmt_vol(info.get("averageVolume10days")),
                       help="Average daily volume over the past 10 days vs. 3-month average — elevated = increased interest")
            site = info.get("website", "")
            mc8.metric("Website", site[:30] if site else "—")
            # 52-week range
            yr_h = fi.get("year_high")
            yr_l = fi.get("year_low")
            px   = result.current_price
            if yr_h or yr_l:
                st.divider()
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("52-Week High", f"${yr_h:.2f}" if yr_h else "—")
                r2.metric("52-Week Low",  f"${yr_l:.2f}" if yr_l else "—")
                if yr_h and px:
                    r3.metric("Off 52W High", f"-{(yr_h-px)/yr_h*100:.1f}%", delta_color="off",
                              help="How far below the 52-week high the stock is currently trading")
                if yr_l and px:
                    r4.metric("Above 52W Low", f"+{(px-yr_l)/yr_l*100:.1f}%", delta_color="off")

        with tab_fund:
            if fund:
                # Verdict
                if fund.score >= 70:
                    st.success("**Bullish** — strong financial health supports the buy case.")
                elif fund.score >= 50:
                    st.info("**Neutral** — acceptable fundamentals; not a headwind to the trade.")
                else:
                    st.warning("**Bearish** — weak fundamentals add risk. Verify this is a temporary correction, not deterioration.")
                # Recovery thesis
                info_f = data.info
                t_eps = info_f.get("trailingEps")
                f_eps = info_f.get("forwardEps")
                fcf   = fund.free_cash_flow
                robs = []
                if fund.score >= 60: robs.append("Fundamentals are solid enough to survive the dip and recover.")
                if fund.revenue_growth_yoy and fund.revenue_growth_yoy > 0:
                    robs.append(f"Revenue is growing ({fund.revenue_growth_yoy:.1%}) — earnings power is intact.")
                if fcf and fcf > 0: robs.append("Positive free cash flow — the company is self-funding and not burning cash during this correction.")
                if t_eps and t_eps > 0: robs.append(f"EPS is positive ({t_eps:.2f}) — the business is currently profitable.")
                if f_eps and t_eps and f_eps > t_eps: robs.append(f"Forward EPS ({f_eps:.2f}) > trailing ({t_eps:.2f}) — analysts expect earnings to grow, validating recovery.")
                de = fund.debt_to_equity
                if de and de < 1.0: robs.append("Low leverage means the company can sustain itself through the dip without refinancing risk.")
                for o in robs[:4]:
                    st.caption(f"• {o}")
                # Observations
                obs = []
                roe = fund.return_on_equity
                if roe is not None:
                    if roe > 0.20:   obs.append(f"Excellent ROE {roe:.1%} — management creates strong shareholder value.")
                    elif roe > 0.10: obs.append(f"Solid ROE {roe:.1%} — profitable and efficient.")
                    elif roe > 0:    obs.append(f"Below-average ROE {roe:.1%} — limited equity returns.")
                    else:            obs.append(f"Negative ROE {roe:.1%} — currently unprofitable; verify it is temporary.")
                if de is not None:
                    if de < 0.3:   obs.append("Very low debt — resilient balance sheet.")
                    elif de < 1.0: obs.append(f"Moderate leverage D/E {de:.1f} — manageable.")
                    else:          obs.append(f"High leverage D/E {de:.1f} — rate-sensitive; watch carefully.")
                rg = fund.revenue_growth_yoy
                if rg is not None:
                    if rg > 0.15:  obs.append(f"Strong revenue growth {rg:.1%} YoY.")
                    elif rg > 0:   obs.append(f"Moderate growth {rg:.1%} YoY.")
                    else:          obs.append(f"Revenue declining {rg:.1%} — investigate if structural or cyclical.")
                for o in obs:
                    st.caption(f"• {o}")
                st.divider()
                # Core metrics row
                cols = st.columns(3)
                cols[0].metric("ROE", f"{fund.return_on_equity:.1%}" if fund.return_on_equity else "—")
                cols[1].metric("D/E Ratio", f"{fund.debt_to_equity:.2f}" if fund.debt_to_equity else "—")
                cols[2].metric("Gross Margin", f"{fund.gross_margins:.1%}" if fund.gross_margins else "—")
                cols2 = st.columns(3)
                cols2[0].metric("Rev Growth YoY", f"{fund.revenue_growth_yoy:.1%}" if fund.revenue_growth_yoy else "—")
                cols2[1].metric("Earnings Growth", f"{fund.earnings_growth:.1%}" if fund.earnings_growth else "—")
                cols2[2].metric("P/E", f"{fund.pe_ratio:.1f}" if fund.pe_ratio else "—")
                # EPS row
                st.caption("**Earnings per Share**")
                e1, e2, e3 = st.columns(3)
                e1.metric("Trailing EPS", f"${t_eps:.2f}" if t_eps else "—",
                          help="Earnings per share over the trailing 12 months. Positive = profitable.")
                e2.metric("Forward EPS", f"${f_eps:.2f}" if f_eps else "—",
                          help="Estimated EPS for the next 12 months. Rising > trailing = growth trajectory.")
                eps_growth = (f_eps - t_eps) / abs(t_eps) if f_eps and t_eps and t_eps != 0 else None
                e3.metric("EPS Growth (est.)", f"{eps_growth:+.1%}" if eps_growth is not None else "—",
                          help="Expected change from trailing to forward EPS — proxy for near-term earnings momentum.")
                # Debt health row
                st.caption("**Debt & Liquidity**")
                d1, d2, d3, d4 = st.columns(4)
                total_debt = info_f.get("totalDebt")
                total_cash = info_f.get("totalCash")
                net_debt   = (total_debt - total_cash) if total_debt and total_cash else None
                d1.metric("Total Debt",    _fmt_cap(total_debt),
                          help="Total interest-bearing debt on the balance sheet.")
                d2.metric("Total Cash",    _fmt_cap(total_cash),
                          help="Cash and short-term investments. High cash = buffer against downturn.")
                d3.metric("Net Debt",      _fmt_cap(net_debt) if net_debt else "—",
                          help="Total Debt minus Cash. Negative = net cash position (very healthy).")
                d4.metric("Current Ratio", f"{fund.current_ratio:.2f}" if fund.current_ratio else "—",
                          help="Current assets / current liabilities. > 1.5 = strong short-term liquidity.")

        with tab_tech:
            if tech:
                # Verdict
                bullish_signals = sum([tech.price_above_200sma, tech.price_above_50sma_before_drop,
                                       tech.macd_bullish_before_drop, tech.higher_highs_higher_lows,
                                       tech.obv_uptrend])
                if tech.score >= 70:
                    st.success(f"**Bullish** — {bullish_signals}/5 technical indicators confirm the uptrend before this pullback.")
                elif tech.score >= 50:
                    st.info(f"**Neutral** — mixed signals ({bullish_signals}/5 bullish). Uptrend may be weakening; watch for confirmation.")
                else:
                    st.warning(f"**Bearish** — only {bullish_signals}/5 technical indicators are positive. The prior trend may be broken.")
                obs = []
                if tech.price_above_200sma:  obs.append("Trading above the 200-day SMA — long-term uptrend intact.")
                else:                         obs.append("Below the 200-day SMA — long-term trend is down; adds risk.")
                if tech.higher_highs_higher_lows: obs.append("Pattern of higher highs and higher lows — momentum structure is healthy.")
                if tech.adx_value and tech.adx_value > 25: obs.append(f"ADX {tech.adx_value:.0f} — strong trend strength (>25 = trending market).")
                elif tech.adx_value:                        obs.append(f"ADX {tech.adx_value:.0f} — weak trend (<25 = choppy/ranging market).")
                if tech.macd_bullish_before_drop: obs.append("MACD was bullish before the drop — momentum was positive pre-correction.")
                for o in obs[:3]:
                    st.caption(f"• {o}")
                st.divider()
                cols = st.columns(3)
                cols[0].metric("ADX", f"{tech.adx_value:.1f}" if tech.adx_value else "—")
                cols[1].metric("RSI at peak", f"{tech.rsi_at_peak:.1f}" if tech.rsi_at_peak else "—")
                cols[2].metric("MACD Bullish", "✅" if tech.macd_bullish_before_drop else "❌")
                cols2 = st.columns(3)
                cols2[0].metric("Above 200 SMA", "✅" if tech.price_above_200sma else "❌")
                cols2[1].metric("Above 50 SMA", "✅" if tech.price_above_50sma_before_drop else "❌")
                cols2[2].metric("HH/HL", "✅" if tech.higher_highs_higher_lows else "❌")

        with tab_corr:
            if corr_result:
                # Verdict
                depth_ok = 0.05 <= corr_result.correction_pct <= 0.15
                vol_ok = corr_result.volume_ratio < 0.90
                macro_ok = corr_result.is_macro_correlated or corr_result.spy_correlation and corr_result.spy_correlation >= 0.55
                if corr_result.score >= 70:
                    st.success("**Bullish** — healthy, orderly pullback with the right characteristics to buy the dip.")
                elif corr_result.score >= 50:
                    st.info("**Neutral** — correction is underway but not all quality signals are present yet.")
                else:
                    st.warning("**Bearish** — correction shows signs of genuine selling pressure, not a simple pullback.")
                obs = []
                p = corr_result.correction_pct
                if 0.05 <= p <= 0.15: obs.append(f"Drop of {p:.1%} is in the sweet spot (5–15%) for a buy-the-dip setup.")
                elif p < 0.05:        obs.append(f"Drop of {p:.1%} is too small — may not yet offer meaningful entry advantage.")
                else:                 obs.append(f"Drop of {p:.1%} is larger than the 15% sweet spot — verify the cause is not fundamental.")
                vr = corr_result.volume_ratio
                if vr < 0.85:   obs.append(f"Volume ratio {vr:.2f}x — low-volume selloff. Sellers are not aggressive; likely profit-taking.")
                elif vr < 1.10: obs.append(f"Volume ratio {vr:.2f}x — near-average volume. Neutral signal.")
                else:           obs.append(f"Volume ratio {vr:.2f}x — elevated volume during decline. Could signal genuine distribution.")
                if corr_result.spy_correlation is not None:
                    if corr_result.spy_correlation >= 0.65: obs.append(f"High SPY correlation ({corr_result.spy_correlation:.2f}) — selloff is macro-driven, not company-specific. Good for a bounce.")
                    else:                                   obs.append(f"Low SPY correlation ({corr_result.spy_correlation:.2f}) — selloff diverges from the market; investigate the cause.")
                for o in obs[:3]:
                    st.caption(f"• {o}")
                st.divider()
                cols = st.columns(3)
                cols[0].metric("Drop", f"{corr_result.correction_pct:.1%}")
                cols[1].metric("Vol Ratio", f"{corr_result.volume_ratio:.2f}x")
                cols[2].metric("RSI at Bottom", f"{corr_result.rsi_at_bottom:.1f}" if corr_result.rsi_at_bottom else "—")
                if corr_result.spy_correlation is not None:
                    st.metric("SPY Correlation", f"{corr_result.spy_correlation:.2f}",
                              help="≥0.65 = macro-driven selloff")

        with tab_news:
            if news:
                # Verdict
                event = news.event_type
                if event in ("macro", "sector", "unrelated"):
                    st.success(f"**Bullish** — event type '{event}' means the drop is NOT caused by company-specific bad news. The business is intact.")
                elif event == "mixed":
                    st.info("**Neutral** — mixed signals. Some macro/sector triggers but also some company-specific mentions. Verify carefully.")
                elif event == "fundamental":
                    st.error("**Bearish** — news suggests company-specific deterioration (earnings miss, guidance cut, scandal). This is NOT a simple dip to buy.")
                else:
                    st.info("**Neutral** — insufficient news data to classify the trigger.")
                obs = []
                if news.downgrade_count > 0:
                    obs.append(f"{news.downgrade_count} analyst downgrade(s) recently — adds near-term selling pressure.")
                if news.pre_earnings:
                    obs.append("Earnings event upcoming — avoid initiating position before earnings; wait for the print.")
                if news.confidence >= 0.7:
                    obs.append(f"High classification confidence ({news.confidence:.0%}) — signal is reliable.")
                elif news.confidence >= 0.5:
                    obs.append(f"Moderate confidence ({news.confidence:.0%}) — treat classification as indicative, not definitive.")
                for o in obs:
                    st.caption(f"• {o}")
                st.divider()
                cols = st.columns(3)
                cols[0].metric("Event Type", news.event_type)
                cols[1].metric("Confidence", f"{news.confidence:.0%}")
                cols[2].metric("Downgrades", news.downgrade_count)
                if news.top_headlines:
                    st.markdown("**Recent headlines:**")
                    for h in news.top_headlines[:4]:
                        st.caption(f"• {h[:90]}")

        with tab_val:
            info = data.info
            pe   = info.get("trailingPE")
            fpe  = info.get("forwardPE")
            peg  = info.get("pegRatio")
            pb   = info.get("priceToBook")
            ev_e = info.get("enterpriseToEbitda")
            target_mean = info.get("targetMeanPrice")
            # Narrative
            val_obs = []
            if peg and peg < 1.0:      val_obs.append(f"PEG ratio {peg:.2f} is below 1 — stock may be undervalued relative to its growth rate.")
            elif peg and peg < 2.0:    val_obs.append(f"PEG ratio {peg:.2f} — fair value range; growth is priced in but not stretched.")
            elif peg:                   val_obs.append(f"PEG ratio {peg:.2f} — above 2 suggests the stock is pricing in high expectations.")
            if pe and fpe and fpe < pe: val_obs.append(f"Forward P/E ({fpe:.1f}) is lower than trailing P/E ({pe:.1f}) — analysts expect earnings growth ahead.")
            elif fpe and pe and fpe > pe * 1.1: val_obs.append(f"Forward P/E ({fpe:.1f}) is higher than trailing — earnings expected to dip next year.")
            if target_mean and current_px:
                upside = (target_mean - current_px) / current_px * 100
                if upside > 15:    val_obs.append(f"Analyst consensus target implies {upside:.0f}% upside — strong buy case from Street.")
                elif upside > 5:   val_obs.append(f"Analyst consensus target implies {upside:.0f}% upside — modest upside.")
                elif upside >= 0:  val_obs.append(f"Analyst target near current price ({upside:.0f}% upside) — Street sees limited near-term upside.")
                else:              val_obs.append(f"Analyst target below current price ({upside:.0f}%) — Street is more cautious than the current price implies.")
            if val_obs:
                for o in val_obs:
                    st.caption(f"• {o}")
                st.divider()
            cols = st.columns(3)
            cols[0].metric("Trailing P/E", f"{pe:.1f}" if pe else "—",
                           help="Price / trailing 12-month earnings. Compare to sector peers.")
            cols[1].metric("Forward P/E",  f"{fpe:.1f}" if fpe else "—",
                           help="Price / estimated next-year earnings. Lower than trailing = earnings growing.")
            cols[2].metric("PEG Ratio",    f"{peg:.2f}" if peg else "—",
                           help="P/E divided by growth rate. < 1 = potentially undervalued, < 2 = fair, > 2 = pricey.")
            cols2 = st.columns(3)
            cols2[0].metric("P/B", f"{pb:.2f}" if pb else "—",
                            help="Price / book value. < 1 = trading below book (asset-value support).")
            cols2[1].metric("EV/EBITDA", f"{ev_e:.1f}" if ev_e else "—",
                            help="Enterprise value / EBITDA. < 10 = reasonable; > 20 = expensive.")
            if target_mean and current_px:
                upside = (target_mean - current_px) / current_px * 100
                cols2[2].metric("Analyst Target", f"${target_mean:.2f}",
                                delta=f"{upside:+.1f}% vs current",
                                delta_color="normal" if upside >= 0 else "inverse",
                                help=f"Mean analyst price target across {info.get('numberOfAnalystOpinions', '?')} opinions.")
            else:
                cols2[2].metric("Analyst Target", "—")

        with tab_analyst:
            rs = data.recommendations_summary
            if rs is not None and not rs.empty:
                row = rs.iloc[0]
                categories_rec = ["strongBuy", "buy", "hold", "sell", "strongSell"]
                labels_rec     = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
                colors_rec     = ["#1a7a30", "#4CAF50", "#FF9800", "#e74c3c", "#8b0000"]
                counts = [int(row.get(c, 0)) for c in categories_rec]
                total_analysts = sum(counts)
                buy_count  = counts[0] + counts[1]
                sell_count = counts[3] + counts[4]
                if total_analysts > 0:
                    buy_pct = buy_count / total_analysts
                    if buy_pct >= 0.6:
                        st.success(f"**Bullish consensus** — {buy_count} of {total_analysts} analysts ({buy_pct:.0%}) rate it Buy or Strong Buy.")
                    elif sell_count / total_analysts >= 0.4:
                        st.warning(f"**Bearish consensus** — {sell_count} of {total_analysts} analysts rate it Sell or Strong Sell.")
                    else:
                        st.info(f"**Neutral consensus** — mixed views among {total_analysts} analysts covering this stock.")
                fig_rec = go.Figure(go.Bar(
                    x=labels_rec, y=counts, marker_color=colors_rec, opacity=0.85,
                    text=counts, textposition="outside",
                ))
                fig_rec.update_layout(height=220, margin=dict(t=10, b=10), showlegend=False,
                                      yaxis_title="# Analysts")
                st.plotly_chart(fig_rec, use_container_width=True)
            else:
                st.info("No recommendations data available.")

            ud = data.upgrades_downgrades
            if ud is not None and not ud.empty:
                st.caption("**Recent upgrades / downgrades**")
                ud_disp = ud.head(8).reset_index()
                st.dataframe(ud_disp.style.hide(axis="index"), use_container_width=True, height=220)

        with tab_own:
            info = data.info
            ins_pct     = info.get("heldPercentInsiders")
            inst_pct    = info.get("heldPercentInstitutions")
            short_pct   = info.get("shortPercentOfFloat")
            short_ratio = info.get("shortRatio")
            # Narrative
            own_obs = []
            if ins_pct and ins_pct > 0.10:  own_obs.append(f"Insider ownership is high ({ins_pct:.1%}) — management has significant skin in the game.")
            elif ins_pct and ins_pct > 0.02: own_obs.append(f"Moderate insider ownership ({ins_pct:.1%}) — some alignment with shareholders.")
            elif ins_pct is not None:         own_obs.append(f"Low insider ownership ({ins_pct:.1%}) — management may be less incentivized by share price.")
            if short_pct and short_pct > 0.20: own_obs.append(f"Heavy short interest ({short_pct:.1%} of float) — potential for a short squeeze if the stock recovers.")
            elif short_pct and short_pct > 0.10: own_obs.append(f"Elevated short interest ({short_pct:.1%}) — some bearish conviction from short sellers; monitor closely.")
            elif short_pct:                       own_obs.append(f"Low short interest ({short_pct:.1%}) — minimal bearish bets against the stock.")
            if short_ratio and short_ratio > 10: own_obs.append(f"Short ratio {short_ratio:.1f} days — high days-to-cover; shorts are vulnerable to a squeeze.")
            for o in own_obs:
                st.caption(f"• {o}")
            if own_obs:
                st.divider()
            cols = st.columns(4)
            cols[0].metric("Insider Ownership", f"{ins_pct:.1%}" if ins_pct else "—",
                           help="% held by insiders (execs, directors). > 5% shows skin in the game.")
            cols[1].metric("Institutional",     f"{inst_pct:.1%}" if inst_pct else "—",
                           help="% held by institutions (funds, ETFs). High = stable shareholder base.")
            cols[2].metric("Short Float %",     f"{short_pct:.1%}" if short_pct else "—",
                           help="% of float sold short. > 20% = heavily shorted; potential short squeeze fuel.")
            cols[3].metric("Short Ratio",       f"{short_ratio:.1f}d" if short_ratio else "—",
                           help="Days to cover at average volume. > 10 days = high short interest.")
            ih = data.institutional_holders
            if ih is not None and not ih.empty:
                st.caption("**Top institutional holders**")
                ih_cols = [c for c in ("Holder", "Shares", "% Out", "Value") if c in ih.columns]
                st.dataframe(ih[ih_cols].head(10).style.hide(axis="index"),
                             use_container_width=True, height=260)


def page_deep_dive():
    st.title("🔎 Deep Dive Analysis")
    st.caption("Full single-ticker analysis — chart with S/R & patterns, fundamentals, risk/reward, analyst consensus.")

    # Build dropdown options from previously analyzed symbols (persisted in session cache)
    cached_syms = sorted(k[9:] for k in st.session_state if k.startswith("dd_cache_"))

    # Consume any pending navigation/pick request BEFORE widgets render
    _pending = st.session_state.pop("_dd_pick_request", None)
    if not _pending:
        nav_sym = st.session_state.pop("deep_dive_nav_symbol", None)
        if nav_sym:
            _pending = nav_sym.upper().strip()
    auto_analyze = st.session_state.pop("dd_auto_analyze", False)

    _NEW_OPT = "＋ Enter new symbol…"
    # Include a pending symbol in options even before its first analysis
    if _pending and _pending not in cached_syms:
        dd_options = [_NEW_OPT, _pending] + cached_syms
    else:
        dd_options = [_NEW_OPT] + cached_syms

    if _pending:
        st.session_state["dd_select"] = _pending
    elif st.session_state.get("dd_select") not in dd_options:
        st.session_state["dd_select"] = _NEW_OPT

    col_sym, col_btn = st.columns([4, 1])
    selected = col_sym.selectbox(
        "Symbol", options=dd_options, key="dd_select", label_visibility="collapsed"
    )
    analyze = col_btn.button("🔍 Analyze", use_container_width=True, type="primary")

    if selected == _NEW_OPT:
        symbol = st.text_input(
            "Ticker symbol", placeholder="e.g. NVDA", key="dd_new_sym",
            label_visibility="collapsed"
        ).upper().strip()
    else:
        symbol = selected

    # Cache age + refresh row (shown only when a cached result exists)
    cache_key = f"dd_cache_{symbol}" if symbol else None
    cached = st.session_state.get(cache_key) if cache_key else None
    run_analysis = analyze or auto_analyze

    if cached:
        age_secs = (datetime.now() - cached["analyzed_at"]).total_seconds()
        h = int(age_secs // 3600)
        m = int((age_secs % 3600) // 60)
        age_str = f"{h}h {m}m ago" if h > 0 else (f"{m}m ago" if m > 0 else "just now")
        info_col, refresh_col = st.columns([5, 1])
        info_col.caption(f"Showing cached analysis · {age_str}")
        if refresh_col.button("🔄 Refresh", use_container_width=True, key="dd_refresh"):
            run_analysis = True

    if not symbol:
        st.info("Select a recently analyzed symbol from the dropdown, or choose **＋ Enter new symbol…** and type a ticker, then click **Analyze**.")
        return

    # Serve from cache when not re-running
    if cached and not run_analysis:
        _render_dd_content(symbol, cached)
        return

    if not run_analysis:
        st.info(f"Click **Analyze** to run a full analysis for **{symbol}**.")
        return

    from data.fetcher import fetch
    from scanner import correction, fundamental, news_classifier, patterns, scorer, technical

    with st.spinner(f"Fetching and analyzing **{symbol}**…"):
        deep_cfg = dict(config)
        deep_cfg["correction"] = dict(config.get("correction", {}))
        deep_cfg["correction"]["min_drop_pct"] = 0.005

        data = fetch(symbol, use_cache=False)
        if data is None:
            st.error(f"Could not fetch data for **{symbol}**. Check the symbol and try again.")
            return

        fi = data.fast_info
        fund        = fundamental.score(data, deep_cfg)
        corr_result, corr_days = correction.detect(data, deep_cfg)
        tech        = technical.score(data, deep_cfg, correction_days=corr_days)
        news        = news_classifier.classify(data, deep_cfg)
        pattern     = patterns.score(data, deep_cfg)
        result      = scorer.compute(
            symbol, fund, tech, corr_result, news, deep_cfg, pattern,
            fi.get("last_price"), fi.get("year_high"),
            data.info.get("sector"), data.info.get("industry"),
        )

    payload = {
        "result": result, "fund": fund, "tech": tech,
        "corr_result": corr_result, "news": news,
        "pattern": pattern, "data": data,
        "analyzed_at": datetime.now(), "deep_cfg": deep_cfg,
    }
    st.session_state[cache_key] = payload
    _render_dd_content(symbol, payload)


# ── Routing ───────────────────────────────────────────────────────────────────

if "Dashboard" in page:
    page_dashboard()
elif "Scanner" in page:
    page_scanner()
elif "Deep Dive" in page:
    page_deep_dive()
elif "Tracker" in page:
    page_tracker()
elif "History" in page:
    page_history()
elif "Watchlist" in page:
    page_watchlist()
