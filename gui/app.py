"""Swing Trading Scanner — Streamlit GUI."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
        background: #fafafa;
    }
    .stProgress > div > div > div { background-color: #28a745; }
    .block-container { padding-top: 1.5rem; }
    thead tr th { background: #f0f2f6 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Swing Scanner")
    st.divider()

    page = st.radio(
        "nav",
        ["🏠  Dashboard", "🔍  Scanner", "🔎  Deep Dive", "📊  Trend Tracker", "📋  History", "⭐  Watchlist"],
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

# ── Helpers ───────────────────────────────────────────────────────────────────

_SIG_EMOJI = {"BUY_DIP": "🟢", "WATCH": "🟡", "AVOID": "🔴"}
_SIG_COLOR = {"BUY_DIP": _BUY_COLOR, "WATCH": _WATCH_COLOR, "AVOID": _AVOID_COLOR}


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
            st.dataframe(_style_table(disp), use_container_width=True, height=280)
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
        st.dataframe(_style_table(disp), use_container_width=True, height=320)
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
    st.dataframe(_style_table(disp), use_container_width=True, height=420)

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


def _wl_simple(wl: pd.DataFrame):
    display = wl.copy()
    display["added_at"] = display["added_at"].str[:10]
    st.dataframe(display.rename(columns={"symbol": "Symbol", "added_at": "Added", "notes": "Notes"}).style.hide(axis="index"),
                 use_container_width=True, height=340)


# ── Deep Dive ─────────────────────────────────────────────────────────────────

def page_deep_dive():
    st.title("🔎 Deep Dive Analysis")
    st.caption("On-demand full analysis: price chart with S/R levels, pattern detection, and all dimension scores.")

    col_sym, col_btn = st.columns([4, 1])
    symbol = col_sym.text_input("Symbol", placeholder="NVDA", label_visibility="collapsed").upper().strip()
    analyze = col_btn.button("🔍 Analyze", use_container_width=True, type="primary")

    if not symbol:
        st.info("Enter a ticker symbol above and click **Analyze**.")
        return
    if not analyze:
        return

    from data.fetcher import fetch
    from scanner import correction, fundamental, news_classifier, patterns, scorer, technical

    with st.spinner(f"Fetching and analyzing **{symbol}**…"):
        deep_cfg = dict(config)
        deep_cfg["correction"] = dict(config.get("correction", {}))
        deep_cfg["correction"]["min_drop_pct"] = 0.005  # allow even tiny pullbacks

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
        # Timeline selector
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

            # Compute SMAs on full history then slice to window
            sma50  = close_full.rolling(50).mean().reindex(window.index)
            sma200 = close_full.rolling(200).mean().reindex(window.index)

            # Two-row subplot: price (75%) + volume (25%)
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.03,
                shared_xaxes=True,
            )

            # Row 1 — Candlestick
            fig.add_trace(go.Candlestick(
                x=window.index,
                open=window["Open"], high=window["High"],
                low=window["Low"],   close=window["Close"],
                name="Price",
                increasing_line_color="#27ae60", increasing_fillcolor="#27ae60",
                decreasing_line_color="#e74c3c", decreasing_fillcolor="#e74c3c",
                showlegend=False,
            ), row=1, col=1)

            # Row 1 — SMA 50
            sma50_clean = sma50.dropna()
            if not sma50_clean.empty:
                fig.add_trace(go.Scatter(
                    x=sma50_clean.index, y=sma50_clean,
                    mode="lines", name="SMA 50",
                    line=dict(color="#FF9800", width=1.5),
                    hovertemplate="SMA50: $%{y:.2f}<extra></extra>",
                ), row=1, col=1)

            # Row 1 — SMA 200
            sma200_clean = sma200.dropna()
            if not sma200_clean.empty:
                fig.add_trace(go.Scatter(
                    x=sma200_clean.index, y=sma200_clean,
                    mode="lines", name="SMA 200",
                    line=dict(color="#9C27B0", width=1.5),
                    hovertemplate="SMA200: $%{y:.2f}<extra></extra>",
                ), row=1, col=1)

            # Row 1 — S/R lines
            for lv in pattern.sr_levels:
                if lv.level_type == "support":
                    color, dash = "#27ae60", "dot"
                    prefix = "S"
                elif lv.level_type == "resistance":
                    color, dash = "#e74c3c", "dot"
                    prefix = "R"
                else:
                    color, dash = "#f39c12", "dot"
                    prefix = "S/R"
                label = f"{prefix} ${lv.price:.2f}"
                if lv.fib_label:
                    label += f" [{lv.fib_label}]"
                    dash = "dashdot"
                fig.add_hline(
                    y=lv.price, line_dash=dash, line_color=color, line_width=1.5,
                    annotation_text=label, annotation_position="right",
                    annotation_font_size=10, annotation_font_color=color,
                    row=1, col=1,
                )

            # Row 1 — Pattern key levels
            for pm in pattern.patterns_detected:
                if pm.key_price:
                    kcolor = "#8e44ad" if pm.signal == "bullish" else "#c0392b"
                    fig.add_hline(
                        y=pm.key_price, line_dash="solid", line_color=kcolor, line_width=2,
                        annotation_text=f"[{pm.name[:12]}] ${pm.key_price:.2f}",
                        annotation_position="left",
                        annotation_font_size=10, annotation_font_color=kcolor,
                        row=1, col=1,
                    )

            # Row 2 — Volume bars (green up-day, red down-day)
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
                height=540,
                margin=dict(t=10, b=10, l=10, r=140),
                hovermode="x unified",
                plot_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=1, col=1)
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", showgrid=True, gridcolor="#f0f0f0", row=1, col=1)
            fig.update_yaxes(title_text="Volume",   showgrid=False, row=2, col=1)
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
            icon  = "🟢" if lv.level_type == "support" else ("🔴" if lv.level_type == "resistance" else "🟡")
            fib   = f" `{lv.fib_label}`" if lv.fib_label else ""
            st.markdown(f"{icon} **${lv.price:.2f}**{fib}  str={lv.strength:.0f}")

    st.divider()

    # ── Dimension score radar ─────────────────────────────────────────────────
    col_radar, col_detail = st.columns([1, 2], gap="large")

    with col_radar:
        st.subheader("🕸 Score Radar")
        categories = ["Fundamental", "Technical", "Correction", "News", "Pattern"]
        values     = [fund.score, tech.score, corr_result.score, news.score, pattern.score]
        fig_r = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=6),
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=320, margin=dict(t=20, b=10, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_detail:
        st.subheader("📋 Analysis Detail")
        tab_fund, tab_tech, tab_corr, tab_news = st.tabs(["Fundamental", "Technical", "Correction", "News"])

        with tab_fund:
            if fund:
                st.metric("Score", f"{fund.score:.0f}/100")
                cols = st.columns(3)
                cols[0].metric("ROE", f"{fund.return_on_equity:.1%}" if fund.return_on_equity else "—")
                cols[1].metric("D/E", f"{fund.debt_to_equity:.2f}" if fund.debt_to_equity else "—")
                cols[2].metric("Gross Margin", f"{fund.gross_margins:.1%}" if fund.gross_margins else "—")
                cols2 = st.columns(3)
                cols2[0].metric("Rev Growth YoY", f"{fund.revenue_growth_yoy:.1%}" if fund.revenue_growth_yoy else "—")
                cols2[1].metric("Earnings Growth", f"{fund.earnings_growth:.1%}" if fund.earnings_growth else "—")
                cols2[2].metric("P/E", f"{fund.pe_ratio:.1f}" if fund.pe_ratio else "—")

        with tab_tech:
            if tech:
                st.metric("Score", f"{tech.score:.0f}/100")
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
                st.metric("Score", f"{corr_result.score:.0f}/100")
                cols = st.columns(3)
                cols[0].metric("Drop", f"{corr_result.correction_pct:.1%}")
                cols[1].metric("Vol Ratio", f"{corr_result.volume_ratio:.2f}x")
                cols[2].metric("RSI at Bottom", f"{corr_result.rsi_at_bottom:.1f}" if corr_result.rsi_at_bottom else "—")
                if corr_result.spy_correlation is not None:
                    st.metric("SPY Correlation", f"{corr_result.spy_correlation:.2f}",
                              help="≥0.65 = macro-driven selloff")

        with tab_news:
            if news:
                st.metric("Score", f"{news.score:.0f}/100")
                cols = st.columns(3)
                cols[0].metric("Event Type", news.event_type)
                cols[1].metric("Confidence", f"{news.confidence:.0%}")
                cols[2].metric("Downgrades", news.downgrade_count)
                if news.top_headlines:
                    st.markdown("**Recent headlines:**")
                    for h in news.top_headlines[:4]:
                        st.caption(f"• {h[:90]}")


# ── Routing ───────────────────────────────────────────────────────────────────

from datetime import datetime  # noqa: E402 (needed here for _scan_age)

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
