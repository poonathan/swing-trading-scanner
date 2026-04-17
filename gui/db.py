import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "swing_trading.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scan_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at          TEXT NOT NULL,
                universe        TEXT,
                symbol_count    INTEGER,
                buy_dip_count   INTEGER,
                watch_count     INTEGER,
                config_json     TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER REFERENCES scan_runs(id) ON DELETE CASCADE,
                scan_date           TEXT NOT NULL,
                run_at              TEXT NOT NULL,
                symbol              TEXT NOT NULL,
                signal              TEXT,
                composite_score     REAL,
                fund_score          REAL,
                tech_score          REAL,
                corr_score          REAL,
                news_score          REAL,
                current_price       REAL,
                year_high           REAL,
                pct_from_high       REAL,
                correction_pct      REAL,
                volume_ratio        REAL,
                event_type          TEXT,
                spy_correlation     REAL,
                is_macro_correlated INTEGER,
                roe                 REAL,
                debt_to_equity      REAL,
                revenue_growth      REAL,
                earnings_growth     REAL,
                was_in_uptrend      INTEGER,
                adx                 REAL,
                rsi_at_bottom       REAL,
                near_support        INTEGER,
                sector              TEXT,
                industry            TEXT,
                reason              TEXT
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                symbol      TEXT PRIMARY KEY,
                added_at    TEXT NOT NULL,
                notes       TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_results_symbol_date ON scan_results(symbol, scan_date);
            CREATE INDEX IF NOT EXISTS idx_results_run_id      ON scan_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_results_signal      ON scan_results(signal);
            CREATE INDEX IF NOT EXISTS idx_results_date        ON scan_results(scan_date);
        """)


def save_scan(results: list, universe: str, config: dict) -> int:
    from scanner.models import CompositeResult
    run_at = datetime.now().isoformat()
    scan_date = run_at[:10]
    buy_dip = sum(1 for r in results if r.signal == "BUY_DIP")
    watch = sum(1 for r in results if r.signal == "WATCH")

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO scan_runs (run_at, universe, symbol_count, buy_dip_count, watch_count, config_json)"
            " VALUES (?,?,?,?,?,?)",
            (run_at, universe, len(results), buy_dip, watch, json.dumps(config, default=str)),
        )
        run_id = cur.lastrowid

        rows = [_flatten_result(r, run_id, scan_date, run_at) for r in results]
        conn.executemany("""
            INSERT INTO scan_results (
                run_id, scan_date, run_at, symbol, signal, composite_score,
                fund_score, tech_score, corr_score, news_score,
                current_price, year_high, pct_from_high,
                correction_pct, volume_ratio, event_type,
                spy_correlation, is_macro_correlated,
                roe, debt_to_equity, revenue_growth, earnings_growth,
                was_in_uptrend, adx, rsi_at_bottom, near_support,
                sector, industry, reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)

    return run_id


def _flatten_result(r, run_id, scan_date, run_at) -> tuple:
    f = r.fundamental
    t = r.technical
    c = r.correction
    n = r.news
    return (
        run_id, scan_date, run_at, r.symbol, r.signal, r.composite_score,
        f.score if f else None,
        t.score if t else None,
        c.score if c else None,
        n.score if n else None,
        r.current_price, r.year_high,
        round(r.pct_from_high * 100, 1) if r.pct_from_high else None,
        round(c.correction_pct * 100, 1) if c else None,
        c.volume_ratio if c else None,
        n.event_type if n else None,
        c.spy_correlation if c else None,
        int(c.is_macro_correlated) if c else None,
        round(f.return_on_equity * 100, 1) if f and f.return_on_equity else None,
        f.debt_to_equity if f else None,
        round(f.revenue_growth_yoy * 100, 1) if f and f.revenue_growth_yoy else None,
        round(f.earnings_growth * 100, 1) if f and f.earnings_growth else None,
        int(t.was_in_uptrend) if t else None,
        t.adx_value if t else None,
        c.rsi_at_bottom if c else None,
        int(c.is_near_support) if c else None,
        r.sector, r.industry, r.reason,
    )


# ── Queries ──────────────────────────────────────────────────────────────────

def get_latest_run() -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM scan_runs ORDER BY run_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def get_run_stats() -> dict:
    with get_conn() as conn:
        total_runs = conn.execute("SELECT COUNT(*) FROM scan_runs").fetchone()[0]
        unique_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM scan_results").fetchone()[0]
        latest = conn.execute("SELECT * FROM scan_runs ORDER BY run_at DESC LIMIT 1").fetchone()
        return {
            "total_runs": total_runs,
            "unique_symbols": unique_symbols,
            "latest_run": dict(latest) if latest else None,
        }


def get_latest_results(signal_filter: str = None, min_score: float = 0) -> pd.DataFrame:
    latest = get_latest_run()
    if not latest:
        return pd.DataFrame()
    with get_conn() as conn:
        params: list = [latest["id"]]
        q = "SELECT * FROM scan_results WHERE run_id = ?"
        if signal_filter:
            q += " AND signal = ?"; params.append(signal_filter)
        if min_score:
            q += " AND composite_score >= ?"; params.append(min_score)
        q += " ORDER BY composite_score DESC"
        rows = conn.execute(q, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_upgrades() -> pd.DataFrame:
    """Symbols that moved up to BUY_DIP in the latest scan vs the previous one."""
    with get_conn() as conn:
        runs = conn.execute("SELECT id FROM scan_runs ORDER BY run_at DESC LIMIT 2").fetchall()
        if len(runs) < 2:
            return pd.DataFrame()
        latest_id, prev_id = runs[0]["id"], runs[1]["id"]
        rows = conn.execute("""
            SELECT curr.*, prev.signal AS prev_signal, prev.composite_score AS prev_score
            FROM scan_results curr
            JOIN scan_results prev ON curr.symbol = prev.symbol
            WHERE curr.run_id = ? AND prev.run_id = ?
              AND curr.signal = 'BUY_DIP'
              AND prev.signal IN ('WATCH', 'AVOID')
            ORDER BY curr.composite_score DESC
        """, (latest_id, prev_id)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_symbol_history(symbol: str) -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_results WHERE symbol = ? ORDER BY run_at ASC", (symbol,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_all_tracked_symbols() -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM scan_results ORDER BY symbol"
        ).fetchall()
    return [r["symbol"] for r in rows]


def get_all_sectors() -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT sector FROM scan_results WHERE sector IS NOT NULL ORDER BY sector"
        ).fetchall()
    return [r["sector"] for r in rows]


def get_history_df(
    date_from=None, date_to=None, signal=None, sector=None, min_score=0, symbol_search=None
) -> pd.DataFrame:
    with get_conn() as conn:
        q = "SELECT * FROM scan_results WHERE 1=1"
        params: list = []
        if date_from:
            q += " AND scan_date >= ?"; params.append(str(date_from))
        if date_to:
            q += " AND scan_date <= ?"; params.append(str(date_to))
        if signal and signal != "All":
            q += " AND signal = ?"; params.append(signal)
        if sector and sector != "All":
            q += " AND sector = ?"; params.append(sector)
        if min_score:
            q += " AND composite_score >= ?"; params.append(min_score)
        if symbol_search:
            q += " AND symbol LIKE ?"; params.append(f"%{symbol_search.upper()}%")
        q += " ORDER BY run_at DESC"
        rows = conn.execute(q, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ── Watchlist ─────────────────────────────────────────────────────────────────

def get_watchlist() -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def get_watchlist_symbols() -> List[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()
    return [r["symbol"] for r in rows]


def add_to_watchlist(symbol: str, notes: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (symbol, added_at, notes) VALUES (?,?,?)",
            (symbol.upper(), datetime.now().isoformat(), notes),
        )


def remove_from_watchlist(symbol: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))


def get_score_deltas() -> pd.DataFrame:
    """For each symbol, compare latest vs previous scan: score delta and signal change."""
    with get_conn() as conn:
        runs = conn.execute("SELECT id FROM scan_runs ORDER BY run_at DESC LIMIT 2").fetchall()
        if len(runs) < 2:
            return pd.DataFrame()
        latest_id, prev_id = runs[0]["id"], runs[1]["id"]
        rows = conn.execute("""
            SELECT
                curr.symbol,
                curr.signal           AS signal,
                curr.composite_score  AS score,
                prev.signal           AS prev_signal,
                prev.composite_score  AS prev_score,
                (curr.composite_score - prev.composite_score) AS delta,
                curr.correction_pct, curr.event_type, curr.sector,
                curr.current_price
            FROM scan_results curr
            JOIN scan_results prev ON curr.symbol = prev.symbol
            WHERE curr.run_id = ? AND prev.run_id = ?
            ORDER BY delta DESC
        """, (latest_id, prev_id)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
