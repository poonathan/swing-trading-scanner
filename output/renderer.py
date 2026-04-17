from typing import List

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from scanner.models import CompositeResult

console = Console()

_SIGNAL_COLORS = {
    "BUY_DIP": "bold green",
    "WATCH": "bold yellow",
    "AVOID": "dim red",
    "INSUFFICIENT_DATA": "dim",
}


def print_results(results: List[CompositeResult], verbose: bool = False, min_score: float = 0) -> None:
    filtered = [r for r in results if r.signal in ("BUY_DIP", "WATCH") or r.composite_score >= min_score]

    if not filtered:
        console.print("[yellow]No opportunities found matching criteria.[/yellow]")
        return

    buy_dip = [r for r in filtered if r.signal == "BUY_DIP"]
    watch = [r for r in filtered if r.signal == "WATCH"]

    if buy_dip:
        console.print(f"\n[bold green]BUY DIP Candidates ({len(buy_dip)})[/bold green]")
        _print_table(buy_dip, verbose)

    if watch:
        console.print(f"\n[bold yellow]WATCH List ({len(watch)})[/bold yellow]")
        _print_table(watch, verbose)


def _print_table(results: List[CompositeResult], verbose: bool) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan", expand=False)

    table.add_column("Symbol", style="bold", min_width=6, no_wrap=True)
    table.add_column("Score", justify="right", min_width=5, no_wrap=True)
    table.add_column("Signal", min_width=8, no_wrap=True)
    table.add_column("Fund", justify="right", min_width=4, no_wrap=True)
    table.add_column("Tech", justify="right", min_width=4, no_wrap=True)
    table.add_column("Corr", justify="right", min_width=4, no_wrap=True)
    table.add_column("News", justify="right", min_width=4, no_wrap=True)
    table.add_column("Drop%", justify="right", min_width=5, no_wrap=True)
    table.add_column("VolRt", justify="right", min_width=5, no_wrap=True)
    table.add_column("Event", min_width=9, no_wrap=True)
    table.add_column("Price", justify="right", min_width=7, no_wrap=True)

    if verbose:
        table.add_column("Reason", min_width=35)

    for r in results:
        signal_style = _SIGNAL_COLORS.get(r.signal, "")
        signal_text = Text(r.signal, style=signal_style)

        drop_str = f"{r.correction.correction_pct:.1%}" if r.correction else "—"
        vol_str = f"{r.correction.volume_ratio:.2f}x" if r.correction else "—"
        event_str = r.news.event_type if r.news else "—"
        price_str = f"${r.current_price:.2f}" if r.current_price else "—"

        fund_str = f"{r.fundamental.score:.0f}" if r.fundamental else "—"
        tech_str = f"{r.technical.score:.0f}" if r.technical else "—"
        corr_str = f"{r.correction.score:.0f}" if r.correction else "—"
        news_str = f"{r.news.score:.0f}" if r.news else "—"

        # Color the event type
        event_color = {
            "macro": "yellow", "sector": "yellow",
            "unrelated": "green", "unknown": "dim",
            "mixed": "orange3", "fundamental": "red",
        }.get(event_str, "white")
        event_display = Text(event_str, style=event_color)

        # Color fund score
        fund_color = "green" if r.fundamental and r.fundamental.score >= 70 else (
            "yellow" if r.fundamental and r.fundamental.score >= 50 else "red"
        )

        row = [
            r.symbol,
            f"{r.composite_score:.1f}",
            signal_text,
            Text(fund_str, style=fund_color),
            tech_str,
            corr_str,
            news_str,
            drop_str,
            vol_str,
            event_display,
            price_str,
        ]
        if verbose:
            row.append(r.reason[:40] if r.reason else "")

        table.add_row(*row)

    console.print(table)


def print_single(result: CompositeResult) -> None:
    """Detailed output for a single ticker analysis."""
    console.print(f"\n[bold]Analysis: {result.symbol}[/bold]")
    console.print(f"Signal: [{_SIGNAL_COLORS.get(result.signal, '')}]{result.signal}[/]  Composite: {result.composite_score:.1f}/100")
    console.print(f"Reason: {result.reason}")

    if result.current_price:
        console.print(f"Price: ${result.current_price:.2f}  52w High: ${result.year_high:.2f}" if result.year_high else f"Price: ${result.current_price:.2f}")

    if result.fundamental:
        f = result.fundamental
        console.print(f"\n[cyan]Fundamental Score: {f.score:.0f}/100[/cyan]")
        if f.return_on_equity is not None:
            console.print(f"  ROE: {f.return_on_equity:.1%}")
        if f.debt_to_equity is not None:
            console.print(f"  D/E: {f.debt_to_equity:.2f}")
        if f.revenue_growth_yoy is not None:
            console.print(f"  Revenue Growth (YoY): {f.revenue_growth_yoy:.1%} ({f.revenue_trend})")
        if f.earnings_growth is not None:
            console.print(f"  Earnings Growth: {f.earnings_growth:.1%}")
        if f.gross_margins is not None:
            console.print(f"  Gross Margin: {f.gross_margins:.1%}")
        if f.pe_ratio is not None:
            console.print(f"  P/E: {f.pe_ratio:.1f}" + (f"  PEG: {f.peg_ratio:.2f}" if f.peg_ratio else ""))

    if result.technical:
        t = result.technical
        console.print(f"\n[cyan]Technical Score: {t.score:.0f}/100[/cyan]")
        console.print(f"  Uptrend confirmed: {t.was_in_uptrend}")
        console.print(f"  Above 200 SMA: {t.price_above_200sma}  Above 50 SMA: {t.price_above_50sma_before_drop}")
        if t.adx_value:
            console.print(f"  ADX: {t.adx_value:.1f}  RSI at peak: {t.rsi_at_peak:.1f}" if t.rsi_at_peak else f"  ADX: {t.adx_value:.1f}")
        console.print(f"  MACD bullish: {t.macd_bullish_before_drop}  HH/HL: {t.higher_highs_higher_lows}  OBV up: {t.obv_uptrend}")

    if result.correction:
        c = result.correction
        console.print(f"\n[cyan]Correction Score: {c.score:.0f}/100[/cyan]")
        console.print(f"  Drop: {c.correction_pct:.1%} from {c.correction_start_date} ({c.days_declining} days)")
        console.print(f"  Volume ratio: {c.volume_ratio:.2f}x  OBV trend: {c.obv_trend}")
        if c.rsi_at_bottom is not None:
            console.print(f"  RSI at bottom: {c.rsi_at_bottom:.1f}")
        console.print(f"  Near support: {c.is_near_support}" + (f" at ${c.support_level:.2f}" if c.support_level else ""))
        if c.spy_correlation is not None:
            console.print(f"  SPY correlation: {c.spy_correlation:.2f} ({'macro-driven' if c.is_macro_correlated else 'stock-specific'})")

    if result.news:
        n = result.news
        console.print(f"\n[cyan]News Score: {n.score:.0f}/100[/cyan]")
        console.print(f"  Event type: {n.event_type}  Confidence: {n.confidence:.0%}")
        console.print(f"  Fundamental signals: {n.fundamental_signal_count:.1f}  Macro signals: {n.macro_signal_count:.1f}")
        if n.downgrade_count:
            console.print(f"  Analyst downgrades: {n.downgrade_count}")
        if n.pre_earnings:
            console.print(f"  [yellow]Earnings approaching[/yellow]")
        if n.top_headlines:
            console.print("  Recent headlines:")
            for h in n.top_headlines[:3]:
                console.print(f"    • {h[:80]}")
