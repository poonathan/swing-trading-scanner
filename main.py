#!/usr/bin/env python3
"""Swing Trading Scanner — find fundamentally strong stocks on unrelated dips."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from data import cache
from output import exporter, renderer
from scanner import pipeline, universe

app = typer.Typer(help="Swing Trading Scanner: buy-the-dip on fundamentally strong stocks")
console = Console()

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        raise typer.Exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def scan(
    symbols: Optional[str] = typer.Option(None, "--symbols", "-s", help="Comma-separated symbols: AAPL,MSFT,NVDA"),
    watchlist: Optional[str] = typer.Option(None, "--watchlist", "-w", help="Path to .txt file with one symbol per line"),
    sp500: bool = typer.Option(False, "--sp500", help="Scan all S&P 500 stocks"),
    min_score: float = typer.Option(58.0, "--min-score", help="Minimum composite score to display"),
    min_correction: float = typer.Option(0.04, "--min-correction", help="Minimum correction % (e.g. 0.05 = 5%)"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export path: results.csv or results.json"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass disk cache"),
    workers: int = typer.Option(5, "--workers", help="Parallel worker threads"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-dimension scores and reason"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml"),
):
    """Scan for fundamentally strong stocks that corrected on unrelated events."""
    _setup_logging(verbose)
    config = _load_config(config_path)
    config["correction"]["min_drop_pct"] = min_correction

    # Load symbols
    syms = universe.load_symbols(symbols=symbols, watchlist_file=watchlist, sp500=sp500)
    if not syms:
        console.print("[red]No symbols to scan.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Swing Trading Scanner[/bold cyan]  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    console.print(f"Scanning [bold]{len(syms)}[/bold] symbols | min score: {min_score} | min correction: {min_correction:.0%}")
    console.print(f"Cache: {'disabled' if no_cache else 'enabled'}  Workers: {workers}\n")

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning...", total=len(syms))

        def on_progress(done, total, sym):
            progress.update(task, completed=done, description=f"Scanned {sym:<6}")

        results = pipeline.scan_universe(
            syms, config, workers=workers, use_cache=not no_cache, progress_cb=on_progress
        )

    # Filter by min score for display
    display = [r for r in results if r.signal in ("BUY_DIP", "WATCH") or r.composite_score >= min_score]

    console.print(f"\n[bold]Found {sum(1 for r in display if r.signal == 'BUY_DIP')} BUY DIP  "
                  f"{sum(1 for r in display if r.signal == 'WATCH')} WATCH candidates[/bold]")

    renderer.print_results(display, verbose=verbose, min_score=min_score)

    if export:
        path = export
        if path.endswith(".json"):
            out = exporter.export_json(display, path)
        else:
            out = exporter.export_csv(display, path)
        console.print(f"\n[green]Exported {len(display)} results to {out}[/green]")


@app.command()
def score(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze in detail"),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml"),
    no_cache: bool = typer.Option(False, "--no-cache"),
):
    """Deep analysis of a single stock."""
    _setup_logging(verbose=False)
    config = _load_config(config_path)

    console.print(f"\n[bold cyan]Analyzing {symbol.upper()}...[/bold cyan]")
    result = pipeline.scan_ticker(symbol.upper(), config, use_cache=not no_cache)

    if result is None:
        console.print(f"[yellow]{symbol}: no result — insufficient data or correction not detected.[/yellow]")
    else:
        renderer.print_single(result)


@app.command()
def clear_cache():
    """Clear the disk cache."""
    count = cache.clear()
    console.print(f"[green]Cleared {count} cache files.[/green]")


@app.command()
def cache_info():
    """Show cache statistics."""
    stats = cache.stats()
    console.print(f"Cache: {stats['files']} files, {stats['size_mb']:.1f} MB")


if __name__ == "__main__":
    app()
