@echo off
setlocal EnableDelayedExpansion

:: ─────────────────────────────────────────────────────────────────────────────
::  Swing Trading Scanner — launcher
::  Usage:  launch.bat          (starts GUI, installs deps if missing)
::          launch.bat --cli    (starts CLI scanner instead)
::          launch.bat --scan "AAPL,NVDA,MSFT"  (quick CLI scan)
:: ─────────────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo  ============================================================
echo   Swing Trading Scanner
echo  ============================================================
echo.

:: ── 1. Locate Python ─────────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found in PATH.
    echo  Please install Python 3.9+ from https://python.org and try again.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python:  %PYVER%

:: ── 2. Check / install dependencies ──────────────────────────────────────────
echo  Checking dependencies...

python -c "import streamlit, plotly, yfinance, pandas, numpy, rich, typer, yaml, feedparser, bs4" >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing missing packages from requirements.txt...
    echo.
    python -m pip install -r requirements.txt --quiet
    if %errorlevel% neq 0 (
        echo  [ERROR] pip install failed. Check your internet connection or run:
        echo    python -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  Packages installed successfully.
) else (
    echo  All dependencies satisfied.
)

echo.

:: ── 3. Route based on argument ────────────────────────────────────────────────
if /i "%~1"=="--cli" (
    echo  Starting CLI scanner...
    echo.
    shift
    python main.py %*
    goto :end
)

if /i "%~1"=="--scan" (
    echo  Running quick scan: %~2
    echo.
    python main.py scan --symbols "%~2" --verbose
    goto :end
)

if /i "%~1"=="--score" (
    echo  Analyzing %~2...
    echo.
    python main.py score "%~2"
    goto :end
)

:: ── 4. Default: launch Streamlit GUI ─────────────────────────────────────────
echo  Launching GUI at http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run gui/app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.base light

:end
echo.
if %errorlevel% neq 0 (
    echo  [ERROR] Exited with code %errorlevel%.
    pause
)
endlocal
