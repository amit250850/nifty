# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

NiftySignalBot is a signal-only options alert system for NSE (NIFTY/BANKNIFTY) and MCX (SILVERM/GOLDM). It scans every 5 minutes, computes technical indicators, selects option strikes, evaluates safety guards, and sends Telegram alerts. Auto-execution (BUY order + OCO GTT) is implemented but currently disabled.

## Daily workflow

```bash
# Step 1: Refresh Kite auth (REQUIRED every day before running bot)
python login.py
# Opens browser → login → paste redirect URL → updates .env with new access token

# Step 2: Run the bot (blocks until Ctrl+C)
python main.py
```

## Development & testing

```bash
# Full pipeline test — works offline, no live market needed
python test_run.py

# Historical backtest (60 days of 1H data)
python backtest.py                      # all symbols
python backtest.py --symbol NIFTY       # specific symbol
python backtest.py --show-medium        # include MEDIUM conviction signals
python backtest.py --no-filter-hours    # ignore 10:00–15:00 IST filter

# Market regime analysis
python market_regime.py                  # full report
python market_regime.py --today          # today's trading recommendation

# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium   # needed for NSE Akamai bypass attempts
```

## Architecture

Signal pipeline runs inside `scan_and_signal()` in `main.py`, called every 5 minutes via APScheduler for each of NIFTY, BANKNIFTY, SILVERM, GOLDM:

1. **`modules/option_chain.py`** — Fetches PCR, Max Pain, OI trend from NSE/Kite
2. **`modules/chart_signals.py`** — Computes EMA 9/21, RSI(14), VWAP, SuperTrend(ATR10) on 1H yfinance data → yields `SignalResult` with conviction HIGH (4/4 indicators aligned) or MEDIUM (3/4)
3. **`modules/strike_selector.py`** — Rounds to ATM, selects 1-OTM strike, fetches premium, computes SL (50%) and target (2×)
4. **`modules/position_guard.py`** — Three checks: no duplicate open position, margin available ≥ lot cost, bid-ask spread ≤ 5%
5. **`modules/gtt_manager.py`** — Fourth guard (conviction ≥ threshold) + optional auto-execute (BUY + OCO GTT). **`ENABLE_AUTO_EXECUTE = False`** — set to `True` for live order placement
6. **`modules/telegram_alert.py`** — Formats and sends alert with full context
7. **`modules/trade_logger.py`** — Appends 19-column record to `trade_log.csv`
8. **`modules/trade_monitor.py`** — Per-cycle: monitors GTT status, trailing stops, P&L alerts, EOD summaries
9. **`modules/oi_tracker.py`** — SQLite snapshots of OI every 5 min; computes 20-min rolling OI trend

## Key configuration

All in `main.py` (hard-coded):
- Alert window: **10:00–15:00 IST** (skips 9:15–10:00 opening noise)
- Min conviction thresholds: NIFTY → MEDIUM, BANKNIFTY → HIGH (backtest-validated)
- DTE minimum: 3 days (Mon/Fri entry only — DTE < 3 has 17–24% win rate)

Credentials in `.env` (gitignored): `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN` (auto-updated by `login.py`), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

## Known issue: NSE option chain blocked by Akamai

NSE's option chain API requires a JS-solved `nsit` cookie. All Python HTTP libraries (requests, nsepython, curl_cffi) return `{}` regardless of headers/fingerprinting.

**Fallback chain:** NSE HTTP → Kite Quote API → estimated premium (0.8% of ATM)

Current Kite account (XJ7473) does **not** have Quote permission, so the chain falls back to estimated premium.
