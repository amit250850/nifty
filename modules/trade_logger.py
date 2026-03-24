"""
modules/trade_logger.py — Trade Signal Logger

Responsibilities:
  • Append every HIGH or MEDIUM conviction signal to trade_log.csv.
  • CSV columns (specification-compliant):
      timestamp, index, direction, strike, expiry, premium, lot_cost,
      stop_loss, target, conviction, PCR, max_pain,
      EMA_signal, RSI, VWAP_signal, SuperTrend,
      entry_price, exit_price, PnL
  • entry_price, exit_price, PnL are left blank — filled manually by user.
  • Creates the CSV with headers if it doesn't exist yet.
  • Thread-safe write using a file lock.

Usage (standalone test):
    python -m modules.trade_logger
"""

import csv
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz

from modules.chart_signals import SignalResult
from modules.strike_selector import StrikeInfo

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# ── File path ──────────────────────────────────────────────────────────────────
# Resolve relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE      = _PROJECT_ROOT / "trade_log.csv"

# ── CSV column headers (spec-defined) ─────────────────────────────────────────
CSV_HEADERS = [
    "timestamp",
    "index",
    "direction",
    "strike",
    "expiry",
    "premium",
    "lot_cost",
    "stop_loss",
    "target",
    "conviction",
    "PCR",
    "max_pain",
    "EMA_signal",
    "RSI",
    "VWAP_signal",
    "SuperTrend",
    "entry_price",   # blank — user fills manually
    "exit_price",    # blank — user fills manually
    "PnL",           # blank — user fills manually
]

# Thread lock to prevent concurrent writes corrupting the CSV
_write_lock = threading.Lock()


# ── Initialisation ────────────────────────────────────────────────────────────

def initialise_log() -> None:
    """
    Create trade_log.csv with header row if it doesn't already exist.
    Safe to call multiple times (idempotent).
    """
    if not LOG_FILE.exists():
        with _write_lock:
            # Double-check inside the lock
            if not LOG_FILE.exists():
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                logger.info("[trade_logger] Created new log file: %s", LOG_FILE)
    else:
        logger.debug("[trade_logger] Log file already exists: %s", LOG_FILE)


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(
    signal:  SignalResult,
    strike:  StrikeInfo,
    oc_data: dict,
) -> dict:
    """
    Build a CSV row dict from signal, strike, and option chain data.

    Args:
        signal:  Chart signal result.
        strike:  Strike selection result.
        oc_data: Option chain data dict.

    Returns:
        Dict mapping CSV column names to values.
    """
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    # Boolean indicators as readable strings
    ema_signal = "Bullish" if signal.ema_bullish else "Bearish"
    vwap_signal = "Above" if signal.vwap_bullish else "Below"
    st_signal   = "Green"  if signal.supertrend_bullish else "Red"

    return {
        "timestamp":   now,
        "index":       signal.symbol,
        "direction":   signal.direction,
        "strike":      f"{strike.otm_strike}{strike.option_type}",
        "expiry":      strike.expiry_date,
        "premium":     strike.premium,
        "lot_cost":    strike.lot_cost,
        "stop_loss":   strike.stop_loss,
        "target":      strike.target,
        "conviction":  signal.conviction,
        "PCR":         oc_data.get("pcr", ""),
        "max_pain":    oc_data.get("max_pain", ""),
        "EMA_signal":  ema_signal,
        "RSI":         signal.rsi_value,
        "VWAP_signal": vwap_signal,
        "SuperTrend":  st_signal,
        "entry_price": "",   # user fills
        "exit_price":  "",   # user fills
        "PnL":         "",   # user fills
    }


# ── Main log function ──────────────────────────────────────────────────────────

def log_signal(
    signal:  SignalResult,
    strike:  StrikeInfo,
    oc_data: dict,
) -> bool:
    """
    Append a trade signal to the CSV log file.

    Args:
        signal:  Chart signal result from chart_signals module.
        strike:  Strike info from strike_selector module.
        oc_data: Option chain dict from option_chain module.

    Returns:
        True on success, False on failure.
    """
    # Ensure the file and headers exist
    initialise_log()

    row = _build_row(signal, strike, oc_data)

    with _write_lock:
        try:
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)
            _pcr = oc_data.get("pcr")
            _pcr_str = f"{_pcr:.2f}" if _pcr is not None else "N/A"
            logger.info(
                "[trade_logger] Logged signal: %s %s %s%s | %s | PCR=%s",
                row["timestamp"],
                signal.symbol,
                strike.otm_strike,
                strike.option_type,
                signal.conviction,
                _pcr_str,
            )
            return True
        except OSError as exc:
            logger.error("[trade_logger] Failed to write to %s: %s", LOG_FILE, exc)
            return False


def get_log_path() -> str:
    """Return the absolute path of the trade log CSV."""
    return str(LOG_FILE)


def count_signals_today() -> int:
    """
    Count how many signals were logged today (IST).

    Returns:
        Integer count of today's signals.
    """
    if not LOG_FILE.exists():
        return 0

    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    count     = 0

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("timestamp", "").startswith(today_str):
                    count += 1
    except OSError as exc:
        logger.error("[trade_logger] Failed to read log: %s", exc)

    return count


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from modules.chart_signals import SignalResult
    from modules.strike_selector import StrikeInfo

    # Initialise CSV
    initialise_log()

    mock_signal = SignalResult(
        symbol             = "NIFTY",
        direction          = "BUY CALL",
        conviction         = "HIGH",
        signals_agreed     = 4,
        ema_bullish        = True,
        rsi_value          = 58.0,
        rsi_bullish        = True,
        vwap_bullish       = True,
        supertrend_bullish = True,
        last_close         = 23050.0,
        last_vwap          = 22980.0,
        last_ema9          = 23020.0,
        last_ema21         = 22950.0,
    )

    mock_strike = StrikeInfo(
        symbol      = "NIFTY",
        direction   = "BUY CALL",
        spot        = 23050.0,
        atm_strike  = 23050,
        otm_strike  = 23100,
        option_type = "CE",
        expiry_date = "24 Apr 2026",
        expiry_raw  = date(2026, 4, 24),
        nfo_symbol  = "NIFTY26APR23100CE",
        premium     = 185.0,
        lot_size    = 75,
        lot_cost    = 13875.0,
        stop_loss   = 92.5,
        target      = 370.0,
    )

    mock_oc = {
        "pcr":                1.12,
        "pcr_trend":          "Rising — Bullish",
        "max_put_oi_strike":  22800,
        "max_call_oi_strike": 23200,
        "max_pain":           23000,
        "iv_rank":            42.5,
    }

    success = log_signal(mock_signal, mock_strike, mock_oc)
    print(f"\n{'✅ Signal logged' if success else '❌ Failed to log'} → {get_log_path()}")
    print(f"Total signals today: {count_signals_today()}")

    # Print last 5 rows of CSV
    print(f"\nLast rows in {LOG_FILE.name}:")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-3:]:
        print(line.rstrip())
