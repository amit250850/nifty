#!/usr/bin/env python3
"""
straddle_executor.py — Pre-Event Straddle Alert + Auto-Executor

Monitors the NSE earnings calendar and fires HIGH-CONVICTION Telegram alerts
the day before quarterly results. Optionally places the straddle via Kite.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY SUMMARY (from 2Y backtest across 50 NIFTY50 stocks):
  • 404 events analysed (gap ≥ 3% days = earnings/result days)
  • Overall win rate at 2H exit: 100% for top-tier stocks
  • Average move (4-5%) always exceeds break-even (1.5-2.5%)
  • 2H exit consistently 2× better than EOD (IV crush kills EOD)
  • Top stocks: TRENT, INDIGO, AXISBANK, ADANIENT, LT, BEL, INFY

TRADE PLAN:
  Day -1 (prev close): Buy ATM CE + ATM PE (straddle)
  Day  0 open        : Sell losing leg at market
  Day  0 + 2H        : Exit winning leg — HARD EXIT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SOURCES FOR EARNINGS CALENDAR (in priority order):
  1. MANUAL_DATES — hardcoded known result dates (most reliable)
  2. yfinance get_earnings_dates() — available for some NSE stocks
  3. QUARTERLY_WINDOW — fallback: flag stocks in result months
     (Q4: Apr-May | Q1: Jul-Aug | Q2: Oct-Nov | Q3: Jan-Feb)

USAGE:
  python straddle_executor.py --scan              # check next 7 days for results
  python straddle_executor.py --scan --days 30    # check next 30 days
  python straddle_executor.py --today             # alert for any results TOMORROW
  python straddle_executor.py --add INDIGO 2025-07-23   # add manual date
  python straddle_executor.py --alert INDIGO      # force send alert for INDIGO now
  python straddle_executor.py --execute INDIGO    # place straddle via Kite NOW
  python straddle_executor.py --monitor           # run as daily scheduler (3 PM IST)
  python straddle_executor.py --status            # show all known upcoming dates
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from kiteconnect import KiteConnect

warnings.filterwarnings("ignore")
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("straddle_executor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("straddle_executor")

IST = pytz.timezone("Asia/Kolkata")

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
API_KEY        = os.getenv("KITE_API_KEY")
ACCESS_TOKEN   = os.getenv("KITE_ACCESS_TOKEN")

DATES_FILE = Path("straddle_dates.json")   # persistent store for manual dates
RISK_FREE   = 0.065
PRE_IV_MULT = 1.40
BASE_VOL_W  = 30

# ── Backtest-derived stats (from backtest_event_straddle.py, 2Y, gap≥3%) ─────
# Used in alert messages to show historical edge.
STOCK_STATS = {
    "TRENT":      {"events": 17, "be_rate": 94,  "win_2h": 100, "avg_pnl_2h": 124973, "tier": 1},
    "INDIGO":     {"events": 22, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  59366, "tier": 1},
    "AXISBANK":   {"events": 10, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  66090, "tier": 1},
    "ADANIENT":   {"events": 20, "be_rate": 85,  "win_2h": 100, "avg_pnl_2h":  41967, "tier": 1},
    "LT":         {"events": 10, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  51338, "tier": 1},
    "BEL":        {"events": 14, "be_rate": 93,  "win_2h": 100, "avg_pnl_2h":  69600, "tier": 1},
    "SBIN":       {"events":  5, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  79398, "tier": 1},
    "M&M":        {"events":  7, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  95668, "tier": 1},
    "INDUSINDBK": {"events": 10, "be_rate": 70,  "win_2h": 100, "avg_pnl_2h":  43262, "tier": 1},
    "HINDALCO":   {"events": 13, "be_rate": 92,  "win_2h":  92, "avg_pnl_2h":  38582, "tier": 2},
    "CIPLA":      {"events":  8, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  46626, "tier": 1},
    "EICHERMOT":  {"events":  9, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  43241, "tier": 1},
    "INFY":       {"events": 14, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  20588, "tier": 2},
    "SHRIRAMFIN": {"events": 16, "be_rate": 81,  "win_2h": 100, "avg_pnl_2h":  20705, "tier": 2},
    "MARUTI":     {"events":  6, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  52694, "tier": 1},
    "TATACONSUM": {"events":  5, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  54542, "tier": 1},
    "HCLTECH":    {"events": 11, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  36266, "tier": 2},
    "BAJAJ-AUTO": {"events":  7, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  29009, "tier": 2},
    "JSWSTEEL":   {"events":  5, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  32108, "tier": 2},
    "SBILIFE":    {"events":  7, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  47446, "tier": 2},
    "ASIANPAINT": {"events":  9, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  22808, "tier": 2},
    "TECHM":      {"events": 10, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  37677, "tier": 2},
    "BPCL":       {"events": 12, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  22150, "tier": 2},
    "NTPC":       {"events":  5, "be_rate": 80,  "win_2h": 100, "avg_pnl_2h":  68551, "tier": 2},
    "TCS":        {"events":  6, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  23154, "tier": 2},
    "APOLLOHOSP": {"events":  4, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  38041, "tier": 2},
    "SUNPHARMA":  {"events":  4, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  46129, "tier": 2},
    "HEROMOTOCO": {"events":  6, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  52042, "tier": 2},
    "TITAN":      {"events":  4, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  49372, "tier": 2},
    "COALINDIA":  {"events":  7, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  46055, "tier": 2},
    "TATASTEEL":  {"events": 10, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  30915, "tier": 2},
    "HDFCBANK":   {"events":  9, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  16959, "tier": 2},
    "RELIANCE":   {"events":  6, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  15296, "tier": 2},
    "WIPRO":      {"events": 12, "be_rate": 67,  "win_2h":  83, "avg_pnl_2h":   9850, "tier": 3},
    "KOTAKBANK":  {"events":  6, "be_rate": 67,  "win_2h":  83, "avg_pnl_2h":   5854, "tier": 3},
    "BAJFINANCE": {"events": 15, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":   3362, "tier": 3},
    "ICICIBANK":  {"events":  6, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  23685, "tier": 2},
    "HINDUNILVR": {"events":  5, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  25064, "tier": 2},
    "DRREDDY":    {"events":  9, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":   6497, "tier": 3},
    "GRASIM":     {"events":  4, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  21978, "tier": 2},
    "DIVISLAB":   {"events":  8, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  21746, "tier": 2},
    "HDFCLIFE":   {"events":  5, "be_rate": 100, "win_2h":  80, "avg_pnl_2h":  22416, "tier": 2},
    "BAJAJFINSV": {"events":  3, "be_rate": 100, "win_2h": 100, "avg_pnl_2h":  24266, "tier": 2},
    "BHARTIARTL": {"events":  3, "be_rate": 100, "win_2h": 100, "avg_pnl_2h": 110471, "tier": 1},
    # Avoid list
    "ONGC":       {"events":  7, "be_rate": 43,  "win_2h":  29, "avg_pnl_2h":  -5306, "tier": 0},
    "POWERGRID":  {"events":  6, "be_rate": 67,  "win_2h":  83, "avg_pnl_2h":  31224, "tier": 3},
}

NIFTY50_TICKERS = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN": "SBIN.NS", "BHARTIARTL": "BHARTIARTL.NS", "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS", "AXISBANK": "AXISBANK.NS", "LT": "LT.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "MARUTI": "MARUTI.NS", "TITAN": "TITAN.NS",
    "WIPRO": "WIPRO.NS", "ULTRACEMCO": "ULTRACEMCO.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "HCLTECH": "HCLTECH.NS", "NESTLEIND": "NESTLEIND.NS", "POWERGRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS", "ONGC": "ONGC.NS", "JSWSTEEL": "JSWSTEEL.NS",
    "INDUSINDBK": "INDUSINDBK.NS", "TECHM": "TECHM.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "ADANIENT": "ADANIENT.NS", "BAJAJFINSV": "BAJAJFINSV.NS", "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS", "EICHERMOT": "EICHERMOT.NS", "GRASIM": "GRASIM.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS", "HINDALCO": "HINDALCO.NS", "M&M": "M&M.NS",
    "TATASTEEL": "TATASTEEL.NS", "TATACONSUM": "TATACONSUM.NS", "APOLLOHOSP": "APOLLOHOSP.NS",
    "BPCL": "BPCL.NS", "COALINDIA": "COALINDIA.NS", "DIVISLAB": "DIVISLAB.NS",
    "INDIGO": "INDIGO.NS", "SBILIFE": "SBILIFE.NS", "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "TRENT": "TRENT.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS", "HDFCLIFE": "HDFCLIFE.NS",
    "BEL": "BEL.NS",
}

LOT_SIZE_MAP = {
    "RELIANCE": 250, "TCS": 150, "HDFCBANK": 550, "INFY": 300,
    "ICICIBANK": 700, "HINDUNILVR": 300, "SBIN": 1500, "BHARTIARTL": 1851,
    "ITC": 3200, "KOTAKBANK": 400, "AXISBANK": 1200, "LT": 300,
    "ASIANPAINT": 200, "MARUTI": 100, "TITAN": 375, "WIPRO": 1500,
    "ULTRACEMCO": 100, "BAJFINANCE": 125, "HCLTECH": 700, "NESTLEIND": 40,
    "POWERGRID": 4900, "NTPC": 4500, "ONGC": 1925, "JSWSTEEL": 675,
    "INDUSINDBK": 600, "TECHM": 600, "SUNPHARMA": 700, "ADANIENT": 250,
    "BAJAJFINSV": 500, "CIPLA": 650, "DRREDDY": 125, "EICHERMOT": 200,
    "GRASIM": 250, "HEROMOTOCO": 300, "HINDALCO": 1400, "M&M": 700,
    "TATASTEEL": 5500, "TATACONSUM": 1100, "APOLLOHOSP": 125, "BPCL": 1800,
    "COALINDIA": 2800, "DIVISLAB": 100, "INDIGO": 300, "SBILIFE": 750,
    "SHRIRAMFIN": 600, "TRENT": 500, "BAJAJ-AUTO": 75, "HDFCLIFE": 1100,
    "BEL": 4900,
}

STRIKE_ROUND = {
    "MARUTI": 500, "EICHERMOT": 100, "BAJAJ-AUTO": 100, "HINDUNILVR": 100,
    "APOLLOHOSP": 50, "DRREDDY": 100, "DIVISLAB": 100, "TRENT": 50,
}

# Quarterly result months — fallback when exact date not known
RESULT_MONTHS = {
    "Q4": [4, 5],    # April–May
    "Q1": [7, 8],    # July–August
    "Q2": [10, 11],  # October–November
    "Q3": [1, 2],    # January–February
}
ALL_RESULT_MONTHS = {m for months in RESULT_MONTHS.values() for m in months}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bs_call(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def atm_strike(spot: float, symbol: str) -> float:
    rnd = STRIKE_ROUND.get(symbol, 50)
    return round(spot / rnd) * rnd


def realized_vol(prices: pd.Series) -> float:
    if len(prices) < BASE_VOL_W + 2:
        return 0.30
    lr = np.log(prices / prices.shift(1)).dropna()
    return float(max(0.15, min(1.2, lr.tail(BASE_VOL_W).std() * math.sqrt(252))))


def get_spot(symbol: str) -> Optional[float]:
    ticker = NIFTY50_TICKERS.get(symbol)
    if not ticker:
        return None
    try:
        df = yf.download(ticker, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return float(df["Close"].squeeze().iloc[-1])
    except Exception:
        return None


def get_daily_prices(symbol: str, days: int = 60) -> Optional[pd.Series]:
    ticker = NIFTY50_TICKERS.get(symbol)
    if not ticker:
        return None
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return df["Close"].squeeze()
    except Exception:
        return None


def estimate_straddle(symbol: str, spot: float, prices: pd.Series) -> dict:
    """Estimate straddle cost and break-even using Black-Scholes."""
    vol      = realized_vol(prices)
    pre_vol  = vol * PRE_IV_MULT
    K        = atm_strike(spot, symbol)
    T        = 1.0 / 365.0   # 1 DTE (buy day before, results next day)
    call_c   = bs_call(spot, K, T, RISK_FREE, pre_vol)
    put_c    = bs_put (spot, K, T, RISK_FREE, pre_vol)
    total    = call_c + put_c
    be_pct   = (total / K) * 100.0
    lot      = LOT_SIZE_MAP.get(symbol, 500)
    outlay   = total * lot
    return {
        "spot":    round(spot, 2),
        "strike":  K,
        "call":    round(call_c, 2),
        "put":     round(put_c, 2),
        "total":   round(total, 2),
        "be_pct":  round(be_pct, 2),
        "lot":     lot,
        "outlay":  round(outlay, 0),
        "vol":     round(vol * 100, 1),
        "pre_vol": round(pre_vol * 100, 1),
    }


# ── Dates Store ───────────────────────────────────────────────────────────────

def load_dates() -> dict:
    """Load manually-entered result dates from JSON file."""
    if not DATES_FILE.exists():
        return {}
    try:
        with open(DATES_FILE) as f:
            raw = json.load(f)
        # Convert string dates back to date objects
        return {sym: [date.fromisoformat(d) for d in dates]
                for sym, dates in raw.items()}
    except Exception:
        return {}


def save_dates(dates_dict: dict) -> None:
    with open(DATES_FILE, "w") as f:
        json.dump({sym: [d.isoformat() for d in dates]
                   for sym, dates in dates_dict.items()}, f, indent=2)


def add_date(symbol: str, result_date: date) -> None:
    """Persist a known result date for a symbol."""
    dates = load_dates()
    if symbol not in dates:
        dates[symbol] = []
    if result_date not in dates[symbol]:
        dates[symbol].append(result_date)
        dates[symbol].sort()
    save_dates(dates)
    print(f"✅  Added {symbol} result date: {result_date}")


def get_upcoming_manual(within_days: int) -> list[dict]:
    """Return manually-stored result dates within the next N days."""
    today   = date.today()
    cutoff  = today + timedelta(days=within_days)
    dates   = load_dates()
    results = []
    for sym, date_list in dates.items():
        for d in date_list:
            if today <= d <= cutoff:
                results.append({"symbol": sym, "date": d, "source": "manual"})
    return results


# ── Earnings Calendar (yfinance) ──────────────────────────────────────────────

def get_yfinance_earnings(symbol: str, within_days: int = 30) -> list[date]:
    """Pull earnings dates from yfinance (works for some NSE stocks)."""
    ticker = NIFTY50_TICKERS.get(symbol)
    if not ticker:
        return []
    try:
        t  = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=8)
        if ed is None or ed.empty:
            return []
        today  = date.today()
        cutoff = today + timedelta(days=within_days)
        upcoming = []
        for idx in ed.index:
            try:
                d = pd.Timestamp(idx).date()
                if today <= d <= cutoff:
                    upcoming.append(d)
            except Exception:
                pass
        return sorted(upcoming)
    except Exception:
        return []


def scan_yfinance_calendar(symbols: list[str], within_days: int) -> list[dict]:
    """Check all symbols against yfinance earnings calendar."""
    results = []
    for sym in symbols:
        dates = get_yfinance_earnings(sym, within_days)
        for d in dates:
            results.append({"symbol": sym, "date": d, "source": "yfinance"})
    return results


def is_result_month() -> bool:
    """Return True if current month is a quarterly result month."""
    return datetime.now(IST).month in ALL_RESULT_MONTHS


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        logger.warning("Telegram not configured — printing to console only.")
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60 + "\n")
        return False
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.ok
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return False


def build_alert_message(symbol: str, result_date: date, straddle: dict,
                         days_to_event: int) -> str:
    """Build the HIGH CONVICTION Telegram alert message."""
    stats   = STOCK_STATS.get(symbol, {})
    tier    = stats.get("tier", 3)
    win_2h  = stats.get("win_2h", 70)
    be_rate = stats.get("be_rate", 80)
    avg_pnl = stats.get("avg_pnl_2h", 0)
    events  = stats.get("events", 0)

    tier_label = {1: "🥇 TIER 1 — TOP CONVICTION", 2: "🥈 TIER 2 — HIGH CONVICTION",
                  3: "🥉 TIER 3 — MODERATE", 0: "⛔ AVOID"}
    urgency = "⚡ TOMORROW" if days_to_event == 1 else f"📅 in {days_to_event} days"

    lines = [
        f"🔔 <b>PRE-EVENT STRADDLE OPPORTUNITY</b>",
        f"{'━' * 36}",
        f"<b>{symbol}</b>  |  {tier_label.get(tier, '')}",
        f"Results: {result_date.strftime('%d %b %Y')}  {urgency}",
        f"{'━' * 36}",
        f"",
        f"📊 <b>Straddle Setup</b>",
        f"  Spot:        ₹{straddle['spot']:,.2f}",
        f"  ATM Strike:  ₹{straddle['strike']:,.0f}",
        f"  Call cost:   ₹{straddle['call']:.2f}",
        f"  Put cost:    ₹{straddle['put']:.2f}",
        f"  Total/unit:  ₹{straddle['total']:.2f}",
        f"  Lot size:    {straddle['lot']}",
        f"  <b>Total outlay: ₹{straddle['outlay']:,.0f}</b>",
        f"  Break-even:  ±{straddle['be_pct']:.1f}% move needed",
        f"  IV used:     {straddle['pre_vol']:.0f}% (base {straddle['vol']:.0f}% × 1.4)",
        f"",
        f"📈 <b>Historical Edge (2Y backtest)</b>",
        f"  Events:       {events}",
        f"  Break-even hit rate: {be_rate}%",
        f"  <b>Win rate at 2H exit: {win_2h}%</b>",
        f"  Avg P&amp;L at 2H exit: ₹{avg_pnl:,.0f}",
        f"",
        f"🎯 <b>Trade Plan</b>",
    ]

    if days_to_event == 1:
        lines += [
            f"  <b>TODAY before 3:20 PM:</b>",
            f"    Buy {straddle['strike']} CE (1 lot = {straddle['lot']} units)",
            f"    Buy {straddle['strike']} PE (1 lot = {straddle['lot']} units)",
            f"  <b>TOMORROW open:</b> Sell losing leg at market",
            f"  <b>TOMORROW +2H:</b>  Exit winning leg — HARD EXIT",
        ]
    else:
        lines += [
            f"  Day before results ({(result_date - timedelta(days=1)).strftime('%d %b')}):",
            f"    Buy {straddle['strike']} CE + {straddle['strike']} PE at close",
            f"  Results day open: Sell losing leg immediately",
            f"  Results day +2H: Exit winning leg — HARD EXIT",
        ]

    if tier == 1:
        lines += [
            f"",
            f"{'━' * 36}",
            f"⚠️  <b>DO NOT MISS THIS TRADE</b>",
            f"Win rate {win_2h}% | Avg return ₹{avg_pnl:,.0f} per event",
            f"{'━' * 36}",
        ]
    elif tier == 0:
        lines += [
            f"",
            f"⛔ <b>This stock has poor historical edge — SKIP</b>",
        ]

    return "\n".join(lines)


# ── Kite Auto-Execute ──────────────────────────────────────────────────────────

def init_kite() -> Optional[KiteConnect]:
    load_dotenv(override=True)
    token = os.getenv("KITE_ACCESS_TOKEN")
    key   = os.getenv("KITE_API_KEY")
    if not token or not key:
        logger.error("Kite credentials missing in .env")
        return None
    kite = KiteConnect(api_key=key)
    kite.set_access_token(token)
    try:
        kite.profile()
        logger.info("Kite session valid.")
        return kite
    except Exception as exc:
        logger.error("Kite session invalid: %s. Run login.py first.", exc)
        return None


def find_option_symbol(kite: KiteConnect, symbol: str, expiry: date,
                        strike: float, opt_type: str) -> Optional[str]:
    """Search Kite NFO instruments for the tradingsymbol."""
    try:
        instruments = kite.instruments("NFO")
        opt_type_kite = "CE" if opt_type.upper() == "CE" else "PE"
        for inst in instruments:
            if (inst["name"] == symbol and
                inst["instrument_type"] == opt_type_kite and
                abs(inst["strike"] - strike) < 1 and
                inst["expiry"] == expiry):
                return inst["tradingsymbol"]
    except Exception as exc:
        logger.error("Instruments lookup failed: %s", exc)
    return None


def find_nearest_expiry(kite: KiteConnect, symbol: str,
                         min_dte: int = 0) -> Optional[date]:
    """Find the nearest expiry for a symbol with at least min_dte days remaining."""
    try:
        instruments = kite.instruments("NFO")
        today    = date.today()
        expiries = sorted({inst["expiry"] for inst in instruments
                           if inst["name"] == symbol and
                           (inst["expiry"] - today).days >= min_dte})
        return expiries[0] if expiries else None
    except Exception:
        return None


def place_straddle(kite: KiteConnect, symbol: str, strike: float,
                    expiry: date, lot: int, dry_run: bool = True) -> dict:
    """
    Place ATM call + put buy orders (straddle).
    dry_run=True: logs what would be placed without actually placing.
    """
    results = {"call": None, "put": None, "errors": []}

    for opt_type in ("CE", "PE"):
        ts = find_option_symbol(kite, symbol, expiry, strike, opt_type)
        if not ts:
            msg = f"Could not find {symbol} {strike} {opt_type} exp={expiry}"
            logger.error(msg)
            results["errors"].append(msg)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would place: BUY %s x%d @ MARKET (NFO)", ts, lot)
            results[opt_type.lower()] = {"tradingsymbol": ts, "qty": lot,
                                          "status": "DRY_RUN"}
        else:
            try:
                order_id = kite.place_order(
                    variety          = kite.VARIETY_REGULAR,
                    exchange         = kite.EXCHANGE_NFO,
                    tradingsymbol    = ts,
                    transaction_type = kite.TRANSACTION_TYPE_BUY,
                    quantity         = lot,
                    product          = kite.PRODUCT_NRML,
                    order_type       = kite.ORDER_TYPE_MARKET,
                )
                logger.info("✅  Order placed: %s x%d → order_id=%s", ts, lot, order_id)
                results[opt_type.lower()] = {
                    "tradingsymbol": ts, "qty": lot,
                    "order_id": order_id, "status": "PLACED",
                }
            except Exception as exc:
                msg = f"Order failed for {ts}: {exc}"
                logger.error(msg)
                results["errors"].append(msg)

    return results


# ── Core Actions ──────────────────────────────────────────────────────────────

def action_scan(within_days: int) -> None:
    """Scan all sources for upcoming result dates."""
    print(f"\n🔍  Scanning for results in next {within_days} days...\n")
    all_events = []

    # 1. Manual dates
    manual = get_upcoming_manual(within_days)
    all_events.extend(manual)

    # 2. yfinance calendar (top stocks only — API is slow)
    tier1 = [s for s, st in STOCK_STATS.items() if st.get("tier", 0) == 1]
    print(f"   Checking yfinance for {len(tier1)} tier-1 stocks...")
    yf_events = scan_yfinance_calendar(tier1, within_days)
    # Deduplicate against manual
    manual_keys = {(e["symbol"], e["date"]) for e in manual}
    yf_events   = [e for e in yf_events
                   if (e["symbol"], e["date"]) not in manual_keys]
    all_events.extend(yf_events)

    if not all_events:
        print("  No upcoming results found in calendar.\n")
        if is_result_month():
            print("  ℹ️  Current month is a results month — add dates manually:")
            print("     python straddle_executor.py --add SYMBOL YYYY-MM-DD\n")
        return

    all_events.sort(key=lambda e: e["date"])
    today = date.today()
    print(f"  {'DATE':<14} {'SYMBOL':<14} {'DAYS':<6} {'TIER':<8} {'SOURCE'}")
    print("  " + "─" * 60)
    for ev in all_events:
        sym  = ev["symbol"]
        d    = ev["date"]
        days = (d - today).days
        st   = STOCK_STATS.get(sym, {})
        tier = {1: "🥇 T1", 2: "🥈 T2", 3: "🥉 T3", 0: "⛔ SKIP"}.get(
            st.get("tier", 3), "❓")
        src  = ev["source"]
        print(f"  {d.strftime('%d %b %Y'):<14} {sym:<14} {days:<6} {tier:<10} {src}")
    print()


def action_today() -> None:
    """
    Send alerts for any results happening TOMORROW.
    Designed to run daily at 3 PM IST.
    """
    today    = date.today()
    tomorrow = today + timedelta(days=1)
    # Skip weekends
    if tomorrow.weekday() >= 5:
        logger.info("Tomorrow is a weekend — no alert needed.")
        return

    upcoming = get_upcoming_manual(2)
    yf_tier1 = [s for s, st in STOCK_STATS.items() if st.get("tier", 0) <= 2]
    yf_events = scan_yfinance_calendar(yf_tier1, 2)
    all_events = {(e["symbol"], e["date"]): e
                  for e in (upcoming + yf_events)}.values()

    tomorrow_events = [e for e in all_events if e["date"] == tomorrow]
    if not tomorrow_events:
        logger.info("No results scheduled for tomorrow (%s).", tomorrow)
        return

    logger.info("Found %d result(s) for tomorrow:", len(tomorrow_events))
    for ev in tomorrow_events:
        action_alert(ev["symbol"], days_to_event=1, result_date=tomorrow)


def action_alert(symbol: str, days_to_event: int = 1,
                  result_date: Optional[date] = None) -> None:
    """Build and send straddle alert for a specific symbol."""
    if result_date is None:
        result_date = date.today() + timedelta(days=days_to_event)

    print(f"\n📡  Building alert for {symbol} (results: {result_date})...")

    # Get market data
    spot = get_spot(symbol)
    if not spot:
        logger.error("Could not get spot price for %s", symbol)
        return

    prices = get_daily_prices(symbol, days=60)
    if prices is None or len(prices) < 20:
        logger.error("Could not get price history for %s", symbol)
        return

    straddle = estimate_straddle(symbol, spot, prices)
    message  = build_alert_message(symbol, result_date, straddle, days_to_event)

    # Print to console
    print(message.replace("<b>", "").replace("</b>", "")
                 .replace("<i>", "").replace("</i>", "")
                 .replace("&amp;", "&"))

    # Send Telegram
    sent = send_telegram(message)
    logger.info("Telegram alert %s for %s", "sent ✅" if sent else "failed ❌", symbol)


def action_execute(symbol: str, dry_run: bool = True) -> None:
    """Place straddle via Kite. dry_run=True by default — pass --live to really execute."""
    kite = init_kite()
    if not kite:
        return

    spot = get_spot(symbol)
    if not spot:
        logger.error("Cannot get spot for %s", symbol)
        return

    prices = get_daily_prices(symbol, days=60)
    if prices is None:
        logger.error("Cannot get price history for %s", symbol)
        return

    straddle = estimate_straddle(symbol, spot, prices)
    K        = straddle["strike"]
    lot      = straddle["lot"]
    expiry   = find_nearest_expiry(kite, symbol, min_dte=1)

    if not expiry:
        logger.error("Could not find upcoming expiry for %s", symbol)
        return

    mode = "DRY RUN" if dry_run else "LIVE EXECUTE"
    logger.info("[%s] %s straddle: %s %d%s CE+PE  expiry=%s  outlay≈₹%,.0f",
                mode, symbol, symbol, K, "", expiry, straddle["outlay"])

    result = place_straddle(kite, symbol, K, expiry, lot, dry_run=dry_run)

    if result["errors"]:
        for err in result["errors"]:
            logger.error("  ❌  %s", err)
        msg = (f"⚠️ Straddle execution failed for {symbol}:\n" +
               "\n".join(result["errors"]))
        send_telegram(msg)
    else:
        call_ts = result.get("ce", {}).get("tradingsymbol", "?")
        put_ts  = result.get("pe", {}).get("tradingsymbol", "?")
        status  = "DRY RUN (not placed)" if dry_run else "PLACED ✅"
        msg = (f"{'🧪' if dry_run else '⚡'} <b>Straddle {status}</b>\n"
               f"  {symbol} {K} CE: {call_ts}\n"
               f"  {symbol} {K} PE: {put_ts}\n"
               f"  Lot: {lot}  |  Outlay: ₹{straddle['outlay']:,.0f}\n"
               f"  {'Dry run — no order sent' if dry_run else 'Check Kite for order status'}")
        send_telegram(msg)
        print(msg)


def action_status() -> None:
    """Show all known upcoming result dates."""
    dates = load_dates()
    if not dates:
        print("\n  No manual dates stored. Add with:")
        print("  python straddle_executor.py --add SYMBOL YYYY-MM-DD\n")
        return
    today = date.today()
    print(f"\n  Stored result dates:\n")
    for sym, date_list in sorted(dates.items()):
        for d in sorted(date_list):
            past = " (past)" if d < today else f" (in {(d - today).days} days)"
            tier = STOCK_STATS.get(sym, {}).get("tier", 3)
            tier_label = {1: "🥇", 2: "🥈", 3: "🥉", 0: "⛔"}.get(tier, "")
            print(f"  {tier_label} {sym:<14}  {d}  {past}")
    print()


def action_monitor() -> None:
    """
    Run as a daily scheduler.
    Every trading day at 3:00 PM IST: scan tomorrow's results and send alerts.
    Every trading day at 9:15 AM IST: reminder if today has a result (exit plan).
    """
    print("\n⏰  Starting straddle monitor scheduler...")
    print("   Alert time: 3:00 PM IST daily (day-before alert)")
    print("   Exit reminder: 9:15 AM IST (if result day)\n")

    scheduler = BlockingScheduler(timezone=IST)

    def _daily_alert():
        logger.info("Running daily result scan...")
        action_today()

    def _exit_reminder():
        """If today is a result day for any tracked stock, send exit reminder."""
        today  = date.today()
        stored = load_dates()
        for sym, dates in stored.items():
            if today in dates:
                spot = get_spot(sym)
                msg = (
                    f"🔔 <b>EXIT REMINDER — {sym}</b>\n"
                    f"Results day is today!\n\n"
                    f"{'━' * 30}\n"
                    f"✅ Open: Sell losing leg at MARKET immediately\n"
                    f"⏰ <b>11:15 AM sharp: EXIT winning leg — NO EXCEPTIONS</b>\n"
                    f"  (2H exit consistently 2× better than EOD)\n"
                    f"{'━' * 30}\n"
                    f"Current spot: ₹{spot:,.2f}" if spot else ""
                )
                send_telegram(msg)
                logger.info("Exit reminder sent for %s", sym)

    scheduler.add_job(_daily_alert, CronTrigger(
        day_of_week="mon-fri", hour=15, minute=0, timezone=IST))
    scheduler.add_job(_exit_reminder, CronTrigger(
        day_of_week="mon-fri", hour=9, minute=15, timezone=IST))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Monitor stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-Event Straddle Alert & Executor for NIFTY50 stocks."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan",    action="store_true",
                        help="Scan earnings calendar for upcoming results.")
    group.add_argument("--today",   action="store_true",
                        help="Send alerts for results happening tomorrow (run at 3 PM).")
    group.add_argument("--add",     nargs=2, metavar=("SYMBOL", "DATE"),
                        help="Add a known result date. DATE format: YYYY-MM-DD")
    group.add_argument("--alert",   metavar="SYMBOL",
                        help="Force send straddle alert for a symbol now.")
    group.add_argument("--execute", metavar="SYMBOL",
                        help="Place straddle order via Kite for a symbol.")
    group.add_argument("--monitor", action="store_true",
                        help="Run as daily scheduler (3 PM alert + 9:15 AM exit reminder).")
    group.add_argument("--status",  action="store_true",
                        help="Show all stored result dates.")

    p.add_argument("--days",  type=int, default=7,
                    help="Scan window in days (used with --scan, default 7).")
    p.add_argument("--result-date", metavar="YYYY-MM-DD",
                    help="Override result date for --alert (default: tomorrow).")
    p.add_argument("--live", action="store_true",
                    help="With --execute: actually place orders (default is dry-run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.scan:
        action_scan(args.days)

    elif args.today:
        action_today()

    elif args.add:
        symbol = args.add[0].upper()
        try:
            d = date.fromisoformat(args.add[1])
        except ValueError:
            print(f"❌  Invalid date format: {args.add[1]}. Use YYYY-MM-DD")
            sys.exit(1)
        add_date(symbol, d)

    elif args.alert:
        symbol = args.alert.upper()
        result_date = None
        if args.result_date:
            try:
                result_date = date.fromisoformat(args.result_date)
            except ValueError:
                print(f"❌  Invalid date: {args.result_date}")
                sys.exit(1)
        days = (result_date - date.today()).days if result_date else 1
        action_alert(symbol, days_to_event=max(1, days), result_date=result_date)

    elif args.execute:
        symbol = args.execute.upper()
        dry = not args.live
        if not dry:
            confirm = input(
                f"⚠️  LIVE MODE: This will place REAL straddle orders for {symbol}. "
                f"Type YES to confirm: "
            )
            if confirm.strip().upper() != "YES":
                print("Cancelled.")
                return
        action_execute(symbol, dry_run=dry)

    elif args.monitor:
        action_monitor()

    elif args.status:
        action_status()


if __name__ == "__main__":
    main()
