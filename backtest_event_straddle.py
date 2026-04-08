#!/usr/bin/env python3
"""
backtest_event_straddle.py — Event-Driven Options Strategy Backtest

Tests two institutional approaches on NIFTY50 stocks:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY 1 — PRE-EVENT STRADDLE
  Buy ATM call + ATM put at close of the day BEFORE a known event
  (earnings/result day identified by next-day gap ≥ threshold%).
  Direction is unknown — profit if the overnight move is larger
  than the straddle cost (break-even = ±straddle_cost / spot).

  Entry  : day-before close → buy ATM straddle (IV inflated 1.4×)
  Exit   : event-day open → sell loser at 15% of cost
           hold winner for 1H / 2H / EOD; model with BS (IV crush)
  P&L    : winner_exit + loser_salvage − straddle_cost

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY 2 — FIRST-CANDLE MOMENTUM (second-wave entry)
  On earnings gap days (≥ threshold%), wait for the first 1-hour
  candle to close BEFORE entering. This skips the initial spike/fade
  and catches the confirmed second wave — the way institutions trade
  post-event continuation.

  Entry  : after first 1H candle close → enter ATM option in
           candle direction (CE if candle bullish, PE if bearish)
  Exit   : hold 1H / 2H / 3H after entry; model with BS
  P&L    : exit_premium − entry_premium (per option, scaled to lot)
  Guard  : only trade if first candle confirms gap direction
           (i.e. candle close > open for gap-up, else skip)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data sources:
  • yfinance daily data (2Y) for gap detection + volatility
  • yfinance 1H data (730 days) for intraday candle simulation
  • Black-Scholes for option premium estimation
  • IV model: base_vol × 1.4 (pre-event) → base_vol × 0.9 (post-event crush)

Usage:
  python backtest_event_straddle.py                         # all 50 stocks, both strategies
  python backtest_event_straddle.py --stock INDIGO          # single stock
  python backtest_event_straddle.py --stock RELIANCE INFY   # multiple stocks
  python backtest_event_straddle.py --strategy straddle     # strategy 1 only
  python backtest_event_straddle.py --strategy momentum     # strategy 2 only
  python backtest_event_straddle.py --gap-min 2             # min gap % (default 3)
  python backtest_event_straddle.py --days 365              # lookback in days (default 730)
  python backtest_event_straddle.py --top 10                # show top N stocks only
  python backtest_event_straddle.py --list-stocks           # list all available tickers
  python backtest_event_straddle.py --lot-size 500          # custom lot size (default: auto)
  python backtest_event_straddle.py --show-trades           # print every individual trade
"""

import argparse
import math
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc — no scipy needed."""
    return 0.5 * math.erfc(-x / math.sqrt(2))

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE      = 0.065       # 6.5% — approximate Indian Gsec rate
PRE_EVENT_IV_MULT   = 1.40        # IV inflates 40% before results (conservative)
POST_EVENT_IV_MULT  = 0.90        # IV crushes to 90% of base after event
LOSER_SALVAGE_PCT   = 0.15        # sell losing leg at 15% of its cost
BASE_VOL_WINDOW     = 30          # days of history for realized vol
MIN_VOL             = 0.15        # floor — 15% annualized vol minimum
MAX_VOL             = 1.20        # cap — 120% annualized vol maximum
STRIKE_ROUND_MAP    = {           # round ATM to nearest N rupees
    "default": 50,
    "TATAMOTORS": 5,
    "TATASTEEL": 5,
    "HINDUNILVR": 100,
    "MARUTI": 500,
    "EICHERMOT": 100,
    "BAJAJ-AUTO": 100,
    "BPCL": 5,
    "COALINDIA": 5,
    "BEL": 5,
}
LOT_SIZE_MAP = {                  # approximate NSE lot sizes
    "RELIANCE": 250, "TCS": 150, "HDFCBANK": 550, "INFY": 300,
    "ICICIBANK": 700, "HINDUNILVR": 300, "SBIN": 1500, "BHARTIARTL": 1851,
    "ITC": 3200, "KOTAKBANK": 400, "AXISBANK": 1200, "LT": 300,
    "ASIANPAINT": 200, "MARUTI": 100, "TITAN": 375, "WIPRO": 1500,
    "ULTRACEMCO": 100, "BAJFINANCE": 125, "HCLTECH": 700, "NESTLEIND": 40,
    "POWERGRID": 4900, "NTPC": 4500, "ONGC": 1925, "JSWSTEEL": 675,
    "INDUSINDBK": 600, "TATAMOTORS": 2400, "TECHM": 600, "SUNPHARMA": 700,
    "ADANIENT": 250, "BAJAJFINSV": 500, "CIPLA": 650, "DRREDDY": 125,
    "EICHERMOT": 200, "GRASIM": 250, "HEROMOTOCO": 300, "HINDALCO": 1400,
    "M&M": 700, "TATASTEEL": 5500, "TATACONSUM": 1100, "APOLLOHOSP": 125,
    "BPCL": 1800, "COALINDIA": 2800, "DIVISLAB": 100, "INDIGO": 300,
    "SBILIFE": 750, "SHRIRAMFIN": 600, "TRENT": 500,
    "BAJAJ-AUTO": 75, "HDFCLIFE": 1100, "BEL": 4900,
}

# ── NIFTY50 Universe ───────────────────────────────────────────────────────────
NIFTY50 = {
    "RELIANCE":   "RELIANCE.NS",  "TCS":        "TCS.NS",
    "HDFCBANK":   "HDFCBANK.NS",  "INFY":       "INFY.NS",
    "ICICIBANK":  "ICICIBANK.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN":       "SBIN.NS",      "BHARTIARTL": "BHARTIARTL.NS",
    "ITC":        "ITC.NS",       "KOTAKBANK":  "KOTAKBANK.NS",
    "AXISBANK":   "AXISBANK.NS",  "LT":         "LT.NS",
    "ASIANPAINT": "ASIANPAINT.NS","MARUTI":     "MARUTI.NS",
    "TITAN":      "TITAN.NS",     "WIPRO":      "WIPRO.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS","BAJFINANCE": "BAJFINANCE.NS",
    "HCLTECH":    "HCLTECH.NS",   "NESTLEIND":  "NESTLEIND.NS",
    "POWERGRID":  "POWERGRID.NS", "NTPC":       "NTPC.NS",
    "ONGC":       "ONGC.NS",      "JSWSTEEL":   "JSWSTEEL.NS",
    "INDUSINDBK": "INDUSINDBK.NS","TATAMOTORS": "TATAMOTORS.NS",
    "TECHM":      "TECHM.NS",     "SUNPHARMA":  "SUNPHARMA.NS",
    "ADANIENT":   "ADANIENT.NS",  "BAJAJFINSV": "BAJAJFINSV.NS",
    "CIPLA":      "CIPLA.NS",     "DRREDDY":    "DRREDDY.NS",
    "EICHERMOT":  "EICHERMOT.NS", "GRASIM":     "GRASIM.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS","HINDALCO":   "HINDALCO.NS",
    "M&M":        "M&M.NS",       "TATASTEEL":  "TATASTEEL.NS",
    "TATACONSUM": "TATACONSUM.NS","APOLLOHOSP": "APOLLOHOSP.NS",
    "BPCL":       "BPCL.NS",      "COALINDIA":  "COALINDIA.NS",
    "DIVISLAB":   "DIVISLAB.NS",  "INDIGO":     "INDIGO.NS",
    "SBILIFE":    "SBILIFE.NS",   "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "TRENT":      "TRENT.NS",     "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HDFCLIFE":   "HDFCLIFE.NS",  "BEL":        "BEL.NS",
}


# ── Black-Scholes Pricer ───────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, S - K)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price. T in years."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if d1 is None:
        return max(0.0, K - S)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def atm_strike(spot: float, symbol: str) -> float:
    """Round spot to nearest ATM strike for this symbol."""
    rnd = STRIKE_ROUND_MAP.get(symbol, STRIKE_ROUND_MAP["default"])
    return round(spot / rnd) * rnd


def realized_vol(close_series: pd.Series, window: int = BASE_VOL_WINDOW) -> float:
    """Annualized realized volatility (log-return std × √252)."""
    if len(close_series) < window + 2:
        return 0.30
    log_ret = np.log(close_series / close_series.shift(1)).dropna()
    vol = float(log_ret.tail(window).std() * math.sqrt(252))
    return max(MIN_VOL, min(MAX_VOL, vol))


# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_daily(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """Download daily OHLCV from yfinance. Returns None on failure."""
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 30:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index = df.index.normalize()
        return df
    except Exception:
        return None


def load_hourly(ticker: str) -> Optional[pd.DataFrame]:
    """
    Download 1H intraday data (yfinance supports ~730 days for 1h).
    Returns None on failure or insufficient data.
    """
    try:
        df = yf.download(ticker, period="730d", interval="1h",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 20:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return df
    except Exception:
        return None


def get_earnings_dates(ticker: str) -> list[date]:
    """
    Try to fetch earnings dates from yfinance.
    Returns list of dates (may be empty for NSE stocks with limited coverage).
    """
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=24)
        if ed is None or ed.empty:
            return []
        dates = []
        for d in ed.index:
            try:
                dt = pd.Timestamp(d).date() if not isinstance(d, date) else d
                dates.append(dt)
            except Exception:
                pass
        return dates
    except Exception:
        return []


def find_gap_events(daily: pd.DataFrame, gap_min_pct: float) -> list[dict]:
    """
    Find all days where open gapped ≥ gap_min_pct% from previous close.
    Returns list of dicts with event metadata.
    """
    events = []
    closes = daily["Close"].squeeze()
    opens  = daily["Open"].squeeze()
    highs  = daily["High"].squeeze()
    lows   = daily["Low"].squeeze()

    for i in range(1, len(daily)):
        prev_close = float(closes.iloc[i - 1])
        today_open = float(opens.iloc[i])
        today_high = float(highs.iloc[i])
        today_low  = float(lows.iloc[i])
        today_close= float(closes.iloc[i])
        if prev_close <= 0:
            continue
        gap_pct = (today_open - prev_close) / prev_close * 100.0
        if abs(gap_pct) < gap_min_pct:
            continue
        events.append({
            "event_date":   daily.index[i].date(),
            "prev_date":    daily.index[i - 1].date(),
            "prev_close":   prev_close,
            "event_open":   today_open,
            "event_high":   today_high,
            "event_low":    today_low,
            "event_close":  today_close,
            "gap_pct":      gap_pct,
            "day_idx":      i,
        })
    return events


# ── Strategy 1 — Pre-Event Straddle ───────────────────────────────────────────

def backtest_straddle(
    symbol: str,
    daily: pd.DataFrame,
    gap_min_pct: float,
    lot_size: Optional[int],
    show_trades: bool,
    earnings_dates: list[date],
) -> dict:
    """
    For each gap event, simulate buying a straddle the day before.

    Methodology:
      1. Entry: day-before close price, ATM straddle using BS with
         base_vol × PRE_EVENT_IV_MULT. DTE assumed = 1 (expiry next day
         after results — weekly/monthly options).
      2. Event day open: determine direction.
      3. Loser leg: sold at LOSER_SALVAGE_PCT × cost.
      4. Winner leg: valued at close-of-event-day using BS with
         base_vol × POST_EVENT_IV_MULT and DTE ≈ 0.
         Also estimate 1H and 2H exit using intraday range proxy.
      5. P&L = winner_exit + loser_salvage - straddle_cost (per unit).
         Scale by lot_size for ₹ P&L.
    """
    ls = lot_size or LOT_SIZE_MAP.get(symbol, 500)
    events = find_gap_events(daily, gap_min_pct)
    if not events:
        return {"symbol": symbol, "strategy": "straddle", "events": 0}

    closes = daily["Close"].squeeze()

    results = {
        "symbol": symbol, "strategy": "straddle",
        "events": 0,
        "wins_eod": 0, "wins_1h": 0, "wins_2h": 0,
        "total_pnl_eod": 0.0, "total_pnl_1h": 0.0, "total_pnl_2h": 0.0,
        "avg_straddle_cost": 0.0,
        "avg_move_pct": 0.0,
        "be_rate": 0.0,   # % of events where move > straddle cost (break-even)
        "trades": [],
    }

    straddle_costs = []
    move_pcts = []
    be_hits = 0

    for ev in events:
        i = ev["day_idx"]
        if i < BASE_VOL_WINDOW + 2:
            continue  # not enough history for vol calculation

        prev_close = ev["prev_close"]
        event_open = ev["event_open"]
        event_close= ev["event_close"]
        event_high = ev["event_high"]
        event_low  = ev["event_low"]
        gap_pct    = ev["gap_pct"]

        # Base vol from history up to day-before
        hist_closes = closes.iloc[max(0, i - BASE_VOL_WINDOW - 5): i]
        base_vol = realized_vol(hist_closes)
        pre_vol  = base_vol * PRE_EVENT_IV_MULT
        post_vol = base_vol * POST_EVENT_IV_MULT

        # ATM strike at prev close
        K = atm_strike(prev_close, symbol)
        T_entry = 1.0 / 365.0    # 1 DTE (buying day before weekly expiry)

        call_cost = bs_call(prev_close, K, T_entry, RISK_FREE_RATE, pre_vol)
        put_cost  = bs_put (prev_close, K, T_entry, RISK_FREE_RATE, pre_vol)
        straddle_cost = call_cost + put_cost

        if straddle_cost <= 0:
            continue

        # Direction from gap
        direction = "up" if gap_pct > 0 else "down"

        # Loser salvage
        if direction == "up":
            winner_cost = call_cost
            loser_cost  = put_cost
        else:
            winner_cost = put_cost
            loser_cost  = call_cost
        loser_salvage = loser_cost * LOSER_SALVAGE_PCT

        # Actual move from prev_close to event_open (gap move)
        move_pct = abs(gap_pct)
        move_pcts.append(move_pct)

        # Break-even: did the move exceed straddle cost as % of strike?
        be_pct = (straddle_cost / K) * 100.0
        if move_pct >= be_pct:
            be_hits += 1

        # ── EOD exit (event day close) ──────────────────────────────────────
        # Winner valued at event close using BS with post-event IV, DTE ≈ 0
        T_eod = 0.25 / 365.0   # ~6 hours remaining (tiny)
        if direction == "up":
            winner_eod = bs_call(event_close, K, T_eod, RISK_FREE_RATE, post_vol)
        else:
            winner_eod = bs_put(event_close, K, T_eod, RISK_FREE_RATE, post_vol)
        # floor: intrinsic value
        if direction == "up":
            winner_eod = max(winner_eod, max(0.0, event_close - K))
        else:
            winner_eod = max(winner_eod, max(0.0, K - event_close))

        pnl_eod = (winner_eod + loser_salvage - straddle_cost) * ls

        # ── 1H exit proxy: spot moves 40% of full day range from open ──────
        # Approximation: after first hour, stock has moved ~40% of day range
        if direction == "up":
            spot_1h = event_open + 0.40 * (event_high - event_open)
        else:
            spot_1h = event_open - 0.40 * (event_open - event_low)
        T_1h = 20.0 / 365.0   # ~5 hours remaining
        if direction == "up":
            winner_1h = bs_call(spot_1h, K, T_1h, RISK_FREE_RATE, post_vol)
            winner_1h = max(winner_1h, max(0.0, spot_1h - K))
        else:
            winner_1h = bs_put(spot_1h, K, T_1h, RISK_FREE_RATE, post_vol)
            winner_1h = max(winner_1h, max(0.0, K - spot_1h))
        pnl_1h = (winner_1h + loser_salvage - straddle_cost) * ls

        # ── 2H exit proxy: spot moves 65% of full day range from open ──────
        if direction == "up":
            spot_2h = event_open + 0.65 * (event_high - event_open)
        else:
            spot_2h = event_open - 0.65 * (event_open - event_low)
        T_2h = 15.0 / 365.0
        if direction == "up":
            winner_2h = bs_call(spot_2h, K, T_2h, RISK_FREE_RATE, post_vol)
            winner_2h = max(winner_2h, max(0.0, spot_2h - K))
        else:
            winner_2h = bs_put(spot_2h, K, T_2h, RISK_FREE_RATE, post_vol)
            winner_2h = max(winner_2h, max(0.0, K - spot_2h))
        pnl_2h = (winner_2h + loser_salvage - straddle_cost) * ls

        results["events"]        += 1
        results["total_pnl_eod"] += pnl_eod
        results["total_pnl_1h"]  += pnl_1h
        results["total_pnl_2h"]  += pnl_2h
        if pnl_eod > 0: results["wins_eod"] += 1
        if pnl_1h  > 0: results["wins_1h"]  += 1
        if pnl_2h  > 0: results["wins_2h"]  += 1
        straddle_costs.append(straddle_cost)

        trade = {
            "date":           ev["event_date"],
            "spot":           round(prev_close, 2),
            "strike":         K,
            "gap_pct":        round(gap_pct, 2),
            "direction":      direction,
            "straddle_cost":  round(straddle_cost, 2),
            "be_pct":         round(be_pct, 2),
            "move_pct":       round(move_pct, 2),
            "pnl_1h":         round(pnl_1h, 0),
            "pnl_2h":         round(pnl_2h, 0),
            "pnl_eod":        round(pnl_eod, 0),
        }
        results["trades"].append(trade)

        if show_trades:
            verdict = "✅" if pnl_eod > 0 else "❌"
            print(
                f"  {verdict} {ev['event_date']}  gap={gap_pct:+.1f}%  "
                f"straddle=₹{straddle_cost:.1f}  BE={be_pct:.1f}%  "
                f"move={move_pct:.1f}%  "
                f"P&L: 1H=₹{pnl_1h:,.0f}  2H=₹{pnl_2h:,.0f}  EOD=₹{pnl_eod:,.0f}"
            )

    n = results["events"]
    if n > 0:
        results["avg_straddle_cost"] = round(float(np.mean(straddle_costs)), 2)
        results["avg_move_pct"]      = round(float(np.mean(move_pcts)), 2)
        results["be_rate"]           = round(be_hits / n * 100, 1)
        results["win_rate_eod"]      = round(results["wins_eod"] / n * 100, 1)
        results["win_rate_1h"]       = round(results["wins_1h"]  / n * 100, 1)
        results["win_rate_2h"]       = round(results["wins_2h"]  / n * 100, 1)
        results["avg_pnl_eod"]       = round(results["total_pnl_eod"] / n, 0)
        results["avg_pnl_1h"]        = round(results["total_pnl_1h"]  / n, 0)
        results["avg_pnl_2h"]        = round(results["total_pnl_2h"]  / n, 0)

    return results


# ── Strategy 2 — First-Candle Momentum ────────────────────────────────────────

def backtest_momentum(
    symbol: str,
    daily: pd.DataFrame,
    hourly: Optional[pd.DataFrame],
    gap_min_pct: float,
    lot_size: Optional[int],
    show_trades: bool,
) -> dict:
    """
    After a gap event, wait for first 1H candle to close before entering.

    Methodology:
      1. Identify gap events from daily data (same as straddle).
      2. For each event day, look up the first 1H candle in hourly data.
         First candle = the candle starting at 9:15 AM IST.
      3. Determine entry direction from first candle:
         - Gap-up AND first candle closes UP → buy CE (confirmed bull)
         - Gap-up AND first candle closes DOWN → SKIP (reversal risk)
         - Gap-down AND first candle closes DOWN → buy PE (confirmed bear)
         - Gap-down AND first candle closes UP → SKIP
      4. Entry premium: BS(spot_at_entry, ATM_K, remaining_T, post_vol)
         where spot_at_entry ≈ first candle close price.
      5. Exit: use subsequent 1H candles (+1H, +2H, +3H) to estimate
         exit spot, reprice option at exit.
      6. P&L = (exit_premium - entry_premium) × lot_size.

    If hourly data is unavailable (yfinance limitation), falls back to
    daily OHLC proxy for a rough estimate.
    """
    ls = lot_size or LOT_SIZE_MAP.get(symbol, 500)
    events = find_gap_events(daily, gap_min_pct)
    closes = daily["Close"].squeeze()

    results = {
        "symbol": symbol, "strategy": "momentum",
        "events_found": len(events),
        "events_traded": 0,
        "events_skipped_reversal": 0,
        "wins_1h": 0, "wins_2h": 0, "wins_3h": 0,
        "total_pnl_1h": 0.0, "total_pnl_2h": 0.0, "total_pnl_3h": 0.0,
        "trades": [],
    }

    has_hourly = hourly is not None and not hourly.empty

    for ev in events:
        i = ev["day_idx"]
        if i < BASE_VOL_WINDOW + 2:
            continue

        gap_pct    = ev["gap_pct"]
        event_date = ev["event_date"]
        prev_close = ev["prev_close"]
        event_open = ev["event_open"]
        event_high = ev["event_high"]
        event_low  = ev["event_low"]
        event_close= ev["event_close"]

        hist_closes = closes.iloc[max(0, i - BASE_VOL_WINDOW - 5): i]
        base_vol    = realized_vol(hist_closes)
        entry_vol   = base_vol * POST_EVENT_IV_MULT   # IV has partially crushed by 1H candle close
        K = atm_strike(event_open, symbol)             # strike at event open (ATM)

        # ── First candle direction ──────────────────────────────────────────
        first_candle_open  = None
        first_candle_close = None
        spot_1h = None   # spot price 1H after entry
        spot_2h = None
        spot_3h = None

        if has_hourly:
            # Filter hourly candles for this event date
            day_candles = hourly[
                (hourly.index.date == event_date) &   # type: ignore[attr-defined]
                (hourly.index.hour >= 9) &
                (hourly.index.hour <= 15)
            ]
            if len(day_candles) >= 2:
                c0 = day_candles.iloc[0]   # first candle (9:15–10:15)
                first_candle_open  = float(c0["Open"].squeeze() if hasattr(c0["Open"], "squeeze") else c0["Open"])
                first_candle_close = float(c0["Close"].squeeze() if hasattr(c0["Close"], "squeeze") else c0["Close"])

                # Subsequent candle closes for exit spots
                spot_1h = float(day_candles.iloc[1]["Close"].squeeze() if hasattr(day_candles.iloc[1]["Close"], "squeeze") else day_candles.iloc[1]["Close"])
                if len(day_candles) >= 3:
                    spot_2h = float(day_candles.iloc[2]["Close"].squeeze() if hasattr(day_candles.iloc[2]["Close"], "squeeze") else day_candles.iloc[2]["Close"])
                else:
                    spot_2h = event_close
                if len(day_candles) >= 4:
                    spot_3h = float(day_candles.iloc[3]["Close"].squeeze() if hasattr(day_candles.iloc[3]["Close"], "squeeze") else day_candles.iloc[3]["Close"])
                else:
                    spot_3h = event_close

        # Fallback to daily OHLC proxy if no hourly
        if first_candle_open is None:
            # Proxy: first candle open = event_open, close = interpolated
            first_candle_open  = event_open
            # Assume first hour moves 35% of daily range
            if gap_pct > 0:
                first_candle_close = event_open + 0.35 * (event_high - event_open)
            else:
                first_candle_close = event_open - 0.35 * (event_open - event_low)
            spot_1h = event_open + 0.50 * (event_close - event_open)
            spot_2h = event_open + 0.75 * (event_close - event_open)
            spot_3h = event_close

        # ── Confirmation check ──────────────────────────────────────────────
        candle_bullish = first_candle_close > first_candle_open
        candle_bearish = first_candle_close < first_candle_open
        gap_up   = gap_pct > 0
        gap_down = gap_pct < 0

        if gap_up and candle_bullish:
            direction  = "up"
            entry_spot = first_candle_close
        elif gap_down and candle_bearish:
            direction  = "down"
            entry_spot = first_candle_close
        else:
            # Candle does NOT confirm gap direction → skip (reversal risk)
            results["events_skipped_reversal"] += 1
            if show_trades:
                print(
                    f"  ⏭  {event_date}  gap={gap_pct:+.1f}%  "
                    f"first candle reversed — SKIP"
                )
            continue

        # ── Entry premium ───────────────────────────────────────────────────
        T_entry = 5.0 / 365.0    # ~5 trading hours remaining in the day
        if direction == "up":
            entry_prem = bs_call(entry_spot, K, T_entry, RISK_FREE_RATE, entry_vol)
            entry_prem = max(entry_prem, max(0.0, entry_spot - K))
        else:
            entry_prem = bs_put(entry_spot, K, T_entry, RISK_FREE_RATE, entry_vol)
            entry_prem = max(entry_prem, max(0.0, K - entry_spot))

        if entry_prem <= 0.5:
            continue   # degenerate — skip

        # ── Exit premiums at +1H, +2H, +3H ─────────────────────────────────
        def option_exit(spot_exit: float, T_remaining_hrs: float) -> float:
            T = max(0.0, T_remaining_hrs / 365.0)
            if direction == "up":
                v = bs_call(spot_exit, K, T, RISK_FREE_RATE, entry_vol)
                return max(v, max(0.0, spot_exit - K))
            else:
                v = bs_put(spot_exit, K, T, RISK_FREE_RATE, entry_vol)
                return max(v, max(0.0, K - spot_exit))

        exit_1h = option_exit(spot_1h,  4.0)
        exit_2h = option_exit(spot_2h,  3.0)
        exit_3h = option_exit(spot_3h,  2.0)

        pnl_1h = (exit_1h - entry_prem) * ls
        pnl_2h = (exit_2h - entry_prem) * ls
        pnl_3h = (exit_3h - entry_prem) * ls

        results["events_traded"]  += 1
        results["total_pnl_1h"]   += pnl_1h
        results["total_pnl_2h"]   += pnl_2h
        results["total_pnl_3h"]   += pnl_3h
        if pnl_1h > 0: results["wins_1h"] += 1
        if pnl_2h > 0: results["wins_2h"] += 1
        if pnl_3h > 0: results["wins_3h"] += 1

        trade = {
            "date":         event_date,
            "spot":         round(event_open, 2),
            "strike":       K,
            "gap_pct":      round(gap_pct, 2),
            "direction":    direction,
            "entry_prem":   round(entry_prem, 2),
            "pnl_1h":       round(pnl_1h, 0),
            "pnl_2h":       round(pnl_2h, 0),
            "pnl_3h":       round(pnl_3h, 0),
            "data_source":  "hourly" if has_hourly else "daily_proxy",
        }
        results["trades"].append(trade)

        if show_trades:
            verdict = "✅" if pnl_2h > 0 else "❌"
            src = "H" if has_hourly else "~"
            print(
                f"  {verdict}[{src}] {event_date}  gap={gap_pct:+.1f}%  "
                f"dir={direction}  entry=₹{entry_prem:.1f}  "
                f"P&L: 1H=₹{pnl_1h:,.0f}  2H=₹{pnl_2h:,.0f}  3H=₹{pnl_3h:,.0f}"
            )

    n = results["events_traded"]
    if n > 0:
        results["win_rate_1h"] = round(results["wins_1h"] / n * 100, 1)
        results["win_rate_2h"] = round(results["wins_2h"] / n * 100, 1)
        results["win_rate_3h"] = round(results["wins_3h"] / n * 100, 1)
        results["avg_pnl_1h"]  = round(results["total_pnl_1h"] / n, 0)
        results["avg_pnl_2h"]  = round(results["total_pnl_2h"] / n, 0)
        results["avg_pnl_3h"]  = round(results["total_pnl_3h"] / n, 0)

    return results


# ── Results Display ────────────────────────────────────────────────────────────

def print_straddle_summary(all_results: list[dict], top: int) -> None:
    print("\n" + "═" * 80)
    print("  STRATEGY 1 — PRE-EVENT STRADDLE RESULTS")
    print("  (Buy ATM call + put at prev close → sell loser at open → hold winner)")
    print("═" * 80)

    valid = [r for r in all_results if r.get("events", 0) > 0]
    if not valid:
        print("  No events found matching criteria.")
        return

    # Sort by total EOD P&L descending
    valid.sort(key=lambda r: r.get("total_pnl_eod", 0), reverse=True)
    if top:
        valid = valid[:top]

    # Header
    print(f"\n  {'STOCK':<14} {'EVENTS':>6} {'BE%':>5} {'AVG MOVE':>9} {'AVG COST':>9} "
          f"{'WIN%(EOD)':>10} {'TOTAL(EOD)':>12} {'WIN%(2H)':>9} {'TOTAL(2H)':>11}")
    print("  " + "─" * 92)

    total_pnl_eod = 0.0
    total_pnl_2h  = 0.0
    for r in valid:
        n   = r.get("events", 0)
        be  = r.get("be_rate", 0)
        mv  = r.get("avg_move_pct", 0)
        sc  = r.get("avg_straddle_cost", 0)
        we  = r.get("win_rate_eod", 0)
        te  = r.get("total_pnl_eod", 0)
        w2  = r.get("win_rate_2h", 0)
        t2  = r.get("total_pnl_2h", 0)
        total_pnl_eod += te
        total_pnl_2h  += t2
        print(
            f"  {r['symbol']:<14} {n:>6}  {be:>4.0f}%  {mv:>7.1f}%  ₹{sc:>7.1f}  "
            f"{we:>8.1f}%  ₹{te:>10,.0f}  {w2:>7.1f}%  ₹{t2:>9,.0f}"
        )

    print("  " + "─" * 92)
    print(f"  {'TOTAL':<14} {'':>6} {'':>5} {'':>9} {'':>9} {'':>10} "
          f"₹{total_pnl_eod:>10,.0f} {'':>9} ₹{total_pnl_2h:>9,.0f}")

    # Best events across all stocks
    all_trades = []
    for r in all_results:
        for t in r.get("trades", []):
            t["symbol"] = r["symbol"]
            all_trades.append(t)

    if all_trades:
        best = sorted(all_trades, key=lambda t: t.get("pnl_eod", 0), reverse=True)[:5]
        worst = sorted(all_trades, key=lambda t: t.get("pnl_eod", 0))[:5]
        print("\n  ── TOP 5 BEST STRADDLE TRADES ──")
        for t in best:
            print(f"  ✅ {t['symbol']} {t['date']}  gap={t['gap_pct']:+.1f}%  "
                  f"move={t['move_pct']:.1f}% vs BE={t['be_pct']:.1f}%  EOD=₹{t['pnl_eod']:,.0f}")
        print("\n  ── TOP 5 WORST STRADDLE TRADES ──")
        for t in worst:
            print(f"  ❌ {t['symbol']} {t['date']}  gap={t['gap_pct']:+.1f}%  "
                  f"move={t['move_pct']:.1f}% vs BE={t['be_pct']:.1f}%  EOD=₹{t['pnl_eod']:,.0f}")


def print_momentum_summary(all_results: list[dict], top: int) -> None:
    print("\n" + "═" * 80)
    print("  STRATEGY 2 — FIRST-CANDLE MOMENTUM RESULTS")
    print("  (Wait for first 1H candle → enter only if confirms gap direction)")
    print("  [H] = actual hourly data  [~] = daily proxy estimate")
    print("═" * 80)

    valid = [r for r in all_results if r.get("events_traded", 0) > 0]
    if not valid:
        print("  No confirmed trades found matching criteria.")
        return

    valid.sort(key=lambda r: r.get("total_pnl_2h", 0), reverse=True)
    if top:
        valid = valid[:top]

    print(f"\n  {'STOCK':<14} {'FOUND':>6} {'TRADED':>7} {'SKIP%':>6} "
          f"{'WIN%(1H)':>9} {'TOT(1H)':>10} {'WIN%(2H)':>9} {'TOT(2H)':>10} {'WIN%(3H)':>9} {'TOT(3H)':>10}")
    print("  " + "─" * 100)

    total_1h = 0.0
    total_2h = 0.0
    total_3h = 0.0
    for r in valid:
        found   = r.get("events_found", 0)
        traded  = r.get("events_traded", 0)
        skipped = r.get("events_skipped_reversal", 0)
        skip_pct= (skipped / found * 100) if found > 0 else 0
        w1  = r.get("win_rate_1h", 0)
        t1  = r.get("total_pnl_1h", 0)
        w2  = r.get("win_rate_2h", 0)
        t2  = r.get("total_pnl_2h", 0)
        w3  = r.get("win_rate_3h", 0)
        t3  = r.get("total_pnl_3h", 0)
        total_1h += t1
        total_2h += t2
        total_3h += t3
        print(
            f"  {r['symbol']:<14} {found:>6} {traded:>7}  {skip_pct:>4.0f}%  "
            f"{w1:>7.1f}%  ₹{t1:>8,.0f}  {w2:>7.1f}%  ₹{t2:>8,.0f}  "
            f"{w3:>7.1f}%  ₹{t3:>8,.0f}"
        )

    print("  " + "─" * 100)
    print(f"  {'TOTAL':<14} {'':>6} {'':>7} {'':>6} {'':>9} "
          f"₹{total_1h:>8,.0f} {'':>9} ₹{total_2h:>8,.0f} {'':>9} ₹{total_3h:>8,.0f}")

    # Skip rate insight
    all_found  = sum(r.get("events_found", 0)  for r in valid)
    all_traded = sum(r.get("events_traded", 0) for r in valid)
    all_skip   = sum(r.get("events_skipped_reversal", 0) for r in valid)
    if all_found > 0:
        print(f"\n  Confirmation filter: {all_skip}/{all_found} events skipped "
              f"({all_skip/all_found*100:.0f}%) — first candle reversed gap direction.")
        print(f"  Only {all_traded}/{all_found} events ({all_traded/all_found*100:.0f}%) "
              f"had confirmed first-candle continuation.")


def print_comparison(s_results: list[dict], m_results: list[dict]) -> None:
    print("\n" + "═" * 80)
    print("  STRATEGY COMPARISON — WHICH IS BETTER?")
    print("═" * 80)

    total_s_eod = sum(r.get("total_pnl_eod", 0) for r in s_results if r.get("events", 0) > 0)
    total_s_2h  = sum(r.get("total_pnl_2h",  0) for r in s_results if r.get("events", 0) > 0)
    total_m_1h  = sum(r.get("total_pnl_1h",  0) for r in m_results if r.get("events_traded", 0) > 0)
    total_m_2h  = sum(r.get("total_pnl_2h",  0) for r in m_results if r.get("events_traded", 0) > 0)

    trades_s = sum(r.get("events", 0) for r in s_results)
    trades_m = sum(r.get("events_traded", 0) for r in m_results)

    wins_s = sum(r.get("wins_eod", 0) for r in s_results)
    wins_m = sum(r.get("wins_2h", 0) for r in m_results)

    wr_s = (wins_s / trades_s * 100) if trades_s > 0 else 0
    wr_m = (wins_m / trades_m * 100) if trades_m > 0 else 0

    print(f"\n  {'Metric':<30} {'Straddle':>15} {'Momentum':>15}")
    print("  " + "─" * 62)
    print(f"  {'Total trades':<30} {trades_s:>15} {trades_m:>15}")
    print(f"  {'Win rate (primary hold)':<30} {wr_s:>14.1f}% {wr_m:>14.1f}%")
    print(f"  {'Total P&L (EOD/2H)':<30} ₹{total_s_eod:>13,.0f} ₹{total_m_2h:>13,.0f}")
    print(f"  {'Total P&L (2H/1H)':<30} ₹{total_s_2h:>13,.0f} ₹{total_m_1h:>13,.0f}")
    print(f"  {'Avg P&L per trade (EOD/2H)':<30} "
          f"₹{(total_s_eod/trades_s if trades_s else 0):>13,.0f} "
          f"₹{(total_m_2h/trades_m if trades_m else 0):>13,.0f}")

    print("\n  Key differences:")
    print("  • Straddle: direction-agnostic, buys BEFORE event, profits if move > straddle cost")
    print("  • Momentum: direction-selective, enters AFTER first candle confirms, skips reversals")
    print("  • Straddle works best on HIGH-CONVICTION events (earnings, budget, RBI)")
    print("  • Momentum works best on TRENDING days; skips 'gap-and-fade' traps")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest pre-event straddle and first-candle momentum on NIFTY50 stocks."
    )
    p.add_argument("--stock",    nargs="+", metavar="SYMBOL",
                   help="One or more stock symbols (e.g. INDIGO RELIANCE). Default: all 50.")
    p.add_argument("--strategy", choices=["straddle", "momentum", "both"], default="both",
                   help="Which strategy to run (default: both).")
    p.add_argument("--gap-min",  type=float, default=3.0, metavar="PCT",
                   help="Minimum gap %% to qualify as an event (default: 3.0).")
    p.add_argument("--days",     type=int,   default=730, metavar="DAYS",
                   help="Lookback period in calendar days (default: 730 ≈ 2 years).")
    p.add_argument("--lot-size", type=int,   default=None, metavar="N",
                   help="Override lot size for all symbols (default: use LOT_SIZE_MAP).")
    p.add_argument("--top",      type=int,   default=0, metavar="N",
                   help="Show only top N stocks by P&L (default: all).")
    p.add_argument("--show-trades", action="store_true",
                   help="Print every individual trade detail.")
    p.add_argument("--list-stocks", action="store_true",
                   help="List all available stock symbols and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_stocks:
        print("\nAvailable NIFTY50 stocks:")
        for sym, tick in sorted(NIFTY50.items()):
            ls = LOT_SIZE_MAP.get(sym, 500)
            print(f"  {sym:<14}  ({tick})  lot={ls}")
        return

    # Determine stock universe
    if args.stock:
        symbols = {s.upper(): NIFTY50[s.upper()] for s in args.stock
                   if s.upper() in NIFTY50}
        missing = [s for s in args.stock if s.upper() not in NIFTY50]
        if missing:
            print(f"⚠  Unknown symbols (ignored): {', '.join(missing)}")
    else:
        symbols = NIFTY50

    run_straddle  = args.strategy in ("straddle", "both")
    run_momentum  = args.strategy in ("momentum", "both")

    print(f"\n🔍  Backtest: {len(symbols)} stocks  |  gap≥{args.gap_min}%  |  "
          f"{args.days}d lookback  |  strategies: {args.strategy}")
    print("─" * 70)

    straddle_results = []
    momentum_results = []

    for sym, ticker in symbols.items():
        print(f"  [{sym}] downloading …", end="\r")

        daily = load_daily(ticker, args.days)
        if daily is None or len(daily) < 40:
            print(f"  [{sym}] ⚠  insufficient daily data — skipped")
            continue

        hourly = None
        if run_momentum:
            hourly = load_hourly(ticker)

        if args.show_trades:
            print(f"\n  ── {sym} ──")

        if run_straddle:
            earnings_dates = []   # yfinance coverage for NSE is sparse; use gap proxy
            sr = backtest_straddle(
                symbol         = sym,
                daily          = daily,
                gap_min_pct    = args.gap_min,
                lot_size       = args.lot_size,
                show_trades    = args.show_trades,
                earnings_dates = earnings_dates,
            )
            straddle_results.append(sr)

        if run_momentum:
            mr = backtest_momentum(
                symbol      = sym,
                daily       = daily,
                hourly      = hourly,
                gap_min_pct = args.gap_min,
                lot_size    = args.lot_size,
                show_trades = args.show_trades,
            )
            momentum_results.append(mr)

        # Brief per-stock summary line
        parts = []
        if run_straddle and straddle_results:
            sr = straddle_results[-1]
            n = sr.get("events", 0)
            tp = sr.get("total_pnl_eod", 0)
            wr = sr.get("win_rate_eod", 0)
            parts.append(f"S: {n}ev {wr:.0f}%wr ₹{tp:,.0f}")
        if run_momentum and momentum_results:
            mr = momentum_results[-1]
            n = mr.get("events_traded", 0)
            tp = mr.get("total_pnl_2h", 0)
            wr = mr.get("win_rate_2h", 0)
            parts.append(f"M: {n}tr {wr:.0f}%wr ₹{tp:,.0f}")
        print(f"  [{sym}] {'  |  '.join(parts) if parts else 'no events'}")

    # Print summaries
    if run_straddle and straddle_results:
        print_straddle_summary(straddle_results, args.top)

    if run_momentum and momentum_results:
        print_momentum_summary(momentum_results, args.top)

    if run_straddle and run_momentum and straddle_results and momentum_results:
        print_comparison(straddle_results, momentum_results)

    print("\n" + "═" * 80)
    print("  NOTES ON METHODOLOGY")
    print("═" * 80)
    print("  • Option premiums estimated via Black-Scholes (not real market prices)")
    print("  • Pre-event IV = realized_vol × 1.40 (conservative; real IV can be 2-3×)")
    print("  • Post-event IV = realized_vol × 0.90 (IV crush modelled)")
    print("  • First-candle exit spots: actual hourly candles where available,")
    print("    daily OHLC proxy otherwise (marked with [~])")
    print("  • Straddle DTE = 1 (assumes weekly expiry next day after results)")
    print("  • Lot sizes from LOT_SIZE_MAP — override with --lot-size N")
    print("  • Results are directional estimates. Real slippage/impact not modelled.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()
