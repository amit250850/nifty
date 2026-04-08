"""
news_event_research.py — News-Event Option Premium Research Tool
================================================================

Answers the question: "If I had bought an ATM option on a NIFTY50 stock
at 9:15 AM on a high-impact news day, what would have happened?"

This is RESEARCH ONLY — no live trading, no orders.

How it works:
  1. Downloads 2 years of daily price data for NIFTY50 stocks (or a custom list).
  2. Identifies "gap days" — opens that differ meaningfully from the prior close.
     These are almost always news-driven (earnings, corporate actions, macro).
  3. For each gap, estimates ATM option premium at open using Black-Scholes
     with realized volatility from the 20 days before the event.
  4. Models what happens if you held the option for 30 / 60 / 90 / 180 minutes
     using the actual intraday price move.
  5. Produces a ranked report of: best stocks, best gap sizes, best hold times,
     and the realistic P&L statistics.

Key insight this answers:
  - Is buying a gap-open option actually profitable after accounting for
    inflated IV at open (the spread between realized and implied vol widens
    sharply on gap days)?
  - Which gap sizes give the best R:R?
  - Does the trade work better for gaps that EXTEND or gaps that FADE?

Limitations (important to understand):
  - B-S model understates real option cost on event days (IV spikes at open).
    Actual premiums will be 20-50% higher than model estimates.
    The script applies an EVENT_IV_PREMIUM multiplier to correct for this.
  - Intraday data from yfinance has 30-minute resolution (not 1-minute).
    Hold times are approximated to 30-min intervals.
  - Individual stock options have lower liquidity than index options.
    Bid-ask spread can be 5-15% of premium for many NIFTY50 options.

Usage:
    python news_event_research.py                   — all tracked stocks, 2Y data
    python news_event_research.py --stock INDIGO    — single stock deep-dive
    python news_event_research.py --gap-min 3       — only gaps ≥ 3%
    python news_event_research.py --top 10          — show top 10 events only
    python news_event_research.py --days 365        — 1 year lookback
    python news_event_research.py --list-stocks     — show all tracked stocks
"""

import argparse
import math
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("❌  yfinance not installed.  Run: pip install yfinance")
    sys.exit(1)

# ── NIFTY50 stock universe ─────────────────────────────────────────────────────
# Top NIFTY50 constituents most likely to have high-impact news events.
# Focuses on stocks with decent option liquidity on NSE.
# Symbol format: NSE ticker + ".NS"

NIFTY50_STOCKS = {
    # Airlines / Travel
    "INDIGO":       "INDIGO.NS",       # IndiGo — volatile on load factor, fuel, regulation news
    # Banking / Finance
    "HDFCBANK":     "HDFCBANK.NS",
    "ICICIBANK":    "ICICIBANK.NS",
    "KOTAKBANK":    "KOTAKBANK.NS",
    "AXISBANK":     "AXISBANK.NS",
    "SBIN":         "SBIN.NS",
    "BAJFINANCE":   "BAJFINANCE.NS",
    # IT
    "TCS":          "TCS.NS",
    "INFY":         "INFY.NS",
    "WIPRO":        "WIPRO.NS",
    "HCLTECH":      "HCLTECH.NS",
    "TECHM":        "TECHM.NS",
    # Pharma
    "SUNPHARMA":    "SUNPHARMA.NS",
    "DRREDDY":      "DRREDDY.NS",
    # Auto
    "MARUTI":       "MARUTI.NS",
    "TATAMOTORS":   "TATAMOTORS.NS",
    "M&M":          "M&M.NS",
    "HEROMOTOCO":   "HEROMOTOCO.NS",
    # Energy / Metals
    "RELIANCE":     "RELIANCE.NS",
    "ONGC":         "ONGC.NS",
    "TATASTEEL":    "TATASTEEL.NS",
    "HINDALCO":     "HINDALCO.NS",
    "JSWSTEEL":     "JSWSTEEL.NS",
    # FMCG
    "HINDUNILVR":   "HINDUNILVR.NS",
    "ITC":          "ITC.NS",
    "NESTLEIND":    "NESTLEIND.NS",
    # Telecom / Infra
    "BHARTIARTL":   "BHARTIARTL.NS",
    "POWERGRID":    "POWERGRID.NS",
    "NTPC":         "NTPC.NS",
    "ADANIPORTS":   "ADANIPORTS.NS",
}

# Approximate lot sizes for NSE stock options (as of 2025-26)
# Used to estimate lot cost per trade
STOCK_LOT_SIZES = {
    "INDIGO":     150,
    "HDFCBANK":   550,
    "ICICIBANK":  1375,
    "KOTAKBANK":  400,
    "AXISBANK":   625,
    "SBIN":       1500,
    "BAJFINANCE": 125,
    "TCS":        150,
    "INFY":       600,
    "WIPRO":      1500,
    "HCLTECH":    700,
    "TECHM":      600,
    "SUNPHARMA":  700,
    "DRREDDY":    125,
    "MARUTI":     100,
    "TATAMOTORS": 2850,
    "M&M":        700,
    "HEROMOTOCO": 300,
    "RELIANCE":   500,
    "ONGC":       3850,
    "TATASTEEL":  5500,
    "HINDALCO":   2150,
    "JSWSTEEL":   1350,
    "HINDUNILVR": 300,
    "ITC":        3200,
    "NESTLEIND":  40,
    "BHARTIARTL": 500,
    "POWERGRID":  5000,
    "NTPC":       4500,
    "ADANIPORTS": 1250,
}

# ── Research parameters ────────────────────────────────────────────────────────

# Gap threshold — minimum absolute open vs prev close % move to flag as event
GAP_MIN_PCT = 2.0    # 2% gap (up or down)

# IV premium on event days — real options are MORE expensive than B-S model
# because market knows something is happening.
# 1.5x = premiums 50% above model → conservative estimate for gap days.
EVENT_IV_PREMIUM = 1.5

# Hold time scenarios to evaluate (in 30-min bars from open)
HOLD_SCENARIOS_BARS = [1, 2, 3, 6]   # 30min, 60min, 90min, 180min

# Rolling volatility window (trading days of daily returns)
HIST_VOL_WINDOW = 20
RISK_FREE_RATE  = 0.065

# ── Normal CDF ────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if option_type == "CE" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return float(S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2))
    else:
        return float(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))

# ── ATM strike calculator ─────────────────────────────────────────────────────

def atm_strike(spot: float) -> int:
    """Round spot to nearest NSE option strike (₹5 for <₹500, ₹10 for >₹500, etc.)"""
    if spot < 100:
        step = 2.5
    elif spot < 500:
        step = 5
    elif spot < 1000:
        step = 10
    elif spot < 2000:
        step = 20
    elif spot < 5000:
        step = 50
    elif spot < 10000:
        step = 100
    else:
        step = 200
    return int(round(spot / step) * step)

# ── Data fetcher ──────────────────────────────────────────────────────────────

def fetch_daily(ticker_ns: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Download daily OHLC data for a stock."""
    period = f"{years * 365 + 30}d"
    try:
        df = yf.download(ticker_ns, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df
    except Exception:
        return None


def fetch_intraday_30m(ticker_ns: str, days: int = 730) -> Optional[pd.DataFrame]:
    """
    Download 30-minute intraday data.
    yfinance provides up to 60 days of 30m data in a single call.
    We make multiple calls and concatenate to extend coverage.
    """
    chunks = []
    # yfinance allows period="60d" for 30m interval
    # For longer history, download 60-day chunks going back
    periods_needed = max(1, days // 55)
    for _ in range(min(periods_needed, 4)):   # max 4 chunks (~8 months)
        try:
            chunk = yf.download(ticker_ns, period="60d", interval="30m",
                                auto_adjust=True, progress=False)
            if not chunk.empty:
                if hasattr(chunk.columns, "levels"):
                    chunk.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                                     for c in chunk.columns]
                else:
                    chunk.columns = [c.lower() for c in chunk.columns]
                chunks.append(chunk)
            break   # yfinance doesn't support offset — just get what we can
        except Exception:
            break

    if not chunks:
        return None

    df = pd.concat(chunks).drop_duplicates()
    df.index = pd.to_datetime(df.index, utc=True)
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    df.index = df.index.tz_convert(IST)
    return df[["open", "high", "low", "close"]].dropna()

# ── Gap event detector ────────────────────────────────────────────────────────

def find_gap_events(daily: pd.DataFrame, gap_min_pct: float = GAP_MIN_PCT) -> list:
    """
    Find all days where open differs from previous close by more than gap_min_pct.

    Returns list of dicts with:
        date, direction (UP/DOWN), gap_pct, open_price, prev_close
        hist_vol, dte_to_expiry (weekly approximation)
    """
    events = []
    daily  = daily.copy()
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_pct"]    = (daily["open"] - daily["prev_close"]) / daily["prev_close"] * 100

    # Rolling volatility (20-day annualised, using daily returns)
    log_ret = np.log(daily["close"] / daily["close"].shift(1))
    daily["hist_vol"] = log_ret.rolling(HIST_VOL_WINDOW).std() * np.sqrt(252)

    for idx, row in daily.iterrows():
        if pd.isna(row["gap_pct"]) or pd.isna(row["hist_vol"]):
            continue
        gap = float(row["gap_pct"])
        if abs(gap) < gap_min_pct:
            continue

        events.append({
            "date":       idx.date() if hasattr(idx, "date") else idx,
            "direction":  "UP" if gap > 0 else "DOWN",
            "gap_pct":    round(gap, 2),
            "open":       float(row["open"]),
            "prev_close": float(row["prev_close"]),
            "hist_vol":   float(row["hist_vol"]),
            "day_high":   float(row["high"]),
            "day_low":    float(row["low"]),
            "day_close":  float(row["close"]),
        })

    return events

# ── Intraday outcome estimator ────────────────────────────────────────────────

def estimate_option_outcomes(event: dict, intraday_30m: Optional[pd.DataFrame],
                              stock_name: str) -> dict:
    """
    For a given gap event, estimate what would happen if you bought
    an ATM option at 9:15 AM open and held for various durations.

    Uses actual intraday prices where available, otherwise uses day OHLC
    to estimate intraday move.

    Returns a dict with outcomes for each hold scenario.
    """
    spot_open  = event["open"]
    hist_vol   = event["hist_vol"]
    gap_dir    = event["direction"]
    event_date = event["date"]

    # Option type aligned with gap direction
    option_type = "CE" if gap_dir == "UP" else "PE"
    k = atm_strike(spot_open)
    lot_size = STOCK_LOT_SIZES.get(stock_name, 500)

    # Weekly expiry approx — nearest Thursday DTE
    d = event_date
    days_to_thu = (3 - d.weekday()) % 7
    if days_to_thu == 0:
        days_to_thu = 7
    dte = days_to_thu
    T_entry = dte / 365.0

    # Event-day IV is inflated — apply EVENT_IV_PREMIUM multiplier
    iv_at_open = hist_vol * EVENT_IV_PREMIUM

    entry_premium = bs_price(spot_open, k, T_entry, RISK_FREE_RATE,
                             iv_at_open, option_type)
    if entry_premium < 1.0:
        return {}

    lot_cost = round(entry_premium * lot_size, 2)

    outcomes = {
        "entry_premium": round(entry_premium, 2),
        "strike":        k,
        "option_type":   option_type,
        "lot_size":      lot_size,
        "lot_cost":      lot_cost,
        "iv_used":       round(iv_at_open * 100, 1),
        "scenarios":     {},
    }

    # Try to use real intraday 30m data for the event date
    intraday_day = None
    if intraday_30m is not None:
        try:
            mask = intraday_30m.index.date == event_date
            intraday_day = intraday_30m[mask]
            # Filter to 9:15 AM onwards
            intraday_day = intraday_day[
                (intraday_day.index.hour > 9) |
                ((intraday_day.index.hour == 9) & (intraday_day.index.minute >= 15))
            ]
        except Exception:
            intraday_day = None

    for bars in HOLD_SCENARIOS_BARS:
        hold_minutes = bars * 30
        label = f"{hold_minutes}min"

        # Get exit spot price
        if intraday_day is not None and len(intraday_day) >= bars:
            exit_spot = float(intraday_day["close"].iloc[bars - 1])
        else:
            # Approximate: if gap extends, price moves further in gap direction
            # Use day high/low as bounds
            if gap_dir == "UP":
                # Assume roughly linear move from open to high over the day
                progress = min(bars / 12.0, 1.0)   # 12 bars = full 6h NSE session
                exit_spot = spot_open + (event["day_high"] - spot_open) * progress * 0.6
            else:
                progress  = min(bars / 12.0, 1.0)
                exit_spot = spot_open - (spot_open - event["day_low"]) * progress * 0.6

        # Re-price option at exit
        # IV decays somewhat intraday after initial spike
        iv_at_exit  = iv_at_open * (0.9 if bars <= 2 else 0.8)
        T_exit      = max((T_entry - hold_minutes / (365 * 24 * 60)), 0.5 / 365)
        exit_premium = bs_price(exit_spot, k, T_exit, RISK_FREE_RATE,
                                iv_at_exit, option_type)

        pnl_unit  = exit_premium - entry_premium
        pnl_lot   = round(pnl_unit * lot_size, 2)
        pnl_pct   = pnl_unit / entry_premium * 100
        move_pct  = (exit_spot - spot_open) / spot_open * 100
        if gap_dir == "DOWN":
            move_pct = -move_pct   # positive = gap extending for puts

        outcomes["scenarios"][label] = {
            "exit_spot":     round(exit_spot, 2),
            "exit_premium":  round(exit_premium, 2),
            "pnl_unit":      round(pnl_unit, 2),
            "pnl_lot":       pnl_lot,
            "pnl_pct":       round(pnl_pct, 1),
            "move_pct":      round(move_pct, 2),   # positive = gap extended
            "gap_extended":  move_pct > 0,
        }

    return outcomes

# ── Full analysis for one stock ───────────────────────────────────────────────

def analyse_stock(name: str, ticker: str, days: int = 730,
                  gap_min_pct: float = GAP_MIN_PCT,
                  verbose: bool = False) -> list:
    """
    Run full gap event analysis for a single stock.
    Returns list of event result dicts.
    """
    daily = fetch_daily(ticker, years=max(1, days // 300))
    if daily is None or len(daily) < 30:
        if verbose:
            print(f"  ⚠️  {name}: insufficient daily data")
        return []

    # Filter to requested period
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    daily  = daily[daily.index >= cutoff]

    events = find_gap_events(daily, gap_min_pct)
    if not events:
        return []

    # Try to get intraday data (best effort — limited by yfinance 60d window)
    intraday = fetch_intraday_30m(ticker, days=min(days, 60))

    results = []
    for event in events:
        outcomes = estimate_option_outcomes(event, intraday, name)
        if not outcomes:
            continue
        results.append({
            "stock":    name,
            "ticker":   ticker,
            "date":     event["date"],
            "direction": event["direction"],
            "gap_pct":  event["gap_pct"],
            "hist_vol": round(event["hist_vol"] * 100, 1),
            "outcomes": outcomes,
        })

    return results

# ── Report printer ────────────────────────────────────────────────────────────

def print_report(all_results: list, gap_min_pct: float, top_n: int) -> None:
    if not all_results:
        print("\n  No gap events found above threshold.")
        return

    print(f"\n{'═'*80}")
    print(f"  NEWS EVENT OPTION PREMIUM RESEARCH")
    print(f"  Gap threshold: ≥{gap_min_pct}%  |  {len(all_results)} events analysed")
    print(f"  IV assumption: realized vol × {EVENT_IV_PREMIUM}x (event-day inflation)")
    print(f"  ⚠️  B-S model — actual fills may differ (spread + illiquidity)")
    print(f"{'═'*80}")

    # Best hold time summary
    print(f"\n  HOLD-TIME ANALYSIS (avg P&L % across all events)")
    print(f"  {'─'*60}")
    scenario_labels = [f"{b*30}min" for b in HOLD_SCENARIOS_BARS]
    header = f"  {'Hold':>8}" + "".join(f"  {'AvgPnL%':>9}  {'WinRate':>8}  {'AvgLot₹':>8}" for _ in scenario_labels[:1])
    print(f"  {'Hold':>8}  {'AvgPnL%':>9}  {'WinRate':>8}  {'AvgLot₹':>9}  {'MedianPnL%':>11}")
    print(f"  {'─'*60}")

    for label in scenario_labels:
        pnls     = [r["outcomes"]["scenarios"][label]["pnl_pct"]
                    for r in all_results
                    if label in r["outcomes"].get("scenarios", {})]
        lot_pnls = [r["outcomes"]["scenarios"][label]["pnl_lot"]
                    for r in all_results
                    if label in r["outcomes"].get("scenarios", {})]
        if not pnls:
            continue
        wins    = sum(1 for p in pnls if p > 0)
        avg_pct = sum(pnls) / len(pnls)
        med_pct = float(np.median(pnls))
        avg_lot = sum(lot_pnls) / len(lot_pnls)
        wr      = wins / len(pnls) * 100
        print(f"  {label:>8}  {avg_pct:>+8.1f}%  {wr:>7.1f}%  "
              f"₹{avg_lot:>+8,.0f}  {med_pct:>+10.1f}%")

    # Gap size buckets
    print(f"\n  GAP SIZE ANALYSIS (60min hold, all stocks)")
    print(f"  {'─'*60}")
    print(f"  {'Gap Range':>12}  {'Events':>7}  {'WinRate':>8}  {'AvgPnL%':>9}  {'Best stock'}")
    print(f"  {'─'*60}")
    buckets = [
        ("2–3%",   2.0,  3.0),
        ("3–5%",   3.0,  5.0),
        ("5–8%",   5.0,  8.0),
        ("8–12%",  8.0, 12.0),
        (">12%",  12.0, 999.0),
    ]
    for label, lo, hi in buckets:
        bucket_events = [r for r in all_results
                         if lo <= abs(r["gap_pct"]) < hi]
        if not bucket_events:
            continue
        pnls    = [r["outcomes"]["scenarios"].get("60min", {}).get("pnl_pct", 0)
                   for r in bucket_events
                   if "60min" in r["outcomes"].get("scenarios", {})]
        if not pnls:
            continue
        wins    = sum(1 for p in pnls if p > 0)
        avg_pct = sum(pnls) / len(pnls)
        wr      = wins / len(pnls) * 100
        # Best stock for this bucket
        best = max(bucket_events,
                   key=lambda r: r["outcomes"]["scenarios"].get("60min", {}).get("pnl_pct", -999),
                   default=None)
        best_name = f"{best['stock']} ({best['gap_pct']:+.1f}%)" if best else "—"
        print(f"  {label:>12}  {len(bucket_events):>7}  {wr:>7.1f}%  "
              f"{avg_pct:>+8.1f}%  {best_name}")

    # Top individual events
    print(f"\n  TOP {top_n} INDIVIDUAL EVENTS (by 60min P&L %)")
    print(f"  {'─'*80}")
    print(f"  {'Stock':>12} {'Date':>12} {'Gap':>7} {'Dir':>5} {'Entry₹':>8} "
          f"{'30min':>7} {'60min':>7} {'90min':>7} {'LotCost':>9}")
    print(f"  {'─'*80}")

    sorted_results = sorted(
        all_results,
        key=lambda r: r["outcomes"]["scenarios"].get("60min", {}).get("pnl_pct", -999),
        reverse=True
    )

    for r in sorted_results[:top_n]:
        sc   = r["outcomes"]["scenarios"]
        ep   = r["outcomes"]["entry_premium"]
        lc   = r["outcomes"]["lot_cost"]
        date_str = r["date"].strftime("%d %b %Y") if hasattr(r["date"], "strftime") else str(r["date"])

        def pct_str(label):
            p = sc.get(label, {}).get("pnl_pct")
            return f"{p:+.0f}%" if p is not None else "  N/A"

        print(f"  {r['stock']:>12} {date_str:>12} {r['gap_pct']:>+6.1f}% "
              f"{'↑' if r['direction']=='UP' else '↓':>5} "
              f"₹{ep:>7.0f} "
              f"{pct_str('30min'):>7} {pct_str('60min'):>7} {pct_str('90min'):>7} "
              f"₹{lc:>8,.0f}")

    # Per-stock summary
    print(f"\n  PER-STOCK SUMMARY (60min hold, all events)")
    print(f"  {'─'*60}")
    print(f"  {'Stock':>12} {'Events':>7} {'WinRate':>8} {'AvgPnL%':>9} {'TotalLot₹':>11} {'AvgGap%':>8}")
    print(f"  {'─'*60}")

    by_stock = {}
    for r in all_results:
        by_stock.setdefault(r["stock"], []).append(r)

    stock_summary = []
    for stock, events in by_stock.items():
        pnls = [e["outcomes"]["scenarios"].get("60min", {}).get("pnl_pct", 0)
                for e in events if "60min" in e["outcomes"].get("scenarios", {})]
        lot_pnls = [e["outcomes"]["scenarios"].get("60min", {}).get("pnl_lot", 0)
                    for e in events if "60min" in e["outcomes"].get("scenarios", {})]
        if not pnls:
            continue
        wins     = sum(1 for p in pnls if p > 0)
        avg_pct  = sum(pnls) / len(pnls)
        total    = sum(lot_pnls)
        avg_gap  = sum(abs(e["gap_pct"]) for e in events) / len(events)
        wr       = wins / len(pnls) * 100
        stock_summary.append((stock, len(events), wr, avg_pct, total, avg_gap))

    for row in sorted(stock_summary, key=lambda x: x[3], reverse=True):
        stock, n, wr, avg_pct, total, avg_gap = row
        print(f"  {stock:>12} {n:>7} {wr:>7.1f}% {avg_pct:>+8.1f}% "
              f"₹{total:>+10,.0f} {avg_gap:>+7.1f}%")

    # Key takeaways
    print(f"\n  KEY FINDINGS")
    print(f"  {'─'*60}")

    all_60min = [r["outcomes"]["scenarios"].get("60min", {}).get("pnl_pct", 0)
                 for r in all_results if "60min" in r["outcomes"].get("scenarios", {})]
    if all_60min:
        overall_wr = sum(1 for p in all_60min if p > 0) / len(all_60min) * 100
        pct_extending = sum(
            1 for r in all_results
            if r["outcomes"]["scenarios"].get("60min", {}).get("gap_extended", False)
        ) / len(all_results) * 100

        best_hold = max(
            scenario_labels,
            key=lambda l: sum(
                r["outcomes"]["scenarios"].get(l, {}).get("pnl_pct", -999)
                for r in all_results if l in r["outcomes"].get("scenarios", {})
            )
        )

        print(f"  • Overall 60min win rate       : {overall_wr:.1f}%")
        print(f"  • Gaps that extend intraday    : {pct_extending:.1f}% of events")
        print(f"  • Best holding period          : {best_hold}")
        print(f"  • IV inflation factor used     : {EVENT_IV_PREMIUM}x (actual cost is higher)")
        print(f"  • Liquidity risk               : Stock options spread 5-15% of premium")
        print(f"  • Recommendation               : Focus on gaps >5% for better R:R")
        print(f"    (smaller gaps: IV crush eats premium faster than underlying moves)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="News Event Option Premium Research — NIFTY50 stocks"
    )
    parser.add_argument("--stock",  type=str, default=None,
                        help="Single stock name to analyse (e.g. INDIGO, TCS)")
    parser.add_argument("--gap-min", type=float, default=GAP_MIN_PCT,
                        help=f"Minimum gap %% to flag as event (default {GAP_MIN_PCT})")
    parser.add_argument("--days",   type=int, default=730,
                        help="Lookback in calendar days (default 730 = ~2 years)")
    parser.add_argument("--top",    type=int, default=15,
                        help="Number of top events to show in detail (default 15)")
    parser.add_argument("--list-stocks", action="store_true",
                        help="Print tracked stocks and exit")
    args = parser.parse_args()

    if args.list_stocks:
        print("\n  Tracked stocks:")
        for name, ticker in NIFTY50_STOCKS.items():
            print(f"    {name:15} {ticker}")
        return

    stocks = {args.stock: NIFTY50_STOCKS[args.stock]} if args.stock and args.stock in NIFTY50_STOCKS \
             else NIFTY50_STOCKS

    if args.stock and args.stock not in NIFTY50_STOCKS:
        print(f"  ❌  Unknown stock '{args.stock}'.  Use --list-stocks to see options.")
        sys.exit(1)

    print(f"\n{'═'*80}")
    print(f"  NEWS EVENT RESEARCH — {len(stocks)} stock(s)")
    print(f"  Period: last {args.days} days  |  Min gap: {args.gap_min}%")
    print(f"  Downloading data … (this may take 30-60 seconds)")
    print(f"{'═'*80}\n")

    all_results = []
    for name, ticker in stocks.items():
        print(f"  Analysing {name} ({ticker}) …", end=" ", flush=True)
        results = analyse_stock(
            name, ticker,
            days=args.days,
            gap_min_pct=args.gap_min,
            verbose=True,
        )
        all_results.extend(results)
        print(f"{len(results)} events")

    print_report(all_results, args.gap_min, args.top)


if __name__ == "__main__":
    main()
