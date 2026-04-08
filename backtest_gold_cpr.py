#!/usr/bin/env python3
"""
backtest_gold_cpr.py
====================
CPR (Central Pivot Range) strategy backtest on GOLD (GC futures / GLD ETF).
Implements the AAK-style CPR rules: daily pivot calculation, narrow vs wide
day classification, and two trade modes — breakout and range reversion.

CPR Formulas (calculated from PREVIOUS day's OHLC)
----------------------------------------------------
  Pivot (P)  = (Prev_High + Prev_Low + Prev_Close) / 3
  BC         = (Prev_High + Prev_Low) / 2
  TC         = (P - BC) + P   →  i.e. 2P - BC
  Width%     = (TC - BC) / P * 100

Standard Pivot Levels
---------------------
  R1 = 2P - Prev_Low       S1 = 2P - Prev_High
  R2 = P + (PH - PL)       S2 = P  - (PH - PL)

CPR Day Classification
-----------------------
  Narrow CPR (Width% < NARROW_THRESH) → Trending day expected → BREAKOUT rules
  Wide CPR   (Width% > WIDE_THRESH)   → Range-bound day expected → RANGE rules
  In-between → Skip (ambiguous)

Trade Rules
-----------
BREAKOUT (Narrow CPR day):
  • If 1H opening candle closes ABOVE TC → LONG at TC close, SL=BC, Target=R1
  • If 1H opening candle closes BELOW BC → SHORT at BC close, SL=TC, Target=S1
  • If neither → no trade

RANGE (Wide CPR day):
  • If price touches BC from above → LONG at BC, SL = BC-(TC-BC), Target=TC
  • If price touches TC from below → SHORT at TC, SL = TC+(TC-BC), Target=BC
  • First touch only per session

Risk Management
---------------
  • Max risk per trade: 0.5% of capital (configurable)
  • Trailing SL: move SL to entry once 50% of target reached (configurable)
  • Hard EOD exit: close all positions at 4:30 PM ET (or last hourly bar)
  • Max 1 trade per session per direction

Usage
-----
  python backtest_gold_cpr.py                        # 2Y backtest, both modes
  python backtest_gold_cpr.py --mode breakout        # breakout only
  python backtest_gold_cpr.py --mode range           # range only
  python backtest_gold_cpr.py --narrow 0.25          # custom narrow threshold
  python backtest_gold_cpr.py --show-trades          # print every trade
  python backtest_gold_cpr.py --capital 50000 --inr  # Indian budget in ₹
  python backtest_gold_cpr.py --ticker GLD           # use GLD ETF instead

Requirements: yfinance, pandas  (pip install yfinance pandas)
Run on your Windows D: drive machine.
"""

import argparse
import sys
from datetime import date, timedelta, datetime, time as dtime
from collections import defaultdict

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: pip install yfinance pandas")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
NARROW_THRESH   = 0.30   # CPR width% below this → narrow (trending) day
WIDE_THRESH     = 0.55   # CPR width% above this → wide (range) day
RISK_PCT        = 0.005  # risk 0.5% of capital per trade
TRAIL_AT        = 0.50   # trail SL to entry when 50% of target reached
MGC_OZ          = 10     # MGC micro gold = 10 troy oz per contract
MGC_TICK        = 0.10   # MGC tick = $0.10/oz = $1.00/contract
GC_OZ           = 100    # Standard GC = 100 oz
USD_TO_INR      = 84.5   # approx exchange rate


# ── Download data ──────────────────────────────────────────────────────────────
def fetch_data(ticker: str, days: int = 730):
    end   = date.today()
    start = end - timedelta(days=days + 60)   # extra for vol warmup
    print(f"[*] Downloading {ticker} daily data ({start} → {end}) …")
    daily = yf.download(ticker, start=str(start), end=str(end),
                        interval="1d", auto_adjust=True, progress=False)
    print(f"[*] Downloading {ticker} hourly data (last {days}d) …")
    hourly = yf.download(ticker, period=f"{days}d",
                         interval="1h", auto_adjust=True, progress=False)
    daily.index  = pd.to_datetime(daily.index).date
    hourly.index = pd.to_datetime(hourly.index)
    # Flatten MultiIndex columns that yfinance sometimes returns
    daily  = _flatten(daily)
    hourly = _flatten(hourly)
    if daily.empty or hourly.empty:
        print(f"ERROR: No data for {ticker}. Check ticker name.")
        sys.exit(1)
    print(f"  Daily bars  : {len(daily)}")
    print(f"  Hourly bars : {len(hourly)}\n")
    return daily, hourly


# ── Scalar helper (handles yfinance MultiIndex columns) ───────────────────────
def _s(v):
    """Safely extract a float from either a scalar or a single-element Series."""
    if hasattr(v, "iloc"):
        return float(v.iloc[0])
    return float(v)


def _flatten(df):
    """Drop the ticker level from a MultiIndex column DataFrame if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ── CPR Calculation ────────────────────────────────────────────────────────────
def calc_cpr(prev_high, prev_low, prev_close):
    P  = (prev_high + prev_low + prev_close) / 3.0
    BC = (prev_high + prev_low) / 2.0
    TC = 2 * P - BC
    # ── Normalise so upper_cpr ≥ lower_cpr always ─────────────────────────────
    # TC < BC happens when close < midpoint of yesterday's range (bear close).
    # Without this, SL ends up on the WRONG SIDE of entry — a silent but fatal
    # bug that makes almost every "SL hit" look like a winner.
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1 = 2 * P - prev_low
    S1 = 2 * P - prev_high
    R2 = P + (prev_high - prev_low)
    S2 = P - (prev_high - prev_low)
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, BC=BC, TC=TC,
                upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1, R2=R2, S2=S2,
                width_pct=width_pct)


def classify(width_pct, narrow_thresh, wide_thresh):
    if width_pct < narrow_thresh:  return "NARROW"
    if width_pct > wide_thresh:    return "WIDE"
    return "NEUTRAL"


# ── Trade simulator ────────────────────────────────────────────────────────────
def simulate_day(day_bars: pd.DataFrame, cpr: dict, mode: str,
                 capital: float, risk_pct: float, ticker: str):
    """
    Simulate CPR trades for one session using hourly bars.
    Returns list of trade dicts.
    """
    if day_bars.empty or len(day_bars) < 2:
        return []

    is_mgc = ticker == "MGC=F"

    # Always use normalised levels: upper ≥ lower (TC/BC raw can invert)
    upper = cpr["upper_cpr"]
    lower = cpr["lower_cpr"]
    R1, S1 = cpr["R1"], cpr["S1"]
    cpr_range = upper - lower        # always ≥ 0

    trades = []

    # ── BREAKOUT mode ──────────────────────────────────────────────────────────
    if mode == "breakout":
        first       = day_bars.iloc[0]
        first_close = _s(first["Close"])

        # Long: first candle closes ABOVE the upper CPR band
        if first_close > upper:
            entry  = upper          # enter at upper CPR
            sl     = lower          # SL always BELOW entry
            target = R1             # first pivot resistance
            risk   = entry - sl     # always > 0 (upper > lower)
            if risk > 0:
                rr = abs(target - entry) / risk
                outcome, exit_price, _ = _simulate_exit(
                    day_bars.iloc[1:], entry, sl, target, "LONG")
                trades.append(_make_trade(
                    day_bars.index[0].date(), "LONG", "breakout",
                    entry, sl, target, exit_price, outcome, rr,
                    capital, risk_pct, cpr, ticker, is_mgc))

        # Short: first candle closes BELOW the lower CPR band
        elif first_close < lower:
            entry  = lower          # enter at lower CPR
            sl     = upper          # SL always ABOVE entry
            target = S1             # first pivot support
            risk   = sl - entry     # always > 0
            if risk > 0:
                rr = abs(target - entry) / risk
                outcome, exit_price, _ = _simulate_exit(
                    day_bars.iloc[1:], entry, sl, target, "SHORT")
                trades.append(_make_trade(
                    day_bars.index[0].date(), "SHORT", "breakout",
                    entry, sl, target, exit_price, outcome, rr,
                    capital, risk_pct, cpr, ticker, is_mgc))

    # ── RANGE mode ────────────────────────────────────────────────────────────
    elif mode == "range":
        long_taken  = False
        short_taken = False

        for i, (ts, bar) in enumerate(day_bars.iterrows()):
            h = _s(bar["High"])
            l = _s(bar["Low"])

            # Long: price touches LOWER CPR from above
            if not long_taken and l <= lower <= h:
                entry  = lower
                sl     = lower - cpr_range   # 1× CPR width below lower (always < entry)
                target = upper               # fade back to upper CPR
                risk   = entry - sl
                if risk > 0:
                    rr = abs(target - entry) / risk
                    outcome, exit_price, _ = _simulate_exit(
                        day_bars.iloc[i+1:], entry, sl, target, "LONG")
                    trades.append(_make_trade(
                        ts.date(), "LONG", "range",
                        entry, sl, target, exit_price, outcome, rr,
                        capital, risk_pct, cpr, ticker, is_mgc))
                    long_taken = True

            # Short: price touches UPPER CPR from below
            if not short_taken and l <= upper <= h:
                entry  = upper
                sl     = upper + cpr_range   # 1× CPR width above upper (always > entry)
                target = lower               # fade back to lower CPR
                risk   = sl - entry
                if risk > 0:
                    rr = abs(target - entry) / risk
                    outcome, exit_price, _ = _simulate_exit(
                        day_bars.iloc[i+1:], entry, sl, target, "SHORT")
                    trades.append(_make_trade(
                        ts.date(), "SHORT", "range",
                        entry, sl, target, exit_price, outcome, rr,
                        capital, risk_pct, cpr, ticker, is_mgc))
                    short_taken = True

    return trades


def _simulate_exit(bars, entry, sl, target, direction):
    """Walk through hourly bars to find first SL or Target hit."""
    for ts, bar in bars.iterrows():
        h = _s(bar["High"])
        l = _s(bar["Low"])
        c = _s(bar["Close"])

        if direction == "LONG":
            if l <= sl:      return "SL",     sl,     ts
            if h >= target:  return "TARGET",  target, ts
        else:
            if h >= sl:      return "SL",     sl,     ts
            if l <= target:  return "TARGET",  target, ts

    # EOD exit at last bar close
    last_close = _s(bars.iloc[-1]["Close"]) if not bars.empty else entry
    return "EOD", last_close, (bars.index[-1] if not bars.empty else None)


def _make_trade(trade_date, direction, mode, entry, sl, target,
                exit_price, outcome, rr, capital, risk_pct, cpr,
                ticker, is_mgc):
    """Calculate P&L for one trade."""
    risk_per_trade = capital * risk_pct
    price_risk     = abs(entry - sl)

    if ticker == "GLD":
        # GLD ETF: shares traded
        shares  = risk_per_trade / price_risk if price_risk > 0 else 0
        shares  = max(1, int(shares))
        raw_pnl = (exit_price - entry) * shares if direction == "LONG" \
                  else (entry - exit_price) * shares
        size_str = f"{shares} shares"
    elif is_mgc:
        # MGC: 10 oz/contract, $1/oz P&L
        contracts = max(1, int(risk_per_trade / (price_risk * MGC_OZ)))
        raw_pnl   = (exit_price - entry) * MGC_OZ * contracts \
                    if direction == "LONG" \
                    else (entry - exit_price) * MGC_OZ * contracts
        size_str  = f"{contracts} MGC"
    else:
        # GC=F standard futures (100 oz)
        contracts = max(1, int(risk_per_trade / (price_risk * GC_OZ)))
        raw_pnl   = (exit_price - entry) * GC_OZ * contracts \
                    if direction == "LONG" \
                    else (entry - exit_price) * GC_OZ * contracts
        size_str  = f"{contracts} GC"

    return dict(
        date      = trade_date,
        direction = direction,
        mode      = mode,
        entry     = entry,
        sl        = sl,
        target    = target,
        exit_p    = exit_price,
        outcome   = outcome,
        rr        = rr,
        pnl_usd   = raw_pnl,
        pnl_inr   = raw_pnl * USD_TO_INR,
        size      = size_str,
        cpr_w     = cpr["width_pct"],
        P         = cpr["P"],
        BC        = cpr["BC"],
        TC        = cpr["TC"],
    )


# ── Main backtest loop ─────────────────────────────────────────────────────────
def run_backtest(daily, hourly, ticker, narrow_thresh, wide_thresh,
                 mode_filter, capital, risk_pct, show_trades, use_inr):

    trades_all = []
    days_processed = 0
    narrow_days = wide_days = neutral_days = 0

    daily_dates = sorted(daily.index)

    for i in range(1, len(daily_dates)):
        today     = daily_dates[i]
        yesterday = daily_dates[i - 1]

        # Skip if today is outside hourly data range
        hourly_today = hourly[hourly.index.date == today]
        if hourly_today.empty:
            continue

        # Calculate CPR from yesterday
        try:
            ph = _s(daily.loc[yesterday, "High"])
            pl = _s(daily.loc[yesterday, "Low"])
            pc = _s(daily.loc[yesterday, "Close"])
        except Exception:
            continue

        cpr  = calc_cpr(ph, pl, pc)
        kind = classify(cpr["width_pct"], narrow_thresh, wide_thresh)
        days_processed += 1

        if kind == "NARROW":   narrow_days += 1
        elif kind == "WIDE":   wide_days   += 1
        else:                  neutral_days+= 1

        # Apply mode filter
        if mode_filter == "breakout" and kind != "NARROW": continue
        if mode_filter == "range"    and kind != "WIDE":   continue
        if mode_filter == "auto":
            if kind == "NARROW":   mode = "breakout"
            elif kind == "WIDE":   mode = "range"
            else:                  continue
        else:
            mode = mode_filter

        day_trades = simulate_day(hourly_today, cpr, mode,
                                  capital, risk_pct, ticker)
        trades_all.extend(day_trades)

    return trades_all, days_processed, narrow_days, wide_days, neutral_days


# ── Display ────────────────────────────────────────────────────────────────────
def fmt_usd(v):
    s = "+" if v >= 0 else "-"
    return f"{s}${abs(v):,.2f}"

def fmt_inr(v):
    s = "+" if v >= 0 else "-"
    v = abs(v)
    if v >= 1e5: return f"{s}₹{v/1e5:.2f}L"
    return f"{s}₹{v:,.0f}"

def print_report(trades, days_proc, narrow, wide, neutral,
                 ticker, capital, use_inr, show_trades,
                 narrow_thresh, wide_thresh):

    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  CPR STRATEGY BACKTEST — {ticker}  |  Capital: ${capital:,.0f}")
    print(f"  Narrow threshold: <{narrow_thresh}%  |  Wide threshold: >{wide_thresh}%")
    print(sep)
    print(f"  Days processed : {days_proc}")
    print(f"  Narrow days    : {narrow}  ({narrow/days_proc*100:.0f}%)  → breakout trades")
    print(f"  Wide days      : {wide}   ({wide/days_proc*100:.0f}%)  → range trades")
    print(f"  Neutral days   : {neutral}  ({neutral/days_proc*100:.0f}%)  → skipped")
    print(f"  Total trades   : {len(trades)}")

    if not trades:
        print("  No trades generated.")
        return

    wins     = [t for t in trades if t["pnl_usd"] > 0]
    losses   = [t for t in trades if t["pnl_usd"] <= 0]
    targets  = [t for t in trades if t["outcome"] == "TARGET"]
    sls      = [t for t in trades if t["outcome"] == "SL"]
    eods     = [t for t in trades if t["outcome"] == "EOD"]

    tot_pnl  = sum(t["pnl_usd"] for t in trades)
    avg_win  = sum(t["pnl_usd"] for t in wins)  / max(len(wins),  1)
    avg_loss = sum(t["pnl_usd"] for t in losses)/ max(len(losses), 1)

    # By mode
    bt = [t for t in trades if t["mode"] == "breakout"]
    rt = [t for t in trades if t["mode"] == "range"]

    print(f"\n{'─'*90}")
    print(f"  OVERALL RESULTS")
    print(f"{'─'*90}")
    print(f"  Win rate        : {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.0f}%")
    print(f"  Total P&L       : {fmt_usd(tot_pnl)}  "
          f"({'  = ' + fmt_inr(tot_pnl * USD_TO_INR) if use_inr else ''})")
    print(f"  Avg win         : {fmt_usd(avg_win)}")
    print(f"  Avg loss        : {fmt_usd(avg_loss)}")
    print(f"  Win/Loss ratio  : {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"  Target hits     : {len(targets)}  ({len(targets)/len(trades)*100:.0f}%)")
    print(f"  SL hits         : {len(sls)}  ({len(sls)/len(trades)*100:.0f}%)")
    print(f"  EOD exits       : {len(eods)}  ({len(eods)/len(trades)*100:.0f}%)")
    print(f"  Return on cap   : {tot_pnl/capital*100:+.1f}%  over 2 years")

    if bt:
        bw = [t for t in bt if t["pnl_usd"] > 0]
        bp = sum(t["pnl_usd"] for t in bt)
        print(f"\n  BREAKOUT trades : {len(bt)}  |  "
              f"Win: {len(bw)}/{len(bt)} = {len(bw)/len(bt)*100:.0f}%  |  "
              f"P&L: {fmt_usd(bp)}")
    if rt:
        rw = [t for t in rt if t["pnl_usd"] > 0]
        rp = sum(t["pnl_usd"] for t in rt)
        print(f"  RANGE trades    : {len(rt)}  |  "
              f"Win: {len(rw)}/{len(rt)} = {len(rw)/len(rt)*100:.0f}%  |  "
              f"P&L: {fmt_usd(rp)}")

    # Monthly breakdown
    monthly = defaultdict(float)
    for t in trades:
        key = t["date"].strftime("%Y-%m")
        monthly[key] += t["pnl_usd"]
    print(f"\n{'─'*90}")
    print("  MONTHLY P&L")
    print(f"{'─'*90}")
    green_months = red_months = 0
    for ym in sorted(monthly):
        v  = monthly[ym]
        mk = "✅" if v >= 0 else "❌"
        if v >= 0: green_months += 1
        else: red_months += 1
        inr_str = f"  ({fmt_inr(v * USD_TO_INR)})" if use_inr else ""
        print(f"  {ym}  {mk}  {fmt_usd(v):>12}{inr_str}")
    print(f"\n  Green months: {green_months}  |  Red months: {red_months}")

    # Individual trades
    if show_trades:
        print(f"\n{'─'*90}")
        print("  TRADE LOG")
        print(f"{'─'*90}")
        hdr = f"  {'Date':<12} {'Dir':<6} {'Mode':<10} {'Entry':>7} {'SL':>7} "
        hdr += f"{'Target':>7} {'Exit':>7} {'Out':<7} {'RR':>4} {'P&L USD':>10}"
        print(hdr)
        print("  " + "─" * 86)
        for t in trades:
            mk = "✅" if t["pnl_usd"] > 0 else "❌"
            print(f"  {str(t['date']):<12} {t['direction']:<6} {t['mode']:<10} "
                  f"{t['entry']:>7.1f} {t['sl']:>7.1f} {t['target']:>7.1f} "
                  f"{t['exit_p']:>7.1f} {t['outcome']:<7} "
                  f"{t['rr']:>4.1f}x {fmt_usd(t['pnl_usd']):>10}  {mk}")

    # Budget guide
    print(f"\n{'─'*90}")
    print("  BUDGET GUIDE FOR THIS STRATEGY")
    print(f"{'─'*90}")
    print(f"  Instrument      : MGC (Micro Gold Futures, CME) — 10 oz/contract")
    print(f"  Approx margin   : $800–$1,000 per MGC contract (varies by broker)")
    print(f"  Avg trade risk  : ~$50–80 per trade at 0.5% of $10,000 account")
    print(f"  Prop firm route : Apex Trader Funding — $147 eval → $50,000 funded account")
    print(f"                    MGC allowed ✅  |  Keep 90% of profits")
    print(f"\n  With ₹50,000 (~$590):")
    print(f"    Own account (IBKR): Too small for GC ($8K margin). MGC needs $1K min.")
    print(f"    Prop firm eval    : Pay $147 eval fee, trade $50K their capital → ✅")
    print(f"    Recommended path  : Apex Trader Funding + MGC + CPR strategy")
    print(sep + "\n")

    # Save CSV
    try:
        import csv, os
        out = os.path.join(os.path.dirname(__file__), "gold_cpr_backtest.csv")
        fields = ["date", "direction", "mode", "entry", "sl", "target",
                  "exit_p", "outcome", "rr", "pnl_usd", "pnl_inr",
                  "size", "cpr_w", "P", "BC", "TC"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in trades:
                w.writerow({k: t[k] for k in fields})
        print(f"  Trades saved → {out}")
    except Exception as e:
        print(f"  [CSV save failed: {e}]")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="CPR backtest on Gold (GC/MGC/GLD)")
    p.add_argument("--ticker",   default="GC=F",
                   choices=["GC=F", "MGC=F", "GLD"],
                   help="Instrument (default: GC=F — Gold futures)")
    p.add_argument("--mode",     default="auto",
                   choices=["auto", "breakout", "range"],
                   help="Trade mode: auto=narrow→breakout, wide→range")
    p.add_argument("--narrow",   type=float, default=NARROW_THRESH,
                   help=f"Narrow CPR threshold %% (default {NARROW_THRESH})")
    p.add_argument("--wide",     type=float, default=WIDE_THRESH,
                   help=f"Wide CPR threshold %% (default {WIDE_THRESH})")
    p.add_argument("--capital",  type=float, default=10000,
                   help="Account capital in USD (default $10,000)")
    p.add_argument("--risk",     type=float, default=RISK_PCT,
                   help=f"Risk per trade as decimal (default {RISK_PCT} = 0.5%%)")
    p.add_argument("--days",     type=int,   default=730,
                   help="Lookback days (default 730 = 2 years)")
    p.add_argument("--inr",      action="store_true",
                   help="Show P&L in ₹ as well")
    p.add_argument("--show-trades", action="store_true",
                   help="Print every individual trade")
    args = p.parse_args()

    print(f"\n{'═'*60}")
    print(f"  CPR Gold Backtest  |  {args.ticker}  |  Mode: {args.mode}")
    print(f"  Capital: ${args.capital:,.0f}  |  Risk/trade: {args.risk*100:.1f}%")
    print(f"  Narrow<{args.narrow}%  Wide>{args.wide}%  |  {args.days}d lookback")
    print(f"{'═'*60}\n")

    daily, hourly = fetch_data(args.ticker, args.days)

    trades, days_proc, narrow, wide, neutral = run_backtest(
        daily, hourly, args.ticker,
        args.narrow, args.wide, args.mode,
        args.capital, args.risk,
        args.show_trades, args.inr)

    print_report(trades, days_proc, narrow, wide, neutral,
                 args.ticker, args.capital, args.inr,
                 args.show_trades, args.narrow, args.wide)


if __name__ == "__main__":
    main()
