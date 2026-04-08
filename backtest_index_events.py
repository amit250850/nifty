#!/usr/bin/env python3
"""
backtest_index_events.py
========================
Pre-event straddle backtest for NIFTY and BANKNIFTY using WEEKLY options
on known macro event dates: RBI MPC, Union Budget, FOMC (India-impact day).

Why weekly options work here
-----------------------------
NIFTY and BANKNIFTY still have weekly Thursday expiries even after SEBI
discontinued stock weeklies. So DTE=1 is real and achievable — straddle
cost is low, break-even is tight, and even a 1.5–2% index move profits.

Strategy
---------
  • Entry : Buy ATM call + put at prev-day close  (DTE=1 weekly)
  • Pre-IV: realised vol × 1.35  (indices spike less than stocks pre-event)
  • Post-IV: realised vol × 0.85 (IV crush modelled)
  • Exit  : 2H (≈11:15 AM) — captures move before crush
  • Loser salvage: 15% of entry premium
  • Also shows EOD result for comparison

Usage
-----
  python backtest_index_events.py                  # both indices, all events
  python backtest_index_events.py --index NIFTY    # NIFTY only
  python backtest_index_events.py --index BANKNIFTY
  python backtest_index_events.py --show-all       # include IV/vol detail
  python backtest_index_events.py --type rbi       # only RBI events
  python backtest_index_events.py --type budget
  python backtest_index_events.py --type fomc

Requirements: yfinance, pandas  (run on Windows D: drive machine)
"""

import argparse
import math
import sys
from datetime import date, timedelta

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: yfinance / pandas not installed.  pip install yfinance pandas")
    sys.exit(1)

# ── Index config ───────────────────────────────────────────────────────────────
INDEX_CONFIG = {
    "NIFTY": {
        "ticker":     "^NSEI",
        "lot_size":   25,           # post-Nov-2024 lot size
        "strike_rnd": 50,
        "label":      "NIFTY 50",
    },
    "BANKNIFTY": {
        "ticker":     "^NSEBANK",
        "lot_size":   15,           # post-Jul-2024 lot size
        "strike_rnd": 100,
        "label":      "BANK NIFTY",
    },
}

# ── Known macro event dates (India market reaction date) ──────────────────────
# For RBI/Budget: the day of announcement (market opens and reacts same day)
# For FOMC: day AFTER the announcement (India opens next morning with the news)
EVENTS = [
    # ── RBI MPC Policy Decisions ──────────────────────────────────────────────
    # Decision announced ~10 AM IST; market spikes/falls on open or at 10 AM
    {"date": date(2024,  4,  5), "type": "rbi",    "label": "RBI Apr-24 (hold)"},
    {"date": date(2024,  6,  7), "type": "rbi",    "label": "RBI Jun-24 (hold)"},
    {"date": date(2024,  8,  8), "type": "rbi",    "label": "RBI Aug-24 (hold)"},
    {"date": date(2024, 10,  9), "type": "rbi",    "label": "RBI Oct-24 (hold)"},
    {"date": date(2024, 12,  6), "type": "rbi",    "label": "RBI Dec-24 (hold)"},
    {"date": date(2025,  2,  7), "type": "rbi",    "label": "RBI Feb-25 (-25bp)"},
    {"date": date(2025,  4,  9), "type": "rbi",    "label": "RBI Apr-25 (-25bp)"},
    {"date": date(2025,  6,  6), "type": "rbi",    "label": "RBI Jun-25"},
    {"date": date(2025,  8,  6), "type": "rbi",    "label": "RBI Aug-25"},
    {"date": date(2025, 10,  8), "type": "rbi",    "label": "RBI Oct-25"},
    {"date": date(2025, 12,  5), "type": "rbi",    "label": "RBI Dec-25"},
    {"date": date(2026,  2,  6), "type": "rbi",    "label": "RBI Feb-26"},
    {"date": date(2026,  4,  9), "type": "rbi",    "label": "RBI Apr-26"},  # upcoming

    # ── Union Budget ──────────────────────────────────────────────────────────
    # Budget speech starts 11 AM — massive volatility
    {"date": date(2024,  2,  1), "type": "budget", "label": "Budget Feb-24 (interim)"},
    {"date": date(2024,  7, 23), "type": "budget", "label": "Budget Jul-24 (full)"},
    {"date": date(2025,  2,  1), "type": "budget", "label": "Budget Feb-25 (full)"},
    {"date": date(2026,  2,  1), "type": "budget", "label": "Budget Feb-26 (full)"},

    # ── FOMC (India reaction day = day after announcement) ────────────────────
    # US announces late night IST; Indian market gaps next morning
    {"date": date(2024,  1, 31), "type": "fomc",   "label": "FOMC Jan-24 (hold)"},
    {"date": date(2024,  3, 21), "type": "fomc",   "label": "FOMC Mar-24 (hold)"},
    {"date": date(2024,  5,  2), "type": "fomc",   "label": "FOMC May-24 (hold)"},
    {"date": date(2024,  6, 13), "type": "fomc",   "label": "FOMC Jun-24 (hold)"},
    {"date": date(2024,  7, 31), "type": "fomc",   "label": "FOMC Jul-24 (hold)"},
    {"date": date(2024,  9, 19), "type": "fomc",   "label": "FOMC Sep-24 (-50bp)"},  # big cut
    {"date": date(2024, 11,  8), "type": "fomc",   "label": "FOMC Nov-24 (-25bp)"},
    {"date": date(2024, 12, 19), "type": "fomc",   "label": "FOMC Dec-24 (-25bp)"},
    {"date": date(2025,  1, 30), "type": "fomc",   "label": "FOMC Jan-25 (hold)"},
    {"date": date(2025,  3, 20), "type": "fomc",   "label": "FOMC Mar-25 (hold)"},
    {"date": date(2025,  5,  8), "type": "fomc",   "label": "FOMC May-25 (hold)"},
    {"date": date(2025,  6, 19), "type": "fomc",   "label": "FOMC Jun-25"},
    {"date": date(2025,  7, 31), "type": "fomc",   "label": "FOMC Jul-25"},
    {"date": date(2025,  9, 18), "type": "fomc",   "label": "FOMC Sep-25"},
    {"date": date(2025, 10, 30), "type": "fomc",   "label": "FOMC Oct-25"},
    {"date": date(2025, 12, 11), "type": "fomc",   "label": "FOMC Dec-25"},
    {"date": date(2026,  1, 29), "type": "fomc",   "label": "FOMC Jan-26"},
    {"date": date(2026,  3, 19), "type": "fomc",   "label": "FOMC Mar-26"},
]

# ── Black-Scholes ──────────────────────────────────────────────────────────────
def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)

def atm_strike(spot, rnd):
    return round(spot / rnd) * rnd

# ── Realised vol ───────────────────────────────────────────────────────────────
def realised_vol(daily, as_of_date, window=30):
    closes = daily["Close"].squeeze()
    hist = closes[closes.index < as_of_date].tail(window)
    if len(hist) < 5:
        return 0.18
    log_ret = [math.log(float(hist.iloc[i]) / float(hist.iloc[i - 1]))
               for i in range(1, len(hist))]
    mean = sum(log_ret) / len(log_ret)
    var  = sum((r - mean)**2 for r in log_ret) / max(len(log_ret) - 1, 1)
    return max(0.10, min(1.0, math.sqrt(var) * math.sqrt(252)))

# ── Simulate one event ─────────────────────────────────────────────────────────
def simulate(daily, event, cfg, verbose):
    ev_date = event["date"]
    closes  = daily["Close"].squeeze()
    opens   = daily["Open"].squeeze()
    highs   = daily["High"].squeeze()
    lows    = daily["Low"].squeeze()

    # entry: prev trading day close
    prev_days = [d for d in closes.index if d < ev_date]
    if not prev_days:
        return None
    entry_date = prev_days[-1]
    entry_spot = float(closes[entry_date])

    # event day data
    if ev_date not in closes.index:
        return None
    ev_open  = float(opens[ev_date])
    ev_close = float(closes[ev_date])
    ev_high  = float(highs[ev_date])
    ev_low   = float(lows[ev_date])

    gap_pct = (ev_open - entry_spot) / entry_spot * 100

    # BS params
    rnd     = cfg["strike_rnd"]
    K       = atm_strike(entry_spot, rnd)
    rv      = realised_vol(daily, entry_date)
    pre_iv  = min(rv * 1.35, 1.0)    # indices spike less than stocks
    post_iv = min(rv * 0.85, 1.0)
    r       = 0.068
    T_pre   = 1.0 / 252              # DTE=1 (weekly, bought day before)
    T_2h    = 0.5 / 252              # ~2 hours left in session

    call_e = bs_call(entry_spot, K, T_pre, r, pre_iv)
    put_e  = bs_put(entry_spot, K, T_pre, r, pre_iv)
    cost   = call_e + put_e
    be_pct = cost / entry_spot * 100

    # 2H proxy spot (open + partial day move toward H or L)
    if gap_pct >= 0:
        spot_2h = ev_open + (ev_high - ev_open) * 0.35
    else:
        spot_2h = ev_open - (ev_open - ev_low) * 0.35

    call_2h = bs_call(spot_2h, K, T_2h, r, post_iv)
    put_2h  = bs_put(spot_2h, K, T_2h, r, post_iv)

    if gap_pct >= 0:
        winner_2h = call_2h
        loser_2h  = max(put_e * 0.15, bs_put(spot_2h, K, T_2h, r, post_iv))
    else:
        winner_2h = put_2h
        loser_2h  = max(call_e * 0.15, bs_call(spot_2h, K, T_2h, r, post_iv))
    exit_2h = winner_2h + loser_2h

    # EOD
    exit_eod = max(ev_close - K, 0.0) + max(K - ev_close, 0.0)

    lot   = cfg["lot_size"]
    pnl_2h  = (exit_2h  - cost) * lot
    pnl_eod = (exit_eod - cost) * lot

    return dict(
        label      = event["label"],
        ev_type    = event["type"],
        ev_date    = ev_date,
        entry_date = entry_date,
        entry_spot = entry_spot,
        strike     = K,
        gap_pct    = gap_pct,
        be_pct     = be_pct,
        cost       = cost,
        exit_2h    = exit_2h,
        exit_eod   = exit_eod,
        pnl_2h     = pnl_2h,
        pnl_eod    = pnl_eod,
        win_2h     = pnl_2h > 0,
        win_eod    = pnl_eod > 0,
        rv_pct     = rv * 100,
        pre_iv_pct = pre_iv * 100,
        lot_size   = lot,
    )

# ── Formatting helpers ─────────────────────────────────────────────────────────
def fmt(v):
    sign = "+" if v >= 0 else "-"
    v = abs(v)
    if v >= 1e5: return f"{sign}₹{v/1e5:.2f}L"
    return f"{sign}₹{v:,.0f}"

TYPE_EMOJI = {"rbi": "🏦", "budget": "📋", "fomc": "🦅"}

# ── Print one index result set ─────────────────────────────────────────────────
def print_table(index_name, trades, verbose, show_type=None):
    if not trades:
        print(f"  [{index_name}] No trades to show.")
        return

    cfg  = INDEX_CONFIG[index_name]
    sep  = "═" * 100
    print(f"\n{sep}")
    print(f"  {cfg['label']}  |  Lot: {cfg['lot_size']}  |  "
          f"Weekly expiry (DTE=1)  |  Exit at 2H (11:15 AM)")
    print(sep)

    hdr = (f"  {'Event':<30} {'Type':<7} {'Date':<12} {'Spot':>7} {'K':>6} "
           f"{'Gap%':>6} {'BE%':>5} {'Cost':>5} "
           f"{'P&L 2H':>10} {'P&L EOD':>10} {'W?':>3}")
    print(hdr)
    print("  " + "─" * 96)

    for t in trades:
        w   = "✅" if t["win_2h"] else "❌"
        em  = TYPE_EMOJI.get(t["ev_type"], "•")
        print(
            f"  {t['label']:<30} {em+t['ev_type']:<7} {str(t['ev_date']):<12} "
            f"{t['entry_spot']:>7,.0f} {t['strike']:>6,.0f} "
            f"{t['gap_pct']:>+6.1f}% {t['be_pct']:>4.2f}% "
            f"{t['cost']:>5.1f} "
            f"{fmt(t['pnl_2h']):>10} {fmt(t['pnl_eod']):>10} {w:>3}"
        )
        if verbose:
            print(f"    RV={t['rv_pct']:.1f}%  PreIV={t['pre_iv_pct']:.1f}%  "
                  f"Entry@{t['entry_date']}  LotSize={t['lot_size']}")

    print("  " + "─" * 96)

    n         = len(trades)
    wins_2h   = sum(1 for t in trades if t["win_2h"])
    wins_eod  = sum(1 for t in trades if t["win_eod"])
    tot_2h    = sum(t["pnl_2h"]  for t in trades)
    tot_eod   = sum(t["pnl_eod"] for t in trades)
    avg_gap   = sum(abs(t["gap_pct"]) for t in trades) / n
    avg_cost  = sum(t["cost"] for t in trades) / n
    avg_be    = sum(t["be_pct"] for t in trades) / n
    outlay    = avg_cost * trades[0]["lot_size"]

    print(f"\n  SUMMARY  ({n} events)")
    print(f"  Win rate — 2H exit : {wins_2h}/{n} = {wins_2h/n*100:.0f}%   "
          f"Total = {fmt(tot_2h)}")
    print(f"  Win rate — EOD exit: {wins_eod}/{n} = {wins_eod/n*100:.0f}%   "
          f"Total = {fmt(tot_eod)}")
    print(f"  Avg move on event  : {avg_gap:.2f}%")
    print(f"  Avg straddle cost  : {avg_cost:.1f} pts  →  ₹{outlay:,.0f}/lot")
    print(f"  Avg break-even     : {avg_be:.2f}%  (need {avg_be:.2f}% move to profit)")
    print(f"  Avg P&L per trade  : 2H={fmt(tot_2h/n)}   EOD={fmt(tot_eod/n)}")
    print(f"  2H beats EOD by    : {fmt(tot_2h - tot_eod)}  over {n} trades")

    # per-type breakdown
    for ev_type in ("rbi", "budget", "fomc"):
        sub = [t for t in trades if t["ev_type"] == ev_type]
        if not sub: continue
        sw  = sum(1 for t in sub if t["win_2h"])
        sp  = sum(t["pnl_2h"] for t in sub)
        print(f"\n  {TYPE_EMOJI[ev_type]} {ev_type.upper():10} : "
              f"{sw}/{len(sub)} wins = {sw/len(sub)*100:.0f}%  |  "
              f"Total 2H P&L = {fmt(sp)}  |  "
              f"Avg/trade = {fmt(sp/len(sub))}")

    # budget guide
    print(f"\n  BUDGET GUIDE  ({index_name}):")
    print(f"    1 lot  : ₹{outlay:,.0f}  outlay  (keep ₹{outlay*1.3:,.0f} with buffer)")
    print(f"    2 lots : ₹{outlay*2:,.0f}  outlay")
    print(f"    5 lots : ₹{outlay*5:,.0f}  outlay")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="NIFTY/BANKNIFTY event straddle backtest (weekly options)")
    p.add_argument("--index",    choices=["NIFTY", "BANKNIFTY", "both"],
                   default="both", help="Which index to test (default: both)")
    p.add_argument("--type",     choices=["rbi", "budget", "fomc", "all"],
                   default="all", help="Filter event type (default: all)")
    p.add_argument("--show-all", action="store_true",
                   help="Show IV / vol details per trade")
    args = p.parse_args()

    today = date.today()
    indices = ["NIFTY", "BANKNIFTY"] if args.index == "both" else [args.index]
    ev_filter = None if args.type == "all" else args.type

    # filter events
    events = [e for e in EVENTS if e["date"] <= today]
    if ev_filter:
        events = [e for e in events if e["type"] == ev_filter]

    print(f"\n{'═'*60}")
    print(f"  Index Event Straddle Backtest")
    print(f"  Events: {len(events)}  |  Type: {args.type}  |  Index: {args.index}")
    print(f"  Strategy: Buy ATM weekly straddle day before event → exit 2H")
    print(f"{'═'*60}")

    data_start = min(e["date"] for e in events) - timedelta(days=60)
    data_end   = max(e["date"] for e in events) + timedelta(days=5)

    all_results = {}

    for idx_name in indices:
        cfg    = INDEX_CONFIG[idx_name]
        ticker = cfg["ticker"]
        print(f"\n[*] Fetching {ticker} from {data_start} to {data_end} …")
        daily = yf.download(ticker, start=str(data_start), end=str(data_end),
                            interval="1d", auto_adjust=True, progress=False)
        if daily.empty:
            print(f"  [!] No data for {ticker} — skipping")
            continue
        daily.index = pd.to_datetime(daily.index).date

        trades = []
        skipped = 0
        for ev in events:
            r = simulate(daily, ev, cfg, args.show_all)
            if r:
                trades.append(r)
            else:
                skipped += 1

        if skipped:
            print(f"  [{idx_name}] {skipped} events skipped (holiday / data gap)")

        all_results[idx_name] = trades
        print_table(idx_name, trades, args.show_all, ev_filter)

    # save CSV
    if all_results:
        try:
            import csv, os
            out = os.path.join(os.path.dirname(__file__), "index_event_straddle.csv")
            fields = ["index", "label", "ev_type", "ev_date", "entry_date",
                      "entry_spot", "strike", "gap_pct", "be_pct", "cost",
                      "exit_2h", "pnl_2h", "win_2h", "exit_eod", "pnl_eod",
                      "win_eod", "rv_pct", "pre_iv_pct", "lot_size"]
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for idx_name, trades in all_results.items():
                    for t in trades:
                        row = {**t, "index": idx_name}
                        w.writerow({k: row[k] for k in fields})
            print(f"\n  Results saved → {out}")
        except Exception as e:
            print(f"  [CSV save failed: {e}]")

    print("\n  KEY RULES:")
    print("  1. Enter day BEFORE the event at 3:15–3:20 PM (closing price)")
    print("  2. On event morning: sell the losing leg at open")
    print("  3. At 11:15 AM: HARD EXIT the winning leg — do not hold past this")
    print("  4. Never hold to EOD — IV crush gives back 30–50% of your 2H gains")
    print("  5. RBI > Budget > FOMC for India impact  (confirm from breakdown)\n")


if __name__ == "__main__":
    main()
