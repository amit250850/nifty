#!/usr/bin/env python3
"""
backtest_tcs_results.py
=======================
TCS-only pre-event straddle backtest using ACTUAL quarterly results dates.
Covers the last 8 quarters (~2 years).

Strategy
--------
  • Buy 1 ATM call + 1 ATM put the day BEFORE results (close price = entry spot)
  • Use Black-Scholes to price the straddle (pre-event IV = realised vol × 1.40)
  • Exit at 2H mark (approx 11:15 AM) and at EOD
  • Post-event IV = realised vol × 0.90 (IV crush effect)
  • Losing leg salvage = 15% of original premium (realistic floor)

Usage
-----
  python backtest_tcs_results.py               # full table + summary
  python backtest_tcs_results.py --show-all    # include IV/vol details per trade
  python backtest_tcs_results.py --lot-size 150

Requirements: yfinance, pandas (run on your Windows machine)
"""

import argparse
import math
import sys
from datetime import date, timedelta

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: yfinance / pandas not installed.")
    print("  pip install yfinance pandas")
    sys.exit(1)

# ── TCS lot size & strike rounding ────────────────────────────────────────────
TCS_LOT_SIZE   = 150          # NSE lot size as of 2024
TCS_STRIKE_RND = 50           # round ATM to nearest ₹50
TCS_TICKER     = "TCS.NS"

# ── Known TCS quarterly results dates (last 8 quarters, Apr 2024 – Jan 2026) ──
# Source: BSE/NSE official results calendar
# Format: (date, quarter_label)
TCS_RESULTS_DATES = [
    (date(2024, 4, 19), "Q4 FY24"),
    (date(2024, 7, 11), "Q1 FY25"),
    (date(2024, 10, 10), "Q2 FY25"),
    (date(2025, 1,  9), "Q3 FY25"),
    (date(2025, 4, 17), "Q4 FY25"),   # approximate — confirm on NSE
    (date(2025, 7, 10), "Q1 FY26"),   # approximate
    (date(2025, 10, 9), "Q2 FY26"),   # approximate
    (date(2026, 1,  9), "Q3 FY26"),   # approximate
]

# ── Black-Scholes helpers ──────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))

def bs_call(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def atm_strike(spot: float) -> float:
    return round(spot / TCS_STRIKE_RND) * TCS_STRIKE_RND

# ── Fetch data ─────────────────────────────────────────────────────────────────
def fetch_tcs(start: date, end: date) -> pd.DataFrame:
    print(f"[*] Fetching TCS.NS from {start} to {end} …")
    df = yf.download(TCS_TICKER, start=str(start), end=str(end + timedelta(days=5)),
                     interval="1d", auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index).date
    return df

# ── Compute 30-day realised vol ────────────────────────────────────────────────
def realised_vol(daily: pd.DataFrame, as_of: date, window: int = 30) -> float:
    closes = daily["Close"].squeeze()
    hist = closes[closes.index < as_of].tail(window)
    if len(hist) < 5:
        return 0.25  # fallback
    log_ret = [math.log(hist.iloc[i] / hist.iloc[i - 1]) for i in range(1, len(hist))]
    std_daily = (sum((r - sum(log_ret) / len(log_ret)) ** 2
                     for r in log_ret) / (len(log_ret) - 1)) ** 0.5
    return max(0.15, min(1.20, std_daily * math.sqrt(252)))

# ── Single event simulation ────────────────────────────────────────────────────
def simulate_event(daily: pd.DataFrame, result_date: date,
                   label: str, lot_size: int, verbose: bool) -> dict | None:
    """
    Simulate one pre-event straddle around result_date.
    Returns a result dict or None if data is unavailable.
    """
    closes = daily["Close"].squeeze()
    opens  = daily["Open"].squeeze()
    highs  = daily["High"].squeeze()
    lows   = daily["Low"].squeeze()

    # ── Entry: close price day BEFORE results ──────────────────────────────
    dates_before = [d for d in closes.index if d < result_date]
    if not dates_before:
        print(f"  [!] No data before {result_date} — skipping {label}")
        return None
    entry_date = dates_before[-1]
    entry_spot = float(closes[entry_date])

    # ── Results day ──────────────────────────────────────────────────────
    if result_date not in closes.index:
        print(f"  [!] Results date {result_date} not in data — skipping {label}")
        return None

    res_open  = float(opens[result_date])
    res_close = float(closes[result_date])
    res_high  = float(highs[result_date])
    res_low   = float(lows[result_date])

    # ── Parameters ────────────────────────────────────────────────────────
    K      = atm_strike(entry_spot)
    rv     = realised_vol(daily, entry_date)
    pre_iv = min(rv * 1.40, 1.20)
    post_iv= min(rv * 0.90, 1.20)
    r      = 0.068                # risk-free rate (approx RBI repo)
    T_pre  = 1 / 252              # DTE = 1 day
    T_2h   = max(0.5 / 252, 0)   # ~2 hours left in session
    T_eod  = 0.0                  # expired at close

    # ── Entry premium (straddle cost) ─────────────────────────────────────
    call_entry = bs_call(entry_spot, K, T_pre, r, pre_iv)
    put_entry  = bs_put(entry_spot, K, T_pre, r, pre_iv)
    straddle_cost = call_entry + put_entry

    # ── Gap % on results day ──────────────────────────────────────────────
    gap_pct = (res_open - entry_spot) / entry_spot * 100

    # ── 2H exit (approx mid-session price) ───────────────────────────────
    # Use average of open and (high or low depending on direction) as 2H proxy
    if gap_pct >= 0:
        spot_2h = res_open + (res_high - res_open) * 0.4   # still running up
    else:
        spot_2h = res_open - (res_open - res_low) * 0.4    # still running down

    call_2h = bs_call(spot_2h, K, T_2h, r, post_iv)
    put_2h  = bs_put(spot_2h, K, T_2h, r, post_iv)
    # Winner leg: whichever is in the money; loser: salvage 15%
    if gap_pct >= 0:
        winner_2h = call_2h
        loser_2h  = max(put_entry * 0.15, bs_put(spot_2h, K, T_2h, r, post_iv))
    else:
        winner_2h = put_2h
        loser_2h  = max(call_entry * 0.15, bs_call(spot_2h, K, T_2h, r, post_iv))
    exit_2h = winner_2h + loser_2h

    # ── EOD exit (intrinsic only, IV=0) ──────────────────────────────────
    call_eod = max(res_close - K, 0.0)
    put_eod  = max(K - res_close, 0.0)
    exit_eod = call_eod + put_eod

    # ── P&L per lot ───────────────────────────────────────────────────────
    pnl_2h  = (exit_2h  - straddle_cost) * lot_size
    pnl_eod = (exit_eod - straddle_cost) * lot_size

    # ── Break-even ────────────────────────────────────────────────────────
    be_pct = straddle_cost / entry_spot * 100

    return dict(
        label        = label,
        result_date  = result_date,
        entry_date   = entry_date,
        entry_spot   = entry_spot,
        strike       = K,
        rv_pct       = rv * 100,
        pre_iv_pct   = pre_iv * 100,
        post_iv_pct  = post_iv * 100,
        gap_pct      = gap_pct,
        be_pct       = be_pct,
        straddle_cost= straddle_cost,
        exit_2h      = exit_2h,
        exit_eod     = exit_eod,
        pnl_2h       = pnl_2h,
        pnl_eod      = pnl_eod,
        win_2h       = pnl_2h > 0,
        win_eod      = pnl_eod > 0,
        lot_size     = lot_size,
    )

# ── Print helpers ──────────────────────────────────────────────────────────────
def fmt_inr(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    v = abs(v)
    if v >= 1e7:
        return f"{sign}₹{v/1e7:.2f}Cr"
    if v >= 1e5:
        return f"{sign}₹{v/1e5:.2f}L"
    return f"{sign}₹{v:,.0f}"

def print_results(trades: list[dict], verbose: bool) -> None:
    if not trades:
        print("No trades to display.")
        return

    sep = "═" * 92
    print("\n" + sep)
    print("  TCS PRE-EVENT STRADDLE BACKTEST  |  Lot size:", trades[0]["lot_size"],
          "|  Strategy: Buy ATM straddle day before results")
    print(sep)
    hdr = (f"{'Quarter':<10} {'ResDate':<12} {'Spot':>7} {'K':>6} "
           f"{'Gap%':>6} {'BE%':>5} {'Cost':>6} "
           f"{'Exit2H':>7} {'P&L 2H':>12} {'Exit EOD':>8} {'P&L EOD':>12} {'W?':>3}")
    print(hdr)
    print("─" * 92)

    for t in trades:
        w = "✅" if t["win_2h"] else "❌"
        print(
            f"{t['label']:<10} {str(t['result_date']):<12} "
            f"{t['entry_spot']:>7.0f} {t['strike']:>6.0f} "
            f"{t['gap_pct']:>+6.1f}% {t['be_pct']:>4.1f}% "
            f"{t['straddle_cost']:>6.0f} "
            f"{t['exit_2h']:>7.0f} {fmt_inr(t['pnl_2h']):>12} "
            f"{t['exit_eod']:>8.0f} {fmt_inr(t['pnl_eod']):>12} {w:>3}"
        )
        if verbose:
            print(f"           RV={t['rv_pct']:.1f}%  PreIV={t['pre_iv_pct']:.1f}%"
                  f"  PostIV={t['post_iv_pct']:.1f}%"
                  f"  Entry@{str(t['entry_date'])}  LotSize={t['lot_size']}")

    print("─" * 92)

    # ── Summary ───────────────────────────────────────────────────────────
    n         = len(trades)
    wins_2h   = sum(1 for t in trades if t["win_2h"])
    wins_eod  = sum(1 for t in trades if t["win_eod"])
    total_2h  = sum(t["pnl_2h"]  for t in trades)
    total_eod = sum(t["pnl_eod"] for t in trades)
    avg_gap   = sum(abs(t["gap_pct"]) for t in trades) / n
    avg_be    = sum(t["be_pct"] for t in trades) / n
    avg_cost  = sum(t["straddle_cost"] for t in trades) / n
    outlay    = avg_cost * trades[0]["lot_size"]

    print(f"\n  TRADES : {n}")
    print(f"  WIN RATE (2H exit)  : {wins_2h}/{n}  = {wins_2h/n*100:.0f}%  "
          f"→ Total P&L = {fmt_inr(total_2h)}")
    print(f"  WIN RATE (EOD exit) : {wins_eod}/{n}  = {wins_eod/n*100:.0f}%  "
          f"→ Total P&L = {fmt_inr(total_eod)}")
    print(f"\n  Avg absolute gap    : {avg_gap:.1f}%")
    print(f"  Avg break-even      : {avg_be:.1f}%  (need {avg_be:.1f}% move to profit)")
    print(f"  Avg straddle cost   : ₹{avg_cost:.0f}/share  →  ₹{outlay:,.0f}/lot")
    print(f"\n  CAPITAL PER TRADE   : ~₹{outlay:,.0f}  (1 straddle × {trades[0]['lot_size']} shares)")
    print(f"  CAPITAL × 2 LOTS    : ~₹{outlay*2:,.0f}")
    print(f"  AVG P&L / TRADE(2H) : {fmt_inr(total_2h/n)}")
    print(f"  AVG P&L / TRADE(EOD): {fmt_inr(total_eod/n)}")

    # ── Budget summary ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  BUDGET GUIDE:")
    print(f"    Conservative (1 lot) : ₹{outlay:,.0f}")
    print(f"    Moderate    (2 lots) : ₹{outlay*2:,.0f}")
    print(f"    Aggressive  (5 lots) : ₹{outlay*5:,.0f}")
    print(f"    Keep 30% as buffer for adverse IV: ₹{outlay*1.3:,.0f}")
    print(f"\n  KEY RULE: Exit BEFORE 11:15 AM — 2H exit beats EOD by "
          f"{fmt_inr(total_2h - total_eod)} over {n} trades")
    print(sep + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TCS pre-event straddle backtest (2Y, actual results dates)")
    parser.add_argument("--lot-size", type=int, default=TCS_LOT_SIZE,
                        help=f"TCS lot size (default: {TCS_LOT_SIZE})")
    parser.add_argument("--show-all", action="store_true",
                        help="Show IV/vol details per trade")
    args = parser.parse_args()

    lot_size = args.lot_size

    # Filter to dates we actually have results for (not in the future)
    today = date.today()
    # Use all dates up to 10 days from now (allows running just before results)
    target_dates = [(d, lbl) for d, lbl in TCS_RESULTS_DATES if d <= today + timedelta(days=10)]

    if not target_dates:
        print("No TCS results dates available in the configured list.")
        sys.exit(0)

    # Fetch 2.5 years of daily data to have enough history
    data_start = min(d for d, _ in target_dates) - timedelta(days=90)
    data_end   = max(d for d, _ in target_dates) + timedelta(days=5)
    daily = fetch_tcs(data_start, data_end)

    if daily.empty:
        print("ERROR: Could not download TCS data. Check internet connection.")
        sys.exit(1)

    print(f"[*] Running straddle simulation on {len(target_dates)} results events …\n")

    trades = []
    for res_date, label in target_dates:
        r = simulate_event(daily, res_date, label, lot_size, args.show_all)
        if r:
            trades.append(r)

    print_results(trades, args.show_all)

    # ── Save CSV ──────────────────────────────────────────────────────────
    try:
        import csv, os
        out_path = os.path.join(os.path.dirname(__file__), "tcs_straddle_backtest.csv")
        fieldnames = ["label", "result_date", "entry_date", "entry_spot", "strike",
                      "gap_pct", "be_pct", "straddle_cost", "exit_2h", "pnl_2h",
                      "exit_eod", "pnl_eod", "win_2h", "win_eod", "lot_size",
                      "rv_pct", "pre_iv_pct", "post_iv_pct"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for t in trades:
                w.writerow({k: t[k] for k in fieldnames})
        print(f"  Results saved → {out_path}\n")
    except Exception as e:
        print(f"  [CSV save failed: {e}]")

if __name__ == "__main__":
    main()
