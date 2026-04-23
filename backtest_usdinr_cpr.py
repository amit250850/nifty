#!/usr/bin/env python3
"""
backtest_usdinr_cpr.py
======================
CPR (Central Pivot Range) backtest for USDINR futures — NSE Currency segment.

Strategy
--------
  Narrow CPR width (<threshold) → trending day expected:
    First candle (9:15–10:15) closes ABOVE upper_cpr → LONG  @ upper+slip, SL=lower-slip, T=R1
    First candle (9:15–10:15) closes BELOW lower_cpr → SHORT @ lower-slip, SL=upper+slip, T=S1
  Wide CPR → SKIP (choppy/range day)

  Also tests a European-session filter (3:00–5:00 PM) as a separate run.

USDINR Contract Specs (NSE)
----------------------------
  Lot size   : 1,000 USD
  P&L        : ₹1,000 per ₹1 (1 paisa) move  [1 USD = 100 paise]
  Margin     : ~₹1,700–2,500 per lot (SPAN + Exposure, ~2–3% of notional)
  Budget ₹35K: 12–18 lots available; use 2 lots/trade safely

  USDINR is quoted as INR per USD (e.g. 84.25)
  1 pip = 0.0025 INR → P&L = 0.0025 × 1000 = ₹2.50 per lot per pip
  1 paisa move (0.01 INR) → ₹10 per lot

Usage
-----
  python backtest_usdinr_cpr.py
  python backtest_usdinr_cpr.py --days 365
  python backtest_usdinr_cpr.py --days 720 --narrow 0.015 --lots 2 --show-trades
  python backtest_usdinr_cpr.py --session european   # 3PM–5PM only
  python backtest_usdinr_cpr.py --session morning     # 9:15AM–12PM only
  python backtest_usdinr_cpr.py --session full        # full day (default)
"""

import argparse
import math
import sys
import warnings
from collections import defaultdict
from datetime import date, timedelta

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("ERROR: pip install pandas yfinance")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKER        = "USDINR=X"          # yfinance symbol for USD/INR spot
COMMISSION    = 30                  # ₹ per trade round-trip (NSE currency brokerage ~₹20–30)
SLIPPAGE_INR  = 0.0025              # 1 pip slippage per side (0.25 paisa)
PNL_PER_LOT   = 1000               # ₹1,000 per lot per ₹1 move in USDINR rate
MARGIN_PER_LOT = 2200              # ₹ per lot (conservative SPAN+EM estimate)

# Session windows (hour, minute) — IST
SESSION_OPEN    = (9, 15)
SESSION_CLOSE   = (17, 0)
MORNING_END     = (12, 0)
EUROPEAN_START  = (15, 0)

DEFAULT_NARROW  = 0.02   # CPR width < 0.02% = narrow (trending day)
DEFAULT_DAYS    = 365
DEFAULT_LOTS    = 2


# ─── DATA ─────────────────────────────────────────────────────────────────────
def fetch_data(days: int) -> pd.DataFrame:
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)
    print(f"  Downloading {TICKER} — 1H, {days} days ({start_dt} → {end_dt}) …")
    df = yf.download(
        tickers      = TICKER,
        start        = str(start_dt),
        end          = str(end_dt),
        interval     = "1h",
        progress     = False,
        auto_adjust  = True,
    )
    if df.empty:
        sys.exit("ERROR: yfinance returned no data for USDINR=X")

    # Flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index)

    # yfinance returns UTC — convert to IST
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")

    print(f"  Downloaded {len(df)} hourly bars  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─── CPR ──────────────────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1        = 2 * P - L
    S1        = 2 * P - H
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1, width_pct=width_pct)


# ─── EXIT SCAN ────────────────────────────────────────────────────────────────
def exit_scan(bars: pd.DataFrame, entry: float, sl: float,
              target: float, direction: str) -> tuple[str, float]:
    for _, bar in bars.iterrows():
        h, l = float(bar["high"]), float(bar["low"])
        if direction == "LONG":
            if l <= sl:     return "SL",     sl
            if h >= target: return "TARGET",  target
        else:
            if h >= sl:     return "SL",     sl
            if l <= target: return "TARGET",  target
    eod = float(bars.iloc[-1]["close"]) if not bars.empty else entry
    return "EOD", eod


# ─── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, narrow_thresh: float,
                 lots: int, session: str) -> dict:
    """
    session: "full" | "morning" | "european"
      full      → use first 1H candle regardless of time
      morning   → only use bars 9:15–12:00 for entry detection
      european  → only use bars 15:00–17:00 for entry detection
    """
    # Build daily OHLC from hourly data (for CPR calc — use previous day)
    daily = (df.resample("1D")
               .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
               .dropna())
    daily.index = daily.index.date

    all_days = sorted(daily.index)
    trades   = []
    month_pnl = defaultdict(float)
    stats     = defaultdict(int)

    for i, today in enumerate(all_days):
        if i == 0:
            continue

        yesterday = all_days[i - 1]
        try:
            pH = float(daily.loc[yesterday, "high"])
            pL = float(daily.loc[yesterday, "low"])
            pC = float(daily.loc[yesterday, "close"])
        except (KeyError, ValueError):
            continue

        cpr   = calc_cpr(pH, pL, pC)
        width = cpr["width_pct"]

        if width >= narrow_thresh:
            stats["wide_skipped"] += 1
            continue
        stats["narrow_days"] += 1

        # Get today's hourly bars within session
        day_bars = df[df.index.date == today].copy()

        h = day_bars.index.hour
        m = day_bars.index.minute
        after_open = (h > SESSION_OPEN[0]) | ((h == SESSION_OPEN[0]) & (m >= SESSION_OPEN[1]))

        if session == "morning":
            day_bars = day_bars[after_open & (h < MORNING_END[0])]
        elif session == "european":
            day_bars = day_bars[h >= EUROPEAN_START[0]]
        else:  # full
            day_bars = day_bars[after_open]

        if len(day_bars) < 2:
            stats["no_intraday"] += 1
            continue

        upper = cpr["upper_cpr"]
        lower = cpr["lower_cpr"]
        first_close = float(day_bars.iloc[0]["close"])

        direction = entry = sl = target = None

        if first_close > upper:
            direction = "LONG"
            entry  = upper + SLIPPAGE_INR
            sl     = lower - SLIPPAGE_INR
            target = cpr["R1"]
        elif first_close < lower:
            direction = "SHORT"
            entry  = lower - SLIPPAGE_INR
            sl     = upper + SLIPPAGE_INR
            target = cpr["S1"]
        else:
            stats["no_signal"] += 1
            continue

        if direction == "LONG"  and target <= entry: stats["no_signal"] += 1; continue
        if direction == "SHORT" and target >= entry: stats["no_signal"] += 1; continue

        outcome, exit_raw = exit_scan(day_bars.iloc[1:], entry, sl, target, direction)

        if outcome in ("TARGET", "SL"):
            exit_price = (exit_raw - SLIPPAGE_INR if direction == "LONG"
                          else exit_raw + SLIPPAGE_INR)
        else:
            exit_price = exit_raw

        raw_move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        net_pnl  = raw_move * PNL_PER_LOT * lots - COMMISSION
        month_key = today.strftime("%b %Y")
        month_pnl[month_key] += net_pnl

        trades.append(dict(
            date      = today,
            direction = direction,
            entry     = round(entry, 4),
            exit      = round(exit_price, 4),
            sl        = round(sl, 4),
            target    = round(target, 4),
            outcome   = outcome,
            move_inr  = round(raw_move, 4),
            net_pnl   = round(net_pnl, 2),
            width_pct = round(width, 4),
        ))

    return dict(trades=trades, month_pnl=month_pnl, stats=stats)


# ─── REPORT ───────────────────────────────────────────────────────────────────
def print_report(result: dict, lots: int, narrow_thresh: float,
                 session: str, days: int) -> None:
    trades    = result["trades"]
    month_pnl = result["month_pnl"]
    stats     = result["stats"]

    if not trades:
        print("\n  No trades generated. Try --narrow with a higher value.\n")
        return

    wins     = [t for t in trades if t["net_pnl"] > 0]
    losses   = [t for t in trades if t["net_pnl"] <= 0]
    targets  = [t for t in trades if t["outcome"] == "TARGET"]
    sls      = [t for t in trades if t["outcome"] == "SL"]
    eods     = [t for t in trades if t["outcome"] == "EOD"]

    total_pnl    = sum(t["net_pnl"] for t in trades)
    avg_win      = sum(t["net_pnl"] for t in wins)  / len(wins)  if wins   else 0
    avg_loss     = sum(t["net_pnl"] for t in losses) / len(losses) if losses else 0
    win_rate     = len(wins) / len(trades) * 100

    margin_used  = lots * MARGIN_PER_LOT
    roi_pct      = total_pnl / margin_used * 100 if margin_used else 0

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
  USDINR CPR Backtest — {session.upper()} session
  Period: {days} days  |  Narrow CPR < {narrow_thresh:.3f}%  |  {lots} lot(s)
╚══════════════════════════════════════════════════════════════════╝

  Contract  : USDINR Futures (NSE Currency segment)
  Lot size  : 1,000 USD  |  Margin ~₹{MARGIN_PER_LOT:,}/lot
  Capital   : ₹{margin_used:,} ({lots} lot margin)  ← well within ₹35K budget

  ── Signal Stats ────────────────────────────────────────────────
  Days scanned     : {stats["narrow_days"] + stats["wide_skipped"]}
  Narrow CPR days  : {stats["narrow_days"]}  ({stats["narrow_days"]/(stats["narrow_days"]+stats["wide_skipped"])*100:.1f}% of days)
  Wide/skipped     : {stats["wide_skipped"]}
  Signal generated : {len(trades)}  (first candle outside CPR)
  No signal (inside CPR) : {stats["no_signal"]}

  ── Trade Results ───────────────────────────────────────────────
  Total trades     : {len(trades)}
  Winners          : {len(wins)}  ({win_rate:.1f}%)
  Losers           : {len(losses)}
  Target hits      : {len(targets)}
  SL hits          : {len(sls)}
  EOD exits        : {len(eods)}
  Avg win          : ₹{avg_win:,.0f}
  Avg loss         : ₹{avg_loss:,.0f}
  Payoff ratio     : {abs(avg_win/avg_loss):.2f}:1  (need >1 to be profitable)

  ── P&L ─────────────────────────────────────────────────────────
  Total net P&L    : ₹{total_pnl:,.0f}
  ROI on margin    : {roi_pct:.1f}%  over {days} days
  Per trade avg    : ₹{total_pnl/len(trades):,.0f}
""")

    # Monthly breakdown
    print("  ── Monthly P&L ─────────────────────────────────────────────────")
    running = 0
    for month in sorted(month_pnl.keys(),
                        key=lambda m: pd.to_datetime(m, format="%b %Y")):
        pnl = month_pnl[month]
        running += pnl
        bar = "█" * int(abs(pnl) / 500) if pnl != 0 else ""
        sign = "+" if pnl >= 0 else ""
        print(f"  {month:<10}  {sign}₹{pnl:>8,.0f}  {bar}")
    print(f"  {'TOTAL':<10}  ₹{running:>8,.0f}\n")


def print_trades(trades: list) -> None:
    print("\n  ── Individual Trades ───────────────────────────────────────────")
    print(f"  {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'Move':>8} {'Outcome':<8} {'P&L':>8}")
    print(f"  {'-'*66}")
    for t in trades:
        sign = "+" if t["net_pnl"] >= 0 else ""
        move_paise = t["move_inr"] * 100
        print(f"  {str(t['date']):<12} {t['direction']:<6} "
              f"{t['entry']:>8.4f} {t['exit']:>8.4f} "
              f"{move_paise:>+7.1f}p  {t['outcome']:<8} "
              f"{sign}₹{t['net_pnl']:>7,.0f}")
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="USDINR CPR Backtest")
    parser.add_argument("--days",        type=int,   default=DEFAULT_DAYS,
                        help="Days of history (default 365)")
    parser.add_argument("--narrow",      type=float, default=DEFAULT_NARROW,
                        help="CPR narrow threshold %% (default 0.02)")
    parser.add_argument("--lots",        type=int,   default=DEFAULT_LOTS,
                        help="Lots per trade (default 2)")
    parser.add_argument("--session",     choices=["full", "morning", "european"],
                        default="full",
                        help="Entry session: full/morning/european (default full)")
    parser.add_argument("--show-trades", action="store_true",
                        help="Print individual trade list")
    args = parser.parse_args()

    print(f"\n  USDINR CPR Backtest")
    print(f"  Narrow threshold : {args.narrow:.3f}%")
    print(f"  Session          : {args.session}")
    print(f"  Lots             : {args.lots}  (margin ~₹{args.lots * MARGIN_PER_LOT:,})")

    df     = fetch_data(args.days)
    result = run_backtest(df, args.narrow, args.lots, args.session)
    print_report(result, args.lots, args.narrow, args.session, args.days)

    if args.show_trades:
        print_trades(result["trades"])

    # Quick comparison across sessions if running default
    if not args.show_trades and args.session == "full":
        print("  ── Session Comparison (same parameters) ────────────────────────")
        for sess in ["morning", "european"]:
            r = run_backtest(df, args.narrow, args.lots, sess)
            t = r["trades"]
            if not t:
                print(f"  {sess:<12}  No trades")
                continue
            wins = len([x for x in t if x["net_pnl"] > 0])
            pnl  = sum(x["net_pnl"] for x in t)
            wr   = wins / len(t) * 100
            print(f"  {sess:<12}  trades={len(t):3d}  win={wr:.0f}%  P&L=₹{pnl:,.0f}")
        print()


if __name__ == "__main__":
    main()
