#!/usr/bin/env python3
"""
backtest_nickel_daily.py
========================
NICKEL CPR backtest using Kite continuous daily data.

WHY NOT THE STANDARD backtest_mcx_cpr.py
-----------------------------------------
  Kite hourly data for NICKEL only covers the last 6 weeks.
  Kite daily data with continuous=True covers 1 full year (257+ bars).
  This script uses daily H/L to simulate intraday exits — more trades,
  statistically meaningful results.

EXIT SIMULATION (daily bar, conservative)
------------------------------------------
  LONG  trade: if day's Low  <= SL     → SL hit  (worst case first)
               elif day's High >= Target → TARGET
               else                      → EOD at close
  SHORT trade: if day's High >= SL     → SL hit  (worst case first)
               elif day's Low  <= Target → TARGET
               else                      → EOD at close

  Conservative rule: if both SL and Target touched → SL assumed hit first.
  This understates performance slightly — gives a lower-bound estimate.

CONTINUOUS DATA CAVEAT
-----------------------
  Kite stitches expired + active contracts into one continuous price series.
  Roll adjustments may cause small price jumps at contract boundaries.
  CPR math is unaffected (CPR uses relative H/L/C within each day).

Usage
-----
  python backtest_nickel_daily.py
  python backtest_nickel_daily.py --narrow 0.30
  python backtest_nickel_daily.py --narrow 0.50
  python backtest_nickel_daily.py --show-trades
  python backtest_nickel_daily.py --days 180
  python backtest_nickel_daily.py --clear-cache
"""

import os, sys, pickle, logging, argparse
from datetime import date, timedelta
from collections import defaultdict

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import pandas as pd
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv pandas")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
NARROW_PCT    = 0.40    # < 0.40% → narrow CPR → signal day
WIDE_PCT      = 1.20    # > 1.20% → wide CPR  → skip
LOTS          = 1
PNL_PER_LOT   = 1_500   # 1,500 kg × ₹1/kg
COMMISSION    = 65
SLIPPAGE      = 0.20    # ₹/kg per side
MARGIN        = 48_866  # Zerodha NRML verified Apr 2026
CAPITAL       = 50_000

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "nickel_daily_cache.pkl")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite():
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        sys.exit("ERROR: run login.py first")
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k


# ─── DATA ─────────────────────────────────────────────────────────────────────
def load_data(kite, days, clear_cache=False):
    if not clear_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            c = pickle.load(f)
        log.info("Cache: %d daily bars  %s → %s",
                 len(c["daily"]), c["start"], c["end"])
        return c

    log.info("Fetching NICKEL continuous daily data …")
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    front = df[(df["name"] == "NICKEL") &
               (df["instrument_type"] == "FUT")].sort_values("expiry").iloc[0]
    token = int(front["instrument_token"])
    log.info("Front contract: %s  token=%d", front["tradingsymbol"], token)

    end   = date.today()
    start = end - timedelta(days=days)

    # continuous=True stitches all historical NICKEL contracts
    hist = kite.historical_data(token, start, end,
                                interval="day", continuous=True, oi=False)
    log.info("Downloaded %d daily bars  %s → %s",
             len(hist), hist[0]["date"], hist[-1]["date"])

    daily = pd.DataFrame(hist)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily = daily.set_index("date").sort_index()

    payload = {"daily": daily,
               "start": daily.index.min(),
               "end":   daily.index.max()}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    log.info("Cached → %s", CACHE_FILE)
    return payload


# ─── CPR MATH ─────────────────────────────────────────────────────────────────
def calc_cpr(H, L, C):
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1        = 2 * P - L
    S1        = 2 * P - H
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1, width_pct=(upper_cpr - lower_cpr) / P * 100)


# ─── EXIT SIMULATION (daily bar, conservative) ────────────────────────────────
def simulate_exit(H, L, C, entry, sl, target, direction):
    """
    Conservative exit: if SL level is touched, assume SL hit first.
    Only credit TARGET if SL level was NOT touched.
    """
    if direction == "LONG":
        if L <= sl:
            return "SL", sl
        if H >= target:
            return "TARGET", target
    else:  # SHORT
        if H >= sl:
            return "SL", sl
        if L <= target:
            return "TARGET", target
    return "EOD", C


# ─── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(data, narrow_thresh):
    daily   = data["daily"]
    trading = sorted(daily.index.tolist())

    trades    = []
    month_pnl = defaultdict(float)
    stats     = defaultdict(int)

    for i, today in enumerate(trading):
        if i == 0:
            continue

        yesterday = trading[i - 1]
        try:
            pH = float(daily.loc[yesterday, "high"])
            pL = float(daily.loc[yesterday, "low"])
            pC = float(daily.loc[yesterday, "close"])
        except (KeyError, ValueError):
            continue

        cpr   = calc_cpr(pH, pL, pC)
        width = cpr["width_pct"]

        if width >= narrow_thresh:
            if width > WIDE_PCT:
                stats["wide_skipped"] += 1
            else:
                stats["neutral_skipped"] += 1
            continue

        stats["narrow_days"] += 1

        try:
            tH = float(daily.loc[today, "high"])
            tL = float(daily.loc[today, "low"])
            tC = float(daily.loc[today, "close"])
        except (KeyError, ValueError):
            stats["no_data"] += 1
            continue

        upper = cpr["upper_cpr"]
        lower = cpr["lower_cpr"]

        # Signal: today's close vs CPR (proxy for first 1H candle)
        if tC > upper:
            direction = "LONG"
            entry  = upper + SLIPPAGE
            sl     = lower - SLIPPAGE
            target = cpr["R1"]
        elif tC < lower:
            direction = "SHORT"
            entry  = lower - SLIPPAGE
            sl     = upper + SLIPPAGE
            target = cpr["S1"]
        else:
            stats["no_signal"] += 1
            continue

        # Sanity check
        if direction == "LONG"  and target <= entry: stats["no_signal"] += 1; continue
        if direction == "SHORT" and target >= entry: stats["no_signal"] += 1; continue

        outcome, exit_price = simulate_exit(tH, tL, tC, entry, sl, target, direction)

        # Apply slippage on SL/Target exits
        if outcome == "TARGET":
            exit_price = exit_price - SLIPPAGE if direction == "LONG" else exit_price + SLIPPAGE
        elif outcome == "SL":
            exit_price = exit_price + SLIPPAGE if direction == "LONG" else exit_price - SLIPPAGE

        raw_move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        net_pnl  = raw_move * PNL_PER_LOT * LOTS - COMMISSION
        win      = net_pnl > 0

        month_pnl[today.strftime("%Y-%m")] += net_pnl
        stats["wins" if win else "losses"] += 1

        trades.append(dict(
            date      = today,
            direction = direction,
            width_pct = round(width, 3),
            entry     = round(entry, 2),
            sl        = round(sl, 2),
            target    = round(target, 2),
            exit      = round(exit_price, 2),
            outcome   = outcome,
            raw_move  = round(raw_move, 2),
            pnl       = round(net_pnl, 2),
            win       = win,
        ))

    log.info("narrow=%d  wide=%d  neutral=%d  no_signal=%d  no_data=%d",
             stats["narrow_days"], stats["wide_skipped"],
             stats["neutral_skipped"], stats["no_signal"], stats["no_data"])
    return trades, dict(month_pnl)


# ─── REPORT ───────────────────────────────────────────────────────────────────
def _inr(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}₹{v/1e5:.2f}L" if abs(v) >= 1e5 else f"{sign}₹{v:,.0f}"


def print_report(trades, month_pnl, data, narrow_thresh, show_trades):
    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  MCX NICKEL CPR BACKTEST — Daily-bar simulation (conservative)")
    print(f"  Coverage   : {data['start']}  →  {data['end']}")
    print(f"  Data       : Kite continuous daily (1 year, stitched contracts)")
    print(f"  Lots/trade : {LOTS}  (1,500 kg)  |  ₹{PNL_PER_LOT:,}/₹1 move")
    print(f"  Commission : ₹{COMMISSION}  |  Slippage : ₹{SLIPPAGE}/kg per side")
    print(f"  Narrow CPR : < {narrow_thresh}%  |  Wide skip : > {WIDE_PCT}%")
    print(f"  Exit rule  : SL checked FIRST (conservative — understates performance)")
    print(f"  Margin     : ₹{MARGIN:,}/lot  |  Buffer on ₹{CAPITAL:,} : ₹{CAPITAL-MARGIN:,}")
    print(SEP)

    if not trades:
        print("  No trades. Try --narrow 0.50 or --clear-cache after login.py")
        print(SEP); return

    total  = len(trades)
    wins   = sum(1 for t in trades if t["win"])
    net    = sum(t["pnl"] for t in trades)
    avg    = net / total
    best   = max(t["pnl"] for t in trades)
    worst  = min(t["pnl"] for t in trades)
    tgts   = sum(1 for t in trades if t["outcome"] == "TARGET")
    sls    = sum(1 for t in trades if t["outcome"] == "SL")
    eods   = sum(1 for t in trades if t["outcome"] == "EOD")
    greens = sum(1 for v in month_pnl.values() if v >= 0)
    reds   = sum(1 for v in month_pnl.values() if v < 0)

    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    lw     = sum(1 for t in longs  if t["win"])
    sw     = sum(1 for t in shorts if t["win"])
    win_pnl  = [t["pnl"] for t in trades if t["win"]]
    loss_pnl = [t["pnl"] for t in trades if not t["win"]]
    avg_win  = sum(win_pnl)  / len(win_pnl)  if win_pnl  else 0
    avg_loss = sum(loss_pnl) / len(loss_pnl) if loss_pnl else 0

    print(f"\n  RESULTS  ({total} trades  |  {data['start']} → {data['end']})")
    print(f"  {'Win rate':<22}: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  {'Net P&L':<22}: {_inr(net)}")
    print(f"  {'Avg / trade':<22}: {_inr(avg)}")
    print(f"  {'Avg win':<22}: {_inr(avg_win)}")
    print(f"  {'Avg loss':<22}: {_inr(avg_loss)}")
    if avg_loss != 0:
        print(f"  {'Payoff ratio':<22}: {abs(avg_win/avg_loss):.2f}x")
    print(f"  {'Best trade':<22}: {_inr(best)}")
    print(f"  {'Worst trade':<22}: {_inr(worst)}")
    print(f"  {'Target hits':<22}: {tgts} ({tgts/total*100:.0f}%)")
    print(f"  {'SL hits':<22}: {sls}  ({sls/total*100:.0f}%)")
    print(f"  {'EOD exits':<22}: {eods} ({eods/total*100:.0f}%)")
    print(f"  {'Longs':<22}: {len(longs)} trades  {lw}W  ({lw/max(len(longs),1)*100:.0f}% win)")
    print(f"  {'Shorts':<22}: {len(shorts)} trades  {sw}W  ({sw/max(len(shorts),1)*100:.0f}% win)")
    print(f"  {'Green months':<22}: {greens}  |  Red: {reds}")

    # Monthly P&L
    if month_pnl:
        max_abs = max(abs(v) for v in month_pnl.values()) or 1
        print(f"\n  ── Monthly P&L {'─'*50}")
        for mo, pnl in sorted(month_pnl.items()):
            bar  = "█" * int(abs(pnl) / max_abs * 20)
            sign = "+" if pnl >= 0 else ""
            print(f"  {mo}  {sign}₹{pnl:>10,.0f}  {bar}")
        print(f"  {'─'*42}")
        print(f"  TOTAL          {_inr(net)}")

    # Trade log
    if show_trades:
        print(f"\n  ── All Trades {'─'*55}")
        print(f"  {'Date':<12} {'Dir':<6} {'Width%':<7} "
              f"{'Entry':>9} {'SL':>9} {'Target':>9} {'Exit':>9} "
              f"{'Move':>8} {'P&L':>9}  Result")
        print(f"  {'─'*90}")
        for t in trades:
            sign = "+" if t["pnl"] >= 0 else ""
            print(f"  {str(t['date']):<12} {t['direction']:<6} {t['width_pct']:<7.3f} "
                  f"{t['entry']:>9.2f} {t['sl']:>9.2f} {t['target']:>9.2f} "
                  f"{t['exit']:>9.2f} {t['raw_move']:>+8.2f} "
                  f"{sign}₹{abs(t['pnl']):>7,.0f}  {t['outcome']}")

    print(f"\n{SEP}")

    if trades:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "nickel_daily_backtest.csv")
        pd.DataFrame(trades).to_csv(csv_path, index=False)
        print(f"  Trades saved → {csv_path}")
        print(SEP)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="NICKEL CPR Daily-Bar Backtest")
    p.add_argument("--narrow",      type=float, default=NARROW_PCT)
    p.add_argument("--days",        type=int,   default=365)
    p.add_argument("--show-trades", action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    args = p.parse_args()

    kite = get_kite()
    data = load_data(kite, args.days, clear_cache=args.clear_cache)
    trades, month_pnl = run_backtest(data, args.narrow)
    print_report(trades, month_pnl, data, args.narrow, args.show_trades)


if __name__ == "__main__":
    main()
