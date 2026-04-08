#!/usr/bin/env python3
"""
backtest_goldpetal_cpr.py
=========================
CPR (Central Pivot Range) strategy backtest on MCX GOLDPETAL (Gold Petal) futures
using Zerodha Kite Connect API for historical data.

Why GOLDPETAL instead of GOLDM?
---------------------------------
  GOLDM margin at current gold prices (₹1,48,000+/10g) = ₹2,27,263 per lot.
  Zerodha has disabled MIS leverage on GOLDM — NRML = MIS = ₹2.27L.
  ₹50,000 budget cannot support even 1 lot of GOLDM.

  GOLDPETAL solves this:
    Lot size      : 1 gram
    Margin/lot    : ~₹2,200 (15% of ₹14,810 lot value)
    10 lots margin: ~₹22,000 → fits ₹50K comfortably ✅

MCX GOLDPETAL Specifications
------------------------------
  Lot size   : 1 gram
  Quote unit : ₹ per gram  (NOT per 10g — GOLDPETAL is ₹/gram)
  Tick size  : ₹1 per gram

  P&L per lot per ₹1 move = 1 gram × ₹1/gram = ₹1 per lot
  With 10 lots: ₹1 price move → ₹10 total P&L
  With 10 lots: ₹100 move    → ₹1,000 total P&L

  Example at gold ₹14,810/gram:
    Lot value     = 1 × ₹14,810 = ₹14,810
    NRML margin   ≈ ₹2,222 per lot  (15% — high due to gold bull run)
    10 lots margin≈ ₹22,220  ← fits ₹50K, leaves ₹27,780 buffer
    SL risk/trade ≈ 0.30% × ₹14,810 × 10 lots = ₹444  (0.9% of ₹50K) ✅

Commission note
---------------
  Zerodha charges per ORDER, not per lot, for commodity futures:
    Entry order   : ₹20 flat (all 10 lots together)
    Exit order    : ₹20 flat
    Exchange/STT  : ~₹10–15 per trade
    Total/trade   : ~₹50–55 flat  → ₹5–5.5 per lot equivalent

CPR Strategy (AAK Method) — BREAKOUT ONLY
------------------------------------------
  Narrow CPR (width% < 0.30%) → trending day
    1H candle closes ABOVE upper_cpr → LONG  at upper_cpr, SL=lower_cpr, Target=R1
    1H candle closes BELOW lower_cpr → SHORT at lower_cpr, SL=upper_cpr, Target=S1
  Wide/Neutral → SKIP (range mode proven -ve on Gold in 2yr GC=F backtest)

Data note
---------
  Same Kite Connect limitation as GOLDM — only active + recent contracts.
  Typically 3–6 months of GOLDPETAL historical data available.
  Cache stored in goldpetal_kite_cache.pkl.

Setup
-----
  1. python login.py
  2. python backtest_goldpetal_cpr.py              (default 10 lots)
  3. python backtest_goldpetal_cpr.py --lots 20
  4. python backtest_goldpetal_cpr.py --show-trades
  5. python backtest_goldpetal_cpr.py --clear-cache
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import date, timedelta, datetime, time as dtime
from collections import defaultdict

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import pandas as pd
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv pandas --break-system-packages")
    sys.exit(1)

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── GOLDPETAL SPECS ─────────────────────────────────────────────────────────
LOT_GRAMS    = 1            # 1 gram per lot
QUOTE_UNIT   = 1            # price quoted in ₹ per gram
PNL_PER_LOT  = LOT_GRAMS / QUOTE_UNIT   # = 1 → ₹1 per lot per ₹1 price move

NARROW_PCT   = 0.30         # CPR width% threshold for breakout day
WIDE_PCT     = 0.55

COMMISSION   = 55           # ₹ FLAT per trade (not per lot — Zerodha charges per order)
                            # ₹20 entry + ₹20 exit + ₹15 exchange/STT/GST
SLIPPAGE     = 0.25         # ₹/gram per side (smaller tick than GOLDM)

SESSION_START = dtime(9, 0)  # MCX morning session
SESSION_END   = dtime(17, 0)

CACHE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "goldpetal_kite_cache.pkl")


# ─── KITE AUTH ────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    load_dotenv()
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        sys.exit("ERROR: KITE_API_KEY / KITE_ACCESS_TOKEN not in .env. Run login.py first.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    log.info("Kite client ready (api_key=%s…)", api_key[:6])
    return kite


# ─── INSTRUMENT DISCOVERY ─────────────────────────────────────────────────────
def get_goldpetal_contracts(kite: KiteConnect) -> pd.DataFrame:
    """Fetch all MCX GOLDPETAL FUT contracts sorted by expiry."""
    log.info("Fetching MCX instruments …")
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    # GOLDPETAL instrument name in Kite is "GOLDPETAL"
    petal = df[
        (df["name"] == "GOLDPETAL") &
        (df["instrument_type"] == "FUT")
    ].sort_values("expiry").reset_index(drop=True)

    log.info("Found %d GOLDPETAL contracts (expiries: %s → %s)",
             len(petal),
             petal["expiry"].min().date() if not petal.empty else "N/A",
             petal["expiry"].max().date() if not petal.empty else "N/A")
    return petal


# ─── DATA FETCHERS ───────────────────────────────────────────────────────────
def _fetch_daily(kite, token, from_dt, to_dt) -> pd.DataFrame:
    try:
        rows = kite.historical_data(token, from_dt, to_dt, interval="day",
                                    continuous=False, oi=False)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.set_index("date")
    except Exception as exc:
        log.warning("daily fetch token=%s: %s", token, exc)
        return pd.DataFrame()


def _fetch_hourly_chunked(kite, token, from_dt, to_dt) -> pd.DataFrame:
    """Kite allows max ~60 days per intraday request — chunk it."""
    all_rows = []
    cur      = from_dt
    while cur <= to_dt:
        chunk_end = min(cur + timedelta(days=58), to_dt)
        try:
            rows = kite.historical_data(token, cur, chunk_end,
                                        interval="60minute",
                                        continuous=False, oi=False)
            if rows:
                all_rows.extend(rows)
        except Exception as exc:
            log.warning("hourly fetch token=%s %s→%s: %s", token, cur, chunk_end, exc)
        cur = chunk_end + timedelta(days=1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── CONTRACT STITCHING ───────────────────────────────────────────────────────
def build_continuous_data(kite, contracts, start, end):
    """
    Stitch monthly GOLDPETAL contracts into one continuous series.
    Rolls 3 days before each expiry.
    """
    daily_parts  = []
    hourly_parts = []

    for i, row in contracts.iterrows():
        expiry = row["expiry"].date()
        token  = int(row["instrument_token"])
        sym    = row["tradingsymbol"]

        c_start = start if i == 0 \
                  else contracts.loc[i - 1, "expiry"].date() - timedelta(days=3)
        c_end   = min(expiry - timedelta(days=1), end)
        c_start = max(c_start, start)

        if c_start > end or c_end < start or c_start > c_end:
            continue

        log.info("  %-22s  %s → %s  (expiry %s)", sym, c_start, c_end, expiry)

        d = _fetch_daily(kite, token, c_start, c_end)
        if not d.empty:
            daily_parts.append(d)

        h = _fetch_hourly_chunked(kite, token, c_start, c_end)
        if not h.empty:
            hourly_parts.append(h)

    if not daily_parts:
        sys.exit(
            "ERROR: No GOLDPETAL data downloaded.\n"
            "       Verify access_token is fresh (run login.py) and\n"
            "       GOLDPETAL contracts appear in kite.instruments('MCX')."
        )

    daily  = (pd.concat(daily_parts).sort_index()
               .pipe(lambda df: df[~df.index.duplicated(keep="last")]))
    hourly = (pd.concat(hourly_parts).sort_values("date")
               .drop_duplicates(subset=["date"]).reset_index(drop=True))
    return daily, hourly


# ─── CACHE ────────────────────────────────────────────────────────────────────
def load_data(kite, start, end, clear_cache=False):
    if not clear_cache and os.path.exists(CACHE_FILE):
        log.info("Loading cache: %s", CACHE_FILE)
        with open(CACHE_FILE, "rb") as f:
            c = pickle.load(f)
        log.info("Cache: daily=%d, hourly=%d  coverage=%s→%s",
                 len(c["daily"]), len(c["hourly"]),
                 c.get("coverage_start", "?"), c.get("coverage_end", "?"))
        return c

    log.info("Downloading GOLDPETAL data  %s → %s …", start, end)
    contracts = get_goldpetal_contracts(kite)
    if contracts.empty:
        sys.exit("ERROR: No GOLDPETAL contracts returned by Kite.")

    relevant = contracts[
        contracts["expiry"].dt.date >= (start - timedelta(days=45))
    ].reset_index(drop=True)
    log.info("Stitching %d relevant contracts …", len(relevant))

    daily, hourly = build_continuous_data(kite, relevant, start, end)

    actual_start = min(daily.index)
    actual_end   = max(daily.index)
    days_covered = (actual_end - actual_start).days

    if days_covered < 90:
        log.warning("⚠  Only %d days of data (%s → %s). "
                    "Kite may not have older expired contracts.",
                    days_covered, actual_start, actual_end)
    else:
        log.info("Coverage: %d days  (%s → %s)", days_covered, actual_start, actual_end)

    payload = dict(daily=daily, hourly=hourly,
                   coverage_start=actual_start, coverage_end=actual_end)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    log.info("Cached → %s", CACHE_FILE)
    return payload


# ─── CPR MATH ────────────────────────────────────────────────────────────────
def calc_cpr(H, L, C):
    """
    Standard CPR from previous day's OHLC.
    upper_cpr = max(TC, BC) always — critical normalisation to prevent
    the SL-above-entry bug on bear-close days.
    """
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1        = 2 * P - L
    R2        = P + (H - L)
    S1        = 2 * P - H
    S2        = P - (H - L)
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, R2=R2, S1=S1, S2=S2, width_pct=width_pct)


# ─── EXIT SCANNER ────────────────────────────────────────────────────────────
def _exit_scan(bars, entry, sl, target, direction):
    for _, bar in bars.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        if direction == "LONG":
            if l <= sl:      return "SL",     sl
            if h >= target:  return "TARGET",  target
        else:
            if h >= sl:      return "SL",     sl
            if l <= target:  return "TARGET",  target
    eod = float(bars.iloc[-1]["close"]) if not bars.empty else entry
    return "EOD", eod


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(data, lots, narrow_thresh=NARROW_PCT):
    daily  = data["daily"]
    hourly = data["hourly"]
    start  = data["coverage_start"]
    end    = data["coverage_end"]

    trading_days = sorted([d for d in daily.index if start <= d <= end])
    log.info("Backtest: %s → %s  (%d trading days)", start, end, len(trading_days))

    trades    = []
    month_pnl = defaultdict(float)
    stats     = defaultdict(int)

    for i, today in enumerate(trading_days):
        if i == 0:
            continue

        yesterday = trading_days[i - 1]
        try:
            pH = float(daily.loc[yesterday, "high"])
            pL = float(daily.loc[yesterday, "low"])
            pC = float(daily.loc[yesterday, "close"])
        except (KeyError, ValueError, TypeError):
            continue

        cpr   = calc_cpr(pH, pL, pC)
        width = cpr["width_pct"]

        # Day classification
        if width < narrow_thresh:
            kind = "NARROW"
        elif width > WIDE_PCT:
            stats["wide_skipped"] += 1
            continue
        else:
            stats["neutral_skipped"] += 1
            continue

        stats["narrow_days"] += 1

        # Today's hourly bars — morning session only
        day_h = hourly[
            (hourly["date"].dt.date == today) &
            (hourly["date"].dt.time >= SESSION_START) &
            (hourly["date"].dt.time <  SESSION_END)
        ].reset_index(drop=True)

        if len(day_h) < 2:
            stats["no_intraday"] += 1
            continue

        upper = cpr["upper_cpr"]
        lower = cpr["lower_cpr"]
        R1    = cpr["R1"]
        S1    = cpr["S1"]

        # Signal from first hourly candle
        first_close = float(day_h.iloc[0]["close"])
        direction = entry = sl = target = None

        if first_close > upper:
            direction = "LONG"
            entry     = upper + SLIPPAGE
            sl        = lower - SLIPPAGE
            target    = R1
        elif first_close < lower:
            direction = "SHORT"
            entry     = lower - SLIPPAGE
            sl        = upper + SLIPPAGE
            target    = S1

        if direction is None:
            stats["no_signal"] += 1
            continue

        # Guard: target must be on correct side of entry after slippage
        if direction == "LONG"  and target <= entry:
            stats["no_signal"] += 1
            continue
        if direction == "SHORT" and target >= entry:
            stats["no_signal"] += 1
            continue

        # Simulate exit
        outcome, exit_raw = _exit_scan(day_h.iloc[1:], entry, sl, target, direction)

        # Exit slippage (except EOD)
        if outcome in ("TARGET", "SL"):
            exit_price = exit_raw - SLIPPAGE if direction == "LONG" \
                         else exit_raw + SLIPPAGE
        else:
            exit_price = exit_raw

        # P&L
        # COMMISSION is flat per trade (not per lot) — Zerodha per-order model
        raw_move = (exit_price - entry) if direction == "LONG" \
                   else (entry - exit_price)
        net_pnl  = raw_move * PNL_PER_LOT * lots - COMMISSION
        win      = net_pnl > 0

        month_pnl[today.strftime("%Y-%m")] += net_pnl
        if win: stats["wins"] += 1
        else:   stats["losses"] += 1

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
            P         = round(cpr["P"], 2),
            R1        = round(R1, 2),
            S1        = round(S1, 2),
        ))

    log.info("narrow=%d  wide_skip=%d  neutral_skip=%d  no_signal=%d  no_intraday=%d",
             stats["narrow_days"], stats["wide_skipped"], stats["neutral_skipped"],
             stats["no_signal"],   stats["no_intraday"])

    return trades, dict(month_pnl)


# ─── REPORT ──────────────────────────────────────────────────────────────────
def _inr(v):
    sign = "+" if v >= 0 else ""
    if abs(v) >= 1e5: return f"{sign}₹{v/1e5:.2f}L"
    return f"{sign}₹{v:,.0f}"


def print_report(trades, month_pnl, lots, data, show_trades):
    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  MCX GOLDPETAL CPR BACKTEST  (Breakout-only, AAK method)")
    print(f"  Coverage  : {data['coverage_start']}  →  {data['coverage_end']}")
    print(f"  Lots/trade: {lots}  |  P&L: ₹{PNL_PER_LOT:.0f}/lot/₹1move  "
          f"→  ₹{PNL_PER_LOT * lots:.0f} per ₹1 move total")
    print(f"  Commission: ₹{COMMISSION} FLAT per trade (Zerodha per-order model)")
    print(f"  Slippage  : ₹{SLIPPAGE}/gram per side")
    print(f"  Narrow CPR: < {NARROW_PCT}%   (Wide/Neutral skipped — proven -ve on Gold)")
    print(SEP)

    if not trades:
        print("  No trades. Check login.py and data coverage.")
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
    reds   = sum(1 for v in month_pnl.values() if v <  0)

    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    lw     = sum(1 for t in longs  if t["win"])
    sw     = sum(1 for t in shorts if t["win"])

    # Avg win/loss
    win_pnl  = [t["pnl"] for t in trades if t["win"]]
    loss_pnl = [t["pnl"] for t in trades if not t["win"]]
    avg_win  = sum(win_pnl)  / len(win_pnl)  if win_pnl  else 0
    avg_loss = sum(loss_pnl) / len(loss_pnl) if loss_pnl else 0

    print(f"\n  RESULTS  ({total} trades  |  {data['coverage_start']} → {data['coverage_end']})")
    print(f"  {'Win rate':<20}: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  {'Net P&L':<20}: {_inr(net)}")
    print(f"  {'Avg / trade':<20}: {_inr(avg)}")
    print(f"  {'Avg win':<20}: {_inr(avg_win)}")
    print(f"  {'Avg loss':<20}: {_inr(avg_loss)}")
    print(f"  {'Win/Loss ratio':<20}: {abs(avg_win/avg_loss):.1f}x"
          if avg_loss != 0 else "")
    print(f"  {'Best trade':<20}: {_inr(best)}")
    print(f"  {'Worst trade':<20}: {_inr(worst)}")
    print(f"  {'Target hits':<20}: {tgts} ({tgts/total*100:.0f}%)")
    print(f"  {'SL hits':<20}: {sls}  ({sls/total*100:.0f}%)")
    print(f"  {'EOD exits':<20}: {eods} ({eods/total*100:.0f}%)")
    print(f"  {'Longs':<20}: {len(longs)} trades  {lw}W "
          f"({lw/max(len(longs),1)*100:.0f}% win)")
    print(f"  {'Shorts':<20}: {len(shorts)} trades  {sw}W "
          f"({sw/max(len(shorts),1)*100:.0f}% win)")
    print(f"  {'Green months':<20}: {greens}  |  Red: {reds}")

    # ── Budget check (real numbers) ────────────────────────────────────────
    gold_gram   = 14810       # ₹/gram (approx current)
    lot_value   = gold_gram * LOT_GRAMS               # ₹14,810 per lot
    margin_lot  = lot_value * 0.15                    # 15% MCX margin
    margin_10   = margin_lot * lots
    sl_risk_lot = NARROW_PCT / 100 * gold_gram * PNL_PER_LOT
    sl_risk_all = sl_risk_lot * lots + COMMISSION

    print(f"\n  ── GOLDPETAL Budget Guide (Gold ≈ ₹{gold_gram:,}/gram) ──────────────")
    print(f"  Lot value       : ₹{lot_value:,}  (1 gram)")
    print(f"  NRML margin/lot : ₹{margin_lot:,.0f}  (15% — check Zerodha for live figure)")
    print(f"  {lots} lots margin    : ₹{margin_10:,.0f}")
    print(f"  ₹50K headroom   : ₹{50000 - margin_10:,.0f}  ✅")
    print(f"  SL risk/trade   : ₹{sl_risk_all:,.0f}  "
          f"({sl_risk_all/50000*100:.1f}% of ₹50K)  ✅")
    print(f"  P&L per ₹1 move : ₹{PNL_PER_LOT * lots:.0f}  "
          f"(₹{PNL_PER_LOT:.0f}/lot × {lots} lots)")
    print(f"  P&L per ₹100 move: ₹{PNL_PER_LOT * lots * 100:,.0f}")

    # ── Monthly P&L ───────────────────────────────────────────────────────
    print(f"\n  ── Monthly P&L ──────────────────────────────────────────────────")
    for m, v in sorted(month_pnl.items()):
        bar = "█" * min(int(abs(v) / 200), 26)
        mk  = "✅" if v >= 0 else "❌"
        print(f"  {m}  {mk}  {_inr(v):>12}  {bar}")

    print(f"\n{SEP}")

    # ── Per-trade log ─────────────────────────────────────────────────────
    if show_trades:
        print(f"\n  {'Date':<12} {'Dir':<6} {'Width':>6}  "
              f"{'Entry':>8} {'SL':>8} {'Target':>8} {'Exit':>8}  "
              f"{'Out':<7} {'Move':>7} {'P&L':>10}")
        print("  " + "─" * 72)
        for t in trades:
            mk = "✅" if t["win"] else "❌"
            print(f"  {str(t['date']):<12} {t['direction']:<6} "
                  f"{t['width_pct']:>6.3f}  "
                  f"{t['entry']:>8.2f} {t['sl']:>8.2f} "
                  f"{t['target']:>8.2f} {t['exit']:>8.2f}  "
                  f"{t['outcome']:<7} "
                  f"{t['raw_move']:>+7.2f} "
                  f"{_inr(t['pnl']):>10}  {mk}")
        print(SEP)

    # ── CSV export ────────────────────────────────────────────────────────
    try:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "goldpetal_cpr_backtest.csv")
        pd.DataFrame(trades).to_csv(out, index=False)
        print(f"  Trades → {out}")
    except Exception as exc:
        log.warning("CSV save failed: %s", exc)

    print(SEP + "\n")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="MCX GOLDPETAL CPR Backtest via Zerodha Kite Connect")
    ap.add_argument("--lots",        type=int,   default=10,
                    help="Lots per trade (default 10; fits ₹50K budget)")
    ap.add_argument("--days",        type=int,   default=730,
                    help="Look-back days (default 730)")
    ap.add_argument("--narrow",      type=float, default=NARROW_PCT,
                    help=f"Narrow threshold %% (default {NARROW_PCT})")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Re-download all data (ignore cache)")
    ap.add_argument("--show-trades", action="store_true",
                    help="Print every trade row")
    args = ap.parse_args()

    end   = date.today()
    start = end - timedelta(days=args.days)

    print(f"\n{'═'*72}")
    print(f"  GOLDPETAL CPR Backtest  |  Kite Connect  |  {start} → {end}")
    print(f"  Lots: {args.lots}  |  P&L/move: ₹{PNL_PER_LOT * args.lots:.0f} per ₹1")
    print(f"  (₹1 price move = ₹{PNL_PER_LOT:.0f}/lot × {args.lots} lots = "
          f"₹{PNL_PER_LOT * args.lots:.0f} total)")
    print(f"{'═'*72}\n")

    kite   = get_kite()
    data   = load_data(kite, start, end, clear_cache=args.clear_cache)
    trades, month_pnl = run_backtest(data, args.lots, narrow_thresh=args.narrow)
    print_report(trades, month_pnl, args.lots, data, show_trades=args.show_trades)


if __name__ == "__main__":
    main()
