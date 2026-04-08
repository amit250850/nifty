#!/usr/bin/env python3
"""
backtest_goldm_cpr.py
=====================
CPR (Central Pivot Range) strategy backtest on MCX GOLDM (Gold Mini) futures
using Zerodha Kite Connect API for historical data.

MCX GOLDM Specifications
------------------------
  Lot size   : 100 grams
  Quote unit : ₹ per 10 grams
  P&L/lot    : ₹10 per ₹1 price move  (100g ÷ 10g = 10 units)
  Session    : 09:00 – 23:30 IST  (we use 09:00–17:00 for CPR daytime session)
  Expiry     : Last day of each calendar month (MCX monthly)

  Example: Gold at ₹9,500/10g
    Lot value  = 10 × 9,500 = ₹95,000
    SPAN margin ≈ ₹8,000–10,000/lot  (changes with MCX circular)
    ₹50,000 budget → 4–5 lots comfortably

CPR Strategy (AAK Method) — BREAKOUT ONLY
------------------------------------------
  Narrow CPR (width% < 0.30%) → trending day
    1H candle closes ABOVE upper_cpr → LONG  at upper_cpr, SL=lower_cpr, Target=R1
    1H candle closes BELOW lower_cpr → SHORT at lower_cpr, SL=upper_cpr, Target=S1

  Wide CPR → SKIP (range mode proven -ve on Gold in GC=F 2-year backtest)
  Neutral  → SKIP

Data coverage note
------------------
  Kite Connect instruments endpoint returns only ACTIVE contracts + recent expireds
  (typically ~6–12 months back, not 2 full years). The code stitches whatever
  contracts are available, reports the actual coverage, and warns if < 2 years.
  On first run, data is downloaded and cached to goldm_kite_cache.pkl so
  subsequent runs are instant.

Setup
-----
  1. python login.py                         (refresh access_token in .env)
  2. python backtest_goldm_cpr.py            (default: 2 lots, last 730 days)
  3. python backtest_goldm_cpr.py --lots 1 --show-trades
  4. python backtest_goldm_cpr.py --clear-cache   (force re-download)

Requirements: kiteconnect python-dotenv pandas
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

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
LOT_GRAMS    = 100          # grams per GOLDM lot
QUOTE_UNIT   = 10           # MCX quotes gold in ₹ per 10 grams
PNL_PER_LOT  = LOT_GRAMS / QUOTE_UNIT   # = 10 → ₹10/lot per ₹1 price move

NARROW_PCT   = 0.30         # CPR width% threshold for breakout day
WIDE_PCT     = 0.55         # CPR width% threshold for range day

COMMISSION   = 60           # ₹ per lot, round-trip (Zerodha ₹20×2 + STT/tax ≈ ₹55–65)
SLIPPAGE     = 0.50         # ₹/10g per side (entry + exit each)

SESSION_START = dtime(9, 0)  # IST — first hourly bar
SESSION_END   = dtime(17, 0) # IST — daytime session close

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "goldm_kite_cache.pkl")


# ─── KITE AUTH ────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    load_dotenv()
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        sys.exit(
            "ERROR: KITE_API_KEY / KITE_ACCESS_TOKEN missing in .env.\n"
            "       Run  python login.py  first to refresh your session."
        )
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    log.info("Kite client initialised (api_key=%s…)", api_key[:6])
    return kite


# ─── INSTRUMENT DISCOVERY ─────────────────────────────────────────────────────
def get_goldm_contracts(kite: KiteConnect) -> pd.DataFrame:
    """
    Return all MCX GOLDM FUT contracts sorted by expiry.

    ⚠  Kite only returns currently active + recently expired contracts.
       For a full 2-year backtest you may only get ~6–12 months of coverage.
       Use --clear-cache after a few months to keep the dataset fresh.
    """
    log.info("Fetching MCX instruments list …")
    rows = kite.instruments("MCX")
    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    goldm = df[
        (df["name"] == "GOLDM") &
        (df["instrument_type"] == "FUT")
    ].sort_values("expiry").reset_index(drop=True)
    log.info("Found %d GOLDM contracts (expiries: %s → %s)",
             len(goldm),
             goldm["expiry"].min().date() if not goldm.empty else "N/A",
             goldm["expiry"].max().date() if not goldm.empty else "N/A")
    return goldm


# ─── RAW DATA FETCHERS ────────────────────────────────────────────────────────
def _fetch_daily(kite: KiteConnect, token: int,
                 from_dt: date, to_dt: date) -> pd.DataFrame:
    try:
        rows = kite.historical_data(
            token, from_dt, to_dt, interval="day",
            continuous=False, oi=False)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")
        return df
    except Exception as exc:
        log.warning("daily fetch failed  token=%s  %s→%s : %s",
                    token, from_dt, to_dt, exc)
        return pd.DataFrame()


def _fetch_hourly_chunked(kite: KiteConnect, token: int,
                          from_dt: date, to_dt: date) -> pd.DataFrame:
    """
    Kite historical_data allows max ~60 days per request for intraday intervals.
    Split into 59-day windows and concatenate.
    """
    all_rows = []
    cur = from_dt
    while cur <= to_dt:
        chunk_end = min(cur + timedelta(days=58), to_dt)
        try:
            rows = kite.historical_data(
                token, cur, chunk_end, interval="60minute",
                continuous=False, oi=False)
            if rows:
                all_rows.extend(rows)
        except Exception as exc:
            log.warning("hourly fetch failed  token=%s  %s→%s : %s",
                        token, cur, chunk_end, exc)
        cur = chunk_end + timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── CONTRACT STITCHING ───────────────────────────────────────────────────────
def build_continuous_data(kite: KiteConnect,
                          contracts: pd.DataFrame,
                          start: date, end: date) -> tuple:
    """
    Build a continuous GOLDM daily + hourly price series by rolling through
    front-month contracts.  Roll 3 calendar days before each contract's expiry.

    Returns: (daily_df, hourly_df)  — both deduplicated and sorted.
    """
    daily_parts  = []
    hourly_parts = []

    for i, row in contracts.iterrows():
        expiry = row["expiry"].date()
        token  = int(row["instrument_token"])
        sym    = row["tradingsymbol"]

        # Window this contract is "active" in our backtest
        if i == 0:
            c_start = start
        else:
            prev_expiry = contracts.loc[i - 1, "expiry"].date()
            c_start = prev_expiry - timedelta(days=3)   # overlap for smooth roll

        c_end   = min(expiry - timedelta(days=1), end)
        c_start = max(c_start, start)

        if c_start > end or c_end < start or c_start > c_end:
            continue

        log.info("  %-20s  %s → %s  (expiry %s)", sym, c_start, c_end, expiry)

        d = _fetch_daily(kite, token, c_start, c_end)
        if not d.empty:
            daily_parts.append(d)

        h = _fetch_hourly_chunked(kite, token, c_start, c_end)
        if not h.empty:
            hourly_parts.append(h)

    if not daily_parts:
        sys.exit(
            "ERROR: No GOLDM data downloaded.\n"
            "       Check that your access_token is valid (run login.py)\n"
            "       and that GOLDM contracts appear in kite.instruments('MCX')."
        )

    daily = (pd.concat(daily_parts)
               .sort_index()
               .pipe(lambda df: df[~df.index.duplicated(keep="last")]))

    hourly = (pd.concat(hourly_parts)
                .sort_values("date")
                .drop_duplicates(subset=["date"])
                .reset_index(drop=True))

    return daily, hourly


# ─── CACHE MANAGER ────────────────────────────────────────────────────────────
def load_data(kite: KiteConnect, start: date, end: date,
              clear_cache: bool = False) -> dict:
    if not clear_cache and os.path.exists(CACHE_FILE):
        log.info("Loading cached data from  %s", CACHE_FILE)
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        log.info("Cache: daily=%d bars, hourly=%d bars, "
                 "coverage=%s → %s",
                 len(cached["daily"]), len(cached["hourly"]),
                 cached.get("coverage_start", "?"),
                 cached.get("coverage_end", "?"))
        return cached

    log.info("Downloading fresh GOLDM data  %s → %s …", start, end)
    contracts = get_goldm_contracts(kite)
    if contracts.empty:
        sys.exit("ERROR: No GOLDM contracts returned by Kite. "
                 "Check MCX segment availability and your subscription.")

    # Only use contracts whose expiry falls within or just before our window
    relevant = contracts[
        contracts["expiry"].dt.date >= (start - timedelta(days=45))
    ].reset_index(drop=True)
    log.info("Stitching %d relevant contracts …", len(relevant))

    daily, hourly = build_continuous_data(kite, relevant, start, end)

    actual_start = min(daily.index)
    actual_end   = max(daily.index)
    days_covered = (actual_end - actual_start).days

    if days_covered < 180:
        log.warning(
            "⚠  Only %d days of data fetched (%s → %s).\n"
            "   Kite may not have expired contracts from 2 years ago.\n"
            "   Run with a shorter --days window or use --clear-cache "
            "after a few months to build coverage incrementally.",
            days_covered, actual_start, actual_end)
    else:
        log.info("Coverage: %d days  (%s → %s)", days_covered,
                 actual_start, actual_end)

    payload = {
        "daily":          daily,
        "hourly":         hourly,
        "req_start":      start,
        "req_end":        end,
        "coverage_start": actual_start,
        "coverage_end":   actual_end,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    log.info("Data cached → %s", CACHE_FILE)
    return payload


# ─── CPR MATH ────────────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    """
    Calculate CPR levels from previous day's OHLC.

    CRITICAL: TC can be < BC when close is in the lower half of the day's range
    (bear-close day). Always normalise: upper_cpr = max(TC, BC).
    Without this, SL ends up ABOVE entry for LONG trades — a silent fatal bug.
    """
    P  = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = 2 * P - BC
    upper_cpr = max(TC, BC)    # ← critical normalisation
    lower_cpr = min(TC, BC)
    R1 = 2 * P - L
    S1 = 2 * P - H
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1, width_pct=width_pct)


# ─── TRADE EXIT SCANNER ───────────────────────────────────────────────────────
def _exit_scan(bars: pd.DataFrame,
               entry: float, sl: float, target: float,
               direction: str) -> tuple:
    """
    Walk hourly bars to find the first SL or Target hit.
    Returns (outcome, exit_price).
    """
    for _, bar in bars.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        if direction == "LONG":
            if l <= sl:      return "SL",     sl
            if h >= target:  return "TARGET",  target
        else:
            if h >= sl:      return "SL",     sl
            if l <= target:  return "TARGET",  target

    # EOD exit at last bar's close
    eod_close = float(bars.iloc[-1]["close"]) if not bars.empty else entry
    return "EOD", eod_close


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(data: dict, lots: int,
                 narrow_thresh: float = NARROW_PCT) -> tuple:
    """
    BREAKOUT-only CPR strategy on GOLDM.

    Returns (trades_list, monthly_pnl_dict).
    """
    daily  = data["daily"]
    hourly = data["hourly"]
    start  = data["coverage_start"]
    end    = data["coverage_end"]

    trading_days = sorted([d for d in daily.index if start <= d <= end])
    log.info("Backtest window: %s → %s  (%d trading days)",
             start, end, len(trading_days))

    trades     = []
    month_pnl  = defaultdict(float)
    stats      = defaultdict(int)

    for i, today in enumerate(trading_days):
        if i == 0:
            continue    # need previous day for CPR

        yesterday = trading_days[i - 1]

        # ── Previous day OHLC for CPR ──────────────────────────────────────
        try:
            pH = float(daily.loc[yesterday, "high"])
            pL = float(daily.loc[yesterday, "low"])
            pC = float(daily.loc[yesterday, "close"])
        except (KeyError, ValueError, TypeError):
            continue

        cpr   = calc_cpr(pH, pL, pC)
        width = cpr["width_pct"]

        # ── Day classification ─────────────────────────────────────────────
        if width < narrow_thresh:
            kind = "NARROW"
        elif width > WIDE_PCT:
            stats["wide_skipped"] += 1
            continue    # range mode proven -ve on Gold
        else:
            stats["neutral_skipped"] += 1
            continue    # neutral — ambiguous

        stats["narrow_days"] += 1

        # ── Get today's intraday hourly bars (morning session only) ────────
        day_h = hourly[
            (hourly["date"].dt.date == today) &
            (hourly["date"].dt.time >= SESSION_START) &
            (hourly["date"].dt.time <  SESSION_END)
        ].reset_index(drop=True)

        if len(day_h) < 2:
            stats["no_intraday"] += 1
            continue

        upper  = cpr["upper_cpr"]
        lower  = cpr["lower_cpr"]
        R1     = cpr["R1"]
        S1     = cpr["S1"]

        # ── Signal from first hourly candle ────────────────────────────────
        first_close = float(day_h.iloc[0]["close"])

        direction = None
        if first_close > upper:
            direction = "LONG"
            entry  = upper + SLIPPAGE
            sl     = lower - SLIPPAGE
            target = R1
        elif first_close < lower:
            direction = "SHORT"
            entry  = lower - SLIPPAGE
            sl     = upper + SLIPPAGE
            target = S1

        if direction is None:
            stats["no_signal"] += 1
            continue

        # ── Guard: target must be on the correct side of entry after slippage ─
        # Degenerate case: CPR width ≈ 0, R1/S1 ends up inside slippage band.
        # e.g. width=0.000% → R1 < upper_cpr + slippage → skip, no edge.
        if direction == "LONG"  and target <= entry:
            stats["no_signal"] += 1
            continue
        if direction == "SHORT" and target >= entry:
            stats["no_signal"] += 1
            continue

        # ── Simulate exit ──────────────────────────────────────────────────
        outcome, exit_raw = _exit_scan(
            day_h.iloc[1:], entry, sl, target, direction)

        # Add slippage on exit (except EOD where we take close as-is)
        if outcome == "TARGET":
            exit_price = exit_raw - SLIPPAGE if direction == "LONG" \
                         else exit_raw + SLIPPAGE
        elif outcome == "SL":
            exit_price = exit_raw - SLIPPAGE if direction == "LONG" \
                         else exit_raw + SLIPPAGE
        else:
            exit_price = exit_raw   # EOD: market close

        # ── P&L ───────────────────────────────────────────────────────────
        raw_move = (exit_price - entry) if direction == "LONG" \
                   else (entry - exit_price)
        net_pnl  = raw_move * PNL_PER_LOT * lots - COMMISSION * lots
        win      = net_pnl > 0

        month_key = today.strftime("%Y-%m")
        month_pnl[month_key] += net_pnl

        if win:  stats["wins"] += 1
        else:    stats["losses"] += 1

        trades.append(dict(
            date      = today,
            kind      = kind,
            direction = direction,
            width_pct = round(width, 3),
            entry     = round(entry, 1),
            sl        = round(sl, 1),
            target    = round(target, 1),
            exit      = round(exit_price, 1),
            outcome   = outcome,
            raw_move  = round(raw_move, 1),
            pnl       = round(net_pnl, 0),
            win       = win,
            P         = round(cpr["P"], 1),
            R1        = round(R1, 1),
            S1        = round(S1, 1),
        ))

    log.info(
        "Signals: narrow=%d, wide_skipped=%d, neutral_skipped=%d, "
        "no_signal=%d, no_intraday=%d",
        stats["narrow_days"], stats["wide_skipped"],
        stats["neutral_skipped"], stats["no_signal"], stats["no_intraday"])

    return trades, dict(month_pnl)


# ─── REPORT ──────────────────────────────────────────────────────────────────
def _inr(v: float) -> str:
    sign = "+" if v >= 0 else ""
    if abs(v) >= 1e5:
        return f"{sign}₹{v/1e5:.2f}L"
    return f"{sign}₹{v:,.0f}"


def print_report(trades: list, month_pnl: dict, lots: int,
                 data: dict, show_trades: bool) -> None:

    SEP = "═" * 72
    cov_start = data.get("coverage_start", "?")
    cov_end   = data.get("coverage_end",   "?")

    print(f"\n{SEP}")
    print(f"  MCX GOLDM CPR BACKTEST  (Breakout-only, AAK method)")
    print(f"  Coverage  : {cov_start}  →  {cov_end}")
    print(f"  Lots/trade: {lots}  |  P&L/lot/₹1move: ₹{PNL_PER_LOT:.0f}")
    print(f"  Commission: ₹{COMMISSION}/lot  |  Slippage: ₹{SLIPPAGE}/side")
    print(f"  Narrow CPR: < {NARROW_PCT}%   (Wide/Neutral days skipped — proven -ve)")
    print(SEP)

    if not trades:
        print("  No trades generated.")
        print("  Check: (a) access_token fresh  (b) GOLDM data coverage >= 90 days")
        print(SEP)
        return

    total  = len(trades)
    wins   = sum(1 for t in trades if t["win"])
    losses = total - wins
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

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  RESULTS  ({total} trades)")
    print(f"  {'Win rate':<18}: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  {'Net P&L':<18}: {_inr(net)}")
    print(f"  {'Avg / trade':<18}: {_inr(avg)}")
    print(f"  {'Best trade':<18}: {_inr(best)}")
    print(f"  {'Worst trade':<18}: {_inr(worst)}")
    print(f"  {'Target hits':<18}: {tgts} ({tgts/total*100:.0f}%)")
    print(f"  {'SL hits':<18}: {sls}  ({sls/total*100:.0f}%)")
    print(f"  {'EOD exits':<18}: {eods} ({eods/total*100:.0f}%)")
    print(f"  {'Longs':<18}: {len(longs)} trades  {lw}W ({lw/max(len(longs),1)*100:.0f}% win)")
    print(f"  {'Shorts':<18}: {len(shorts)} trades  {sw}W ({sw/max(len(shorts),1)*100:.0f}% win)")
    print(f"  {'Green months':<18}: {greens}  |  Red months: {reds}")

    # ── Budget check ─────────────────────────────────────────────────────────
    gold_price = 9500   # approximate ₹/10g
    lot_val    = gold_price * (LOT_GRAMS / QUOTE_UNIT)
    margin_est = int(lot_val * 0.10)   # ~10% SPAN+exposure margin
    sl_risk    = NARROW_PCT / 100 * gold_price * PNL_PER_LOT   # per lot per trade
    budget_50k_lots = 50000 // margin_est
    print(f"\n  ── GOLDM Budget Guide (Gold ≈ ₹{gold_price}/10g) ───────────────")
    print(f"  Lot value       ≈ ₹{lot_val:,.0f}")
    print(f"  SPAN margin/lot ≈ ₹{margin_est:,.0f}  (changes daily, check Zerodha)")
    print(f"  ₹50,000 allows  ≈ {budget_50k_lots} lots")
    print(f"  SL risk/lot     ≈ ₹{sl_risk:.0f}/trade  (narrow CPR 0.30% of price)")
    print(f"  {lots}-lot risk      ≈ ₹{lots*sl_risk:.0f}/trade  "
          f"({lots*sl_risk/50000*100:.1f}% of ₹50K)  ✅")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    print(f"\n  ── Monthly P&L ──────────────────────────────────────────────────")
    for m, v in sorted(month_pnl.items()):
        bar = "█" * min(int(abs(v) / 300), 24)
        mk  = "✅" if v >= 0 else "❌"
        print(f"  {m}  {mk}  {_inr(v):>12}  {bar}")

    print(f"\n{SEP}")

    # ── Trade log ─────────────────────────────────────────────────────────────
    if show_trades:
        print(f"\n  {'Date':<12} {'Dir':<6} {'Width':>6}  "
              f"{'Entry':>8} {'SL':>8} {'Target':>8} {'Exit':>8}  "
              f"{'Out':<7} {'P&L':>10}")
        print("  " + "─" * 72)
        for t in trades:
            mk = "✅" if t["win"] else "❌"
            print(f"  {str(t['date']):<12} {t['direction']:<6} "
                  f"{t['width_pct']:>6.3f}  "
                  f"{t['entry']:>8.1f} {t['sl']:>8.1f} "
                  f"{t['target']:>8.1f} {t['exit']:>8.1f}  "
                  f"{t['outcome']:<7} {_inr(t['pnl']):>10}  {mk}")
        print(SEP)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    try:
        df = pd.DataFrame(trades)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "goldm_cpr_backtest.csv")
        df.to_csv(out, index=False)
        print(f"  Trade log → {out}")
    except Exception as exc:
        log.warning("CSV save failed: %s", exc)

    print(SEP + "\n")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="CPR backtest — MCX GOLDM via Zerodha Kite Connect",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lots",        type=int,   default=2,
                    help="Lots per trade (default 2; ≈ ₹16K–20K margin)")
    ap.add_argument("--days",        type=int,   default=730,
                    help="Look-back window in calendar days (default 730 = 2yr)")
    ap.add_argument("--narrow",      type=float, default=NARROW_PCT,
                    help=f"Narrow CPR threshold %% (default {NARROW_PCT})")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Ignore existing cache and re-download all data")
    ap.add_argument("--show-trades", action="store_true",
                    help="Print every trade row after the summary")
    args = ap.parse_args()

    end   = date.today()
    start = end - timedelta(days=args.days)

    print(f"\n{'═'*72}")
    print(f"  GOLDM CPR Backtest  |  Kite Connect  |  {start} → {end}")
    print(f"  Lots: {args.lots}  |  Narrow threshold: {args.narrow}%")
    print(f"{'═'*72}\n")

    kite = get_kite()
    data = load_data(kite, start, end, clear_cache=args.clear_cache)

    trades, month_pnl = run_backtest(data, args.lots, narrow_thresh=args.narrow)
    print_report(trades, month_pnl, args.lots, data,
                 show_trades=args.show_trades)


if __name__ == "__main__":
    main()
