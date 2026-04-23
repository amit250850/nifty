#!/usr/bin/env python3
"""
backtest_mcx_cpr.py
===================
Unified MCX commodity CPR (Central Pivot Range) backtest — AAK method.
Supports: GOLDPETAL, GOLDGUINEA, SILVERM  (bread & butter symbols)

CPR Strategy (Breakout-only)
-----------------------------
  Narrow CPR day (width% < threshold) → trending:
    First 1H candle closes ABOVE upper_cpr → LONG  @ upper+slip, SL=lower-slip, T=R1
    First 1H candle closes BELOW lower_cpr → SHORT @ lower-slip, SL=upper+slip, T=S1
  Wide/neutral day → SKIP (choppy, no edge)

  Narrow CPR → price is coiling; breakout tends to follow through to R1/S1.
  Wide CPR   → price already moved; mean-reversion likely, breakout fails.

Usage
-----
  python backtest_mcx_cpr.py --symbol GOLDPETAL
  python backtest_mcx_cpr.py --symbol GOLDGUINEA
  python backtest_mcx_cpr.py --symbol SILVERM
  python backtest_mcx_cpr.py --symbol CRUDEOILM
  python backtest_mcx_cpr.py --symbol NICKEL
  python backtest_mcx_cpr.py --symbol CRUDEOILM  --narrow 0.40 --show-trades
  python backtest_mcx_cpr.py --symbol NICKEL      --narrow 0.30 --days 360

MCX Commodity Specs (budget guide — Zerodha NRML margins Apr 2026)
--------------------------------------------------------------------
  GOLDPETAL  : 1g   lot, ₹1/lot/₹1 move    | margin ~₹2,200/lot | 10 lots = ₹22,000  ✅
  GOLDGUINEA : 8g   lot, ₹1/lot/₹1 move    | margin ~₹17,800/lot |  2 lots = ₹35,600  ✅
  SILVERMICRO: 1kg  lot, ₹1/lot/₹1 move    | margin ~₹25,000/lot |  1 lot  = ₹25,000  ✅
  SILVERM    : 5kg  lot, ₹5/lot/₹1 move    | margin ~₹1.25L/lot  |  1 lot  = ₹1.25L   ❌ silver tripled
  CRUDEOILM  : 10bbl lot, ₹10/lot/₹1 move  | margin ~₹28,600/lot |  1 lot  = ₹28,600  ✅
  NICKEL     : 1500kg lot, ₹1500/lot/₹1    | margin ~₹48,953/lot |  1 lot  = ₹49,000  ⚠️ tight
  ALUMINIUM  : 5MT  lot — margin ₹1.73L/lot → ❌ needs ₹1.8L
  ZINC       : 5MT  lot — margin ₹1.56L/lot → ❌ needs ₹1.6L
  LEAD       : 5MT  lot — margin ₹72K/lot   → ❌ needs ₹73K
  COPPER     : 2.5MT lot — margin ₹3.05L/lot→ ❌ needs ₹3.1L
  NATURALGAS : 1250 MMBTU — margin ₹68K/lot → ❌ needs ₹68K
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import date, timedelta, time as dtime
from collections import defaultdict

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import pandas as pd
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv pandas")
    sys.exit(1)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── COMMODITY CONFIGS ────────────────────────────────────────────────────────
# pnl_per_lot : ₹ P&L per 1 lot when price moves ₹1 in quote units
# slippage    : ₹ per quote-unit per side (e.g. ₹/gram for GOLDPETAL, ₹/kg for SILVERM)
# narrow_pct  : CPR width% threshold for "trending day" signal
# wide_pct    : CPR width% above which day is skipped as range/choppy
# ref_price   : approximate current price per quote-unit (for budget display only)
# margin_pct  : approximate NRML margin as % of lot value (SEBI SPAN+EM estimate)
# default_lots: default number of lots per trade
# expiry_type : "monthly" | "bimonthly"

CONFIGS = {
    "GOLDPETAL": dict(
        kite_name    = "GOLDPETAL",
        lot_unit     = "1g",
        quote_desc   = "₹ per gram",
        pnl_per_lot  = 1,       # 1g × ₹1/g = ₹1
        commission   = 65,
        slippage     = 0.25,    # ₹/gram per side
        narrow_pct   = 0.30,    # validated on gold in backtest_goldguinea_cpr.py
        wide_pct     = 0.55,
        ref_price    = 9_000,   # ₹/gram
        margin_pct   = 0.15,
        default_lots = 10,
        expiry_type  = "monthly",
    ),
    "GOLDGUINEA": dict(
        kite_name    = "GOLDGUINEA",
        lot_unit     = "8g",
        quote_desc   = "₹ per 8g lot",
        pnl_per_lot  = 1,       # price quoted per 8g lot — ₹1 price move = ₹1 P&L
        commission   = 65,
        slippage     = 4.00,    # ₹/lot per side (₹0.50/gram × 8g)
        narrow_pct   = 0.35,
        wide_pct     = 0.55,
        ref_price    = 72_000,  # ₹/8g lot (8 × ₹9,000/gram approx)
        margin_pct   = 0.15,
        default_lots = 3,
        expiry_type  = "bimonthly",
    ),
    "SILVERM": dict(
        kite_name    = "SILVERM",
        lot_unit     = "5kg",
        quote_desc   = "₹ per kg",
        pnl_per_lot  = 5,       # 5kg × ₹1/kg = ₹5 per lot per ₹1 move
        commission   = 65,
        slippage     = 10.0,    # ₹/kg per side (wider spread than gold)
        narrow_pct   = 0.40,    # silver more volatile than gold → wider threshold
        wide_pct     = 0.90,
        ref_price    = 90_000,  # ₹/kg
        margin_pct   = 0.10,    # ~10% SPAN+EM
        default_lots = 1,
        expiry_type  = "monthly",
    ),
    # ── Within ₹50K budget ─────────────────────────────────────────────────────
    "CRUDEOILM": dict(
        kite_name    = "CRUDEOILM",
        lot_unit     = "10bbl",
        quote_desc   = "₹ per barrel",
        pnl_per_lot  = 10,      # 10 barrels × ₹1/barrel = ₹10 per lot per ₹1 move
        commission   = 65,
        slippage     = 1.0,     # ₹1/barrel per side (1 tick)
        narrow_pct   = 0.50,    # crude more volatile than gold; tune via backtest
        wide_pct     = 1.50,
        ref_price    = 8_500,   # ₹/barrel (approx Apr 2026)
        margin_pct   = 3.37,    # ≈ ₹28,607 on ₹8,500 ref; display only
        default_lots = 1,
        expiry_type  = "monthly",
    ),
    "NICKEL": dict(
        kite_name    = "NICKEL",
        lot_unit     = "250kg",
        quote_desc   = "₹ per kg",
        pnl_per_lot  = 250,     # 250 kg × ₹1/kg = ₹250 per lot per ₹1 move (MCX reduced lot)
        commission   = 65,
        slippage     = 0.20,    # ₹/kg per side (2 ticks)
        narrow_pct   = 0.40,
        wide_pct     = 1.20,
        ref_price    = 1_750,   # ₹/kg (approx Apr 2026)
        margin_pct   = 1.87,    # ≈ ₹48,953; display only (Zerodha NRML May 2026)
        default_lots = 1,
        expiry_type  = "monthly",
    ),
    # ── SILVERMIC — 1 kg micro lot (fits ₹87K account; Kite name = SILVERMIC) ──
    "SILVERMIC": dict(
        kite_name    = "SILVERMIC",
        lot_unit     = "1kg",
        quote_desc   = "₹ per kg",
        pnl_per_lot  = 1,       # 1kg × ₹1/kg = ₹1 per lot per ₹1 move
        commission   = 65,
        slippage     = 10.0,    # ₹/kg per side (same spread as SILVERM)
        narrow_pct   = 0.45,    # widened from 0.40 — backtest: +₹1.40L vs +₹1.22L, 9/0 green
        wide_pct     = 0.90,
        ref_price    = 250_000, # ₹/kg (Apr 2026 — silver at ~₹2.5L/kg)
        margin_pct   = 0.25,    # ~25% NRML ≈ ₹62,500/lot at current price ✅
        default_lots = 1,
        expiry_type  = "monthly",
    ),
    # ── NATURALGAS — 1,250 MMBTU full lot (margin ~₹68K, fits ₹87K tight) ──────
    "NATURALGAS": dict(
        kite_name    = "NATURALGAS",
        lot_unit     = "1250 MMBTU",
        quote_desc   = "₹ per MMBTU",
        pnl_per_lot  = 1_250,   # 1,250 MMBTU × ₹1/MMBTU = ₹1,250 per lot per ₹1 move
        commission   = 65,
        slippage     = 0.50,    # ₹/MMBTU per side (1-2 ticks)
        narrow_pct   = 0.50,    # gas is more volatile than metals — start wide
        wide_pct     = 1.50,
        ref_price    = 255,     # ₹/MMBTU (Apr 2026)
        margin_pct   = 21.0,    # ~21% NRML ≈ ₹68K/lot at current price
        default_lots = 1,
        expiry_type  = "monthly",
    ),
    # ── NATGASMINI — 250 MMBTU mini lot (margin ~₹14K, can run 4 lots on ₹87K) ─
    "NATGASMINI": dict(
        kite_name    = "NATGASMINI",
        lot_unit     = "250 MMBTU",
        quote_desc   = "₹ per MMBTU",
        pnl_per_lot  = 250,     # 250 MMBTU × ₹1/MMBTU = ₹250 per lot per ₹1 move
        commission   = 65,
        slippage     = 0.50,    # ₹/MMBTU per side
        narrow_pct   = 0.50,    # same as NATURALGAS — identical underlying
        wide_pct     = 1.50,
        ref_price    = 255,     # ₹/MMBTU (Apr 2026)
        margin_pct   = 21.0,    # ~21% NRML ≈ ₹14K/lot → 4 lots = ₹56K on ₹87K
        default_lots = 4,
        expiry_type  = "monthly",
    ),
}

# MCX day session (same for all above)
SESSION_START = dtime(9, 0)
SESSION_END   = dtime(17, 0)

CACHE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── KITE AUTH ────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    load_dotenv()
    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        sys.exit("ERROR: KITE_API_KEY / KITE_ACCESS_TOKEN not in .env — run login.py first.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    log.info("Kite ready (api_key=%s…)", api_key[:6])
    return kite


# ─── INSTRUMENT DISCOVERY ─────────────────────────────────────────────────────
def get_contracts(kite: KiteConnect, kite_name: str) -> pd.DataFrame:
    log.info("Fetching MCX instruments …")
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    out = df[
        (df["name"] == kite_name) &
        (df["instrument_type"] == "FUT")
    ].sort_values("expiry").reset_index(drop=True)
    log.info("Found %d %s contracts (expiries: %s → %s)",
             len(out), kite_name,
             out["expiry"].min().date() if not out.empty else "N/A",
             out["expiry"].max().date() if not out.empty else "N/A")
    return out


# ─── DATA FETCHERS ────────────────────────────────────────────────────────────
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
    all_rows = []
    cur = from_dt
    while cur <= to_dt:
        chunk_end = min(cur + timedelta(days=58), to_dt)
        try:
            rows = kite.historical_data(token, cur, chunk_end,
                                        interval="60minute",
                                        continuous=False, oi=False)
            if rows:
                all_rows.extend(rows)
        except Exception as exc:
            log.warning("hourly fetch %s→%s: %s", cur, chunk_end, exc)
        cur = chunk_end + timedelta(days=1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── CONTRACT STITCHING ───────────────────────────────────────────────────────
def build_continuous_data(kite, contracts, start, end):
    """Stitch contracts rolling 3 days before each expiry."""
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

        log.info("  %-26s  %s → %s  (expiry %s)", sym, c_start, c_end, expiry)

        d = _fetch_daily(kite, token, c_start, c_end)
        if not d.empty:
            daily_parts.append(d)

        h = _fetch_hourly_chunked(kite, token, c_start, c_end)
        if not h.empty:
            hourly_parts.append(h)

    if not daily_parts:
        sys.exit(
            f"ERROR: No data downloaded.\n"
            f"  • Verify access_token is fresh (run login.py)\n"
            f"  • Check contracts exist: python -c \"from kiteconnect import KiteConnect; "
            f"import os; from dotenv import load_dotenv; load_dotenv(); "
            f"k=KiteConnect(api_key=os.getenv('KITE_API_KEY')); "
            f"k.set_access_token(os.getenv('KITE_ACCESS_TOKEN')); "
            f"[print(x['name'], x['tradingsymbol'], x['expiry']) "
            f"for x in k.instruments('MCX') if x['instrument_type']=='FUT'][:5]\""
        )

    daily  = (pd.concat(daily_parts).sort_index()
               .pipe(lambda df: df[~df.index.duplicated(keep="last")]))
    hourly = (pd.concat(hourly_parts).sort_values("date")
               .drop_duplicates(subset=["date"]).reset_index(drop=True))
    return daily, hourly


# ─── CACHE ────────────────────────────────────────────────────────────────────
def load_data(kite, symbol, cfg, start, end, clear_cache=False):
    cache_file = os.path.join(CACHE_DIR, f"{symbol.lower()}_mcx_cpr_cache.pkl")

    if not clear_cache and os.path.exists(cache_file):
        log.info("Loading cache: %s", cache_file)
        with open(cache_file, "rb") as f:
            c = pickle.load(f)
        log.info("Cache: daily=%d, hourly=%d  %s→%s",
                 len(c["daily"]), len(c["hourly"]),
                 c.get("coverage_start", "?"), c.get("coverage_end", "?"))
        return c

    log.info("Downloading %s data  %s → %s …", symbol, start, end)
    contracts = get_contracts(kite, cfg["kite_name"])
    if contracts.empty:
        sys.exit(f"ERROR: No {symbol} contracts returned by Kite.")

    relevant = contracts[
        contracts["expiry"].dt.date >= (start - timedelta(days=90))
    ].reset_index(drop=True)
    log.info("Stitching %d relevant contracts …", len(relevant))

    daily, hourly = build_continuous_data(kite, relevant, start, end)

    actual_start = min(daily.index)
    actual_end   = max(daily.index)
    days_covered = (actual_end - actual_start).days

    if days_covered < 60:
        log.warning("⚠  Only %d days of data (%s → %s). "
                    "Kite may not have older expired contracts.",
                    days_covered, actual_start, actual_end)
    else:
        log.info("Coverage: %d days  (%s → %s)", days_covered, actual_start, actual_end)

    payload = dict(daily=daily, hourly=hourly,
                   coverage_start=actual_start, coverage_end=actual_end)
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)
    log.info("Cached → %s", cache_file)
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
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1, width_pct=width_pct)


# ─── EXIT SCANNER ─────────────────────────────────────────────────────────────
def _exit_scan(bars, entry, sl, target, direction):
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


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest(data, cfg, lots, narrow_thresh):
    daily  = data["daily"]
    hourly = data["hourly"]
    start  = data["coverage_start"]
    end    = data["coverage_end"]

    slip         = cfg["slippage"]
    pnl_per_lot  = cfg["pnl_per_lot"]
    commission   = cfg["commission"]
    wide_thresh  = cfg["wide_pct"]

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

        if width < narrow_thresh:
            stats["narrow_days"] += 1
        elif width > wide_thresh:
            stats["wide_skipped"] += 1
            continue
        else:
            stats["neutral_skipped"] += 1
            continue

        # Hourly bars for today (day session only)
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

        first_close = float(day_h.iloc[0]["close"])
        direction = entry = sl = target = None

        if first_close > upper:
            direction = "LONG"
            entry  = upper + slip
            sl     = lower - slip
            target = R1
        elif first_close < lower:
            direction = "SHORT"
            entry  = lower - slip
            sl     = upper + slip
            target = S1

        if direction is None:
            stats["no_signal"] += 1
            continue

        # Degenerate guard
        if direction == "LONG"  and target <= entry:
            stats["no_signal"] += 1; continue
        if direction == "SHORT" and target >= entry:
            stats["no_signal"] += 1; continue

        outcome, exit_raw = _exit_scan(day_h.iloc[1:], entry, sl, target, direction)

        if outcome in ("TARGET", "SL"):
            exit_price = exit_raw - slip if direction == "LONG" else exit_raw + slip
        else:
            exit_price = exit_raw

        raw_move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
        net_pnl  = raw_move * pnl_per_lot * lots - commission
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
            P         = round(cpr["P"], 2),
            R1        = round(R1, 2),
            S1        = round(S1, 2),
        ))

    log.info("narrow=%d  wide_skip=%d  neutral_skip=%d  no_signal=%d  no_intraday=%d",
             stats["narrow_days"], stats["wide_skipped"], stats["neutral_skipped"],
             stats["no_signal"],   stats["no_intraday"])
    return trades, dict(month_pnl)


# ─── REPORT ───────────────────────────────────────────────────────────────────
def _inr(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}₹{v/1e5:.2f}L" if abs(v) >= 1e5 else f"{sign}₹{v:,.0f}"


def print_report(symbol, trades, month_pnl, cfg, lots, data, show_trades, narrow_thresh):
    SEP = "═" * 72
    lot_val    = cfg["ref_price"] * lots           # rough lot value for N lots
    margin_est = lot_val * cfg["margin_pct"]
    free_cap   = 50_000 - margin_est

    print(f"\n{SEP}")
    print(f"  MCX {symbol} CPR BACKTEST  (Breakout-only, AAK method)")
    print(f"  Coverage   : {data['coverage_start']}  →  {data['coverage_end']}")
    print(f"  Lots/trade : {lots}  ({cfg['lot_unit']} each)  |  "
          f"₹{cfg['pnl_per_lot'] * lots}/₹1 move total")
    print(f"  Quote      : {cfg['quote_desc']}")
    print(f"  Commission : ₹{cfg['commission']} flat/trade")
    print(f"  Slippage   : ₹{cfg['slippage']}/{cfg['quote_desc'].split()[-1]} per side")
    print(f"  Narrow CPR : < {narrow_thresh}%  |  Wide skip : > {cfg['wide_pct']}%")
    print(f"  Budget est : margin ≈ ₹{margin_est:,.0f}  |  free ≈ ₹{free_cap:,.0f} on ₹50K")
    print(SEP)

    if not trades:
        print("  No trades found.")
        print(f"  → Try --narrow {narrow_thresh + 0.2:.2f} to widen the narrow threshold")
        print("  → Run login.py to refresh access_token, then --clear-cache")
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

    print(f"\n  RESULTS  ({total} trades  |  {data['coverage_start']} → {data['coverage_end']})")
    print(f"  {'Win rate':<22}: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"  {'Net P&L':<22}: {_inr(net)}")
    print(f"  {'Avg / trade':<22}: {_inr(avg)}")
    print(f"  {'Avg win':<22}: {_inr(avg_win)}")
    print(f"  {'Avg loss':<22}: {_inr(avg_loss)}")
    if avg_loss != 0:
        print(f"  {'Win/Loss ratio':<22}: {abs(avg_win / avg_loss):.2f}x")
    print(f"  {'Best trade':<22}: {_inr(best)}")
    print(f"  {'Worst trade':<22}: {_inr(worst)}")
    print(f"  {'Target hits':<22}: {tgts} ({tgts/total*100:.0f}%)")
    print(f"  {'SL hits':<22}: {sls}  ({sls/total*100:.0f}%)")
    print(f"  {'EOD exits':<22}: {eods} ({eods/total*100:.0f}%)")
    print(f"  {'Longs':<22}: {len(longs)} trades  {lw}W  ({lw/max(len(longs),1)*100:.0f}% win)")
    print(f"  {'Shorts':<22}: {len(shorts)} trades  {sw}W  ({sw/max(len(shorts),1)*100:.0f}% win)")
    print(f"  {'Green months':<22}: {greens}  |  Red: {reds}")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    if month_pnl:
        max_abs = max(abs(v) for v in month_pnl.values()) or 1
        print(f"\n  ── Monthly P&L {'─'*50}")
        for mo, pnl in sorted(month_pnl.items()):
            bar  = "█" * int(abs(pnl) / max_abs * 20)
            sign = "+" if pnl >= 0 else ""
            print(f"  {mo}  {sign}₹{pnl:>9,.0f}  {bar}")
        print(f"  {'─'*40}")
        print(f"  TOTAL          {_inr(net)}")

    # ── Trade log ─────────────────────────────────────────────────────────────
    if show_trades:
        print(f"\n  ── All Trades {'─'*55}")
        hdr = (f"  {'Date':<12} {'Dir':<6} {'Width%':<7} "
               f"{'Entry':>9} {'SL':>9} {'Target':>9} {'Exit':>9} "
               f"{'Move':>8} {'P&L':>9}  {'Result'}")
        print(hdr)
        print(f"  {'─'*95}")
        for t in trades:
            sign = "+" if t["pnl"] >= 0 else ""
            print(
                f"  {str(t['date']):<12} {t['direction']:<6} {t['width_pct']:<7.3f} "
                f"{t['entry']:>9.2f} {t['sl']:>9.2f} {t['target']:>9.2f} "
                f"{t['exit']:>9.2f} {t['raw_move']:>+8.2f} "
                f"{sign}₹{t['pnl']:>7,.0f}  {t['outcome']}"
            )

    print(f"\n{SEP}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if trades:
        csv_path = os.path.join(CACHE_DIR, f"{symbol.lower()}_cpr_backtest.csv")
        pd.DataFrame(trades).to_csv(csv_path, index=False)
        print(f"  Trades saved → {csv_path}")
        print(SEP)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    symbols = list(CONFIGS.keys())
    p = argparse.ArgumentParser(
        description="MCX Commodity CPR Backtest (AAK Method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available symbols: {', '.join(symbols)}"
    )
    p.add_argument("--symbol", required=True, choices=symbols,
                   help="MCX commodity to backtest")
    p.add_argument("--lots",        type=int,   default=None,
                   help="Number of lots per trade (overrides per-symbol default)")
    p.add_argument("--days",        type=int,   default=120,
                   help="Days of history to fetch (default: 120)")
    p.add_argument("--narrow",      type=float, default=None,
                   help="Narrow CPR width%% threshold (overrides per-symbol default)")
    p.add_argument("--show-trades", action="store_true",
                   help="Print every individual trade in the report")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete cached data and re-download from Kite")
    args = p.parse_args()

    cfg          = CONFIGS[args.symbol]
    lots         = args.lots   if args.lots   is not None else cfg["default_lots"]
    narrow_thresh = args.narrow if args.narrow is not None else cfg["narrow_pct"]

    kite  = get_kite()
    end   = date.today()
    start = end - timedelta(days=args.days)

    data   = load_data(kite, args.symbol, cfg, start, end, clear_cache=args.clear_cache)
    trades, month_pnl = run_backtest(data, cfg, lots, narrow_thresh)
    print_report(args.symbol, trades, month_pnl, cfg, lots, data,
                 show_trades=args.show_trades, narrow_thresh=narrow_thresh)


if __name__ == "__main__":
    main()
