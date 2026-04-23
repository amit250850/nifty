#!/usr/bin/env python3
"""
backtest_goldguinea_cpr.py
==========================
CPR (Central Pivot Range) strategy backtest on MCX GOLDGUINEA futures
using Zerodha Kite Connect API for historical data.

MCX GOLDGUINEA Specifications
------------------------------
  Lot size   : 8 grams
  Quote unit : ₹ per gram
  Tick size  : ₹1 per gram

  P&L per lot per ₹1 move = 8 grams × ₹1/gram = ₹8 per lot
  With 2 lots: ₹1 price move  → ₹16 total P&L
  With 2 lots: ₹100 move      → ₹1,600 total P&L

  Expiry cycle: BIMONTHLY (Feb, Apr, Jun, Aug, Oct, Dec) — 6 contracts/year.
  Unlike GOLDPETAL (monthly), each GOLDGUINEA contract covers ~2 months.
  Stitching rolls 3 days before expiry; front-month alert rolls 7 days before.

  Budget at gold ₹14,810/gram:
    Lot value     = 8 × ₹14,810 = ₹1,18,480
    NRML margin   ≈ ₹17,772 per lot  (15%)
    2 lots margin ≈ ₹35,544  ← fits ₹50K, leaves ₹14,456 buffer ✅
    3 lots margin ≈ ₹53,316  ← slightly over ₹50K ❌

  vs GOLDPETAL (10 lots):
    Margin ≈ ₹22,220  (more headroom but ₹10/₹1 move vs ₹16/₹1 move here)

Commission note
---------------
  Zerodha charges per ORDER, not per lot, for commodity futures:
    Entry order : ₹20 flat (all 2 lots together)
    Exit order  : ₹20 flat
    STT + Exch  : ~₹15–25 per trade (higher than GOLDPETAL due to larger lot value)
    Total/trade : ~₹60–65 flat

CPR Strategy (AAK Method) — BREAKOUT ONLY
------------------------------------------
  Narrow CPR (width% < 0.30%) → trending day
    1H candle closes ABOVE upper_cpr → LONG  at upper_cpr+slip, SL=lower_cpr-slip, T=R1
    1H candle closes BELOW lower_cpr → SHORT at lower_cpr-slip, SL=upper_cpr+slip, T=S1
  Wide/Neutral → SKIP (range mode proven -ve on Gold)

Setup
-----
  1. python login.py
  2. python backtest_goldguinea_cpr.py              (default 2 lots)
  3. python backtest_goldguinea_cpr.py --lots 3
  4. python backtest_goldguinea_cpr.py --show-trades
  5. python backtest_goldguinea_cpr.py --clear-cache
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

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── GOLDGUINEA SPECS ─────────────────────────────────────────────────────────
LOT_GRAMS    = 8            # 8 grams per lot
QUOTE_UNIT   = 8            # price quoted in ₹ per 8g (the full lot) — same convention
                            # as GOLDM (100g lot, quoted per 10g) and GOLD (1kg, per 10g).
                            # Confirmed: Apr 2026 price ~₹1,19,713/lot ÷ 8 = ₹14,964/gram
                            #            which matches GOLDM's ₹14,810/gram. ✓
PNL_PER_LOT  = LOT_GRAMS / QUOTE_UNIT   # = 1 → ₹1 per lot per ₹1 price move
                                          # (price is in ₹ per 8g, not per gram)

NARROW_PCT   = 0.30         # CPR width% threshold for breakout day
WIDE_PCT     = 0.55

COMMISSION   = 65           # ₹ FLAT per round-trip (₹20 entry + ₹20 exit + ~₹25 STT/exch/GST)
SLIPPAGE     = 4.00         # ₹ per 8g lot per side  (= ₹0.50/gram × 8g — same per-gram
                            # slippage as GOLDPETAL ₹0.25 scaled for lower liquidity)

SESSION_START = dtime(9, 0)
SESSION_END   = dtime(17, 0)

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "goldguinea_kite_cache.pkl")


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
def get_goldguinea_contracts(kite: KiteConnect) -> pd.DataFrame:
    """Fetch all MCX GOLDGUINEA FUT contracts sorted by expiry."""
    log.info("Fetching MCX instruments …")
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    guinea = df[
        (df["name"] == "GOLDGUINEA") &
        (df["instrument_type"] == "FUT")
    ].sort_values("expiry").reset_index(drop=True)

    log.info("Found %d GOLDGUINEA contracts (expiries: %s → %s)",
             len(guinea),
             guinea["expiry"].min().date() if not guinea.empty else "N/A",
             guinea["expiry"].max().date() if not guinea.empty else "N/A")
    return guinea


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
    """Stitch monthly GOLDGUINEA contracts. Rolls 3 days before each expiry."""
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

        log.info("  %-24s  %s → %s  (expiry %s)", sym, c_start, c_end, expiry)

        d = _fetch_daily(kite, token, c_start, c_end)
        if not d.empty:
            daily_parts.append(d)

        h = _fetch_hourly_chunked(kite, token, c_start, c_end)
        if not h.empty:
            hourly_parts.append(h)

    if not daily_parts:
        sys.exit(
            "ERROR: No GOLDGUINEA data downloaded.\n"
            "       Verify access_token is fresh (run login.py) and\n"
            "       GOLDGUINEA contracts appear in kite.instruments('MCX').\n"
            "       Try: python -c \"from kiteconnect import KiteConnect; "
            "import os; from dotenv import load_dotenv; load_dotenv(); "
            "k=KiteConnect(api_key=os.getenv('KITE_API_KEY')); "
            "k.set_access_token(os.getenv('KITE_ACCESS_TOKEN')); "
            "print([x for x in k.instruments('MCX') if x['name']=='GOLDGUINEA'][:3])\""
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

    log.info("Downloading GOLDGUINEA data  %s → %s …", start, end)
    contracts = get_goldguinea_contracts(kite)
    if contracts.empty:
        sys.exit("ERROR: No GOLDGUINEA contracts returned by Kite.")

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


# ─── CPR MATH ─────────────────────────────────────────────────────────────────
def calc_cpr(H, L, C):
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)   # always normalised — prevents SL-above-entry bug
    lower_cpr = min(TC, BC)
    R1        = 2 * P - L
    R2        = P + (H - L)
    S1        = 2 * P - H
    S2        = P - (H - L)
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, R2=R2, S1=S1, S2=S2, width_pct=width_pct)


# ─── EXIT SCANNER ─────────────────────────────────────────────────────────────
def _exit_scan(bars, entry, sl, target, direction):
    for _, bar in bars.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])
        if direction == "LONG":
            if l <= sl:     return "SL",     sl
            if h >= target: return "TARGET",  target
        else:
            if h >= sl:     return "SL",     sl
            if l <= target: return "TARGET",  target
    eod = float(bars.iloc[-1]["close"]) if not bars.empty else entry
    return "EOD", eod


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
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

        if width < narrow_thresh:
            pass  # tradeable day — fall through
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

        # Degenerate guard: target must clear entry after slippage
        if direction == "LONG"  and target <= entry:
            stats["no_signal"] += 1
            continue
        if direction == "SHORT" and target >= entry:
            stats["no_signal"] += 1
            continue

        outcome, exit_raw = _exit_scan(day_h.iloc[1:], entry, sl, target, direction)

        if outcome in ("TARGET", "SL"):
            exit_price = exit_raw - SLIPPAGE if direction == "LONG" \
                         else exit_raw + SLIPPAGE
        else:
            exit_price = exit_raw

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


# ─── REPORT ───────────────────────────────────────────────────────────────────
def _inr(v):
    sign = "+" if v >= 0 else ""
    if abs(v) >= 1e5: return f"{sign}₹{v/1e5:.2f}L"
    return f"{sign}₹{v:,.0f}"


def print_report(trades, month_pnl, lots, data, show_trades):
    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  MCX GOLDGUINEA CPR BACKTEST  (Breakout-only, AAK method)")
    print(f"  Coverage  : {data['coverage_start']}  →  {data['coverage_end']}")
    print(f"  Lots/trade: {lots}  |  P&L: ₹{PNL_PER_LOT:.0f}/lot/₹1move  "
          f"→  ₹{PNL_PER_LOT * lots:.0f} per ₹1 move total")
    print(f"  Quote unit: ₹ per 8g lot (NOT per gram — like GOLDM's per-10g quote)")
    print(f"  Commission: ₹{COMMISSION} FLAT per trade (Zerodha per-order model)")
    print(f"  Slippage  : ₹{SLIPPAGE}/lot per side (= ₹{SLIPPAGE/LOT_GRAMS:.2f}/gram)")
    print(f"  Narrow CPR: < {NARROW_PCT}%   (Wide/Neutral skipped — proven -ve on Gold)")
    print(SEP)

    if not trades:
        print("  No trades found.")
        print("  → Check: kite.instruments('MCX') returns 'GOLDGUINEA' entries")
        print("  → Run login.py to refresh access_token")
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
        print(f"  {'Win/Loss ratio':<22}: {abs(avg_win/avg_loss):.1f}x")
    print(f"  {'Best trade':<22}: {_inr(best)}")
    print(f"  {'Worst trade':<22}: {_inr(worst)}")
    print(f"  {'Target hits':<22}: {tgts} ({tgts/total*100:.0f}%)")
    print(f"  {'SL hits':<22}: {sls}  ({sls/total*100:.0f}%)")
    print(f"  {'EOD exits':<22}: {eods} ({eods/total*100:.0f}%)")
    print(f"  {'Longs':<22}: {len(longs)} trades  {lw}W "
          f"({lw/max(len(longs),1)*100:.0f}% win)")
    print(f"  {'Shorts':<22}: {len(shorts)} trades  {sw}W "
          f"({sw/max(len(shorts),1)*100:.0f}% win)")
    print(f"  {'Green months':<22}: {greens}  |  Red: {reds}")

    # ── Budget check ──────────────────────────────────────────────────────
    gold_gram   = 14810
    lot_value   = gold_gram * LOT_GRAMS          # 8g × ₹14,810 = ₹1,18,480
    margin_lot  = lot_value * 0.15
    margin_tot  = margin_lot * lots
    # SLIPPAGE is in ₹/lot (per 8g quote); convert to per-gram for display
    sl_risk_price = NARROW_PCT / 100 * (gold_gram * LOT_GRAMS)  # width in ₹/lot price
    sl_risk_lot   = sl_risk_price * PNL_PER_LOT                  # ₹ per lot
    sl_risk_all   = sl_risk_lot * lots + COMMISSION

    print(f"\n  ── GOLDGUINEA Budget Guide (Gold ≈ ₹{gold_gram:,}/gram) ──────────")
    print(f"  Lot size        : 8 grams  |  Quote: ₹ per 8g lot (NOT per gram)")
    print(f"  Lot value       : ₹{lot_value:,}  (8g × ₹{gold_gram:,}/gram)")
    print(f"  NRML margin/lot : ₹{margin_lot:,.0f}  (15% — verify on Kite)")
    print(f"  {lots} lots margin    : ₹{margin_tot:,.0f}")
    print(f"  ₹50K headroom   : ₹{50000 - margin_tot:,.0f}")
    print(f"  SL risk/trade   : ₹{sl_risk_all:,.0f}  "
          f"({sl_risk_all/50000*100:.1f}% of ₹50K)")
    print(f"  P&L per ₹1 move : ₹{PNL_PER_LOT * lots:.0f}  "
          f"(₹{PNL_PER_LOT:.0f}/lot × {lots} lots)  [price in ₹/8g]")
    print(f"  P&L per ₹100 move: ₹{PNL_PER_LOT * lots * 100:,.0f}  (₹100 in lot price)")

    # ── vs GOLDPETAL comparison ───────────────────────────────────────────
    print(f"\n  ── vs GOLDPETAL (10 lots) ─────────────────────────────────────")
    print(f"  GOLDGUINEA {lots}L margin : ₹{margin_tot:,.0f}  |  P&L/₹1 lot-price = ₹{PNL_PER_LOT*lots}")
    print(f"  GOLDPETAL 10L margin : ₹22,220           |  P&L/₹1 gram-price = ₹10")
    print(f"  Note: both track same gold price — GOLDGUINEA prices scale 8x larger.")
    print(f"  Equivalent comparison: GOLDGUINEA ₹8 move ≈ GOLDPETAL ₹1 move (same ₹ P&L).")

    # ── Monthly P&L ───────────────────────────────────────────────────────
    print(f"\n  ── Monthly P&L ──────────────────────────────────────────────────")
    for m, v in sorted(month_pnl.items()):
        bar = "█" * min(int(abs(v) / 200), 26)
        mk  = "✅" if v >= 0 else "❌"
        print(f"  {m}  {mk}  {_inr(v):>12}  {bar}")

    print(f"\n{SEP}")

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

    try:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "goldguinea_cpr_backtest.csv")
        pd.DataFrame(trades).to_csv(out, index=False)
        print(f"  Trades → {out}")
    except Exception as exc:
        log.warning("CSV save failed: %s", exc)

    print(SEP + "\n")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="MCX GOLDGUINEA CPR Backtest via Zerodha Kite Connect")
    ap.add_argument("--lots",        type=int,   default=2,
                    help="Lots per trade (default 2; fits ₹50K budget)")
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
    print(f"  GOLDGUINEA CPR Backtest  |  Kite Connect  |  {start} → {end}")
    print(f"  Lots: {args.lots}  |  P&L/move: ₹{PNL_PER_LOT * args.lots:.0f} per ₹1")
    print(f"  (₹1 price move = ₹{PNL_PER_LOT:.0f}/lot × {args.lots} lots = "
          f"₹{PNL_PER_LOT * args.lots:.0f} total)")
    print(f"  Margin estimate: ₹{14810 * 8 * 0.15 * args.lots:,.0f} "
          f"for {args.lots} lots (15% of lot value, verify on Kite)")
    print(f"{'═'*72}\n")

    kite   = get_kite()
    data   = load_data(kite, start, end, clear_cache=args.clear_cache)
    trades, month_pnl = run_backtest(data, args.lots, narrow_thresh=args.narrow)
    print_report(trades, month_pnl, args.lots, data, show_trades=args.show_trades)


if __name__ == "__main__":
    main()
