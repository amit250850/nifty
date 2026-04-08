#!/usr/bin/env python3
"""
goldpetal_autoexec.py
=====================
Auto-execute for MCX GOLDPETAL CPR breakout strategy.
Runs at 10:02 IST (Mon–Fri) via APScheduler in main.py.

FLOW
────
  08:55  Morning alert fires — CPR levels sent to Telegram
  09:00  MCX opens — watch price action
  10:00  First hourly candle closes
  10:02  ← THIS SCRIPT RUNS
           1. Re-calculates CPR from yesterday's OHLC
           2. Fetches 09:00–10:00 60-min candle close
           3. Decision logic:
                close > upper_cpr  → LONG  : market BUY  10 lots NRML
                close < lower_cpr  → SHORT : market SELL 10 lots NRML
                inside CPR zone    → no trade today
           4. Writes trade state → data/goldpetal_trade.json
           5. Sends Telegram execution or skip confirmation

GUARD
─────
  GOLDPETAL_AUTO_EXECUTE = False (default — change to True for live orders)
  When False: full logic runs, decision is logged + Telegr alerteds, NO order placed.
  Set True only after confirming the alert flow is working correctly.

PRODUCT
───────
  NRML — holds until manually squared or monitor squares at 16:55 IST.
  NOT MIS — MIS is disabled by Zerodha on GOLDPETAL during high-volatility periods.

Requires: .env with KITE_API_KEY, KITE_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
          Run login.py first each morning to refresh access_token.
"""

import os
import sys
import json
import logging
from datetime import date, timedelta, datetime, time as dtime

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import requests
    import pandas as pd
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests pandas "
          "--break-system-packages")
    sys.exit(1)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Feature flag — SET True only when ready for live orders ───────────────────
GOLDPETAL_AUTO_EXECUTE: bool = True

LOTS         = 10
PNL_PER_LOT  = 1        # ₹1 per lot per ₹1 move
COMMISSION   = 55       # flat per trade (Zerodha per-order model)
SLIPPAGE     = 0.25     # ₹/gram per side

NARROW_HIGH     = 0.10  # Only HIGH/MEDIUM/STANDARD days fire auto-exec
NARROW_STANDARD = 0.30  # Skip if width > 0.30% (SKIP grade)

MARGIN_PCT   = 0.15
CAPITAL      = 50_000

STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "goldpetal_trade.json",
)

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── KITE ──────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        sys.exit("ERROR: Missing KITE_API_KEY / KITE_ACCESS_TOKEN. Run login.py first.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT ────────────────────────────────────────────────────────────────
def get_front_month(kite: KiteConnect) -> tuple:
    """Return (token, tradingsymbol, expiry) for active front-month GOLDPETAL."""
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    petal = df[
        (df["name"] == "GOLDPETAL") &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= date.today())
    ].sort_values("expiry").reset_index(drop=True)

    if petal.empty:
        raise RuntimeError("No active GOLDPETAL contracts found.")

    front = petal.iloc[0]
    dte   = (front["expiry"].date() - date.today()).days
    if dte <= 5 and len(petal) > 1:
        front = petal.iloc[1]

    log.info("Contract: %s  expiry=%s", front["tradingsymbol"], front["expiry"].date())
    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── CPR ───────────────────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1 = 2 * P - L
    S1 = 2 * P - H
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, S1=S1,
                width_pct=(upper_cpr - lower_cpr) / P * 100)


def get_yesterday_ohlc(kite: KiteConnect, token: int) -> dict:
    """Fetch yesterday's daily OHLC bar."""
    today   = date.today()
    from_dt = today - timedelta(days=7)   # buffer for weekends/holidays
    to_dt   = today - timedelta(days=1)
    rows    = kite.historical_data(token, from_dt, to_dt,
                                   interval="day", continuous=False, oi=False)
    if not rows:
        raise RuntimeError("No daily data returned — is today a holiday?")
    yesterday = rows[-1]
    return {
        "H": float(yesterday["high"]),
        "L": float(yesterday["low"]),
        "C": float(yesterday["close"]),
        "date": yesterday["date"],
    }


def get_first_hourly_candle(kite: KiteConnect, token: int) -> dict:
    """Fetch the 09:00–10:00 60-minute candle for today."""
    today   = date.today()
    from_dt = datetime.combine(today, dtime(9, 0))
    to_dt   = datetime.combine(today, dtime(10, 2))
    rows    = kite.historical_data(token, from_dt, to_dt,
                                   interval="60minute", continuous=False, oi=False)
    if not rows:
        raise RuntimeError("No 60-min candle returned — MCX not open yet?")
    candle = rows[0]  # 09:00 candle
    return {
        "open":  float(candle["open"]),
        "high":  float(candle["high"]),
        "low":   float(candle["low"]),
        "close": float(candle["close"]),
    }


# ─── STATE FILE ────────────────────────────────────────────────────────────────
def load_state() -> dict:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.info("Trade state saved → %s", STATE_FILE)


# ─── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("\n" + "━" * 56 + "\n" + text + "\n" + "━" * 56)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=10,
    )
    if r.ok:
        log.info("Telegram ✅")
    else:
        log.error("Telegram failed %s: %s", r.status_code, r.text[:120])


def _p(v): return f"₹{v:,.2f}"
def _r(v): return f"₹{v:,.0f}"


# ─── PLACE ORDER ───────────────────────────────────────────────────────────────
def place_order(kite: KiteConnect, symbol: str, direction: str, lots: int) -> str | None:
    """
    Place a NRML market order on MCX GOLDPETAL.
    Returns order_id string on success, None on failure or dry-run.
    """
    tx = kite.TRANSACTION_TYPE_BUY if direction == "LONG" else kite.TRANSACTION_TYPE_SELL

    if not GOLDPETAL_AUTO_EXECUTE:
        log.info("[DRY-RUN] Would place %s MARKET %d lots %s NRML — GOLDPETAL_AUTO_EXECUTE=False",
                 direction, lots, symbol)
        return "DRY-RUN"

    try:
        order_id = kite.place_order(
            tradingsymbol    = symbol,
            exchange         = "MCX",
            transaction_type = tx,
            quantity         = lots,         # 1 lot = 1 gram; qty = number of lots
            product          = kite.PRODUCT_NRML,
            order_type       = kite.ORDER_TYPE_MARKET,
            variety          = kite.VARIETY_REGULAR,
        )
        log.info("✅ %s order placed: %s qty=%d → order_id=%s", direction, symbol, lots, order_id)
        return str(order_id)
    except Exception as exc:
        log.error("Order failed: %s", exc)
        return None


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def run_goldpetal_autoexec() -> None:
    now       = datetime.now()
    today_str = now.strftime("%a %d %b %Y")

    log.info("=" * 56)
    log.info("GOLDPETAL Auto-Exec  —  %s  10:02 IST", today_str)
    log.info("=" * 56)

    # ── Guard: don't trade if there's already an open GOLDPETAL position today ──
    existing = load_state()
    if existing.get("status") == "open" and existing.get("date") == str(date.today()):
        log.warning("Open GOLDPETAL trade already exists for today — skipping auto-exec.")
        send_telegram(
            "⚠️ <b>GOLDPETAL Auto-Exec Skipped</b>\n"
            "Open position already registered for today.\n"
            f"Direction: {existing['direction']}  Entry: {_p(existing['entry'])}"
        )
        return

    kite = get_kite()
    token, sym, expiry = get_front_month(kite)

    # ── Step 1: Recalculate CPR from yesterday ──────────────────────────────────
    prev   = get_yesterday_ohlc(kite, token)
    cpr    = calc_cpr(prev["H"], prev["L"], prev["C"])
    width  = cpr["width_pct"]
    upper  = cpr["upper_cpr"]
    lower  = cpr["lower_cpr"]

    log.info("Yesterday: H=%s L=%s C=%s", _p(prev["H"]), _p(prev["L"]), _p(prev["C"]))
    log.info("CPR: upper=%s lower=%s width=%.3f%%", _p(upper), _p(lower), width)

    # ── Guard: wide day — SKIP grade means no trade ─────────────────────────────
    if width >= NARROW_STANDARD:
        log.info("CPR width %.3f%% ≥ 0.30%% — SKIP grade, no trade.", width)
        send_telegram(
            f"⚪ <b>GOLDPETAL Auto-Exec — {today_str}</b>\n"
            f"No trade — CPR width {width:.3f}% (SKIP grade ≥ 0.30%)\n"
            f"<i>Morning alert already flagged this as a no-trade day.</i>"
        )
        return

    # ── Step 2: Fetch the 09:00–10:00 candle ───────────────────────────────────
    candle = get_first_hourly_candle(kite, token)
    close  = candle["close"]
    log.info("09:00–10:00 candle: O=%s H=%s L=%s C=%s",
             _p(candle["open"]), _p(candle["high"]), _p(candle["low"]), _p(close))

    # ── Step 3: Direction decision ──────────────────────────────────────────────
    if close > upper:
        direction = "LONG"
        entry     = upper + SLIPPAGE
        sl        = lower - SLIPPAGE
        target    = cpr["R1"]
    elif close < lower:
        direction = "SHORT"
        entry     = lower - SLIPPAGE
        sl        = upper + SLIPPAGE
        target    = cpr["S1"]
    else:
        log.info("Candle close %s inside CPR [%s – %s] — no trade.", _p(close), _p(lower), _p(upper))
        send_telegram(
            f"🚫 <b>GOLDPETAL Auto-Exec — {today_str}</b>\n"
            f"No trade — 10:00 candle closed <b>inside CPR zone</b>\n"
            f"  Close:     {_p(close)}\n"
            f"  Upper CPR: {_p(upper)}\n"
            f"  Lower CPR: {_p(lower)}\n"
            f"<i>Price needs to break and close outside CPR to trigger.</i>"
        )
        return

    # ── Degenerate guard: target must be profitable after slippage ──────────────
    if direction == "LONG"  and target <= entry:
        log.warning("Degenerate LONG: target %s ≤ entry %s — aborting.", _p(target), _p(entry))
        return
    if direction == "SHORT" and target >= entry:
        log.warning("Degenerate SHORT: target %s ≥ entry %s — aborting.", _p(target), _p(entry))
        return

    # ── Risk/reward for notification ────────────────────────────────────────────
    if direction == "LONG":
        risk   = (entry - sl)     * PNL_PER_LOT * LOTS
        reward = (target - entry) * PNL_PER_LOT * LOTS
    else:
        risk   = (sl - entry)     * PNL_PER_LOT * LOTS
        reward = (entry - target) * PNL_PER_LOT * LOTS

    rr  = f"1 : {reward/risk:.1f}" if risk > 0 and reward > 0 else "–"
    net = reward - COMMISSION

    arrow = "📈" if direction == "LONG" else "📉"
    mode  = "⚡ LIVE ORDER" if GOLDPETAL_AUTO_EXECUTE else "🔒 DRY-RUN (no order placed)"

    # ── Step 4: Place order ─────────────────────────────────────────────────────
    order_id = place_order(kite, sym, direction, LOTS)
    order_ok  = order_id is not None

    # ── Step 5: Save state for monitor ─────────────────────────────────────────
    state = {
        "date":       str(date.today()),
        "symbol":     sym,
        "token":      token,
        "expiry":     str(expiry),
        "direction":  direction,
        "lots":       LOTS,
        "entry":      entry,
        "sl":         sl,
        "target":     target,
        "order_id":   order_id,
        "status":     "open" if order_ok else "order_failed",
        "opened_at":  now.isoformat(),
        # alert-sent flags (used by monitor to prevent duplicate Telegrams)
        "alerts_sent": [],
    }
    save_state(state)

    # ── Step 6: Telegram confirmation ──────────────────────────────────────────
    order_line = (
        f"✅ Order placed — ID: <code>{order_id}</code>"
        if order_ok and GOLDPETAL_AUTO_EXECUTE
        else f"🔒 Dry-run — no real order (GOLDPETAL_AUTO_EXECUTE=False)"
    )
    if not order_ok and GOLDPETAL_AUTO_EXECUTE:
        order_line = "❌ <b>Order FAILED</b> — check Kite / logs immediately"

    send_telegram(
        f"{arrow} <b>GOLDPETAL {direction} — {today_str}</b>  [{mode}]\n"
        f"<code>{sym}</code>  exp {expiry}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"<b>10:00 candle close: {_p(close)}</b>\n"
        f"  Upper CPR: {_p(upper)}   Lower CPR: {_p(lower)}\n\n"
        f"<b>Trade</b>\n"
        f"  Direction : {direction}  ·  {LOTS} lots NRML\n"
        f"  Entry     : {_p(entry)}  (market fill ~this)\n"
        f"  Stop-Loss : {_p(sl)}\n"
        f"  Target    : {_p(target)}  ({"R1" if direction == "LONG" else "S1"})\n"
        f"  Risk      : {_r(risk)}   Reward: {_r(reward)}   RR: {rr}\n"
        f"  Net P&L at target: {_r(net)} (after ₹{COMMISSION} commission)\n\n"
        f"{order_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 Monitor running — will alert on SL/target hit or at 16:55"
    )

    log.info("Auto-exec complete: direction=%s entry=%s sl=%s target=%s order_id=%s",
             direction, _p(entry), _p(sl), _p(target), order_id)


# ─── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_goldpetal_autoexec()
