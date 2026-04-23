#!/usr/bin/env python3
"""
usdinr_cpr_monitor.py
=====================
Intraday CPR trade monitor for NSE CDS USDINR futures.
Runs every 5 min from 9:45 AM – 4:55 PM IST via APScheduler in main.py.

Since trades are placed manually in Kite after the 08:55 morning alert,
this monitor:

  1. From 9:45 IST — Checks live LTP vs CPR levels. If a breakout signal
     exists (LTP outside CPR), initialises state and sends "Signal Active"
     Telegram with levels to watch.

  2. Every 5 min after — Fetches LTP, validates CPR signal is still active
     (LTP remains outside CPR on breakout side), sends 5-min reminder.

  3. At 16:45 IST — Sends EOD square-off reminder. CDS closes at 17:00.

State file:
  data/usdinr_cpr_state.json

  Reset each calendar day. Persists across bot restarts.

USDINR specs
  10 lots · lot_size=1000 USD · P&L=₹1000/lot per ₹1/USD · commission=₹30
  slippage=₹0.0025/side · margin~₹2200/lot · CPR threshold 0.020%
"""

import os
import sys
import json
import logging
from datetime import date, datetime, timedelta, time as dtime

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import requests
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests pandas yfinance")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

LOTS            = 10
LOT_SIZE_USD    = 1000
PNL_PER_LOT     = LOT_SIZE_USD   # ₹1,000 per lot per ₹1/USD move
COMMISSION      = 30
SLIPPAGE        = 0.0025          # 1 tick per side
NARROW_THRESHOLD = 0.020          # % — must match usdinr_morning_alert.py
EXPIRY_ROLL_DAYS = 3              # roll to next contract when DTE <= 3

SQ_OFF_HOUR   = 16
SQ_OFF_MINUTE = 45

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE  = os.path.join(DATA_DIR, "usdinr_cpr_state.json")

# ─── LOGGING ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _p(v): return f"₹{v:.4f}"
def _r(v): return f"₹{v:,.0f}"


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing KITE_API_KEY / KITE_ACCESS_TOKEN. Run login.py.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT ───────────────────────────────────────────────────────────────
def get_front_month(kite: KiteConnect) -> tuple:
    """Return (token, tradingsymbol, expiry) for active front-month USDINR futures."""
    rows = kite.instruments("CDS")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    contracts = df[
        (df["name"] == "USDINR") &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= date.today())
    ].sort_values("expiry").reset_index(drop=True)

    if contracts.empty:
        raise RuntimeError("No active USDINR contracts found.")

    front = contracts.iloc[0]
    dte   = (front["expiry"].date() - date.today()).days
    if dte <= EXPIRY_ROLL_DAYS and len(contracts) > 1:
        front = contracts.iloc[1]

    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── OHLC / CPR ───────────────────────────────────────────────────────────────
YF_TICKER = "USDINR=X"

def get_yesterday_ohlc() -> dict | None:
    """
    Fetch yesterday's OHLC via yfinance (spot rate).
    Spot and futures are within a few paise — fine for CPR levels.
    Ignores zero-range bars (holidays / bad data where H=L=C).
    """
    try:
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=10)
        df = yf.download(YF_TICKER, start=str(start_dt), end=str(end_dt),
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            log.error("[USDINR] yfinance returned no data for %s", YF_TICKER)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df[df["high"] != df["low"]]  # drop zero-range bars
        if df.empty:
            return None
        r = df.iloc[-1]
        log.info("[USDINR] OHLC (yfinance %s): H=%.4f  L=%.4f  C=%.4f  date=%s",
                 YF_TICKER, r["high"], r["low"], r["close"], df.index[-1].date())
        return {"H": float(r["high"]), "L": float(r["low"]), "C": float(r["close"])}
    except Exception as exc:
        log.error("[USDINR] yfinance OHLC fetch failed: %s", exc)
        return None


def calc_cpr(H, L, C) -> dict:
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


def get_first_hourly_close(kite: KiteConnect, token: int) -> float | None:
    """Fetch the 09:00–10:00 hourly candle close (the CPR signal candle)."""
    today = date.today()
    try:
        rows = kite.historical_data(token, today, today,
                                    interval="60minute", continuous=False, oi=False)
        if not rows:
            return None
        first = rows[0]
        if pd.Timestamp(first["date"]).hour == 9:
            return float(first["close"])
        return None
    except Exception as exc:
        log.error("[USDINR] First hourly close fetch failed: %s", exc)
        return None


def get_ltp(kite: KiteConnect, tradingsymbol: str) -> float:
    """Fetch current LTP for a CDS (currency futures) symbol."""
    key  = f"CDS:{tradingsymbol}"
    data = kite.ltp([key])
    return float(data[key]["last_price"])


# ─── STATE FILE ───────────────────────────────────────────────────────────────
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[USDINR] Could not load state: %s", exc)
        return {}


def save_state(state: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def alert_sent(state: dict, code: str) -> bool:
    return code in state.get("alerts_sent", [])


def mark_alert(state: dict, code: str) -> None:
    state.setdefault("alerts_sent", [])
    if code not in state["alerts_sent"]:
        state["alerts_sent"].append(code)


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("\n" + "━" * 56 + "\n" + text + "\n" + "━" * 56)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if r.ok:
            log.info("[USDINR] Telegram ✓")
        else:
            log.error("[USDINR] Telegram failed %s: %s", r.status_code, r.text[:80])
    except Exception as exc:
        log.error("[USDINR] Telegram error: %s", exc)


# ─── MONITOR CYCLE ────────────────────────────────────────────────────────────
def _run_cycle(kite: KiteConnect, now: datetime) -> None:
    today_str = now.strftime("%a %d %b %Y")
    state     = load_state()

    # ── Reset if state is from a previous day ────────────────────────────────
    if state.get("date") and state["date"] != str(date.today()):
        log.info("[USDINR] Stale state from %s — resetting.", state["date"])
        state = {}
        save_state(state)

    # ── Initialise state (first run of the day) ───────────────────────────────
    if not state:
        log.info("[USDINR] No state — detecting signal.")

        try:
            token, tradingsym, expiry = get_front_month(kite)
        except Exception as exc:
            log.error("[USDINR] get_front_month failed: %s", exc)
            return

        ohlc = get_yesterday_ohlc()
        if not ohlc:
            log.warning("[USDINR] No yesterday OHLC — skipping init.")
            return

        cpr   = calc_cpr(ohlc["H"], ohlc["L"], ohlc["C"])
        width = cpr["width_pct"]

        log.info("[USDINR] CPR width=%.4f%%  threshold=%.3f%%  %s",
                 width, NARROW_THRESHOLD,
                 "NARROW ✓" if width < NARROW_THRESHOLD else "WIDE — no trade today")

        if width >= NARROW_THRESHOLD:
            save_state({"date": str(date.today()), "status": "no_trade"})
            return

        upper = cpr["upper_cpr"]
        lower = cpr["lower_cpr"]

        # From 9:45 use live LTP; after 10:05 use completed 1H candle close
        if now.hour == 9 or (now.hour == 10 and now.minute < 5):
            try:
                price_check = get_ltp(kite, tradingsym)
            except Exception as exc:
                log.error("[USDINR] LTP fetch failed: %s", exc)
                return
            if price_check <= 0:
                log.warning("[USDINR] LTP=%.4f invalid (market not yet open?). Skipping.", price_check)
                return
            price_label = "LTP"
        else:
            price_check = get_first_hourly_close(kite, token)
            if price_check is None:
                log.warning("[USDINR] First hourly candle not ready yet.")
                return
            if price_check <= 0:
                log.warning("[USDINR] First hourly close=%.4f invalid. Skipping.", price_check)
                return
            price_label = "1H close"

        log.info("[USDINR] %s=%.4f  CPR[%.4f – %.4f]  → %s",
                 price_label, price_check, lower, upper,
                 "LONG" if price_check > upper else
                 "SHORT" if price_check < lower else "INSIDE — no trade")

        if price_check > upper:
            direction = "LONG"
            entry     = upper + SLIPPAGE
            sl        = lower - SLIPPAGE
            target    = cpr["R1"]
        elif price_check < lower:
            direction = "SHORT"
            entry     = lower - SLIPPAGE
            sl        = upper + SLIPPAGE
            target    = cpr["S1"]
        else:
            save_state({"date": str(date.today()), "status": "no_trade"})
            return

        state = dict(
            date        = str(date.today()),
            symbol      = tradingsym,
            token       = token,
            direction   = direction,
            entry       = round(entry, 4),
            sl          = round(sl, 4),
            target      = round(target, 4),
            cpr_upper   = round(upper, 4),
            cpr_lower   = round(lower, 4),
            status      = "open",
            alerts_sent = [],
            last_pulse  = now.isoformat(),
        )
        save_state(state)

        # Fetch current LTP to show alongside calculated entry
        try:
            ltp_now = get_ltp(kite, tradingsym)
        except Exception:
            ltp_now = price_check

        arrow  = "📈" if direction == "LONG" else "📉"
        risk   = abs(entry - sl)     * PNL_PER_LOT * LOTS
        reward = abs(target - entry) * PNL_PER_LOT * LOTS
        rr_str = f"1 : {reward/risk:.1f}" if risk > 0 and reward > 0 else "–"

        breakout_side  = "above" if direction == "LONG" else "below"
        breakout_level = upper if direction == "LONG" else lower

        gap_pct = abs(ltp_now - entry) / entry * 100
        chase_note = ""
        if gap_pct > 0.05:
            side = "already past entry" if (
                (direction == "LONG" and ltp_now > entry) or
                (direction == "SHORT" and ltp_now < entry)
            ) else "hasn't reached entry"
            chase_note = f"\n⚠️  LTP {_p(ltp_now)} — {side} ({gap_pct:.2f}%). Skip if missed."

        send_telegram(
            f"💵 <b>USDINR — Signal Detected</b>  [{today_str}]\n"
            f"<code>{tradingsym}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{arrow}  <b>{direction}</b>  ·  {LOTS} lots NRML  ·  NSE CDS\n\n"
            f"  {price_label}          : {_p(price_check)}\n"
            f"  CPR breakout {breakout_side}  : {_p(breakout_level)}\n"
            f"  LTP now            : {_p(ltp_now)}\n\n"
            f"  <b>Entry</b>   {_p(entry)}\n"
            f"  <b>SL</b>      {_p(sl)}\n"
            f"  <b>Target</b>  {_p(target)}  (R1/S1)\n"
            f"  Risk {_r(risk)}  →  Reward {_r(reward)}  →  RR {rr_str}"
            f"{chase_note}\n\n"
            f"<i>Place order manually in Kite if not already done.\n"
            f"Reminders every 5 min while setup stays valid. EOD exit 16:45.</i>"
        )
        return

    # ── No-trade state — nothing to do ────────────────────────────────────────
    if state.get("status") == "no_trade":
        log.info("[USDINR] No signal today (CPR too wide or price inside CPR).")
        return

    # ── Signal active — fetch LTP ─────────────────────────────────────────────
    tradingsym = state["symbol"]
    direction  = state["direction"]
    entry      = float(state["entry"])
    sl         = float(state["sl"])
    target     = float(state["target"])
    cpr_upper  = float(state["cpr_upper"])
    cpr_lower  = float(state["cpr_lower"])
    arrow      = "📈" if direction == "LONG" else "📉"

    try:
        ltp = get_ltp(kite, tradingsym)
    except Exception as exc:
        log.error("[USDINR] LTP fetch failed: %s", exc)
        return

    if ltp <= 0:
        log.warning("[USDINR] LTP=%.4f invalid (market closed?). Skipping cycle.", ltp)
        return

    log.info("[USDINR] %-5s  LTP=%.4f  entry=%.4f  SL=%.4f  target=%.4f",
             direction, ltp, entry, sl, target)

    # ── EOD reminder (16:45) ─────────────────────────────────────────────────
    sq_off = now.replace(hour=SQ_OFF_HOUR, minute=SQ_OFF_MINUTE,
                         second=0, microsecond=0)
    if now >= sq_off:
        if not alert_sent(state, "eod_reminder"):
            mark_alert(state, "eod_reminder")
            state["status"] = "no_trade"
            save_state(state)
            send_telegram(
                f"⏰ <b>USDINR — EOD: Exit if In Trade</b>  [{today_str}]\n"
                f"<code>{tradingsym}</code>\n\n"
                f"{arrow} {direction}  ·  {LOTS} lots\n"
                f"  LTP    : {_p(ltp)}\n"
                f"  Entry  : {_p(entry)}\n"
                f"  SL     : {_p(sl)}   Target: {_p(target)}\n\n"
                f"<i>16:45 IST — Exit manually in Kite. CDS closes 17:00.</i>"
            )
        return

    # ── Check if CPR setup is still valid ────────────────────────────────────
    still_valid = (ltp > cpr_upper) if direction == "LONG" else (ltp < cpr_lower)

    if not still_valid:
        log.info("[USDINR] %-5s  LTP=%.4f back inside CPR [%.4f–%.4f] — signal invalidated.",
                 direction, ltp, cpr_lower, cpr_upper)
        return

    # ── CPR still broken — send 5-min reminder ───────────────────────────────
    last_pulse = datetime.fromisoformat(state.get("last_pulse", now.isoformat()))
    if (now - last_pulse).total_seconds() >= 5 * 60:
        state["last_pulse"] = now.isoformat()
        save_state(state)
        send_telegram(
            f"💵 <b>USDINR — Signal Active</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} <b>{direction}</b>  ·  <code>{tradingsym}</code>\n\n"
            f"  LTP    : <b>{_p(ltp)}</b>\n"
            f"  Entry  : {_p(entry)}\n"
            f"  SL     : {_p(sl)}\n"
            f"  Target : {_p(target)}\n\n"
            f"<i>Price outside CPR [{_p(cpr_lower)}–{_p(cpr_upper)}] — setup valid. Enter if suits you.</i>"
        )


# ─── MAIN ENTRY (called by APScheduler in main.py) ────────────────────────────
def run_usdinr_cpr_monitor() -> None:
    """Monitor USDINR CPR trade. Called every 5 min from 9:45 AM via main.py."""
    now = datetime.now()
    # Don't run before 9:45 — wait for a bit of price action after CDS open
    if now.hour < 9 or (now.hour == 9 and now.minute < 45):
        return

    try:
        kite = get_kite()
    except Exception as exc:
        log.error("[usdinr_monitor] Kite init failed: %s", exc)
        return

    try:
        _run_cycle(kite, now)
    except Exception as exc:
        log.error("[usdinr_monitor] Unhandled error: %s", exc)


# ─── STANDALONE ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_usdinr_cpr_monitor()
