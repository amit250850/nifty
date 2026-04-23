#!/usr/bin/env python3
"""
mcx_cpr_monitor.py
==================
Intraday CPR trade monitor for MCX GOLDGUINEA, SILVERMIC, CRUDEOILM and NICKEL.
Runs every 5 min from 10:05 AM – 17:55 PM IST via APScheduler in main.py.

Since these symbols have no auto-execute, trades are placed manually in Kite
after the 08:55 morning alert. This monitor:

  1. At 10:05 IST — Detects which direction was signaled (first 1H candle
     close vs CPR levels). If a signal exists, initialises a state file and
     sends a "Trade Active" Telegram confirming entry levels to watch.

  2. Every 5 min after — Fetches LTP, checks:
       • SL hit (price crossed back through CPR zone)
       • Target hit (R1 for LONG, S1 for SHORT)
       • Approaching SL  (within 30% of SL move) — alert once
       • Near target     (70% of target move captured) — alert once
       • Hourly P&L pulse (sent every 60 min while trade open)

  3. At 17:55 IST — Sends EOD square-off reminder. MCX trades until 23:30
     but CPR day trades are closed end-of-day.

State files (one per symbol):
  data/goldguinea_cpr_state.json
  data/silverm_cpr_state.json

  Created at 10:05, reset each day. Persists across bot restarts.

SYMBOL CONFIGS (must match morning alerts):
  GOLDGUINEA : 3 lots, PNL_PER_LOT=1,  slippage=4.0/lot,  commission=65
  SILVERMIC  : 1 lot,  PNL_PER_LOT=1,  slippage=10.0/kg,  commission=65
  SILVERM excluded — margin ₹1.25L exceeds ₹87K capital limit
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
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests pandas")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


# Auto square-off reminder time (IST)
SQ_OFF_HOUR   = 22
SQ_OFF_MINUTE = 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Per-symbol specs — must match morning alerts exactly
SYMBOL_SPECS = {
    "GOLDGUINEA": dict(
        kite_name    = "GOLDGUINEA",
        lots         = 3,       # ₹87K account: 3 lots × ₹17,772 = ₹53,316 margin
        pnl_per_lot  = 1,       # ₹1 per lot per ₹1 lot-price move
        commission   = 65,
        slippage     = 4.0,     # ₹/lot per side
        narrow_pct   = 0.35,    # must match morning alert threshold
        expiry_roll  = 7,       # roll N days before expiry
        state_file   = os.path.join(DATA_DIR, "goldguinea_cpr_state.json"),
        emoji        = "💛",
        quote_unit   = "lot",
    ),
    "SILVERMIC": dict(
        kite_name    = "SILVERMIC",
        lots         = 1,
        pnl_per_lot  = 1,       # ₹1 per lot per ₹1/kg move (1 kg lot)
        commission   = 65,
        slippage     = 10.0,    # ₹/kg per side
        narrow_pct   = 0.45,    # must match silvermic_morning_alert.py threshold
        expiry_roll  = 5,
        state_file   = os.path.join(DATA_DIR, "silvermic_cpr_state.json"),
        emoji        = "🪙",
        quote_unit   = "kg",
    ),
    "NATURALGAS": dict(
        kite_name    = "NATURALGAS",
        lots         = 1,
        pnl_per_lot  = 1_250,   # 1,250 MMBTU × ₹1/MMBTU = ₹1,250/lot/₹1 move
        commission   = 65,
        slippage     = 0.50,    # ₹/MMBTU per side
        narrow_pct   = 0.50,    # must match naturalgas_morning_alert.py threshold
        expiry_roll  = 5,
        state_file   = os.path.join(DATA_DIR, "naturalgas_cpr_state.json"),
        emoji        = "🔥",
        quote_unit   = "MMBTU",
    ),
    # SILVERM excluded — margin ₹1.25L exceeds ₹87K capital limit
    "CRUDEOILM": dict(
        kite_name    = "CRUDEOILM",
        lots         = 1,
        pnl_per_lot  = 10,      # 10 barrels × ₹1/barrel = ₹10/lot/₹1 move
        commission   = 65,
        slippage     = 1.0,     # ₹1/barrel per side (1 tick)
        narrow_pct   = 0.50,    # must match crudeoilm_morning_alert.py threshold
        expiry_roll  = 5,
        state_file   = os.path.join(DATA_DIR, "crudeoilm_cpr_state.json"),
        emoji        = "🛢",
        quote_unit   = "bbl",
    ),
    "NICKEL": dict(
        kite_name    = "NICKEL",
        lots         = 1,
        pnl_per_lot  = 250,     # 250 kg × ₹1/kg = ₹250/lot/₹1 move (MCX reduced lot)
        commission   = 65,
        slippage     = 0.20,    # ₹0.20/kg per side (2 ticks)
        narrow_pct   = 0.40,    # must match nickel_morning_alert.py threshold
        expiry_roll  = 5,
        state_file   = os.path.join(DATA_DIR, "nickel_cpr_state.json"),
        emoji        = "🔩",
        quote_unit   = "kg",
    ),
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _p(v): return f"₹{v:,.2f}"
def _r(v): return f"₹{v:,.0f}"


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing KITE_API_KEY / KITE_ACCESS_TOKEN. Run login.py.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT ───────────────────────────────────────────────────────────────
def get_front_month(kite: KiteConnect, kite_name: str, roll_days: int) -> tuple:
    """Return (token, tradingsymbol, expiry) for active front-month contract."""
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    contracts = df[
        (df["name"] == kite_name) &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= date.today())
    ].sort_values("expiry").reset_index(drop=True)

    if contracts.empty:
        raise RuntimeError(f"No active {kite_name} contracts found.")

    front = contracts.iloc[0]
    dte   = (front["expiry"].date() - date.today()).days
    if dte <= roll_days and len(contracts) > 1:
        front = contracts.iloc[1]

    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── OHLC / CPR ───────────────────────────────────────────────────────────────
def get_yesterday_ohlc(kite: KiteConnect, token: int) -> dict | None:
    """Fetch yesterday's OHLC for CPR calculation."""
    to_dt   = date.today() - timedelta(days=1)
    from_dt = to_dt - timedelta(days=5)
    try:
        rows = kite.historical_data(token, from_dt, to_dt,
                                    interval="day", continuous=False, oi=False)
        if not rows:
            return None
        r = rows[-1]
        return {"H": float(r["high"]), "L": float(r["low"]), "C": float(r["close"])}
    except Exception as exc:
        log.error("OHLC fetch failed: %s", exc)
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
        # First candle of the MCX day session
        first = rows[0]
        if pd.Timestamp(first["date"]).hour == 9:
            return float(first["close"])
        return None
    except Exception as exc:
        log.error("First hourly close fetch failed: %s", exc)
        return None


def get_ltp(kite: KiteConnect, tradingsymbol: str) -> float:
    """Fetch current LTP for an MCX symbol."""
    data = kite.ltp([f"MCX:{tradingsymbol}"])
    return float(data[f"MCX:{tradingsymbol}"]["last_price"])


# ─── STATE FILE ───────────────────────────────────────────────────────────────
def load_state(state_file: str) -> dict:
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not load state %s: %s", state_file, exc)
        return {}


def save_state(state: dict, state_file: str) -> None:
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
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
            log.info("[mcx_monitor] Telegram ✅")
        else:
            log.error("[mcx_monitor] Telegram failed %s: %s", r.status_code, r.text[:80])
    except Exception as exc:
        log.error("[mcx_monitor] Telegram error: %s", exc)


# ─── SINGLE SYMBOL MONITOR CYCLE ──────────────────────────────────────────────
def _monitor_symbol(kite: KiteConnect, symbol: str, spec: dict, now: datetime) -> None:
    today_str  = now.strftime("%a %d %b %Y")
    state_file = spec["state_file"]
    lots       = spec["lots"]
    pnl_per_lot = spec["pnl_per_lot"]
    commission  = spec["commission"]
    slip        = spec["slippage"]
    emoji       = spec["emoji"]

    state = load_state(state_file)

    # ── Reset if state is from a previous day ─────────────────────────────────
    if state.get("date") and state["date"] != str(date.today()):
        log.info("[%s] Stale state from %s — resetting.", symbol, state["date"])
        state = {}
        save_state(state, state_file)

    # ── Initialise state at first run (10:05 checks first 1H candle) ──────────
    if not state:
        log.info("[%s] No state yet — detecting signal from first 1H candle.", symbol)

        try:
            token, tradingsym, expiry = get_front_month(
                kite, spec["kite_name"], spec["expiry_roll"]
            )
        except Exception as exc:
            log.error("[%s] get_front_month failed: %s", symbol, exc)
            return

        ohlc = get_yesterday_ohlc(kite, token)
        if not ohlc:
            log.warning("[%s] No yesterday OHLC — skipping init.", symbol)
            return

        cpr = calc_cpr(ohlc["H"], ohlc["L"], ohlc["C"])
        width = cpr["width_pct"]

        # Only initialise if it was a narrow CPR day (signal was sent this morning)
        log.info("[%s] CPR check — width=%.3f%%  threshold=%.2f%%  %s",
                 symbol, width, spec["narrow_pct"],
                 "NARROW ✓" if width < spec["narrow_pct"] else "WIDE ✗ → no trade today")
        if width >= spec["narrow_pct"]:
            # Save a "no_trade" state so we don't keep retrying
            save_state({"date": str(date.today()), "status": "no_trade"}, state_file)
            return

        upper = cpr["upper_cpr"]
        lower = cpr["lower_cpr"]

        # From 9:45 use live LTP directly; after 10:05 use completed 1H candle close
        if now.hour == 9 or (now.hour == 10 and now.minute < 5):
            price_check = get_ltp(kite, tradingsym)
            first_close = price_check   # alias for Telegram message
            price_label = "LTP"
            log.info("[%s] Using LTP=%.2f vs CPR[%.2f – %.2f] (pre-candle check at %s)",
                     symbol, price_check, lower, upper, now.strftime("%H:%M"))
        else:
            price_check = get_first_hourly_close(kite, token)
            if price_check is None:
                log.warning("[%s] First hourly candle not ready yet.", symbol)
                return
            first_close = price_check
            price_label = "1H close"
            log.info("[%s] First candle close=%.2f  CPR[%.2f – %.2f]", symbol, price_check, lower, upper)

        log.info("[%s] Direction check: %.2f vs CPR[%.2f – %.2f] → %s",
                 symbol, price_check, lower, upper,
                 "LONG" if price_check > upper else "SHORT" if price_check < lower else "INSIDE — no trade")

        if price_check > upper:
            direction = "LONG"
            entry  = upper + slip
            sl     = lower - slip
            target = cpr["R1"]
        elif price_check < lower:
            direction = "SHORT"
            entry  = lower - slip
            sl     = upper + slip
            target = cpr["S1"]
        else:
            save_state({"date": str(date.today()), "status": "no_trade"}, state_file)
            return

        state = dict(
            date        = str(date.today()),
            symbol      = tradingsym,
            token       = token,
            direction   = direction,
            entry       = round(entry, 2),
            sl          = round(sl, 2),
            target      = round(target, 2),
            lots        = lots,
            cpr_upper   = round(upper, 2),
            cpr_lower   = round(lower, 2),
            status      = "open",
            alerts_sent = [],
            last_pulse  = now.isoformat(),
        )
        save_state(state, state_file)

        # Fetch current LTP to show alongside entry (price may have moved)
        try:
            ltp_now = get_ltp(kite, tradingsym)
        except Exception:
            ltp_now = first_close

        # Signal detected → send Telegram
        arrow = "📈" if direction == "LONG" else "📉"
        risk   = abs(entry - sl)     * pnl_per_lot * lots
        reward = abs(target - entry) * pnl_per_lot * lots
        rr_str = f"1 : {reward/risk:.1f}" if risk > 0 and reward > 0 else "–"

        # Label and boundary differ by direction
        if direction == "LONG":
            breakout_label = "CPR breakout above"
            breakout_level = upper
        else:
            breakout_label = "CPR breakout below"
            breakout_level = lower

        # Warn user if LTP has moved far from entry (don't chase)
        gap = abs(ltp_now - entry)
        gap_pct = gap / entry * 100
        chase_note = ""
        if gap_pct > 0.3:
            side = "already past entry" if (direction == "LONG" and ltp_now > entry) or (direction == "SHORT" and ltp_now < entry) else "hasn't reached entry"
            chase_note = f"\n⚠️  LTP {_p(ltp_now)} — {side} by {gap_pct:.1f}%. Skip if missed."

        send_telegram(
            f"{emoji} <b>{symbol} — Trade Signal Detected</b>  [{today_str}]\n"
            f"<code>{tradingsym}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{arrow}  <b>{direction}</b>  ·  {lots} lot{'s' if lots > 1 else ''} NRML\n\n"
            f"  {price_label:<18} : {_p(first_close)}\n"
            f"  {breakout_label} : {_p(breakout_level)}\n"
            f"  LTP now            : {_p(ltp_now)}\n\n"
            f"  <b>Entry</b>   {_p(entry)}\n"
            f"  <b>SL</b>      {_p(sl)}\n"
            f"  <b>Target</b>  {_p(target)}  (R1/S1)\n"
            f"  Risk {_r(risk)}  →  Reward {_r(reward)}  →  RR {rr_str}"
            f"{chase_note}\n\n"
            f"<i>Place order manually in Kite if not already done.\n"
            f"Monitor will track LTP and alert on SL / target hits.</i>"
        )
        return

    # ── No-trade state — nothing to do ────────────────────────────────────────
    if state.get("status") == "no_trade":
        log.info("[%s] No signal today (CPR too wide or first candle inside CPR).", symbol)
        return

    # ── Signal active — fetch LTP ─────────────────────────────────────────────
    tradingsym = state["symbol"]
    token      = state["token"]
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
        log.error("[%s] LTP fetch failed: %s", symbol, exc)
        return

    log.info("[%s] %-5s  LTP=%s  entry=%s  SL=%s  target=%s",
             symbol, direction, _p(ltp), _p(entry), _p(sl), _p(target))

    total_risk   = abs(entry - sl)
    total_reward = abs(target - entry)

    if direction == "LONG":
        risk_used     = max(0.0, entry - ltp)
        reward_earned = max(0.0, ltp - entry)
        sl_hit        = ltp <= sl
        target_hit    = ltp >= target
        still_valid   = ltp > cpr_upper
    else:
        risk_used     = max(0.0, ltp - entry)
        reward_earned = max(0.0, entry - ltp)
        sl_hit        = ltp >= sl
        target_hit    = ltp <= target
        still_valid   = ltp < cpr_lower

    risk_pct   = risk_used   / total_risk   if total_risk   > 0 else 0
    reward_pct = reward_earned / total_reward if total_reward > 0 else 0
    net_pnl    = (reward_earned if direction == "LONG" else reward_earned) * pnl_per_lot * lots - commission

    # ── EOD reminder ─────────────────────────────────────────────────────────
    sq_off_time = now.replace(hour=SQ_OFF_HOUR, minute=SQ_OFF_MINUTE,
                              second=0, microsecond=0)
    if now >= sq_off_time:
        if not alert_sent(state, "eod_reminder"):
            mark_alert(state, "eod_reminder")
            state["status"] = "no_trade"
            save_state(state, state_file)
            pnl_sign = "+" if net_pnl >= 0 else ""
            send_telegram(
                f"⏰ {emoji} <b>{symbol} — EOD Square-Off</b>  [{today_str}]\n"
                f"<code>{tradingsym}</code>\n\n"
                f"{arrow} {direction}  ·  {lots} lot{'s' if lots > 1 else ''}\n"
                f"  LTP    : {_p(ltp)}\n"
                f"  Entry  : {_p(entry)}\n"
                f"  SL     : {_p(sl)}   Target : {_p(target)}\n"
                f"  Est. P&L : {pnl_sign}{_r(net_pnl)}\n\n"
                f"<b>Exit manually in Kite if still in this trade.</b>"
            )
        return

    # ── Target hit ───────────────────────────────────────────────────────────
    if target_hit and not alert_sent(state, "target_hit"):
        mark_alert(state, "target_hit")
        state["status"] = "no_trade"
        save_state(state, state_file)
        send_telegram(
            f"🎯 {emoji} <b>{symbol} — TARGET HIT!</b>  [{now.strftime('%H:%M')} IST]\n"
            f"<code>{tradingsym}</code>\n\n"
            f"{arrow} {direction}  ·  {lots} lot{'s' if lots > 1 else ''}\n"
            f"  LTP    : <b>{_p(ltp)}</b>\n"
            f"  Target : {_p(target)}  ✅\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Est. P&L : <b>+{_r(abs(target - entry) * pnl_per_lot * lots - commission)}</b>\n\n"
            f"<b>Exit trade in Kite now. Well done!</b>"
        )
        return

    # ── SL hit ───────────────────────────────────────────────────────────────
    if sl_hit and not alert_sent(state, "sl_hit"):
        mark_alert(state, "sl_hit")
        state["status"] = "no_trade"
        save_state(state, state_file)
        send_telegram(
            f"🛑 {emoji} <b>{symbol} — STOP LOSS HIT</b>  [{now.strftime('%H:%M')} IST]\n"
            f"<code>{tradingsym}</code>\n\n"
            f"{arrow} {direction}  ·  {lots} lot{'s' if lots > 1 else ''}\n"
            f"  LTP    : <b>{_p(ltp)}</b>\n"
            f"  SL     : {_p(sl)}  ❌\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Est. Loss : <b>-{_r(abs(entry - sl) * pnl_per_lot * lots + commission)}</b>\n\n"
            f"<b>Exit trade in Kite now. Take the loss, protect capital.</b>"
        )
        return

    # ── CPR rejection — signal invalidated ───────────────────────────────────
    if not still_valid:
        log.info("[%s] %-5s  LTP=%s — back inside CPR [%.2f–%.2f], signal invalidated.",
                 symbol, direction, _p(ltp), cpr_lower, cpr_upper)
        if not alert_sent(state, "cpr_rejection"):
            mark_alert(state, "cpr_rejection")
            state["status"] = "no_trade"
            save_state(state, state_file)
            send_telegram(
                f"⛔ {emoji} <b>{symbol} — SIGNAL INVALIDATED</b>  [{now.strftime('%H:%M')} IST]\n"
                f"<code>{tradingsym}</code>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{arrow} <b>{direction}</b> breakout failed — price returned inside CPR\n\n"
                f"  LTP now  : <b>{_p(ltp)}</b>\n"
                f"  CPR zone : {_p(cpr_lower)} – {_p(cpr_upper)}\n"
                f"  Entry was: {_p(entry)}\n\n"
                f"🚫 <b>Do NOT enter this trade.</b>\n"
                f"📌 <b>If already in — hold. Let SL manage the exit.</b>\n\n"
                f"<i>Breakout failed pre-entry. Price back inside CPR.\n"
                f"Wait for tomorrow's signal.</i>"
            )
        else:
            save_state(state, state_file)
        return

    # ── Entry window closed (after 11:30 IST — too late for fresh entry) ─────
    entry_window_open = now.hour < 11 or (now.hour == 11 and now.minute <= 30)
    if not entry_window_open and not alert_sent(state, "entry_window_closed"):
        mark_alert(state, "entry_window_closed")
        save_state(state, state_file)
        send_telegram(
            f"⏰ {emoji} <b>{symbol} — Entry Window Closed</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} {direction}  ·  <code>{tradingsym}</code>\n\n"
            f"  LTP    : {_p(ltp)}\n"
            f"  Entry  : {_p(entry)}  ·  SL : {_p(sl)}  ·  Target : {_p(target)}\n\n"
            f"<b>Past 11:30 IST — do not enter a fresh CPR breakout trade.</b>\n"
            f"<i>If already in: hold and let SL / target manage the exit.</i>"
        )

    # ── Near-target alert (once, at 70% of reward captured) ──────────────────
    if reward_pct >= 0.70 and not alert_sent(state, "near_target"):
        mark_alert(state, "near_target")
        save_state(state, state_file)
        send_telegram(
            f"🎯 {emoji} <b>{symbol} — Near Target</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} {direction}  ·  <code>{tradingsym}</code>\n\n"
            f"  LTP    : <b>{_p(ltp)}</b>  ({reward_pct*100:.0f}% of move done)\n"
            f"  Target : {_p(target)}\n"
            f"  Entry  : {_p(entry)}\n\n"
            f"<i>Consider trailing SL to breakeven ({_p(entry)}) to protect profit.</i>"
        )

    # ── Near-SL alert (once, at 70% of risk used) ────────────────────────────
    if risk_pct >= 0.70 and not alert_sent(state, "near_sl"):
        mark_alert(state, "near_sl")
        save_state(state, state_file)
        send_telegram(
            f"⚠️ {emoji} <b>{symbol} — Approaching SL</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} {direction}  ·  <code>{tradingsym}</code>\n\n"
            f"  LTP    : <b>{_p(ltp)}</b>  ({risk_pct*100:.0f}% of risk used)\n"
            f"  SL     : {_p(sl)}\n"
            f"  Entry  : {_p(entry)}\n\n"
            f"<i>Price is close to your stop loss. Be ready to exit.</i>"
        )

    # ── Entry level reached alert (gap-up retrace scenario, once) ────────────
    gap_from_entry = (ltp - entry) if direction == "LONG" else (entry - ltp)
    gap_pct        = gap_from_entry / entry * 100
    at_entry       = abs(gap_pct) <= 0.15

    if at_entry and entry_window_open and not alert_sent(state, "entry_reached"):
        mark_alert(state, "entry_reached")
        save_state(state, state_file)
        send_telegram(
            f"📍 {emoji} <b>{symbol} — Entry Level Reached</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} <b>{direction}</b>  ·  <code>{tradingsym}</code>\n\n"
            f"  LTP    : <b>{_p(ltp)}</b>\n"
            f"  Entry  : {_p(entry)}\n"
            f"  SL     : {_p(sl)}\n"
            f"  Target : {_p(target)}\n\n"
            f"<b>Price is at entry level — you can enter now if you haven't.</b>\n"
            f"<i>Watch: if price slips into CPR zone [{_p(cpr_lower)}–{_p(cpr_upper)}], skip trade.</i>"
        )

    # ── Hourly status pulse (silent monitoring — one update per hour) ─────────
    last_pulse = datetime.fromisoformat(state.get("last_pulse", now.isoformat()))
    if (now - last_pulse).total_seconds() >= 60 * 60:
        state["last_pulse"] = now.isoformat()
        save_state(state, state_file)
        pnl_sign = "+" if net_pnl >= 0 else ""
        send_telegram(
            f"{emoji} <b>{symbol}</b>  [{now.strftime('%H:%M')} IST]\n"
            f"{arrow} {direction}  ·  LTP <b>{_p(ltp)}</b>  ·  "
            f"Est. P&L {pnl_sign}{_r(net_pnl)}\n"
            f"SL {_p(sl)}  ·  Target {_p(target)}"
        )


# ─── MAIN ENTRY (called by APScheduler in main.py) ────────────────────────────
def run_mcx_cpr_monitor() -> None:
    """Monitor GOLDGUINEA and SILVERM CPR trades. Called every 5 min from 9:45 AM."""
    now = datetime.now()
    # Don't run before 9:45 — MCX needs a few minutes of price action after open
    if now.hour < 9 or (now.hour == 9 and now.minute < 45):
        return

    try:
        kite = get_kite()
    except Exception as exc:
        log.error("[mcx_monitor] Kite init failed: %s", exc)
        return

    for symbol, spec in SYMBOL_SPECS.items():
        try:
            _monitor_symbol(kite, symbol, spec, now)
        except Exception as exc:
            log.error("[mcx_monitor][%s] Unhandled error: %s", symbol, exc)


# ─── STANDALONE ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_mcx_cpr_monitor()
