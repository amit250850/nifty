#!/usr/bin/env python3
"""
goldm_morning_alert.py
======================
Pre-market GOLDM CPR alert — runs at 08:55 IST, sends a Telegram message
with today's CPR levels, conviction grade, exact entry/SL/target, and
risk/reward in ₹ before MCX opens at 09:00.

CPR Conviction Grades (AAK method)
------------------------------------
  🔴 HIGH     : CPR width < 0.10%  → Very compressed, big move expected
  🟡 MEDIUM   : CPR width 0.10–0.20% → Strong trending day
  🟢 STANDARD : CPR width 0.20–0.30% → Borderline narrow, trade with normal size
  ⚪ SKIP     : CPR width > 0.30%  → Wide or Neutral day, no trade

  ⭐ Virgin CPR bonus: CPR zone NOT touched in last 5 sessions → adds ⭐

MCX GOLDM Specs (as of April 2026)
------------------------------------
  Lot size  : 100 grams
  Quote unit: ₹ per 10 grams
  P&L/lot   : ₹10 per ₹1 price move
  MIS margin: ~₹30,000–38,000 per lot at current gold prices
  Session   : 09:00–23:30 IST  (CPR daytime session 09:00–17:00)

  ✅ With ₹50,000: trade 1 lot MIS comfortably
  ✅ No separate commodity fund needed — Zerodha single ledger since 2024

How to place the trade in Kite
---------------------------------
  1. Watch first hourly candle (09:00–10:00)
  2. If it closes ABOVE upper_cpr → place LIMIT BUY at upper_cpr
     Product: MIS | SL: lower_cpr | Target: R1
  3. If it closes BELOW lower_cpr → place LIMIT SELL at lower_cpr
     Product: MIS | SL: upper_cpr | Target: S1
  4. SKIP if first candle closes inside CPR band

Schedule (add to crontab or Windows Task Scheduler):
  55 8 * * 1-5  python /path/to/goldm_morning_alert.py

  Or add to main.py scheduler:
  scheduler.add_job(send_goldm_cpr_alert, "cron", hour=8, minute=55,
                    day_of_week="mon-fri", timezone=IST)

Setup
-----
  1. python login.py  (refresh access token daily)
  2. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
  3. python goldm_morning_alert.py  (test it manually first)
"""

import os
import sys
import logging
from datetime import date, timedelta, datetime

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import requests
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests --break-system-packages")
    sys.exit(1)

# ─── SETUP ────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── GOLDM SPECS ─────────────────────────────────────────────────────────────
LOT_GRAMS   = 100
QUOTE_UNIT  = 10
PNL_PER_LOT = LOT_GRAMS / QUOTE_UNIT    # ₹10 per lot per ₹1 move

NARROW_HIGH     = 0.10   # 🔴 HIGH conviction
NARROW_MEDIUM   = 0.20   # 🟡 MEDIUM conviction
NARROW_STANDARD = 0.30   # 🟢 STANDARD conviction
SLIPPAGE        = 0.50   # ₹/10g per side


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        log.error("KITE_API_KEY / KITE_ACCESS_TOKEN missing. Run login.py first.")
        sys.exit(1)
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT LOOKUP ────────────────────────────────────────────────────────
def get_front_month_token(kite: KiteConnect) -> tuple:
    """
    Find the current front-month GOLDM futures contract.
    Rolls to the next contract if within 5 days of expiry.
    Returns (instrument_token, tradingsymbol, expiry_date).
    """
    import pandas as pd
    instruments = kite.instruments("MCX")
    df = pd.DataFrame(instruments)
    df["expiry"] = pd.to_datetime(df["expiry"])
    goldm = df[
        (df["name"] == "GOLDM") &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= date.today())
    ].sort_values("expiry").reset_index(drop=True)

    if goldm.empty:
        raise RuntimeError("No active GOLDM contracts found. Check MCX instruments.")

    # Use the next contract if front month expires in ≤ 5 days
    front = goldm.iloc[0]
    days_to_expiry = (front["expiry"].date() - date.today()).days
    if days_to_expiry <= 5 and len(goldm) > 1:
        log.info("Front month expires in %d days — rolling to next contract", days_to_expiry)
        front = goldm.iloc[1]
        days_to_expiry = (front["expiry"].date() - date.today()).days

    log.info("Active contract: %s  (expiry %s, DTE=%d)",
             front["tradingsymbol"], front["expiry"].date(), days_to_expiry)
    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── HISTORICAL OHLC ──────────────────────────────────────────────────────────
def get_recent_daily(kite: KiteConnect, token: int, days: int = 10) -> list:
    """
    Fetch the last `days` trading days of daily OHLC.
    Kite returns dates in IST; weekends/holidays are automatically skipped.
    """
    to_dt   = date.today() - timedelta(days=1)   # yesterday
    from_dt = to_dt - timedelta(days=days + 5)   # buffer for weekends
    rows = kite.historical_data(token, from_dt, to_dt, interval="day",
                                continuous=False, oi=False)
    return rows


# ─── CPR CALCULATION ─────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)   # always ≥ lower (normalised)
    lower_cpr = min(TC, BC)
    R1        = 2 * P - L
    R2        = P + (H - L)
    S1        = 2 * P - H
    S2        = P - (H - L)
    width_pct = (upper_cpr - lower_cpr) / P * 100
    return dict(P=P, BC=BC, TC=TC,
                upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, R2=R2, S1=S1, S2=S2,
                width_pct=width_pct)


# ─── VIRGIN CPR CHECK ─────────────────────────────────────────────────────────
def is_virgin_cpr(cpr: dict, recent_rows: list) -> bool:
    """
    Virgin CPR = today's CPR zone was never touched in the last 5 sessions.
    If price never entered [lower_cpr, upper_cpr] in recent history → Virgin ⭐
    """
    upper = cpr["upper_cpr"]
    lower = cpr["lower_cpr"]
    recent = recent_rows[:-1]   # exclude yesterday (that's what built today's CPR)
    last5  = recent[-5:] if len(recent) >= 5 else recent
    for bar in last5:
        bar_h = float(bar["high"])
        bar_l = float(bar["low"])
        # Check if price entered the CPR zone at any point
        if bar_l <= upper and bar_h >= lower:
            return False   # CPR zone was touched — not virgin
    return True   # never touched → Virgin CPR ⭐


# ─── CONVICTION GRADE ────────────────────────────────────────────────────────
def grade(width_pct: float, virgin: bool) -> tuple:
    """Returns (emoji, label, should_trade)"""
    if width_pct < NARROW_HIGH:
        g = ("🔴", "HIGH", True)
    elif width_pct < NARROW_MEDIUM:
        g = ("🟡", "MEDIUM", True)
    elif width_pct < NARROW_STANDARD:
        g = ("🟢", "STANDARD", True)
    else:
        return ("⚪", "SKIP — Wide/Neutral day", False)

    if virgin:
        return (g[0] + "⭐", g[1] + " + Virgin CPR", g[2])
    return g


# ─── TELEGRAM SENDER ─────────────────────────────────────────────────────────
def send_telegram(msg: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        log.warning("Telegram not configured — printing to console instead.")
        print("\n" + "="*60)
        print(msg)
        print("="*60 + "\n")
        return
    url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id":    TG_CHAT_ID,
        "text":       msg,
        "parse_mode": "HTML",
    }, timeout=10)
    if resp.status_code == 200:
        log.info("Telegram alert sent ✅")
    else:
        log.error("Telegram failed: %s  %s", resp.status_code, resp.text)


# ─── BUDGET HELPER ────────────────────────────────────────────────────────────
def budget_lines(cpr: dict, lots: int = 1) -> str:
    price        = cpr["P"]
    lot_value    = price * (LOT_GRAMS / QUOTE_UNIT)
    nrml_margin  = lot_value * 0.045     # ~4.5% MCX SPAN+exposure
    mis_margin   = nrml_margin * 0.50    # Zerodha MIS = 50% NRML

    sl_range     = cpr["upper_cpr"] - cpr["lower_cpr"]
    sl_risk_lot  = sl_range * PNL_PER_LOT
    tgt_range    = cpr["R1"] - cpr["upper_cpr"]
    tgt_profit_lot = tgt_range * PNL_PER_LOT

    return (
        f"💰 <b>Budget (1 lot MIS)</b>\n"
        f"   Lot value   : ₹{lot_value:,.0f}\n"
        f"   MIS margin  : ~₹{mis_margin:,.0f}  (intraday)\n"
        f"   Max SL risk : ₹{sl_risk_lot:,.0f} per lot\n"
        f"   Target gain : ₹{tgt_profit_lot:,.0f} per lot  (R1)\n"
        f"   RR ratio    : 1:{tgt_profit_lot/sl_risk_lot:.1f}"
        if sl_risk_lot > 0 else ""
    )


# ─── MAIN ALERT ──────────────────────────────────────────────────────────────
def send_goldm_cpr_alert() -> None:
    log.info("=" * 55)
    log.info("GOLDM CPR Morning Alert — %s", datetime.now().strftime("%d %b %Y %H:%M IST"))
    log.info("=" * 55)

    kite = get_kite()

    # Get active contract
    token, symbol, expiry = get_front_month_token(kite)

    # Fetch recent daily bars (need yesterday + last 5 for Virgin CPR check)
    rows = get_recent_daily(kite, token, days=10)
    if len(rows) < 2:
        log.error("Not enough historical data. Market may be holiday. Aborting.")
        return

    yesterday = rows[-1]
    prev_H    = float(yesterday["high"])
    prev_L    = float(yesterday["low"])
    prev_C    = float(yesterday["close"])
    prev_date = yesterday["date"].date() if hasattr(yesterday["date"], "date") \
                else yesterday["date"]

    log.info("Yesterday (%s): H=₹%s  L=₹%s  C=₹%s",
             prev_date, f"{prev_H:,.0f}", f"{prev_L:,.0f}", f"{prev_C:,.0f}")

    # Calculate CPR
    cpr = calc_cpr(prev_H, prev_L, prev_C)

    # Conviction grade
    virgin       = is_virgin_cpr(cpr, rows)
    emoji, label, should_trade = grade(cpr["width_pct"], virgin)

    today_str = date.today().strftime("%d %b %Y")
    dow       = date.today().strftime("%A")

    if not should_trade:
        # Wide/Neutral day — short alert
        msg = (
            f"📊 <b>GOLDM CPR — {today_str} ({dow})</b>\n"
            f"Contract: {symbol}  (exp {expiry})\n\n"
            f"{emoji}  <b>{label}</b>\n"
            f"CPR width: {cpr['width_pct']:.3f}%  → Wide/Neutral\n\n"
            f"🚫 <b>NO TRADE TODAY</b>\n"
            f"Range-bound day expected. Sit on hands.\n\n"
            f"📐 <i>Levels for reference:</i>\n"
            f"   P    : ₹{cpr['P']:,.0f}\n"
            f"   Upper: ₹{cpr['upper_cpr']:,.0f}\n"
            f"   Lower: ₹{cpr['lower_cpr']:,.0f}\n"
            f"   R1   : ₹{cpr['R1']:,.0f}  |  S1: ₹{cpr['S1']:,.0f}"
        )
        send_telegram(msg)
        log.info("Wide/Neutral day — no trade alert sent.")
        return

    # ── NARROW DAY — full trade alert ─────────────────────────────────────
    # Entry / SL / Target for both directions (we don't know first candle yet)
    long_entry  = round(cpr["upper_cpr"] + SLIPPAGE, 1)
    long_sl     = round(cpr["lower_cpr"] - SLIPPAGE, 1)
    long_target = round(cpr["R1"], 1)

    short_entry  = round(cpr["lower_cpr"] - SLIPPAGE, 1)
    short_sl     = round(cpr["upper_cpr"] + SLIPPAGE, 1)
    short_target = round(cpr["S1"], 1)

    # Risk/reward per lot
    long_risk   = (long_entry  - long_sl)   * PNL_PER_LOT
    long_reward = (long_target - long_entry) * PNL_PER_LOT if long_target > long_entry else 0
    short_risk  = (short_sl - short_entry)  * PNL_PER_LOT
    short_reward= (short_entry - short_target) * PNL_PER_LOT if short_target < short_entry else 0

    long_rr  = f"1:{long_reward/long_risk:.1f}"   if long_risk  > 0 else "N/A"
    short_rr = f"1:{short_reward/short_risk:.1f}" if short_risk > 0 else "N/A"

    lot_value   = cpr["P"] * (LOT_GRAMS / QUOTE_UNIT)
    mis_margin  = lot_value * 0.045 * 0.50    # 50% of 4.5% NRML

    # Virgin CPR message
    virgin_line = "⭐ <b>VIRGIN CPR</b> — not touched in last 5 sessions. Extra conviction.\n" \
                  if virgin else ""

    msg = (
        f"🥇 <b>GOLDM CPR Alert — {today_str} ({dow})</b>\n"
        f"Contract : {symbol}  (expiry {expiry})\n"
        f"Based on : {prev_date} OHLC\n\n"

        f"{emoji}  <b>Conviction: {label}</b>\n"
        f"CPR width: {cpr['width_pct']:.3f}%  → NARROW day → BREAKOUT expected\n"
        f"{virgin_line}\n"

        f"📐 <b>CPR Levels</b>\n"
        f"   Pivot (P) : ₹{cpr['P']:,.0f}\n"
        f"   Upper CPR : ₹{cpr['upper_cpr']:,.0f}\n"
        f"   Lower CPR : ₹{cpr['lower_cpr']:,.0f}\n"
        f"   R1        : ₹{cpr['R1']:,.0f}  |  R2: ₹{cpr['R2']:,.0f}\n"
        f"   S1        : ₹{cpr['S1']:,.0f}  |  S2: ₹{cpr['S2']:,.0f}\n\n"

        f"⏰ <b>Wait for 09:00–10:00 candle close, then:</b>\n\n"

        f"📈 <b>LONG setup</b>  (if candle closes ABOVE ₹{cpr['upper_cpr']:,.0f})\n"
        f"   Buy  : ₹{long_entry:,.1f}  (limit, product=MIS)\n"
        f"   SL   : ₹{long_sl:,.1f}\n"
        f"   Tgt  : ₹{long_target:,.0f}  (R1)\n"
        f"   Risk : ₹{long_risk:,.0f}/lot  |  RR {long_rr}\n\n"

        f"📉 <b>SHORT setup</b>  (if candle closes BELOW ₹{cpr['lower_cpr']:,.0f})\n"
        f"   Sell : ₹{short_entry:,.1f}  (limit, product=MIS)\n"
        f"   SL   : ₹{short_sl:,.1f}\n"
        f"   Tgt  : ₹{short_target:,.0f}  (S1)\n"
        f"   Risk : ₹{short_risk:,.0f}/lot  |  RR {short_rr}\n\n"

        f"🚫 <b>SKIP</b> if candle closes inside CPR band "
        f"(₹{cpr['lower_cpr']:,.0f}–₹{cpr['upper_cpr']:,.0f})\n\n"

        f"💰 <b>Budget guide (1 lot MIS, ₹50K account)</b>\n"
        f"   Lot value  : ₹{lot_value:,.0f}\n"
        f"   MIS margin : ~₹{mis_margin:,.0f}  ✅ fits ₹50K\n"
        f"   Max loss   : ₹{long_risk:,.0f}  ({long_risk/50000*100:.1f}% of ₹50K)\n\n"

        f"ℹ️ <i>No separate commodity transfer needed — Zerodha single ledger.\n"
        f"MIS auto squares off before MCX close. EOD carry: use NRML instead.</i>"
    )

    send_telegram(msg)
    log.info("CPR alert sent — %s conviction, width=%.3f%%", label, cpr["width_pct"])
    log.info("LONG:  entry=₹%s  SL=₹%s  tgt=₹%s  risk=₹%s",
             f"{long_entry:,.0f}", f"{long_sl:,.0f}",
             f"{long_target:,.0f}", f"{long_risk:,.0f}")
    log.info("SHORT: entry=₹%s  SL=₹%s  tgt=₹%s  risk=₹%s",
             f"{short_entry:,.0f}", f"{short_sl:,.0f}",
             f"{short_target:,.0f}", f"{short_risk:,.0f}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    send_goldm_cpr_alert()
