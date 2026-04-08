#!/usr/bin/env python3
"""
goldpetal_morning_alert.py
==========================
Pre-market MCX GOLDPETAL CPR alert.
Runs at 08:55 IST → Telegram message arrives before MCX opens at 09:00.

Strategy  : AAK CPR Breakout — 10 lots GOLDPETAL, NRML product
Conviction: based on CPR width + Virgin CPR check (last 5 sessions)
Action    : Watch 09:00–10:00 candle, then place limit order manually in Kite

GOLDPETAL specs
---------------
  Lot size      : 1 gram
  Quote         : ₹ per gram
  P&L per lot   : ₹1 per ₹1 move
  10 lots P&L   : ₹10 per ₹1 move
  Commission    : ₹55 flat per trade (Zerodha per-order model)
  NRML margin   : ~₹2,200/lot → ₹22,000 for 10 lots → ₹28,000 free on ₹50K
  Slippage      : ₹0.25/gram per side

Run schedule (add to Windows Task Scheduler or crontab)
--------------------------------------------------------
  Time    : 08:55 IST  Mon–Fri
  Command : python D:\\Nifty\\NiftySignalBot\\goldpetal_morning_alert.py

Or via main.py APScheduler (already wired in at 08:55).

Requires login.py to be run first each morning to refresh access_token.
"""

import os
import sys
import logging
from datetime import date, timedelta, datetime

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import requests
    import pandas as pd
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests pandas "
          "--break-system-packages")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

LOTS         = 10
PNL_PER_LOT  = 1       # ₹1/lot per ₹1 price move
COMMISSION   = 55      # flat per trade
SLIPPAGE     = 0.25    # ₹/gram per side

NARROW_HIGH     = 0.10   # 🔴 HIGH
NARROW_MEDIUM   = 0.20   # 🟡 MEDIUM
NARROW_STANDARD = 0.30   # 🟢 STANDARD

MARGIN_PCT   = 0.15    # 15% MCX NRML margin on GOLDPETAL
CAPITAL      = 50_000  # ₹50,000 reference capital

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        sys.exit("ERROR: Missing KITE_API_KEY / KITE_ACCESS_TOKEN in .env. "
                 "Run login.py first.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT ───────────────────────────────────────────────────────────────
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
    if dte <= 5 and len(petal) > 1:          # roll 5 days before expiry
        front = petal.iloc[1]
        dte   = (front["expiry"].date() - date.today()).days

    log.info("Contract: %s  expiry=%s  DTE=%d",
             front["tradingsymbol"], front["expiry"].date(), dte)
    return int(front["instrument_token"]), front["tradingsymbol"], \
           front["expiry"].date()


# ─── OHLC ────────────────────────────────────────────────────────────────────
def get_recent_daily(kite: KiteConnect, token: int, n: int = 10) -> list:
    """Last n trading days of daily OHLC (includes today-minus-1 at minimum)."""
    to_dt   = date.today() - timedelta(days=1)
    from_dt = to_dt - timedelta(days=n + 5)
    rows = kite.historical_data(token, from_dt, to_dt,
                                interval="day", continuous=False, oi=False)
    return rows


# ─── CPR MATH ────────────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)   # always normalised — prevents SL-above-entry bug
    lower_cpr = min(TC, BC)
    R1 = 2 * P - L;  R2 = P + (H - L)
    S1 = 2 * P - H;  S2 = P - (H - L)
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, R2=R2, S1=S1, S2=S2,
                width_pct=(upper_cpr - lower_cpr) / P * 100)


# ─── VIRGIN CPR ──────────────────────────────────────────────────────────────
def is_virgin(cpr: dict, rows: list) -> bool:
    """True if CPR zone was never touched in the last 5 sessions."""
    upper, lower = cpr["upper_cpr"], cpr["lower_cpr"]
    for bar in rows[-6:-1]:          # last 5 bars before yesterday
        if float(bar["low"]) <= upper and float(bar["high"]) >= lower:
            return False
    return True


# ─── CONVICTION ──────────────────────────────────────────────────────────────
def conviction(width: float, virgin: bool) -> tuple:
    """Returns (label, colour_emoji, trade_flag)."""
    if   width < NARROW_HIGH:     base = ("HIGH",     "🔴", True)
    elif width < NARROW_MEDIUM:   base = ("MEDIUM",   "🟡", True)
    elif width < NARROW_STANDARD: base = ("STANDARD", "🟢", True)
    else:                         return ("SKIP",      "⚪", False)
    label = base[0] + (" + Virgin ⭐" if virgin else "")
    return label, base[1], base[2]


# ─── TELEGRAM ────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    if not TG_TOKEN or not TG_CHAT_ID:
        print("\n" + "━" * 56 + "\n" + text + "\n" + "━" * 56)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT_ID, "text": text,
              "parse_mode": "HTML"},
        timeout=10,
    )
    if r.ok:
        log.info("Telegram sent ✅")
    else:
        log.error("Telegram failed %s: %s", r.status_code, r.text[:120])


# ─── FORMAT HELPERS ──────────────────────────────────────────────────────────
def _p(v):  return f"₹{v:,.2f}"   # price
def _r(v):  return f"₹{v:,.0f}"   # rupees


# ─── BUILD MESSAGE ───────────────────────────────────────────────────────────
def build_skip_message(symbol, expiry, prev_date, cpr, width, today_str, dow):
    return (
        f"📊 <b>GOLDPETAL CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}\n\n"
        f"⚪  <b>NO TRADE — Wide / Neutral Day</b>\n"
        f"CPR width: {width:.3f}%  (need &lt; 0.30% for breakout)\n\n"
        f"<b>Levels for reference</b>\n"
        f"  Pivot      {_p(cpr['P'])}\n"
        f"  Upper CPR  {_p(cpr['upper_cpr'])}\n"
        f"  Lower CPR  {_p(cpr['lower_cpr'])}\n"
        f"  R1 / S1    {_p(cpr['R1'])}  /  {_p(cpr['S1'])}\n\n"
        f"<i>Sit on hands. Next signal tomorrow.</i>"
    )


def build_trade_message(symbol, expiry, prev_date, cpr,
                        conv_label, conv_emoji, virgin,
                        today_str, dow):

    width     = cpr["width_pct"]
    upper     = cpr["upper_cpr"]
    lower     = cpr["lower_cpr"]

    # Entry / SL / Target with slippage baked in
    l_entry   = upper + SLIPPAGE
    l_sl      = lower - SLIPPAGE
    l_target  = cpr["R1"]
    s_entry   = lower - SLIPPAGE
    s_sl      = upper + SLIPPAGE
    s_target  = cpr["S1"]

    # Risks and rewards per lot × LOTS
    l_risk    = (l_entry - l_sl)    * PNL_PER_LOT * LOTS
    l_reward  = (l_target - l_entry)* PNL_PER_LOT * LOTS
    s_risk    = (s_sl - s_entry)    * PNL_PER_LOT * LOTS
    s_reward  = (s_entry - s_target)* PNL_PER_LOT * LOTS

    l_rr = f"1 : {l_reward/l_risk:.1f}" if l_risk > 0 and l_reward > 0 else "–"
    s_rr = f"1 : {s_reward/s_risk:.1f}" if s_risk > 0 and s_reward > 0 else "–"

    # Budget
    gold_price  = cpr["P"]
    lot_val     = gold_price * 1        # 1 gram per lot
    margin_lot  = lot_val * MARGIN_PCT
    margin_10   = margin_lot * LOTS
    headroom    = CAPITAL - margin_10
    max_loss    = l_risk + COMMISSION   # worst case after commission
    loss_pct    = max_loss / CAPITAL * 100

    # Virgin line
    virgin_line = "⭐ <b>Virgin CPR</b> — zone untouched in last 5 sessions\n" \
                  if virgin else ""

    return (
        f"🥇 <b>GOLDPETAL CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}  |  ref: {prev_date}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{conv_emoji}  <b>Conviction: {conv_label}</b>\n"
        f"CPR width: {width:.3f}%  →  Narrow / Breakout day\n"
        f"{virgin_line}\n"

        f"<b>📐 CPR Levels</b>\n"
        f"  Pivot      <b>{_p(cpr['P'])}</b>\n"
        f"  Upper CPR  <b>{_p(upper)}</b>\n"
        f"  Lower CPR  <b>{_p(lower)}</b>\n"
        f"  R1  {_p(cpr['R1'])}   R2  {_p(cpr['R2'])}\n"
        f"  S1  {_p(cpr['S1'])}   S2  {_p(cpr['S2'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"⏰  <b>Watch 09:00–10:00 candle close, then:</b>\n\n"

        f"📈  <b>LONG</b>  if close &gt; {_p(upper)}\n"
        f"  Buy    <b>{_p(l_entry)}</b>  (limit · NRML · 10 lots)\n"
        f"  SL     <b>{_p(l_sl)}</b>\n"
        f"  Target <b>{_p(l_target)}</b>  (R1)\n"
        f"  Risk {_r(l_risk)}  →  Reward {_r(l_reward)}  →  RR {l_rr}\n\n"

        f"📉  <b>SHORT</b>  if close &lt; {_p(lower)}\n"
        f"  Sell   <b>{_p(s_entry)}</b>  (limit · NRML · 10 lots)\n"
        f"  SL     <b>{_p(s_sl)}</b>\n"
        f"  Target <b>{_p(s_target)}</b>  (S1)\n"
        f"  Risk {_r(s_risk)}  →  Reward {_r(s_reward)}  →  RR {s_rr}\n\n"

        f"🚫  <b>SKIP</b> if close inside  {_p(lower)} – {_p(upper)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        f"💰  <b>10 lots · NRML · ₹50K account</b>\n"
        f"  Margin   ~{_r(margin_10)}   Free  ~{_r(headroom)}\n"
        f"  Max loss  {_r(max_loss)}  ({loss_pct:.1f}% of capital)\n"
        f"  P&L/₹1 move  ₹{PNL_PER_LOT * LOTS}  "
        f"  P&L/₹100 move  {_r(PNL_PER_LOT * LOTS * 100)}"
    )


# ─── MAIN ────────────────────────────────────────────────────────────────────
def send_goldpetal_cpr_alert() -> None:
    now       = datetime.now()
    today_str = now.strftime("%a %d %b %Y")
    dow       = now.strftime("%A")

    log.info("=" * 56)
    log.info("GOLDPETAL CPR Alert  —  %s  08:55 IST", today_str)
    log.info("=" * 56)

    kite              = get_kite()
    token, sym, expiry = get_front_month(kite)
    rows              = get_recent_daily(kite, token, n=10)

    if len(rows) < 2:
        log.error("Insufficient data (holiday?). Aborting.")
        return

    yesterday  = rows[-1]
    prev_H     = float(yesterday["high"])
    prev_L     = float(yesterday["low"])
    prev_C     = float(yesterday["close"])
    prev_date  = yesterday["date"].strftime("%a %d %b") \
                 if hasattr(yesterday["date"], "strftime") \
                 else str(yesterday["date"])

    log.info("Yesterday: H=%s  L=%s  C=%s",
             _p(prev_H), _p(prev_L), _p(prev_C))

    cpr    = calc_cpr(prev_H, prev_L, prev_C)
    width  = cpr["width_pct"]
    virgin = is_virgin(cpr, rows)

    conv_label, conv_emoji, trade_flag = conviction(width, virgin)

    log.info("CPR: upper=%s  lower=%s  width=%.3f%%  conv=%s  virgin=%s",
             _p(cpr["upper_cpr"]), _p(cpr["lower_cpr"]),
             width, conv_label, virgin)

    if trade_flag:
        msg = build_trade_message(sym, expiry, prev_date, cpr,
                                  conv_label, conv_emoji, virgin,
                                  today_str, dow)
    else:
        msg = build_skip_message(sym, expiry, prev_date, cpr,
                                 width, today_str, dow)

    send_telegram(msg)


# ─── ENTRY ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    send_goldpetal_cpr_alert()
