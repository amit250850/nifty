#!/usr/bin/env python3
"""
crudeoilm_morning_alert.py
==========================
Pre-market MCX CRUDEOILM (Mini Crude Oil) CPR alert.
Runs at 08:55 IST → Telegram arrives before MCX opens at 09:00.

Strategy  : AAK CPR Breakout — 1 lot CRUDEOILM, NRML product
Conviction: based on CPR width (< 0.50% = narrow/breakout day)

CRUDEOILM specs
---------------
  Lot size      : 10 barrels
  Quote         : ₹ per barrel
  P&L per lot   : ₹10 per ₹1/barrel move  (10 barrels × ₹1)
  Commission    : ₹65 flat per round-trip
  NRML margin   : ~₹28,600/lot  (Zerodha NRML, verified Apr 2026)
  Slippage      : ₹1.0/barrel per side  (1 tick)
  Expiry cycle  : MONTHLY — roll 5 days before expiry

  1 lot on ₹50K account:
    Margin ≈ ₹28,600  →  Free ≈ ₹21,400
    P&L: ₹10 per ₹1/barrel move · ₹100 move → ₹1,000

Backtest (Oct 2025 – Apr 2026, 1 lot):
  55% win rate · 4.33x payoff · +₹12,250 in 6 months
  SHORTs especially strong: 69% win rate

Run schedule
------------
  Time  : 08:55 IST  Mon–Fri
  Via main.py APScheduler. Or standalone: python crudeoilm_morning_alert.py
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
    print("ERROR: pip install kiteconnect python-dotenv requests pandas")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

LOTS            = 1
PNL_PER_LOT     = 10        # 10 barrels × ₹1/barrel = ₹10/lot/₹1 move
COMMISSION      = 65
SLIPPAGE        = 1.0       # ₹/barrel per side (1 tick)
MARGIN_PER_LOT  = 28_600    # actual Zerodha NRML (verified Apr 2026)
CAPITAL         = 50_000

# CPR width thresholds (% of pivot price)
# Crude is more volatile than gold — thresholds are wider
NARROW_HIGH     = 0.20   # HIGH conviction
NARROW_MEDIUM   = 0.35   # MEDIUM conviction
NARROW_STANDARD = 0.50   # STANDARD (matches backtest narrow_pct)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── KITE ─────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        sys.exit("ERROR: Missing KITE_API_KEY / KITE_ACCESS_TOKEN. Run login.py first.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


# ─── INSTRUMENT ───────────────────────────────────────────────────────────────
def get_front_month(kite: KiteConnect) -> tuple:
    """Return (token, tradingsymbol, expiry) for active front-month CRUDEOILM."""
    rows = kite.instruments("MCX")
    df   = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])

    contracts = df[
        (df["name"] == "CRUDEOILM") &
        (df["instrument_type"] == "FUT") &
        (df["expiry"].dt.date >= date.today())
    ].sort_values("expiry").reset_index(drop=True)

    if contracts.empty:
        raise RuntimeError("No active CRUDEOILM contracts found.")

    front = contracts.iloc[0]
    dte   = (front["expiry"].date() - date.today()).days
    if dte <= 5 and len(contracts) > 1:   # roll 5 days before expiry
        front = contracts.iloc[1]
        dte   = (front["expiry"].date() - date.today()).days

    log.info("Contract: %s  expiry=%s  DTE=%d",
             front["tradingsymbol"], front["expiry"].date(), dte)
    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── OHLC ─────────────────────────────────────────────────────────────────────
def get_recent_daily(kite: KiteConnect, token: int, n: int = 10) -> list:
    to_dt   = date.today() - timedelta(days=1)
    from_dt = to_dt - timedelta(days=n + 5)
    return kite.historical_data(token, from_dt, to_dt,
                                interval="day", continuous=False, oi=False)


# ─── CPR MATH ─────────────────────────────────────────────────────────────────
def calc_cpr(H: float, L: float, C: float) -> dict:
    P         = (H + L + C) / 3.0
    BC        = (H + L) / 2.0
    TC        = 2 * P - BC
    upper_cpr = max(TC, BC)
    lower_cpr = min(TC, BC)
    R1 = 2 * P - L;  R2 = P + (H - L)
    S1 = 2 * P - H;  S2 = P - (H - L)
    return dict(P=P, upper_cpr=upper_cpr, lower_cpr=lower_cpr,
                R1=R1, R2=R2, S1=S1, S2=S2,
                width_pct=(upper_cpr - lower_cpr) / P * 100)


# ─── VIRGIN CPR ───────────────────────────────────────────────────────────────
def is_virgin(cpr: dict, rows: list) -> bool:
    upper, lower = cpr["upper_cpr"], cpr["lower_cpr"]
    for bar in rows[-6:-1]:
        if float(bar["low"]) <= upper and float(bar["high"]) >= lower:
            return False
    return True


# ─── CONVICTION ───────────────────────────────────────────────────────────────
def conviction(width: float, virgin: bool) -> tuple:
    if   width < NARROW_HIGH:     base = ("HIGH",     "🔴", True)
    elif width < NARROW_MEDIUM:   base = ("MEDIUM",   "🟡", True)
    elif width < NARROW_STANDARD: base = ("STANDARD", "🟢", True)
    else:                         return ("SKIP",      "⚪", False)
    label = base[0] + (" + Virgin ⭐" if virgin else "")
    return label, base[1], base[2]


# ─── TELEGRAM ─────────────────────────────────────────────────────────────────
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
        log.info("Telegram sent ✓")
    else:
        log.error("Telegram failed %s: %s", r.status_code, r.text[:120])


def _p(v): return f"₹{v:,.2f}"
def _r(v): return f"₹{v:,.0f}"


# ─── BUILD MESSAGES ───────────────────────────────────────────────────────────
def build_skip_message(symbol, expiry, prev_date, cpr, width, today_str, dow):
    return (
        f"🛢 <b>CRUDEOILM CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}\n\n"
        f"⚪  <b>NO TRADE — Wide / Neutral Day</b>\n"
        f"CPR width: {width:.3f}%  (need &lt; {NARROW_STANDARD}% for breakout)\n\n"
        f"<b>Levels for reference</b>\n"
        f"  Pivot      {_p(cpr['P'])}/bbl\n"
        f"  Upper CPR  {_p(cpr['upper_cpr'])}/bbl\n"
        f"  Lower CPR  {_p(cpr['lower_cpr'])}/bbl\n"
        f"  R1 / S1    {_p(cpr['R1'])}  /  {_p(cpr['S1'])}\n\n"
        f"<i>Sit on hands. Next signal tomorrow.</i>"
    )


def build_trade_message(symbol, expiry, prev_date, cpr,
                        conv_label, conv_emoji, virgin,
                        today_str, dow):
    width  = cpr["width_pct"]
    upper  = cpr["upper_cpr"]
    lower  = cpr["lower_cpr"]

    l_entry  = upper + SLIPPAGE
    l_sl     = lower - SLIPPAGE
    l_target = cpr["R1"]
    s_entry  = lower - SLIPPAGE
    s_sl     = upper + SLIPPAGE
    s_target = cpr["S1"]

    l_risk   = (l_entry - l_sl)     * PNL_PER_LOT * LOTS
    l_reward = (l_target - l_entry) * PNL_PER_LOT * LOTS
    s_risk   = (s_sl - s_entry)     * PNL_PER_LOT * LOTS
    s_reward = (s_entry - s_target) * PNL_PER_LOT * LOTS

    l_rr = f"1 : {l_reward/l_risk:.1f}" if l_risk > 0 and l_reward > 0 else "–"
    s_rr = f"1 : {s_reward/s_risk:.1f}" if s_risk > 0 and s_reward > 0 else "–"

    margin_tot = MARGIN_PER_LOT * LOTS
    headroom   = CAPITAL - margin_tot
    max_loss   = max(l_risk, s_risk) + COMMISSION
    loss_pct   = max_loss / CAPITAL * 100

    virgin_line = "⭐ <b>Virgin CPR</b> — zone untouched in last 5 sessions\n" \
                  if virgin else ""

    return (
        f"🛢 <b>CRUDEOILM CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}  |  ref: {prev_date}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{conv_emoji}  <b>Conviction: {conv_label}</b>\n"
        f"CPR width: {width:.3f}%  →  Narrow / Breakout day\n"
        f"{virgin_line}\n"

        f"<b>📐 CPR Levels  (₹/barrel)</b>\n"
        f"  Pivot      <b>{_p(cpr['P'])}</b>\n"
        f"  Upper CPR  <b>{_p(upper)}</b>\n"
        f"  Lower CPR  <b>{_p(lower)}</b>\n"
        f"  R1  {_p(cpr['R1'])}   R2  {_p(cpr['R2'])}\n"
        f"  S1  {_p(cpr['S1'])}   S2  {_p(cpr['S2'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"⏰  <b>Watch 09:00–10:00 candle close, then:</b>\n\n"

        f"📈  <b>LONG</b>  if close &gt; {_p(upper)}\n"
        f"  Buy    <b>{_p(l_entry)}</b>  (limit · NRML · {LOTS} lot)\n"
        f"  SL     <b>{_p(l_sl)}</b>\n"
        f"  Target <b>{_p(l_target)}</b>  (R1)\n"
        f"  Risk {_r(l_risk)}  →  Reward {_r(l_reward)}  →  RR {l_rr}\n\n"

        f"📉  <b>SHORT</b>  if close &lt; {_p(lower)}\n"
        f"  Sell   <b>{_p(s_entry)}</b>  (limit · NRML · {LOTS} lot)\n"
        f"  SL     <b>{_p(s_sl)}</b>\n"
        f"  Target <b>{_p(s_target)}</b>  (S1)\n"
        f"  Risk {_r(s_risk)}  →  Reward {_r(s_reward)}  →  RR {s_rr}\n\n"

        f"🚫  <b>SKIP</b> if close inside  {_p(lower)} – {_p(upper)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        f"💰  <b>{LOTS} lot · NRML · ₹50K account</b>\n"
        f"  10 bbl/lot  ·  P&L ₹{PNL_PER_LOT}/lot per ₹1/bbl move\n"
        f"  Margin   ~{_r(margin_tot)}   Free  ~{_r(headroom)}\n"
        f"  Max loss  ~{_r(max_loss)}  ({loss_pct:.1f}% of capital)\n"
        f"  P&L/₹10 move  {_r(PNL_PER_LOT * LOTS * 10)}   "
        f"P&L/₹100 move  {_r(PNL_PER_LOT * LOTS * 100)}\n\n"
        f"<i>Backtest: 55% win · 4.33x payoff · SHORTs 69% win. "
        f"EOD exit 17:00 IST.</i>"
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def send_crudeoilm_cpr_alert() -> None:
    now       = datetime.now()
    today_str = now.strftime("%a %d %b %Y")
    dow       = now.strftime("%A")

    log.info("=" * 56)
    log.info("CRUDEOILM CPR Alert  —  %s  08:55 IST", today_str)
    log.info("=" * 56)

    kite               = get_kite()
    token, sym, expiry = get_front_month(kite)
    rows               = get_recent_daily(kite, token, n=10)

    if len(rows) < 2:
        log.error("Insufficient data (holiday?). Aborting.")
        return

    yesterday = rows[-1]
    prev_H    = float(yesterday["high"])
    prev_L    = float(yesterday["low"])
    prev_C    = float(yesterday["close"])
    prev_date = yesterday["date"].strftime("%a %d %b") \
                if hasattr(yesterday["date"], "strftime") \
                else str(yesterday["date"])

    log.info("Yesterday: H=%s  L=%s  C=%s", _p(prev_H), _p(prev_L), _p(prev_C))

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


if __name__ == "__main__":
    send_crudeoilm_cpr_alert()
