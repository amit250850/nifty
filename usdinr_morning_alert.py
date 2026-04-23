#!/usr/bin/env python3
"""
usdinr_morning_alert.py
=======================
Pre-market NSE CDS USDINR CPR alert.
Runs at 08:55 IST → Telegram arrives before CDS opens at 09:00.

Strategy  : CPR Breakout — 10 lots USDINR, NRML product
Conviction: CPR width < 0.020% = NARROW (trade), else skip

USDINR specs
-------------
  Lot size      : 1,000 USD
  Quote         : ₹ per USD (INR/USD rate)
  P&L per lot   : ₹1,000 per ₹1/USD move  (1,000 USD × ₹1)
  10 lots P&L   : ₹10,000 per ₹1/USD move
  Commission    : ₹30 flat per round-trip
  Margin/lot    : ~₹2,200  →  ₹22,000 for 10 lots
  Slippage      : ₹0.0025/USD per side  (1 tick = 0.25 paise)
  Expiry cycle  : Monthly (last business day of month). Roll 3 days before.
  Session       : NSE CDS  09:00–17:00 IST

Run schedule
------------
  Time    : 08:55 IST  Mon–Fri
  Via main.py APScheduler (wired in at 08:55).
  Or standalone: python usdinr_morning_alert.py

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
    import yfinance as yf
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests pandas yfinance "
          "--break-system-packages")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

LOTS            = 10
LOT_SIZE_USD    = 1000          # USD per lot
PNL_PER_LOT     = LOT_SIZE_USD  # ₹1,000 per lot per ₹1/USD move
COMMISSION      = 30            # ₹ flat per round-trip
SLIPPAGE        = 0.0025        # ₹/USD per side (1 tick)
MARGIN_PER_LOT  = 2200          # approximate NRML margin in ₹

NARROW_THRESHOLD = 0.020        # % — must match usdinr_cpr_monitor.py

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
        raise RuntimeError(
            "No active USDINR contracts found in kite.instruments('CDS').\n"
            "Check that CDS is accessible and your login is fresh."
        )

    front = contracts.iloc[0]
    dte   = (front["expiry"].date() - date.today()).days
    if dte <= 3 and len(contracts) > 1:   # roll 3 days before expiry
        front = contracts.iloc[1]
        dte   = (front["expiry"].date() - date.today()).days

    log.info("Contract: %s  expiry=%s  DTE=%d",
             front["tradingsymbol"], front["expiry"].date(), dte)
    return int(front["instrument_token"]), front["tradingsymbol"], front["expiry"].date()


# ─── OHLC via yfinance ────────────────────────────────────────────────────────
YF_TICKER = "USDINR=X"

def get_recent_daily(n: int = 10) -> pd.DataFrame:
    """Fetch recent daily OHLC for USDINR from yfinance (spot rate, reliable)."""
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=n + 10)
    df = yf.download(YF_TICKER, start=str(start_dt), end=str(end_dt),
                     interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {YF_TICKER}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Drop zero-range bars (holidays or bad data where H=L=C)
    df = df[df["high"] != df["low"]]
    return df


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
def is_virgin(cpr: dict, df: pd.DataFrame) -> bool:
    """True if CPR zone was not touched in the last 5 sessions (before yesterday)."""
    upper, lower = cpr["upper_cpr"], cpr["lower_cpr"]
    # df is sorted ascending; check the 5 bars before the last one (yesterday)
    check = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]
    for _, row in check.iterrows():
        if row["low"] <= upper and row["high"] >= lower:
            return False
    return True


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


def _p(v): return f"₹{v:.4f}"
def _r(v): return f"₹{v:,.0f}"


# ─── BUILD MESSAGES ───────────────────────────────────────────────────────────
def build_skip_message(symbol, expiry, prev_date, cpr, width, today_str, dow):
    return (
        f"💵 <b>USDINR CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}\n\n"
        f"⚪  <b>NO TRADE — Wide / Neutral Day</b>\n"
        f"CPR width: {width:.4f}%  (need &lt; {NARROW_THRESHOLD:.3f}% for breakout)\n\n"
        f"<b>Levels for reference</b>\n"
        f"  Pivot      {_p(cpr['P'])}\n"
        f"  Upper CPR  {_p(cpr['upper_cpr'])}\n"
        f"  Lower CPR  {_p(cpr['lower_cpr'])}\n"
        f"  R1 / S1    {_p(cpr['R1'])}  /  {_p(cpr['S1'])}\n\n"
        f"<i>Wide CPR = RBI intervention likely. Sit on hands. Next signal tomorrow.</i>"
    )


def build_trade_message(symbol, expiry, prev_date, cpr,
                        virgin, today_str, dow):
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
    max_loss   = l_risk + COMMISSION
    conv_emoji = "🔴" if width < 0.010 else "🟡" if width < 0.015 else "🟢"
    conv_label = "HIGH" if width < 0.010 else "MEDIUM" if width < 0.015 else "STANDARD"
    virgin_line = "⭐ <b>Virgin CPR</b> — zone untouched in last 5 sessions\n" \
                  if virgin else ""

    return (
        f"💵 <b>USDINR CPR — {today_str} ({dow})</b>\n"
        f"<code>{symbol}</code>  exp {expiry}  |  ref: {prev_date}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        f"{conv_emoji}  <b>Conviction: {conv_label}</b>  —  Narrow CPR / Breakout day\n"
        f"CPR width: {width:.4f}%\n"
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
        f"  Buy    <b>{_p(l_entry)}</b>  (limit · NRML · {LOTS} lots)\n"
        f"  SL     <b>{_p(l_sl)}</b>\n"
        f"  Target <b>{_p(l_target)}</b>  (R1)\n"
        f"  Risk {_r(l_risk)}  →  Reward {_r(l_reward)}  →  RR {l_rr}\n\n"

        f"📉  <b>SHORT</b>  if close &lt; {_p(lower)}\n"
        f"  Sell   <b>{_p(s_entry)}</b>  (limit · NRML · {LOTS} lots)\n"
        f"  SL     <b>{_p(s_sl)}</b>\n"
        f"  Target <b>{_p(s_target)}</b>  (S1)\n"
        f"  Risk {_r(s_risk)}  →  Reward {_r(s_reward)}  →  RR {s_rr}\n\n"

        f"🚫  <b>SKIP</b> if close inside  {_p(lower)} – {_p(upper)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        f"💰  <b>{LOTS} lots · NRML · NSE CDS</b>\n"
        f"  Lot size : 1,000 USD  ·  P&L: {_r(PNL_PER_LOT)}/lot per ₹1/USD move\n"
        f"  Margin   ~{_r(margin_tot)}   ({LOTS} lots × ~{_r(MARGIN_PER_LOT)}/lot)\n"
        f"  Max loss  {_r(max_loss)}\n"
        f"  EOD exit  16:45 IST  (CDS closes 17:00)\n\n"
        f"<i>Narrow CPR = RBI not intervening today → trending move likely.</i>"
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def send_usdinr_cpr_alert() -> None:
    now       = datetime.now()
    today_str = now.strftime("%a %d %b %Y")
    dow       = now.strftime("%A")

    log.info("=" * 56)
    log.info("USDINR CPR Alert  —  %s  08:55 IST", today_str)
    log.info("=" * 56)

    kite           = get_kite()
    _, sym, expiry = get_front_month(kite)

    df = get_recent_daily(n=10)
    if len(df) < 2:
        log.error("Insufficient yfinance data (holiday?). Aborting.")
        return

    yesterday = df.iloc[-1]
    prev_H    = float(yesterday["high"])
    prev_L    = float(yesterday["low"])
    prev_C    = float(yesterday["close"])
    prev_date = yesterday.name.strftime("%a %d %b") \
                if hasattr(yesterday.name, "strftime") else str(yesterday.name)

    log.info("Yesterday: H=%.4f  L=%.4f  C=%.4f  (yfinance USDINR=X)", prev_H, prev_L, prev_C)

    cpr    = calc_cpr(prev_H, prev_L, prev_C)
    width  = cpr["width_pct"]
    virgin = is_virgin(cpr, df)

    log.info("CPR: upper=%.4f  lower=%.4f  width=%.4f%%  virgin=%s",
             cpr["upper_cpr"], cpr["lower_cpr"], width, virgin)

    if width >= NARROW_THRESHOLD:
        msg = build_skip_message(sym, expiry, prev_date, cpr, width, today_str, dow)
    else:
        msg = build_trade_message(sym, expiry, prev_date, cpr, virgin, today_str, dow)

    send_telegram(msg)


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    send_usdinr_cpr_alert()
