#!/usr/bin/env python3
"""
goldpetal_monitor.py
====================
Active trade monitor for MCX GOLDPETAL CPR positions.
Runs every 5 minutes from 10:05 AM – 4:55 PM IST via APScheduler in main.py.

WHAT IT DOES EACH CYCLE
────────────────────────
  1. Loads data/goldpetal_trade.json — if no open trade, returns immediately.
  2. Fetches current LTP of GOLDPETAL via kite.ltp().
  3. Checks SL and Target hit:
       SL hit    → square off with market order + Telegram alert + mark closed
       Target hit → square off with market order + Telegram alert + mark closed
  4. Approach warnings (sent once per session):
       "near_sl"     — LTP within 30% of SL distance from entry
       "near_target" — LTP 70% of the way to target from entry
  5. Auto square-off at 16:55 IST if trade is still open (MCX gold session ends
     at 17:00 IST for agricultural segment; GOLDPETAL uses this session).
  6. Writes updated state back to goldpetal_trade.json after each action.

STATE FILE: data/goldpetal_trade.json
  Created by goldpetal_autoexec.py at 10:02 IST.
  Persists across bot restarts — if main.py crashes and restarts mid-session,
  the monitor picks up the open trade immediately.

SQUARE-OFF
──────────
  Opposite of the entry order:
    LONG  trade → SELL  market order
    SHORT trade → BUY   market order
  Product: NRML. Exchange: MCX. Quantity: same as entry lots.

  When GOLDPETAL_AUTO_EXECUTE = False in goldpetal_autoexec.py, the entry
  order was never placed. The monitor still logs + Telegrams what it WOULD do,
  but skips the actual square-off order (no open position exists in Kite).
"""

import os
import sys
import json
import logging
from datetime import date, datetime, time as dtime, timedelta

try:
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect
    import requests
except ImportError:
    print("ERROR: pip install kiteconnect python-dotenv requests --break-system-packages")
    sys.exit(1)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY", "")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")
TG_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Warning thresholds
NEAR_SL_FACTOR     = 0.30   # alert when LTP is within 30% of SL move from entry
NEAR_TARGET_FACTOR = 0.70   # alert when LTP is 70% of the way to target

# Auto-square-off time (IST) — 5 min before GOLDPETAL session end
AUTO_SQ_OFF_HOUR   = 16
AUTO_SQ_OFF_MINUTE = 55

STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "goldpetal_trade.json",
)

# ─── LOGGING ───────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─── HELPERS ───────────────────────────────────────────────────────────────────
def _p(v): return f"₹{v:,.2f}"
def _r(v): return f"₹{v:,.0f}"


# ─── STATE ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[monitor] Could not load state: %s", exc)
        return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def alert_already_sent(state: dict, alert_code: str) -> bool:
    return alert_code in state.get("alerts_sent", [])


def mark_alert_sent(state: dict, alert_code: str) -> None:
    if "alerts_sent" not in state:
        state["alerts_sent"] = []
    if alert_code not in state["alerts_sent"]:
        state["alerts_sent"].append(alert_code)


# ─── KITE ──────────────────────────────────────────────────────────────────────
def get_kite() -> KiteConnect:
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing KITE_API_KEY / KITE_ACCESS_TOKEN. Run login.py.")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


def get_ltp(kite: KiteConnect, token: int, symbol: str) -> float:
    """Fetch last traded price for GOLDPETAL token."""
    instrument_key = f"MCX:{symbol}"
    data = kite.ltp([instrument_key])
    return float(data[instrument_key]["last_price"])


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
        log.info("[monitor] Telegram ✅")
    else:
        log.error("[monitor] Telegram failed %s: %s", r.status_code, r.text[:80])


# ─── SQUARE-OFF ────────────────────────────────────────────────────────────────
def square_off(kite: KiteConnect, state: dict, reason: str) -> str | None:
    """
    Place a market order to close the open GOLDPETAL position.
    LONG → SELL  |  SHORT → BUY
    Returns order_id or 'DRY-RUN' / None.
    """
    direction = state["direction"]
    symbol    = state["symbol"]
    lots      = state["lots"]

    # Import flag from autoexec at runtime to stay in sync
    try:
        from goldpetal_autoexec import GOLDPETAL_AUTO_EXECUTE
    except ImportError:
        GOLDPETAL_AUTO_EXECUTE = False

    tx = kite.TRANSACTION_TYPE_SELL if direction == "LONG" else kite.TRANSACTION_TYPE_BUY

    if not GOLDPETAL_AUTO_EXECUTE:
        log.info("[monitor][DRY-RUN] Would %s %d lots %s — reason: %s",
                 "SELL" if direction == "LONG" else "BUY", lots, symbol, reason)
        return "DRY-RUN"

    try:
        order_id = kite.place_order(
            tradingsymbol    = symbol,
            exchange         = "MCX",
            transaction_type = tx,
            quantity         = lots,
            product          = kite.PRODUCT_NRML,
            order_type       = kite.ORDER_TYPE_MARKET,
            variety          = kite.VARIETY_REGULAR,
        )
        log.info("[monitor] ✅ Square-off placed: %s qty=%d → order_id=%s  reason=%s",
                 symbol, lots, order_id, reason)
        return str(order_id)
    except Exception as exc:
        log.error("[monitor] Square-off FAILED: %s", exc)
        return None


# ─── P&L CALC ──────────────────────────────────────────────────────────────────
def calc_pnl(state: dict, exit_price: float) -> float:
    """Net P&L for the closed trade."""
    direction  = state["direction"]
    entry      = state["entry"]
    lots       = state["lots"]
    commission = 55  # flat both legs combined (Zerodha per-order model)

    move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
    return move * lots - commission   # PNL_PER_LOT = 1 for GOLDPETAL


# ─── MAIN MONITOR CYCLE ────────────────────────────────────────────────────────
def run_goldpetal_monitor() -> None:
    state = load_state()

    # No state file or not today's trade
    if not state:
        log.debug("[monitor] No trade state file found.")
        return

    if state.get("date") != str(date.today()):
        log.debug("[monitor] State is from %s, not today — skipping.", state.get("date"))
        return

    if state.get("status") != "open":
        log.debug("[monitor] Trade status=%s — nothing to monitor.", state.get("status"))
        return

    # ── Setup ──────────────────────────────────────────────────────────────────
    symbol    = state["symbol"]
    token     = state["token"]
    direction = state["direction"]
    entry     = float(state["entry"])
    sl        = float(state["sl"])
    target    = float(state["target"])
    lots      = state["lots"]
    now       = datetime.now()
    today_str = now.strftime("%a %d %b %Y")
    arrow     = "📈" if direction == "LONG" else "📉"

    # ── Fetch LTP ──────────────────────────────────────────────────────────────
    try:
        kite = get_kite()
        ltp  = get_ltp(kite, token, symbol)
    except Exception as exc:
        log.error("[monitor] Could not fetch LTP: %s", exc)
        return

    log.info("[monitor] %s LTP=%s  entry=%s  sl=%s  target=%s",
             symbol, _p(ltp), _p(entry), _p(sl), _p(target))

    # ── Compute progress toward target / SL ────────────────────────────────────
    total_move  = abs(target - entry)
    total_risk  = abs(entry - sl)
    current_pnl = calc_pnl(state, ltp)

    if direction == "LONG":
        pct_to_target = (ltp - entry) / total_move  if total_move > 0 else 0
        pct_to_sl     = (entry - ltp) / total_risk  if total_risk > 0 else 0
    else:
        pct_to_target = (entry - ltp) / total_move  if total_move > 0 else 0
        pct_to_sl     = (ltp - entry) / total_risk  if total_risk > 0 else 0

    # ── Check: auto square-off time (16:55 IST) ────────────────────────────────
    sq_off_time = now.replace(hour=AUTO_SQ_OFF_HOUR, minute=AUTO_SQ_OFF_MINUTE,
                              second=0, microsecond=0)
    if now >= sq_off_time:
        log.info("[monitor] 16:55 reached — auto square-off trigger.")
        sq_order = square_off(kite, state, "EOD_SQUAREOFF")
        net_pnl  = calc_pnl(state, ltp)

        state["status"]       = "closed"
        state["exit_reason"]  = "EOD_SQUAREOFF"
        state["exit_price"]   = ltp
        state["exit_order_id"]= sq_order
        state["closed_at"]    = now.isoformat()
        state["net_pnl"]      = net_pnl
        save_state(state)

        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
        send_telegram(
            f"⏰ <b>GOLDPETAL EOD Square-Off — {today_str}</b>\n"
            f"<code>{symbol}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{arrow} {direction}  ·  {lots} lots  ·  16:55 close\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Exit   : {_p(ltp)}  (market fill ~this)\n"
            f"  SL     : {_p(sl)}   Target: {_p(target)}\n\n"
            f"{pnl_emoji}  <b>Net P&L: {_r(net_pnl)}</b>  (incl. ₹55 commission)\n\n"
            f"<i>Position squared off at 16:55 — MCX GOLDPETAL session ends 17:00.</i>"
        )
        return

    # ── Check: TARGET HIT ──────────────────────────────────────────────────────
    target_hit = (ltp >= target) if direction == "LONG" else (ltp <= target)
    if target_hit:
        log.info("[monitor] 🎯 TARGET HIT! LTP=%s ≥ target=%s", _p(ltp), _p(target))
        sq_order = square_off(kite, state, "TARGET")
        net_pnl  = calc_pnl(state, target)   # use target price for clean P&L

        state["status"]        = "closed"
        state["exit_reason"]   = "TARGET"
        state["exit_price"]    = ltp
        state["exit_order_id"] = sq_order
        state["closed_at"]     = now.isoformat()
        state["net_pnl"]       = net_pnl
        save_state(state)

        send_telegram(
            f"🎯 <b>GOLDPETAL TARGET HIT — {today_str}</b>\n"
            f"<code>{symbol}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{arrow} {direction}  ·  {lots} lots\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Target : {_p(target)}\n"
            f"  Exit   : {_p(ltp)}  ✅\n\n"
            f"🟢  <b>Net P&L: +{_r(net_pnl)}</b>  (incl. ₹55 commission)\n\n"
            f"<i>Square-off order placed. Verify fill in Kite app.</i>"
        )
        return

    # ── Check: STOP-LOSS HIT ───────────────────────────────────────────────────
    sl_hit = (ltp <= sl) if direction == "LONG" else (ltp >= sl)
    if sl_hit:
        log.info("[monitor] 🛑 SL HIT! LTP=%s ≤ sl=%s", _p(ltp), _p(sl))
        sq_order = square_off(kite, state, "SL")
        net_pnl  = calc_pnl(state, sl)   # use SL price for clean P&L

        state["status"]        = "closed"
        state["exit_reason"]   = "SL"
        state["exit_price"]    = ltp
        state["exit_order_id"] = sq_order
        state["closed_at"]     = now.isoformat()
        state["net_pnl"]       = net_pnl
        save_state(state)

        send_telegram(
            f"🛑 <b>GOLDPETAL STOP-LOSS HIT — {today_str}</b>\n"
            f"<code>{symbol}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{arrow} {direction}  ·  {lots} lots\n"
            f"  Entry  : {_p(entry)}\n"
            f"  SL     : {_p(sl)}\n"
            f"  Exit   : {_p(ltp)}  ❌\n\n"
            f"🔴  <b>Net P&L: {_r(net_pnl)}</b>  (incl. ₹55 commission)\n\n"
            f"<i>Square-off order placed. Worst-case daily loss contained.</i>"
        )
        return

    # ── Warning: Approaching SL (within 30% of SL distance) ───────────────────
    if pct_to_sl >= NEAR_SL_FACTOR and not alert_already_sent(state, "near_sl"):
        mark_alert_sent(state, "near_sl")
        save_state(state)
        send_telegram(
            f"⚠️ <b>GOLDPETAL — Approaching SL</b>  [{today_str}]\n"
            f"{arrow} {direction}  ·  LTP: {_p(ltp)}\n"
            f"  SL     : {_p(sl)}\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Unrealised P&L: {_r(current_pnl)}\n\n"
            f"<i>Price is within 30% of SL. Stay alert — no action taken yet.</i>"
        )
        return

    # ── Warning: Near Target (70% of move captured) ────────────────────────────
    if pct_to_target >= NEAR_TARGET_FACTOR and not alert_already_sent(state, "near_target"):
        mark_alert_sent(state, "near_target")
        save_state(state)
        send_telegram(
            f"✨ <b>GOLDPETAL — Near Target</b>  [{today_str}]\n"
            f"{arrow} {direction}  ·  LTP: {_p(ltp)}\n"
            f"  Target : {_p(target)}\n"
            f"  Entry  : {_p(entry)}\n"
            f"  Unrealised P&L: ~{_r(current_pnl)}\n\n"
            f"<i>70% of target move captured. Auto-exit fires at {_p(target)}.</i>"
        )

    log.info("[monitor] Trade open — LTP=%s  to_target=%.0f%%  to_sl=%.0f%%  unrealised=%s",
             _p(ltp), pct_to_target * 100, pct_to_sl * 100, _r(current_pnl))


# ─── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_goldpetal_monitor()
