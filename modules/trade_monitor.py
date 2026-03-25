"""
modules/trade_monitor.py — Active Trade Lifecycle Monitor

After a BUY order is placed and OCO GTT is set, this module tracks every active
trade until it is fully closed — either by the GTT (SL or target) or manually.

WHAT IT DOES EVERY SCAN CYCLE (monitor_active_trades)
──────────────────────────────────────────────────────
  1. Loads the active trade registry (data/active_trades.json).
  2. For each registered trade:
       a. Checks if the GTT is still active via kite.get_gtts().
       b. If GTT triggered → finds the exit trade in kite.trades(), computes P&L,
          sends a Telegram result notification (SL hit / Target hit / Manual exit).
       c. If still open → reads unrealised P&L from kite.positions() and sends
          one-time Telegram alerts at key thresholds:
            - Approaching SL  : premium ≤ 120% of SL price
            - Near target      : premium ≥ 80% of target price
            - Trail-stop point : premium ≥ 50% of target move (SL moved to breakeven)
  3. Trailing stop: when 50% of the target move is captured, the GTT SL leg is
     modified to breakeven (entry price) via kite.modify_gtt().
  4. At 3:20 PM IST (NSE) and 11:20 PM IST (MCX) sends an end-of-day summary
     of all trades from today's CSV log.

REGISTRY FILE
─────────────
Stored at  data/active_trades.json  in the project root.
Each entry persists across bot restarts — if the bot crashes and restarts, it
will still pick up monitoring of any open GTTs from the previous session.

THRESHOLDS
──────────
APPROACHING_SL_FACTOR   = 1.20   # alert when LTP ≤ SL × 1.20  (20% above SL)
NEAR_TARGET_FACTOR       = 0.80   # alert when LTP ≥ Target × 0.80  (80% of way)
TRAIL_STOP_FACTOR        = 0.50   # trail SL to breakeven at 50% of target move
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "active_trades.json",
)

APPROACHING_SL_FACTOR = 1.20   # alert when LTP ≤ SL × 1.20
NEAR_TARGET_FACTOR    = 0.80   # alert when LTP ≥ entry + (target-entry) × 0.80
TRAIL_STOP_FACTOR     = 0.50   # trail SL to breakeven when 50% of move captured

IST_OFFSET = timedelta(hours=5, minutes=30)


# ══════════════════════════════════════════════════════════════════════════════
# Registry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_registry() -> list:
    """Load active trades from JSON file. Returns [] if file doesn't exist."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("[trade_monitor] Could not load registry: %s", exc)
        return []


def _save_registry(trades: list) -> None:
    """Persist active trade registry to JSON file."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("[trade_monitor] Could not save registry: %s", exc)


def register_trade(
    symbol:        str,
    tradingsymbol: str,
    exchange:      str,
    direction:     str,
    entry_price:   float,
    lot_size:      int,
    gtt_id:        int,
    sl_price:      float,
    target_price:  float,
    order_id:      str = "",
) -> None:
    """
    Add a newly executed trade to the monitoring registry.

    Called immediately after a successful BUY order + GTT placement in gtt_manager.

    Args:
        symbol:        Underlying, e.g. 'NIFTY'.
        tradingsymbol: Option tradingsymbol, e.g. 'NIFTY26APR23050PE'.
        exchange:      'NFO' or 'MCX'.
        direction:     'BUY CALL' or 'BUY PUT'.
        entry_price:   Premium at which the option was bought (₹ per share).
        lot_size:      Number of shares per lot.
        gtt_id:        Zerodha GTT order ID (for monitoring + modification).
        sl_price:      Stop-loss trigger price (₹ per share premium).
        target_price:  Target trigger price (₹ per share premium).
        order_id:      Kite buy order ID (for reference).
    """
    trades = _load_registry()
    trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    entry = {
        "id":             trade_id,
        "symbol":         symbol,
        "tradingsymbol":  tradingsymbol,
        "exchange":       exchange,
        "direction":      direction,
        "entry_price":    entry_price,
        "lot_size":       lot_size,
        "gtt_id":         gtt_id,
        "sl_price":       sl_price,
        "target_price":   target_price,
        "order_id":       order_id,
        "entry_time":     datetime.now().isoformat(),
        # Alert-sent flags — prevent duplicate Telegram spam
        "sl_alerted":     False,
        "target_alerted": False,
        "trail_done":     False,
    }

    trades.append(entry)
    _save_registry(trades)
    logger.info(
        "[trade_monitor] Registered: %s  entry=₹%.1f  SL=₹%.1f  target=₹%.1f  GTT=%s",
        trade_id, entry_price, sl_price, target_price, gtt_id,
    )


def get_active_trades() -> list:
    """Return all currently monitored trades."""
    return _load_registry()


# ══════════════════════════════════════════════════════════════════════════════
# GTT & position lookups
# ══════════════════════════════════════════════════════════════════════════════

def _get_gtt_status(kite, gtt_id: int) -> Optional[dict]:
    """
    Return the GTT dict for `gtt_id`, or None if not found or API fails.
    Possible status values: 'active', 'triggered', 'disabled', 'expired', 'cancelled'.
    """
    try:
        gtts = kite.get_gtts()
        for g in gtts:
            if g.get("id") == gtt_id:
                return g
        return None   # not in list — likely triggered and expired from API
    except Exception as exc:
        logger.warning("[trade_monitor] get_gtts() failed: %s", exc)
        return None


def _get_position_ltp(kite, tradingsymbol: str) -> Optional[float]:
    """
    Return the last traded price (LTP) of an open position from kite.positions().
    Returns None if position is closed or API fails.
    """
    try:
        pos_data = kite.positions()
        for p in pos_data.get("net", []):
            if p.get("tradingsymbol") == tradingsymbol and p.get("quantity", 0) != 0:
                return float(p.get("last_price", 0) or 0)
        return None
    except Exception as exc:
        logger.warning("[trade_monitor] positions() LTP lookup failed: %s", exc)
        return None


def _get_unrealised_pnl(kite, tradingsymbol: str) -> Optional[float]:
    """Return unrealised P&L in ₹ for an open position from kite.positions()."""
    try:
        pos_data = kite.positions()
        for p in pos_data.get("net", []):
            if p.get("tradingsymbol") == tradingsymbol and p.get("quantity", 0) != 0:
                return float(p.get("unrealised", p.get("pnl", 0)) or 0)
        return None
    except Exception as exc:
        logger.warning("[trade_monitor] positions() P&L lookup failed: %s", exc)
        return None


def _find_exit_trade(kite, tradingsymbol: str, entry_time_str: str) -> Optional[dict]:
    """
    Find the SELL trade for `tradingsymbol` in today's kite.trades() that occurred
    after the entry time. Returns the trade dict or None if not found.
    """
    try:
        all_trades = kite.trades()
        entry_dt = datetime.fromisoformat(entry_time_str)

        for t in all_trades:
            if (t.get("tradingsymbol") == tradingsymbol
                    and t.get("transaction_type") == "SELL"):
                ts_raw = t.get("order_timestamp") or t.get("fill_timestamp", "")
                try:
                    trade_dt = datetime.strptime(str(ts_raw)[:19], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if trade_dt >= entry_dt:
                    return t
        return None
    except Exception as exc:
        logger.warning("[trade_monitor] trades() lookup failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Notification formatters
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_pnl(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}₹{abs(pnl):,.0f}"


def _duration(entry_time_str: str) -> str:
    """Human-readable duration since entry."""
    try:
        delta = datetime.now() - datetime.fromisoformat(entry_time_str)
        total_min = int(delta.total_seconds() / 60)
        if total_min < 60:
            return f"{total_min}min"
        h, m = divmod(total_min, 60)
        return f"{h}h {m}min"
    except Exception:
        return "—"


def _exit_message(trade: dict, exit_price: float, pnl: float, exit_type: str) -> str:
    """Format a Telegram message for a closed trade."""
    icon = "✅" if pnl >= 0 else "❌"
    result_label = {
        "target":  "🎯 TARGET HIT",
        "sl":      "🛑 STOP LOSS HIT",
        "manual":  "🤚 MANUALLY EXITED",
        "unknown": "⚡ POSITION CLOSED",
    }.get(exit_type, "⚡ POSITION CLOSED")

    lot_pnl = pnl * trade["lot_size"]

    return (
        f"{icon} {trade['symbol']} {trade['tradingsymbol']}\n"
        f"{result_label}\n"
        f"Entry  : ₹{trade['entry_price']:,.1f}\n"
        f"Exit   : ₹{exit_price:,.1f}\n"
        f"P&L    : {_fmt_pnl(lot_pnl)} ({_fmt_pnl(pnl)}/share)\n"
        f"Duration: {_duration(trade['entry_time'])}\n"
        f"GTT ID  : {trade['gtt_id']}"
    )


def _alert_approaching_sl(trade: dict, ltp: float, pnl: float) -> str:
    lot_pnl = pnl * trade["lot_size"] if pnl is not None else 0
    return (
        f"⚠️ {trade['symbol']} — APPROACHING STOP LOSS\n"
        f"Current premium : ₹{ltp:,.1f}\n"
        f"SL trigger      : ₹{trade['sl_price']:,.1f}\n"
        f"Floating P&L    : {_fmt_pnl(lot_pnl)}\n"
        f"↳ Be ready to cut if SL is breached"
    )


def _alert_near_target(trade: dict, ltp: float, pnl: float) -> str:
    lot_pnl = pnl * trade["lot_size"] if pnl is not None else 0
    return (
        f"💡 {trade['symbol']} — NEAR TARGET\n"
        f"Current premium : ₹{ltp:,.1f}\n"
        f"Target trigger  : ₹{trade['target_price']:,.1f}\n"
        f"Floating P&L    : {_fmt_pnl(lot_pnl)}\n"
        f"↳ Target GTT is set — consider partial booking if IV drops"
    )


def _alert_trail_done(trade: dict, ltp: float, new_sl: float) -> str:
    return (
        f"🔄 {trade['symbol']} — STOP LOSS TRAILED TO BREAKEVEN\n"
        f"Current premium : ₹{ltp:,.1f}\n"
        f"New SL          : ₹{new_sl:,.1f} (entry = breakeven)\n"
        f"Original SL was : ₹{trade['sl_price']:,.1f}\n"
        f"↳ Trade is now risk-free — profits locked in on downside"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trailing stop
# ══════════════════════════════════════════════════════════════════════════════

def _attempt_trail_stop(kite, trade: dict, ltp: float) -> bool:
    """
    If the premium has moved 50%+ toward the target, modify the GTT's SL leg
    to the entry price (breakeven), eliminating downside risk.

    Returns True if trail was applied successfully.
    """
    entry  = trade["entry_price"]
    target = trade["target_price"]
    sl     = trade["sl_price"]

    target_move    = target - entry          # total expected move
    captured_move  = ltp    - entry          # how much is actually captured

    if target_move <= 0 or captured_move < target_move * TRAIL_STOP_FACTOR:
        return False

    new_sl = round(entry, 1)   # move SL to breakeven

    try:
        kite.modify_gtt(
            trigger_id     = trade["gtt_id"],
            trigger_type   = "two-leg",
            tradingsymbol  = trade["tradingsymbol"],
            exchange       = trade["exchange"],
            trigger_values = [new_sl, target],
            last_price     = ltp,
            orders         = [
                {
                    "transaction_type": "SELL",
                    "quantity":         trade["lot_size"],
                    "product":          "MIS",
                    "order_type":       "LIMIT",
                    "price":            round(new_sl * 0.80, 1),
                },
                {
                    "transaction_type": "SELL",
                    "quantity":         trade["lot_size"],
                    "product":          "MIS",
                    "order_type":       "LIMIT",
                    "price":            round(target, 1),
                },
            ],
        )
        logger.info(
            "[trade_monitor] Trail stop applied: %s SL moved ₹%.1f → ₹%.1f",
            trade["tradingsymbol"], sl, new_sl,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[trade_monitor] Trail stop modify_gtt failed for GTT %s: %s",
            trade["gtt_id"], exc,
        )
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Main monitor loop
# ══════════════════════════════════════════════════════════════════════════════

def monitor_active_trades(kite, send_alert_fn) -> None:
    """
    Called every scan cycle. Checks every registered trade and dispatches
    Telegram notifications based on GTT status and floating P&L.

    Args:
        kite:          Authenticated KiteConnect instance.
        send_alert_fn: Callable (str) → bool — telegram_alert.send_alert.
    """
    trades = _load_registry()
    if not trades:
        return

    logger.info("[trade_monitor] Monitoring %d active trade(s).", len(trades))
    updated_trades = []

    for trade in trades:
        ts    = trade["tradingsymbol"]
        gtt_id = trade.get("gtt_id")

        # ── 1. Check GTT status ──────────────────────────────────────────
        gtt = _get_gtt_status(kite, gtt_id) if gtt_id else None
        gtt_status = gtt.get("status", "unknown") if gtt else "not_found"

        trade_closed = gtt_status in ("triggered", "disabled", "expired",
                                      "cancelled", "not_found")

        if trade_closed:
            # ── 2. Find the exit trade to compute P&L ────────────────────
            exit_trade = _find_exit_trade(kite, ts, trade["entry_time"])

            if exit_trade:
                exit_price = float(exit_trade.get("average_price", 0)
                                   or exit_trade.get("price", 0))
            else:
                exit_price = trade["entry_price"]   # fallback — can't find fill

            pnl_per_share = exit_price - trade["entry_price"]
            lot_pnl       = pnl_per_share * trade["lot_size"]

            # Determine how it closed
            if exit_price <= trade["sl_price"] * 1.05:
                exit_type = "sl"
            elif exit_price >= trade["target_price"] * 0.95:
                exit_type = "target"
            elif gtt_status in ("disabled", "cancelled"):
                exit_type = "manual"
            else:
                exit_type = "unknown"

            msg = _exit_message(trade, exit_price, pnl_per_share, exit_type)
            logger.info(
                "[trade_monitor] Trade closed: %s  exit=₹%.1f  P&L=₹%.0f  [%s]",
                ts, exit_price, lot_pnl, exit_type,
            )
            try:
                send_alert_fn(msg)
            except Exception as exc:
                logger.error("[trade_monitor] Could not send exit alert: %s", exc)

            # Trade is done — do NOT add to updated_trades (removes from registry)
            continue

        # ── 3. Trade still open: floating P&L + threshold alerts ─────────
        ltp = _get_position_ltp(kite, ts)
        if ltp is None or ltp <= 0:
            updated_trades.append(trade)
            continue

        pnl_per_share = ltp - trade["entry_price"]
        unrealised    = pnl_per_share * trade["lot_size"]

        logger.info(
            "[trade_monitor] Open: %s  LTP=₹%.1f  P&L=₹%.0f  GTT=%s",
            ts, ltp, unrealised, gtt_status,
        )

        # ── 3a. Approaching SL alert (one-time) ──────────────────────────
        if (not trade.get("sl_alerted")
                and ltp <= trade["sl_price"] * APPROACHING_SL_FACTOR):
            try:
                send_alert_fn(_alert_approaching_sl(trade, ltp, pnl_per_share))
            except Exception as exc:
                logger.error("[trade_monitor] SL approach alert failed: %s", exc)
            trade["sl_alerted"] = True
            logger.info("[trade_monitor] SL approach alert sent for %s", ts)

        # ── 3b. Near target alert (one-time) ─────────────────────────────
        target_threshold = (trade["entry_price"]
                            + (trade["target_price"] - trade["entry_price"])
                            * NEAR_TARGET_FACTOR)
        if (not trade.get("target_alerted")
                and ltp >= target_threshold):
            try:
                send_alert_fn(_alert_near_target(trade, ltp, pnl_per_share))
            except Exception as exc:
                logger.error("[trade_monitor] Near target alert failed: %s", exc)
            trade["target_alerted"] = True
            logger.info("[trade_monitor] Near target alert sent for %s", ts)

        # ── 3c. Trail stop to breakeven (one-time) ────────────────────────
        if not trade.get("trail_done"):
            trailed = _attempt_trail_stop(kite, trade, ltp)
            if trailed:
                new_sl = round(trade["entry_price"], 1)
                try:
                    send_alert_fn(_alert_trail_done(trade, ltp, new_sl))
                except Exception as exc:
                    logger.error("[trade_monitor] Trail alert failed: %s", exc)
                trade["trail_done"]   = True
                trade["sl_price"]     = new_sl   # keep registry in sync
                trade["sl_alerted"]   = False    # reset — new SL to watch
                logger.info("[trade_monitor] Trail stop done for %s", ts)

        updated_trades.append(trade)

    _save_registry(updated_trades)


# ══════════════════════════════════════════════════════════════════════════════
# End-of-day summary
# ══════════════════════════════════════════════════════════════════════════════

def send_eod_summary(send_alert_fn, log_csv_path: str) -> None:
    """
    Build and send an end-of-day summary from today's trade_log.csv.

    Called by a scheduled job at 3:20 PM (NSE) and 11:20 PM (MCX).

    Shows:
      • Total trades signalled today
      • Breakdown by symbol and conviction
      • Auto-executed trades with P&L (if monitoring registry has exits today)
    """
    today_str = date.today().isoformat()

    try:
        import csv
        today_trades = []
        if os.path.exists(log_csv_path):
            with open(log_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp", "")
                    if ts.startswith(today_str):
                        today_trades.append(row)
    except Exception as exc:
        logger.warning("[trade_monitor] EOD CSV read failed: %s", exc)
        today_trades = []

    if not today_trades:
        send_alert_fn(
            f"📋 End-of-Day Summary — {today_str}\n"
            f"No signals were generated today."
        )
        return

    total   = len(today_trades)
    high    = sum(1 for t in today_trades if t.get("conviction") == "HIGH")
    medium  = sum(1 for t in today_trades if t.get("conviction") == "MEDIUM")
    by_sym  = {}
    for t in today_trades:
        sym = t.get("symbol", "?")
        by_sym[sym] = by_sym.get(sym, 0) + 1

    sym_lines = "\n".join(f"  {sym}: {cnt}" for sym, cnt in sorted(by_sym.items()))

    msg = (
        f"📋 End-of-Day Summary — {today_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total signals : {total}\n"
        f"  HIGH        : {high}\n"
        f"  MEDIUM      : {medium}\n"
        f"\nBy symbol:\n{sym_lines}\n"
        f"\n⚠️ P&L tracking active only for auto-executed trades.\n"
        f"   Manual trades: check Zerodha P&L tab."
    )
    try:
        send_alert_fn(msg)
        logger.info("[trade_monitor] EOD summary sent.")
    except Exception as exc:
        logger.error("[trade_monitor] EOD summary failed to send: %s", exc)
