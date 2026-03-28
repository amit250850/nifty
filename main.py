"""
main.py — NiftySignalBot Entry Point

Signal-only options alert system for NIFTY, BANKNIFTY, SILVERM, GOLDM.
Signals are sent to Telegram for manual trading.

AUTO-EXECUTE MODE (optional — controlled by gtt_manager.ENABLE_AUTO_EXECUTE):
  When enabled, a BUY order + OCO GTT (SL + Target) is placed automatically
  ONLY when ALL four guards pass simultaneously:
      Guard 1 — Conviction  : HIGH (4/4 signals agree)
      Guard 2 — PCR         : ideal for direction
      Guard 3 — Budget      : available margin ≥ lot cost
      Guard 4 — Liquidity   : bid-ask spread ≤ 5% of premium
  If any guard fails → manual alert only, no order.

Architecture:
  • APScheduler runs scan_and_signal() every 5 minutes, 9:00 AM – 11:30 PM IST.
  • Each cycle:
      1. Scan option chain via Kite Connect NFO API (NIFTY/BANKNIFTY) or
         MCX spot-only (SILVERM/GOLDM).
      2. Record OI snapshot in SQLite (NIFTY/BANKNIFTY only).
      3. Compute chart signals (EMA/RSI/VWAP/SuperTrend) via Kite + yfinance.
      4. If conviction ≥ MEDIUM: select strike, run all guards, send alert.
      5. If all 4 guards pass: auto-execute BUY + OCO GTT (when enabled).
      6. Log to CSV.

Usage:
    python login.py     ← once per trading day (refreshes Kite access token)
    python main.py      ← starts the bot
"""

import logging
import os
import sys
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from modules.option_chain    import scan_option_chain
from modules.chart_signals   import compute_signals
from modules.strike_selector import select_strike
from modules.telegram_alert  import send_full_alert, send_error_alert
from modules.trade_logger    import log_signal, initialise_log
from modules.position_guard  import has_open_position, check_margin, check_liquidity
from modules.gtt_manager     import (
    find_option_tradingsymbol, all_guards_pass, execute_trade,
    ENABLE_AUTO_EXECUTE,
)
from modules.oi_tracker      import initialise_db as init_oi_db, record_snapshot, get_oi_trend
from modules.trade_monitor   import monitor_active_trades, monitor_unregistered_positions, send_eod_summary

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("niftysignalbot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ── Timezone & config ──────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

load_dotenv()
API_KEY      = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

SYMBOLS                = ["NIFTY", "BANKNIFTY", "SILVERM", "GOLDM"]
SCAN_INTERVAL_MINUTES  = 5

# ── Per-symbol minimum conviction for sending alerts ───────────────────────────
# Derived from 60-day backtest (DTE≥3, SL cooldown, --no-vix):
#
#   NIFTY HIGH   : 18 trades, 61.1% win, +₹26,682  ← trade
#   NIFTY MEDIUM :  9 trades, 66.7% win, +₹15,712  ← trade (MEDIUM actually better!)
#   BANKNIFTY HIGH : 16 trades, 43.8% win, +₹17,002  ← trade (R:R saves it in trends)
#   BANKNIFTY MEDIUM: 11 trades, 45.5% win, +₹968  ← SKIP (₹88/trade avg, not worth it)
#
# BANKNIFTY MEDIUM is barely breakeven (+₹968 over 60 days) and generates
# consecutive SL streaks in choppy markets. HIGH only for BANKNIFTY.
# NIFTY signals are consistently reliable at both conviction levels.
SYMBOL_MIN_CONVICTION: dict[str, str] = {
    "NIFTY":     "MEDIUM",   # Both HIGH and MEDIUM are profitable
    "BANKNIFTY": "HIGH",     # MEDIUM gives only +₹968/60 days — skip
    "SILVERM":   "MEDIUM",   # No backtest data yet — default to MEDIUM
    "GOLDM":     "MEDIUM",   # No backtest data yet — default to MEDIUM
}

# ── NSE hours (equity index options) ──────────────────────────────────────────
MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
# NSE Telegram alerts: 10:00 AM – 3:00 PM
# Skips 9:15–10:00 AM opening — first 45 min has extreme volatility, fake breakouts,
# and algo-driven spikes that flush retail before the real trend establishes.
# Signals fired in this window have much lower reliability.
ALERT_START_H, ALERT_START_M = 10, 0
ALERT_END_H,   ALERT_END_M   = 15, 0

# ── MCX hours (commodity options — SILVERM/GOLDM trade until 11:30 PM IST) ────
MCX_OPEN_H,  MCX_OPEN_M  = 9,  0
MCX_CLOSE_H, MCX_CLOSE_M = 23, 30
# MCX symbols that run the extended session
MCX_SYMBOLS = {"SILVERM", "GOLDM"}

# ── GOLDM budget filter ────────────────────────────────────────────────────────
# Skip GOLDM alerts if the option lot cost exceeds this threshold
GOLDM_MAX_LOT_COST = 50_000

# Global Kite client
kite: KiteConnect = None


# ── Kite initialisation ────────────────────────────────────────────────────────

def initialise_kite() -> bool:
    """
    Initialise global Kite Connect client from .env credentials.
    Kite is used only for:
      • Session validation (profile check)
      • NFO instruments list (expiry date lookup)

    Returns True on success, False if credentials are missing/invalid.
    """
    global kite, ACCESS_TOKEN

    load_dotenv(override=True)
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

    if not API_KEY:
        logger.critical("KITE_API_KEY not found in .env. Exiting.")
        return False

    if not ACCESS_TOKEN:
        logger.critical(
            "KITE_ACCESS_TOKEN not set. Run `python login.py` first."
        )
        return False

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    try:
        profile = kite.profile()
        logger.info(
            "✅  Kite session active — User: %s (%s)",
            profile.get("user_name", "?"),
            profile.get("user_id",   "?"),
        )
        return True
    except KiteException as exc:
        logger.error("❌  Kite token validation failed: %s", exc)
        logger.error("Run `python login.py` to refresh your access token.")
        return False


def is_symbol_market_open(symbol: str) -> bool:
    """
    Return True if the market for this specific symbol is currently open.

    NSE symbols (NIFTY, BANKNIFTY): Mon–Fri 9:15 AM – 3:30 PM IST
    MCX symbols (SILVERM):            Mon–Fri 9:00 AM – 11:30 PM IST
    """
    now = datetime.now(IST)
    if now.weekday() >= 5:    # Saturday / Sunday
        return False
    if symbol in MCX_SYMBOLS:
        open_time  = now.replace(hour=MCX_OPEN_H,  minute=MCX_OPEN_M,  second=0, microsecond=0)
        close_time = now.replace(hour=MCX_CLOSE_H, minute=MCX_CLOSE_M, second=0, microsecond=0)
    else:
        open_time  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
        close_time = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_time <= now <= close_time


def is_market_open() -> bool:
    """Return True if ANY symbol's market is currently open (drives scheduler wake-up)."""
    return any(is_symbol_market_open(s) for s in SYMBOLS)


def is_alert_window_for(symbol: str) -> bool:
    """
    Per-symbol Telegram alert window:

    NSE (NIFTY, BANKNIFTY):
      10:00 AM – 3:00 PM — skips the first 45 minutes of opening-noise
      algo spikes and fake breakouts (9:15–10:00 AM).

    MCX (SILVERM, GOLDM):
      Full MCX session 9:00 AM – 11:30 PM — commodity moves happen
      morning AND evening (US/London market open). The HIGH/MEDIUM
      conviction threshold naturally prevents constant pinging.
    """
    now = datetime.now(IST)
    if symbol in MCX_SYMBOLS:
        open_time  = now.replace(hour=MCX_OPEN_H,  minute=MCX_OPEN_M,  second=0, microsecond=0)
        close_time = now.replace(hour=MCX_CLOSE_H, minute=MCX_CLOSE_M, second=0, microsecond=0)
        return open_time <= now <= close_time
    else:
        alert_start = now.replace(hour=ALERT_START_H, minute=ALERT_START_M, second=0, microsecond=0)
        alert_end   = now.replace(hour=ALERT_END_H,   minute=ALERT_END_M,   second=0, microsecond=0)
        return alert_start <= now <= alert_end


def _is_token_expiry_error(exc: Exception) -> bool:
    """Detect Kite token-expiry exceptions by message content."""
    return isinstance(exc, KiteException) and (
        "token" in str(exc).lower() or "TokenException" in type(exc).__name__
    )


# ── Main scan cycle ────────────────────────────────────────────────────────────

def scan_and_signal() -> None:
    """
    Full scan-and-signal cycle. Called every 5 minutes by APScheduler.

    Extended flow vs previous version:
      • OI snapshot recorded each cycle (NIFTY/BANKNIFTY) → OI trend in alerts
      • Duplicate position check before each alert (suppresses repeat signals)
      • Margin check included in alert message (₹ available vs ₹ needed)
      • Tradingsymbol looked up for liquidity check + GTT placement
      • AUTO-EXECUTE: BUY + OCO GTT placed when all 4 guards pass (when enabled)
    """
    if not is_market_open():
        logger.info("⏸  All markets closed — skipping scan cycle.")
        return

    now_str = datetime.now(IST).strftime("%H:%M:%S IST")
    logger.info("=" * 60)
    logger.info("🔍  Starting scan cycle at %s", now_str)
    logger.info("=" * 60)

    # ── Active trade monitor (runs every cycle before new signals) ──────────────
    # Checks GTT status, sends exit notifications, floating P&L alerts,
    # and applies trailing stop if the 50% target-move milestone is reached.
    try:
        from modules.telegram_alert import send_alert
        monitor_active_trades(kite, send_alert)
    except Exception as exc:
        logger.warning("[monitor] Active trade check failed (non-fatal): %s", exc)

    # ── Manual position watcher ──────────────────────────────────────────────────
    # Watches ALL open option positions from kite.positions() — including those
    # placed manually via Zerodha web/app that are NOT in the bot's registry.
    # Sends P&L advisory alerts (down 25/40%, up 40/70%, trail-stop suggestion).
    try:
        from modules.telegram_alert import send_alert
        monitor_unregistered_positions(kite, send_alert)
    except Exception as exc:
        logger.warning("[monitor] Manual position watcher failed (non-fatal): %s", exc)

    for symbol in SYMBOLS:
        if not is_symbol_market_open(symbol):
            logger.info("─── %s — market closed, skipping ───", symbol)
            continue

        logger.info("─── Processing %s ───", symbol)

        # ── Step 1: Option Chain ────────────────────────────────────────────
        try:
            oc_data = scan_option_chain(symbol, kite=kite)
        except Exception as exc:
            logger.error("[%s] Option chain scan failed: %s", symbol, exc)
            oc_data = None

        if oc_data is None:
            logger.warning("[%s] Skipping — option chain unavailable.", symbol)
            continue

        spot = oc_data.get("underlying", 0.0)
        if spot <= 0:
            logger.warning("[%s] Invalid spot price: %s", symbol, spot)
            continue

        pcr_val     = oc_data.get("pcr")
        maxpain_val = oc_data.get("max_pain")
        ivrank_val  = oc_data.get("iv_rank")
        logger.info(
            "[%s] Spot=%.2f  PCR=%s  MaxPain=%s  IVRank=%s",
            symbol, spot,
            f"{pcr_val:.2f}"     if pcr_val     is not None else "N/A",
            str(maxpain_val)     if maxpain_val  is not None else "N/A",
            f"{ivrank_val:.1f}%" if ivrank_val   is not None else "N/A",
        )

        # ── Step 1b: OI snapshot (NSE only — feeds trend tracker) ──────────
        try:
            record_snapshot(symbol, oc_data)
        except Exception as exc:
            logger.debug("[%s] OI snapshot error (non-fatal): %s", symbol, exc)

        # ── Step 2: Chart Signals ──────────────────────────────────────────
        try:
            signal = compute_signals(symbol, kite=kite)
        except Exception as exc:
            logger.error("[%s] Chart signal error: %s", symbol, exc)
            continue

        if signal is None:
            logger.info("[%s] ⏭  NO TRADE — insufficient signal agreement.", symbol)
            continue

        logger.info(
            "[%s] Signal: %s | Conviction: %s | Agreed: %d/4",
            symbol, signal.direction, signal.conviction, signal.signals_agreed,
        )

        # ── Step 2b: Per-symbol conviction gate ────────────────────────────
        # Backtest (60 days) shows BANKNIFTY MEDIUM gives only +₹968 over 60
        # days (45.5% win, ₹88 avg/trade) vs NIFTY MEDIUM at 66.7% / +₹15,712.
        # Skip below-threshold conviction signals to reduce noise and SL churn.
        _min_conv = SYMBOL_MIN_CONVICTION.get(symbol, "MEDIUM")
        _conv_rank = {"MEDIUM": 1, "HIGH": 2}
        if _conv_rank.get(signal.conviction, 0) < _conv_rank.get(_min_conv, 1):
            logger.info(
                "[%s] ⏭  Conviction gate — got %s but min required is %s. "
                "BANKNIFTY MEDIUM: +₹968/60 days (45.5%% win) — not worth trading.",
                symbol, signal.conviction, _min_conv,
            )
            continue

        # ── Step 3: Duplicate position check ──────────────────────────────
        # If an open option position already exists for this symbol+direction,
        # skip the alert entirely — prevents doubling into a losing trade.
        try:
            if has_open_position(kite, symbol, signal.direction):
                logger.info(
                    "[%s] ⏭  Duplicate suppressed — open %s position already exists.",
                    symbol, signal.direction,
                )
                continue
        except Exception as exc:
            logger.debug("[%s] Position check error (non-fatal): %s", symbol, exc)

        # ── Step 4: Strike Selection ───────────────────────────────────────
        try:
            strike = select_strike(
                kite       = kite,
                symbol     = symbol,
                spot       = spot,
                direction  = signal.direction,
                oc_data    = oc_data,
                conviction = signal.conviction,
            )
        except KiteException as exc:
            if _is_token_expiry_error(exc):
                msg = "⚠️  Kite access token expired. Run `python login.py` and restart."
                logger.warning(msg)
                send_error_alert(msg)
                return
            logger.error("[%s] Kite error in strike selection: %s", symbol, exc)
            continue
        except Exception as exc:
            logger.error("[%s] Strike selection failed: %s", symbol, exc)
            continue

        if strike is None:
            logger.warning("[%s] Strike selection returned None — skipping.", symbol)
            continue

        logger.info(
            "[%s] Strike: %s%s  Expiry: %s  Premium: ₹%.2f  LotCost: ₹%.0f",
            symbol, strike.otm_strike, strike.option_type,
            strike.expiry_date, strike.premium, strike.lot_cost,
        )

        # ── Step 4b: DTE guard — only trade Monday/Friday ──────────────────
        # Backtest (60 days, 114 HIGH signals) win rate by DTE:
        #   DTE 0  (expiry day):  ~0%  win rate — theta trap
        #   DTE 1  (Wed/Thu):     17%  win rate — not viable
        #   DTE 2  (Tue):         24%  win rate — not viable
        #   DTE 3+ (Mon/Fri):     50-55% win rate — profitable
        # Only enter when option has at least 3 calendar days to expiry.
        # This means Monday (DTE≈3) and Friday (DTE≈6) entries only.
        _MIN_DTE = 3
        _today = datetime.now(IST).date()
        try:
            _expiry_date = strike.expiry_raw.date() if hasattr(strike.expiry_raw, "date") else strike.expiry_raw
            _dte = (_expiry_date - _today).days
        except Exception:
            _dte = 999   # unknown — allow through safely
        if _dte < _MIN_DTE:
            logger.info(
                "[%s] ⏭  DTE=%d < %d — skipping (Tue/Wed/Thu entry). "
                "Strategy only trades Mon/Fri (DTE≥3). "
                "Backtest: DTE<3 has <25%% win rate. Expiry: %s",
                symbol, _dte, _MIN_DTE, strike.expiry_date,
            )
            continue

        # ── Step 4c: GOLDM hard budget cap ────────────────────────────────
        if symbol == "GOLDM" and strike.lot_cost >= GOLDM_MAX_LOT_COST:
            logger.info(
                "[GOLDM] Lot cost ₹%.0f ≥ ₹%.0f cap — skipping.", strike.lot_cost, GOLDM_MAX_LOT_COST
            )
            continue

        # ── Step 5: Margin check ───────────────────────────────────────────
        margin_ok, available_margin = check_margin(kite, symbol, strike.lot_cost)

        # ── Step 6: OI trend (NIFTY/BANKNIFTY only) ───────────────────────
        oi_trend = None
        if symbol not in MCX_SYMBOLS:
            try:
                oi_trend = get_oi_trend(symbol)
            except Exception as exc:
                logger.debug("[%s] OI trend error (non-fatal): %s", symbol, exc)

        # ── Step 7: Tradingsymbol lookup + liquidity check ─────────────────
        # Required for both liquidity assessment and GTT placement.
        exchange       = "MCX" if symbol in MCX_SYMBOLS else "NFO"
        tradingsymbol  = None
        liquidity_info = None

        try:
            tradingsymbol = find_option_tradingsymbol(
                kite         = kite,
                symbol       = symbol,
                expiry_raw   = strike.expiry_raw,
                strike_price = float(strike.otm_strike),
                option_type  = strike.option_type,
                exchange     = exchange,
            )
        except Exception as exc:
            logger.debug("[%s] Tradingsymbol lookup error (non-fatal): %s", symbol, exc)

        if tradingsymbol:
            try:
                liquidity_info = check_liquidity(kite, tradingsymbol, exchange)
            except Exception as exc:
                logger.debug("[%s] Liquidity check error (non-fatal): %s", symbol, exc)

        liquid = liquidity_info.get("liquid", True) if liquidity_info else True

        # ── Step 8: Evaluate auto-execute guards ───────────────────────────
        # PCR ideal check (reuse telegram_alert logic for consistency)
        from modules.telegram_alert import evaluate_pcr
        pcr_ideal = True   # default for MCX (no PCR)
        if pcr_val is not None:
            pcr_ideal = evaluate_pcr(pcr_val, signal.direction)["ideal"]

        auto_ok, auto_reason = all_guards_pass(
            conviction = signal.conviction,
            pcr_ideal  = pcr_ideal,
            margin_ok  = margin_ok,
            liquid     = liquid,
            exchange   = exchange,
        )

        # ── Step 9: Telegram Alert ─────────────────────────────────────────
        trade_result = None
        if is_alert_window_for(symbol):
            # Build the extra_info dict for the alert formatter
            extra_info = {
                "available_margin": available_margin,
                "liquidity":        liquidity_info,
                "oi_trend":         oi_trend,
                "auto_ok":          auto_ok,
                "auto_reason":      auto_reason,
                "tradingsymbol":    tradingsymbol,
                "auto_exec_armed":  ENABLE_AUTO_EXECUTE,
            }
            try:
                sent = send_full_alert(signal, strike, oc_data, extra_info=extra_info)
                logger.info("[%s] %s Telegram alert.", symbol,
                            "✅" if sent else "⚠️  Failed to send")
            except Exception as exc:
                logger.error("[%s] Telegram error: %s", symbol, exc)

            # ── Step 10: Auto-execute (all 4 guards must pass) ────────────
            if auto_ok and tradingsymbol and ENABLE_AUTO_EXECUTE:
                try:
                    trade_result = execute_trade(
                        kite          = kite,
                        tradingsymbol = tradingsymbol,
                        exchange      = exchange,
                        lot_size      = strike.lot_size,
                        ltp           = strike.premium,
                        sl_price      = strike.stop_loss,
                        target_price  = strike.target,
                        symbol        = symbol,
                    )
                    if trade_result:
                        from modules.telegram_alert import send_alert
                        send_alert(trade_result["message"])
                        logger.info("[%s] Auto-execute result: %s", symbol, trade_result["message"])
                except Exception as exc:
                    logger.error("[%s] Auto-execute error: %s", symbol, exc)

        else:
            window_hint = "10:00 AM–3:00 PM" if symbol not in MCX_SYMBOLS else "9:00 AM–11:30 PM"
            logger.info("[%s] ⏰ Outside alert window (%s) — signal logged only.", symbol, window_hint)

        # ── Step 11: Log to CSV ────────────────────────────────────────────
        try:
            logged = log_signal(signal, strike, oc_data)
            logger.info("[%s] %s CSV log.", symbol,
                        "✅" if logged else "⚠️  Failed to log")
        except Exception as exc:
            logger.error("[%s] Logging error: %s", symbol, exc)

        time.sleep(2)   # rate-limit between symbols

    logger.info("✔  Scan cycle complete.\n")


# ── APScheduler setup ──────────────────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    """
    Schedule scan_and_signal() every 5 minutes across the full combined
    market window: 9:00 AM – 11:30 PM IST (Mon–Fri).

    The combined window covers both NSE (closes 3:30 PM) and MCX SILVERM
    (closes 11:30 PM). The per-symbol is_symbol_market_open() check inside
    the job skips NSE symbols after 3:30 PM automatically — no extra logic needed.
    """
    scheduler = BlockingScheduler(timezone=IST)

    trigger = CronTrigger(
        day_of_week        = "mon-fri",
        hour               = f"{MCX_OPEN_H}-{MCX_CLOSE_H}",   # 9 – 23 (11 PM)
        minute             = "0/5",
        timezone           = IST,
    )

    scheduler.add_job(
        func               = scan_and_signal,
        trigger            = trigger,
        id                 = "signal_scan",
        name               = "NSE + MCX Signal Scanner",
        misfire_grace_time = 60,
        coalesce           = True,
    )

    # ── End-of-Day Summary jobs ─────────────────────────────────────────────
    # NSE: 3:20 PM — NSE closes 3:30 PM, GTTs settle by 3:20 PM
    # MCX: 11:20 PM — MCX SILVERM/GOLDM close 11:30 PM
    def _nse_eod():
        from modules.telegram_alert import send_alert
        send_eod_summary(send_alert, log_csv_path="trade_log.csv")

    def _mcx_eod():
        from modules.telegram_alert import send_alert
        send_eod_summary(send_alert, log_csv_path="trade_log.csv")

    scheduler.add_job(
        func    = _nse_eod,
        trigger = CronTrigger(day_of_week="mon-fri", hour=15, minute=20, timezone=IST),
        id      = "nse_eod_summary",
        name    = "NSE End-of-Day Summary",
        misfire_grace_time = 120,
        coalesce           = True,
    )
    scheduler.add_job(
        func    = _mcx_eod,
        trigger = CronTrigger(day_of_week="mon-fri", hour=23, minute=20, timezone=IST),
        id      = "mcx_eod_summary",
        name    = "MCX End-of-Day Summary",
        misfire_grace_time = 120,
        coalesce           = True,
    )

    return scheduler


# ── Startup banner ─────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  🟢  NiftySignalBot — OPTIONS SIGNAL SYSTEM")
    exec_mode = "⚡ AUTO-EXECUTE + MONITOR" if ENABLE_AUTO_EXECUTE else "🔒 SIGNAL-ONLY (alert mode)"
    print(f"  Mode: {exec_mode}")
    print("=" * 60)
    print(f"  Symbols   : {', '.join(SYMBOLS)}")
    print(f"  NSE hours : 9:15 AM – 3:30 PM  | Alerts: 10:00 AM–3:00 PM (skips 9:15–10:00 opening)")
    print(f"  MCX hours : 9:00 AM – 11:30 PM | Alerts: full session (SILVERM, GOLDM)")
    print(f"  Scan      : Every {SCAN_INTERVAL_MINUTES} min (Mon–Fri)")
    print(f"  Chart data: yfinance (^NSEI, ^NSEBANK) + Kite MCX historical")
    print(f"  OC + LTP  : Kite Connect NFO API (NIFTY/BANKNIFTY) | MCX: spot-only")
    print(f"  Budget    : NIFTY/BANKNIFTY ₹10k–₹20k/lot | SILVERM ~₹5k–₹15k | GOLDM <₹50k/lot")
    print(f"  Guards    : Duplicate check | Margin | Liquidity | Conviction+PCR")
    exec_status = "⚡ ENABLED" if ENABLE_AUTO_EXECUTE else "🔒 DISABLED (alert-only)"
    print(f"  Auto-exec : {exec_status}")
    print(f"  OI Tracker: data/oi_tracker.db (20-min rolling trend)")
    print(f"  Log file  : trade_log.csv")
    print("=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Startup sequence:
      1. Validate Kite session (needed only for NFO expiry lookup).
      2. Initialise trade log CSV.
      3. Run one immediate scan if market is open.
      4. Start APScheduler blocking loop.
    """
    print_banner()

    if not initialise_kite():
        logger.critical("Cannot start — Kite session invalid. Run login.py first.")
        sys.exit(1)

    initialise_log()
    logger.info("Trade log: %s", os.path.abspath("trade_log.csv"))

    init_oi_db()
    logger.info("OI tracker DB ready.")

    if ENABLE_AUTO_EXECUTE:
        logger.warning(
            "⚡ AUTO-EXECUTE is ENABLED — BUY orders will be placed when all guards pass!"
        )
    else:
        logger.info("🔒 Auto-execute is DISABLED — alert-only mode.")

    # Immediate scan on startup if market is open
    if is_market_open():
        logger.info("Market is open — running initial scan now …")
        try:
            scan_and_signal()
        except Exception as exc:
            logger.error("Initial scan failed: %s", exc)
    else:
        now_ist = datetime.now(IST)
        logger.info(
            "All markets closed (%s IST). "
            "Scheduler activates at 9:00 AM on the next trading day.",
            now_ist.strftime("%H:%M"),
        )

    scheduler = build_scheduler()
    logger.info(
        "⏰  Scheduler started — every %d min during market hours. Press Ctrl+C to stop.\n",
        SCAN_INTERVAL_MINUTES,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑  NiftySignalBot stopped.")
    except Exception as exc:
        logger.critical("Scheduler crashed: %s", exc)
        raise


if __name__ == "__main__":
    main()