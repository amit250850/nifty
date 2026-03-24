"""
main.py — NiftySignalBot Entry Point

Signal-only options alert system for NIFTY and BANKNIFTY.
NO auto order execution — all signals are sent to Telegram for manual trading.

Architecture:
  • APScheduler runs scan_and_signal() every 5 minutes, 9:15 AM – 3:30 PM IST.
  • Each cycle:
      1. Scan option chain (NIFTY + BANKNIFTY) via nsepython.
      2. Compute chart signals (EMA/RSI/VWAP/SuperTrend) via yfinance (free).
      3. If conviction ≥ MEDIUM: select strike using NSE chain LTP (free),
         send Telegram alert, log to CSV.
      4. If conviction = NO TRADE: skip silently.
  • Kite is used only for: session validation + NFO instrument list (expiry lookup).
  • Kite token expiry is caught gracefully — warning logged, cycle skips.

Usage:
    python login.py     ← once per trading day
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

# ── NSE hours (equity index options) ──────────────────────────────────────────
MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
# NSE Telegram alerts: 9:15 AM – 3:00 PM (full NSE session, stops 30 min before close)
ALERT_END_H, ALERT_END_M = 15, 0

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
      9:15 AM – 11:00 AM only — avoids notification overload all day.

    MCX (SILVERM):
      Full MCX session 9:00 AM – 11:30 PM — silver moves happen morning
      AND evening (US/London market open). The HIGH/MEDIUM conviction
      threshold naturally prevents constant pinging.
    """
    now = datetime.now(IST)
    if symbol in MCX_SYMBOLS:
        open_time  = now.replace(hour=MCX_OPEN_H,  minute=MCX_OPEN_M,  second=0, microsecond=0)
        close_time = now.replace(hour=MCX_CLOSE_H, minute=MCX_CLOSE_M, second=0, microsecond=0)
        return open_time <= now <= close_time
    else:
        open_time = now.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
        alert_end = now.replace(hour=ALERT_END_H,   minute=ALERT_END_M,   second=0, microsecond=0)
        return open_time <= now <= alert_end


def _is_token_expiry_error(exc: Exception) -> bool:
    """Detect Kite token-expiry exceptions by message content."""
    return isinstance(exc, KiteException) and (
        "token" in str(exc).lower() or "TokenException" in type(exc).__name__
    )


# ── Main scan cycle ────────────────────────────────────────────────────────────

def scan_and_signal() -> None:
    """
    Full scan-and-signal cycle. Called every 5 minutes by APScheduler.

    Data sources used:
      • Kite Connect NFO API          → spot price, PCR, OI, MaxPain, LTP map
                                        (replaces unreliable NSE HTTP session)
      • yfinance                      → 1H OHLCV for EMA/RSI/VWAP/SuperTrend
      • Kite Connect MCX API          → real MCX SILVERM futures price

    No Kite historical data or Kite LTP subscription required.
    """
    if not is_market_open():
        logger.info("⏸  All markets closed — skipping scan cycle.")
        return

    now_str = datetime.now(IST).strftime("%H:%M:%S IST")
    logger.info("=" * 60)
    logger.info("🔍  Starting scan cycle at %s", now_str)
    logger.info("=" * 60)

    for symbol in SYMBOLS:
        # Skip symbols whose market is currently closed
        if not is_symbol_market_open(symbol):
            logger.info("─── %s — market closed, skipping ───", symbol)
            continue

        logger.info("─── Processing %s ───", symbol)

        # ── Step 1: Option Chain ────────────────────────────────────────────
        # Primary: Kite Connect NFO API (NIFTY/BANKNIFTY) — no NSE cookies needed
        # Fallback: NSE HTTP session (if kite unavailable)
        # SILVERM: spot-only via yfinance (no free MCX option chain)
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

        pcr_val    = oc_data.get("pcr")
        maxpain_val = oc_data.get("max_pain")
        ivrank_val  = oc_data.get("iv_rank")
        logger.info(
            "[%s] Spot=%.2f  PCR=%s  MaxPain=%s  IVRank=%s",
            symbol, spot,
            f"{pcr_val:.2f}"    if pcr_val    is not None else "N/A",
            str(maxpain_val)    if maxpain_val is not None else "N/A",
            f"{ivrank_val:.1f}%" if ivrank_val  is not None else "N/A",
        )

        # ── Step 2: Chart Signals ──────────────────────────────────────────
        # For SILVERM: uses Kite MCX historical_data (real ₹ price).
        # For NIFTY/BANKNIFTY: uses yfinance (^NSEI / ^NSEBANK).
        # Falls back to yfinance SI=F proxy if Kite MCX data fails.
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

        # ── Step 3: Strike Selection (NSE chain LTP → Kite fallback) ──────
        try:
            strike = select_strike(
                kite       = kite,
                symbol     = symbol,
                spot       = spot,
                direction  = signal.direction,
                oc_data    = oc_data,          # ← NSE chain LTP used first (free)
                conviction = signal.conviction, # ← scales target: HIGH=1.5×, MEDIUM=1.2×
            )
        except KiteException as exc:
            if _is_token_expiry_error(exc):
                msg = "⚠️  Kite access token expired. Run `python login.py` and restart."
                logger.warning(msg)
                send_error_alert(msg)
                return   # abort cycle
            logger.error("[%s] Kite error in strike selection: %s", symbol, exc)
            continue
        except Exception as exc:
            logger.error("[%s] Strike selection failed: %s", symbol, exc)
            continue

        if strike is None:
            logger.warning("[%s] Strike selection returned None — skipping.", symbol)
            continue

        logger.info(
            "[%s] Strike: %s%s  Expiry: %s  Premium: ₹%.2f  "
            "LotCost: ₹%.0f  Source: %s",
            symbol, strike.otm_strike, strike.option_type,
            strike.expiry_date, strike.premium, strike.lot_cost,
            strike.ltp_source,
        )

        # ── Step 3b: GOLDM budget filter ──────────────────────────────────
        if symbol == "GOLDM" and strike.lot_cost >= GOLDM_MAX_LOT_COST:
            logger.info(
                "[GOLDM] Lot cost ₹%.0f ≥ ₹%.0f budget cap — skipping alert.",
                strike.lot_cost, GOLDM_MAX_LOT_COST,
            )
            continue

        # ── Step 4: Telegram Alert (per-symbol window) ────────────────────
        # NSE (NIFTY/BANKNIFTY): 9:15 AM – 3:00 PM
        # MCX (SILVERM/GOLDM):   Full session 9:00 AM – 11:30 PM
        if is_alert_window_for(symbol):
            try:
                sent = send_full_alert(signal, strike, oc_data)
                logger.info("[%s] %s Telegram alert.", symbol,
                            "✅" if sent else "⚠️  Failed to send")
            except Exception as exc:
                logger.error("[%s] Telegram error: %s", symbol, exc)
        else:
            window_hint = "9:15 AM–3:00 PM" if symbol not in MCX_SYMBOLS else "9:00 AM–11:30 PM"
            logger.info("[%s] ⏰ Outside alert window (%s) — signal logged only.", symbol, window_hint)

        # ── Step 5: Log to CSV ─────────────────────────────────────────────
        try:
            logged = log_signal(signal, strike, oc_data)
            logger.info("[%s] %s CSV log.", symbol,
                        "✅" if logged else "⚠️  Failed to log")
        except Exception as exc:
            logger.error("[%s] Logging error: %s", symbol, exc)

        # Brief pause between symbols to respect NSE rate limits
        time.sleep(2)

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
    return scheduler


# ── Startup banner ─────────────────────────────────────────────────────────────

def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  🟢  NiftySignalBot — OPTIONS SIGNAL SYSTEM")
    print("  ⚠️   SIGNAL ONLY — No auto execution")
    print("=" * 60)
    print(f"  Symbols   : {', '.join(SYMBOLS)}")
    print(f"  NSE hours : 9:15 AM – 3:30 PM  | Alerts: 9:15 AM–3:00 PM")
    print(f"  MCX hours : 9:00 AM – 11:30 PM | Alerts: full session (SILVERM, GOLDM)")
    print(f"  Scan      : Every {SCAN_INTERVAL_MINUTES} min (Mon–Fri)")
    print(f"  Chart data: yfinance (^NSEI, ^NSEBANK) + Kite MCX historical")
    print(f"  OC + LTP  : Kite Connect NFO API (NIFTY/BANKNIFTY) | MCX: spot-only")
    print(f"  Budget    : NIFTY/BANKNIFTY ₹10k–₹20k/lot | SILVERM ~₹5k–₹15k | GOLDM <₹50k/lot")
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