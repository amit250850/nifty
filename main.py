"""
main.py — NiftySignalBot Entry Point

Signal-only options alert system for NIFTY and BANKNIFTY.
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
  • APScheduler runs scan_and_signal() every 5 minutes, 9:15 AM – 3:30 PM IST.
  • Each cycle:
      1. Scan option chain via Kite Connect NFO API (NIFTY/BANKNIFTY).
      2. Record OI snapshot in SQLite.
      3. Compute chart signals (EMA/RSI/SuperTrend/EMA50) via Kite + yfinance.
      4. If conviction ≥ MEDIUM: select strike, run all guards, send alert.
      5. If all 4 guards pass: auto-execute BUY + OCO GTT (when enabled).
      6. Log to CSV.

Usage:
    python login.py     ← once per trading day (refreshes Kite access token)
    python main.py      ← starts the bot
"""

import dataclasses
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
from modules.strike_selector import (
    select_strike,
    fetch_ltp_from_oc,
    fetch_ltp_from_kite as _fetch_ltp_kite,
    build_nfo_symbol,
    SL_PCT_DEFAULT,
    TARGET_MULT,
)
from modules.telegram_alert  import send_full_alert, send_error_alert
from modules.trade_logger    import log_signal, initialise_log
from modules.position_guard  import has_open_position, check_margin, check_liquidity
from modules.gtt_manager     import (
    find_option_tradingsymbol, all_guards_pass, execute_trade,
    ENABLE_AUTO_EXECUTE,
)
from modules.oi_tracker        import initialise_db as init_oi_db, record_snapshot, get_oi_trend
from modules.trade_monitor     import monitor_active_trades, monitor_unregistered_positions, send_eod_summary
from modules.earnings_calendar import (
    get_results_today, get_results_yesterday, get_results_this_week,
    refresh_cache as refresh_earnings, is_result_season, add_manual_date,
)

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

SYMBOLS                = []   # NSE options scanner DISABLED — 12-month backtest: -₹1.05L (NIFTY 31.4% win, BANKNIFTY 37.9% win)
SCAN_INTERVAL_MINUTES  = 5

# ── NIFTY/BANKNIFTY directional options — SUPPRESSED (loss-making) ─────────────
# 12-month backtest (May 2025 – Apr 2026):
#   NIFTY  HIGH: 51 trades, 31.4% win, -₹59,552 (5 green / 7 red months)
#   BNKFTY HIGH: 58 trades, 37.9% win, -₹45,253 (3 green / 8 red months)
#
# Root cause: 2025 was choppy/range-bound — directional options buying requires
# trending markets to overcome theta decay. Edge only exists in strong trending
# regimes (Oct 2025 – Mar 2026 showed positive months).
#
# Re-enable condition: switch SYMBOLS back to ["NIFTY"] when India VIX > 18
# AND market is in a confirmed directional trend (e.g. 3+ consecutive trend days).
# Profitable substitute: MCX CPR commodities (all 4 green — see below).
SYMBOL_MIN_CONVICTION: dict[str, str] = {
    "NIFTY": "HIGH",   # HIGH only if/when re-enabled
}

# ── Straddle stats (backtest-derived, 2Y gap≥3%, 2H exit) ─────────────────────
# Source: backtest_event_straddle.py — buy ATM CE+PE at prev close, exit winner
# at 2H after gap open, sell loser at open (15% salvage).
#
# IMPORTANT — what this backtest measures vs what it DOESN'T:
#   ✅ Detects all ≥3% gap days as proxy for earnings/event days (2Y history)
#   ✅ Models IV inflation (1.4× realized vol pre-event) + IV crush post-event
#   ⚠️  Real pre-event IV can be 2–3× realized vol → actual cost higher → real
#       profits ~20–40% lower than shown. Edge still exists for T1 stocks.
#   ❌ MACRO EVENTS (FOMC/RBI/Budget) tested separately → 0–25% win, -₹130K
#      These are NOT traded by this bot. Only individual stock earnings are fired.
#
# Tier guide: T1 = strong consistent edge (≥10 events, ≥92% win, high avg P&L)
#             T2 = good edge but smaller avg P&L or fewer events
#             AVOID = backtest shows net loss
STRADDLE_STATS: dict[str, dict] = {
    # ── TIER 1 — strong, consistent ──────────────────────────────────────────
    "TRENT":      {"tier": 1, "win_2h": 100, "avg_pnl_2h": 122266, "outlay": 51500},
    "INDIGO":     {"tier": 1, "win_2h": 100, "avg_pnl_2h":  60722, "outlay": 24000},
    "BEL":        {"tier": 1, "win_2h": 100, "avg_pnl_2h":  69600, "outlay": 28400},
    "M&M":        {"tier": 1, "win_2h": 100, "avg_pnl_2h":  97609, "outlay": 39900},
    "SBIN":       {"tier": 1, "win_2h": 100, "avg_pnl_2h":  71099, "outlay": 27000},
    "AXISBANK":   {"tier": 1, "win_2h": 100, "avg_pnl_2h":  66700, "outlay": 21000},
    "MARUTI":     {"tier": 1, "win_2h": 100, "avg_pnl_2h":  51957, "outlay": 19800},
    "LT":         {"tier": 1, "win_2h": 100, "avg_pnl_2h":  51525, "outlay": 16200},
    "EICHERMOT":  {"tier": 1, "win_2h": 100, "avg_pnl_2h":  46891, "outlay": 17500},
    "TATACONSUM": {"tier": 1, "win_2h": 100, "avg_pnl_2h":  54543, "outlay": 24200},
    "CIPLA":      {"tier": 1, "win_2h": 100, "avg_pnl_2h":  46626, "outlay": 11000},
    "ADANIENT":   {"tier": 1, "win_2h": 100, "avg_pnl_2h":  40501, "outlay": 17000},
    "INDUSINDBK": {"tier": 1, "win_2h": 100, "avg_pnl_2h":  40651, "outlay": 18000},
    "HEROMOTOCO": {"tier": 1, "win_2h": 100, "avg_pnl_2h":  54006, "outlay": 21500},
    "TITAN":      {"tier": 1, "win_2h": 100, "avg_pnl_2h":  49372, "outlay": 18400},
    "BHARTIARTL": {"tier": 1, "win_2h": 100, "avg_pnl_2h": 110471, "outlay": 41300},
    # ── TIER 2 — good edge, smaller P&L or fewer events ──────────────────────
    "NTPC":       {"tier": 2, "win_2h": 100, "avg_pnl_2h":  68551, "outlay": 42300},
    "HINDALCO":   {"tier": 2, "win_2h":  93, "avg_pnl_2h":  38574, "outlay": 24600},
    "SHRIRAMFIN": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  21295, "outlay": 10900},
    "TECHM":      {"tier": 2, "win_2h": 100, "avg_pnl_2h":  37677, "outlay": 16600},
    "SBILIFE":    {"tier": 2, "win_2h": 100, "avg_pnl_2h":  47446, "outlay": 18400},
    "SUNPHARMA":  {"tier": 2, "win_2h": 100, "avg_pnl_2h":  49192, "outlay": 18500},
    "COALINDIA":  {"tier": 2, "win_2h": 100, "avg_pnl_2h":  46055, "outlay": 20200},
    "TATASTEEL":  {"tier": 2, "win_2h": 100, "avg_pnl_2h":  30915, "outlay": 18700},
    "JSWSTEEL":   {"tier": 2, "win_2h": 100, "avg_pnl_2h":  32108, "outlay": 12000},
    "BAJAJ-AUTO": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  28820, "outlay":  9800},
    "HCLTECH":    {"tier": 2, "win_2h": 100, "avg_pnl_2h":  36266, "outlay": 16800},
    "ASIANPAINT": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  21535, "outlay":  9500},
    "APOLLOHOSP": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  38041, "outlay": 13500},
    "ULTRACEMCO": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  43160, "outlay": 17500},
    "DIVISLAB":   {"tier": 2, "win_2h": 100, "avg_pnl_2h":  23807, "outlay":  9000},
    "BPCL":       {"tier": 2, "win_2h": 100, "avg_pnl_2h":  22934, "outlay":  5200},
    "BAJAJFINSV": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  27329, "outlay": 13000},
    "HINDUNILVR": {"tier": 2, "win_2h": 100, "avg_pnl_2h":  25064, "outlay": 10800},
    "GRASIM":     {"tier": 2, "win_2h": 100, "avg_pnl_2h":  21978, "outlay":  8500},
    "HDFCBANK":   {"tier": 2, "win_2h": 100, "avg_pnl_2h":  17541, "outlay":  8000},
    "ICICIBANK":  {"tier": 2, "win_2h": 100, "avg_pnl_2h":  26724, "outlay": 12400},
    "TCS":        {"tier": 2, "win_2h": 100, "avg_pnl_2h":  23154, "outlay":  7000},
    "INFY":       {"tier": 2, "win_2h": 100, "avg_pnl_2h":  20587, "outlay":  6900},
    "RELIANCE":   {"tier": 2, "win_2h": 100, "avg_pnl_2h":  14588, "outlay":  4700},
    "HDFCLIFE":   {"tier": 2, "win_2h":  80, "avg_pnl_2h":  22416, "outlay":  8300},
    "POWERGRID":  {"tier": 2, "win_2h":  83, "avg_pnl_2h":  31224, "outlay": 14700},
    # ── AVOID — backtest net loss ─────────────────────────────────────────────
    "ONGC":       {"tier": 0, "win_2h":  29, "avg_pnl_2h":  -5306, "outlay": 27300},
}
TIER_LABEL = {1: "🥇 T1", 2: "🥈 T2", 3: "🥉 T3", 0: "⛔ AVOID"}

# ── NSE hours (equity index options) ──────────────────────────────────────────
MARKET_OPEN_H,  MARKET_OPEN_M  = 9,  15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
# NSE Telegram alerts: 10:00 AM – 3:00 PM
# Skips 9:15–10:00 AM opening — first 45 min has extreme volatility, fake breakouts,
# and algo-driven spikes that flush retail before the real trend establishes.
# Signals fired in this window have much lower reliability.
ALERT_START_H, ALERT_START_M = 10, 0
ALERT_END_H,   ALERT_END_M   = 15, 0

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
    Return True if the NSE market is currently open.
    Mon–Fri 9:15 AM – 3:30 PM IST.
    """
    now = datetime.now(IST)
    if now.weekday() >= 5:    # Saturday / Sunday
        return False
    open_time  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_time = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_time <= now <= close_time


def is_mcx_session_open() -> bool:
    """MCX trades until 23:30 — keep scheduler alive until 22:00 for EOD reminder."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return now.replace(hour=10, minute=0, second=0, microsecond=0) <= now <= now.replace(hour=22, minute=0, second=0, microsecond=0)


def is_market_open() -> bool:
    """Return True if NSE or MCX session is open (drives scheduler wake-up)."""
    return any(is_symbol_market_open(s) for s in SYMBOLS) or is_mcx_session_open()


def is_alert_window_for(symbol: str) -> bool:
    """
    NSE Telegram alert window: 10:00 AM – 3:00 PM.
    Skips the first 45 minutes of opening-noise, algo spikes,
    and fake breakouts (9:15–10:00 AM).
    """
    now = datetime.now(IST)
    alert_start = now.replace(hour=ALERT_START_H, minute=ALERT_START_M, second=0, microsecond=0)
    alert_end   = now.replace(hour=ALERT_END_H,   minute=ALERT_END_M,   second=0, microsecond=0)
    return alert_start <= now <= alert_end


def _is_token_expiry_error(exc: Exception) -> bool:
    """Detect Kite token-expiry exceptions by message content."""
    return isinstance(exc, KiteException) and (
        "token" in str(exc).lower() or "TokenException" in type(exc).__name__
    )


# ── Daily Briefing & Straddle Intelligence ─────────────────────────────────────

def _send_telegram_raw(message: str) -> None:
    """Send a plain Telegram message (reuses telegram_alert internals)."""
    try:
        from modules.telegram_alert import send_alert
        send_alert(message)
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


def _straddle_line(ev: dict) -> str:
    """Format one event as a compact straddle radar line."""
    from datetime import date as _date
    sym   = ev["symbol"]
    d     = _date.fromisoformat(ev["date"])
    stats = STRADDLE_STATS.get(sym, {})
    tier  = TIER_LABEL.get(stats.get("tier", 3), "")
    win   = stats.get("win_2h", "?")
    pnl   = stats.get("avg_pnl_2h", 0)
    out   = stats.get("outlay", 0)
    today = _date.today()
    days  = (d - today).days
    when  = "⚡ TOMORROW" if days == 1 else (f"📅 {d.strftime('%a %d %b')}" if days > 1 else "📌 TODAY")
    return (f"  {when}: <b>{sym}</b> {tier}  "
            f"win={win}%  avg=₹{pnl:,.0f}  outlay≈₹{out:,.0f}")


def send_morning_briefing() -> None:
    """
    Daily 8:45 AM briefing sent to Telegram.
    Covers: market regime (brief), straddle radar for the week,
    and an urgent action line if results are tomorrow.
    """
    from datetime import date as _date
    import pytz as _pytz

    now_ist = datetime.now(IST)
    today   = _date.today()
    weekday = now_ist.strftime("%A")

    # ── Regime summary (non-fatal if market_regime unavailable) ──────────────
    regime_line = ""
    try:
        from market_regime import get_regime_signal
        regime = get_regime_signal()
        bias   = regime.get("bias", "NEUTRAL").upper()
        conf   = regime.get("confidence", "low").title()
        bias_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(bias, "⚪")
        regime_line = f"\n🧭 <b>Market Regime:</b> {bias_emoji} {bias} ({conf} confidence)"
    except Exception:
        pass   # market_regime is optional — briefing continues without it

    # ── Straddle radar ────────────────────────────────────────────────────────
    week_events = get_results_this_week(days=7)
    today_results = get_results_today()

    radar_lines = []
    for ev in week_events:
        radar_lines.append(_straddle_line(ev))

    if not radar_lines and is_result_season():
        radar_lines = ["  📭 No confirmed dates yet — check NSE calendar"]

    radar_block = "\n".join(radar_lines) if radar_lines else "  📭 No upcoming results in next 7 days"

    # ── Action line — results TODAY after market = buy straddle before close ──
    action_line = ""
    if today_results:
        names = ", ".join(e["symbol"] for e in today_results)
        action_line = (
            f"\n\n⚠️ <b>ACTION REQUIRED TODAY</b>\n"
            f"<b>{names}</b> results TONIGHT after market!\n"
            f"Buy straddle (CE + PE at ATM) before 3:20 PM.\n"
            f"Full alert with strike/premium at 3:00 PM."
        )

    msg = (
        f"🌅 <b>NiftySignalBot — Morning Briefing</b>\n"
        f"{today.strftime('%A, %d %b %Y')}"
        f"{regime_line}\n\n"
        f"📊 <b>Straddle Radar — Next 7 Days:</b>\n"
        f"{radar_block}"
        f"{action_line}\n\n"
        f"🔒 Bot scanning 10:00 AM – 3:00 PM | Auto-exec: "
        f"{'⚡ ON' if ENABLE_AUTO_EXECUTE else '🔒 OFF'}"
    )
    _send_telegram_raw(msg)
    logger.info("Morning briefing sent.")


def _last_thursday_of_month(d):
    """Return the last Thursday of d's month (NSE monthly expiry day)."""
    from datetime import date as _d, timedelta as _td
    import calendar as _cal
    last_day = _d(d.year, d.month, _cal.monthrange(d.year, d.month)[1])
    # walk back from last_day until we hit Thursday (weekday=3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - _td(days=offset)


def _options_dte(result_date) -> int:
    """
    Calendar days from result_date to the monthly expiry Thursday.
    If result_date is after that Thursday (shouldn't happen), rolls to next month.
    """
    from datetime import date as _d, timedelta as _td
    expiry = _last_thursday_of_month(result_date)
    if expiry < result_date:                        # result is after this month's expiry
        nxt = _d(result_date.year + (result_date.month // 12),
                  (result_date.month % 12) + 1, 1)
        expiry = _last_thursday_of_month(nxt)
    return (expiry - result_date).days


def send_straddle_prealert() -> None:
    """
    3:00 PM daily job — send full straddle alert for any stock
    with results TODAY after market, while market is still open to act.

    Correct timing:
      - Results announced tonight after 3:30 PM
      - Buy straddle NOW before 3:20 PM close
      - Gap happens TOMORROW morning
      - Exit TOMORROW: sell loser at open, winner at 11:15 AM

    DTE gate: only fire when monthly expiry is ≤5 calendar days away
    from the results date (i.e. last week before expiry).  Mid-month
    results have options priced with 10-15 days of extra time-value,
    pushing the break-even above TCS/NIFTY50's typical 4-6% move —
    the edge disappears. Alert suppressed with reason logged.
    """
    import math as _math
    import numpy as _np
    import yfinance as _yf
    from datetime import date as _date

    today_events = get_results_today()
    if not today_events:
        logger.info("[straddle] No results today — no pre-alert needed.")
        return

    for ev in today_events:
        sym = ev["symbol"]
        stats = STRADDLE_STATS.get(sym, {})
        tier  = stats.get("tier", 3)
        if tier == 0:
            logger.info("[straddle] Skipping %s — tier 0 (avoid list).", sym)
            continue

        # ── DTE gate: monthly options behave like weeklies only in last 5 days ──
        try:
            result_date = _date.fromisoformat(ev["date"])
            dte = _options_dte(result_date)
            expiry_thu = _last_thursday_of_month(result_date)
            if dte > 5:
                logger.info(
                    "[straddle] %s — results on %s but DTE=%d to expiry %s. "
                    "Mid-month: monthly options have %.0f days extra time-value "
                    "→ BE%% too high for typical move. Suppressing alert.",
                    sym, ev["date"], dte, expiry_thu, dte - 1,
                )
                continue
            logger.info(
                "[straddle] %s — DTE=%d to expiry %s ✅ — options behave like weeklies.",
                sym, dte, expiry_thu,
            )
        except Exception as _dte_err:
            logger.warning("[straddle] DTE check failed for %s: %s — proceeding.", sym, _dte_err)

        # Fetch live spot + vol
        try:
            from modules.earnings_calendar import NIFTY50_TICKERS as _tickers
            ticker_str = _tickers.get(sym)
            df = _yf.download(ticker_str, period="40d", interval="1d",
                              auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError("no data")
            closes  = df["Close"].squeeze()
            spot    = float(closes.iloc[-1])
            log_ret = _np.log(closes / closes.shift(1)).dropna()
            vol     = float(max(0.15, min(1.2, log_ret.tail(30).std() * _math.sqrt(252))))
            pre_vol = vol * 1.40

            # ATM strike
            rnd  = {"MARUTI": 500, "EICHERMOT": 100, "BAJAJ-AUTO": 100}.get(sym, 50)
            K    = round(spot / rnd) * rnd
            T    = 1.0 / 365.0
            r    = 0.065

            def _ncdf(x):
                return 0.5 * _math.erfc(-x / _math.sqrt(2))
            def _bsc(S, K, T, r, s):
                if T <= 0 or s <= 0: return max(0, S - K)
                d1 = (_math.log(S/K) + (r + 0.5*s*s)*T) / (s*_math.sqrt(T))
                d2 = d1 - s*_math.sqrt(T)
                return S*_ncdf(d1) - K*_math.exp(-r*T)*_ncdf(d2)
            def _bsp(S, K, T, r, s):
                if T <= 0 or s <= 0: return max(0, K - S)
                d1 = (_math.log(S/K) + (r + 0.5*s*s)*T) / (s*_math.sqrt(T))
                d2 = d1 - s*_math.sqrt(T)
                return K*_math.exp(-r*T)*_ncdf(-d2) - S*_ncdf(-d1)

            call_c = _bsc(spot, K, T, r, pre_vol)
            put_c  = _bsp(spot, K, T, r, pre_vol)
            total  = call_c + put_c
            be_pct = (total / K) * 100

            from modules.earnings_calendar import NIFTY50_TICKERS as _tickers2
            LOT = {
                "RELIANCE": 250, "TCS": 150, "HDFCBANK": 550, "INFY": 300,
                "ICICIBANK": 700, "SBIN": 1500, "AXISBANK": 1200, "LT": 300,
                "ASIANPAINT": 200, "MARUTI": 100, "TITAN": 375, "WIPRO": 1500,
                "BAJFINANCE": 125, "HCLTECH": 700, "JSWSTEEL": 675, "NTPC": 4500,
                "ONGC": 1925, "INDUSINDBK": 600, "TECHM": 600, "SUNPHARMA": 700,
                "ADANIENT": 250, "BAJAJFINSV": 500, "CIPLA": 650, "DRREDDY": 125,
                "EICHERMOT": 200, "GRASIM": 250, "HEROMOTOCO": 300, "HINDALCO": 1400,
                "M&M": 700, "TATASTEEL": 5500, "TATACONSUM": 1100, "APOLLOHOSP": 125,
                "BPCL": 1800, "COALINDIA": 2800, "DIVISLAB": 100, "INDIGO": 300,
                "SBILIFE": 750, "SHRIRAMFIN": 600, "TRENT": 500, "BAJAJ-AUTO": 75,
                "HDFCLIFE": 1100, "BEL": 4900, "BHARTIARTL": 1851,
            }
            lot    = LOT.get(sym, 500)
            outlay = total * lot
            win_2h = stats.get("win_2h", 70)
            avg_pnl= stats.get("avg_pnl_2h", 0)
            tier_lbl = {1: "🥇 TIER 1", 2: "🥈 TIER 2", 3: "🥉 TIER 3"}.get(tier, "")

            result_date_str = _date.fromisoformat(ev["date"]).strftime("%d %b %Y")

            # ── Weekend / expiry warnings ─────────────────────────────────
            is_friday   = result_date.weekday() == 4
            next_trading = result_date + __import__("datetime").timedelta(days=3 if is_friday else 1)
            is_expiry_next = (_last_thursday_of_month(result_date) == next_trading)

            weekend_warn = (
                f"\n⚠️ <b>FRIDAY RESULTS — Weekend theta risk</b>\n"
                f"  Both legs decay Sat + Sun (3 calendar days held).\n"
                f"  Outlay is higher-stakes than weekday results.\n"
            ) if is_friday else ""

            expiry_warn = (
                f"\n🔴 <b>EXPIRY {next_trading.strftime('%a %d %b')} — Exit by 10:00 AM!</b>\n"
                f"  Winning leg has near-zero time value on expiry day.\n"
                f"  Do NOT wait until 11:15 AM — gamma risk increases fast.\n"
                f"  Exit winner at 10:00 AM sharp.\n"
            ) if is_expiry_next else ""

            exit_day_str = next_trading.strftime("%a %d %b")
            winner_exit  = "10:00 AM" if is_expiry_next else "11:15 AM"

            msg = (
                f"🔔 <b>STRADDLE ALERT — BUY NOW</b>\n"
                f"{'━' * 34}\n"
                f"<b>{sym}</b>  {tier_lbl}\n"
                f"Results: {result_date_str}  ⚡ TONIGHT after market\n"
                f"{'━' * 34}\n\n"
                f"📊 <b>Straddle Setup</b>\n"
                f"  Spot: ₹{spot:,.2f}  |  Strike: {K}\n"
                f"  Call: ₹{call_c:.2f}  +  Put: ₹{put_c:.2f}\n"
                f"  <b>Total outlay: ₹{outlay:,.0f}</b>  (lot={lot})\n"
                f"  Break-even: ±{be_pct:.1f}% move needed\n\n"
                f"📈 <b>Historical Edge (2Y backtest)</b>\n"
                f"  <b>Win rate at 2H exit: {win_2h}%</b>\n"
                f"  Avg P&amp;L per trade: ₹{avg_pnl:,.0f}\n\n"
                f"🎯 <b>Trade Plan</b>\n"
                f"  <b>TODAY before 3:20 PM:</b>\n"
                f"    Buy {K} CE  +  Buy {K} PE\n"
                f"  <b>{exit_day_str} open:</b> Sell losing leg at market\n"
                f"  <b>{exit_day_str} {winner_exit}:</b> Exit winning leg ← HARD EXIT\n"
                f"{weekend_warn}{expiry_warn}\n"
                f"{'━' * 34}\n"
                f"⚠️ <b>BUY BEFORE CLOSE — {win_2h}% win rate | ₹{avg_pnl:,.0f} avg</b>"
            )
            _send_telegram_raw(msg)
            logger.info("[straddle] Pre-alert sent for %s.", sym)

        except Exception as exc:
            logger.warning("[straddle] Pre-alert failed for %s: %s", sym, exc)


def send_exit_reminder() -> None:
    """
    9:15 AM job — if yesterday was a results day for any tracked stock,
    send an exit reminder: sell losing leg at open + exit winner at 11:15 AM.
    Results are announced after market close, so the gap happens the NEXT morning.

    Special cases:
    - Expiry today: exit winner by 10:00 AM (gamma risk spikes near expiry)
    - Weekend held: both legs have 3 days of extra theta — loser may be near zero
    """
    from datetime import date as _date
    yesterday_events = get_results_yesterday()
    if not yesterday_events:
        return

    today = _date.today()
    is_expiry_today = (_last_thursday_of_month(today) == today)

    for ev in yesterday_events:
        sym = ev["symbol"]

        # Was this a Friday results? (held over weekend)
        result_date   = _date.fromisoformat(ev["date"])
        held_weekend  = result_date.weekday() == 4   # Friday

        if is_expiry_today:
            winner_exit  = "10:00 AM"
            expiry_note  = (
                f"\n🔴 <b>EXPIRY TODAY — Exit winner by 10:00 AM sharp!</b>\n"
                f"  Time value is near zero. Gamma risk spikes after 10 AM.\n"
                f"  Do NOT wait until 11:15 AM today.\n"
            )
        else:
            winner_exit  = "11:15 AM"
            expiry_note  = ""

        weekend_note = (
            f"\n⚠️ <b>Weekend held — both legs decayed 3 days.</b>\n"
            f"  Loser premium may be near zero — sell quickly at open.\n"
        ) if held_weekend else ""

        msg = (
            f"⏰ <b>STRADDLE EXIT — {sym}</b>\n"
            f"{'━' * 34}\n"
            f"Results were last night. Gap is open. Act now!\n"
            f"{weekend_note}{expiry_note}\n"
            f"  ✅ <b>RIGHT NOW at open (9:15 AM):</b>\n"
            f"     Sell the LOSING leg at MARKET\n\n"
            f"  ⏰ <b>{winner_exit} — HARD EXIT:</b>\n"
            f"     Exit the winning leg regardless of price\n"
            f"     Do NOT hold to EOD — IV crush kills the premium\n"
            f"{'━' * 34}"
        )
        _send_telegram_raw(msg)
        logger.info("[straddle] Exit reminder sent for %s.", sym)


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

        # ── Step 2c: Weekday MEDIUM suppression ────────────────────────────
        # On Tue/Wed/Thu weekly options have DTE<3 — MEDIUM signals have <50%
        # win rate on these days. Suppress ALL MEDIUM alerts Tue/Wed/Thu for
        # both NIFTY and BANKNIFTY. HIGH conviction signals still go through.
        _weekday = datetime.now(IST).weekday()  # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
        if _weekday in (1, 2, 3) and signal.conviction == "MEDIUM":
            logger.info(
                "[%s] ⏭  MEDIUM suppressed on Tue/Wed/Thu (DTE<3, sub-50%% win). "
                "Waiting for Mon/Fri or HIGH conviction signal.",
                symbol,
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

        # ── Step 5: Margin check ───────────────────────────────────────────
        margin_ok, available_margin = check_margin(kite, symbol, strike.lot_cost)

        # ── Step 6: OI trend ──────────────────────────────────────────────
        oi_trend = None
        try:
            oi_trend = get_oi_trend(symbol)
        except Exception as exc:
            logger.debug("[%s] OI trend error (non-fatal): %s", symbol, exc)

        # ── Step 7: Tradingsymbol lookup + liquidity check ─────────────────
        # Required for both liquidity assessment and GTT placement.
        exchange       = "NFO"
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

        # If liquidity_info is None (tradingsymbol lookup failed), treat as illiquid.
        # Unknown depth = do not auto-execute. Human can still trade the alert manually.
        liquid = liquidity_info.get("liquid", False) if liquidity_info else False

        # ── Step 7b: ATM liquidity fallback ────────────────────────────────
        # If the 1-OTM strike is illiquid (no depth data or wide spread),
        # try the ATM strike instead before giving up on auto-execute.
        #
        # ATM always has tighter spreads and higher volume — far-OTM strikes
        # (e.g. BANKNIFTY 50700 PE with ₹101 premium) often show "depth
        # unavailable" while the ATM (50900 PE) is fully tradeable.
        #
        # We only swap the execution strike — the Telegram alert still shows
        # the original OTM signal so you see exactly what the strategy fired.
        atm_fallback_used = False
        if (not liquid
                and strike.otm_strike != strike.atm_strike):           # not already ATM
            try:
                atm_ts = find_option_tradingsymbol(
                    kite         = kite,
                    symbol       = symbol,
                    expiry_raw   = strike.expiry_raw,
                    strike_price = float(strike.atm_strike),
                    option_type  = strike.option_type,
                    exchange     = exchange,
                )
                if atm_ts:
                    atm_liq = check_liquidity(kite, atm_ts, exchange)
                    if atm_liq.get("liquid", False):
                        # ATM is liquid — fetch its premium and rebuild StrikeInfo
                        atm_prem = fetch_ltp_from_oc(
                            oc_data, strike.atm_strike, strike.option_type, symbol
                        )
                        if not atm_prem:
                            atm_prem = _fetch_ltp_kite(kite, atm_ts, exchange)
                        if atm_prem and atm_prem > 0:
                            tgt_mult = TARGET_MULT.get(signal.conviction, 1.5)
                            strike = dataclasses.replace(
                                strike,
                                otm_strike = strike.atm_strike,
                                nfo_symbol = atm_ts,
                                premium    = round(atm_prem, 2),
                                lot_cost   = round(atm_prem * strike.lot_size, 2),
                                stop_loss  = round(atm_prem * (1 - SL_PCT_DEFAULT), 2),
                                target     = round(atm_prem * tgt_mult, 2),
                            )
                            tradingsymbol    = atm_ts
                            liquidity_info   = atm_liq
                            liquid           = True
                            atm_fallback_used = True
                            logger.info(
                                "[%s] 🔄 ATM fallback: OTM illiquid → using ATM %d%s "
                                "(₹%.2f, %s). Telegram alert still shows OTM signal.",
                                symbol, strike.atm_strike, strike.option_type,
                                atm_prem, atm_liq.get("label", ""),
                            )
            except Exception as exc:
                logger.debug("[%s] ATM fallback error (non-fatal): %s", symbol, exc)

        # ── Step 8: Evaluate auto-execute guards ───────────────────────────
        # PCR ideal check (reuse telegram_alert logic for consistency)
        from modules.telegram_alert import evaluate_pcr
        pcr_ideal = True
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
                "available_margin":   available_margin,
                "liquidity":          liquidity_info,
                "oi_trend":           oi_trend,
                "auto_ok":            auto_ok,
                "auto_reason":        auto_reason,
                "tradingsymbol":      tradingsymbol,
                "auto_exec_armed":    ENABLE_AUTO_EXECUTE,
                "atm_fallback_used":  atm_fallback_used,
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
            logger.info("[%s] ⏰ Outside alert window (10:00 AM–3:00 PM) — signal logged only.", symbol)

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
    Schedule scan_and_signal() every 5 minutes during NSE market hours:
    9:15 AM – 3:30 PM IST (Mon–Fri).
    """
    scheduler = BlockingScheduler(timezone=IST)

    trigger = CronTrigger(
        day_of_week        = "mon-fri",
        hour               = f"{MARKET_OPEN_H}-{MARKET_CLOSE_H}",   # 9 – 15
        minute             = "0/5",
        timezone           = IST,
    )

    scheduler.add_job(
        func               = scan_and_signal,
        trigger            = trigger,
        id                 = "signal_scan",
        name               = "NSE Signal Scanner",
        misfire_grace_time = 60,
        coalesce           = True,
    )

    # ── 8:45 AM — Morning Briefing (regime + straddle radar) ───────────────────
    scheduler.add_job(
        func    = send_morning_briefing,
        trigger = CronTrigger(day_of_week="mon-fri", hour=8, minute=45, timezone=IST),
        id      = "morning_briefing",
        name    = "Morning Briefing",
        misfire_grace_time = 300,
        coalesce           = True,
    )

    # ── 8:55 AM — MCX CPR Pre-Market Alerts (MCX opens 9:00 AM) ────────────────
    # All fire at 08:55 IST, one Telegram per symbol.
    # ACTIVE (backtest-profitable, ₹87K account):
    #   GOLDPETAL  : 10 lots · ₹22K margin · ₹10/₹1 move  | 83.9% win, +₹31K/6mo ✅
    #   GOLDGUINEA :  3 lots · ₹53K margin · ₹3/₹1 move   | 71.9% win, +₹41K/6mo ✅
    #   CRUDEOILM  :  1 lot  · ₹29K margin · ₹10/₹1 move  | 55.2% win, +₹12K/4mo ✅
    #   NICKEL     :  1 lot  · ₹49K margin · ₹250/₹1 move  | 76.9% win, +₹21K/4mo  ✅
    #   SILVERMIC  :  1 lot  · ₹63K margin · ₹1/₹1 move   | 79.5% win, +₹1.22L/9mo ✅
    #   NATURALGAS :  1 lot  · ₹68K margin · ₹1,250/₹1 move| 80.8% win, +₹69K/4mo  ✅
    # SUPPRESSED (over budget — silver tripled since 2023):
    #   SILVERM    :  margin ₹1.25L — exceeds ₹87K capital limit ❌

    def _goldpetal_alert():
        try:
            from goldpetal_morning_alert import send_goldpetal_cpr_alert
            send_goldpetal_cpr_alert()
        except Exception as exc:
            logger.error("GOLDPETAL CPR alert failed: %s", exc)

    def _goldguinea_alert():
        try:
            from goldguinea_morning_alert import send_goldguinea_cpr_alert
            send_goldguinea_cpr_alert()
        except Exception as exc:
            logger.error("GOLDGUINEA CPR alert failed: %s", exc)

    def _silverm_alert():
        try:
            from silverm_morning_alert import send_silverm_cpr_alert
            send_silverm_cpr_alert()
        except Exception as exc:
            logger.error("SILVERM CPR alert failed: %s", exc)

    def _silvermic_alert():
        try:
            from silvermic_morning_alert import send_silvermic_cpr_alert
            send_silvermic_cpr_alert()
        except Exception as exc:
            logger.error("SILVERMIC CPR alert failed: %s", exc)

    def _crudeoilm_alert():
        try:
            from crudeoilm_morning_alert import send_crudeoilm_cpr_alert
            send_crudeoilm_cpr_alert()
        except Exception as exc:
            logger.error("CRUDEOILM CPR alert failed: %s", exc)

    def _nickel_alert():
        try:
            from nickel_morning_alert import send_nickel_cpr_alert
            send_nickel_cpr_alert()
        except Exception as exc:
            logger.error("NICKEL CPR alert failed: %s", exc)

    def _naturalgas_alert():
        try:
            from naturalgas_morning_alert import send_naturalgas_cpr_alert
            send_naturalgas_cpr_alert()
        except Exception as exc:
            logger.error("NATURALGAS CPR alert failed: %s", exc)

    def _usdinr_alert():
        try:
            from usdinr_morning_alert import send_usdinr_cpr_alert
            send_usdinr_cpr_alert()
        except Exception as exc:
            logger.error("USDINR CPR alert failed: %s", exc)

    for _job_id, _job_name, _job_func in [
        ("goldpetal_cpr_alert",   "GOLDPETAL CPR Pre-Market Alert",   _goldpetal_alert),
        ("goldguinea_cpr_alert",  "GOLDGUINEA CPR Pre-Market Alert",  _goldguinea_alert),
        ("silvermic_cpr_alert",   "SILVERMIC CPR Pre-Market Alert",   _silvermic_alert),
        # SILVERM suppressed — margin ₹1.25L exceeds ₹87K capital (silver tripled)
        ("crudeoilm_cpr_alert",   "CRUDEOILM CPR Pre-Market Alert",   _crudeoilm_alert),
        ("nickel_cpr_alert",      "NICKEL CPR Pre-Market Alert",      _nickel_alert),
        ("naturalgas_cpr_alert",  "NATURALGAS CPR Pre-Market Alert",  _naturalgas_alert),
        ("usdinr_cpr_alert",      "USDINR CPR Pre-Market Alert",      _usdinr_alert),
    ]:
        scheduler.add_job(
            func    = _job_func,
            trigger = CronTrigger(day_of_week="mon-fri", hour=8, minute=55, timezone=IST),
            id      = _job_id,
            name    = _job_name,
            misfire_grace_time = 300,
            coalesce           = True,
        )

    # ── 10:02 AM — GOLDPETAL Auto-Execute (checks 09:00–10:00 candle) ───────────
    # Fetches the first 60-min candle close, compares to CPR levels.
    # LONG if close > upper_cpr | SHORT if close < lower_cpr | skip if inside.
    # Places NRML market order when GOLDPETAL_AUTO_EXECUTE = True in autoexec.py.
    def _goldpetal_autoexec():
        try:
            from goldpetal_autoexec import run_goldpetal_autoexec
            run_goldpetal_autoexec()
        except Exception as exc:
            logger.error("GOLDPETAL auto-exec failed: %s", exc)

    scheduler.add_job(
        func    = _goldpetal_autoexec,
        trigger = CronTrigger(day_of_week="mon-fri", hour=10, minute=2, timezone=IST),
        id      = "goldpetal_autoexec",
        name    = "GOLDPETAL CPR Auto-Execute",
        misfire_grace_time = 120,
        coalesce           = True,
    )

    # ── 10:05 AM – 4:55 PM — GOLDPETAL Trade Monitor (every 5 min) ─────────────
    # Polls LTP against SL / target; auto-squares at 16:55 IST if still open.
    # Sends near-SL and near-target warning Telegrams (once per session each).
    def _goldpetal_monitor():
        try:
            from goldpetal_monitor import run_goldpetal_monitor
            run_goldpetal_monitor()
        except Exception as exc:
            logger.error("GOLDPETAL monitor failed: %s", exc)

    scheduler.add_job(
        func    = _goldpetal_monitor,
        trigger = CronTrigger(
            day_of_week = "mon-fri",
            hour        = "10-16",
            minute      = "5,10,15,20,25,30,35,40,45,50,55",
            timezone    = IST,
        ),
        id      = "goldpetal_monitor",
        name    = "GOLDPETAL Trade Monitor",
        misfire_grace_time = 60,
        coalesce           = True,
    )

    # ── 10:05 AM – 5:55 PM — MCX CPR Monitor: GOLDGUINEA + CRUDEOILM + NICKEL ────
    # SILVERM excluded — margin ₹1.25L exceeds ₹50K capital limit.
    # Detects breakout direction from first 1H candle at 10:05.
    # Then monitors LTP → sends near-SL, near-target, target/SL hit, hourly pulse.
    # EOD square-off reminder fires at 17:55 IST (user's 6 PM cutoff).
    def _mcx_cpr_monitor():
        try:
            from mcx_cpr_monitor import run_mcx_cpr_monitor
            run_mcx_cpr_monitor()
        except Exception as exc:
            logger.error("MCX CPR monitor failed: %s", exc)

    scheduler.add_job(
        func    = _mcx_cpr_monitor,
        trigger = CronTrigger(
            day_of_week = "mon-fri",
            hour        = "9-21",
            minute      = "5,10,15,20,25,30,35,40,45,50,55",
            timezone    = IST,
        ),
        id      = "mcx_cpr_monitor",
        name    = "MCX CPR Monitor (GOLDGUINEA + CRUDEOILM + NICKEL)",
        misfire_grace_time = 60,
        coalesce           = True,
    )

    # ── 9:45 AM – 4:55 PM — USDINR CPR Monitor (every 5 min) ────────────────────
    # NSE CDS session 09:00–17:00 IST. EOD square-off reminder at 16:45.
    def _usdinr_cpr_monitor():
        try:
            from usdinr_cpr_monitor import run_usdinr_cpr_monitor
            run_usdinr_cpr_monitor()
        except Exception as exc:
            logger.error("USDINR CPR monitor failed: %s", exc)

    scheduler.add_job(
        func    = _usdinr_cpr_monitor,
        trigger = CronTrigger(
            day_of_week = "mon-fri",
            hour        = "9-16",
            minute      = "5,10,15,20,25,30,35,40,45,50,55",
            timezone    = IST,
        ),
        id      = "usdinr_cpr_monitor",
        name    = "USDINR CPR Monitor",
        misfire_grace_time = 60,
        coalesce           = True,
    )

    # ── 9:15 AM — Straddle exit reminder (if today is a result day) ────────────
    scheduler.add_job(
        func    = send_exit_reminder,
        trigger = CronTrigger(day_of_week="mon-fri", hour=9, minute=15, timezone=IST),
        id      = "exit_reminder",
        name    = "Straddle Exit Reminder",
        misfire_grace_time = 120,
        coalesce           = True,
    )

    # ── 3:00 PM — Pre-event straddle alert (for tomorrow's results) ────────────
    scheduler.add_job(
        func    = send_straddle_prealert,
        trigger = CronTrigger(day_of_week="mon-fri", hour=15, minute=0, timezone=IST),
        id      = "straddle_prealert",
        name    = "Straddle Pre-Alert",
        misfire_grace_time = 120,
        coalesce           = True,
    )

    # ── 3:20 PM — End-of-Day Summary ───────────────────────────────────────────
    def _eod():
        from modules.telegram_alert import send_alert
        send_eod_summary(send_alert, log_csv_path="trade_log.csv")

    scheduler.add_job(
        func    = _eod,
        trigger = CronTrigger(day_of_week="mon-fri", hour=15, minute=20, timezone=IST),
        id      = "eod_summary",
        name    = "NSE End-of-Day Summary",
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
    print(f"  ACTIVE STRATEGIES (backtest-profitable):")
    print(f"    MCX CPR  : GOLDPETAL 10L (83.9% win) · GOLDGUINEA 2L (71.9%) · CRUDEOILM 1L (55.2%) · NICKEL 1L (41.6% / 3.73x)")
    print(f"    CDS CPR  : USDINR 10L")
    print(f"    Straddle : NIFTY50 earnings — DTE<=5 gate (T1/T2 stocks)")
    print(f"  SUPPRESSED STRATEGIES (loss-making / over-budget):")
    print(f"    NSE opts : NIFTY -59K · BANKNIFTY -45K  (12mo, 31-38% win — choppy 2025)")
    print(f"    SILVERM  : margin 1.25L > 50K capital limit (silver tripled)")
    exec_status = "ENABLED" if ENABLE_AUTO_EXECUTE else "DISABLED (alert-only)"
    print(f"  Auto-exec : {exec_status}")
    print("─" * 60)
    print(f"  8:45 AM   : Morning briefing (regime + straddle radar)")
    print(f"  8:55 AM   : GOLDPETAL CPR pre-market alert  [83.9% win / 0 red months]")
    print(f"  8:55 AM   : GOLDGUINEA CPR pre-market alert [71.9% win / 0 red months]")
    print(f"  8:55 AM   : CRUDEOILM CPR pre-market alert  [55.2% win / 4.33x payoff]")
    print(f"  8:55 AM   : NICKEL CPR pre-market alert     [41.6% win / 3.73x payoff]")
    print(f"  8:55 AM   : USDINR CPR pre-market alert (NSE CDS)")
    print(f"  9:15 AM   : Straddle exit reminder (if result day)")
    print(f"  10:02 AM  : GOLDPETAL auto-exec (10:00 candle check)")
    print(f"  10:05 AM+ : GOLDPETAL monitor every 5 min (SL/target/EOD)")
    print(f"  9:45 AM+  : GOLDGUINEA + NICKEL + CRUDEOILM CPR monitor (LTP vs CPR)")
    print(f"  9:45 AM+  : USDINR signal reminders (LTP vs CPR, EOD 16:45)")
    print(f"  3:00 PM   : Pre-event straddle alert (tomorrow's earnings, DTE<=5 only)")
    print(f"  3:20 PM   : EOD summary")
    print("─" * 60)
    print(f"  OI Tracker: data/oi_tracker.db | Log: trade_log.csv")
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

    # ── Refresh earnings calendar (runs fast, cached 24h) ──────────────────
    try:
        refresh_earnings()
        logger.info("Earnings calendar refreshed.")
    except Exception as exc:
        logger.warning("Earnings calendar refresh failed (non-fatal): %s", exc)

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