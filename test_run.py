"""
test_run.py — Full pipeline test (works outside market hours too)

Tests all 5 modules in sequence:
  1. Option Chain  — nsepython (mock if NSE closed)
  2. Chart Signals — yfinance (works 24/7, no Kite needed)
  3. Strike Selector — NSE chain LTP first, Kite fallback
  4. Telegram Alert
  5. Trade Logger (CSV)

Run with:
    python test_run.py
"""

import logging
import os
import sys
from datetime import date

# Force UTF-8 output on Windows so emoji characters don't cause codec errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pytz
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from modules.chart_signals   import SignalResult, compute_signals
from modules.option_chain    import scan_option_chain
from modules.strike_selector import StrikeInfo, select_strike
from modules.telegram_alert  import send_full_alert
from modules.trade_logger    import initialise_log, log_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()

IST = pytz.timezone("Asia/Kolkata")

# ── Kite (only needed for NFO expiry lookup) ───────────────────────────────────
kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

initialise_log()

# ── Mock option chain (used when NSE is closed) ────────────────────────────────
MOCK_OC = {
    "NIFTY": {
        "symbol":             "NIFTY",
        "underlying":         23050.0,
        "pcr":                1.12,
        "pcr_trend":          "Rising — Bullish",
        "max_call_oi_strike": 23200,
        "max_put_oi_strike":  22800,
        "max_pain":           23000,
        "atm_iv":             14.5,
        "iv_rank":            42.5,
        "call_oi_changes":    [],
        "put_oi_changes":     [],
        "strike_ltp_map":     {},   # empty — will fall back to Kite or mock premium
    },
    "BANKNIFTY": {
        "symbol":             "BANKNIFTY",
        "underlying":         49500.0,
        "pcr":                0.95,
        "pcr_trend":          "Neutral",
        "max_call_oi_strike": 50000,
        "max_put_oi_strike":  49000,
        "max_pain":           49500,
        "atm_iv":             16.2,
        "iv_rank":            38.0,
        "call_oi_changes":    [],
        "put_oi_changes":     [],
        "strike_ltp_map":     {},
    },
}


def make_mock_signal(symbol: str, spot: float) -> SignalResult:
    """Fallback mock signal for testing when yfinance returns NO TRADE."""
    return SignalResult(
        symbol             = symbol,
        direction          = "BUY CALL",
        conviction         = "MEDIUM",
        signals_agreed     = 3,
        ema_bullish        = True,
        rsi_value          = 55.0,
        rsi_bullish        = True,
        vwap_bullish       = True,
        supertrend_bullish = False,
        last_close         = spot,
        last_vwap          = spot - 50,
        last_ema9          = spot + 10,
        last_ema21         = spot - 20,
    )


def make_mock_strike(symbol: str, spot: float, direction: str) -> StrikeInfo:
    """Fallback mock strike when Kite LTP and NSE chain both unavailable."""
    step = 50 if symbol == "NIFTY" else 100
    lot  = 75 if symbol == "NIFTY" else 30
    atm  = int(round(spot / step) * step)
    otm  = atm + step if "CALL" in direction else atm - step
    otype = "CE" if "CALL" in direction else "PE"
    return StrikeInfo(
        symbol      = symbol,
        direction   = direction,
        spot        = spot,
        atm_strike  = atm,
        otm_strike  = otm,
        option_type = otype,
        expiry_date = "24 Apr 2026",
        expiry_raw  = date(2026, 4, 24),
        nfo_symbol  = f"{symbol}26APR{otm}{otype}",
        premium     = 185.0,
        lot_size    = lot,
        lot_cost    = round(185.0 * lot, 2),
        stop_loss   = 92.5,
        target      = 370.0,
        ltp_source  = "mock",
        expiry_type = "weekly",
    )


# ── Main test loop ─────────────────────────────────────────────────────────────
for symbol in ["NIFTY", "BANKNIFTY"]:
    print(f"\n{'='*55}")
    print(f"  Testing {symbol}")
    print(f"{'='*55}")

    # ── Module 1: Option Chain (nsepython) ─────────────────────────────────
    print("\n[1/5] Option Chain Scanner (nsepython)")
    oc = scan_option_chain(symbol)
    if oc:
        strikes_with_ltp = len(oc.get("strike_ltp_map", {}))
        print(f"  ✅ LIVE data → Spot={oc['underlying']}  PCR={oc['pcr']}  "
              f"MaxPain={oc['max_pain']}  LTP strikes={strikes_with_ltp}")
    else:
        oc = MOCK_OC[symbol]
        print(f"  ⚠️  NSE closed — using MOCK data  "
              f"(spot={oc['underlying']}  PCR={oc['pcr']})")

    spot = oc["underlying"]

    # ── Module 2: Chart Signals (yfinance — free, works 24/7) ─────────────
    print("\n[2/5] Chart Signal Engine (yfinance)")
    try:
        signal = compute_signals(symbol)   # no kite arg needed
        if signal:
            # If OC was mock, use the real yfinance close as spot
            # so strike selection uses an accurate price
            if not oc.get("strike_ltp_map") or len(oc["strike_ltp_map"]) == 0:
                spot = signal.last_close
                print(f"  ℹ️  OC was mock — using yfinance close as spot: {spot}")

            print(f"  ✅ LIVE signal → {signal.direction} | {signal.conviction} "
                  f"({signal.signals_agreed}/4 agree)")
            print(f"     EMA={'✅' if signal.ema_bullish else '❌'}  "
                  f"RSI={signal.rsi_value:.1f} {'✅' if signal.rsi_bullish else '❌'}  "
                  f"VWAP={'✅' if signal.vwap_bullish else '❌'}  "
                  f"ST={'✅' if signal.supertrend_bullish else '❌'}")
            print(f"     Close={signal.last_close}  VWAP={signal.last_vwap}  "
                  f"EMA9={signal.last_ema9}  EMA21={signal.last_ema21}")
        else:
            print(f"  ⚠️  NO TRADE from live indicators — using MOCK signal for rest of test")
            signal = make_mock_signal(symbol, spot)
    except Exception as exc:
        print(f"  ❌ yfinance error: {exc}")
        signal = make_mock_signal(symbol, spot)
        print(f"  ⚠️  Using MOCK signal")

    # ── Module 3: Strike Selector (NSE LTP → Kite fallback) ───────────────
    print("\n[3/5] Strike Selector")
    try:
        strike = select_strike(
            kite      = kite,
            symbol    = symbol,
            spot      = spot,
            direction = signal.direction,
            oc_data   = oc,        # NSE chain LTP used first (free)
        )
        if strike:
            src = strike.ltp_source
            src_tag = "✅ NSE chain" if src == "nse_chain" else \
                      "✅ Kite API"  if src == "kite_api"  else "⚠️ mock"
            print(f"  {src_tag} → {strike.nfo_symbol}")
            print(f"     Expiry={strike.expiry_date}  Premium=₹{strike.premium}  "
                  f"LotCost=₹{strike.lot_cost:,.0f}")
            print(f"     SL=₹{strike.stop_loss}  Target=₹{strike.target}")
        else:
            print(f"  ❌ Strike selection failed — using MOCK strike")
            strike = make_mock_strike(symbol, spot, signal.direction)
    except Exception as exc:
        print(f"  ❌ Strike error: {exc}")
        strike = make_mock_strike(symbol, spot, signal.direction)
        print(f"  ⚠️  Using MOCK strike: {strike.nfo_symbol}")

    # ── Module 4: Telegram Alert ───────────────────────────────────────────
    print("\n[4/5] Telegram Alert")
    try:
        sent = send_full_alert(signal, strike, oc)
        print(f"  {'✅ Message sent to Telegram!' if sent else '❌ Send failed — check BOT_TOKEN and CHAT_ID in .env'}")
    except Exception as exc:
        print(f"  ❌ Telegram error: {exc}")

    # ── Module 5: Trade Logger ─────────────────────────────────────────────
    print("\n[5/5] Trade Logger (CSV)")
    try:
        logged = log_signal(signal, strike, oc)
        print(f"  {'✅ Row appended to trade_log.csv' if logged else '❌ CSV write failed'}")
    except Exception as exc:
        print(f"  ❌ Logger error: {exc}")

print(f"\n{'='*55}")
print("  Test complete — check Telegram for alert messages.")
print(f"{'='*55}\n")