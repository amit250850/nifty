"""
modules/strike_selector.py — Strike & Expiry Selector

Responsibilities:
  • Round spot price to ATM strike (Nifty→nearest 50, BankNifty→nearest 100,
    GOLDM→nearest 100 per-10g).
  • Recommend 1-strike OTM for capital efficiency.
  • Identify nearest WEEKLY expiry (Thursday for NIFTY, Wednesday for BANKNIFTY)
    or nearest MONTHLY expiry (MCX GOLDM — monthly only, no weekly options).
  • Fetch option premium using this priority:
      1. NSE option chain LTP map (free, no Kite permission needed)
      2. Kite LTP API (fallback if NSE data unavailable)
  • Calculate lot cost, stop-loss (50%), and target (2×).

Weekly expiry schedule (NSE):
  NIFTY     → every Thursday
  BANKNIFTY → every Wednesday (changed from Thursday in 2023)

MCX SILVERM (Silver Mini):
  Monthly expiry only. Lot = 5 kg, price quoted per kg → multiplier = 5.
  lot_cost = premium_per_kg × 5.
  ATM step = ₹1000 (MCX SILVERM strikes in ₹1000/kg increments).
  Budget guide: Silver at ~₹2.4L/kg → ATM premium ~₹2,400–₹3,600/kg
  → lot cost ₹12,000–₹18,000 ✅ fits ₹30k–₹50k budget with 2–3 lots.

Usage (standalone test):
    python -m modules.strike_selector
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# ── Constants ─────────────────────────────────────────────────────────────────

# Lot sizes
#   NIFTY/BANKNIFTY: index lot sizes
#   SILVERM: 5 kg lot, price per kg → premium × 5 = lot cost
LOT_SIZES = {
    "NIFTY":     75,
    "BANKNIFTY": 30,
    "SILVERM":   5,     # 5 kg lot / 1 kg per price unit = ×5 multiplier
    "GOLDM":     10,    # 10 grams per lot, price quoted per 10g
}

ATM_STEP = {
    "NIFTY":     50,
    "BANKNIFTY": 100,
    "SILVERM":   1000,  # MCX SILVERM strikes in ₹1000/kg increments
    "GOLDM":     100,   # MCX GOLDM strikes in ₹100/10g increments
}

# Weekly expiry weekday (Monday=0 … Sunday=6) — NSE index options only
WEEKLY_EXPIRY_DAY = {
    "NIFTY":     3,   # Thursday
    "BANKNIFTY": 2,   # Wednesday (NSE changed from Thu to Wed in 2023)
}

# Symbols that use MCX exchange and have monthly expiry only
MCX_MONTHLY_SYMBOLS = {"SILVERM", "GOLDM"}

CALL         = "CE"
PUT          = "PE"
EXCHANGE_NFO = "NFO"
EXCHANGE_MCX = "MCX"

MAX_RETRIES   = 3
RETRY_DELAY_S = 1

# ── Expiry time-decay safety ───────────────────────────────────────────────────
# Options with very few days to expiry (DTE) have aggressive theta decay and
# almost no recovery time if the trade goes against you.
#
# If the nearest weekly expiry is fewer than MIN_DTE_DAYS away, the bot
# automatically rolls to the FOLLOWING week.
#
# Effect with MIN_DTE_DAYS = 2:
#   NIFTY (Thursday expiry):
#     Mon → DTE 3 ✅ use current  |  Tue → DTE 2 ✅ use current
#     Wed → DTE 1 ❌ roll to +7   |  Thu → DTE 0 ❌ roll to +7
#   BANKNIFTY (Wednesday expiry):
#     Mon → DTE 2 ✅ use current  |  Tue → DTE 1 ❌ roll to +7
#     Wed → DTE 0 ❌ roll to +7   |  Thu → DTE 6 ✅ use current
MIN_DTE_DAYS = 2

# ── Risk / Reward parameters ───────────────────────────────────────────────────
# Smaller targets = higher accuracy. 2× was ambitious; use conviction-scaled targets.
#
#   HIGH   (4/4 indicators): target 1.5× — strong signal, good R:R (1 : 1.5)
#   MEDIUM (3/4 indicators): target 1.2× — more conservative, better win rate
#
# SL: 40% standard. Tightened to 30% when DTE ≤ 2 — near-expiry theta is brutal
# and you need to cut losses faster before the option decays to zero.
SL_PCT_DEFAULT     = 0.40   # DTE ≥ 3
SL_PCT_NEAR_EXPIRY = 0.30   # DTE ≤ 2 (tighter — less time to recover)
TARGET_MULT = {
    "HIGH":   1.5,
    "MEDIUM": 1.2,
}


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class StrikeInfo:
    """All computed strike and trade parameters."""
    symbol:        str
    direction:     str
    spot:          float
    atm_strike:    int
    otm_strike:    int
    option_type:   str
    expiry_date:   str        # human-readable e.g. '27 Mar 2026'
    expiry_raw:    date       # Python date object
    nfo_symbol:    str        # e.g. 'NIFTY26MAR23100CE'
    premium:       float
    lot_size:      int
    lot_cost:      float
    stop_loss:     float
    target:        float
    ltp_source:      str        # 'nse_chain', 'kite_api', or 'mock'
    expiry_type:     str        # 'weekly' or 'monthly'
    days_to_expiry:  int  = 0   # calendar days from today to expiry


# ── ATM / OTM helpers ─────────────────────────────────────────────────────────

def round_to_atm(spot: float, symbol: str) -> int:
    step = ATM_STEP.get(symbol, 50)
    return int(round(spot / step) * step)


def get_otm_strike(atm: int, direction: str, symbol: str) -> int:
    step = ATM_STEP.get(symbol, 50)
    return atm + step if "CALL" in direction else atm - step


# ── Expiry helpers ────────────────────────────────────────────────────────────

def get_nearest_weekly_expiry(symbol: str, kite=None) -> date:
    """
    Return the nearest usable weekly expiry date for the given NSE symbol.

    PRIMARY path (kite provided):
      Reads actual expiry dates from kite.instruments("NFO") and picks the
      nearest one with DTE ≥ MIN_DTE_DAYS.  This correctly handles exchange
      holidays (e.g. April 2 Ram Navami, April 3 Good Friday) where NSE shifts
      the expiry to an earlier date that pure weekday arithmetic would miss.

    FALLBACK (kite=None or API failure):
      Uses weekday arithmetic — Thursday for NIFTY, Wednesday for BANKNIFTY.
      Less reliable around holidays but avoids hard dependency on live Kite data.

    Args:
        symbol: 'NIFTY' or 'BANKNIFTY'.
        kite:   Authenticated KiteConnect instance (optional but strongly preferred).

    Returns:
        Next usable weekly expiry as a Python date.
    """
    today = date.today()

    # ── Primary: read real expiry dates from Kite NFO instruments ─────────────
    if kite is not None:
        try:
            instruments = kite.instruments("NFO")
            expiries = set()
            for inst in instruments:
                if (inst.get("name", "").upper() == symbol.upper()
                        and inst.get("instrument_type") in ("CE", "PE")):
                    exp = inst.get("expiry")
                    if exp is None:
                        continue
                    exp_date = exp if isinstance(exp, date) else exp.date()
                    if exp_date >= today:
                        expiries.add(exp_date)

            if expiries:
                # Sort ascending; pick first date with DTE ≥ MIN_DTE_DAYS
                for exp_date in sorted(expiries):
                    dte = (exp_date - today).days
                    if dte >= MIN_DTE_DAYS:
                        logger.debug(
                            "[strike_selector] %s next expiry from Kite: %s (DTE=%d)",
                            symbol, exp_date, dte,
                        )
                        return exp_date
                # All known expiries are too close — take the furthest available
                furthest = sorted(expiries)[-1]
                logger.warning(
                    "[strike_selector] All %s expiries within MIN_DTE_DAYS; "
                    "using furthest: %s", symbol, furthest,
                )
                return furthest
        except Exception as exc:
            logger.warning(
                "[strike_selector] NFO instruments fetch failed for %s expiry "
                "(%s) — falling back to calendar calc", symbol, exc,
            )

    # ── Fallback: weekday arithmetic ───────────────────────────────────────────
    expiry_day = WEEKLY_EXPIRY_DAY.get(symbol, 3)   # 3=Thursday, 2=Wednesday
    days_ahead = (expiry_day - today.weekday()) % 7
    if days_ahead < MIN_DTE_DAYS:
        days_ahead += 7
    fallback_date = today + timedelta(days=days_ahead)
    logger.debug(
        "[strike_selector] %s expiry (calendar fallback): %s", symbol, fallback_date
    )
    return fallback_date


def get_nearest_mcx_monthly_expiry(kite, symbol: str = "SILVERM") -> Optional[date]:
    """
    Find the nearest MCX monthly expiry for the given symbol from Kite.

    Reads real expiry dates from Kite's MCX instruments list so holidays
    are handled correctly. Falls back to a computed estimate on failure.

    Args:
        kite:   Authenticated KiteConnect instance.
        symbol: MCX instrument name, e.g. 'SILVERM'.
    """
    try:
        instruments = kite.instruments(exchange=EXCHANGE_MCX)
    except Exception as exc:
        logger.warning("[strike_selector] MCX instruments fetch failed: %s — estimating", exc)
        instruments = []

    today    = date.today()
    expiries = set()

    for inst in instruments:
        if (inst.get("name") == symbol
                and inst.get("instrument_type") in ("CE", "PE")
                and inst.get("expiry")):
            exp = inst["expiry"]
            if isinstance(exp, datetime):
                exp = exp.date()
            if exp >= today:
                expiries.add(exp)

    if expiries:
        future = sorted(expiries)
        return future[0]

    # Fallback: estimate next expiry as ~20th of current or next month
    logger.warning("[strike_selector] No MCX %s expiries from Kite — using estimate", symbol)
    today = date.today()
    est   = date(today.year, today.month, 20)
    if est <= today:                    # this month's expiry already passed
        m = today.month + 1
        y = today.year + (1 if m > 12 else 0)
        m = m if m <= 12 else m - 12
        est = date(y, m, 20)
    while est.weekday() >= 5:          # push forward past weekends
        est += timedelta(days=1)
    return est


def get_nearest_monthly_expiry(kite) -> Optional[date]:
    """
    Find the nearest monthly expiry from Kite NFO instruments list.
    Monthly = last Thursday of the month (day >= 24, weekday == 3).
    Falls back to nearest weekly if no monthly found.

    Args:
        kite: Authenticated KiteConnect instance.

    Returns:
        Nearest monthly expiry date, or None on API failure.
    """
    try:
        instruments = kite.instruments(exchange=EXCHANGE_NFO)
    except Exception as exc:
        logger.error("[strike_selector] Failed to fetch instruments: %s", exc)
        return None

    today    = date.today()
    expiries = set()

    for inst in instruments:
        if (inst.get("name") in ("NIFTY", "BANKNIFTY")
                and inst.get("instrument_type") in ("CE", "PE")
                and inst.get("expiry")):
            exp = inst["expiry"]
            if isinstance(exp, datetime):
                exp = exp.date()
            if exp >= today:
                expiries.add(exp)

    if not expiries:
        return None

    # Monthly = last Thursday of month (day >= 24)
    monthly = sorted([e for e in expiries if e.weekday() == 3 and e.day >= 24])
    future  = [e for e in monthly if e >= today]
    return future[0] if future else (min(expiries) if expiries else None)


def format_expiry_for_symbol(expiry: date) -> str:
    """e.g. date(2026,3,27) → '26MAR'"""
    return expiry.strftime("%y%b").upper()


def format_expiry_display(expiry: date) -> str:
    """e.g. '27 Mar 2026'  — cross-platform (%-d fails on Windows)."""
    return f"{expiry.day} {expiry.strftime('%b %Y')}"


def build_nfo_symbol(symbol: str, expiry: date, strike: int,
                     option_type: str) -> str:
    """
    Build Kite trading symbol for NFO (equity index) options.

    Weekly format  (non-month-end): NIFTY2532723100CE  (YYMMDD style)
    Monthly format (month-end Thu): NIFTY26MAR23100CE  (YYMON style)

    Args:
        symbol:      Index name (e.g. 'NIFTY', 'BANKNIFTY').
        expiry:      Expiry date.
        strike:      Strike price.
        option_type: 'CE' or 'PE'.

    Returns:
        Kite NFO symbol string.
    """
    # Determine if this is a monthly expiry (last Thursday, day >= 24)
    is_monthly = (expiry.weekday() == 3 and expiry.day >= 24)

    if is_monthly:
        expiry_str = expiry.strftime("%y%b").upper()    # e.g. 26MAR
    else:
        # Weekly: YY + single-char month + DD
        month_map  = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6',
                      7:'7', 8:'8', 9:'9', 10:'O', 11:'N', 12:'D'}
        yy         = expiry.strftime("%y")
        mm         = month_map[expiry.month]
        dd         = expiry.strftime("%d")
        expiry_str = f"{yy}{mm}{dd}"

    return f"{symbol}{expiry_str}{strike}{option_type}"


def build_mcx_symbol(symbol: str, expiry: date, strike: int,
                     option_type: str) -> str:
    """
    Build Kite MCX trading symbol for commodity options (e.g. GOLDM).

    MCX options always use monthly format: SILVERM26APR238000CE
    Format: {NAME}{YY}{MON3}{STRIKE}{TYPE}

    Args:
        symbol:      Commodity name (e.g. 'GOLDM').
        expiry:      Expiry date.
        strike:      Strike price (₹ per 10g for GOLDM).
        option_type: 'CE' or 'PE'.

    Returns:
        Kite MCX symbol string.
    """
    expiry_str = expiry.strftime("%y%b").upper()    # e.g. 26APR
    return f"{symbol}{expiry_str}{strike}{option_type}"


# ── Premium fetch — NSE option chain (primary, free) ─────────────────────────

def fetch_ltp_from_oc(oc_data: Optional[dict], strike: int,
                      option_type: str, symbol: str) -> Optional[float]:
    """
    Look up option premium from the option chain data already fetched
    by option_chain.py. No additional API call needed.

    Args:
        oc_data:     Result dict from scan_option_chain() with 'strike_ltp_map'.
        strike:      Strike price to look up.
        option_type: 'CE' or 'PE'.
        symbol:      For step-size calculation when doing nearby lookup.

    Returns:
        LTP float if found and > 0, else None.
    """
    if oc_data is None:
        return None

    ltp_map = oc_data.get("strike_ltp_map", {})
    if not ltp_map:
        return None

    entry = ltp_map.get(strike)

    # If exact strike not found, try nearby strikes (within 2 steps)
    if entry is None:
        step = ATM_STEP.get(symbol, 50)
        for delta in [step, -step, 2 * step, -2 * step]:
            entry = ltp_map.get(strike + delta)
            if entry:
                logger.info(
                    "[strike_selector] Exact strike %d not in OC map, "
                    "using nearby %d", strike, strike + delta,
                )
                break

    if entry is None:
        return None

    ltp = entry.get(option_type, 0.0)
    return float(ltp) if ltp > 0 else None


# ── Premium fetch — Kite API (fallback) ───────────────────────────────────────

def fetch_ltp_from_kite(kite, trading_symbol: str,
                        exchange: str = EXCHANGE_NFO) -> Optional[float]:
    """
    Fetch LTP from Kite Connect API. Fallback when NSE/MCX chain data unavailable.
    Requires Kite 'Market Quotes' permission.

    Args:
        kite:           Authenticated KiteConnect instance.
        trading_symbol: Kite symbol string (e.g. 'NIFTY2532723100CE' or 'SILVERM26APR238000CE').
        exchange:       'NFO' for equity index options, 'MCX' for commodity options.
    """
    full_symbol = f"{exchange}:{trading_symbol}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ltp_data = kite.ltp([full_symbol])
            if not ltp_data:
                raise ValueError(f"Empty LTP response for {full_symbol}")
            # Kite sometimes normalises the key — try exact match first,
            # then fall back to the first value in the response dict.
            if full_symbol in ltp_data:
                ltp = ltp_data[full_symbol]["last_price"]
            else:
                ltp = list(ltp_data.values())[0]["last_price"]
                logger.debug("[strike_selector] Kite key mismatch — used first value")
            logger.info("[strike_selector] Kite LTP for %s = %.2f", trading_symbol, ltp)
            return float(ltp)
        except Exception as exc:
            logger.warning(
                "[strike_selector] Kite LTP attempt %d/%d failed for %s: %s",
                attempt, MAX_RETRIES, trading_symbol, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)
    return None


# ── Main function ─────────────────────────────────────────────────────────────

def select_strike(kite, symbol: str, spot: float, direction: str,
                  oc_data: Optional[dict] = None,
                  use_weekly: bool = True,
                  conviction: str = "HIGH") -> Optional[StrikeInfo]:
    """
    Select the optimal option strike and fetch its premium.

    Supports both NSE (NIFTY, BANKNIFTY) and MCX (SILVERM) symbols.

    Premium fetch priority:
      1. NSE option chain LTP map (free, from oc_data) if available
      2. Kite LTP API (fallback — works for both NFO and MCX)

    Target is conviction-scaled (smaller = more achievable):
      HIGH   (4/4) → 1.5× premium   SL = 40%
      MEDIUM (3/4) → 1.2× premium   SL = 40%

    Args:
        kite:       Authenticated KiteConnect instance.
        symbol:     'NIFTY', 'BANKNIFTY', or 'SILVERM'.
        spot:       Current spot price (₹/kg for SILVERM, index value for others).
        direction:  'BUY CALL' or 'BUY PUT'.
        oc_data:    Option chain dict from scan_option_chain() — pass for free LTP.
        use_weekly: True = weekly expiry (NSE only); ignored for MCX symbols.
        conviction: 'HIGH' or 'MEDIUM' — drives the target multiplier.

    Returns:
        StrikeInfo dataclass, or None on failure.
    """
    option_type = CALL if "CALL" in direction else PUT
    lot_size    = LOT_SIZES.get(symbol, 75)

    # Determine exchange and symbol builder
    is_mcx      = symbol in MCX_MONTHLY_SYMBOLS
    exchange    = EXCHANGE_MCX if is_mcx else EXCHANGE_NFO

    # ── 1. ATM strike ────────────────────────────────────────────────────
    atm = round_to_atm(spot, symbol)
    step = ATM_STEP.get(symbol, 50)
    otm_direction = 1 if "CALL" in direction else -1   # +1 for CE, -1 for PE

    logger.info(
        "[strike_selector] %s spot=%.2f  ATM=%d  type=%s  exchange=%s",
        symbol, spot, atm, option_type, exchange,
    )

    # ── 2. Expiry selection ──────────────────────────────────────────────
    if is_mcx:
        expiry      = get_nearest_mcx_monthly_expiry(kite, symbol)
        expiry_type = "monthly"
        if expiry is None:
            logger.error("[strike_selector] Cannot determine MCX expiry for %s", symbol)
            return None
    elif use_weekly:
        expiry      = get_nearest_weekly_expiry(symbol, kite=kite)  # Kite-aware
        expiry_type = "weekly"
    else:
        expiry      = get_nearest_monthly_expiry(kite)
        expiry_type = "monthly"
        if expiry is None:
            logger.warning("[strike_selector] Monthly expiry fetch failed, "
                           "falling back to weekly")
            expiry      = get_nearest_weekly_expiry(symbol, kite=kite)
            expiry_type = "weekly"

    logger.info("[strike_selector] %s  Expiry: %s (%s)",
                exchange, expiry, expiry_type)

    # ── 3. Premium fetch ─────────────────────────────────────────────────
    premium     = None
    ltp_source  = "unknown"
    strike_used = atm + step * otm_direction   # default 1-strike OTM

    # ── 3a. NSE option chain (free, NFO only) ────────────────────────────
    if oc_data is not None and not is_mcx:
        otm = atm + step * otm_direction
        premium = fetch_ltp_from_oc(oc_data, otm, option_type, symbol)
        if premium and premium > 0:
            ltp_source  = "nse_chain"
            strike_used = otm
            trading_sym = build_nfo_symbol(symbol, expiry, otm, option_type)
            logger.info("[strike_selector] ✅ NSE chain LTP: ₹%.2f for %d%s",
                        premium, otm, option_type)
        else:
            premium = fetch_ltp_from_oc(oc_data, atm, option_type, symbol)
            if premium and premium > 0:
                ltp_source  = "nse_chain"
                strike_used = atm
                trading_sym = build_nfo_symbol(symbol, expiry, atm, option_type)
                logger.info("[strike_selector] OTM not in chain, using ATM from NSE: ₹%.2f",
                            premium)

    # ── 3b. Kite LTP API (fallback for NFO; primary for MCX) ────────────
    if not premium or premium <= 0:
        logger.info("[strike_selector] Using Kite LTP API (%s) …", exchange)

        if is_mcx:
            # Try 1-strike OTM first; fall back to ATM if OTM has no liquidity.
            otm     = atm + step * otm_direction
            otm_sym = build_mcx_symbol(symbol, expiry, otm, option_type)
            premium = fetch_ltp_from_kite(kite, otm_sym, exchange=exchange)
            if premium and premium > 0:
                ltp_source  = "kite_api"
                strike_used = otm
                trading_sym = otm_sym
                logger.info("[strike_selector] ✅ %s OTM: %d%s  ₹%.2f  lot=₹%.0f",
                            symbol, otm, option_type, premium, premium * lot_size)
            else:
                atm_sym = build_mcx_symbol(symbol, expiry, atm, option_type)
                premium = fetch_ltp_from_kite(kite, atm_sym, exchange=exchange)
                if premium and premium > 0:
                    ltp_source  = "kite_api"
                    strike_used = atm
                    trading_sym = atm_sym
                    logger.info("[strike_selector] %s OTM no LTP — using ATM: "
                                "%d%s  ₹%.2f  lot=₹%.0f",
                                symbol, atm, option_type, premium, premium * lot_size)

        else:
            # NFO: try 1-strike OTM then ATM
            otm     = atm + step * otm_direction
            otm_sym = build_nfo_symbol(symbol, expiry, otm, option_type)
            premium = fetch_ltp_from_kite(kite, otm_sym, exchange=exchange)
            if premium and premium > 0:
                ltp_source  = "kite_api"
                strike_used = otm
                trading_sym = otm_sym
            else:
                atm_sym = build_nfo_symbol(symbol, expiry, atm, option_type)
                premium = fetch_ltp_from_kite(kite, atm_sym, exchange=exchange)
                if premium and premium > 0:
                    ltp_source  = "kite_api"
                    strike_used = atm
                    trading_sym = atm_sym

    if not premium or premium <= 0:
        logger.error("[strike_selector] Could not fetch premium for %s "
                     "by any method", symbol)
        return None

    # ── 4. DTE-aware costs ────────────────────────────────────────────────
    dte = (expiry - date.today()).days   # calendar days remaining to expiry

    # Near-expiry (DTE ≤ 2): tighten SL to 30% — theta decay is steep and
    # there is little time to recover, so exit sooner if trade moves against you.
    sl_pct      = SL_PCT_NEAR_EXPIRY if dte <= 2 else SL_PCT_DEFAULT
    target_mult = TARGET_MULT.get(conviction, 1.5)
    lot_cost    = round(premium * lot_size,          2)
    # stop_loss = EXIT PRICE at which to cut the trade (same basis as target)
    # e.g. 30% SL on ₹4,890 premium → exit if option falls to ₹3,423, not ₹1,467
    stop_loss   = round(premium * (1.0 - sl_pct),   2)
    target      = round(premium * target_mult,       2)
    sl_pct_disp = int(sl_pct * 100)

    logger.info(
        "[strike_selector] %s%s  ₹%.2f (src=%s, %s)  DTE=%d  "
        "LotCost=₹%.0f  SL=₹%.2f (exit at %d%% loss)  Target=₹%.2f (%.1f×)  [%s]",
        strike_used, option_type, premium, ltp_source, expiry_type, dte,
        lot_cost, stop_loss, sl_pct_disp, target, target_mult, conviction,
    )

    return StrikeInfo(
        symbol          = symbol,
        direction       = direction,
        spot            = round(spot, 2),
        atm_strike      = atm,
        otm_strike      = strike_used,
        option_type     = option_type,
        expiry_date     = format_expiry_display(expiry),
        expiry_raw      = expiry,
        nfo_symbol      = trading_sym,
        premium         = round(premium, 2),
        lot_size        = lot_size,
        lot_cost        = lot_cost,
        stop_loss       = stop_loss,
        target          = target,
        ltp_source      = ltp_source,
        expiry_type     = expiry_type,
        days_to_expiry  = dte,
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

    print(f"\nNext NIFTY weekly expiry     : {get_nearest_weekly_expiry('NIFTY',     kite=kite)}")
    print(f"Next BANKNIFTY weekly expiry : {get_nearest_weekly_expiry('BANKNIFTY', kite=kite)}\n")

    test_cases = [
        ("NIFTY",     23134.0, "BUY PUT"),
        ("BANKNIFTY", 53453.0, "BUY PUT"),
    ]
    for sym, spot, dirn in test_cases:
        print(f"{'='*55}\n  {sym} | {dirn} | spot={spot}")
        info = select_strike(kite, sym, spot, dirn,
                             oc_data=None, use_weekly=True)
        if info:
            print(f"  NFO       : {info.nfo_symbol}  ({info.expiry_type})")
            print(f"  Expiry    : {info.expiry_date}")
            print(f"  Premium   : ₹{info.premium}  (source: {info.ltp_source})")
            print(f"  Lot Cost  : ₹{info.lot_cost:,.0f}")
            print(f"  SL / Tgt  : ₹{info.stop_loss} / ₹{info.target}")
        else:
            print("  ❌  Strike selection failed")