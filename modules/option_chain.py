"""
modules/option_chain.py — NSE + MCX Option Chain Scanner
==========================================================
PRIMARY path  (when kite is provided):
  • NIFTY / BANKNIFTY — fetches live option chain via Kite Connect (NFO).
      Uses kite.instruments("NFO") for expiry/strike list and kite.quote()
      for live OI + LTP data.  No NSE HTTP session required.
  • SILVERM (MCX) — returns spot via yfinance SI=F proxy (no option chain
      data available on free tier).

FALLBACK path  (kite=None):
  • NIFTY / BANKNIFTY — falls back to direct NSE HTTP API with cookie warmup.
  • Useful for standalone testing without a live Kite session.

Responsibilities:
  • Put-Call Ratio (PCR) from total OI.
  • Max Call OI strike (resistance) and Max Put OI strike (support).
  • OI change direction for top-N strikes each side.
  • Max Pain price.
  • IV approximation (Kite path: skipped — set to 0; NSE path: from chain).
  • strike_ltp_map — lookup dict used by strike_selector to price options.

Usage (standalone test):
    python -m modules.option_chain
"""

import time
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import requests
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_N_STRIKES  = 5
MAX_RETRIES    = 3
RETRY_DELAY_S  = 3

# MCX commodity symbols — no free option chain; spot from yfinance
MCX_SYMBOLS = {"SILVERM", "GOLDM"}

# Index LTP symbols on Kite (used to get spot price for ATM calc)
KITE_INDEX_SYMBOLS = {
    "NIFTY":     "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
}

# How many strikes each side of ATM to include in the option chain scan
# NIFTY step=50  → wings=20 covers ATM ± 1000 pts
# BANKNIFTY step=100 → wings=15 covers ATM ± 1500 pts
NFO_CONFIG = {
    "NIFTY":     {"step": 50,  "wings": 20},
    "BANKNIFTY": {"step": 100, "wings": 15},
}

# ── Instruments cache (module-level) ──────────────────────────────────────────
# kite.instruments("NFO") downloads ~60 MB — cache for 30 min to avoid hammering
_INST_CACHE: dict = {
    "instruments": None,
    "fetched_at":  None,
}
_INST_CACHE_TTL_SECONDS = 30 * 60   # 30 minutes


def _get_nfo_instruments(kite) -> list:
    """
    Return NFO instruments, refreshing the in-memory cache if stale (> 30 min).
    """
    now = datetime.utcnow()
    cached_at = _INST_CACHE.get("fetched_at")
    if (
        _INST_CACHE["instruments"] is not None
        and cached_at is not None
        and (now - cached_at).total_seconds() < _INST_CACHE_TTL_SECONDS
    ):
        return _INST_CACHE["instruments"]

    logger.info("[option_chain] Refreshing NFO instruments cache …")
    instruments = kite.instruments("NFO")
    _INST_CACHE["instruments"] = instruments
    _INST_CACHE["fetched_at"]  = now
    logger.info("[option_chain] NFO instruments cached: %d entries", len(instruments))
    return instruments


# ── MCX commodity spot (SILVERM, GOLDM) ──────────────────────────────────────

# yfinance proxy config per MCX symbol
# SILVERM: COMEX SI=F (USD/troy oz) → ₹/kg  (×1000/31.1035)
# GOLDM:   COMEX GC=F (USD/troy oz) → ₹/10g (×10/31.1035)
_MCX_YF_CONFIG = {
    "SILVERM": {
        "ticker":     "SI=F",
        "unit_label": "₹/kg",
        "formula":    lambda price, fx: round(price * fx * 1000 / 31.1035, 0),
    },
    "GOLDM": {
        "ticker":     "GC=F",
        "unit_label": "₹/10g",
        "formula":    lambda price, fx: round(price * fx * 10 / 31.1035, 0),
    },
}


def _get_mcx_commodity_spot_via_kite(kite, symbol: str) -> Optional[float]:
    """
    Get live MCX commodity spot price in native INR units directly from Kite LTP.

    Finds the nearest active futures contract for the given symbol from
    kite.instruments("MCX"), then calls kite.ltp() — no currency conversion needed.

    Returns:
        Spot price in native MCX units (₹/kg for SILVERM, ₹/10g for GOLDM),
        or None on failure.
    """
    try:
        instruments = kite.instruments("MCX")
        today       = date.today()

        futures = []
        for inst in instruments:
            if inst.get("name") == symbol and inst.get("instrument_type") == "FUT":
                exp = inst.get("expiry")
                if exp is None:
                    continue
                exp_date = exp if isinstance(exp, date) else exp.date()
                if exp_date >= today:
                    futures.append((exp_date, inst["instrument_token"], inst["tradingsymbol"]))

        if not futures:
            logger.warning("[option_chain] No active %s FUT found in MCX instruments", symbol)
            return None

        futures.sort(key=lambda x: x[0])   # nearest expiry first
        _, __, tradingsym = futures[0]

        kite_symbol = f"MCX:{tradingsym}"
        ltp_data    = kite.ltp([kite_symbol])

        val   = ltp_data.get(kite_symbol, {})
        price = val.get("last_price", 0.0) if isinstance(val, dict) else float(val)

        if price <= 0:
            if ltp_data:
                first_val = list(ltp_data.values())[0]
                price = first_val.get("last_price", 0.0) if isinstance(first_val, dict) else 0.0

        if price > 0:
            unit = _MCX_YF_CONFIG.get(symbol, {}).get("unit_label", "")
            logger.info("[option_chain] %s spot via Kite MCX LTP: ₹%.0f%s  (%s)",
                        symbol, price, "/" + unit.lstrip("₹/") if unit else "", tradingsym)
            return float(price)

        logger.warning("[option_chain] Kite MCX LTP returned zero for %s", kite_symbol)
        return None

    except Exception as exc:
        logger.warning("[option_chain] Kite MCX %s LTP failed: %s — will try yfinance",
                       symbol, exc)
        return None


def _get_mcx_commodity_spot_via_yfinance(symbol: str) -> Optional[float]:
    """
    Calculate MCX commodity spot price in native INR units via COMEX yfinance proxy.

    Uses period="5d" so a contract-roll data gap on any single day is bridged.

    Returns:
        Spot price in native MCX units, or None on failure.
    """
    cfg = _MCX_YF_CONFIG.get(symbol)
    if cfg is None:
        logger.error("[option_chain] No yfinance proxy config for MCX symbol %s", symbol)
        return None

    try:
        comm_df = yf.download(cfg["ticker"], period="5d", interval="1h",
                              auto_adjust=True, progress=False)
        fx_df   = yf.download("USDINR=X",   period="5d", interval="1h",
                              auto_adjust=True, progress=False)

        def _last_close(df):
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return float(close.dropna().iloc[-1])

        comm_price = _last_close(comm_df)
        usdinr     = _last_close(fx_df)
        spot       = cfg["formula"](comm_price, usdinr)

        logger.info(
            "[option_chain] %s spot (yfinance proxy): %s=%.3f  USDINR=%.2f  → %s=%.0f",
            symbol, cfg["ticker"], comm_price, usdinr, cfg["unit_label"], spot,
        )
        return spot

    except Exception as exc:
        logger.error("[option_chain] %s yfinance spot failed: %s", symbol, exc)
        return None


def _get_mcx_commodity_spot(kite=None, symbol: str = "SILVERM") -> Optional[float]:
    """
    Get MCX commodity spot price in native INR units.

    Priority:
      1. Kite MCX LTP (real INR price, live MCX session) — when kite provided
      2. yfinance COMEX proxy (USD converted via USDINR) — fallback
    """
    if kite is not None:
        spot = _get_mcx_commodity_spot_via_kite(kite, symbol)
        if spot is not None:
            return spot
        logger.warning("[option_chain] Kite MCX spot failed for %s — falling back to yfinance proxy",
                       symbol)

    return _get_mcx_commodity_spot_via_yfinance(symbol)


# ── Kite-based NSE option chain ───────────────────────────────────────────────

def _get_nearest_expiry(instruments: list, symbol: str) -> Optional[date]:
    """
    Return the nearest upcoming expiry date from the NFO instruments list
    for the given underlying (e.g. 'NIFTY', 'BANKNIFTY').
    """
    today = date.today()
    expiries = set()
    for inst in instruments:
        if (
            inst.get("name") == symbol
            and inst.get("instrument_type") in ("CE", "PE")
        ):
            exp = inst.get("expiry")
            if exp:
                # kiteconnect returns expiry as a date object already
                exp_date = exp if isinstance(exp, date) else exp.date()
                if exp_date >= today:
                    expiries.add(exp_date)

    return min(expiries) if expiries else None


def _get_index_spot(kite, symbol: str) -> Optional[float]:
    """
    Fetch the current spot (last traded price) for a NSE index via Kite LTP.

    Falls back to yfinance if Kite fails.
    """
    kite_sym = KITE_INDEX_SYMBOLS.get(symbol)
    if kite_sym and kite is not None:
        try:
            ltp_data = kite.ltp([kite_sym])
            val = ltp_data.get(kite_sym, {})
            price = val.get("last_price", 0.0) if isinstance(val, dict) else float(val)
            if price > 0:
                logger.info("[option_chain] %s spot via Kite LTP: %.2f", symbol, price)
                return float(price)
        except Exception as exc:
            logger.warning("[option_chain] Kite LTP for %s failed: %s — trying yfinance", symbol, exc)

    # yfinance fallback
    yf_ticker = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(symbol)
    if yf_ticker:
        try:
            df = yf.download(yf_ticker, period="1d", interval="1m",
                             auto_adjust=True, progress=False)
            if not df.empty:
                close = df["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                spot = float(close.dropna().iloc[-1])
                logger.info("[option_chain] %s spot via yfinance: %.2f", symbol, spot)
                return spot
        except Exception as exc:
            logger.error("[option_chain] yfinance spot fallback failed for %s: %s", symbol, exc)

    return None


def _scan_nse_via_kite(symbol: str, kite) -> Optional[dict]:
    """
    Fetch live NIFTY/BANKNIFTY option chain data entirely through Kite Connect.

    Steps:
      1. Load (cached) NFO instruments list.
      2. Determine nearest weekly expiry for the symbol.
      3. Fetch index spot price via kite.ltp().
      4. Filter instruments to ATM ± wings strikes for that expiry.
      5. Call kite.quote() in batches of 500 to get OI + LTP.
      6. Build DataFrame and compute PCR / MaxPain / MaxOI / LTP map.

    Returns:
        Full option-chain result dict (same schema as NSE path), or None on failure.
    """
    config = NFO_CONFIG.get(symbol, {"step": 50, "wings": 20})
    step   = config["step"]
    wings  = config["wings"]

    # ── 1. Get instruments ───────────────────────────────────────────────
    try:
        instruments = _get_nfo_instruments(kite)
    except Exception as exc:
        logger.error("[option_chain] Failed to get NFO instruments: %s", exc)
        return None

    # ── 2. Nearest expiry ────────────────────────────────────────────────
    expiry = _get_nearest_expiry(instruments, symbol)
    if expiry is None:
        logger.error("[option_chain] No upcoming expiry found for %s", symbol)
        return None
    logger.info("[option_chain] %s nearest expiry: %s", symbol, expiry)

    # ── 3. Spot price ────────────────────────────────────────────────────
    spot = _get_index_spot(kite, symbol)
    if spot is None or spot <= 0:
        logger.error("[option_chain] Cannot determine spot for %s", symbol)
        return None

    # ── 4. Filter strikes around ATM ────────────────────────────────────
    atm        = round(spot / step) * step
    min_strike = atm - wings * step
    max_strike = atm + wings * step

    filtered = [
        inst for inst in instruments
        if (
            inst.get("name") == symbol
            and inst.get("instrument_type") in ("CE", "PE")
            and (
                (inst.get("expiry").date()
                 if hasattr(inst.get("expiry"), "date")
                 else inst.get("expiry")) == expiry
            )
            and min_strike <= inst.get("strike", 0) <= max_strike
        )
    ]

    if not filtered:
        logger.error(
            "[option_chain] No %s options found for expiry=%s, "
            "strikes %d–%d", symbol, expiry, min_strike, max_strike
        )
        return None

    logger.info("[option_chain] %s: %d option contracts to quote (expiry=%s, ATM=%d)",
                symbol, len(filtered), expiry, atm)

    # ── 5. Quote in batches ──────────────────────────────────────────────
    kite_symbols = [f"NFO:{inst['tradingsymbol']}" for inst in filtered]
    BATCH_SIZE   = 500
    all_quotes   = {}

    for i in range(0, len(kite_symbols), BATCH_SIZE):
        batch = kite_symbols[i : i + BATCH_SIZE]
        try:
            batch_quotes = kite.quote(batch)
            all_quotes.update(batch_quotes)
        except Exception as exc:
            logger.warning("[option_chain] Kite quote batch %d failed: %s", i // BATCH_SIZE + 1, exc)

    if not all_quotes:
        logger.error("[option_chain] Kite quote returned nothing for %s", symbol)
        return None

    # ── 6. Build per-strike DataFrame ─────────────────────────────────
    # Pivot CE/PE rows into one row per strike
    strike_data: dict[int, dict] = {}

    for inst in filtered:
        ks    = f"NFO:{inst['tradingsymbol']}"
        q     = all_quotes.get(ks, {})
        k     = int(inst.get("strike", 0))
        itype = inst.get("instrument_type")   # "CE" or "PE"

        if k not in strike_data:
            strike_data[k] = {
                "strike":       k,
                "ce_oi":        0, "ce_change_oi": 0, "ce_iv": 0.0, "ce_ltp": 0.0,
                "pe_oi":        0, "pe_change_oi": 0, "pe_iv": 0.0, "pe_ltp": 0.0,
            }

        oi  = int(q.get("oi", 0))
        ltp = float(q.get("last_price", 0.0))

        # Approximate OI change: current OI minus day-low OI
        # (not the same as previous-day change, but gives directional signal)
        oi_day_low = int(q.get("oi_day_low", oi))
        oi_change  = oi - oi_day_low

        if itype == "CE":
            strike_data[k]["ce_oi"]        = oi
            strike_data[k]["ce_change_oi"] = oi_change
            strike_data[k]["ce_ltp"]       = ltp
        else:
            strike_data[k]["pe_oi"]        = oi
            strike_data[k]["pe_change_oi"] = oi_change
            strike_data[k]["pe_ltp"]       = ltp

    df = pd.DataFrame(list(strike_data.values()))
    df = df[df["strike"] > 0].copy()
    df.sort_values("strike", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        logger.warning("[option_chain] Empty DataFrame after building from Kite quotes for %s", symbol)
        return None

    # ── 7. Compute metrics ───────────────────────────────────────────────
    pcr        = calculate_pcr(df)
    max_oi     = get_max_oi_strikes(df)
    oi_changes = get_oi_change_direction(df, spot)
    max_pain   = calculate_max_pain(df)
    ltp_map    = _build_strike_ltp_map(df)

    # Kite quotes don't include IV → skip IV rank (set to None)
    expiry_str = f"{expiry.day} {expiry.strftime('%b %Y')}"

    total_call_oi = int(df["ce_oi"].sum())
    total_put_oi  = int(df["pe_oi"].sum())

    result = {
        "symbol":             symbol,
        "underlying":         round(spot, 2),
        "expiry":             expiry_str,
        "pcr":                pcr,
        "pcr_trend":          pcr_trend_label(pcr),
        "call_oi":            total_call_oi,   # total call OI (for OI trend tracker)
        "put_oi":             total_put_oi,    # total put  OI (for OI trend tracker)
        "max_call_oi_strike": max_oi["max_call_oi_strike"],
        "max_put_oi_strike":  max_oi["max_put_oi_strike"],
        "call_oi_changes":    oi_changes["call_oi_changes"],
        "put_oi_changes":     oi_changes["put_oi_changes"],
        "max_pain":           max_pain,
        "atm_iv":             0.0,
        "iv_rank":            None,           # not available from Kite quotes
        "strike_ltp_map":     ltp_map,
        "data_source":        "kite",
    }

    logger.info(
        "[option_chain] %s (Kite) → spot=%.2f  PCR=%.2f  MaxPain=%d  "
        "Strikes with LTP: %d  Expiry: %s",
        symbol, spot, pcr, max_pain, len(ltp_map), expiry_str,
    )
    return result


# ── NSE HTTP fallback (used when kite=None) ────────────────────────────────────

NSE_BASE    = "https://www.nseindia.com"
NSE_OC_URL  = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"

NSE_HEADERS = {
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Connection":      "keep-alive",
}

NSE_WARMUP_URLS = [
    NSE_BASE + "/",
    NSE_BASE + "/market-data/live-equity-market?symbol=NIFTY",
    NSE_BASE + "/option-chain",
]
NSE_WARMUP_DELAY_S = 1.0


def _make_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    for url in NSE_WARMUP_URLS:
        try:
            session.get(url, timeout=10)
            time.sleep(NSE_WARMUP_DELAY_S)
        except Exception:
            pass
    time.sleep(0.5)
    return session


def _safe_fetch_option_chain(symbol: str) -> Optional[dict]:
    """Fetch raw NSE option chain with retry.  Returns dict or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session  = _make_nse_session()
            url      = NSE_OC_URL.format(symbol=symbol)
            response = session.get(url, timeout=15)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type:
                raise ValueError(
                    f"Non-JSON response (Content-Type: {content_type}). "
                    "NSE may have blocked the request."
                )

            data = response.json()

            section = None
            for key in ("records", "filtered"):
                if key in data and isinstance(data[key], dict):
                    section = data[key]
                    break

            if section is None:
                logger.warning(
                    "[option_chain] Neither 'records' nor 'filtered' found. "
                    "Top-level keys: %s", list(data.keys())
                )
                raise KeyError("records/filtered")

            records     = section.get("data", [])
            expiry_list = section.get("expiryDates", [])
            underlying  = float(section.get("underlyingValue", 0))

            if not records:
                raise ValueError("Empty option chain data returned by NSE")

            return {
                "records":     records,
                "expiry_list": expiry_list,
                "underlying":  underlying,
            }

        except Exception as exc:
            logger.warning(
                "[option_chain] NSE attempt %d/%d failed for %s: %s",
                attempt, MAX_RETRIES, symbol, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S)

    logger.error("[option_chain] All NSE retries exhausted for %s", symbol)
    return None


def _build_dataframe_from_nse(records: list) -> pd.DataFrame:
    """Flatten raw NSE option-chain records into a tidy DataFrame."""
    rows = []
    for rec in records:
        strike  = rec.get("strikePrice", 0)
        ce_data = rec.get("CE", {})
        pe_data = rec.get("PE", {})
        rows.append({
            "strike":       strike,
            "ce_oi":        ce_data.get("openInterest",         0),
            "ce_change_oi": ce_data.get("changeinOpenInterest", 0),
            "ce_iv":        ce_data.get("impliedVolatility",    0),
            "ce_ltp":       ce_data.get("lastPrice",            0),
            "pe_oi":        pe_data.get("openInterest",         0),
            "pe_change_oi": pe_data.get("changeinOpenInterest", 0),
            "pe_iv":        pe_data.get("impliedVolatility",    0),
            "pe_ltp":       pe_data.get("lastPrice",            0),
        })

    df = pd.DataFrame(rows)
    df = df[df["strike"] > 0].copy()
    df.sort_values("strike", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _scan_nse_via_http(symbol: str) -> Optional[dict]:
    """Fallback: scan NIFTY/BANKNIFTY via direct NSE HTTP API."""
    logger.info("[option_chain] Using NSE HTTP fallback for %s …", symbol)

    raw = _safe_fetch_option_chain(symbol)
    if raw is None:
        return None

    underlying = float(raw["underlying"])
    df         = _build_dataframe_from_nse(raw["records"])

    if df.empty:
        logger.warning("[option_chain] Empty NSE option chain for %s", symbol)
        return None

    pcr        = calculate_pcr(df)
    max_oi     = get_max_oi_strikes(df)
    oi_changes = get_oi_change_direction(df, underlying)
    max_pain   = calculate_max_pain(df)
    iv_data    = approximate_iv_rank(df)
    ltp_map    = _build_strike_ltp_map(df)

    result = {
        "symbol":             symbol,
        "underlying":         round(underlying, 2),
        "pcr":                pcr,
        "pcr_trend":          pcr_trend_label(pcr),
        "max_call_oi_strike": max_oi["max_call_oi_strike"],
        "max_put_oi_strike":  max_oi["max_put_oi_strike"],
        "call_oi_changes":    oi_changes["call_oi_changes"],
        "put_oi_changes":     oi_changes["put_oi_changes"],
        "max_pain":           max_pain,
        "atm_iv":             iv_data["atm_iv"],
        "iv_rank":            iv_data["iv_rank"],
        "strike_ltp_map":     ltp_map,
        "data_source":        "nse_http",
    }

    logger.info(
        "[option_chain] %s (NSE HTTP) → spot=%.2f  PCR=%.2f  MaxPain=%d  "
        "IVRank=%.1f%%  Strikes: %d",
        symbol, underlying, pcr, max_pain, iv_data["iv_rank"], len(ltp_map),
    )
    return result


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_strike_ltp_map(df: pd.DataFrame) -> dict:
    """
    Build lookup dict: { strike_int: {'CE': ltp_float, 'PE': ltp_float} }
    """
    ltp_map = {}
    for _, row in df.iterrows():
        strike = int(row["strike"])
        ltp_map[strike] = {
            "CE": float(row["ce_ltp"]),
            "PE": float(row["pe_ltp"]),
        }
    return ltp_map


# ── Calculations ───────────────────────────────────────────────────────────────

def calculate_pcr(df: pd.DataFrame) -> float:
    total_call_oi = df["ce_oi"].sum()
    total_put_oi  = df["pe_oi"].sum()
    if total_call_oi == 0:
        return 0.0
    return round(total_put_oi / total_call_oi, 2)


def get_max_oi_strikes(df: pd.DataFrame) -> dict:
    max_call_strike = int(df.loc[df["ce_oi"].idxmax(), "strike"])
    max_put_strike  = int(df.loc[df["pe_oi"].idxmax(), "strike"])
    return {
        "max_call_oi_strike": max_call_strike,
        "max_put_oi_strike":  max_put_strike,
    }


def get_oi_change_direction(df: pd.DataFrame, underlying: float) -> dict:
    top_calls = df.nlargest(TOP_N_STRIKES, "ce_oi")[
        ["strike", "ce_oi", "ce_change_oi"]
    ].copy()
    top_calls["direction"] = top_calls["ce_change_oi"].apply(
        lambda x: "Buildup" if x > 0 else ("Unwinding" if x < 0 else "Neutral")
    )
    top_puts = df.nlargest(TOP_N_STRIKES, "pe_oi")[
        ["strike", "pe_oi", "pe_change_oi"]
    ].copy()
    top_puts["direction"] = top_puts["pe_change_oi"].apply(
        lambda x: "Buildup" if x > 0 else ("Unwinding" if x < 0 else "Neutral")
    )
    return {
        "call_oi_changes": top_calls.to_dict("records"),
        "put_oi_changes":  top_puts.to_dict("records"),
    }


def calculate_max_pain(df: pd.DataFrame) -> int:
    strikes      = df["strike"].values
    total_losses = []
    for k in strikes:
        ce_loss = np.sum(np.maximum(0, k - strikes) * df["ce_oi"].values)
        pe_loss = np.sum(np.maximum(0, strikes - k) * df["pe_oi"].values)
        total_losses.append(ce_loss + pe_loss)
    return int(strikes[int(np.argmin(total_losses))])


def approximate_iv_rank(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["mid_iv"] = (df["ce_iv"] + df["pe_iv"]) / 2
    df_valid     = df[df["mid_iv"] > 0]
    if df_valid.empty:
        return {"atm_iv": 0.0, "iv_low": 0.0, "iv_high": 0.0, "iv_rank": 0.0}
    mid_idx  = len(df_valid) // 2
    atm_iv   = float(df_valid.iloc[mid_idx]["mid_iv"])
    iv_low   = float(df_valid["mid_iv"].min())
    iv_high  = float(df_valid["mid_iv"].max())
    iv_range = iv_high - iv_low
    iv_rank  = round((atm_iv - iv_low) / iv_range * 100, 1) if iv_range > 0 else 50.0
    return {
        "atm_iv":  round(atm_iv, 2),
        "iv_low":  round(iv_low, 2),
        "iv_high": round(iv_high, 2),
        "iv_rank": iv_rank,
    }


def pcr_trend_label(pcr: float) -> str:
    if pcr >= 1.2:   return "Bullish (High PCR)"
    elif pcr >= 1.0: return "Rising — Bullish"
    elif pcr >= 0.8: return "Neutral"
    elif pcr >= 0.6: return "Falling — Bearish"
    else:            return "Bearish (Low PCR)"


# ── Main public API ────────────────────────────────────────────────────────────

def scan_option_chain(symbol: str, kite=None) -> Optional[dict]:
    """
    Full option-chain scan for a given symbol.

    Args:
        symbol: 'NIFTY', 'BANKNIFTY', or 'SILVERM'.
        kite:   Live KiteConnect instance (optional).
                When provided, NIFTY/BANKNIFTY use Kite API (preferred).
                When None, falls back to NSE HTTP session.

    Returns dict with keys:
        underlying, pcr, pcr_trend, max_call_oi_strike, max_put_oi_strike,
        call_oi_changes, put_oi_changes, max_pain, atm_iv, iv_rank,
        strike_ltp_map, data_source, is_mcx (for SILVERM only)

    Returns None on failure.
    """
    # ── MCX: spot only, no option chain ──────────────────────────────────
    if symbol in MCX_SYMBOLS:
        spot = _get_mcx_commodity_spot(kite=kite, symbol=symbol)
        if spot is None:
            logger.error("[option_chain] Cannot get %s spot price", symbol)
            return None
        src_label = "kite_mcx_ltp" if kite is not None else "yfinance_spot"
        logger.info("[option_chain] %s spot=₹%.0f [%s] (OC not scanned)",
                    symbol, spot, src_label)
        return {
            "symbol":             symbol,
            "underlying":         spot,
            "pcr":                None,
            "pcr_trend":          "N/A",
            "max_call_oi_strike": None,
            "max_put_oi_strike":  None,
            "call_oi_changes":    [],
            "put_oi_changes":     [],
            "max_pain":           None,
            "atm_iv":             0.0,
            "iv_rank":            None,
            "strike_ltp_map":     {},
            "is_mcx":             True,
            "data_source":        src_label,
        }

    # ── NSE index: try Kite first, fall back to NSE HTTP ─────────────────
    logger.info("[option_chain] Scanning %s …", symbol)

    if kite is not None:
        result = _scan_nse_via_kite(symbol, kite)
        if result is not None:
            return result
        logger.warning(
            "[option_chain] Kite OC scan failed for %s — falling back to NSE HTTP", symbol
        )

    return _scan_nse_via_http(symbol)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    from kiteconnect import KiteConnect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv()
    api_key      = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    kite_client = None
    if api_key and access_token:
        kite_client = KiteConnect(api_key=api_key)
        kite_client.set_access_token(access_token)
        try:
            profile = kite_client.profile()
            print(f"✅  Kite session: {profile.get('user_name')}")
        except Exception as e:
            print(f"⚠️  Kite session invalid: {e} — will use NSE HTTP fallback")
            kite_client = None
    else:
        print("⚠️  KITE_API_KEY / KITE_ACCESS_TOKEN not set — using NSE HTTP fallback")

    for sym in ["NIFTY", "BANKNIFTY", "SILVERM"]:
        print(f"\n{'='*60}\n  Option Chain Scan — {sym}\n{'='*60}")
        result = scan_option_chain(sym, kite=kite_client)
        if result:
            ltp_map = result.pop("strike_ltp_map", {})
            print(json.dumps(result, indent=2, default=str))
            sample = dict(list(ltp_map.items())[:5])
            print(f"\n  Sample strike LTPs: {sample}")
        else:
            print(f"  ❌  Failed to fetch data for {sym}")
        time.sleep(2)