"""
modules/earnings_calendar.py — NSE Earnings Calendar with Auto-Refresh

Fetches upcoming quarterly result dates for NIFTY50 stocks from NSE's
corporate event API. Falls back to yfinance if NSE is unreachable.
Caches results to earnings_cache.json (refreshed every 24 hours).

Sources (in priority order):
  1. NSE event-calendar API  — most accurate, India-specific
  2. yfinance earnings_dates — partial NSE coverage
  3. Quarterly window heuristic — flags stocks in result months
     (Apr-May = Q4, Jul-Aug = Q1, Oct-Nov = Q2, Jan-Feb = Q3)

Public API:
  get_results_on(target_date)  → list of symbols with results on that date
  get_results_this_week()      → list of (symbol, date) tuples for next 7 days
  get_results_tomorrow()       → list of symbols with results tomorrow
  refresh_cache()              → force-refresh from NSE/yfinance
"""

import json
import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

warnings.filterwarnings("ignore")
logger = logging.getLogger("earnings_calendar")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_FILE     = Path("earnings_cache.json")
CACHE_TTL_HRS  = 24          # refresh cache every 24 hours
NSE_BASE       = "https://www.nseindia.com"
NSE_EVENT_API  = f"{NSE_BASE}/api/event-calendar"
NSE_HEADERS    = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36",
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "Referer":          "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
    "Connection":       "keep-alive",
}
RESULT_KEYWORDS = {
    "quarterly results", "financial results", "unaudited results",
    "audited results", "board meeting", "q1 results", "q2 results",
    "q3 results", "q4 results",
}

# NIFTY50 NSE symbols → yfinance tickers
NIFTY50_TICKERS = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN": "SBIN.NS", "BHARTIARTL": "BHARTIARTL.NS", "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS", "AXISBANK": "AXISBANK.NS", "LT": "LT.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "MARUTI": "MARUTI.NS", "TITAN": "TITAN.NS",
    "WIPRO": "WIPRO.NS", "ULTRACEMCO": "ULTRACEMCO.NS", "BAJFINANCE": "BAJFINANCE.NS",
    "HCLTECH": "HCLTECH.NS", "NESTLEIND": "NESTLEIND.NS", "POWERGRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS", "ONGC": "ONGC.NS", "JSWSTEEL": "JSWSTEEL.NS",
    "INDUSINDBK": "INDUSINDBK.NS", "TECHM": "TECHM.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "ADANIENT": "ADANIENT.NS", "BAJAJFINSV": "BAJAJFINSV.NS", "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS", "EICHERMOT": "EICHERMOT.NS", "GRASIM": "GRASIM.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS", "HINDALCO": "HINDALCO.NS", "M&M": "M&M.NS",
    "TATASTEEL": "TATASTEEL.NS", "TATACONSUM": "TATACONSUM.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS", "BPCL": "BPCL.NS", "COALINDIA": "COALINDIA.NS",
    "DIVISLAB": "DIVISLAB.NS", "INDIGO": "INDIGO.NS", "SBILIFE": "SBILIFE.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS", "TRENT": "TRENT.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HDFCLIFE": "HDFCLIFE.NS", "BEL": "BEL.NS",
}
NIFTY50_SET = set(NIFTY50_TICKERS.keys())

# Months when quarterly results are typically announced
RESULT_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save earnings cache: %s", exc)


def _cache_fresh(cache: dict) -> bool:
    """Return True if cache was refreshed within TTL."""
    ts = cache.get("refreshed_at")
    if not ts:
        return False
    try:
        refreshed = datetime.fromisoformat(ts)
        return (datetime.now() - refreshed).total_seconds() < CACHE_TTL_HRS * 3600
    except Exception:
        return False


# ── NSE Fetcher ───────────────────────────────────────────────────────────────

def _nse_session() -> Optional[requests.Session]:
    """Create a requests session with NSE cookies (required for API auth)."""
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        # Hit homepage first to get cookies (nseappid, nsit etc.)
        resp = session.get(NSE_BASE, timeout=10)
        if not resp.ok:
            return None
        return session
    except Exception as exc:
        logger.debug("NSE session init failed: %s", exc)
        return None


def _fetch_nse_events(days_ahead: int = 30) -> list[dict]:
    """
    Fetch upcoming corporate results from NSE event-calendar API.
    Returns list of dicts: {symbol, date, purpose, source}.
    """
    session = _nse_session()
    if not session:
        return []

    results = []
    try:
        today   = date.today()
        from_dt = today.strftime("%d-%m-%Y")
        to_dt   = (today + timedelta(days=days_ahead)).strftime("%d-%m-%Y")

        resp = session.get(
            NSE_EVENT_API,
            params={"index": "equities"},
            timeout=15,
        )
        if not resp.ok:
            logger.debug("NSE event API returned %d", resp.status_code)
            return []

        events = resp.json()
        if not isinstance(events, list):
            return []

        for ev in events:
            symbol  = ev.get("symbol", "").upper().strip()
            purpose = ev.get("purpose", "").lower().strip()
            date_str= ev.get("date", "") or ev.get("bm_date", "")

            # Filter: only NIFTY50 stocks + result-related events
            if symbol not in NIFTY50_SET:
                continue
            if not any(kw in purpose for kw in RESULT_KEYWORDS):
                continue

            # Parse date (NSE returns DD-MMM-YYYY or YYYY-MM-DD)
            ev_date = None
            for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    ev_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    pass
            if not ev_date:
                continue
            if not (today <= ev_date <= today + timedelta(days=days_ahead)):
                continue

            results.append({
                "symbol":  symbol,
                "date":    ev_date.isoformat(),
                "purpose": purpose,
                "source":  "nse",
            })

        logger.info("NSE API: found %d upcoming results for NIFTY50 stocks.", len(results))
    except Exception as exc:
        logger.debug("NSE event fetch failed: %s", exc)

    return results


# ── yfinance Fetcher ──────────────────────────────────────────────────────────

def _fetch_yfinance_events(symbols: list[str], days_ahead: int = 30) -> list[dict]:
    """
    Pull earnings dates from yfinance for the given symbols.
    Coverage for NSE stocks is partial but useful as a fallback.
    """
    import yfinance as yf

    results = []
    today   = date.today()
    cutoff  = today + timedelta(days=days_ahead)

    for sym in symbols:
        ticker_str = NIFTY50_TICKERS.get(sym)
        if not ticker_str:
            continue
        try:
            t  = yf.Ticker(ticker_str)
            ed = t.get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                continue
            for idx in ed.index:
                try:
                    d = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
                    if today <= d <= cutoff:
                        results.append({
                            "symbol":  sym,
                            "date":    d.isoformat(),
                            "purpose": "quarterly results (yfinance)",
                            "source":  "yfinance",
                        })
                except Exception:
                    pass
        except Exception:
            pass

    logger.info("yfinance: found %d upcoming results.", len(results))
    return results


# ── Refresh ───────────────────────────────────────────────────────────────────

def refresh_cache(days_ahead: int = 30, force: bool = False) -> dict:
    """
    Refresh the earnings calendar from NSE + yfinance.
    Returns the updated cache dict.
    """
    cache = _load_cache()
    if not force and _cache_fresh(cache):
        logger.debug("Earnings cache is fresh — skipping refresh.")
        return cache

    logger.info("Refreshing earnings calendar (NSE + yfinance)...")
    all_events = []

    # 1. Try NSE API
    nse_events = _fetch_nse_events(days_ahead)
    all_events.extend(nse_events)

    # 2. yfinance for any NIFTY50 stock not already covered by NSE
    nse_symbols = {e["symbol"] for e in nse_events}
    remaining   = [s for s in NIFTY50_SET if s not in nse_symbols]
    if remaining:
        yf_events = _fetch_yfinance_events(remaining, days_ahead)
        # Deduplicate: skip if NSE already has that symbol on that date
        nse_keys = {(e["symbol"], e["date"]) for e in nse_events}
        yf_events = [e for e in yf_events if (e["symbol"], e["date"]) not in nse_keys]
        all_events.extend(yf_events)

    # Deduplicate and sort
    seen = set()
    deduped = []
    for ev in sorted(all_events, key=lambda e: (e["date"], e["symbol"])):
        key = (ev["symbol"], ev["date"])
        if key not in seen:
            seen.add(key)
            deduped.append(ev)

    cache = {
        "refreshed_at": datetime.now().isoformat(),
        "events":       deduped,
    }
    _save_cache(cache)
    logger.info("Earnings cache saved: %d events.", len(deduped))
    return cache


# ── Public API ────────────────────────────────────────────────────────────────

def _get_events(days_ahead: int = 30) -> list[dict]:
    """Return cached events, refreshing if stale."""
    cache = _load_cache()
    if not _cache_fresh(cache):
        cache = refresh_cache(days_ahead)
    return cache.get("events", [])


def get_results_on(target_date: date) -> list[dict]:
    """Return all NIFTY50 stocks with results on target_date."""
    target_str = target_date.isoformat()
    return [e for e in _get_events() if e["date"] == target_str]


def get_results_tomorrow() -> list[dict]:
    """Return stocks with results tomorrow (most critical — straddle buy day is TODAY)."""
    tomorrow = date.today() + timedelta(days=1)
    # Skip to next weekday if tomorrow is weekend
    while tomorrow.weekday() >= 5:
        tomorrow += timedelta(days=1)
    return get_results_on(tomorrow)


def get_results_this_week(days: int = 7) -> list[dict]:
    """Return all results in the next N days, sorted by date."""
    today   = date.today()
    cutoff  = today + timedelta(days=days)
    events  = _get_events()
    upcoming = []
    for ev in events:
        try:
            ev_date = date.fromisoformat(ev["date"])
            if today <= ev_date <= cutoff:
                upcoming.append(ev)
        except Exception:
            pass
    return sorted(upcoming, key=lambda e: e["date"])


def get_results_today() -> list[dict]:
    """Return stocks with results announced today (straddle buy day — results after market)."""
    return get_results_on(date.today())


def get_results_yesterday() -> list[dict]:
    """Return stocks with results announced yesterday (exit reminder day — gap already happened)."""
    yesterday = date.today() - timedelta(days=1)
    # Skip back past weekend — Friday results exit on Monday morning
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)
    return get_results_on(yesterday)


def is_result_season() -> bool:
    """True if current month is a quarterly result month."""
    return datetime.now().month in RESULT_MONTHS


def add_manual_date(symbol: str, result_date: date) -> None:
    """Manually inject a result date (persists in cache)."""
    cache = _load_cache()
    events = cache.get("events", [])
    key = (symbol.upper(), result_date.isoformat())
    existing_keys = {(e["symbol"], e["date"]) for e in events}
    if key not in existing_keys:
        events.append({
            "symbol":  symbol.upper(),
            "date":    result_date.isoformat(),
            "purpose": "quarterly results (manual)",
            "source":  "manual",
        })
        events.sort(key=lambda e: (e["date"], e["symbol"]))
        cache["events"] = events
        if "refreshed_at" not in cache:
            cache["refreshed_at"] = datetime.now().isoformat()
        _save_cache(cache)
        logger.info("Manual date added: %s → %s", symbol.upper(), result_date)
