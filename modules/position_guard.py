"""
modules/position_guard.py — Position Deduplication + Margin + Liquidity Checks

Prevents over-trading and unsafe entries by running three checks before every alert:

  1. DUPLICATE CHECK  — is there already an open option position in this direction?
     If yes, suppress the alert entirely (prevents doubling into a losing trade).

  2. MARGIN CHECK     — is available cash ≥ lot cost?
     If not, flag it in the alert so you know before you tap 'Buy'.

  3. LIQUIDITY CHECK  — is the bid-ask spread on the recommended strike acceptable?
     Wide spreads (> 5% of premium) mean you'll leak money on entry and exit.
     Uses the depth data from kite.quote() which is already authorised.

All checks are non-fatal — failures return safe defaults so the alert still fires
rather than silently suppressing. The trader always gets information, not silence.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Bid-ask spread > this % of LTP = flag as illiquid
SPREAD_WARN_PCT  = 5.0    # yellow warning
SPREAD_BLOCK_PCT = 10.0   # red — strongly advise against

# MCX symbols use commodity margin segment
_MCX_SYMBOLS = {"SILVERM", "GOLDM"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DUPLICATE POSITION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_open_option_positions(kite) -> list:
    """
    Fetch all currently open option positions (non-zero net quantity) from Kite.
    Returns list of position dicts, or [] on any failure.
    """
    try:
        all_pos = kite.positions()
        net     = all_pos.get("net", [])
        return [
            p for p in net
            if p.get("quantity", 0) != 0
            and p.get("instrument_type", "") in ("CE", "PE")
        ]
    except Exception as exc:
        logger.warning("[position_guard] positions() failed: %s", exc)
        return []


def has_open_position(kite, symbol: str, direction: str) -> bool:
    """
    Return True if there is an open option position for this symbol + direction.

    Uses a substring match on tradingsymbol so it catches all expiries/strikes
    for the given underlying (e.g. any 'NIFTY...PE' position blocks a BUY PUT).

    Args:
        kite:      Authenticated KiteConnect instance.
        symbol:    e.g. 'NIFTY', 'BANKNIFTY', 'SILVERM'.
        direction: 'BUY CALL' or 'BUY PUT'.

    Returns:
        True  → duplicate open position exists; suppress alert.
        False → clear to proceed.
    """
    opt_type  = "CE" if "CALL" in direction else "PE"
    open_pos  = _fetch_open_option_positions(kite)

    for pos in open_pos:
        ts  = pos.get("tradingsymbol", "")
        itype = pos.get("instrument_type", "")
        qty   = pos.get("quantity", 0)
        if symbol.upper() in ts.upper() and itype == opt_type:
            logger.info(
                "[position_guard] Duplicate suppressed: existing %s qty=%d "
                "blocks new %s %s signal",
                ts, qty, symbol, direction,
            )
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# 2. MARGIN CHECK
# ══════════════════════════════════════════════════════════════════════════════

def get_available_margin(kite, segment: str = "equity") -> Optional[float]:
    """
    Return available cash margin for the given segment.

    Args:
        kite:    Authenticated KiteConnect instance.
        segment: 'equity' (NSE) or 'commodity' (MCX).

    Returns:
        Float balance in ₹, or None if the API call fails.
    """
    try:
        margins  = kite.margins()
        seg_data = margins.get(segment, {})
        avail    = seg_data.get("available", {})

        # Prefer live_balance (intraday cash after MTM); fallback to cash
        for key in ("live_balance", "cash", "opening_balance"):
            val = avail.get(key)
            if val is not None:
                return float(val)

        # Some accounts surface net at top level
        net = seg_data.get("net")
        if net is not None:
            return float(net)

        return None

    except Exception as exc:
        logger.warning("[position_guard] margins() failed (%s): %s", segment, exc)
        return None


def check_margin(kite, symbol: str, lot_cost: float) -> Tuple[bool, Optional[float]]:
    """
    Verify that available margin covers the trade's lot cost.

    Args:
        kite:     Authenticated KiteConnect instance.
        symbol:   Trading symbol — determines equity vs commodity segment.
        lot_cost: Total cost of 1 lot in ₹.

    Returns:
        (sufficient: bool, available_margin: float | None)
        If the API call fails, returns (True, None) — don't block on unknown.
    """
    segment   = "commodity" if symbol in _MCX_SYMBOLS else "equity"
    available = get_available_margin(kite, segment)

    if available is None:
        return True, None   # unknown margin → don't block

    sufficient = available >= lot_cost
    if not sufficient:
        logger.warning(
            "[position_guard] Low margin for %s: available ₹%.0f < lot cost ₹%.0f",
            symbol, available, lot_cost,
        )
    return sufficient, available


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIQUIDITY CHECK (bid-ask spread via kite.quote() depth)
# ══════════════════════════════════════════════════════════════════════════════

def check_liquidity(kite, tradingsymbol: str, exchange: str = "NFO") -> dict:
    """
    Assess the bid-ask spread and volume of the recommended option strike.

    Uses the 5-level market depth returned by kite.quote() — same call already
    used for OI data in option_chain.py, so no extra quota is consumed.

    Args:
        kite:          Authenticated KiteConnect instance.
        tradingsymbol: Option tradingsymbol, e.g. 'NIFTY25MAR23100PE'.
        exchange:      'NFO' or 'MCX'.

    Returns:
        dict with keys:
          ltp         (float) — last traded price of the option
          best_bid    (float) — highest bid price (what buyers will pay)
          best_ask    (float) — lowest ask price  (what sellers want)
          spread      (float) — best_ask − best_bid  in ₹
          spread_pct  (float) — spread as % of LTP
          liquid      (bool)  — True if spread_pct ≤ SPREAD_WARN_PCT
          volume      (int)   — contracts traded today
          label       (str)   — human-readable liquidity verdict
    """
    kite_key = f"{exchange}:{tradingsymbol}"
    default  = {
        "ltp": 0.0, "best_bid": 0.0, "best_ask": 0.0,
        "spread": 0.0, "spread_pct": 0.0,
        "liquid": True, "volume": 0,
        "label": "ℹ️ Liquidity data unavailable",
    }

    try:
        q    = kite.quote([kite_key])
        data = q.get(kite_key, {})
        if not data:
            return default

        ltp    = float(data.get("last_price", 0.0))
        volume = int(data.get("volume", 0))
        depth  = data.get("depth", {})
        bids   = depth.get("buy",  [])
        asks   = depth.get("sell", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0

        if best_bid > 0 and best_ask > 0 and ltp > 0:
            spread     = best_ask - best_bid
            spread_pct = spread / ltp * 100
        else:
            spread = spread_pct = 0.0

        liquid = spread_pct <= SPREAD_WARN_PCT

        if spread_pct == 0.0:
            label = "ℹ️ Depth unavailable — verify liquidity manually"
        elif spread_pct > SPREAD_BLOCK_PCT:
            label = (
                f"⛔ Very illiquid (spread ₹{spread:.1f} = {spread_pct:.1f}% of premium)"
                f" — consider skipping or adjusting strike"
            )
        elif spread_pct > SPREAD_WARN_PCT:
            label = (
                f"⚠️ Thin liquidity (spread ₹{spread:.1f} = {spread_pct:.1f}% of premium)"
                f" — use limit order, not market"
            )
        else:
            label = (
                f"✅ Liquid (spread ₹{spread:.1f} = {spread_pct:.1f}% of premium)"
            )

        logger.info(
            "[position_guard] Liquidity %s: LTP=%.1f bid=%.1f ask=%.1f "
            "spread=%.1f%% vol=%d",
            tradingsymbol, ltp, best_bid, best_ask, spread_pct, volume,
        )

        return {
            "ltp": ltp, "best_bid": best_bid, "best_ask": best_ask,
            "spread": spread, "spread_pct": spread_pct,
            "liquid": liquid, "volume": volume, "label": label,
        }

    except Exception as exc:
        logger.warning(
            "[position_guard] Liquidity check failed for %s: %s", tradingsymbol, exc
        )
        return default
