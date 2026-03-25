"""
modules/gtt_manager.py — Auto-Execute + OCO GTT (Stop-Loss & Target)

AUTO-EXECUTE LOGIC
──────────────────
A BUY order is placed automatically ONLY when ALL four guards pass:

  Guard 1 — Conviction  : signal.conviction == "HIGH"
  Guard 2 — PCR         : pcr_ideal == True  (PCR confirms direction)
  Guard 3 — Budget      : available margin ≥ lot cost
  Guard 4 — Liquidity   : bid-ask spread ≤ SPREAD_WARN_PCT (5%)

If any guard fails → manual alert only (no auto-order).
If all guards pass → buy 1 lot at market, then immediately place OCO GTT.

OCO GTT (One Cancels Other)
────────────────────────────
Two exit orders placed simultaneously on the option you just bought:
  • Lower leg — SL  : fires SELL when premium drops to stop_loss price
  • Upper leg — Target: fires SELL when premium rises to target price
Whichever triggers first, Zerodha cancels the other automatically.

⚠️  IMPORTANT — READ BEFORE USE:
  1. Auto-execute is scoped to NFO only (NIFTY, BANKNIFTY).
     MCX (SILVERM, GOLDM) requires manual entry — GTT note is sent instead.
  2. The buy order uses MIS (intraday) product — the position squares off
     automatically at 3:20 PM if not manually closed or GTT-triggered earlier.
  3. Set ENABLE_AUTO_EXECUTE = True in this file to arm auto-execution.
     It defaults to False so you can test alerts first without live orders.
  4. GTT persists for 1 year. Cancel via Kite app if the signal was skipped.
"""

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Feature flag ───────────────────────────────────────────────────────────────
# Set True only when you are ready for live auto-execution.
# While False: alerts fire, GTT note is included, but NO orders are placed.
ENABLE_AUTO_EXECUTE: bool = False

# Supported exchanges for auto-execution
AUTO_EXEC_EXCHANGES = {"NFO"}

# GTT limit price buffers
SL_LIMIT_BUFFER     = 0.80   # SL sell limit = sl_trigger × 0.80
                              # (wide buffer so a gap-down still fills)
TARGET_LIMIT_BUFFER = 1.00   # Target sell limit = target_trigger
                              # (fills immediately when price reaches target)


# ── Tradingsymbol lookup ───────────────────────────────────────────────────────

def find_option_tradingsymbol(
    kite,
    symbol:       str,
    expiry_raw:   date,
    strike_price: float,
    option_type:  str,
    exchange:     str = "NFO",
) -> Optional[str]:
    """
    Look up the exact Kite tradingsymbol for an option from the instruments list.

    Args:
        kite:         Authenticated KiteConnect instance.
        symbol:       Underlying name, e.g. 'NIFTY', 'BANKNIFTY'.
        expiry_raw:   Python date object for the expiry.
        strike_price: Strike price as float, e.g. 23100.0.
        option_type:  'CE' or 'PE'.
        exchange:     'NFO' or 'MCX'.

    Returns:
        Tradingsymbol string like 'NIFTY25MAR23100PE', or None if not found.
    """
    try:
        instruments = kite.instruments(exchange)
    except Exception as exc:
        logger.warning("[gtt_manager] instruments(%s) failed: %s", exchange, exc)
        return None

    for inst in instruments:
        if inst.get("name", "").upper() != symbol.upper():
            continue
        if inst.get("instrument_type", "").upper() != option_type.upper():
            continue
        if abs(float(inst.get("strike", 0)) - float(strike_price)) > 0.5:
            continue

        exp = inst.get("expiry")
        if exp is None:
            continue
        exp_date = exp if isinstance(exp, date) else exp.date()
        if exp_date == expiry_raw:
            return inst.get("tradingsymbol")

    logger.warning(
        "[gtt_manager] No tradingsymbol found: %s %s K=%s %s on %s",
        exchange, symbol, strike_price, option_type, expiry_raw,
    )
    return None


# ── Auto-execute guard check ───────────────────────────────────────────────────

def all_guards_pass(
    conviction:   str,
    pcr_ideal:    bool,
    margin_ok:    bool,
    liquid:       bool,
    exchange:     str,
) -> tuple[bool, str]:
    """
    Evaluate all four auto-execution guards.

    Returns:
        (all_pass: bool, reason: str)
        reason is a human-readable explanation shown in the Telegram alert.
    """
    failures = []

    if exchange not in AUTO_EXEC_EXCHANGES:
        return False, f"Auto-exec not supported for {exchange} (NFO only)"

    if conviction != "HIGH":
        failures.append(f"conviction={conviction} (need HIGH)")
    if not pcr_ideal:
        failures.append("PCR not ideal for direction")
    if not margin_ok:
        failures.append("insufficient margin")
    if not liquid:
        failures.append("low liquidity (wide spread)")

    if failures:
        return False, "Auto-exec skipped: " + " | ".join(failures)

    return True, "All guards passed ✅"


# ── Buy order ─────────────────────────────────────────────────────────────────

def place_buy_order(
    kite,
    tradingsymbol: str,
    exchange:      str,
    quantity:      int,
    symbol:        str = "",
) -> Optional[str]:
    """
    Place a MIS market BUY order for the option.

    Args:
        kite:          Authenticated KiteConnect instance.
        tradingsymbol: Option tradingsymbol, e.g. 'NIFTY25MAR23100PE'.
        exchange:      'NFO' or 'MCX'.
        quantity:      Number of shares (lot_size × num_lots).
        symbol:        Underlying name for logging only.

    Returns:
        Order ID string on success, or None on failure.
    """
    if not ENABLE_AUTO_EXECUTE:
        logger.info(
            "[gtt_manager] ENABLE_AUTO_EXECUTE=False — BUY order suppressed for %s",
            tradingsymbol,
        )
        return None

    try:
        order_id = kite.place_order(
            tradingsymbol   = tradingsymbol,
            exchange        = exchange,
            transaction_type= kite.TRANSACTION_TYPE_BUY,
            quantity        = quantity,
            product         = kite.PRODUCT_MIS,
            order_type      = kite.ORDER_TYPE_MARKET,
            variety         = kite.VARIETY_REGULAR,
        )
        logger.info(
            "[gtt_manager] ✅ BUY order placed: %s qty=%d → order_id=%s",
            tradingsymbol, quantity, order_id,
        )
        return str(order_id)

    except Exception as exc:
        logger.error(
            "[gtt_manager] BUY order failed for %s qty=%d: %s",
            tradingsymbol, quantity, exc,
        )
        return None


# ── OCO GTT (stop-loss + target) ──────────────────────────────────────────────

def place_oco_gtt(
    kite,
    tradingsymbol: str,
    exchange:      str,
    quantity:      int,
    ltp:           float,
    sl_price:      float,
    target_price:  float,
) -> Optional[dict]:
    """
    Place a two-leg OCO GTT for SL + target exit on the option position.

    The GTT is placed on the SELL side — it assumes you hold (or will hold)
    `quantity` units of `tradingsymbol`.

    When the lower trigger (sl_price) is hit → SL SELL order fires.
    When the upper trigger (target_price) is hit → Target SELL order fires.
    The surviving leg is automatically cancelled by Zerodha.

    Args:
        kite:          Authenticated KiteConnect instance.
        tradingsymbol: Option tradingsymbol.
        exchange:      'NFO' or 'MCX'.
        quantity:      Shares to sell (lot_size × num_lots).
        ltp:           Current option LTP at signal time (GTT reference price).
        sl_price:      Stop-loss SELL trigger (₹ premium level).
        target_price:  Target SELL trigger (₹ premium level).

    Returns:
        dict {gtt_id, tradingsymbol, sl_trigger, target_trigger} or None.
    """
    if exchange not in AUTO_EXEC_EXCHANGES:
        logger.info("[gtt_manager] GTT not placed for %s (MCX — set manually).", exchange)
        return None

    if ltp <= 0 or sl_price <= 0 or target_price <= 0:
        logger.warning(
            "[gtt_manager] Invalid GTT prices: ltp=%.1f sl=%.1f target=%.1f",
            ltp, sl_price, target_price,
        )
        return None

    if sl_price >= ltp:
        logger.warning(
            "[gtt_manager] SL %.1f ≥ LTP %.1f — invalid GTT; skipping", sl_price, ltp
        )
        return None

    if target_price <= ltp:
        logger.warning(
            "[gtt_manager] Target %.1f ≤ LTP %.1f — invalid GTT; skipping",
            target_price, ltp,
        )
        return None

    sl_limit     = round(sl_price     * SL_LIMIT_BUFFER,     1)
    target_limit = round(target_price * TARGET_LIMIT_BUFFER, 1)

    try:
        gtt_id = kite.place_gtt(
            trigger_type   = kite.GTT_TYPE_OCO,
            tradingsymbol  = tradingsymbol,
            exchange       = exchange,
            trigger_values = [sl_price, target_price],   # [lower, upper]
            last_price     = ltp,
            orders         = [
                {   # Leg 1 — SL: fires when price drops to sl_price
                    "transaction_type": kite.TRANSACTION_TYPE_SELL,
                    "quantity":         quantity,
                    "product":          kite.PRODUCT_MIS,
                    "order_type":       kite.ORDER_TYPE_LIMIT,
                    "price":            sl_limit,
                },
                {   # Leg 2 — Target: fires when price rises to target_price
                    "transaction_type": kite.TRANSACTION_TYPE_SELL,
                    "quantity":         quantity,
                    "product":          kite.PRODUCT_MIS,
                    "order_type":       kite.ORDER_TYPE_LIMIT,
                    "price":            target_limit,
                },
            ],
        )

        result = {
            "gtt_id":         gtt_id,
            "tradingsymbol":  tradingsymbol,
            "sl_trigger":     sl_price,
            "target_trigger": target_price,
            "sl_limit":       sl_limit,
            "target_limit":   target_limit,
        }
        logger.info(
            "[gtt_manager] ✅ GTT OCO placed: ID=%s  %s  SL=%.1f  Target=%.1f",
            gtt_id, tradingsymbol, sl_price, target_price,
        )
        return result

    except Exception as exc:
        logger.error(
            "[gtt_manager] GTT placement failed for %s: %s", tradingsymbol, exc
        )
        return None


# ── Execute full trade (buy + GTT) ────────────────────────────────────────────

def execute_trade(
    kite,
    tradingsymbol: str,
    exchange:      str,
    lot_size:      int,
    ltp:           float,
    sl_price:      float,
    target_price:  float,
    symbol:        str = "",
) -> dict:
    """
    Full auto-execute flow: place BUY order then immediately set OCO GTT.

    Returns:
        dict with keys:
          executed    (bool)   — True if BUY order was placed
          order_id    (str)    — Kite order ID, or None
          gtt_id      (int)    — GTT ID, or None
          gtt_placed  (bool)   — True if GTT OCO was set
          message     (str)    — summary for Telegram
    """
    result = {
        "executed":   False,
        "order_id":   None,
        "gtt_id":     None,
        "gtt_placed": False,
        "message":    "",
    }

    if not ENABLE_AUTO_EXECUTE:
        result["message"] = (
            "🔒 Auto-execute is DISABLED (ENABLE_AUTO_EXECUTE=False)\n"
            "   Set it to True in gtt_manager.py when ready for live orders."
        )
        return result

    # 1. Place buy order
    order_id = place_buy_order(kite, tradingsymbol, exchange, lot_size, symbol)
    if order_id is None:
        result["message"] = "❌ BUY order failed — check logs. Set GTT manually."
        return result

    result["executed"] = True
    result["order_id"] = order_id

    # 2. Place OCO GTT immediately after buy
    gtt = place_oco_gtt(
        kite, tradingsymbol, exchange, lot_size, ltp, sl_price, target_price
    )

    if gtt:
        result["gtt_id"]    = gtt["gtt_id"]
        result["gtt_placed"] = True
        result["message"]   = (
            f"✅ AUTO-EXECUTED\n"
            f"   Order ID : {order_id}\n"
            f"   GTT ID   : {gtt['gtt_id']} (OCO — SL + Target set)\n"
            f"   SL Trigger  : ₹{sl_price:,.1f}  (Limit: ₹{gtt['sl_limit']:,.1f})\n"
            f"   TGT Trigger : ₹{target_price:,.1f}\n"
            f"   ⚠️ Cancel GTT #{gtt['gtt_id']} in Kite if you want to exit manually"
        )

        # Register trade for active monitoring (live P&L + alerts + trail stop)
        try:
            from modules.trade_monitor import register_trade
            register_trade(
                symbol        = symbol,
                tradingsymbol = tradingsymbol,
                exchange      = exchange,
                direction     = "",           # caller can pass if needed
                entry_price   = ltp,
                lot_size      = lot_size,
                gtt_id        = gtt["gtt_id"],
                sl_price      = sl_price,
                target_price  = target_price,
                order_id      = order_id,
            )
        except Exception as exc:
            logger.warning("[gtt_manager] Trade registration failed (non-fatal): %s", exc)

    else:
        result["message"] = (
            f"✅ BUY placed (Order: {order_id})\n"
            f"   ❌ GTT failed — SET SL ₹{sl_price:,.1f} AND TARGET ₹{target_price:,.1f} MANUALLY IN KITE NOW"
        )

    return result


# ── Utility: cancel GTT ────────────────────────────────────────────────────────

def cancel_gtt(kite, gtt_id: int) -> bool:
    """Cancel a GTT by ID. Returns True on success."""
    try:
        kite.delete_gtt(gtt_id)
        logger.info("[gtt_manager] GTT %d cancelled.", gtt_id)
        return True
    except Exception as exc:
        logger.error("[gtt_manager] Failed to cancel GTT %d: %s", gtt_id, exc)
        return False
