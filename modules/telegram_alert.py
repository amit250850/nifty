"""
modules/telegram_alert.py — Telegram Alert Sender

Formats and sends trade signal alerts to a Telegram chat.

PCR Validation logic (shown in every alert):
  BUY CALL signal → PCR ≥ 1.0 = ✅ IDEAL   | PCR < 1.0 = ⚠️ NOT IDEAL
  BUY PUT  signal → PCR ≤ 0.8 = ✅ IDEAL   | PCR > 0.8 = ⚠️ NOT IDEAL

Usage (standalone test):
    python -m modules.telegram_alert
"""

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

try:
    from telegram import Bot
    from telegram.error import TelegramError
except ImportError:
    raise ImportError(
        "python-telegram-bot not installed. Run: pip install python-telegram-bot==20.7"
    )

from modules.chart_signals   import SignalResult
from modules.strike_selector import StrikeInfo

logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ── PCR thresholds ─────────────────────────────────────────────────────────────
PCR_BULLISH_MIN = 1.0   # PCR must be >= this for BUY CALL to be IDEAL
PCR_BEARISH_MAX = 0.8   # PCR must be <= this for BUY PUT  to be IDEAL


# ── PCR validation ─────────────────────────────────────────────────────────────

def evaluate_pcr(pcr: float, direction: str) -> dict:
    """
    Check whether the PCR reading supports the trade direction.

    BUY CALL: PCR >= 1.0 means more puts than calls outstanding —
              traders are hedging downside, which is actually bullish for the index.
    BUY PUT:  PCR <= 0.8 means calls dominate, market is complacent —
              bearish signal when combined with bearish chart.

    Args:
        pcr:       Current Put-Call Ratio.
        direction: 'BUY CALL' or 'BUY PUT'.

    Returns:
        dict with keys: 'ideal' (bool), 'label' (str), 'note' (str)
    """
    is_call = "CALL" in direction

    if is_call:
        ideal = pcr >= PCR_BULLISH_MIN
        if pcr >= 1.2:
            note = "Very bullish — strong put accumulation"
        elif pcr >= 1.0:
            note = "Bullish — puts outnumber calls"
        elif pcr >= 0.8:
            note = "Neutral — wait for PCR to rise above 1.0"
        else:
            note = "Caution — calls dominate, bearish skew"
    else:  # BUY PUT
        ideal = pcr <= PCR_BEARISH_MAX
        if pcr <= 0.6:
            note = "Very bearish — calls heavily dominate"
        elif pcr <= 0.8:
            note = "Bearish — calls outnumber puts"
        elif pcr <= 1.0:
            note = "Neutral — wait for PCR to drop below 0.8"
        else:
            note = "Caution — put accumulation suggests bounce"

    label = "✅ IDEAL" if ideal else "⚠️ NOT IDEAL — skip or reduce size"
    return {"ideal": ideal, "label": label, "note": note}


# ── Message formatter ──────────────────────────────────────────────────────────

def _direction_emoji(direction: str) -> str:
    return "📈" if "CALL" in direction else "📉"


def _signal_tick(is_bullish: bool, direction: str) -> str:
    """✅ if indicator agrees with direction, ❌ if it disagrees."""
    trade_bullish = "CALL" in direction
    return "✅" if is_bullish == trade_bullish else "❌"


def format_alert_message(
    signal:     SignalResult,
    strike:     StrikeInfo,
    oc_data:    dict,
    extra_info: dict = None,
) -> str:
    """
    Compose the full Telegram alert message.

    Args:
        signal:     SignalResult from chart_signals module.
        strike:     StrikeInfo from strike_selector module.
        oc_data:    Option chain dict from option_chain module.
        extra_info: Optional dict with keys:
                      available_margin (float|None)
                      liquidity        (dict|None)
                      oi_trend         (dict|None)
                      auto_ok          (bool)
                      auto_reason      (str)
                      tradingsymbol    (str|None)
                      auto_exec_armed  (bool)

    Returns:
        Formatted plain-text message string.
    """
    extra_info = extra_info or {}
    direction  = signal.direction
    emoji      = _direction_emoji(direction)

    # ── DTE display ───────────────────────────────────────────────────────
    dte = getattr(strike, "days_to_expiry", None)
    dte_label = f" | {dte}d left" if dte is not None else ""

    # Near-expiry: SL was already tightened to 30% in strike_selector
    if dte is not None and dte <= 2:
        sl_pct_label  = "30%"
        theta_warning = (
            "\n⏰ NEAR EXPIRY — Theta decay is aggressive"
            "\n   SL tightened to 30%. Cut quickly if it moves against you."
        )
    else:
        sl_pct_label  = "40%"
        theta_warning = ""

    # ── Margin line ────────────────────────────────────────────────────────
    avail_margin = extra_info.get("available_margin")
    if avail_margin is not None:
        margin_ok  = avail_margin >= strike.lot_cost
        margin_ico = "✅" if margin_ok else "⚠️"
        margin_line = (
            f"\n💰 Margin: ₹{avail_margin:,.0f} available | "
            f"₹{strike.lot_cost:,.0f} needed {margin_ico}"
        )
    else:
        margin_line = ""

    # ── SL display: exit price + max loss amount ───────────────────────────
    # stop_loss is the EXIT PRICE (e.g. ₹3,423 for a ₹4,890 entry at 30% SL)
    # max_loss_unit = the ₹ amount you lose per share if SL is hit
    max_loss_unit = round(strike.premium - strike.stop_loss, 0)
    max_loss_lot  = round(max_loss_unit * strike.lot_size, 0)

    # ── Header ────────────────────────────────────────────────────────────
    header = (
        f"🔔 {signal.symbol} SIGNAL — {signal.conviction} CONVICTION\n"
        f"Direction: {direction} {emoji}\n"
        f"Strike: {strike.otm_strike} {strike.option_type} (1 OTM)\n"
        f"Expiry: {strike.expiry_date} ({strike.expiry_type}{dte_label})\n"
        f"Premium: ₹{strike.premium:,.0f} | Lot Cost: ₹{strike.lot_cost:,.0f}\n"
        f"Stop Loss: ₹{strike.stop_loss:,.0f} exit price"
        f" | Max loss: ₹{max_loss_unit:,.0f}/unit = ₹{max_loss_lot:,.0f}/lot ({sl_pct_label})"
        f"{theta_warning}\n"
        f"Target: ₹{strike.target:,.0f} ({'1.5×' if signal.conviction == 'HIGH' else '1.2×'} premium — {signal.conviction})"
        f"{margin_line}"
    )

    # ── PCR validation section ────────────────────────────────────────────
    pcr       = oc_data.get("pcr", None)   # None for MCX symbols (no option chain)
    is_mcx    = oc_data.get("is_mcx", False)
    pcr_trend = oc_data.get("pcr_trend", "—")
    max_put   = oc_data.get("max_put_oi_strike",  "—")
    max_call  = oc_data.get("max_call_oi_strike", "—")
    max_pain  = oc_data.get("max_pain", "—")

    if is_mcx or pcr is None:
        # MCX Gold — no free option chain data available
        oc_section = (
            f"\n\n📊 Option Chain\n"
            f"PCR: N/A (MCX — no free option chain data)\n"
            f"PCR Signal: ⚪ N/A — chart signals only for MCX Silver\n"
            f"  → No PCR filter applies here; rely on chart conviction\n"
            f"Max Put OI:  N/A\n"
            f"Max Call OI: N/A\n"
            f"Max Pain:    N/A"
        )
        pcr_ideal = True   # no PCR to gate on, fall through to trade note
    else:
        pcr_eval = evaluate_pcr(pcr, direction)

        if "CALL" in direction:
            pcr_rule = f"(Need PCR ≥ {PCR_BULLISH_MIN} for BUY CALL)"
        else:
            pcr_rule = f"(Need PCR ≤ {PCR_BEARISH_MAX} for BUY PUT)"

        oc_section = (
            f"\n\n📊 Option Chain\n"
            f"PCR: {pcr} ({pcr_trend})\n"
            f"PCR Signal: {pcr_eval['label']}\n"
            f"  → {pcr_eval['note']}\n"
            f"  → {pcr_rule}\n"
            f"Max Put OI:  {max_put} (Support)\n"
            f"Max Call OI: {max_call} (Resistance)\n"
            f"Max Pain:    {max_pain}"
        )
        pcr_ideal = pcr_eval["ideal"]

    # ── Chart signals section ──────────────────────────────────────────────
    ema_label = "Bullish crossover" if signal.ema_bullish else "Bearish crossover"
    st_label  = "Green" if signal.supertrend_bullish else "Red"

    if getattr(signal, "is_mcx", False):
        # ── MCX Silver display — commodity-optimised labels ────────────────
        # RSI: show actual threshold (< 45 bearish | > 55 bullish | 45-55 neutral)
        rsi_v = signal.rsi_value
        if rsi_v < 45:
            rsi_label = f"{rsi_v:.0f} (< 45 — bearish)"
            rsi_tick  = _signal_tick(False, direction)   # False = bearish
        elif rsi_v > 55:
            rsi_label = f"{rsi_v:.0f} (> 55 — bullish)"
            rsi_tick  = _signal_tick(True, direction)    # True  = bullish
        else:
            rsi_label = f"{rsi_v:.0f} (45–55 — neutral)"
            rsi_tick  = "—"   # neutral: no tick

        # EMA50 trend — stored in vwap_bullish / last_vwap fields
        ema50_pos = "Above EMA50" if signal.vwap_bullish else "Below EMA50"

        # SuperTrend is the gate — if we got here it PASSED; always ✅
        st_gate_label = "Green (gate ✅)" if signal.supertrend_bullish else "Red (gate ✅)"

        chart_section = (
            f"\n\n📈 Chart Signals (1H) — MCX Silver\n"
            f"EMA 9/21:    {ema_label} {_signal_tick(signal.ema_bullish, direction)}\n"
            f"RSI(14):     {rsi_label} {rsi_tick}\n"
            f"EMA50 Trend: {ema50_pos} {_signal_tick(signal.vwap_bullish, direction)}\n"
            f"SuperTrend:  {st_gate_label}"
        )

        # Conviction note for MCX footer
        non_st_agree = signal.signals_agreed - 1   # subtract SuperTrend
        if signal.conviction == "HIGH":
            trade_note = (
                f"✅ TAKE TRADE — SuperTrend + {non_st_agree}/3 signals confirm\n"
                f"   All indicators aligned — full position size"
            )
        else:  # MEDIUM
            trade_note = (
                f"⚡ PROCEED WITH CAUTION — SuperTrend + {non_st_agree}/3 confirm\n"
                f"   Half position size — momentum signals partially aligned"
            )

    else:
        # ── NSE display — standard labels ─────────────────────────────────
        rsi_pos  = "above 50" if signal.rsi_bullish else "below 50"
        vwap_pos = "Price above" if signal.vwap_bullish else "Price below"

        chart_section = (
            f"\n\n📈 Chart Signals (1H)\n"
            f"EMA:        {ema_label} {_signal_tick(signal.ema_bullish, direction)}\n"
            f"RSI:        {signal.rsi_value:.0f} ({rsi_pos}) {_signal_tick(signal.rsi_bullish, direction)}\n"
            f"VWAP:       {vwap_pos} {_signal_tick(signal.vwap_bullish, direction)}\n"
            f"SuperTrend: {st_label} {_signal_tick(signal.supertrend_bullish, direction)}"
        )

        # Trade note for NSE
        if pcr_ideal:
            trade_note = "✅ TAKE TRADE — Chart + OC both confirm"
        else:
            trade_note = "⚠️ WAIT — Chart signals fired but PCR not ideal\n   Consider skipping or halving position size"

    iv_rank = oc_data.get("iv_rank", None)
    iv_line = f"\n📐 IV Rank: {iv_rank:.1f}%" if iv_rank is not None else ""

    # ── OI Trend section (NSE only) ────────────────────────────────────────
    oi_trend = extra_info.get("oi_trend")
    if oi_trend:
        oi_trend_line = (
            f"\n📉 OI Trend: {oi_trend['trend_emoji']} {oi_trend['trend_label']}"
        )
    else:
        oi_trend_line = ""

    # ── Liquidity section ──────────────────────────────────────────────────
    liquidity = extra_info.get("liquidity")
    if liquidity:
        vol_str   = f" | Vol: {liquidity['volume']:,}" if liquidity.get("volume") else ""
        liq_line  = f"\n💧 Liquidity: {liquidity['label']}{vol_str}"
    else:
        liq_line = ""

    # ── Auto-execute / GTT section ─────────────────────────────────────────
    auto_ok         = extra_info.get("auto_ok",         False)
    auto_reason     = extra_info.get("auto_reason",     "")
    auto_exec_armed = extra_info.get("auto_exec_armed", False)
    tradingsymbol   = extra_info.get("tradingsymbol",   None)

    if auto_exec_armed and auto_ok:
        # All guards passed AND auto-execute is armed
        gtt_line = (
            f"\n\n⚡ AUTO-EXECUTE ARMED\n"
            f"   BUY order + OCO GTT will be placed automatically\n"
            f"   SL: ₹{strike.stop_loss:,.0f}  →  Target: ₹{strike.target:,.0f}\n"
            f"   ⚠️ Check GTT IDs in Kite app after order"
        )
    elif not auto_exec_armed and auto_ok:
        # All guards passed but auto-execute is disabled — show "perfect setup"
        gtt_line = (
            f"\n\n✅ PERFECT SETUP — All 4 guards green\n"
            f"   Budget ✅  Liquid ✅  HIGH conviction ✅  PCR ideal ✅\n"
            f"   Buy manually + set GTT SL ₹{strike.stop_loss:,.0f} / Target ₹{strike.target:,.0f}"
            + (f"\n   Kite instrument: {tradingsymbol}" if tradingsymbol else "")
        )
    elif auto_reason:
        # Some guard failed — show which one so trader knows why
        gtt_line = f"\n\n🔒 Manual trade only\n   {auto_reason}"
    else:
        gtt_line = ""

    footer = f"\n\n🏁 Decision Guide\n{trade_note}"

    return (
        header
        + oc_section
        + chart_section
        + iv_line
        + oi_trend_line
        + liq_line
        + gtt_line
        + footer
    )


# ── Async send ─────────────────────────────────────────────────────────────────

async def _async_send(message: str) -> bool:
    """Send a Telegram message asynchronously."""
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("[telegram_alert] BOT_TOKEN or CHAT_ID missing in .env")
        return False
    try:
        bot = Bot(token=BOT_TOKEN)
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=None)
        logger.info("[telegram_alert] ✅  Message sent to chat_id=%s", CHAT_ID)
        return True
    except TelegramError as exc:
        logger.error("[telegram_alert] TelegramError: %s", exc)
        return False
    except Exception as exc:
        logger.error("[telegram_alert] Unexpected error: %s", exc)
        return False


def send_alert(message: str) -> bool:
    """Synchronous wrapper for async Telegram send. Safe for APScheduler."""
    try:
        return asyncio.run(_async_send(message))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_send(message))
        finally:
            loop.close()


def send_full_alert(signal: SignalResult, strike: StrikeInfo,
                    oc_data: dict, extra_info: dict = None) -> bool:
    """Format and send a complete trade alert."""
    message = format_alert_message(signal, strike, oc_data, extra_info=extra_info)
    logger.info("[telegram_alert] Sending alert:\n%s", message)
    return send_alert(message)


def send_error_alert(text: str) -> bool:
    """Send a plain error/warning notification."""
    return send_alert(f"⚠️ NiftySignalBot Warning\n{text}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    from modules.chart_signals   import SignalResult
    from modules.strike_selector import StrikeInfo

    # Test 1: HIGH conviction BUY PUT with IDEAL PCR
    mock_signal = SignalResult(
        symbol="NIFTY", direction="BUY PUT", conviction="HIGH",
        signals_agreed=4, ema_bullish=False, rsi_value=39.7,
        rsi_bullish=False, vwap_bullish=False, supertrend_bullish=False,
        last_close=23134.65, last_vwap=23280.0,
        last_ema9=23196.43, last_ema21=23294.4,
    )
    mock_strike = StrikeInfo(
        symbol="NIFTY", direction="BUY PUT", spot=23134.65,
        atm_strike=23150, otm_strike=23100, option_type="PE",
        expiry_date="27 Mar 2026", expiry_raw=date(2026, 3, 27),
        nfo_symbol="NIFTY2532723100PE", premium=185.0, lot_size=75,
        lot_cost=13875.0, stop_loss=92.5, target=370.0,
        ltp_source="nse_chain", expiry_type="weekly",
    )

    # Test with IDEAL PCR for BUY PUT (PCR = 0.72 ≤ 0.8)
    mock_oc_ideal = {
        "pcr": 0.72, "pcr_trend": "Falling — Bearish",
        "max_put_oi_strike": 22800, "max_call_oi_strike": 23200,
        "max_pain": 23000, "iv_rank": 38.5,
    }

    # Test with NOT IDEAL PCR for BUY PUT (PCR = 1.15 > 0.8)
    mock_oc_not_ideal = {
        "pcr": 1.15, "pcr_trend": "Rising — Bullish",
        "max_put_oi_strike": 22800, "max_call_oi_strike": 23200,
        "max_pain": 23000, "iv_rank": 38.5,
    }

    for label, oc in [("IDEAL PCR", mock_oc_ideal),
                       ("NOT IDEAL PCR", mock_oc_not_ideal)]:
        print(f"\n{'='*55}")
        print(f"  Test: {label}")
        print("="*55)
        msg = format_alert_message(mock_signal, mock_strike, oc)
        print(msg)

    answer = input("\nSend the IDEAL PCR version to Telegram? (y/n): ").strip().lower()
    if answer == "y":
        send_full_alert(mock_signal, mock_strike, mock_oc_ideal)
        print("✅  Sent!")