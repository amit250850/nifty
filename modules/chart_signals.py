"""
modules/chart_signals.py — Chart Signal Engine

Data source priority for MCX SILVERM:
  1. Kite historical_data() — actual MCX SILVERM futures price in ₹/kg (real data)
  2. yfinance SI=F fallback  — COMEX Silver in USD (proxy, used only if Kite fails)

For NSE symbols (NIFTY, BANKNIFTY): always yfinance (^NSEI, ^NSEBANK).

Yahoo Finance tickers (fallback / NSE):
  NIFTY     → ^NSEI
  BANKNIFTY → ^NSEBANK
  SILVERM   → SI=F   (fallback only — COMEX Silver futures)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NIFTY / BANKNIFTY signal model (4 equal-weight signals)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. EMA 9/21 crossover
  2. RSI(14) > / < 50
  3. VWAP (session-anchored, resets daily)
  4. SuperTrend (ATR 10, multiplier 3)

  Conviction: HIGH = 4/4 | MEDIUM = 3/4 | < 3 = NO TRADE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SILVERM (MCX Silver) signal model — optimised for commodities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Silver is a trending commodity; SuperTrend is its strongest indicator.
  No PCR/OI confirmation is available, so we compensate with stricter rules:

  Gate  :  SuperTrend MUST agree with signal direction
            → Counter-SuperTrend signals are BLOCKED entirely

  Indicators (scored after gate passes):
    1. EMA 9/21 crossover         (trend momentum)
    2. RSI(14) < 45 bear / > 55 bull  (tighter than NSE's 50 threshold)
    3. Price vs EMA50             (replaces VWAP — more stable over the
                                   14.5h MCX session than session-VWAP)

  Conviction:
    HIGH   = SuperTrend ✅ + all 3 remaining agree (4/4 effectively)
    MEDIUM = SuperTrend ✅ + 2/3 remaining agree
    < 2/3  = NO TRADE (insufficient confirmation without PCR safety net)

  RSI neutral zone (45–55):
    RSI between 45–55 is ambiguous for commodities; it does NOT count as
    a confirming signal in either direction but does not block the trade.
    → Required pass threshold drops from 3 to 2 for the remaining signals.

Usage (standalone test):
    python -m modules.chart_signals
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance not installed. Run: pip install yfinance")

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# ── Yahoo Finance tickers (NSE + fallback for MCX) ───────────────────────────
YF_TICKERS = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SILVERM":   "SI=F",     # fallback only — COMEX Silver futures in USD
    "GOLDM":     "GC=F",     # fallback only — COMEX Gold futures in USD
}

# Symbols that should use Kite MCX historical data (real INR price)
MCX_SYMBOLS = {"SILVERM", "GOLDM"}

# yfinance supports max 60 days for 1h interval
LOOKBACK_PERIOD = "60d"
CANDLE_INTERVAL = "1h"
MCX_LOOKBACK_DAYS = 60    # days of history to fetch from Kite for MCX signals

# ── NSE indicator parameters ──────────────────────────────────────────────────
EMA_FAST      = 9
EMA_SLOW      = 21
RSI_PERIOD    = 14
ST_ATR_PERIOD = 10
ST_MULTIPLIER = 3.0
MIN_CANDLES   = 50

# ── MCX Silver indicator parameters (commodity-optimised) ─────────────────────
# Tighter RSI thresholds — Silver can stay near 45-55 for extended periods;
# only consider RSI a confirming signal when it has real directional commitment.
MCX_RSI_BEARISH = 45    # RSI < 45 = bearish signal for Silver
MCX_RSI_BULLISH = 55    # RSI > 55 = bullish signal for Silver

# EMA50 trend filter — replaces VWAP for MCX.
# Session VWAP is unreliable for the 14.5-hour MCX Silver session because:
#   • Silver trades from 9 AM to 11:30 PM IST
#   • COMEX opens at ~7:30 PM IST and dominates price for the final 4 hours
#   • A session VWAP anchored at 9 AM becomes stale by evening
# EMA50 (1H) ≈ 2 weeks of trend context — more robust and time-stable.
EMA_TREND = 50


# ── Signal result dataclass ───────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Container for all chart signal outputs."""
    symbol:          str
    direction:       str          # 'BUY CALL' or 'BUY PUT'
    conviction:      str          # 'HIGH' or 'MEDIUM'
    signals_agreed:  int          # number of aligned signals

    ema_bullish:        bool
    rsi_value:          float
    rsi_bullish:        bool      # True = RSI bullish (>50 NSE | >55 MCX)
    vwap_bullish:       bool      # True = price above VWAP (NSE) / EMA50 (MCX)
    supertrend_bullish: bool

    last_close:   float
    last_vwap:    float           # VWAP for NSE | EMA50 value for MCX
    last_ema9:    float
    last_ema21:   float
    is_mcx:       bool = False    # True for SILVERM — changes Telegram labels
    generated_at: datetime = field(default_factory=lambda: datetime.now(IST))


# ── Native indicator implementations ──────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA using pandas ewm (Wilder-compatible span)."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI using Wilder's smoothing (EWM with alpha=1/period)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-anchored VWAP. Resets each trading day.
    Groups candles by date and computes cumulative TP×Vol / cumVol.
    NaN values (zero-volume candles, weekend/after-hours data from yfinance)
    are forward-filled so the last valid VWAP is always available.
    """
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]        = df["typical_price"] * df["volume"]
    df["_date"]         = df.index.date

    vwap = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("_date"):
        cum_tpv = group["tp_vol"].cumsum()
        cum_vol = group["volume"].cumsum().replace(0, np.nan)
        vwap.loc[group.index] = cum_tpv / cum_vol

    vwap = vwap.ffill()
    return vwap


def compute_supertrend(df: pd.DataFrame, period: int = ST_ATR_PERIOD,
                       multiplier: float = ST_MULTIPLIER) -> pd.Series:
    """SuperTrend indicator. Returns 1 (bullish/green) or -1 (bearish/red)."""
    hl2 = (df["high"] + df["low"]) / 2
    tr  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend       = pd.Series(1, index=df.index)
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        prev_close = df["close"].iloc[i - 1]
        curr_close = df["close"].iloc[i]

        final_upper.iloc[i] = (
            upper.iloc[i]
            if upper.iloc[i] < final_upper.iloc[i - 1] or prev_close > final_upper.iloc[i - 1]
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            lower.iloc[i]
            if lower.iloc[i] > final_lower.iloc[i - 1] or prev_close < final_lower.iloc[i - 1]
            else final_lower.iloc[i - 1]
        )

        if   trend.iloc[i - 1] == -1 and curr_close > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] ==  1 and curr_close < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend


def _last_valid(series: pd.Series, fallback: float = 0.0) -> float:
    """Return the last non-NaN value; fallback if entire Series is NaN."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else fallback


# ── Kite MCX historical data (real MCX price in ₹) ────────────────────────────

def _fetch_mcx_ohlcv_via_kite(kite, symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch 1H OHLCV for an MCX commodity futures contract via Kite Connect.

    Uses the nearest active futures contract (not options) so we get the
    actual MCX price in ₹, not a USD proxy.

    Returns:
        DataFrame with lowercase [open, high, low, close, volume] columns,
        IST-indexed, or None on failure.
    """
    try:
        instruments = kite.instruments(exchange="MCX")
        today       = datetime.now(IST).date()

        futures = []
        for inst in instruments:
            if inst.get("name") != symbol:
                continue
            if inst.get("instrument_type") != "FUT":
                continue
            exp = inst.get("expiry")
            if exp is None:
                continue
            exp_date = exp.date() if hasattr(exp, "date") else exp
            if exp_date >= today:
                futures.append((exp_date, inst["instrument_token"]))

        if not futures:
            logger.warning("[chart_signals] No active %s MCX futures found", symbol)
            return None

        futures.sort(key=lambda x: x[0])
        token = futures[0][1]

        from_dt = datetime.now(IST) - timedelta(days=MCX_LOOKBACK_DAYS)
        to_dt   = datetime.now(IST)

        candles = kite.historical_data(
            instrument_token = token,
            from_date        = from_dt,
            to_date          = to_dt,
            interval         = "60minute",
            continuous       = False,
            oi               = False,
        )

        if not candles:
            logger.warning("[chart_signals] Kite returned empty candles for %s", symbol)
            return None

        df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])

        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize(IST)
        else:
            df["date"] = df["date"].dt.tz_convert(IST)

        df = df.set_index("date").sort_index()
        df.dropna(subset=["close"], inplace=True)

        logger.info(
            "[chart_signals] ✅ Kite MCX data: %d candles for %s (₹/kg, real MCX price)",
            len(df), symbol,
        )
        return df

    except Exception as exc:
        logger.warning(
            "[chart_signals] Kite MCX historical_data failed for %s: %s — will try yfinance",
            symbol, exc,
        )
        return None


# ── yfinance data fetch ────────────────────────────────────────────────────────

def fetch_historical_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch 1-hour OHLCV candles from Yahoo Finance.

    Returns:
        pd.DataFrame with lowercase columns [open, high, low, close, volume],
        or None on failure.
    """
    ticker = YF_TICKERS.get(symbol)
    if ticker is None:
        logger.error("[chart_signals] Unknown symbol: %s", symbol)
        return None

    try:
        df = yf.download(
            tickers     = ticker,
            period      = LOOKBACK_PERIOD,
            interval    = CANDLE_INTERVAL,
            progress    = False,
            auto_adjust = True,
        )
    except Exception as exc:
        logger.error("[chart_signals] yfinance download failed for %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        logger.warning("[chart_signals] No data returned from yfinance for %s", symbol)
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)

    logger.info("[chart_signals] yfinance: %d candles for %s (%s)",
                len(df), symbol, ticker)
    return df


# ── MCX commodity signal logic (SILVERM, GOLDM) ───────────────────────────────

def _compute_mcx_signals(
    symbol: str, df: pd.DataFrame, data_src: str,
    last_close: float, last_ema9: float, last_ema21: float,
    rsi_value: float, ema_bullish: bool, st_bullish: bool,
) -> Optional[SignalResult]:
    """
    Commodity-optimised signal logic for MCX symbols (SILVERM, GOLDM).

    Rules:
      • SuperTrend is a MANDATORY gate — if it disagrees, return None
      • EMA50 replaces session VWAP (more stable over the long MCX session)
      • RSI thresholds are tighter: < 45 = bearish, > 55 = bullish
        RSI 45–55 = neutral (not counted as a confirming signal)
      • Conviction: HIGH = ST ✅ + all 3 secondary agree
                   MEDIUM = ST ✅ + 2/3 secondary agree
                   else  = NO TRADE

    Args:
        All intermediate indicator values already computed by compute_signals().

    Returns:
        SignalResult if signal is valid, None otherwise.
    """
    # ── EMA50 trend filter (replaces VWAP for MCX) ────────────────────────
    df["ema50"]   = compute_ema(df["close"], EMA_TREND)
    last_ema50    = _last_valid(df["ema50"], fallback=last_close)
    trend_bullish = last_close > last_ema50

    # ── RSI with commodity-specific thresholds ────────────────────────────
    rsi_is_bullish = rsi_value > MCX_RSI_BULLISH   # > 55
    rsi_is_bearish = rsi_value < MCX_RSI_BEARISH   # < 45
    rsi_neutral    = not rsi_is_bullish and not rsi_is_bearish   # 45–55

    # ── Determine signal direction from all indicators ────────────────────
    # Score: count how many indicators favour each direction
    # When RSI is neutral it contributes 0 to both sides
    indicators_bullish = [ema_bullish, trend_bullish, st_bullish]
    if rsi_is_bullish:
        indicators_bullish.append(True)
    elif rsi_is_bearish:
        indicators_bullish.append(False)
    # rsi_neutral: don't append — pool size shrinks to 3

    pool_size     = len(indicators_bullish)
    bullish_votes = sum(indicators_bullish)
    bearish_votes = pool_size - bullish_votes

    if bullish_votes > bearish_votes:
        direction          = "BUY CALL"
        supertrend_agrees  = st_bullish
    elif bearish_votes > bullish_votes:
        direction          = "BUY PUT"
        supertrend_agrees  = not st_bullish
    else:
        logger.info("[chart_signals] %s — NO TRADE (SILVERM: signals tied %d:%d)",
                    symbol, bullish_votes, bearish_votes)
        return None

    # ── MANDATORY SuperTrend gate ─────────────────────────────────────────
    # Silver trends strongly; a counter-SuperTrend entry is high-risk without
    # PCR/OI confirmation available on MCX.
    if not supertrend_agrees:
        logger.info(
            "[chart_signals] %s — NO TRADE (SuperTrend GATE: ST=%s disagrees with %s)",
            symbol,
            "Green (Bullish)" if st_bullish else "Red (Bearish)",
            direction,
        )
        return None

    # ── Conviction: count non-SuperTrend confirming signals ──────────────
    # SuperTrend is the gate — remaining 3 indicators (EMA, RSI, EMA50)
    # determine conviction level.  RSI neutral = doesn't help OR hurt.
    if direction == "BUY CALL":
        non_st_agree = sum([
            ema_bullish,
            trend_bullish,
            rsi_is_bullish,      # False if neutral — won't count
        ])
    else:  # BUY PUT
        non_st_agree = sum([
            not ema_bullish,
            not trend_bullish,
            rsi_is_bearish,      # False if neutral — won't count
        ])

    if non_st_agree >= 3:
        conviction = "HIGH"
    elif non_st_agree >= 2:
        conviction = "MEDIUM"
    else:
        logger.info(
            "[chart_signals] %s — NO TRADE (SILVERM: %d/3 secondary signals; need ≥ 2)",
            symbol, non_st_agree,
        )
        return None

    # Signals agreed = SuperTrend + non_st_agree
    signals_agreed = 1 + non_st_agree

    logger.info(
        "[chart_signals] %s → %s | %s | EMA=%s RSI=%.1f(thr=45/55) EMA50=%s ST=✅(GATE)  "
        "[src=%s] | ST+%d/3 secondary agree",
        symbol, direction, conviction,
        "✅" if ema_bullish else "❌",
        rsi_value,
        "✅" if trend_bullish else "❌",
        data_src, non_st_agree,
    )

    return SignalResult(
        symbol             = symbol,
        direction          = direction,
        conviction         = conviction,
        signals_agreed     = signals_agreed,
        ema_bullish        = ema_bullish,
        rsi_value          = rsi_value,
        rsi_bullish        = rsi_is_bullish,   # True = RSI > 55 for MCX
        vwap_bullish       = trend_bullish,    # EMA50 trend stored in this field
        supertrend_bullish = st_bullish,
        last_close         = round(last_close, 2),
        last_vwap          = round(last_ema50, 2),  # EMA50 value stored here
        last_ema9          = round(last_ema9,  2),
        last_ema21         = round(last_ema21, 2),
        is_mcx             = True,
    )


# ── Main signal computation ───────────────────────────────────────────────────

def compute_signals(symbol: str, kite=None) -> Optional[SignalResult]:
    """
    Fetch 1H OHLCV and compute chart signals.

    For SILVERM:  Uses commodity-optimised rules (SuperTrend gate, tight RSI,
                  EMA50 trend filter).  See module docstring for full details.
    For NIFTY/BANKNIFTY: Standard 4-indicator equal-weight model.

    Args:
        symbol: 'NIFTY', 'BANKNIFTY', or 'SILVERM'.
        kite:   Authenticated KiteConnect instance (optional; enables real MCX data).

    Returns:
        SignalResult or None (NO TRADE).
    """
    df       = None
    data_src = "yfinance"

    if symbol in MCX_SYMBOLS and kite is not None:
        df = _fetch_mcx_ohlcv_via_kite(kite, symbol)
        if df is not None:
            data_src = "kite_mcx"

    if df is None:
        df = fetch_historical_data(symbol)
        if symbol in MCX_SYMBOLS and data_src != "kite_mcx":
            logger.warning(
                "[chart_signals] %s using yfinance SI=F proxy (USD) — "
                "signals may lag MCX during US off-hours", symbol,
            )

    if df is None or len(df) < MIN_CANDLES:
        logger.warning(
            "[chart_signals] Insufficient data for %s (%d candles, need %d)",
            symbol, len(df) if df is not None else 0, MIN_CANDLES,
        )
        return None

    # ── Common indicators (computed for all symbols) ───────────────────────
    df["ema9"]       = compute_ema(df["close"], EMA_FAST)
    df["ema21"]      = compute_ema(df["close"], EMA_SLOW)
    df["rsi"]        = compute_rsi(df["close"], RSI_PERIOD)
    df["supertrend"] = compute_supertrend(df)

    last_close  = _last_valid(df["close"])
    last_ema9   = _last_valid(df["ema9"])
    last_ema21  = _last_valid(df["ema21"])
    rsi_value   = round(_last_valid(df["rsi"], fallback=50.0), 2)
    st_value    = _last_valid(df["supertrend"], fallback=-1)

    ema_bullish = bool(last_ema9 > last_ema21)
    st_bullish  = bool(st_value == 1)

    # ── Route to symbol-specific conviction logic ──────────────────────────
    if symbol in MCX_SYMBOLS:
        return _compute_mcx_signals(
            symbol, df, data_src,
            last_close, last_ema9, last_ema21,
            rsi_value, ema_bullish, st_bullish,
        )

    # ── NSE path: EMA50 trend + standard RSI threshold ─────────────────────
    # NOTE: VWAP was replaced with EMA50 because yfinance index tickers
    # (^NSEI / ^NSEBANK) return zero volume → VWAP = NaN → always votes
    # bearish via fallback → HIGH BUY CALL signals could never fire.
    # EMA50 is a causal, volume-independent trend filter (same approach
    # already used for MCX commodities) and gives the 4th indicator real
    # signal power for both directions.
    df["ema50"]    = compute_ema(df["close"], EMA_TREND)
    last_ema50     = _last_valid(df["ema50"], fallback=last_close)
    ema50_bullish  = last_close > last_ema50
    rsi_bullish    = rsi_value > 50

    bullish_count = sum([ema_bullish, rsi_bullish, ema50_bullish, st_bullish])
    bearish_count = 4 - bullish_count

    if   bullish_count == 4: direction, conviction = "BUY CALL", "HIGH"
    elif bullish_count == 3: direction, conviction = "BUY CALL", "MEDIUM"
    elif bearish_count == 4: direction, conviction = "BUY PUT",  "HIGH"
    elif bearish_count == 3: direction, conviction = "BUY PUT",  "MEDIUM"
    else:
        logger.info(
            "[chart_signals] %s — NO TRADE (bullish=%d, bearish=%d)",
            symbol, bullish_count, bearish_count,
        )
        return None

    signals_agreed = max(bullish_count, bearish_count)

    result = SignalResult(
        symbol             = symbol,
        direction          = direction,
        conviction         = conviction,
        signals_agreed     = signals_agreed,
        ema_bullish        = ema_bullish,
        rsi_value          = rsi_value,
        rsi_bullish        = rsi_bullish,
        vwap_bullish       = ema50_bullish,   # field reused — stores EMA50 result for NSE
        supertrend_bullish = st_bullish,
        last_close         = round(last_close, 2),
        last_vwap          = round(last_ema50, 2),   # field reused — stores EMA50 value for NSE
        last_ema9          = round(last_ema9,  2),
        last_ema21         = round(last_ema21, 2),
        is_mcx             = False,
    )

    logger.info(
        "[chart_signals] %s → %s | %s | EMA9/21=%s RSI=%.1f EMA50=%s ST=%s  [src=%s]",
        symbol, direction, conviction,
        "✅" if ema_bullish else "❌",
        rsi_value,
        "✅" if ema50_bullish else "❌",
        "✅" if st_bullish else "❌",
        data_src,
    )
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for sym in ["NIFTY", "BANKNIFTY", "SILVERM"]:
        print(f"\n{'='*55}\n  Chart Signals — {sym}\n{'='*55}")
        result = compute_signals(sym)
        if result:
            print(f"  Direction  : {result.direction}")
            print(f"  Conviction : {result.conviction}")
            print(f"  is_mcx     : {result.is_mcx}")
            print(f"  EMA 9/21   : {'Bullish ✅' if result.ema_bullish else 'Bearish ❌'}")
            if result.is_mcx:
                rsi_thr = 45 if not result.rsi_bullish else 55
                print(f"  RSI(14)    : {result.rsi_value} "
                      f"({'> 55 bullish' if result.rsi_bullish else '< 45 bearish' if result.rsi_value < 45 else '45–55 neutral'})")
                print(f"  EMA50 Trend: price {'above ✅' if result.vwap_bullish else 'below ❌'}"
                      f"  (EMA50={result.last_vwap})")
                print(f"  SuperTrend : {'Green ✅ (GATE — passed)' if result.supertrend_bullish else 'Red ✅ (GATE — passed)'}")
            else:
                print(f"  RSI(14)    : {result.rsi_value} "
                      f"({'above ✅' if result.rsi_bullish else 'below ❌'} 50)")
                print(f"  VWAP       : price {'above ✅' if result.vwap_bullish else 'below ❌'}"
                      f"  (VWAP={result.last_vwap})")
                print(f"  SuperTrend : {'Green ✅' if result.supertrend_bullish else 'Red ❌'}")
            print(f"  Last Close : {result.last_close}")
        else:
            print("  ⏭  NO TRADE")