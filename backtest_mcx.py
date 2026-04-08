"""
backtest_mcx.py — MCX Commodity Options Backtester (GOLDM / SILVERM)
=====================================================================

Applies the EXACT same 4-indicator model used by the live bot to
GOLDM (Gold Mini) and SILVERM (Silver Mini) commodity charts from
yfinance, then simulates monthly option trades using Black-Scholes.

Signal logic (identical to backtest.py and chart_signals.py):
  1. EMA 9/21 crossover
  2. RSI(14) > / < 50
  3. SuperTrend (ATR 10, multiplier 3)
  4. EMA50 trend
  HIGH   = all 4 agree
  MEDIUM = 3/4 agree

Key differences vs. index backtest:
  • Data source  : GC=F (Gold COMEX) / SI=F (Silver COMEX) from yfinance
                   Converted to INR using live USDINR=X rate
  • Expiry       : Monthly only — MCX options expire last Tuesday of month
  • Strike step  : GOLDM ₹500, SILVERM ₹1000 per kg
  • Lot size     : GOLDM 10g, SILVERM 5 kg
  • Trading hrs  : 9:00 AM – 11:30 PM IST (extended session)
  • Alert window : 9:00 AM – 11:00 PM IST (skip last 30 min before close)
  • No PCR       : MCX does not have publicly accessible PCR
  • No DTE filter: Monthly options have 20-30 DTE at entry — no need to skip days
  • No VIX       : India VIX tracks NIFTY only; MCX vol is commodity-driven

Usage:
    python backtest_mcx.py                       — GOLDM + SILVERM, 60 days
    python backtest_mcx.py --symbol GOLDM
    python backtest_mcx.py --symbol SILVERM
    python backtest_mcx.py --days 120            — extend lookback
    python backtest_mcx.py --show-medium         — include MEDIUM signals
    python backtest_mcx.py --no-filter-hours     — all hours, not just alert window
"""

import argparse
import math
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("❌  yfinance not installed.  Run: pip install yfinance")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")

# yfinance tickers for MCX commodities (COMEX proxies)
MCX_YF_CONFIG = {
    "GOLDM": {
        "price_ticker": "GC=F",           # COMEX Gold — USD/troy oz
        "fx_ticker":    "USDINR=X",
        "inr_formula":  lambda p, fx: p * fx * 10 / 31.1035,   # → ₹/10g
        "unit":         "₹/10g",
        "atm_step":     500,              # MCX GOLDM strike spacing ₹500/10g
        "lot_size":     10,               # 10 grams per lot
        "interval":     "1h",
        "description":  "Gold Mini (MCX)",
    },
    "SILVERM": {
        "price_ticker": "SI=F",           # COMEX Silver — USD/troy oz
        "fx_ticker":    "USDINR=X",
        "inr_formula":  lambda p, fx: p * fx * 1000 / 31.1035, # → ₹/kg
        "unit":         "₹/kg",
        "atm_step":     1000,             # MCX SILVERM strike spacing ₹1000/kg
        "lot_size":     5,                # 5 kg per lot
        "interval":     "1h",
        "description":  "Silver Mini (MCX)",
    },
}

# Indicator parameters — mirrors chart_signals.py exactly
EMA_FAST      = 9
EMA_SLOW      = 21
RSI_PERIOD    = 14
ST_ATR_PERIOD = 10
ST_MULTIPLIER = 3.0
EMA_WARMUP    = 50   # candles before trusting indicators

# Trade parameters — mirrors strike_selector.py
SL_PCT      = 0.40   # exit at 60% of entry premium
TARGET_MULT = 1.50   # exit at 150% of entry premium

# Black-Scholes
RISK_FREE_RATE  = 0.065
IV_MULTIPLIER   = 1.25    # commodity realized vol × 1.25 ≈ typical IV premium
HIST_VOL_WINDOW = 20
MIN_PREMIUM     = 20.0    # ₹ — skip deep OTM options (per 10g for GOLDM, per kg for SILVERM)

# MCX trading hours IST
MCX_OPEN_H  = 9
MCX_CLOSE_H = 23
MCX_CLOSE_M = 30

# Alert window: skip last 30 min before close (23:00–23:30) to avoid noise
ALERT_START_H = 9
ALERT_END_H   = 23   # up to 11 PM

# Post-SL cooldown — same concept as index backtest
SL_COOLDOWN_HOURS = 3

# ── Normal CDF (no scipy needed) ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ── Black-Scholes pricer ──────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:
    """Standard Black-Scholes European option price. T in years."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if option_type == "CE" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return float(S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2))
    else:
        return float(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))

# ── Indicator implementations (exact copy from backtest.py) ──────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_supertrend(df: pd.DataFrame, period: int = ST_ATR_PERIOD,
                       multiplier: float = ST_MULTIPLIER) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    tr  = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr         = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    upper       = hl2 + multiplier * atr
    lower       = hl2 - multiplier * atr
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

# ── MCX monthly expiry calculator ────────────────────────────────────────────

def mcx_monthly_expiry(ref_date: date) -> date:
    """
    Return the nearest upcoming MCX monthly options expiry.

    MCX Gold/Silver options expire on the last Tuesday of the expiry month.
    If that Tuesday has already passed (or is today), return next month's last Tuesday.

    Note: The actual NSE/MCX expiry date may shift by ±1 day around holidays.
    This is a close approximation for backtesting purposes.
    """
    # Find last Tuesday of current month
    def last_tuesday_of_month(y: int, m: int) -> date:
        # Start from last day of month and walk backwards to Tuesday
        if m == 12:
            last_day = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(y, m + 1, 1) - timedelta(days=1)
        while last_day.weekday() != 1:   # 1 = Tuesday
            last_day -= timedelta(days=1)
        return last_day

    exp = last_tuesday_of_month(ref_date.year, ref_date.month)

    # If this month's expiry has passed (or is today), use next month
    if exp <= ref_date:
        m = ref_date.month + 1
        y = ref_date.year
        if m > 12:
            m = 1
            y += 1
        exp = last_tuesday_of_month(y, m)

    return exp

# ── USDINR fetch ──────────────────────────────────────────────────────────────

def fetch_usdinr(days: int) -> Optional[pd.Series]:
    """
    Download USDINR hourly data from yfinance.
    Returns a Series indexed by datetime (UTC) with INR per USD.
    Falls back to a fixed rate if download fails.
    """
    period = f"{days + 30}d"
    try:
        df = yf.download("USDINR=X", period=period, interval="1h",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty USDINR data")
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index, utc=True)
        return close.dropna()
    except Exception as exc:
        print(f"  ⚠️  USDINR fetch failed ({exc}) — using fixed rate ₹84/USD")
        return None

# ── Price data fetch ──────────────────────────────────────────────────────────

def fetch_data(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Download commodity price data from yfinance and convert to INR.

    Returns a clean DataFrame with columns:
        open, high, low, close  (all in INR in MCX native units)
    Indexed by datetime in IST.
    """
    cfg    = MCX_YF_CONFIG[symbol]
    period = f"{days + 40}d"   # extra buffer for rolling vol warmup

    print(f"  Downloading {symbol} ({cfg['price_ticker']}) — 1H, {days} days …")

    # ── Commodity price (USD) ─────────────────────────────────────────────
    try:
        raw = yf.download(cfg["price_ticker"], period=period, interval="1h",
                          auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  ❌  No data for {cfg['price_ticker']}")
            return None
    except Exception as exc:
        print(f"  ❌  Download failed for {cfg['price_ticker']}: {exc}")
        return None

    # Flatten MultiIndex columns if present
    if hasattr(raw.columns, "levels"):
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.to_datetime(raw.index, utc=True)

    # ── USDINR rate ───────────────────────────────────────────────────────
    fx_series = fetch_usdinr(days)
    FIXED_RATE = 84.0   # fallback if yfinance unavailable

    # ── Convert to INR ────────────────────────────────────────────────────
    def to_inr(price_usd: pd.Series) -> pd.Series:
        if fx_series is not None:
            # Forward-fill FX rate to match price timestamps
            fx_aligned = fx_series.reindex(price_usd.index, method="ffill").ffill()
            fx_aligned = fx_aligned.fillna(FIXED_RATE)
            return price_usd.apply(
                lambda p: cfg["inr_formula"](p, fx_aligned.get(price_usd.index[0], FIXED_RATE))
                if pd.isna(p) else cfg["inr_formula"](p, fx_aligned.get(price_usd.index[price_usd.tolist().index(p)], FIXED_RATE) if p in price_usd.tolist() else FIXED_RATE)
            )
        else:
            return price_usd.apply(lambda p: cfg["inr_formula"](p, FIXED_RATE))

    # Simpler vectorised conversion — much faster
    if fx_series is not None:
        fx_aligned = fx_series.reindex(raw.index, method="ffill").ffill().fillna(FIXED_RATE)
    else:
        fx_aligned = pd.Series(FIXED_RATE, index=raw.index)

    df = pd.DataFrame(index=raw.index)
    formula = cfg["inr_formula"]
    df["open"]  = raw["open"].combine(fx_aligned, lambda p, fx: formula(p, fx) if pd.notna(p) else np.nan)
    df["high"]  = raw["high"].combine(fx_aligned, lambda p, fx: formula(p, fx) if pd.notna(p) else np.nan)
    df["low"]   = raw["low"].combine(fx_aligned,  lambda p, fx: formula(p, fx) if pd.notna(p) else np.nan)
    df["close"] = raw["close"].combine(fx_aligned, lambda p, fx: formula(p, fx) if pd.notna(p) else np.nan)

    # ── Convert index to IST ──────────────────────────────────────────────
    df.index = df.index.tz_convert(IST)

    # ── Filter to MCX trading hours only (9:00 AM – 11:30 PM IST) ────────
    df = df[
        ((df.index.hour > MCX_OPEN_H) |
         (df.index.hour == MCX_OPEN_H)) &
        ((df.index.hour < MCX_CLOSE_H) |
         ((df.index.hour == MCX_CLOSE_H) & (df.index.minute <= MCX_CLOSE_M)))
    ]

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    print(f"  ✅  {symbol}: {len(df)} candles after filtering  "
          f"(latest close: {cfg['unit']} {df['close'].iloc[-1]:,.0f})")
    return df

# ── Rolling vol (annualised for MCX 1H candles) ───────────────────────────────

def rolling_vol(close: pd.Series, window: int = HIST_VOL_WINDOW) -> pd.Series:
    """
    Annualised rolling realised volatility for MCX 1H candles.

    MCX trades ~14.5 hours/day (9 AM – 11:30 PM) vs NSE's 6.25h.
    Per year: 252 × 14.5 ≈ 3654 hourly candles.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(3654) * IV_MULTIPLIER

# ── Signal scanner ────────────────────────────────────────────────────────────

def scan_signals(df: pd.DataFrame, symbol: str,
                 min_conviction: str = "HIGH",
                 filter_hours: bool = True) -> list:
    """
    Scan the price DataFrame and return a list of signal dicts.

    Each signal dict contains:
        ts, direction, conviction, signals_agreed, spot, vol,
        entry_premium, lot_cost, expiry, dte, strike, option_type
    """
    cfg      = MCX_YF_CONFIG[symbol]
    atm_step = cfg["atm_step"]
    lot_size = cfg["lot_size"]

    # ── Compute indicators ────────────────────────────────────────────────
    close = df["close"]
    df    = df.copy()

    df["ema_fast"]    = compute_ema(close, EMA_FAST)
    df["ema_slow"]    = compute_ema(close, EMA_SLOW)
    df["rsi"]         = compute_rsi(close)
    df["st_trend"]    = compute_supertrend(df)
    df["ema50"]       = compute_ema(close, 50)
    df["hist_vol"]    = rolling_vol(close)

    signals = []

    for i in range(EMA_WARMUP, len(df)):
        row = df.iloc[i]
        ts  = df.index[i]

        # ── Alert window filter ───────────────────────────────────────────
        if filter_hours:
            h = ts.hour
            if not (ALERT_START_H <= h < ALERT_END_H):
                continue

        # ── MCX weekday filter — skip weekends ────────────────────────────
        if ts.weekday() >= 5:
            continue

        spot = float(row["close"])
        if spot <= 0:
            continue

        # ── EMA crossover ─────────────────────────────────────────────────
        ema_bull = float(row["ema_fast"]) > float(row["ema_slow"])

        # ── RSI ───────────────────────────────────────────────────────────
        rsi_val  = float(row["rsi"])
        rsi_bull = rsi_val > 50

        # ── SuperTrend ────────────────────────────────────────────────────
        st_bull  = int(row["st_trend"]) == 1

        # ── EMA50 ─────────────────────────────────────────────────────────
        ema50_bull = spot > float(row["ema50"])

        # ── Count agreement ───────────────────────────────────────────────
        bulls = [ema_bull, rsi_bull, st_bull, ema50_bull]
        bears = [not b for b in bulls]

        bull_count = sum(bulls)
        bear_count = sum(bears)

        if bull_count == 4:
            direction, conviction, agreed = "BUY CALL", "HIGH",   4
        elif bear_count == 4:
            direction, conviction, agreed = "BUY PUT",  "HIGH",   4
        elif bull_count == 3:
            direction, conviction, agreed = "BUY CALL", "MEDIUM", 3
        elif bear_count == 3:
            direction, conviction, agreed = "BUY PUT",  "MEDIUM", 3
        else:
            continue

        # ── Conviction gate ───────────────────────────────────────────────
        conv_rank = {"MEDIUM": 1, "HIGH": 2}
        if conv_rank.get(conviction, 0) < conv_rank.get(min_conviction, 1):
            continue

        # ── Option pricing via Black-Scholes ──────────────────────────────
        hist_vol = float(row["hist_vol"])
        if pd.isna(hist_vol) or hist_vol <= 0:
            continue

        option_type = "CE" if "CALL" in direction else "PE"
        atm    = round(spot / atm_step) * atm_step
        strike = atm + atm_step if option_type == "CE" else atm - atm_step

        expiry = mcx_monthly_expiry(ts.date())
        dte    = (expiry - ts.date()).days
        T      = dte / 365.0

        premium = bs_price(spot, strike, T, RISK_FREE_RATE, hist_vol, option_type)
        if premium < MIN_PREMIUM:
            continue

        lot_cost = round(premium * lot_size, 2)

        signals.append({
            "ts":            ts,
            "direction":     direction,
            "conviction":    conviction,
            "agreed":        agreed,
            "spot":          round(spot, 2),
            "vol":           round(hist_vol * 100, 1),
            "strike":        strike,
            "option_type":   option_type,
            "premium":       round(premium, 2),
            "lot_cost":      lot_cost,
            "expiry":        expiry,
            "dte":           dte,
        })

    return signals

# ── Trade simulator ───────────────────────────────────────────────────────────

def simulate_trades(df: pd.DataFrame, signals: list,
                    symbol: str) -> list:
    """
    Simulate trade outcomes by re-pricing the option on each subsequent candle
    until SL, Target, or forced expiry exit.

    Returns a list of completed trade dicts.
    """
    cfg      = MCX_YF_CONFIG[symbol]
    lot_size = cfg["lot_size"]
    close    = df["close"]
    hist_vol = rolling_vol(close)

    completed  = []
    open_trade = None
    sl_cooldown: dict = {"BUY CALL": None, "BUY PUT": None}

    signal_idx = 0
    signals_sorted = sorted(signals, key=lambda x: x["ts"])

    for i, (ts, row) in enumerate(df.iterrows()):
        spot = float(row["close"])
        vol  = float(hist_vol.iloc[i]) if i < len(hist_vol) and not pd.isna(hist_vol.iloc[i]) else 0.2

        # ── Check if open trade should be exited ──────────────────────────
        if open_trade is not None:
            expiry = open_trade["expiry"]
            dte    = (expiry - ts.date()).days
            T      = max(dte / 365.0, 1 / 365.0)

            current_prem = bs_price(
                spot, open_trade["strike"], T,
                RISK_FREE_RATE, vol if vol > 0 else 0.2,
                open_trade["option_type"]
            )

            entry_prem = open_trade["entry_premium"]
            sl_price   = entry_prem * (1 - SL_PCT)
            tgt_price  = entry_prem * TARGET_MULT

            # Forced exit on expiry day at 3:30 PM IST
            forced_exit = (ts.date() >= expiry and ts.hour >= 15 and ts.minute >= 15)

            exit_reason = None
            exit_price  = current_prem

            if forced_exit:
                exit_reason = "EXPIRY"
                exit_price  = max(current_prem, 0.0)
            elif current_prem <= sl_price:
                exit_reason = "SL"
                exit_price  = sl_price
                sl_cooldown[open_trade["direction"]] = ts
            elif current_prem >= tgt_price:
                exit_reason = "TARGET"
                exit_price  = tgt_price

            if exit_reason:
                pnl_per_unit = exit_price - entry_prem
                pnl_lot      = round(pnl_per_unit * lot_size, 2)
                open_trade.update({
                    "exit_ts":     ts,
                    "exit_price":  round(exit_price, 2),
                    "exit_reason": exit_reason,
                    "pnl_unit":    round(pnl_per_unit, 2),
                    "pnl_lot":     pnl_lot,
                    "hold_hours":  round((ts - open_trade["entry_ts"]).total_seconds() / 3600, 1),
                })
                completed.append(open_trade)
                open_trade = None

        # ── Check for new signal entry ────────────────────────────────────
        if open_trade is None:
            while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]["ts"] <= ts:
                sig = signals_sorted[signal_idx]
                signal_idx += 1

                if sig["ts"] != ts:
                    continue

                # SL cooldown check
                cooldown_ts = sl_cooldown.get(sig["direction"])
                if cooldown_ts is not None:
                    hours_since_sl = (ts - cooldown_ts).total_seconds() / 3600
                    if hours_since_sl < SL_COOLDOWN_HOURS:
                        continue

                # No duplicate position
                open_trade = {
                    "symbol":        symbol,
                    "direction":     sig["direction"],
                    "conviction":    sig["conviction"],
                    "entry_ts":      ts,
                    "entry_premium": sig["premium"],
                    "lot_cost":      sig["lot_cost"],
                    "strike":        sig["strike"],
                    "option_type":   sig["option_type"],
                    "expiry":        sig["expiry"],
                    "dte_at_entry":  sig["dte"],
                    "spot_at_entry": sig["spot"],
                    "vol_at_entry":  sig["vol"],
                }
                break

    # Force-close any still-open trade at end of data
    if open_trade is not None:
        last_ts   = df.index[-1]
        last_spot = float(df["close"].iloc[-1])
        last_vol  = float(hist_vol.iloc[-1]) if not pd.isna(hist_vol.iloc[-1]) else 0.2
        expiry    = open_trade["expiry"]
        dte       = max((expiry - last_ts.date()).days, 1)
        T         = dte / 365.0
        exit_prem = bs_price(last_spot, open_trade["strike"], T,
                             RISK_FREE_RATE, last_vol, open_trade["option_type"])
        pnl_unit = exit_prem - open_trade["entry_premium"]
        open_trade.update({
            "exit_ts":     last_ts,
            "exit_price":  round(exit_prem, 2),
            "exit_reason": "DATA_END",
            "pnl_unit":    round(pnl_unit, 2),
            "pnl_lot":     round(pnl_unit * lot_size, 2),
            "hold_hours":  round((last_ts - open_trade["entry_ts"]).total_seconds() / 3600, 1),
        })
        completed.append(open_trade)

    return completed

# ── Report printer ────────────────────────────────────────────────────────────

def print_report(trades: list, symbol: str, days: int,
                 min_conviction: str) -> None:
    cfg = MCX_YF_CONFIG[symbol]

    print(f"\n{'═'*70}")
    print(f"  {symbol} ({cfg['description']}) — {days}-day backtest")
    print(f"  Min conviction: {min_conviction}  |  Lot size: {cfg['lot_size']} units")
    print(f"{'═'*70}")

    if not trades:
        print("  No trades generated.")
        return

    total        = len(trades)
    wins         = [t for t in trades if t["pnl_lot"] > 0]
    losses       = [t for t in trades if t["pnl_lot"] <= 0]
    total_pnl    = sum(t["pnl_lot"] for t in trades)
    win_rate     = len(wins) / total * 100
    avg_hold     = sum(t["hold_hours"] for t in trades) / total
    avg_win      = sum(t["pnl_lot"] for t in wins)  / len(wins)  if wins   else 0
    avg_loss     = sum(t["pnl_lot"] for t in losses) / len(losses) if losses else 0

    targets  = sum(1 for t in trades if t["exit_reason"] == "TARGET")
    sls      = sum(1 for t in trades if t["exit_reason"] == "SL")
    expiries = sum(1 for t in trades if t["exit_reason"] in ("EXPIRY", "DATA_END"))

    # Group by conviction
    by_conv = {}
    for t in trades:
        c = t["conviction"]
        by_conv.setdefault(c, []).append(t)

    print(f"\n  OVERALL RESULTS")
    print(f"  {'─'*50}")
    print(f"  Total trades     : {total}")
    print(f"  Win rate         : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total P&L        : ₹{total_pnl:,.0f}")
    print(f"  Avg win / loss   : ₹{avg_win:,.0f} / ₹{avg_loss:,.0f}")
    print(f"  Avg hold time    : {avg_hold:.1f}h")
    print(f"  Exits: TGT={targets}  SL={sls}  Expiry={expiries}")

    if len(by_conv) > 1:
        print(f"\n  BY CONVICTION")
        print(f"  {'─'*50}")
        for conv in ["HIGH", "MEDIUM"]:
            grp = by_conv.get(conv, [])
            if not grp:
                continue
            g_wins = [t for t in grp if t["pnl_lot"] > 0]
            g_pnl  = sum(t["pnl_lot"] for t in grp)
            g_wr   = len(g_wins) / len(grp) * 100
            print(f"  {conv:8}: {len(grp):3} trades  "
                  f"WR={g_wr:.1f}%  P&L=₹{g_pnl:,.0f}")

    print(f"\n  TRADE LOG")
    print(f"  {'─'*70}")
    hdr = f"  {'Date':12} {'Dir':9} {'Conv':7} {'Entry':>7} {'Exit':>7} {'P&L':>8}  {'Reason':8}  {'DTEin':>5}"
    print(hdr)
    print(f"  {'─'*70}")
    cumulative = 0
    for t in sorted(trades, key=lambda x: x["entry_ts"]):
        cumulative += t["pnl_lot"]
        date_str  = t["entry_ts"].strftime("%d %b %H:%M")
        dir_short = "CALL" if "CALL" in t["direction"] else "PUT "
        pnl_str   = f"₹{t['pnl_lot']:+,.0f}"
        print(f"  {date_str:12} {dir_short:9} {t['conviction']:7} "
              f"₹{t['entry_premium']:>6.0f} ₹{t['exit_price']:>6.0f} "
              f"{pnl_str:>8}  {t['exit_reason']:8}  {t['dte_at_entry']:>5}d")
    print(f"  {'─'*70}")
    print(f"  Cumulative P&L: ₹{cumulative:,.0f}\n")

# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(symbol: str, days: int = 60,
                 min_conviction: str = "HIGH",
                 filter_hours: bool = True) -> list:
    print(f"\n{'━'*70}")
    print(f"  MCX BACKTEST — {symbol}")
    print(f"{'━'*70}")

    df = fetch_data(symbol, days=days)
    if df is None or df.empty:
        print(f"  ❌  No data for {symbol} — skipping.")
        return []

    # Trim to requested period
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    df_bt  = df[df.index >= cutoff].copy()

    print(f"  Backtest window: {df_bt.index[0].strftime('%d %b %Y')} → "
          f"{df_bt.index[-1].strftime('%d %b %Y')}  ({len(df_bt)} candles)")

    signals = scan_signals(df, symbol,   # use full df for indicator warmup
                           min_conviction=min_conviction,
                           filter_hours=filter_hours)

    # Filter signals to backtest window
    signals = [s for s in signals if s["ts"] >= cutoff]
    print(f"  Signals found: {len(signals)} ({min_conviction})")

    trades = simulate_trades(df_bt, signals, symbol)
    print_report(trades, symbol, days, min_conviction)
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCX Commodity Options Backtest — GOLDM / SILVERM"
    )
    parser.add_argument("--symbol", choices=["GOLDM", "SILVERM", "ALL"],
                        default="ALL", help="Symbol to backtest")
    parser.add_argument("--days",   type=int, default=60,
                        help="Lookback period in calendar days (default 60)")
    parser.add_argument("--show-medium", action="store_true",
                        help="Include MEDIUM conviction signals")
    parser.add_argument("--no-filter-hours", action="store_true",
                        help="Ignore alert window — use all trading hours")
    args = parser.parse_args()

    min_conv     = "MEDIUM" if args.show_medium else "HIGH"
    filter_hours = not args.no_filter_hours
    symbols      = list(MCX_YF_CONFIG.keys()) if args.symbol == "ALL" else [args.symbol]

    print("\n" + "═" * 70)
    print("  MCX Commodity Options Backtest")
    print(f"  Symbols: {', '.join(symbols)}  |  Days: {args.days}  |  Min conv: {min_conv}")
    print("═" * 70)

    all_trades = {}
    for sym in symbols:
        all_trades[sym] = run_backtest(
            sym, days=args.days,
            min_conviction=min_conv,
            filter_hours=filter_hours,
        )

    # Combined summary
    if len(symbols) > 1:
        all_t = [t for trades in all_trades.values() for t in trades]
        if all_t:
            total_pnl = sum(t["pnl_lot"] for t in all_t)
            wins      = sum(1 for t in all_t if t["pnl_lot"] > 0)
            wr        = wins / len(all_t) * 100
            print(f"\n{'═'*70}")
            print(f"  COMBINED SUMMARY  ({len(all_t)} trades)")
            print(f"  Win rate  : {wr:.1f}%")
            print(f"  Total P&L : ₹{total_pnl:,.0f}")
            print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
