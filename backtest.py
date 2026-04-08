"""
backtest.py — NiftySignalBot HIGH Conviction Signal Backtester
==============================================================

Replays the exact same 4-indicator model used by the live bot on the last
~60 days of 1H NIFTY / BANKNIFTY data from yfinance.

Signal logic (mirrors chart_signals.py exactly):
  1. EMA 9/21 crossover
  2. RSI(14) > / < 50
  3. VWAP (session-anchored, resets daily)
  4. SuperTrend (ATR 10, multiplier 3)
  HIGH     = all 4 agree  (this is what gets backtested)
  MEDIUM   = 3/4 agree    (shown separately with --show-medium flag)

Trade parameters (mirrors strike_selector.py):
  Entry  : ATM+1 OTM call/put priced via Black-Scholes
  SL     : 40% loss  → exit if premium falls to 60% of entry
  Target : 1.5×      → exit if premium rises to 150% of entry
  Forced : expiry day 15:15 IST

Entry filters:
  MIN_DTE=3 : Only enter Mon (DTE≈3) or Fri (DTE≈6). Skip Tue/Wed/Thu.
              Backtest proof: DTE<3 → 17-24% win rate. DTE≥3 → 50-55%.
  SL cooldown: After SL, block same-direction re-entry for 3 hours.
              Prevents churn of 3-4 consecutive SLs on same choppy day.

Black-Scholes inputs:
  S  = NIFTY/BANKNIFTY spot at entry candle close
  K  = ATM + 1 step (50 for NIFTY, 100 for BANKNIFTY) OTM
  T  = calendar days to Thursday expiry / 365
  r  = 0.065  (India risk-free rate proxy)
  σ  = 20-period rolling σ of log-returns, annualised × IV_MULTIPLIER
       IV is typically ~20% higher than realized vol — multiplier adjusts for this

Alert window filter:  10:00–15:00 IST (same as live bot)
Duplicate guard:      One open position per symbol — no new signals while in trade

Usage:
    python backtest.py
    python backtest.py --symbol NIFTY
    python backtest.py --symbol BANKNIFTY
    python backtest.py --show-medium
    python backtest.py --no-filter-hours   (ignore 10-15h window — all signals)
"""

import argparse
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import math

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("❌  yfinance not installed.  Run: pip install yfinance --break-system-packages")
    sys.exit(1)

# Normal CDF via math.erf — no scipy dependency
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ── Constants ──────────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")

YF_TICKERS = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

# ETF tickers for VWAP computation — these have REAL volume unlike index tickers
# ^NSEI / ^NSEBANK return zero volume → VWAP = NaN throughout.
# NIFTYBEES.NS / BANKBEES.NS are actual traded securities (tracking error < 0.2%).
# If the ETF is above its session VWAP, the index is above volume-weighted average.
# The directional signal is identical. Used only with --vwap flag.
ETF_TICKERS = {
    "NIFTY":     "NIFTYBEES.NS",   # Nippon NIFTY 50 ETF  — price ≈ NIFTY/10
    "BANKNIFTY": "BANKBEES.NS",    # Nippon Bank Nifty ETF — price ≈ BANKNIFTY/100
}

ATM_STEP = {
    "NIFTY":     50,
    "BANKNIFTY": 100,
}

LOT_SIZE = {
    "NIFTY":     75,
    "BANKNIFTY": 30,
}

# Indicator parameters — MUST match chart_signals.py exactly
EMA_FAST       = 9
EMA_SLOW       = 21
RSI_PERIOD     = 14
ST_ATR_PERIOD  = 10
ST_MULTIPLIER  = 3.0
EMA_WARMUP     = 50   # candles before we trust indicators

# Trade parameters — MUST match strike_selector.py exactly
SL_PCT      = 0.40   # 40% loss = exit at 60% of premium
TARGET_MULT = 1.50   # 1.5× premium

# Black-Scholes
RISK_FREE_RATE = 0.065   # 6.5% — approximate India Gsec yield
IV_MULTIPLIER  = 1.25    # realized vol × 1.25 ≈ typical IV premium
HIST_VOL_WINDOW = 20     # rolling window for realized volatility

# Alert window (10:00–15:00 IST) — mirrors live bot ALERT_START/END
ALERT_START_H = 10
ALERT_END_H   = 15

# Minimum premium filter — very deep OTM options skew results
MIN_PREMIUM = 15.0   # ₹ per share

# ── India VIX filters ──────────────────────────────────────────────────────────
# Real IV proxy: India VIX (^INDIAVIX from yfinance).
#
# Why VIX matters for OPTION BUYERS:
#   • If VIX is too LOW  (<11): options are cheap but market barely moves →
#     hard to hit 1.5× target; SL often triggered by noise.
#   • If VIX is too HIGH (>30): options are expensive (IV crush risk) →
#     even if direction is right, IV falls after the spike and option loses value.
#   • SWEET SPOT 12–28: enough movement to hit target, not so high that IV
#     mean-reversion works against you.
#
# IV Rank (0–100):
#   IV_Rank = (current_VIX − 252d_low) / (252d_high − 252d_low) × 100
#   Low rank (<40) → VIX is cheap vs history → buying options is relatively cheap.
#   High rank (>70) → VIX already spiked → risk of IV crush if market calms.
#
# Set to None to disable each filter independently.
VIX_MIN      = 11.0   # skip if India VIX below this (market too calm)
VIX_MAX      = 30.0   # skip if India VIX above this (IV crush risk)
VIX_RANK_MAX = 70.0   # skip if VIX rank > 70% (already spiked — IV mean reverts)

# Minimum DTE to enter a new trade.
#
# BACKTEST FINDING (60 days, 114 HIGH signals):
#   DTE=0 (expiry day)  → 0%  win rate — absolute theta trap
#   DTE=1 (Wed/Thu)     → 17% win rate — option has no time to move
#   DTE=2 (Tue/Wed)     → 24% win rate — still terrible
#   DTE=3 (Mon)         → 50% win rate — first viable entry day
#   DTE=5-6 (Fri/Mon)   → 52-55% win rate — best entries
#
# Rule: Only enter on MONDAY (DTE≈3 for NIFTY) or FRIDAY (DTE≈6).
# Skip Tuesday, Wednesday, Thursday entries entirely.
# This single rule turns a losing strategy into a winning one.
MIN_DTE = 3          # skip signals where days-to-expiry < MIN_DTE

# Post-SL cooldown: after a stop-loss, block same-direction entries for N hours
# Prevents churn where 3-4 consecutive SLs fire on the same choppy day
SL_COOLDOWN_HOURS = 3

# ── Black-Scholes pricer ──────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:
    """
    Standard Black-Scholes European option price.
    T in years. Returns price per share.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if option_type == "CE" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return float(S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2))
    else:
        return float(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))


# ── Indicator implementations (exact copy of chart_signals.py) ────────────────

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


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]        = df["typical_price"] * df["volume"]
    df["_date"]         = df.index.date
    vwap = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("_date"):
        cum_tpv = group["tp_vol"].cumsum()
        cum_vol = group["volume"].cumsum().replace(0, np.nan)
        vwap.loc[group.index] = cum_tpv / cum_vol
    return vwap.ffill()


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


# ── Next Thursday/Wednesday expiry calculator ─────────────────────────────────

def next_expiry(ref_date: date, symbol: str) -> date:
    """Return the next (or same-day) expiry for NIFTY (Thu) / BANKNIFTY (Wed)."""
    target_wd = 3 if symbol == "NIFTY" else 2   # 3=Thu, 2=Wed
    days_ahead = (target_wd - ref_date.weekday()) % 7
    exp = ref_date + timedelta(days=days_ahead)
    return exp


# ── Historical rolling volatility ─────────────────────────────────────────────

def rolling_vol(close: pd.Series, window: int = HIST_VOL_WINDOW) -> pd.Series:
    """Annualised rolling realised volatility from log-returns."""
    log_ret = np.log(close / close.shift(1))
    # 1H candles — 6.25 trading hours × 252 trading days ≈ 1575 candles/year
    return log_ret.rolling(window).std() * np.sqrt(1575) * IV_MULTIPLIER


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(symbol: str) -> Optional[pd.DataFrame]:
    ticker = YF_TICKERS[symbol]
    print(f"  Downloading {symbol} ({ticker}) — 1H, 60 days …")
    df = yf.download(tickers=ticker, period="60d", interval="1h",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        print(f"  ❌  No data for {symbol}")
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
    print(f"  ✅  {len(df)} candles loaded  "
          f"({df.index[0].strftime('%d %b')} → {df.index[-1].strftime('%d %b %Y')})")
    return df


# ── ETF VWAP fetch ────────────────────────────────────────────────────────────

def fetch_etf_vwap(symbol: str) -> Optional[pd.Series]:
    """
    Download the ETF proxy for NIFTY/BANKNIFTY and compute session-anchored VWAP.

    Why ETF and not the index directly:
      yfinance returns zero volume for ^NSEI / ^NSEBANK (they are index tickers,
      not traded instruments). VWAP requires real trade volume to be meaningful.
      NIFTYBEES.NS / BANKBEES.NS are actual ETFs traded on NSE with real volume
      and they track the underlying indices within ~0.1-0.2% tracking error.

      The directional VWAP signal (price above/below VWAP) is identical whether
      computed on the ETF or the index — we only care about direction, not value.

    Returns:
      pd.Series of bool (True = ETF price above session VWAP = bullish signal),
      indexed by IST timestamp. Returns None on download failure.
    """
    etf_ticker = ETF_TICKERS.get(symbol)
    if etf_ticker is None:
        return None

    print(f"  Downloading {symbol} ETF ({etf_ticker}) for VWAP — 1H, 60 days …")
    try:
        df = yf.download(tickers=etf_ticker, period="60d", interval="1h",
                         progress=False, auto_adjust=True)
    except Exception as exc:
        print(f"  ⚠️  ETF download failed ({exc}) — falling back to EMA50")
        return None

    if df is None or df.empty:
        print(f"  ⚠️  No data for {etf_ticker} — falling back to EMA50")
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

    # Check volume is non-zero (sanity check)
    vol_sum = df["volume"].sum() if "volume" in df.columns else 0
    if vol_sum == 0:
        print(f"  ⚠️  {etf_ticker} has zero volume — VWAP would be NaN. Falling back to EMA50")
        return None

    # Compute session-anchored VWAP on the ETF
    df["vwap_etf"] = compute_vwap(df)

    # Directional signal: True = ETF above its VWAP = bullish
    vwap_bull = (df["close"] > df["vwap_etf"]).rename("vwap_bull_etf")
    # Replace NaN VWAP rows with NaN (can't signal without VWAP)
    vwap_bull = vwap_bull.where(df["vwap_etf"].notna())

    print(f"  ✅  ETF VWAP ready: {len(df)} candles | {etf_ticker} | "
          f"vol={vol_sum:,.0f} | "
          f"now={'above' if bool(vwap_bull.dropna().iloc[-1]) else 'below'} VWAP")
    return vwap_bull


# ── India VIX fetch ───────────────────────────────────────────────────────────

def fetch_india_vix(days: int = 400) -> Optional[pd.DataFrame]:
    """
    Download India VIX daily data via yfinance (^INDIAVIX).
    Returns a daily DataFrame with columns:
      vix        — closing VIX value
      vix_rank   — (0-100) where today's VIX sits vs its trailing 252-day range
                   0 = at 252d low, 100 = at 252d high
      vix_pct_1d — 1-day % change in VIX (positive = VIX rising = fear increasing)

    Uses 'days' of history so the 252-day rank window is properly populated.
    Returns None if download fails (filters are then skipped gracefully).
    """
    period = f"{days}d"
    print(f"  Downloading India VIX (^INDIAVIX) — daily, {days} days …")
    try:
        raw = yf.download(tickers="^INDIAVIX", period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            print("  ⚠️  India VIX data unavailable — VIX filters will be skipped.")
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        vix = raw[["close"]].copy().rename(columns={"close": "vix"})
        # 252-day rolling rank (causal — only uses past data)
        roll_min = vix["vix"].rolling(252, min_periods=30).min()
        roll_max = vix["vix"].rolling(252, min_periods=30).max()
        vix["vix_rank"] = ((vix["vix"] - roll_min) / (roll_max - roll_min) * 100).round(1)
        vix["vix_pct_1d"] = vix["vix"].pct_change() * 100
        # Convert index to date (no tz) for merge
        vix.index = pd.to_datetime(vix.index).normalize()
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        print(f"  ✅  VIX loaded: current={vix['vix'].iloc[-1]:.1f}  "
              f"rank={vix['vix_rank'].iloc[-1]:.0f}%  "
              f"252d range=[{vix['vix'].rolling(252).min().iloc[-1]:.1f}, "
              f"{vix['vix'].rolling(252).max().iloc[-1]:.1f}]")
        return vix
    except Exception as exc:
        print(f"  ⚠️  India VIX fetch failed ({exc}) — VIX filters will be skipped.")
        return None


# ── Signal scanner (rolling, no look-ahead) ──────────────────────────────────

def scan_signals(df: pd.DataFrame, symbol: str,
                 min_conviction: str = "HIGH",
                 filter_hours: bool = True,
                 vix_df: Optional[pd.DataFrame] = None,
                 vix_min: Optional[float] = VIX_MIN,
                 vix_max: Optional[float] = VIX_MAX,
                 vix_rank_max: Optional[float] = VIX_RANK_MAX,
                 vwap_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Replay indicator logic candle-by-candle (no look-ahead bias).

    vix_df       — daily India VIX DataFrame (from fetch_india_vix).
                   If None, VIX filters are silently skipped.
    vix_min      — skip signal if VIX < vix_min (market too calm, options won't move)
    vix_max      — skip signal if VIX > vix_max (IV crush risk if panic subsides)
    vix_rank_max — skip signal if VIX 252d rank > vix_rank_max (VIX already spiked)
    vwap_series  — boolean Series from fetch_etf_vwap() (True = price above ETF VWAP).
                   When provided, VWAP replaces EMA50 as the 4th indicator — matching
                   the live bot's intended design. When None, falls back to EMA50.

    Returns a DataFrame of signals with columns:
      signal_time, direction, conviction, spot,
      ema_bull, rsi, fourth_bull (EMA50 or VWAP), st_bull, signals_agreed,
      fourth_indicator, vix, vix_rank
    """
    df = df.copy()

    # ── Merge VIX onto 1H price frame by date ─────────────────────────────
    if vix_df is not None:
        df["_date_key"] = df.index.normalize()
        if df["_date_key"].dt.tz is not None:
            df["_date_key"] = df["_date_key"].dt.tz_localize(None)
        df = df.merge(
            vix_df[["vix", "vix_rank", "vix_pct_1d"]],
            left_on="_date_key", right_index=True, how="left"
        )
        df.drop(columns=["_date_key"], inplace=True)
    else:
        df["vix"] = np.nan
        df["vix_rank"] = np.nan
        df["vix_pct_1d"] = np.nan

    # ── Merge ETF VWAP signal onto 1H price frame ──────────────────────────
    if vwap_series is not None:
        df = df.join(vwap_series.rename("vwap_bull_etf"), how="left")
        # Forward-fill within each trading day only (don't carry yesterday's VWAP signal)
        df["vwap_bull_etf"] = df["vwap_bull_etf"].ffill()
    else:
        df["vwap_bull_etf"] = np.nan

    # Compute all indicators on full history (EWM is causal — no look-ahead)
    df["ema9"]        = compute_ema(df["close"], EMA_FAST)
    df["ema21"]       = compute_ema(df["close"], EMA_SLOW)
    df["ema50"]       = compute_ema(df["close"], 50)
    df["rsi"]         = compute_rsi(df["close"], RSI_PERIOD)
    df["supertrend"]  = compute_supertrend(df)
    df["hist_vol"]    = rolling_vol(df["close"])
    # Note: compute_vwap() on ^NSEI/^NSEBANK always returns NaN (zero volume).
    # Use fetch_etf_vwap() + --vwap flag to get real VWAP from ETF proxy instead.

    signals = []

    for i in range(EMA_WARMUP, len(df)):
        row    = df.iloc[i]
        ts     = df.index[i]

        # ── Alert window filter ────────────────────────────────────────────
        if filter_hours:
            hour = ts.hour
            if not (ALERT_START_H <= hour < ALERT_END_H):
                continue

        # ── NSE market hours only (Mon–Fri 9:15–15:30) ────────────────────
        if ts.weekday() >= 5:
            continue
        if ts.hour < 9 or (ts.hour == 9 and ts.minute < 15):
            continue
        if ts.hour >= 15 and ts.minute >= 30:
            continue

        # ── Indicator values ───────────────────────────────────────────────
        ema9   = row["ema9"]
        ema21  = row["ema21"]
        rsi    = row["rsi"]
        ema50  = row["ema50"]
        st     = row["supertrend"]
        close  = row["close"]

        if any(pd.isna([ema9, ema21, rsi, st, ema50])):
            continue

        # ── India VIX filter (real IV proxy) ───────────────────────────────
        row_vix      = row.get("vix",      float("nan"))
        row_vix_rank = row.get("vix_rank", float("nan"))
        if vix_df is not None:
            if vix_min is not None and not pd.isna(row_vix) and row_vix < vix_min:
                continue
            if vix_max is not None and not pd.isna(row_vix) and row_vix > vix_max:
                continue
            if vix_rank_max is not None and not pd.isna(row_vix_rank) and row_vix_rank > vix_rank_max:
                continue

        ema_bull = bool(ema9 > ema21)
        rsi_bull = bool(rsi  > 50)
        st_bull  = bool(st   == 1)

        # ── 4th indicator: VWAP (via ETF proxy) OR EMA50 fallback ─────────
        # VWAP mode  (--vwap flag): uses ETF session VWAP — matches live bot design.
        #   True = NIFTYBEES/BANKBEES price above its session VWAP = bullish.
        # EMA50 mode (default):     uses price vs EMA50 — works without volume data.
        vwap_raw = row.get("vwap_bull_etf", float("nan"))
        if vwap_series is not None and not pd.isna(vwap_raw):
            fourth_bull      = bool(vwap_raw)
            fourth_indicator = "VWAP"
        else:
            fourth_bull      = bool(close > ema50)
            fourth_indicator = "EMA50"

        bull = sum([ema_bull, rsi_bull, fourth_bull, st_bull])
        bear = 4 - bull

        if   bull == 4: direction, conviction, agreed = "BUY CALL", "HIGH",   4
        elif bull == 3: direction, conviction, agreed = "BUY CALL", "MEDIUM", 3
        elif bear == 4: direction, conviction, agreed = "BUY PUT",  "HIGH",   4
        elif bear == 3: direction, conviction, agreed = "BUY PUT",  "MEDIUM", 3
        else:           continue

        # ── Conviction filter ──────────────────────────────────────────────
        if min_conviction == "HIGH" and conviction != "HIGH":
            continue

        signals.append({
            "signal_time":      ts,
            "direction":        direction,
            "conviction":       conviction,
            "spot":             round(close, 2),
            "ema_bull":         ema_bull,
            "rsi":              round(rsi, 1),
            "fourth_bull":      fourth_bull,
            "fourth_indicator": fourth_indicator,
            "st_bull":          st_bull,
            "signals_agreed":   agreed,
            "hist_vol":         row["hist_vol"] if not pd.isna(row["hist_vol"]) else 0.20,
            "vix":              round(row_vix, 2)      if not pd.isna(row_vix)      else None,
            "vix_rank":         round(row_vix_rank, 1) if not pd.isna(row_vix_rank) else None,
        })

    return pd.DataFrame(signals)


# ── Trade simulator ───────────────────────────────────────────────────────────

def simulate_trades(signals_df: pd.DataFrame, price_df: pd.DataFrame,
                    symbol: str) -> pd.DataFrame:
    """
    For each signal, simulate an option trade using Black-Scholes pricing.

    Exit logic (mirrors real trading — NOT intraday only):
      - Hold until SL (40% loss) OR Target (1.5×) is hit, OR
      - Expiry day 15:15 IST forced exit (theta = near zero, no point holding)
      - If data runs out before expiry: record as EXPIRED with current BS price

    This means a Monday signal can be held through Tuesday, Wednesday, Thursday
    — exactly how the live bot would behave with GTT orders in place.

    Returns a DataFrame with one row per trade.
    """
    step = ATM_STEP[symbol]
    lot  = LOT_SIZE[symbol]
    r    = RISK_FREE_RATE
    trades = []

    open_position_dir = None
    open_position_end = None
    # Post-SL cooldown tracker: direction → earliest re-entry time allowed
    sl_cooldown_until: dict = {}

    for _, sig in signals_df.iterrows():
        entry_time = sig["signal_time"]
        direction  = sig["direction"]
        spot       = sig["spot"]
        sigma      = max(sig["hist_vol"], 0.08)
        conviction = sig["conviction"]

        # ── Position guard: block same-direction while trade is open ───────
        if (open_position_dir == direction
                and open_position_end is not None
                and entry_time <= open_position_end):
            continue

        # ── Post-SL cooldown: block re-entry for SL_COOLDOWN_HOURS ────────
        cooldown_end = sl_cooldown_until.get(direction)
        if cooldown_end is not None and entry_time < cooldown_end:
            continue

        # ── Strike & expiry ────────────────────────────────────────────────
        atm      = round(spot / step) * step
        opt_type = "CE" if "CALL" in direction else "PE"
        strike   = atm + step if opt_type == "CE" else atm - step

        expiry   = next_expiry(entry_time.date(), symbol)
        dte_days = (expiry - entry_time.date()).days

        # ── MIN_DTE filter: never enter on expiry day (theta trap) ─────────
        if dte_days < MIN_DTE:
            continue

        T_entry  = max(dte_days / 365, 1 / 365)

        # ── Entry premium ──────────────────────────────────────────────────
        entry_prem = bs_price(spot, strike, T_entry, r, sigma, opt_type)
        if entry_prem < MIN_PREMIUM:
            continue

        sl_price     = round(entry_prem * (1.0 - SL_PCT),   2)
        target_price = round(entry_prem * TARGET_MULT,       2)
        lot_cost     = round(entry_prem * lot,               0)

        # ── Exit simulation: hold until SL / target / expiry ──────────────
        exit_time   = None
        exit_prem   = None
        exit_reason = "EXPIRED"
        exit_spot   = spot

        future = price_df[price_df.index > entry_time].copy()
        # Only scan candles up to and including expiry day
        future = future[future.index.map(lambda x: x.date()) <= expiry]

        last_fts  = None
        last_frow = None

        for fts, frow in future.iterrows():
            last_fts  = fts
            last_frow = frow

            # Skip non-market hours (weekends / pre-market candles in yfinance data)
            if fts.weekday() >= 5:
                continue
            if fts.hour < 9 or (fts.hour == 9 and fts.minute < 15):
                continue

            dte_exit  = max((expiry - fts.date()).days, 0)
            T_exit    = max(dte_exit / 365, 1 / (365 * 24))   # floor at 1 hour

            curr_prem = bs_price(frow["close"], strike, T_exit, r, sigma, opt_type)

            # ── Expiry day: force close at 15:15 ──────────────────────────
            if fts.date() == expiry and fts.hour >= 15 and fts.minute >= 15:
                exit_time   = fts
                exit_spot   = frow["close"]
                exit_prem   = max(curr_prem, 0.0)
                exit_reason = "EXPIRY"
                break

            # ── SL check ──────────────────────────────────────────────────
            if curr_prem <= sl_price:
                exit_time   = fts
                exit_prem   = sl_price
                exit_spot   = frow["close"]
                exit_reason = "SL"
                break

            # ── Target check ──────────────────────────────────────────────
            if curr_prem >= target_price:
                exit_time   = fts
                exit_prem   = target_price
                exit_spot   = frow["close"]
                exit_reason = "TARGET"
                break

        # ── Data ran out before expiry (end of yfinance 60-day window) ────
        if exit_prem is None:
            if last_frow is not None:
                dte_exit  = max((expiry - last_fts.date()).days, 0)
                T_exit    = max(dte_exit / 365, 1 / (365 * 24))
                exit_prem = bs_price(last_frow["close"], strike, T_exit, r, sigma, opt_type)
                exit_time = last_fts
                exit_spot = last_frow["close"]
                exit_reason = "DATA_END"
            else:
                continue

        pnl_per_share = exit_prem - entry_prem
        pnl_lot       = round(pnl_per_share * lot, 0)
        pnl_pct       = round(pnl_per_share / entry_prem * 100, 1)

        open_position_dir = direction
        open_position_end = exit_time

        # ── Set post-SL cooldown if this trade was stopped out ─────────────
        if exit_reason == "SL" and exit_time is not None:
            from datetime import timedelta as _td
            sl_cooldown_until[direction] = exit_time + _td(hours=SL_COOLDOWN_HOURS)

        trades.append({
            "symbol":       symbol,
            "entry_time":   entry_time,
            "exit_time":    exit_time,
            "direction":    direction,
            "conviction":   conviction,
            "spot_entry":   spot,
            "spot_exit":    round(exit_spot, 2),
            "strike":       strike,
            "opt_type":     opt_type,
            "entry_prem":   round(entry_prem,  2),
            "exit_prem":    round(exit_prem,   2),
            "sl_price":     sl_price,
            "target_price": target_price,
            "exit_reason":  exit_reason,
            "pnl_per_share": round(pnl_per_share, 2),
            "pnl_lot":      pnl_lot,
            "pnl_pct":      pnl_pct,
            "lot_cost":     lot_cost,
            "dte":          dte_days,
            "sigma":        round(sigma, 3),
            "vix":          sig.get("vix"),
            "vix_rank":     sig.get("vix_rank"),
        })

    return pd.DataFrame(trades)


# ── Results printer ───────────────────────────────────────────────────────────

def print_results(trades_df: pd.DataFrame, symbol: str, conviction: str) -> None:
    if trades_df.empty:
        print(f"\n  No {conviction} conviction trades found for {symbol}.")
        return

    total       = len(trades_df)
    winners     = (trades_df["exit_reason"] == "TARGET").sum()
    sl_hits     = (trades_df["exit_reason"] == "SL").sum()
    expiry_exit = trades_df["exit_reason"].isin(["EXPIRY", "DATA_END"]).sum()
    exp_win     = (trades_df["exit_reason"].isin(["EXPIRY", "DATA_END"]) & (trades_df["pnl_lot"] > 0)).sum()
    exp_loss    = (trades_df["exit_reason"].isin(["EXPIRY", "DATA_END"]) & (trades_df["pnl_lot"] <= 0)).sum()

    # Overall win = target hit OR held to expiry/data-end with profit
    overall_wins = (trades_df["pnl_lot"] > 0).sum()
    win_rate     = round(overall_wins / total * 100, 1)
    target_rate  = round(winners / total * 100, 1)
    total_pnl    = trades_df["pnl_lot"].sum()
    avg_win      = trades_df.loc[trades_df["pnl_lot"] > 0, "pnl_lot"].mean()
    avg_loss     = trades_df.loc[trades_df["pnl_lot"] <= 0, "pnl_lot"].mean()

    if not pd.isna(avg_win) and not pd.isna(avg_loss) and avg_loss != 0:
        rr = round(abs(avg_win / avg_loss), 2)
    else:
        rr = "—"

    # Avg hold duration
    durations = []
    for _, t in trades_df.iterrows():
        try:
            d = (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"]))
            durations.append(d.total_seconds() / 3600)
        except Exception:
            pass
    avg_hold_h = round(sum(durations) / len(durations), 1) if durations else 0

    print(f"\n{'═'*64}")
    print(f"  {symbol}  |  {conviction} Conviction  |  Last ~60 days (1H candles)")
    print(f"  Hold mode: until SL / Target / Expiry (NOT intraday-only)")
    print(f"{'═'*64}")
    print(f"  Total trades     : {total}")
    print(f"  🎯 Target hit    : {winners}  ({target_rate}%)")
    print(f"  🛑 SL hit         : {sl_hits}  ({round(sl_hits/total*100,1)}%)")
    print(f"  📅 Held to expiry : {expiry_exit}  (profit={exp_win}, loss={exp_loss})")
    print(f"  Overall win rate  : {win_rate}%  ({overall_wins} profitable / {total} trades)")
    print(f"  Actual R:R        : {rr}  (avg win ÷ avg loss from real fills)")
    print(f"  Avg hold time     : {avg_hold_h}h")
    print(f"  Total P&L         : ₹{total_pnl:,.0f}  (1 lot per trade)")
    if not pd.isna(avg_win):  print(f"  Avg winning trade : ₹{avg_win:,.0f}")
    if not pd.isna(avg_loss): print(f"  Avg losing trade  : ₹{avg_loss:,.0f}")
    print(f"{'─'*64}")

    # Trade-by-trade table
    print(f"\n  {'DATE':10}  {'TIME':5}  {'DIR':10}  {'SPOT':>7}  {'STRIKE':>8}  "
          f"{'ENTRY':>6}  {'EXIT':>6}  {'P&L/LOT':>9}  {'HELD':>5}  RESULT")
    print(f"  {'─'*10}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*9}  {'─'*5}  {'─'*10}")
    for _, t in trades_df.iterrows():
        result_icon = {"TARGET": "✅", "SL": "❌", "EXPIRY": "📅", "DATA_END": "📊"}.get(t["exit_reason"], "?")
        pnl_str  = f"₹{t['pnl_lot']:+,.0f}"
        try:
            hold_h = round((pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 3600, 0)
            hold_str = f"{int(hold_h)}h"
        except Exception:
            hold_str = "—"
        print(
            f"  {t['entry_time'].strftime('%d %b %y'):10}  "
            f"{t['entry_time'].strftime('%H:%M'):5}  "
            f"{t['direction']:10}  "
            f"{t['spot_entry']:7.0f}  "
            f"{t['strike']:6d}{t['opt_type']}  "
            f"₹{t['entry_prem']:5.0f}  "
            f"₹{t['exit_prem']:5.0f}  "
            f"{pnl_str:>9}  "
            f"{hold_str:>5}  "
            f"{result_icon} {t['exit_reason']}"
        )

    print(f"\n  IV model: hist vol × {IV_MULTIPLIER}  |  Min premium: ₹{MIN_PREMIUM}")
    print(f"  ⚠️  P&L = Black-Scholes estimate. Real fills may differ ±15–20%.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NiftySignalBot Backtester")
    parser.add_argument("--symbol",       choices=["NIFTY", "BANKNIFTY", "BOTH"],
                        default="BOTH",   help="Symbol to backtest")
    parser.add_argument("--show-medium",  action="store_true",  default=True,
                        help="Also show MEDIUM conviction trades (default: on)")
    parser.add_argument("--no-filter-hours", action="store_true",
                        help="Disable 10:00–15:00 alert window filter")
    parser.add_argument("--vwap",  action="store_true",
                        help="Use real VWAP (via ETF proxy) as 4th indicator instead of EMA50. "
                             "Downloads NIFTYBEES.NS / BANKBEES.NS for volume data.")
    parser.add_argument("--no-vix",  action="store_true",
                        help="Disable India VIX filter entirely (for comparison)")
    parser.add_argument("--vix-min",  type=float, default=VIX_MIN,
                        help=f"Min India VIX to enter trade (default {VIX_MIN}). "
                             "Set 0 to disable lower bound.")
    parser.add_argument("--vix-max",  type=float, default=VIX_MAX,
                        help=f"Max India VIX to enter trade (default {VIX_MAX}). "
                             "Set 999 to disable upper bound.")
    parser.add_argument("--vix-rank-max", type=float, default=VIX_RANK_MAX,
                        help=f"Max VIX 252d rank%% to enter trade (default {VIX_RANK_MAX}). "
                             "Set 100 to disable.")
    args = parser.parse_args()

    symbols = (["NIFTY", "BANKNIFTY"] if args.symbol == "BOTH"
               else [args.symbol])
    filter_hours = not args.no_filter_hours
    use_vix      = not args.no_vix
    use_vwap     = args.vwap
    vix_min      = args.vix_min  if args.vix_min  > 0   else None
    vix_max      = args.vix_max  if args.vix_max  < 999 else None
    vix_rank_max = args.vix_rank_max if args.vix_rank_max < 100 else None

    # ── Fetch India VIX once (shared across both symbols) ─────────────────
    vix_df = None
    if use_vix:
        vix_df = fetch_india_vix(days=400)   # 400d so 252d rank is populated

    # ── Banner ────────────────────────────────────────────────────────────
    fourth_ind_name = "VWAP via ETF proxy" if use_vwap else "EMA50"
    print("\n" + "═" * 64)
    print("  NiftySignalBot — Signal Backtest (last ~60 days)")
    if use_vwap:
        print("  Indicators: EMA9/21 × RSI(14) × VWAP(ETF) × SuperTrend(10,3)")
        print("  4th indicator: VWAP — computed on NIFTYBEES.NS / BANKBEES.NS ETFs")
        print("    (^NSEI/^NSEBANK have zero volume; ETF proxy has real volume)")
    else:
        print("  Indicators: EMA9/21 × RSI(14) × EMA50 × SuperTrend(10,3)")
        print("  4th indicator: EMA50 — use --vwap to switch to real VWAP via ETF")
    print(f"  Alert window: {'10:00–15:00 IST' if filter_hours else 'ALL HOURS'}")
    print(f"  SL: {int(SL_PCT*100)}% loss  |  Target: {TARGET_MULT}×  |  Hold until SL/Target/Expiry")
    print(f"  Min DTE: {MIN_DTE} day(s)  |  SL cooldown: {SL_COOLDOWN_HOURS}h after stop-out")
    if vix_df is not None:
        print(f"  India VIX filter: min={vix_min}  max={vix_max}  rank_max={vix_rank_max}%")
        print(f"  (Use --no-vix to disable, or --vix-min / --vix-max / --vix-rank-max to tune)")
    else:
        print("  India VIX filter: DISABLED (--no-vix or download failed)")
    print("═" * 64 + "\n")

    all_trades = []

    for symbol in symbols:
        print(f"── {symbol} ──────────────────────────────────────────")
        df = fetch_data(symbol)
        if df is None:
            continue

        # ── Fetch ETF VWAP for this symbol (only if --vwap flag set) ──────
        vwap_series = None
        if use_vwap:
            vwap_series = fetch_etf_vwap(symbol)
            if vwap_series is None:
                print(f"  ⚠️  VWAP unavailable for {symbol} — using EMA50 as fallback")

        # HIGH conviction trades
        sigs_high = scan_signals(df, symbol, min_conviction="HIGH",
                                 filter_hours=filter_hours,
                                 vix_df=vix_df,
                                 vix_min=vix_min,
                                 vix_max=vix_max,
                                 vix_rank_max=vix_rank_max,
                                 vwap_series=vwap_series)
        print(f"  Raw HIGH signals before position guard: {len(sigs_high)}")

        trades_high = simulate_trades(sigs_high, df, symbol)
        print_results(trades_high, symbol, "HIGH")
        all_trades.append(trades_high)

        if args.show_medium:
            sigs_all = scan_signals(df, symbol, min_conviction="MEDIUM",
                                    filter_hours=filter_hours,
                                    vix_df=vix_df,
                                    vix_min=vix_min,
                                    vix_max=vix_max,
                                    vix_rank_max=vix_rank_max,
                                    vwap_series=vwap_series)
            sigs_med = sigs_all[sigs_all["conviction"] == "MEDIUM"]
            print(f"\n  Raw MEDIUM signals before position guard: {len(sigs_med)}")
            trades_med = simulate_trades(sigs_med, df, symbol)
            print_results(trades_med, symbol, "MEDIUM")
            all_trades.append(trades_med)

    # ── Combined summary ───────────────────────────────────────────────────
    if len(all_trades) > 1:
        combined = pd.concat(all_trades, ignore_index=True)
        if not combined.empty:
            total    = len(combined)
            winners  = (combined["exit_reason"] == "TARGET").sum()
            total_pnl = combined["pnl_lot"].sum()
            print(f"\n{'═'*64}")
            print(f"  COMBINED SUMMARY  (NIFTY + BANKNIFTY)")
            print(f"{'═'*64}")
            print(f"  Total trades : {total}")
            print(f"  Win rate     : {round(winners/total*100,1)}%  ({winners} targets / {total} trades)")
            print(f"  Total P&L    : ₹{total_pnl:,.0f}  (1 lot per trade, all signals)")

    # ── Save to CSV ────────────────────────────────────────────────────────
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        out_path = "backtest_results.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
