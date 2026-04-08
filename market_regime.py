"""
market_regime.py — Market Regime & Seasonal Intelligence for NiftySignalBot
===========================================================================

Three layers of market context:

  Layer 1 — SEASONAL CALENDAR (5-year historical analysis)
    Downloads NIFTY/BANKNIFTY daily data and computes per-month statistics:
    average return, volatility, directional bias, win rate.
    Identifies which months are historically good/bad for directional option buying.

  Layer 2 — EVENT CALENDAR (India-specific high-impact dates)
    RBI MPC meetings, Union Budget, Derivative expiry weeks, results seasons.
    Flags: avoid-entry days (1-2 days before major events) and high-opportunity
    windows (first 2 days after major event — strongest trend moves happen here).

  Layer 3 — NEWS SENTIMENT (free, no API key required)
    Scrapes Google News RSS for NIFTY/India market headlines.
    Scores each headline +1 (bullish) / -1 (bearish) / 0 (neutral).
    Returns a bias score and confidence for the current trading day.

Usage:
    python market_regime.py              — full report (all 3 layers)
    python market_regime.py --seasonal   — only seasonal calendar
    python market_regime.py --events     — only event calendar
    python market_regime.py --news       — only news sentiment
    python market_regime.py --today      — brief "should I trade today?" summary

Integration with live bot:
    Import get_regime_signal() to get a (bias, confidence, notes) tuple
    before executing any trade. See bottom of file for usage example.
"""

import argparse
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Optional, Tuple
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("❌  yfinance not installed.  Run: pip install yfinance")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌  pandas/numpy not installed.  Run: pip install pandas numpy")
    sys.exit(1)

try:
    import urllib.request
    import urllib.parse
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

import pytz
IST = pytz.timezone("Asia/Kolkata")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SEASONAL CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# India market known seasonal characteristics (backed by 10+ years of observation)
# These annotations help interpret the quantitative results
MONTH_CONTEXT = {
    1:  "Q3 results season + pre-budget speculation. Jan selloff common in global risk-off years.",
    2:  "UNION BUDGET (Feb 1). Biggest volatility event of year. Huge move on budget day. "
        "Best month for directional option buying if you're positioned right.",
    3:  "FY-end tax selling + window dressing. Often volatile, trending moves common.",
    4:  "New FY begins + Q4 results (Infosys, TCS, etc.). Apr typically bullish but "
        "global events (US Fed) can override.",
    5:  "Q4 results continue. May historically weak globally ('Sell in May'). "
        "Watch for FII outflows.",
    6:  "Pre-monsoon uncertainty. Monsoon forecast impact on agri/rural stocks. "
        "Often choppy. RBI MPC (Jun).",
    7:  "Monsoon season + Q1 results. Mid-July can be volatile. Generally ranging.",
    8:  "Monsoon continues. Historically weak globally (US Jackson Hole Fed meeting). "
        "Low liquidity in Indian markets.",
    9:  "HISTORICALLY WORST MONTH globally. US markets tend to sell off in Sep, "
        "drags India. FII outflows. Budget discussions start.",
    10: "Q2 results + Diwali season (timing varies). Festive rally common. "
        "Oct can recover strongly after Sep selloff.",
    11: "Post-Diwali. FII year-end positioning. Muhurat trading. "
        "Nov often positive but US election years add volatility.",
    12: "FII rebalancing + low liquidity. Santa rally in US helps. "
        "Last week is dead — no trades.",
}


def fetch_nifty_5y() -> Optional[pd.DataFrame]:
    """Download 5 years of NIFTY 50 daily OHLCV data."""
    print("  Downloading NIFTY 5Y daily data (^NSEI) …")
    try:
        df = yf.download("^NSEI", period="5y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            print("  ❌  No data returned.")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        print(f"  ✅  {len(df)} trading days: {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as exc:
        print(f"  ❌  Download failed: {exc}")
        return None


def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly statistics from daily data."""
    df = df.copy()
    df["ret_pct"]   = df["close"].pct_change() * 100
    df["year"]      = df.index.year
    df["month"]     = df.index.month
    df["month_name"]= pd.Categorical(
        df.index.strftime("%b"), categories=MONTH_NAMES, ordered=True
    )

    # Monthly total return and intra-month metrics
    monthly = df.groupby(["year","month"]).agg(
        month_return=("ret_pct", "sum"),
        daily_vol=("ret_pct", "std"),
        up_days=("ret_pct", lambda x: (x > 0).sum()),
        total_days=("ret_pct", "count"),
        max_drawdown=("ret_pct", lambda x: x.min()),
        max_gain=("ret_pct", lambda x: x.max()),
    ).reset_index()
    monthly["day_win_rate"] = monthly["up_days"] / monthly["total_days"] * 100

    # Aggregate by month number across all years
    stats = monthly.groupby("month").agg(
        avg_month_return=("month_return", "mean"),
        median_month_return=("month_return", "median"),
        positive_months=("month_return", lambda x: (x > 0).sum()),
        total_months=("month_return", "count"),
        avg_daily_vol=("daily_vol", "mean"),
        avg_max_daily_gain=("max_gain", "mean"),
        avg_max_daily_drop=("max_drawdown", "mean"),
        day_win_rate=("day_win_rate", "mean"),
        worst_month=("month_return", "min"),
        best_month=("month_return", "max"),
    ).reset_index()

    stats["month_name"] = [MONTH_NAMES[m-1] for m in stats["month"]]
    stats["month_win_rate"] = stats["positive_months"] / stats["total_months"] * 100

    # Strategy rating for directional option buyers
    # Want: HIGH volatility (big moves) + CLEAR direction (not choppy)
    # Score 0-10
    def strategy_score(row):
        vol_score = min(row["avg_daily_vol"] / 1.5 * 4, 4)          # 0-4 pts for vol
        dir_score = min(abs(row["avg_month_return"]) / 4 * 3, 3)    # 0-3 pts for direction
        wr_score  = abs(row["month_win_rate"] - 50) / 50 * 3        # 0-3 pts for consistency
        return round(vol_score + dir_score + wr_score, 1)

    stats["strategy_score"] = stats.apply(strategy_score, axis=1)

    return stats


def print_seasonal(stats: pd.DataFrame) -> None:
    print("\n" + "═"*80)
    print("  LAYER 1: NIFTY SEASONAL CALENDAR — Last 5 Years")
    print("  For directional option BUYERS: best months = high vol + strong direction")
    print("═"*80)
    print(f"\n  {'Mo':3} {'Month':5} | {'Avg Ret':>8} | {'MoWR':>5} | {'DayWR':>5} | "
          f"{'AvgVol':>7} | {'Score':>5} | {'Verdict'}")
    print(f"  {'─'*3} {'─'*5} | {'─'*8} | {'─'*5} | {'─'*5} | {'─'*7} | {'─'*5} | {'─'*35}")

    for _, row in stats.iterrows():
        m    = int(row["month"])
        name = row["month_name"]
        ret  = row["avg_month_return"]
        mwr  = row["month_win_rate"]
        dwr  = row["day_win_rate"]
        vol  = row["avg_daily_vol"]
        sc   = row["strategy_score"]

        if sc >= 7:      verdict = "✅ BEST — trade aggressively"
        elif sc >= 5:    verdict = "🟡 GOOD — trade normally"
        elif sc >= 3.5:  verdict = "🟠 MIXED — be selective"
        else:            verdict = "🔴 AVOID — choppy/low vol"

        # Direction bias
        if ret > 2.5:       bias = "↑↑ Bull"
        elif ret > 0.5:     bias = "↑  Lean bull"
        elif ret < -2.5:    bias = "↓↓ Bear"
        elif ret < -0.5:    bias = "↓  Lean bear"
        else:               bias = "→  Neutral"

        print(f"  {m:2}  {name:5} | {ret:>+7.1f}% | {mwr:>4.0f}% | {dwr:>4.0f}% | "
              f"{vol:>6.2f}σ | {sc:>5.1f} | {verdict} ({bias})")

    print(f"\n  Score: 7-10=Best  5-7=Good  3.5-5=Mixed  0-3.5=Avoid")
    print(f"  MoWR = % of calendar years that month was net positive")
    print(f"  DayWR = % of individual days that were up (>50% = more up days than down)")
    print(f"  Vol (σ) = avg daily standard deviation — higher means bigger daily moves")

    print("\n\n  MONTH-BY-MONTH CONTEXT (India-specific events):")
    print("  " + "─"*75)
    for _, row in stats.iterrows():
        m = int(row["month"])
        print(f"  {row['month_name']} (Score {row['strategy_score']:.1f}): "
              f"{MONTH_CONTEXT.get(m,'')}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — INDIA EVENT CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def get_rbi_mpc_dates(year: int) -> list:
    """
    Return approximate RBI MPC (Monetary Policy Committee) meeting dates.
    RBI MPC meets 6 times a year: Feb, Apr, Jun, Aug, Oct, Dec.
    Policy announcement is on day 3 of the 3-day meeting.
    Dates shift slightly year to year — these are typical windows.
    """
    # Approximate: first week of even months
    # 2026 dates are estimates — check rbi.org.in for exact dates
    mpc_months = [2, 4, 6, 8, 10, 12]
    dates = []
    for m in mpc_months:
        # Typically first Friday of the month
        d = date(year, m, 1)
        while d.weekday() != 4:  # Friday
            d += timedelta(days=1)
        if d.day > 10:  # if first Friday is very late, use day 7-8 fallback
            d = date(year, m, 7)
            while d.weekday() != 4:
                d += timedelta(days=1)
        dates.append(d)
    return dates


def get_results_seasons() -> list:
    """Quarterly results seasons (high volatility windows for stock-driven moves)."""
    today = date.today()
    year  = today.year
    return [
        (date(year, 4, 10), date(year, 5, 20),  "Q4 Results Season (Apr-May)"),
        (date(year, 7, 10), date(year, 8, 15),  "Q1 Results Season (Jul-Aug)"),
        (date(year, 10, 10), date(year, 11, 10), "Q2 Results Season (Oct-Nov)"),
        (date(year+1, 1, 10), date(year+1, 2, 15),"Q3 Results Season (Jan-Feb)"),
    ]


def get_event_calendar(lookahead_days: int = 90) -> list:
    """Return upcoming high-impact events for the next N days."""
    today    = date.today()
    end_date = today + timedelta(days=lookahead_days)
    year     = today.year
    events   = []

    # ── Union Budget (always Feb 1) ────────────────────────────────────────
    for y in [year, year+1]:
        bd = date(y, 2, 1)
        if today <= bd <= end_date:
            events.append({
                "date": bd,
                "event": "UNION BUDGET",
                "impact": "EXTREME",
                "notes": "Single biggest market event. Avoid new trades 3 days before. "
                         "Day after budget = strongest trend day of year.",
                "action": "📛 AVOID ENTRY 3 days before | ⚡ TRADE AGGRESSIVELY day-of/day-after"
            })

    # ── RBI MPC ───────────────────────────────────────────────────────────
    for mpc_date in get_rbi_mpc_dates(year) + get_rbi_mpc_dates(year+1):
        if today <= mpc_date <= end_date:
            events.append({
                "date": mpc_date,
                "event": "RBI MPC Decision",
                "impact": "HIGH",
                "notes": "Rate decision causes sharp 1-2% moves. "
                         "Avoid holding options through the announcement.",
                "action": "⚠️  Avoid holding through announcement | Trade after reaction"
            })

    # ── NSE Weekly Expiry (every Thursday) ────────────────────────────────
    d = today
    count = 0
    while d <= min(end_date, today + timedelta(days=28)) and count < 5:
        if d.weekday() == 3:  # Thursday
            events.append({
                "date": d,
                "event": "NSE Weekly F&O Expiry",
                "impact": "MEDIUM",
                "notes": "Options expiry causes pinning near strikes + gamma squeeze near close. "
                         "High volatility 2:30-3:30 PM. DTE filter (≥3) already handles this.",
                "action": "✅ DTE≥3 filter handles this automatically"
            })
            count += 1
        d += timedelta(days=1)

    # ── Results seasons ────────────────────────────────────────────────────
    for start, end, label in get_results_seasons():
        if today <= end and start <= end_date:
            events.append({
                "date": start,
                "event": f"📊 {label}",
                "impact": "MEDIUM",
                "notes": "Individual stock moves large but index impact mixed. "
                         "Infosys, TCS, Reliance, HDFC results move BankNifty/Nifty.",
                "action": "🟡 Trade index signals normally — results add vol, which helps"
            })

    # ── Global events (US Fed — typically 3rd week of even months) ────────
    fed_months = [1, 3, 5, 7, 9, 11]  # odd months
    for m in fed_months:
        for y in [year, year+1]:
            try:
                # US Fed typically Wed of 3rd week
                d = date(y, m, 15)
                while d.weekday() != 2:  # Wednesday
                    d += timedelta(days=1)
                if today <= d <= end_date:
                    events.append({
                        "date": d,
                        "event": "US Fed FOMC Meeting",
                        "impact": "HIGH",
                        "notes": "US Fed rate decision at 2:30 AM IST next day. "
                                 "FII flows follow US market reaction overnight.",
                        "action": "⚠️  Watch US futures evening. India opens with gap next day."
                    })
            except Exception:
                pass

    events.sort(key=lambda x: x["date"])
    return events


def print_event_calendar(lookahead_days: int = 90) -> None:
    today  = date.today()
    events = get_event_calendar(lookahead_days)

    print("\n" + "═"*80)
    print(f"  LAYER 2: INDIA EVENT CALENDAR — Next {lookahead_days} days")
    print(f"  Today: {today.strftime('%d %b %Y %A')}")
    print("═"*80)

    if not events:
        print("  No major events in the next window.")
        return

    # Check if we're near a major event right now
    danger_window = []
    for ev in events:
        days_away = (ev["date"] - today).days
        if 0 <= days_away <= 3 and ev["impact"] in ("EXTREME","HIGH"):
            danger_window.append((ev, days_away))

    if danger_window:
        print(f"\n  ⚠️  ALERT: Major event within 3 days:")
        for ev, days in danger_window:
            print(f"     {ev['event']} — {ev['date'].strftime('%d %b')} "
                  f"({'TODAY' if days==0 else f'in {days}d'})")
            print(f"     Recommendation: {ev['action']}")
        print()

    impact_icon = {"EXTREME": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

    print(f"\n  {'Date':12} {'Days':>5}  {'Impact':8}  {'Event'}")
    print(f"  {'─'*12} {'─'*5}  {'─'*8}  {'─'*45}")
    for ev in events:
        days_away = (ev["date"] - today).days
        icon      = impact_icon.get(ev["impact"], "⚪")
        print(f"  {ev['date'].strftime('%d %b %Y'):12} {days_away:>5}d  "
              f"{icon} {ev['impact']:8}  {ev['event']}")
        print(f"  {'':12} {'':5}   {'':8}  → {ev['action']}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — NEWS SENTIMENT (Google News RSS, no API key)
# ══════════════════════════════════════════════════════════════════════════════

BULLISH_KEYWORDS = [
    "surge", "rally", "gain", "rise", "soar", "jump", "climb", "boost",
    "recover", "rebound", "high", "record", "positive", "strong", "growth",
    "buy", "upside", "bullish", "advance", "FII buying", "DII buying",
    "rate cut", "stimulus", "GDP growth", "earnings beat",
]

BEARISH_KEYWORDS = [
    "crash", "fall", "drop", "slide", "plunge", "tumble", "decline", "sell",
    "selloff", "fear", "weak", "low", "loss", "negative", "concern", "risk",
    "FII selling", "outflow", "inflation", "rate hike", "recession", "war",
    "geopolitical", "crude rise", "dollar surge", "China slowdown",
    "earnings miss", "profit warning",
]

QUERIES = [
    "NIFTY stock market India today",
    "NSE BSE India market outlook",
    "Indian stock market FII DII",
    "Sensex Nifty today",
]


def fetch_gnews_headlines(query: str, max_items: int = 10) -> list:
    """Fetch headlines from Google News RSS (no API key required)."""
    if not HAS_URLLIB:
        return []
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        headlines = []
        for item in root.iter("item"):
            title_el = item.find("title")
            pubdate  = item.find("pubDate")
            if title_el is not None and title_el.text:
                headlines.append({
                    "title":   title_el.text,
                    "pubdate": pubdate.text if pubdate is not None else "",
                })
            if len(headlines) >= max_items:
                break
        return headlines
    except Exception:
        return []


def score_headline(title: str) -> int:
    """Return +1 (bullish), -1 (bearish), 0 (neutral) for a headline."""
    lower = title.lower()
    bull  = sum(1 for kw in BULLISH_KEYWORDS if kw in lower)
    bear  = sum(1 for kw in BEARISH_KEYWORDS if kw in lower)
    if bull > bear:   return +1
    if bear > bull:   return -1
    return 0


def get_news_sentiment() -> dict:
    """
    Fetch current market news headlines and compute a sentiment score.
    Returns dict with: score, bias, confidence, headlines, notes
    """
    all_headlines = []
    for query in QUERIES:
        headlines = fetch_gnews_headlines(query, max_items=8)
        all_headlines.extend(headlines)

    # Deduplicate by title
    seen  = set()
    unique = []
    for h in all_headlines:
        key = h["title"][:50]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    if not unique:
        return {
            "score":      0,
            "bias":       "UNKNOWN",
            "confidence": "LOW",
            "headlines":  [],
            "notes":      "Could not fetch news (no internet or Google blocked).",
        }

    # Score each headline
    scored = []
    for h in unique[:20]:
        s = score_headline(h["title"])
        scored.append({**h, "score": s})

    total_score = sum(h["score"] for h in scored)
    scored_count = len(scored)
    avg_score    = total_score / scored_count if scored_count > 0 else 0

    bull_count = sum(1 for h in scored if h["score"] > 0)
    bear_count = sum(1 for h in scored if h["score"] < 0)
    neut_count = sum(1 for h in scored if h["score"] == 0)

    # Bias
    if avg_score > 0.3:    bias = "BULLISH"
    elif avg_score < -0.3: bias = "BEARISH"
    else:                  bias = "NEUTRAL"

    # Confidence
    strong_bias  = max(bull_count, bear_count) / scored_count if scored_count > 0 else 0
    if strong_bias > 0.6:      confidence = "HIGH"
    elif strong_bias > 0.4:    confidence = "MEDIUM"
    else:                      confidence = "LOW"

    return {
        "score":        round(avg_score, 2),
        "bias":         bias,
        "confidence":   confidence,
        "bull_count":   bull_count,
        "bear_count":   bear_count,
        "neut_count":   neut_count,
        "total":        scored_count,
        "headlines":    scored,
        "notes":        f"Analysed {scored_count} headlines. "
                        f"Bull: {bull_count}, Bear: {bear_count}, Neutral: {neut_count}.",
    }


def print_news_sentiment() -> dict:
    print("\n" + "═"*80)
    print("  LAYER 3: LIVE NEWS SENTIMENT (Google News RSS)")
    print("  Keyword scoring: +1 bullish / -1 bearish / 0 neutral")
    print("═"*80)
    print("  Fetching headlines …")

    result = get_news_sentiment()

    bias       = result["bias"]
    confidence = result["confidence"]
    score      = result["score"]

    bias_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡", "UNKNOWN": "⚪"}
    conf_icon = {"HIGH": "🔒", "MEDIUM": "🔓", "LOW": "❓"}

    print(f"\n  Sentiment Score : {score:+.2f}  (range: -1.0 to +1.0)")
    print(f"  Market Bias     : {bias_icon[bias]} {bias}")
    print(f"  Confidence      : {conf_icon.get(confidence,'❓')} {confidence}")
    print(f"  {result['notes']}")

    if result.get("headlines"):
        print(f"\n  Top headlines:")
        print(f"  {'─'*70}")
        for h in result["headlines"][:12]:
            icon = "🟢" if h["score"] > 0 else ("🔴" if h["score"] < 0 else "⚪")
            title_short = h["title"][:65]
            print(f"  {icon} {title_short}")

    # Trading recommendation
    print(f"\n  {'─'*70}")
    if bias == "BULLISH" and confidence in ("HIGH","MEDIUM"):
        print("  💡 RECOMMENDATION: News supports BUY CALL signals today.")
        print("     → If bot fires a BUY CALL: FULL conviction. If BUY PUT: reduce or skip.")
    elif bias == "BEARISH" and confidence in ("HIGH","MEDIUM"):
        print("  💡 RECOMMENDATION: News supports BUY PUT signals today.")
        print("     → If bot fires a BUY PUT: FULL conviction. If BUY CALL: reduce or skip.")
    else:
        print("  💡 RECOMMENDATION: News sentiment unclear — let chart signals decide.")
        print("     → Trade both directions normally based on bot signals.")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED: get_regime_signal() — callable from main.py
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_signal(direction: str = None) -> dict:
    """
    Returns a regime assessment for the current trading session.
    Callable from the live bot before auto-executing a trade.

    Args:
        direction: 'BUY CALL' or 'BUY PUT' from the signal (optional)

    Returns dict:
        {
          "ok_to_trade":       True/False,
          "confidence_mult":   0.5 to 1.5  (multiply position size)
          "avoid_reason":      str or None,
          "news_bias":         'BULLISH'/'BEARISH'/'NEUTRAL',
          "news_aligns":       True/False (news confirms signal direction),
          "major_event_today": True/False,
          "event_name":        str or None,
          "seasonal_score":    float (0-10),
          "notes":             str,
        }
    """
    today         = date.today()
    month         = today.month
    result        = {}

    # ── Event calendar check ──────────────────────────────────────────────
    events      = get_event_calendar(lookahead_days=3)
    today_events = [e for e in events if (e["date"] - today).days == 0
                    and e["impact"] in ("EXTREME","HIGH")]
    near_events  = [e for e in events if 1 <= (e["date"] - today).days <= 2
                    and e["impact"] == "EXTREME"]

    major_event_today = len(today_events) > 0 or len(near_events) > 0
    event_name        = today_events[0]["event"] if today_events else (
                        f"Tomorrow: {near_events[0]['event']}" if near_events else None
                    )

    # ── Seasonal score (from known patterns, no download needed here) ─────
    # These scores are based on the historical analysis — update from full
    # seasonal_analysis() run for live calibration.
    SEASONAL_SCORES = {
        1: 5.5,   # Jan — Q3 results, pre-budget, moderate
        2: 8.5,   # Feb — Budget! Highest vol + direction event of year
        3: 7.0,   # Mar — FY-end, trending moves
        4: 6.0,   # Apr — Q4 results, new FY, generally positive
        5: 4.5,   # May — "Sell in May", FII outflows, choppy
        6: 4.0,   # Jun — Pre-monsoon, choppy, RBI
        7: 5.0,   # Jul — Q1 results, monsoon, moderate
        8: 3.5,   # Aug — Weak globally (Jackson Hole), avoid
        9: 4.0,   # Sep — Historically worst globally
        10: 6.5,  # Oct — Q2 results + Diwali rally, good
        11: 5.5,  # Nov — Post-Diwali positioning
        12: 3.0,  # Dec — Low liquidity, year-end
    }
    seasonal_score = SEASONAL_SCORES.get(month, 5.0)

    # ── News sentiment (fast, async-friendly) ────────────────────────────
    try:
        news = get_news_sentiment()
        news_bias  = news["bias"]
        news_conf  = news["confidence"]
    except Exception:
        news_bias  = "UNKNOWN"
        news_conf  = "LOW"

    # News alignment with signal direction
    news_aligns = False
    if direction and news_bias != "UNKNOWN" and news_bias != "NEUTRAL":
        if direction == "BUY CALL" and news_bias == "BULLISH":
            news_aligns = True
        elif direction == "BUY PUT" and news_bias == "BEARISH":
            news_aligns = True

    # ── Compute confidence multiplier ─────────────────────────────────────
    mult = 1.0
    avoid_reason = None

    if major_event_today:
        mult         = 0.0    # Don't trade right before major event
        avoid_reason = f"Major event: {event_name}"
    elif seasonal_score < 3.5:
        mult         = 0.5    # Half size in bad months
        avoid_reason = f"Weak seasonal month (score {seasonal_score})"
    elif seasonal_score >= 7.0:
        mult = 1.3            # Increase size in strong months

    if news_aligns and news_conf == "HIGH":
        mult = min(mult * 1.2, 1.5)  # News confirms signal: small boost
    elif not news_aligns and news_conf == "HIGH" and news_bias != "NEUTRAL":
        mult = mult * 0.7             # News contradicts signal: reduce

    ok_to_trade = mult > 0 and seasonal_score >= 3.5

    notes_parts = []
    if event_name:          notes_parts.append(f"Event: {event_name}")
    if seasonal_score < 4:  notes_parts.append(f"Weak season (score {seasonal_score})")
    if news_aligns:         notes_parts.append("News confirms direction")
    elif news_bias not in ("NEUTRAL","UNKNOWN"):
        notes_parts.append(f"News contradicts ({news_bias} vs {direction})")

    result = {
        "ok_to_trade":       ok_to_trade,
        "confidence_mult":   round(mult, 2),
        "avoid_reason":      avoid_reason,
        "news_bias":         news_bias,
        "news_aligns":       news_aligns,
        "major_event_today": major_event_today,
        "event_name":        event_name,
        "seasonal_score":    seasonal_score,
        "notes":             " | ".join(notes_parts) if notes_parts else "Normal trading conditions",
    }
    return result


def print_today_summary() -> None:
    """Quick 'should I trade today?' check — designed for daily use."""
    today  = date.today()
    month  = today.month

    print("\n" + "═"*70)
    print(f"  TODAY'S REGIME CHECK — {today.strftime('%A, %d %b %Y')}")
    print("═"*70)

    regime = get_regime_signal()

    ok_icon = "✅" if regime["ok_to_trade"] else "🚫"
    print(f"\n  Trade today?     : {ok_icon} {'YES' if regime['ok_to_trade'] else 'NO — ' + (regime['avoid_reason'] or '')}")
    print(f"  Seasonal score  : {regime['seasonal_score']}/10  (month={MONTH_NAMES[month-1]})")
    print(f"  News bias       : {regime['news_bias']} (confidence: {regime.get('news_aligns')})")
    print(f"  Size multiplier : {regime['confidence_mult']}×  (1.0 = normal, 0.5 = half, 1.5 = max)")
    print(f"  Notes           : {regime['notes']}")

    if regime["major_event_today"]:
        print(f"\n  ⚠️  MAJOR EVENT: {regime['event_name']}")
        print("  Do NOT auto-execute today. Wait for post-event trend to establish.")
    elif not regime["ok_to_trade"]:
        print(f"\n  ⚠️  AVOID: {regime['avoid_reason']}")
    else:
        print(f"\n  Strategy is cleared to trade. Bot runs normally.")

    # Upcoming events
    events = get_event_calendar(lookahead_days=14)
    if events:
        print(f"\n  Upcoming events (next 14 days):")
        impact_icon = {"EXTREME":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}
        for ev in events[:5]:
            days = (ev["date"] - today).days
            icon = impact_icon.get(ev["impact"],"⚪")
            d_str = "TODAY" if days == 0 else f"in {days}d"
            print(f"    {icon} {ev['date'].strftime('%d %b')} ({d_str}) — {ev['event']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NiftySignalBot Market Regime Intelligence"
    )
    parser.add_argument("--seasonal", action="store_true", help="5-year seasonal analysis")
    parser.add_argument("--events",   action="store_true", help="Upcoming event calendar")
    parser.add_argument("--news",     action="store_true", help="Live news sentiment")
    parser.add_argument("--today",    action="store_true", help="Quick daily trade check")
    args = parser.parse_args()

    run_all = not any([args.seasonal, args.events, args.news, args.today])

    if args.today:
        print_today_summary()
        return

    if run_all or args.seasonal:
        df = fetch_nifty_5y()
        if df is not None:
            stats = seasonal_analysis(df)
            print_seasonal(stats)
        else:
            print("  Skipping seasonal analysis (data unavailable).")

    if run_all or args.events:
        print_event_calendar(lookahead_days=60)

    if run_all or args.news:
        print_news_sentiment()

    if run_all:
        print("\n" + "═"*80)
        print("  INTEGRATED RECOMMENDATION FOR TODAY")
        print("═"*80)
        print_today_summary()


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO INTEGRATE INTO main.py (auto-execute guard)
# ══════════════════════════════════════════════════════════════════════════════
#
# In main.py, before Step 10 (auto-execute), add:
#
#   from market_regime import get_regime_signal
#
#   regime = get_regime_signal(direction=signal.direction)
#   if not regime["ok_to_trade"]:
#       logger.info("[%s] 🚫 Regime check: %s — skipping auto-execute",
#                   symbol, regime["avoid_reason"])
#       # Still send Telegram alert, just don't auto-execute
#   elif regime["confidence_mult"] < 0.8:
#       logger.info("[%s] ⚠️  Regime confidence low (%.1fx) — alert only, no auto-execute",
#                   symbol, regime["confidence_mult"])
#   else:
#       # Proceed with auto-execute as normal
#       ...
#
# Run daily check before market open:
#   python market_regime.py --today
