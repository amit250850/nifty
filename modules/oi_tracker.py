"""
modules/oi_tracker.py — OI Time-Series Tracker (SQLite)

Stores a Put-Call OI snapshot every scan cycle (every 5 minutes) and computes
a rolling OI trend over the last N snapshots to add real context to alerts.

WHY THIS MATTERS
────────────────
A single-point PCR (what we had before) tells you the ratio right now.
OI TREND tells you what's changing — which is what actually predicts direction:

  PCR RISING  + Put OI building  → fresh put writing = bearish hedging
                                   → index traders expect upside (BULLISH for index)

  PCR FALLING + Call OI building → fresh call writing = bearish positioning
                                   → smart money selling calls (BEARISH for index)

  Call OI dropping (unwinding)   → short covering → BULLISH squeeze incoming
  Put OI dropping (unwinding)    → put writers exiting → BEARISH pressure ahead

The trend is computed over OI_TREND_LOOKBACK snapshots (default 4 × 5min = 20min).

DB FILE
───────
Stored at  data/oi_tracker.db  in the project root.
Auto-created on first run. Snapshots older than 7 days are auto-pruned.
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH          = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "oi_tracker.db",
)
OI_TREND_LOOKBACK = 4     # snapshots back  (4 × 5min = 20-minute window)
PRUNE_AFTER_DAYS  = 7     # delete rows older than this


# ── DB setup ───────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Open (and auto-create directory for) the SQLite OI database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def initialise_db() -> None:
    """Create the oi_snapshots table and indexes if they don't exist."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS oi_snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                symbol     TEXT    NOT NULL,
                pcr        REAL,
                call_oi    INTEGER,
                put_oi     INTEGER,
                underlying REAL,
                max_pain   REAL
            )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_sym_ts ON oi_snapshots(symbol, ts)"
        )
    logger.debug("[oi_tracker] DB initialised at %s", DB_PATH)


# ── Write ──────────────────────────────────────────────────────────────────────

def record_snapshot(symbol: str, oc_data: dict) -> None:
    """
    Store one OI snapshot for a symbol.

    Silently skips MCX symbols (no OI data available — pcr is None).
    Also auto-prunes rows older than PRUNE_AFTER_DAYS on each write.

    Args:
        symbol:  e.g. 'NIFTY', 'BANKNIFTY'.
        oc_data: Option chain dict from scan_option_chain().
    """
    pcr    = oc_data.get("pcr")
    is_mcx = oc_data.get("is_mcx", False)
    if is_mcx or pcr is None:
        return

    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    call_oi  = oc_data.get("call_oi")
    put_oi   = oc_data.get("put_oi")
    underly  = oc_data.get("underlying")
    max_pain = oc_data.get("max_pain")

    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO oi_snapshots(ts,symbol,pcr,call_oi,put_oi,underlying,max_pain) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, symbol, pcr, call_oi, put_oi, underly, max_pain),
            )
            # Prune stale rows (keep DB small — only last week matters)
            cutoff = (datetime.now() - timedelta(days=PRUNE_AFTER_DAYS)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            c.execute(
                "DELETE FROM oi_snapshots WHERE symbol=? AND ts < ?",
                (symbol, cutoff),
            )
        logger.debug("[oi_tracker] Snapshot for %s: PCR=%.2f  Call OI=%s  Put OI=%s",
                     symbol, pcr, call_oi, put_oi)
    except Exception as exc:
        logger.warning("[oi_tracker] Failed to record snapshot for %s: %s", symbol, exc)


# ── Read + trend ───────────────────────────────────────────────────────────────

def get_oi_trend(symbol: str, lookback: int = OI_TREND_LOOKBACK) -> Optional[dict]:
    """
    Compute the OI trend for a symbol over the last `lookback` snapshots.

    Compares the most recent snapshot to the oldest one in the lookback window,
    giving a change-over-time view rather than a single-point reading.

    Returns:
        dict with keys:
          pcr_now        (float) — current PCR
          pcr_change     (float) — positive = rising (bullish), negative = falling
          call_oi_change (int)   — positive = building, negative = unwinding
          put_oi_change  (int)   — positive = building, negative = unwinding
          trend_label    (str)   — human-readable trend summary
          trend_emoji    (str)   — 🟢 / 🔴 / ⚪
          window_min     (int)   — how many minutes the window covers

        or None if fewer than 2 snapshots are available (new session).
    """
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT pcr, call_oi, put_oi FROM oi_snapshots "
                "WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                (symbol, lookback + 1),
            ).fetchall()
    except Exception as exc:
        logger.warning("[oi_tracker] Trend query failed for %s: %s", symbol, exc)
        return None

    if len(rows) < 2:
        return None   # not enough history yet — normal for first few scans

    latest = rows[0]
    oldest = rows[-1]

    pcr_now  = float(latest["pcr"]  or 0.0)
    pcr_old  = float(oldest["pcr"]  or 0.0)
    call_now = int(latest["call_oi"] or 0)
    call_old = int(oldest["call_oi"] or 0)
    put_now  = int(latest["put_oi"]  or 0)
    put_old  = int(oldest["put_oi"]  or 0)

    pcr_chg  = round(pcr_now - pcr_old, 3)
    call_chg = call_now - call_old
    put_chg  = put_now  - put_old
    window   = (len(rows) - 1) * 5   # minutes (5-min scan interval)

    # ── Interpret the change ────────────────────────────────────────────────
    if abs(pcr_chg) < 0.03 and abs(call_chg) < 50_000 and abs(put_chg) < 50_000:
        # Essentially flat
        trend_label = f"OI stable (PCR {pcr_chg:+.2f} in {window}min)"
        trend_emoji = "⚪"

    elif pcr_chg > 0.03 and put_chg > 0:
        # Put OI growing + PCR rising = fresh put writing = bearish hedging = BULLISH
        trend_label = (
            f"PCR ↑{pcr_chg:+.2f} in {window}min — "
            f"Put buildup (+{put_chg:,}) = bearish hedging = bullish for index"
        )
        trend_emoji = "🟢"

    elif pcr_chg < -0.03 and call_chg > 0:
        # Call OI growing + PCR falling = fresh call writing = BEARISH
        trend_label = (
            f"PCR ↓{pcr_chg:+.2f} in {window}min — "
            f"Call buildup (+{call_chg:,}) = complacency = bearish"
        )
        trend_emoji = "🔴"

    elif call_chg < -50_000:
        # Significant call unwinding = short covering = BULLISH
        trend_label = (
            f"Call OI dropping ({call_chg:,}) in {window}min — "
            f"short covering = bullish squeeze"
        )
        trend_emoji = "🟢"

    elif put_chg < -50_000:
        # Significant put unwinding = protection being removed = BEARISH
        trend_label = (
            f"Put OI dropping ({put_chg:,}) in {window}min — "
            f"put unwinding = bearish pressure"
        )
        trend_emoji = "🔴"

    else:
        trend_label = f"PCR {pcr_chg:+.2f} in {window}min — mixed OI signals"
        trend_emoji = "⚪"

    return {
        "pcr_now":        pcr_now,
        "pcr_change":     pcr_chg,
        "call_oi_change": call_chg,
        "put_oi_change":  put_chg,
        "trend_label":    trend_label,
        "trend_emoji":    trend_emoji,
        "window_min":     window,
    }
