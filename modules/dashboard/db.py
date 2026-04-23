# modules/dashboard/db.py
"""
SQLite event logging for detection history.
All database access in MalTwin goes through this module.

Database file: config.DB_PATH  (logs/maltwin.db)
WAL mode:      enabled on every connection  (SRS REL-4)
Permissions:   600 after creation           (SRS SEC-3)

Schema
------
CREATE TABLE IF NOT EXISTS detection_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,   -- ISO 8601 UTC: "2025-04-22T14:35:22.123456"
    file_name        TEXT    NOT NULL,
    sha256           TEXT    NOT NULL,   -- 64-char hex
    file_format      TEXT    NOT NULL,   -- 'PE' or 'ELF'
    file_size        INTEGER NOT NULL,   -- bytes
    predicted_family TEXT    NOT NULL,
    confidence       REAL    NOT NULL,   -- softmax probability [0.0, 1.0]
    device_used      TEXT    NOT NULL    -- 'cpu', 'cuda', 'cuda:0', …
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_family    ON detection_events(predicted_family);
"""
import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import config


@contextmanager
def get_connection(db_path: Path):
    """
    Context manager for SQLite connections.
    Sets WAL journal mode and row_factory on every connection.
    Commits on clean exit, rolls back on exception.

    Usage:
        with get_connection(db_path) as conn:
            conn.execute(...)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row                 # column-name access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """
    Create the database, table, and indexes if they do not exist.
    Sets file permissions to 600 after creation.
    Safe to call multiple times (IF NOT EXISTS guards).
    Called once at app.py startup.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                file_name        TEXT    NOT NULL,
                sha256           TEXT    NOT NULL,
                file_format      TEXT    NOT NULL,
                file_size        INTEGER NOT NULL,
                predicted_family TEXT    NOT NULL,
                confidence       REAL    NOT NULL,
                device_used      TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp "
            "ON detection_events(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_family "
            "ON detection_events(predicted_family)"
        )
    os.chmod(db_path, 0o600)    # SRS SEC-3: owner read/write only


def log_detection_event(
    db_path: Path,
    file_name: str,
    sha256: str,
    file_format: str,
    file_size: int,
    predicted_family: str,
    confidence: float,
    device_used: str,
) -> None:
    """
    Insert one detection event. Retries once on failure.
    NEVER raises — a DB failure must not block displaying the detection result.
    """
    timestamp = datetime.utcnow().isoformat()
    sql = """
        INSERT INTO detection_events
            (timestamp, file_name, sha256, file_format, file_size,
             predicted_family, confidence, device_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        timestamp, file_name, sha256, file_format, file_size,
        predicted_family, confidence, device_used,
    )
    for attempt in range(2):
        try:
            with get_connection(db_path) as conn:
                conn.execute(sql, params)
            return
        except Exception as e:
            if attempt == 0:
                time.sleep(0.1)
            else:
                print(f"[MalTwin] DB write failed after retry: {e}", file=sys.stderr)


def get_recent_events(db_path: Path, limit: int = 5) -> list[dict]:
    """
    Return the `limit` most recent detection events, newest first.
    Returns empty list if DB does not exist or on any error.
    """
    if not db_path.exists():
        return []
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM detection_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []


def get_detection_stats(db_path: Path) -> dict:
    """
    Aggregate statistics for the home dashboard KPI cards.

    Returns:
        {
            'total_analyzed': int,
            'total_malware':  int,   # same as total (all detections are malware)
            'total_benign':   int,   # always 0 for now
            'model_accuracy': float | None,  # from eval_metrics.json if available
        }
    """
    if not db_path.exists():
        return {'total_analyzed': 0, 'total_malware': 0,
                'total_benign': 0, 'model_accuracy': None}
    try:
        with get_connection(db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM detection_events"
            ).fetchone()[0]
        acc = None
        metrics_path = config.PROCESSED_DIR / 'eval_metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                acc = json.load(f).get('accuracy')
        return {
            'total_analyzed': total,
            'total_malware':  total,
            'total_benign':   0,
            'model_accuracy': acc,
        }
    except Exception:
        return {'total_analyzed': 0, 'total_malware': 0,
                'total_benign': 0, 'model_accuracy': None}


def get_events_by_date_range(db_path: Path, days_back: int = 7) -> list[dict]:
    """
    Return all events from the last `days_back` days.
    Used by the home page activity chart.
    Returns empty list if DB missing or on any error.
    """
    if not db_path.exists():
        return []
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, predicted_family FROM detection_events "
                "WHERE timestamp >= ? ORDER BY timestamp ASC",
                (cutoff,),
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []
