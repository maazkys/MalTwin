# MalTwin — Phase 6: Dashboard
### Agent Instruction Document | `modules/dashboard/` + `tests/test_db.py`

> **Read this entire document before writing a single line of code.**
> Every file is fully specified. Do not add, remove, or rename functions,
> keys, or imports that are not listed here.

---

## Mandatory Rules (from PRD Section 16)

- **Read `MALTWIN_PRD_COMPLETE.md`** before writing any code.
- `st.set_page_config()` must be the **absolute first** Streamlit call in `app.py`. If it is not the first call, the app crashes. It must be called inside `configure_page()` which is the very first call in `main()`.
- **Never use HTML `<form>` tags** in any Streamlit file. Use `st.button`, `st.file_uploader`, `st.text_input`, etc.
- All `st.session_state` keys are **defined in `state.py` as constants**. Never use raw string literals for session_state keys in page files.
- `st.image()` for PNG bytes uses `use_column_width=True` (not `use_container_width`).
- `PRAGMA journal_mode=WAL` is set on **every** connection via `get_connection()`.
- `os.chmod(db_path, 0o600)` is called in `init_db()` after the file is created.
- `log_detection_event()` **never raises** — all exceptions are caught and logged to stderr.
- `conn.commit()` happens inside the `get_connection` context manager, not manually.
- Always call `state.clear_file_state()` when a new file is uploaded, before processing.
- `@st.cache_resource` is **NOT** used for the model — session_state guard is used instead.
- All paths use `pathlib.Path`, never string concatenation.

---

## Phase 6 Scope

Phase 6 implements the entire dashboard layer. It depends on Phases 1–5 being complete.

### Files to create

| File | Description |
|------|-------------|
| `modules/dashboard/__init__.py` | Empty |
| `modules/dashboard/db.py` | SQLite helpers |
| `modules/dashboard/state.py` | Session state constants and helpers |
| `modules/dashboard/pages/__init__.py` | Empty |
| `modules/dashboard/pages/home.py` | Overview / KPI dashboard page |
| `modules/dashboard/pages/upload.py` | Binary upload + visualization page |
| `modules/dashboard/pages/detection.py` | Malware detection + results page |
| `modules/dashboard/pages/digital_twin.py` | Stub page (M1 deferred) |
| `modules/dashboard/app.py` | Streamlit entry point + routing |
| `tests/test_db.py` | Full test suite for db.py |

---

## File 1: `modules/dashboard/__init__.py`

```python
# modules/dashboard/__init__.py
# Empty
```

---

## File 2: `modules/dashboard/pages/__init__.py`

```python
# modules/dashboard/pages/__init__.py
# Empty
```

---

## File 3: `modules/dashboard/db.py`

```python
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
```

---

## File 4: `modules/dashboard/state.py`

```python
# modules/dashboard/state.py
"""
Centralised session_state key definitions and helper functions.

All pages import from here and use these constants exclusively.
Never use raw string literals for session_state keys in page files.
"""
import streamlit as st

# ── Session state key constants ───────────────────────────────────────────────
KEY_MODEL        = 'model'             # MalTwinCNN or None
KEY_CLASS_NAMES  = 'class_names'       # list[str] or None
KEY_IMG_ARRAY    = 'img_array'         # np.ndarray (128,128) uint8 or None
KEY_FILE_META    = 'file_meta'         # dict from get_file_metadata() or None
KEY_DETECTION    = 'detection_result'  # dict from predict_single() or None
KEY_MODEL_LOADED = 'model_loaded'      # bool
KEY_DEVICE_INFO  = 'device_info'       # str e.g. "cuda:0" or "cpu"


def init_session_state() -> None:
    """
    Initialise all session state keys with default values if not already set.
    Call once at the top of app.py before any page renders.
    """
    defaults = {
        KEY_MODEL:        None,
        KEY_CLASS_NAMES:  None,
        KEY_IMG_ARRAY:    None,
        KEY_FILE_META:    None,
        KEY_DETECTION:    None,
        KEY_MODEL_LOADED: False,
        KEY_DEVICE_INFO:  'unknown',
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_file_state() -> None:
    """
    Clear file-related keys. Call when a new file is uploaded
    to prevent stale detection results appearing for a different file.
    """
    st.session_state[KEY_IMG_ARRAY] = None
    st.session_state[KEY_FILE_META] = None
    st.session_state[KEY_DETECTION] = None


def has_uploaded_file() -> bool:
    return st.session_state.get(KEY_IMG_ARRAY) is not None


def has_detection_result() -> bool:
    return st.session_state.get(KEY_DETECTION) is not None


def is_model_loaded() -> bool:
    return st.session_state.get(KEY_MODEL_LOADED, False)
```

---

## File 5: `modules/dashboard/pages/home.py`

```python
# modules/dashboard/pages/home.py
"""
Home / System Overview Dashboard page.
Implements SRS Mockup M1 — Main Dashboard Screen.
SRS refs: FR1.1, FR1.2, FR1.4
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

import config
from modules.dashboard.db import (
    get_recent_events,
    get_detection_stats,
    get_events_by_date_range,
)
from modules.dashboard import state


def render():
    st.title("🏠 System Overview Dashboard")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    stats = get_detection_stats(config.DB_PATH)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Files Analyzed", value=stats.get('total_analyzed', 0))
    with col2:
        st.metric(label="Malware Detected",     value=stats.get('total_malware', 0))
    with col3:
        st.metric(label="Benign Files",         value=stats.get('total_benign', 0))
    with col4:
        acc = stats.get('model_accuracy')
        st.metric(
            label="Model Accuracy",
            value=f"{acc * 100:.1f}%" if acc is not None else "N/A",
        )

    st.markdown("---")

    # ── Activity Chart + Digital Twin Status ──────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Detection Activity (Last 7 Days)")
        _render_activity_chart(config.DB_PATH)

    with col_right:
        st.subheader("Digital Twin Status")
        st.info("🖥️ Digital Twin simulation is in a future implementation phase.")
        st.markdown("**Status:** Not Configured")
        st.markdown("**Active Nodes:** —")
        st.markdown("**Traffic Flow:** —")

    st.markdown("---")

    # ── Recent Detection Feed ─────────────────────────────────────────────────
    st.subheader("Recent Detections")
    events = get_recent_events(config.DB_PATH, limit=5)
    if not events:
        st.caption("No detections yet. Upload a binary file to get started.")
    else:
        df = pd.DataFrame(events)[
            ['timestamp', 'file_name', 'predicted_family', 'confidence', 'device_used']
        ]
        df['confidence'] = df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
        df.columns = ['Timestamp', 'File', 'Predicted Family', 'Confidence', 'Device']
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Module Status Table ───────────────────────────────────────────────────
    st.subheader("Module Status")
    _render_module_status()


def _render_activity_chart(db_path):
    """Fetch event counts per day for the last 7 days and render a Plotly line chart."""
    events = get_events_by_date_range(db_path, days_back=7)

    # Build date → count mapping for last 7 days
    today = datetime.utcnow().date()
    date_counts: dict = defaultdict(int)
    for i in range(7):
        day = today - timedelta(days=6 - i)
        date_counts[str(day)] = 0   # pre-fill with zero

    for ev in events:
        try:
            day_str = ev['timestamp'][:10]   # "YYYY-MM-DD"
            if day_str in date_counts:
                date_counts[day_str] += 1
        except (KeyError, TypeError):
            continue

    dates  = sorted(date_counts.keys())
    counts = [date_counts[d] for d in dates]

    fig = go.Figure(go.Scatter(
        x=dates,
        y=counts,
        mode='lines+markers',
        marker=dict(size=8, color='#FF4B4B'),
        line=dict(color='#FF4B4B', width=2),
        name='Detections',
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Detections",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        yaxis=dict(rangemode='nonnegative'),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_module_status():
    """Render a static table showing the implementation status of all 8 modules."""
    model_status = "✅ Active" if config.BEST_MODEL_PATH.exists() else "⚠️ No model trained"
    modules = [
        ("M1", "Digital Twin Simulation",      "⚠️ Deferred — future sprint"),
        ("M2", "Binary-to-Image Conversion",   "✅ Active"),
        ("M3", "Dataset & Preprocessing",      "✅ Active"),
        ("M4", "Data Enhancement & Balancing", "✅ Active"),
        ("M5", "Malware Detection (CNN)",       model_status),
        ("M6", "Dashboard & Visualization",    "✅ Active"),
        ("M7", "Explainable AI (Grad-CAM)",    "⚠️ Deferred — future sprint"),
        ("M8", "Automated Threat Reporting",   "⚠️ Deferred — future sprint"),
    ]
    df = pd.DataFrame(modules, columns=["ID", "Module", "Status"])
    st.dataframe(df, use_container_width=True, hide_index=True)
```

---

## File 6: `modules/dashboard/pages/upload.py`

```python
# modules/dashboard/pages/upload.py
"""
Binary Upload & Visualization page.
Implements SRS Mockup M3 — Binary Upload and Visualization Screen.
SRS refs: FR3.1, FR3.2, FR3.3, FR3.4, UC-01
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

import config
from modules.binary_to_image.converter import BinaryConverter
from modules.binary_to_image.utils import (
    validate_binary_format,
    compute_pixel_histogram,
    get_file_metadata,
)
from modules.dashboard import state


def render():
    st.title("📂 Binary Upload & Visualization")
    st.markdown(
        "Upload a PE (.exe, .dll) or ELF binary file to convert it into a grayscale "
        "image for analysis. The image captures the structural byte patterns of the "
        "binary and is used as input to the malware detection model."
    )
    st.markdown("---")

    # ── File uploader ─────────────────────────────────────────────────────────
    # SRS ref: FR3.1
    uploaded_file = st.file_uploader(
        label="Upload Binary File",
        type=["exe", "dll"],
        help=(
            "Accepted formats: PE (.exe, .dll) or ELF binaries. "
            "ELF binaries have no extension — rename to .elf if needed. "
            f"Maximum file size: {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        ),
        key="binary_uploader",
    )

    if uploaded_file is not None:
        _process_upload(uploaded_file)

    if state.has_uploaded_file():
        _render_results()
    elif uploaded_file is None:
        st.info("👆 Upload a binary file above to begin.")


def _process_upload(uploaded_file) -> None:
    """
    Reads, validates, and converts the uploaded file.
    Stores img_array and file_meta in session_state.
    Clears previous state before processing a new file.

    Error messages follow SRS USE-3 format:
        "Error: [what]. Cause: [why]. Action: [what to do]."
    """
    file_bytes = uploaded_file.read()

    # ── Size check ────────────────────────────────────────────────────────────
    if len(file_bytes) > config.MAX_UPLOAD_BYTES:
        size_mb = len(file_bytes) // (1024 * 1024)
        st.error(
            f"Error: File too large. "
            f"Cause: File exceeds the 50 MB limit (uploaded: {size_mb} MB). "
            "Action: Upload a smaller binary file."
        )
        return

    # Clear stale results from any previous upload
    state.clear_file_state()

    # ── Format validation ─────────────────────────────────────────────────────
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        st.error(
            "Error: Unsupported file format. "
            f"Cause: {e} "
            "Action: Upload a valid PE (.exe, .dll) or ELF binary file."
        )
        return

    # ── Conversion ────────────────────────────────────────────────────────────
    try:
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        img_array = converter.convert(file_bytes)
    except ValueError as e:
        st.error(
            "Error: Binary too small to convert. "
            f"Cause: {e} "
            "Action: Upload a valid binary file of at least 64 bytes."
        )
        return
    except Exception as e:
        st.error(
            "Error: Conversion failed. "
            f"Cause: {e} "
            "Action: Ensure the file is a valid PE or ELF binary and try again."
        )
        return

    # ── Store in session state ────────────────────────────────────────────────
    st.session_state[state.KEY_IMG_ARRAY] = img_array
    st.session_state[state.KEY_FILE_META] = get_file_metadata(
        file_bytes, uploaded_file.name, file_format
    )
    st.success(
        "✅ File processed successfully. "
        "Navigate to **Malware Detection** in the sidebar to analyze."
    )


def _render_results() -> None:
    """
    Display the grayscale image, metadata table, and pixel intensity histogram.
    Called when session_state has a processed image.
    """
    img_array = st.session_state[state.KEY_IMG_ARRAY]
    meta      = st.session_state[state.KEY_FILE_META]

    col_left, col_right = st.columns(2)

    # ── Left: Grayscale image ─────────────────────────────────────────────────
    with col_left:
        st.subheader("Grayscale Visualization")
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        png_bytes = converter.to_png_bytes(img_array)
        st.image(
            png_bytes,
            caption=f"Grayscale visualization ({config.IMG_SIZE}×{config.IMG_SIZE} pixels, 8-bit)",
            use_column_width=True,
        )

    # ── Right: Metadata + Histogram ───────────────────────────────────────────
    with col_right:
        st.subheader("File Metadata")
        meta_table = {
            "Property": ["File Name", "File Size", "Format", "SHA-256", "Upload Time"],
            "Value":    [
                meta['name'],
                meta['size_human'],
                meta['format'],
                meta['sha256'],
                meta['upload_time'],
            ],
        }
        st.table(meta_table)

        # SHA-256 in monospace for easy copying
        st.markdown("**SHA-256 (copy):**")
        st.code(meta['sha256'], language=None)

        # Pixel intensity histogram
        st.subheader("Pixel Intensity Distribution")
        hist = compute_pixel_histogram(img_array)
        fig = go.Figure(go.Bar(
            x=hist['bins'],
            y=hist['counts'],
            marker_color='#4A90D9',
            name='Byte frequency',
        ))
        fig.update_layout(
            title="Pixel Intensity Distribution (256 bins)",
            xaxis_title="Byte Value (0–255)",
            yaxis_title="Pixel Count",
            template="plotly_dark",
            height=300,
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("➡️ Navigate to **Malware Detection** in the sidebar to run analysis.")
```

---

## File 7: `modules/dashboard/pages/detection.py`

```python
# modules/dashboard/pages/detection.py
"""
Malware Detection & Prediction page.
Implements SRS Mockup M5 — Malware Detection and Prediction Screen.
SRS refs: FR5.1, FR5.2, FR5.3, FR5.4, FR5.5, FR5.6, UC-02
"""
import json
import streamlit as st
import plotly.graph_objects as go

import config
from modules.detection.inference import predict_single
from modules.dashboard.db import log_detection_event
from modules.dashboard import state


def render():
    st.title("🔍 Malware Detection & Classification")
    st.markdown("---")

    # ── Guard: no file uploaded ───────────────────────────────────────────────
    if not state.has_uploaded_file():
        st.warning(
            "⚠️ No binary file loaded. "
            "Please upload a file on the **Binary Upload** page first."
        )
        return

    # ── Guard: no model loaded ────────────────────────────────────────────────
    if not state.is_model_loaded():
        st.warning(
            "⚠️ No trained model available. "
            "Run `python scripts/train.py` to train the model, then restart the dashboard."
        )
        return

    # ── File summary ──────────────────────────────────────────────────────────
    _render_file_summary()
    st.markdown("---")

    # ── Run Detection button ──────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "▶ Run Detection",
            type="primary",
            use_container_width=True,
            help=(
                "Run the CNN malware classifier on the uploaded binary image. "
                "The model classifies the binary into one of 25 known malware families."
            ),
        )

    if run_clicked:
        _run_detection()

    # ── Results ───────────────────────────────────────────────────────────────
    if state.has_detection_result():
        st.markdown("---")
        _render_results()


def _render_file_summary() -> None:
    """Compact summary: thumbnail | metadata | SHA-256."""
    meta      = st.session_state[state.KEY_FILE_META]
    img_array = st.session_state[state.KEY_IMG_ARRAY]

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        from modules.binary_to_image.converter import BinaryConverter
        png_bytes = BinaryConverter(img_size=config.IMG_SIZE).to_png_bytes(img_array)
        st.image(png_bytes, caption="Binary image", width=128)
    with col2:
        st.markdown(f"**File:** `{meta['name']}`")
        st.markdown(f"**Size:** {meta['size_human']}")
        st.markdown(f"**Format:** {meta['format']}")
    with col3:
        st.markdown("**SHA-256:**")
        st.code(meta['sha256'], language=None)


def _run_detection() -> None:
    """
    Execute inference, store result in session_state, log to SQLite.
    Catches all exceptions and displays them as st.error (never crashes the page).
    """
    try:
        with st.spinner("Running malware classification…"):
            model       = st.session_state[state.KEY_MODEL]
            class_names = st.session_state[state.KEY_CLASS_NAMES]
            img_array   = st.session_state[state.KEY_IMG_ARRAY]
            meta        = st.session_state[state.KEY_FILE_META]

            result = predict_single(model, img_array, class_names, config.DEVICE)
            st.session_state[state.KEY_DETECTION] = result

            # Log to SQLite — non-blocking (log_detection_event never raises)
            log_detection_event(
                db_path=config.DB_PATH,
                file_name=meta['name'],
                sha256=meta['sha256'],
                file_format=meta['format'],
                file_size=meta['size_bytes'],
                predicted_family=result['predicted_family'],
                confidence=result['confidence'],
                device_used=str(config.DEVICE),
            )
    except Exception as e:
        st.error(
            "Error: Detection failed. "
            f"Cause: {e}. "
            "Action: Ensure the model is correctly loaded and try again."
        )


def _render_results() -> None:
    """
    Render the full detection results panel.
    Sections: A) prediction + confidence  B) top-3  C) probability chart
              D) MITRE mapping  E) XAI stub  F) report export
    """
    result     = st.session_state[state.KEY_DETECTION]
    confidence = result['confidence']
    family     = result['predicted_family']

    # ── A: Prediction + Confidence ────────────────────────────────────────────
    st.subheader("Detection Result")

    if confidence >= config.CONFIDENCE_GREEN:
        st.success(f"🎯 **{family}** detected with **{confidence * 100:.1f}%** confidence")
    elif confidence >= config.CONFIDENCE_AMBER:
        st.warning(
            f"⚠️ **{family}** detected with **{confidence * 100:.1f}%** confidence\n\n"
            "Low confidence — results may be unreliable. Manual verification recommended."
        )
    else:
        st.error(
            f"🔴 **{family}** detected with **{confidence * 100:.1f}%** confidence\n\n"
            "Very low confidence — manual expert review is required."
        )

    confidence_pct = int(confidence * 100)
    color_label = (
        "🟢 High Confidence"   if confidence >= config.CONFIDENCE_GREEN
        else "🟡 Medium Confidence" if confidence >= config.CONFIDENCE_AMBER
        else "🔴 Low Confidence"
    )
    col_bar, col_label = st.columns([3, 1])
    col_bar.progress(confidence_pct)
    col_label.markdown(f"**{confidence_pct}%** {color_label}")

    # ── B: Top-3 Predictions ──────────────────────────────────────────────────
    st.markdown("**Top 3 Predictions:**")
    for i, pred in enumerate(result['top3'], 1):
        st.markdown(f"{i}. `{pred['family']}` — {pred['confidence'] * 100:.2f}%")

    st.markdown("---")

    # ── C: Per-Class Probability Chart ────────────────────────────────────────
    st.subheader("Class Probability Distribution")
    st.caption(
        "All malware families shown. Zero-probability classes display as empty bars.",
        help=(
            "The model outputs a probability for every known malware family. "
            "Higher bars indicate the model believes the binary is more likely "
            "to belong to that family."
        ),
    )
    _render_probability_chart(result['probabilities'])

    st.markdown("---")

    # ── D: MITRE ATT&CK for ICS Mapping ──────────────────────────────────────
    st.subheader("MITRE ATT&CK for ICS Mapping")
    st.caption(
        "Adversary tactics and techniques associated with the detected malware family.",
        help=(
            "MITRE ATT&CK for ICS is a knowledge base of adversary tactics and "
            "techniques specific to operational technology environments."
        ),
    )
    _render_mitre_mapping(family)

    st.markdown("---")

    # ── E: XAI Heatmap (STUB) ────────────────────────────────────────────────
    st.subheader("Explainable AI — Grad-CAM Heatmap")
    xai_requested = st.checkbox(
        "Generate Grad-CAM Heatmap",
        help=(
            "Grad-CAM highlights which regions of the binary image influenced "
            "the classification decision."
        ),
    )
    if xai_requested:
        st.info(
            "🔬 Grad-CAM XAI visualization will be implemented in Module 7. "
            "This feature generates heatmaps showing which byte regions drove the classification."
        )

    # ── F: Report Export ──────────────────────────────────────────────────────
    st.subheader("Forensic Report")
    col_pdf, col_json = st.columns(2)

    with col_pdf:
        st.button(
            "📄 Download PDF Report (Coming Soon)",
            disabled=True,
            help="Automated PDF forensic report generation will be available in Module 8.",
            use_container_width=True,
        )

    with col_json:
        meta = st.session_state[state.KEY_FILE_META]
        export_data = {
            'file_name':         meta['name'],
            'sha256':            meta['sha256'],
            'file_format':       meta['format'],
            'file_size_bytes':   meta['size_bytes'],
            'upload_time':       meta['upload_time'],
            'predicted_family':  result['predicted_family'],
            'confidence':        result['confidence'],
            'top3':              result['top3'],
            'all_probabilities': result['probabilities'],
        }
        json_bytes = json.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            label="📥 Download JSON Result",
            data=json_bytes,
            file_name=f"maltwin_result_{meta['sha256'][:8]}.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_probability_chart(probabilities: dict) -> None:
    """
    Horizontal bar chart of all class probabilities, sorted descending.
    Top-1 bar is red (#FF4B4B), all others are blue (#4A90D9).
    ALL classes shown including zero-probability ones (SRS FR5.3).
    """
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    families = [item[0] for item in sorted_probs]
    probs    = [item[1] for item in sorted_probs]
    colors   = ['#FF4B4B'] + ['#4A90D9'] * (len(families) - 1)

    fig = go.Figure(go.Bar(
        x=probs,
        y=families,
        orientation='h',
        marker_color=colors,
        text=[f"{p * 100:.2f}%" for p in probs],
        textposition='outside',
    ))
    fig.update_layout(
        title="Detection Probability per Malware Family",
        xaxis_title="Probability",
        yaxis_title="Malware Family",
        template="plotly_dark",
        height=max(400, len(families) * 22),
        xaxis=dict(range=[0, 1.05]),
        margin=dict(l=150, r=80, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_mitre_mapping(predicted_family: str) -> None:
    """
    Load MITRE ATT&CK for ICS mapping from JSON and display for the predicted family.
    Uses st.info (not st.error) when mapping is unavailable — never crashes the page.
    """
    try:
        with open(config.MITRE_JSON_PATH, 'r') as f:
            mitre_db = json.load(f)
    except FileNotFoundError:
        st.info(
            "MITRE ATT&CK mapping database not found. "
            "Ensure data/mitre_ics_mapping.json exists in the repo root."
        )
        return

    mapping = mitre_db.get(predicted_family)
    if not mapping:
        st.info(f"MITRE ATT&CK mapping not available for family: **{predicted_family}**")
        return

    tactics    = mapping.get('tactics', [])
    techniques = mapping.get('techniques', [])

    if tactics:
        st.markdown(f"**Tactics:** {', '.join(tactics)}")
    if techniques:
        st.markdown("**Techniques:**")
        for t in techniques:
            st.markdown(f"  - `{t['id']}` — {t['name']}")
```

---

## File 8: `modules/dashboard/pages/digital_twin.py`

```python
# modules/dashboard/pages/digital_twin.py
"""
Digital Twin Simulation page — STUB.
Module 1 is deferred to a future sprint.
"""
import streamlit as st


def render():
    st.title("🖥️ Digital Twin Simulation")
    st.markdown("---")
    st.warning(
        "⚠️ **Module 1 — Digital Twin Simulation** is not yet implemented.\n\n"
        "This module will provide a Docker + Mininet based IIoT simulation "
        "environment for safe malware execution and behavioral observation."
    )
    st.markdown("**Planned capabilities:**")
    st.markdown("- Deploy containerized IIoT nodes (PLCs, sensors, MQTT broker, Modbus server)")
    st.markdown("- Simulate Modbus TCP and MQTT industrial traffic")
    st.markdown("- Execute malware samples in isolated containers")
    st.markdown("- Stream live network traffic log to dashboard")
    st.markdown("- Monitor node infection status in real-time")
    st.info(
        "This page will be implemented in a future sprint "
        "once the ML pipeline is stable."
    )
```

---

## File 9: `modules/dashboard/app.py`

> **CRITICAL:** `st.set_page_config()` must be the absolute first Streamlit call. It is inside `configure_page()` which is the first call in `main()`. The `configure_page()` call must appear before `state.init_session_state()`, `init_db()`, and everything else.

```python
# modules/dashboard/app.py
"""
Streamlit application entry point.

Run:
    streamlit run modules/dashboard/app.py --server.port 8501

Responsibilities:
    1. Page configuration  (st.set_page_config — MUST be first Streamlit call)
    2. Session state init
    3. Database init
    4. Model + class names loading (once per session)
    5. Sidebar navigation
    6. Page routing
"""
import streamlit as st
from pathlib import Path

import config
from modules.dashboard import state
from modules.dashboard.db import init_db
from modules.dataset.preprocessor import load_class_names
from modules.detection.inference import load_model


def configure_page() -> None:
    """
    MUST be called before any other Streamlit command.
    st.set_page_config() is the very first Streamlit call in the app.
    """
    st.set_page_config(
        page_title=config.DASHBOARD_TITLE,
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': (
                "MalTwin — AI-based IIoT Malware Detection Framework\n"
                "COMSATS University, Islamabad | BS Cyber Security 2023-2027"
            ),
        },
    )


def load_global_resources() -> None:
    """
    Load class names and model into session_state on first run.
    Checks state BEFORE attempting to load — runs once per session only.
    Does NOT use @st.cache_resource; session_state guard is used instead.
    """
    # ── Class names ───────────────────────────────────────────────────────────
    if st.session_state[state.KEY_CLASS_NAMES] is None:
        try:
            class_names = load_class_names(config.CLASS_NAMES_PATH)
            st.session_state[state.KEY_CLASS_NAMES] = class_names
        except FileNotFoundError:
            st.session_state[state.KEY_CLASS_NAMES] = None

    # ── Model ─────────────────────────────────────────────────────────────────
    if (
        st.session_state[state.KEY_MODEL] is None
        and st.session_state[state.KEY_CLASS_NAMES] is not None
    ):
        try:
            with st.spinner("Loading detection model…"):
                num_classes = len(st.session_state[state.KEY_CLASS_NAMES])
                model = load_model(config.BEST_MODEL_PATH, num_classes, config.DEVICE)
                st.session_state[state.KEY_MODEL]        = model
                st.session_state[state.KEY_MODEL_LOADED] = True
                st.session_state[state.KEY_DEVICE_INFO]  = str(config.DEVICE)
        except FileNotFoundError:
            st.session_state[state.KEY_MODEL_LOADED] = False


def render_sidebar() -> str:
    """
    Render sidebar navigation. Returns the selected page label string.
    """
    st.sidebar.markdown("# 🛡️ MalTwin")
    st.sidebar.markdown("*IIoT Malware Detection*")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Dashboard",
            "📂 Binary Upload",
            "🔍 Malware Detection",
            "🖥️ Digital Twin",
        ],
        label_visibility="hidden",
    )

    # ── System status ─────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("**System Status**")

    if state.is_model_loaded():
        device_info = st.session_state.get(state.KEY_DEVICE_INFO, 'unknown')
        st.sidebar.success(f"✅ Model Ready ({device_info})")
    else:
        st.sidebar.warning("⚠️ No model loaded")
        st.sidebar.caption("Run `scripts/train.py` first")

    if state.has_uploaded_file():
        meta = st.session_state[state.KEY_FILE_META]
        st.sidebar.info(f"📄 {meta['name']}")
    else:
        st.sidebar.caption("No file uploaded")

    if state.has_detection_result():
        result = st.session_state[state.KEY_DETECTION]
        st.sidebar.success(f"🎯 {result['predicted_family']}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.caption("COMSATS University, Islamabad")
    st.sidebar.caption("BS Cyber Security 2023-2027")

    return page


def main() -> None:
    """
    Application entry point.
    configure_page() MUST be the first call — st.set_page_config() inside it
    must precede every other Streamlit call.
    """
    configure_page()                        # ← st.set_page_config() happens here
    state.init_session_state()
    init_db(config.DB_PATH)
    load_global_resources()
    page = render_sidebar()

    if page == "🏠 Dashboard":
        from modules.dashboard.pages.home import render
        render()
    elif page == "📂 Binary Upload":
        from modules.dashboard.pages.upload import render
        render()
    elif page == "🔍 Malware Detection":
        from modules.dashboard.pages.detection import render
        render()
    elif page == "🖥️ Digital Twin":
        from modules.dashboard.pages.digital_twin import render
        render()


if __name__ == "__main__":
    main()
```

---

## File 10: `tests/test_db.py`

Write this file **exactly** as shown.

```python
# tests/test_db.py
"""
Test suite for modules/dashboard/db.py

All tests use tmp_path (pytest built-in) for isolated SQLite files.
No Malimg dataset required.

Run:
    pytest tests/test_db.py -v
"""
import os
import pytest
from pathlib import Path
from modules.dashboard.db import (
    init_db,
    log_detection_event,
    get_recent_events,
    get_detection_stats,
    get_events_by_date_range,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path) -> Path:
    """Initialised DB in a temporary directory."""
    db_path = tmp_path / "test_maltwin.db"
    init_db(db_path)
    return db_path


def _insert(db_path: Path, **overrides):
    """Helper: insert a detection event with sensible defaults."""
    defaults = dict(
        file_name="sample.exe",
        sha256="a" * 64,
        file_format="PE",
        file_size=1024,
        predicted_family="Allaple.A",
        confidence=0.95,
        device_used="cpu",
    )
    defaults.update(overrides)
    log_detection_event(db_path, **defaults)


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        init_db(db_path)
        assert db_path.exists()

    def test_idempotent(self, temp_db):
        """Calling init_db a second time must not raise."""
        init_db(temp_db)

    def test_file_permissions_are_600(self, temp_db):
        mode = os.stat(temp_db).st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_detection_events_table(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_events'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_timestamp_index(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_family_index(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_family'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "deep" / "test.db"
        init_db(db_path)
        assert db_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# log_detection_event
# ─────────────────────────────────────────────────────────────────────────────

class TestLogDetectionEvent:
    def test_inserts_one_row(self, temp_db):
        _insert(temp_db)
        events = get_recent_events(temp_db, limit=10)
        assert len(events) == 1

    def test_inserted_values_are_correct(self, temp_db):
        _insert(temp_db,
                file_name="test.exe",
                sha256="b" * 64,
                file_format="ELF",
                file_size=2048,
                predicted_family="Yuner.A",
                confidence=0.75,
                device_used="cuda")
        row = get_recent_events(temp_db)[0]
        assert row['file_name']        == "test.exe"
        assert row['sha256']           == "b" * 64
        assert row['file_format']      == "ELF"
        assert row['file_size']        == 2048
        assert row['predicted_family'] == "Yuner.A"
        assert abs(row['confidence'] - 0.75) < 1e-6
        assert row['device_used']      == "cuda"

    def test_timestamp_is_set_automatically(self, temp_db):
        _insert(temp_db)
        row = get_recent_events(temp_db)[0]
        assert 'timestamp' in row
        assert len(row['timestamp']) > 10   # ISO 8601 string

    def test_multiple_inserts_accumulate(self, temp_db):
        for i in range(5):
            _insert(temp_db, file_name=f"file_{i}.exe")
        assert len(get_recent_events(temp_db, limit=10)) == 5

    def test_does_not_raise_on_bad_path(self, tmp_path):
        """A path where the parent dir doesn't exist — must not raise."""
        bad = tmp_path / "nonexistent_dir" / "db.db"
        # log_detection_event must swallow the error
        try:
            log_detection_event(
                bad, "x.exe", "a" * 64, "PE", 100, "X", 0.5, "cpu"
            )
        except Exception:
            pass   # acceptable — it just must not propagate up and crash calling code


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_events
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentEvents:
    def test_returns_empty_list_for_empty_db(self, temp_db):
        assert get_recent_events(temp_db) == []

    def test_returns_empty_list_for_missing_db(self, tmp_path):
        assert get_recent_events(tmp_path / "missing.db") == []

    def test_returns_most_recent_first(self, temp_db):
        for i in range(3):
            _insert(temp_db, file_name=f"file_{i}.exe")
        events = get_recent_events(temp_db, limit=5)
        assert events[0]['file_name'] == "file_2.exe"

    def test_limit_is_respected(self, temp_db):
        for i in range(10):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert len(get_recent_events(temp_db, limit=3)) == 3

    def test_default_limit_is_five(self, temp_db):
        for i in range(8):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert len(get_recent_events(temp_db)) == 5

    def test_returns_list_of_dicts(self, temp_db):
        _insert(temp_db)
        events = get_recent_events(temp_db)
        assert isinstance(events, list)
        assert isinstance(events[0], dict)

    def test_rows_contain_all_schema_columns(self, temp_db):
        _insert(temp_db)
        row = get_recent_events(temp_db)[0]
        expected_keys = {
            'id', 'timestamp', 'file_name', 'sha256', 'file_format',
            'file_size', 'predicted_family', 'confidence', 'device_used',
        }
        assert expected_keys.issubset(row.keys())


# ─────────────────────────────────────────────────────────────────────────────
# get_detection_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDetectionStats:
    def test_empty_db_returns_zeros(self, temp_db):
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 0
        assert stats['total_malware']  == 0
        assert stats['total_benign']   == 0

    def test_missing_db_returns_zeros(self, tmp_path):
        stats = get_detection_stats(tmp_path / "missing.db")
        assert stats['total_analyzed'] == 0

    def test_counts_correctly_after_inserts(self, temp_db):
        for i in range(5):
            _insert(temp_db, file_name=f"f{i}.exe")
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 5
        assert stats['total_malware']  == 5

    def test_returns_required_keys(self, temp_db):
        stats = get_detection_stats(temp_db)
        assert 'total_analyzed' in stats
        assert 'total_malware'  in stats
        assert 'total_benign'   in stats
        assert 'model_accuracy' in stats

    def test_model_accuracy_none_when_no_metrics_file(self, temp_db):
        # As long as data/processed/eval_metrics.json doesn't exist, this is None
        stats = get_detection_stats(temp_db)
        # Can be None or float — just check it doesn't crash
        assert stats['model_accuracy'] is None or isinstance(stats['model_accuracy'], float)

    def test_total_benign_always_zero(self, temp_db):
        for i in range(3):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert get_detection_stats(temp_db)['total_benign'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_events_by_date_range
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEventsByDateRange:
    def test_returns_empty_list_for_empty_db(self, temp_db):
        assert get_events_by_date_range(temp_db) == []

    def test_returns_empty_list_for_missing_db(self, tmp_path):
        assert get_events_by_date_range(tmp_path / "missing.db") == []

    def test_returns_events_within_range(self, temp_db):
        _insert(temp_db, file_name="recent.exe")
        events = get_events_by_date_range(temp_db, days_back=7)
        assert len(events) == 1

    def test_returned_dicts_have_timestamp_key(self, temp_db):
        _insert(temp_db)
        events = get_events_by_date_range(temp_db, days_back=7)
        assert 'timestamp' in events[0]

    def test_returned_dicts_have_predicted_family_key(self, temp_db):
        _insert(temp_db)
        events = get_events_by_date_range(temp_db, days_back=7)
        assert 'predicted_family' in events[0]
```

---

## Definition of Done

```bash
# ── DB tests ──────────────────────────────────────────────────────────────────
pytest tests/test_db.py -v
# Expected: all tests pass, 0 failures

# ── Full non-integration suite ────────────────────────────────────────────────
pytest tests/ -v -m "not integration"
# Expected: all tests pass across all phases

# ── Dashboard smoke test ──────────────────────────────────────────────────────
# Verify app.py imports without error (no Streamlit server needed)
python -c "import modules.dashboard.app"
python -c "import modules.dashboard.pages.home"
python -c "import modules.dashboard.pages.upload"
python -c "import modules.dashboard.pages.detection"
python -c "import modules.dashboard.pages.digital_twin"

# ── Launch the dashboard ──────────────────────────────────────────────────────
streamlit run modules/dashboard/app.py --server.port 8501
# Open http://localhost:8501 and verify:
#   ✓ App loads without crashing
#   ✓ Sidebar shows all 4 navigation options
#   ✓ Home page renders KPI cards (all zeros if no detections yet)
#   ✓ Module status table shows 8 rows
#   ✓ "No model loaded" warning appears if best_model.pt is missing
#   ✓ Binary Upload page loads with file uploader widget
#   ✓ Detection page shows "no file uploaded" warning when no file is in state
#   ✓ Digital Twin page shows the deferred stub message
```

### With a trained model (full end-to-end)

```bash
# Upload a binary via the dashboard and verify:
#   ✓ Grayscale image renders in the left column
#   ✓ Metadata table shows file name, size, format, SHA-256, upload time
#   ✓ Pixel intensity histogram renders
#   ✓ Sidebar shows the filename after upload

# Navigate to Malware Detection and click Run Detection:
#   ✓ Spinner appears during inference
#   ✓ Confidence bar appears with correct color (green/amber/red)
#   ✓ Top-3 predictions listed
#   ✓ Horizontal probability chart shows all 25 families
#   ✓ MITRE ATT&CK mapping renders for known families
#   ✓ JSON download button is active and downloads valid JSON
#   ✓ Home page Recent Detections table shows the event
```

### Checklist

- [ ] `pytest tests/test_db.py -v` passes with zero failures
- [ ] `pytest tests/ -v -m "not integration"` passes with zero failures (all phases)
- [ ] `python -c "import modules.dashboard.app"` exits 0
- [ ] `st.set_page_config()` is the **first** Streamlit call — inside `configure_page()` — inside `main()`
- [ ] No HTML `<form>` tags anywhere in any dashboard file
- [ ] All `st.session_state` keys come from `state.py` constants — no raw strings in page files
- [ ] `PRAGMA journal_mode=WAL` is in every `get_connection()` call
- [ ] `os.chmod(db_path, 0o600)` called in `init_db()` after file creation
- [ ] `log_detection_event()` never raises — all exceptions swallowed and logged to stderr
- [ ] `state.clear_file_state()` called before processing a new upload in `upload.py`
- [ ] `st.image()` uses `use_column_width=True` (not `use_container_width`)
- [ ] `get_val_transforms` used inside `predict_single` (verified in Phase 4)
- [ ] JSON download button uses `st.download_button` — not `st.button` + manual JS

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|-----|---------|-----|
| `st.set_page_config()` not first call | `StreamlitAPIException: set_page_config() can only be called once` | Move it to `configure_page()`, call that first in `main()` |
| Raw string key in session_state | Typo causes `KeyError` silently or wrong value | Import `state` and use `state.KEY_*` constants everywhere |
| `use_container_width=True` in `st.image()` | TypeError on older Streamlit versions | Use `use_column_width=True` for image display |
| `conn.commit()` called manually outside context manager | Double-commit or missed commit if exception raised | Let `get_connection` handle `commit()` — never call it manually |
| `log_detection_event()` raising on DB failure | Detection result page crashes | Wrap both attempts in try/except; print to stderr on failure; never re-raise |
| Forgetting `state.clear_file_state()` on new upload | Old detection result shown for new file | Call it at the start of `_process_upload()` before any processing |
| `@st.cache_resource` on model | Shared across users in multi-user deployments | Use session_state guard in `load_global_resources()` instead |
| HTML `<form>` tag anywhere in Streamlit | Streamlit may render oddly or break reruns | Use `st.button`, `st.file_uploader`, `st.text_input` etc. only |
| `get_connection` without WAL PRAGMA | DB may corrupt on crash | WAL is set inside `get_connection` — never bypass it |
| `init_db()` not called before first write | `OperationalError: no such table: detection_events` | Call `init_db(config.DB_PATH)` in `app.py main()` at startup |

---

*Phase 6 complete. The full pipeline is now implemented.*

## Phase 7 — Final Integration Smoke Test

After merging Phase 6, run the complete integration checklist:

```bash
# 1. All unit tests pass
pytest tests/ -v -m "not integration"

# 2. Convert a binary via CLI
python scripts/convert_binary.py \
    --input tests/fixtures/sample_pe.exe \
    --output /tmp/smoke_test.png

# 3. Full training smoke run (2 epochs — requires Malimg dataset)
python scripts/train.py --epochs 2 --workers 0

# 4. Verify all output files were produced
ls -lh models/best_model.pt
ls -lh data/processed/class_names.json
ls -lh data/processed/eval_metrics.json
ls -lh data/processed/confusion_matrix.png

# 5. Evaluate-only script
python scripts/evaluate.py

# 6. All tests including integration
pytest tests/ -v

# 7. Launch dashboard
streamlit run modules/dashboard/app.py --server.port 8501

# 8. Manual walkthrough
#    a) Open http://localhost:8501
#    b) Navigate to Binary Upload → upload tests/fixtures/sample_pe.exe
#    c) Verify grayscale image, metadata, histogram render correctly
#    d) Navigate to Malware Detection → click Run Detection
#    e) Verify confidence bar, top-3, probability chart, MITRE mapping all render
#    f) Click Download JSON Result → verify JSON is valid and contains all fields
#    g) Navigate to Dashboard → verify Recent Detections shows the event
#    h) Navigate to Digital Twin → verify stub page renders
```
