# MalTwin — Implementation Step 3: Dataset Gallery + Detection History Filtering
### SRS refs: FR1.4, M6 FE-5, M8 FE-5

> Complete Steps 1 and 2 first. This step has no dependency on Grad-CAM or
> the reporting pipeline, but the full regression suite must be green before
> starting here.

---

## What This Step Delivers

| Item | Status before | Status after |
|---|---|---|
| `modules/dashboard/pages/gallery.py` | Does not exist | Full dataset gallery page |
| `modules/dashboard/app.py` | 4-page routing | 5-page routing with gallery |
| `modules/dashboard/pages/home.py` | Last-5 events, no filter | Filterable history table + expanded query |
| `modules/dashboard/db.py` | `get_recent_events(limit=5)` only | `get_filtered_events()` + `get_family_list()` added |
| `tests/test_gallery.py` | Does not exist | Gallery unit tests |
| `tests/test_db.py` | Existing suite | `TestGetFilteredEvents` + `TestGetFamilyList` added |

---

## Mandatory Rules

- The gallery page loads images **directly from `config.DATA_DIR`** using `pathlib.Path` — it never calls a DataLoader or imports PyTorch.
- Images are loaded with `PIL.Image.open()` in mode `'L'` then displayed via `st.image()`.
- The gallery renders a **grid of columns** using `st.columns()` — never a single-column list.
- Gallery page gracefully handles missing dataset: shows an info message, never crashes.
- `get_filtered_events()` uses **parameterised SQL** only — no string formatting into queries.
- All new DB functions return empty list on any error — never raise.
- `get_family_list()` returns families sorted alphabetically, with `"All Families"` prepended.
- Detection history filter state lives in `st.session_state` — not URL params.
- The gallery and history pages use `@st.cache_data` with a `ttl` for expensive disk reads.
- No new session_state keys are needed — history filters are local widget state.

---

## File 1: Add two functions to `modules/dashboard/db.py`

Add these functions at the bottom of the existing `db.py`. Do not modify any existing functions.

```python
def get_filtered_events(
    db_path: Path,
    family_filter: str | None = None,
    min_confidence: float = 0.0,
    days_back: int | None = None,
    limit: int = 100,
    sort_desc: bool = True,
) -> list[dict]:
    """
    Return detection events with optional filtering and sorting.

    Args:
        db_path:        Path to SQLite database.
        family_filter:  If not None and not 'All Families', filter by predicted_family.
        min_confidence: Minimum confidence threshold (0.0 = no filter).
        days_back:      If not None, only return events from last N days.
        limit:          Maximum number of rows to return (default 100).
        sort_desc:      If True, newest first. If False, oldest first.

    Returns:
        list[dict] — one dict per row, all schema columns present.
        Empty list if DB does not exist or on any error.

    Implementation:
        Build WHERE clauses dynamically, always using ? placeholders.
        Never use f-strings or % formatting in SQL.
    """
    if not db_path.exists():
        return []
    try:
        clauses = []
        params  = []

        if family_filter and family_filter != 'All Families':
            clauses.append('predicted_family = ?')
            params.append(family_filter)

        if min_confidence > 0.0:
            clauses.append('confidence >= ?')
            params.append(min_confidence)

        if days_back is not None:
            from datetime import datetime, timedelta
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
            clauses.append('timestamp >= ?')
            params.append(cutoff)

        where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
        order = 'DESC' if sort_desc else 'ASC'
        sql   = (
            f"SELECT * FROM detection_events "
            f"{where} "
            f"ORDER BY id {order} "
            f"LIMIT ?"
        )
        params.append(limit)

        with get_connection(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    except Exception:
        return []


def get_family_list(db_path: Path) -> list[str]:
    """
    Return a sorted list of all distinct malware families in the detection log,
    with 'All Families' prepended as the default option.

    Returns:
        ['All Families', 'Allaple.A', 'Rbot!gen', ...]
        ['All Families'] if DB is empty or does not exist.
    """
    if not db_path.exists():
        return ['All Families']
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT predicted_family "
                "FROM detection_events "
                "ORDER BY predicted_family ASC"
            ).fetchall()
        families = [row[0] for row in rows]
        return ['All Families'] + families
    except Exception:
        return ['All Families']
```

---

## File 2: `modules/dashboard/pages/gallery.py`

```python
# modules/dashboard/pages/gallery.py
"""
Dataset Gallery page — per-family sample image grid.

SRS refs: M6 FE-5
Displays sample grayscale images from each of the 25 Malimg malware families
directly from config.DATA_DIR. No PyTorch or DataLoader involved.

Layout:
    - Sidebar family selector (radio or selectbox)
    - Main area: family info header + NxM image grid
    - Grid columns configurable via a slider (default 4)
    - Max images per family configurable (default 12)
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np

import config


@st.cache_data(ttl=300, show_spinner=False)
def _load_family_names(data_dir: str) -> list[str]:
    """
    Scan data_dir for subdirectory names (one per family).
    Cached for 5 minutes — avoids repeated disk scans.
    Returns sorted list of family name strings.
    Returns [] if data_dir does not exist.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


@st.cache_data(ttl=300, show_spinner=False)
def _load_sample_images(
    data_dir: str,
    family_name: str,
    max_images: int = 12,
) -> list[np.ndarray]:
    """
    Load up to max_images PNG images from data_dir/family_name/.
    Returns list of uint8 numpy arrays shape (H, W).
    Returns [] if directory is missing or has no PNGs.
    Cached per (data_dir, family_name, max_images).
    """
    family_dir = Path(data_dir) / family_name
    if not family_dir.exists():
        return []

    png_files = sorted(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))
    png_files = png_files[:max_images]

    images = []
    for path in png_files:
        try:
            img = Image.open(path).convert('L')      # grayscale
            images.append(np.array(img))
        except Exception:
            continue   # skip corrupt files silently
    return images


@st.cache_data(ttl=300, show_spinner=False)
def _count_family_images(data_dir: str, family_name: str) -> int:
    """Return total PNG count for a family (not capped at max_images)."""
    family_dir = Path(data_dir) / family_name
    if not family_dir.exists():
        return 0
    return len(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))


def render():
    st.title("🖼️ Dataset Gallery")
    st.markdown(
        "Browse sample grayscale images from each malware family in the Malimg dataset. "
        "Each image is the raw byte structure of a malware binary visualised as a "
        "128×128 greyscale image. Distinctive textures and patterns are visible per family."
    )
    st.markdown("---")

    # ── Dataset availability check ────────────────────────────────────────────
    family_names = _load_family_names(str(config.DATA_DIR))

    if not family_names:
        st.info(
            "📂 Dataset not found. "
            f"The Malimg dataset was not detected at `{config.DATA_DIR}`. "
            "Download it from Kaggle and extract it so that each malware family "
            "has its own subfolder under `data/malimg/`."
        )
        st.markdown("**Expected structure:**")
        st.code(
            "data/malimg/\n"
            "├── Allaple.A/\n"
            "│   ├── 00a5c6a6.png\n"
            "│   └── ...\n"
            "├── Rbot!gen/\n"
            "└── ...",
            language=None,
        )
        return

    # ── Controls sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🖼️ Gallery Controls")
        st.divider()

        selected_family = st.selectbox(
            "Malware Family",
            options=family_names,
            index=0,
            help="Select a malware family to view sample images.",
        )

        max_images = st.slider(
            "Max images to show",
            min_value=4,
            max_value=24,
            value=12,
            step=4,
            help="Maximum number of sample images displayed per family.",
        )

        n_cols = st.slider(
            "Grid columns",
            min_value=2,
            max_value=6,
            value=4,
            help="Number of columns in the image grid.",
        )

        st.divider()
        st.markdown(f"**{len(family_names)} families** in dataset")

    # ── Family overview strip (all families mini-preview) ─────────────────────
    st.subheader("All Families — Quick Overview")
    st.caption(
        "One representative image per family. Click the family selector in the sidebar "
        "for a full grid."
    )

    _render_overview_strip(family_names, str(config.DATA_DIR))

    st.markdown("---")

    # ── Selected family detail grid ───────────────────────────────────────────
    total_count = _count_family_images(str(config.DATA_DIR), selected_family)
    st.subheader(f"📁 {selected_family}")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Samples", total_count)
    col_info2.metric("Showing", min(max_images, total_count))
    col_info3.metric("Image Size", "128 × 128 px")

    # MITRE context for selected family
    try:
        from modules.reporting.mitre_mapper import get_mitre_mapping
        mitre = get_mitre_mapping(selected_family)
        if mitre['found']:
            with st.expander("MITRE ATT&CK for ICS Context", expanded=False):
                st.markdown(f"**Description:** {mitre['description']}")
                st.markdown(f"**Tactics:** {', '.join(mitre['tactics'])}")
                for tech in mitre['techniques']:
                    st.markdown(f"- `{tech['id']}` — {tech['name']}")
    except Exception:
        pass   # MITRE context is a nice-to-have; never crash the gallery

    st.markdown("---")

    with st.spinner(f"Loading {min(max_images, total_count)} images…"):
        images = _load_sample_images(str(config.DATA_DIR), selected_family, max_images)

    if not images:
        st.warning(f"No PNG images found in `data/malimg/{selected_family}/`.")
        return

    _render_image_grid(images, selected_family, n_cols)


def _render_overview_strip(family_names: list[str], data_dir: str) -> None:
    """
    Render one representative image per family in a horizontal scrolling strip.
    Uses 8 columns per row, wraps to next row automatically.
    """
    STRIP_COLS = 8
    chunks = [
        family_names[i:i + STRIP_COLS]
        for i in range(0, len(family_names), STRIP_COLS)
    ]

    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, family in zip(cols, chunk):
            images = _load_sample_images(data_dir, family, max_images=1)
            with col:
                if images:
                    st.image(
                        images[0],
                        caption=family,
                        use_column_width=True,
                        clamp=True,
                    )
                else:
                    st.caption(family)
                    st.markdown("_(no image)_")


def _render_image_grid(
    images: list[np.ndarray],
    family_name: str,
    n_cols: int,
) -> None:
    """
    Render a grid of grayscale images with filenames as captions.
    Fills rows left-to-right, wraps at n_cols.
    """
    for row_start in range(0, len(images), n_cols):
        row_images = images[row_start:row_start + n_cols]
        cols = st.columns(n_cols)
        for col, img_array in zip(cols, row_images):
            with col:
                st.image(
                    img_array,
                    caption=f"{family_name} #{row_start + cols.index(col) + 1}",
                    use_column_width=True,
                    clamp=True,
                )
```

---

## File 3: Update `modules/dashboard/pages/home.py`

### 3a — Replace `_render_recent_detections()` section

Find and replace the entire `# ── Recent Detection Feed` block in `render()`:

```python
    # ── Recent Detection Feed (with filters) ──────────────────────────────────
    st.subheader("Detection History")
    _render_history_section()
```

### 3b — Add `_render_history_section()` function

Add this new function at the bottom of `home.py`:

```python
def _render_history_section() -> None:
    """
    Filterable, sortable detection history table.
    SRS refs: FR1.4, M8 FE-5
    """
    import pandas as pd
    from modules.dashboard.db import get_filtered_events, get_family_list

    # ── Filter controls ───────────────────────────────────────────────────────
    with st.expander("🔍 Filter & Sort", expanded=False):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            family_options = get_family_list(config.DB_PATH)
            family_filter = st.selectbox(
                "Malware Family",
                options=family_options,
                index=0,
                key="history_family_filter",
            )

        with col_f2:
            min_conf = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                format="%.0f%%",
                key="history_min_conf",
            )

        with col_f3:
            days_options = {
                "All time":    None,
                "Last 7 days":  7,
                "Last 30 days": 30,
                "Last 90 days": 90,
            }
            days_label = st.selectbox(
                "Time Range",
                options=list(days_options.keys()),
                index=0,
                key="history_days",
            )
            days_back = days_options[days_label]

        with col_f4:
            sort_desc = st.radio(
                "Sort Order",
                options=["Newest first", "Oldest first"],
                index=0,
                key="history_sort",
            ) == "Newest first"

        limit = st.slider(
            "Max rows",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            key="history_limit",
        )

    # ── Fetch events ──────────────────────────────────────────────────────────
    events = get_filtered_events(
        db_path=config.DB_PATH,
        family_filter=family_filter if family_filter != 'All Families' else None,
        min_confidence=min_conf,
        days_back=days_back,
        limit=limit,
        sort_desc=sort_desc,
    )

    if not events:
        st.caption(
            "No detections match the current filters. "
            "Upload a binary file on the Binary Upload page to get started."
        )
        return

    # ── Build display dataframe ───────────────────────────────────────────────
    df = pd.DataFrame(events)

    # Format columns for display
    df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
    df['timestamp']      = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    display_df = df[[
        'timestamp', 'file_name', 'predicted_family',
        'confidence_pct', 'file_format', 'device_used',
    ]].copy()
    display_df.columns = [
        'Timestamp', 'File', 'Predicted Family',
        'Confidence', 'Format', 'Device',
    ]

    # ── Summary metrics ───────────────────────────────────────────────────────
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rows shown", len(events))
    col_m2.metric(
        "Avg confidence",
        f"{df['confidence'].mean() * 100:.1f}%",
    )
    col_m3.metric(
        "Unique families",
        df['predicted_family'].nunique(),
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export to CSV",
        data=csv_bytes,
        file_name="maltwin_detection_history.csv",
        mime="text/csv",
        help="Download the filtered detection history as a CSV file.",
    )
```

---

## File 4: Update `modules/dashboard/app.py`

### 4a — Add gallery to sidebar navigation

In `render_sidebar()`, replace the `options` list in `st.sidebar.radio()`:

```python
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Dashboard",
            "📂 Binary Upload",
            "🔍 Malware Detection",
            "🖼️ Dataset Gallery",
            "🖥️ Digital Twin",
        ],
        label_visibility="hidden",
    )
```

### 4b — Add gallery route in `main()`

Add the gallery branch to the `if/elif` routing block:

```python
    elif page == "🖼️ Dataset Gallery":
        from modules.dashboard.pages.gallery import render
        render()
```

---

## File 5: `tests/test_gallery.py`

```python
"""
Test suite for modules/dashboard/pages/gallery.py
and the new db.py functions: get_filtered_events, get_family_list.

All tests use tmp_path — no real Malimg dataset required.

Run:
    pytest tests/test_gallery.py -v
"""
import os
import cv2
import pytest
import numpy as np
from pathlib import Path

from modules.dashboard.db import (
    init_db,
    log_detection_event,
    get_filtered_events,
    get_family_list,
)
from modules.dashboard.pages.gallery import (
    _load_family_names,
    _load_sample_images,
    _count_family_images,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture
def populated_db(temp_db):
    """DB with 10 events across 3 families."""
    families = ['Allaple.A', 'Rbot!gen', 'VB.AT']
    confs    = [0.95, 0.72, 0.45, 0.88, 0.61, 0.90, 0.55, 0.83, 0.40, 0.77]
    for i, conf in enumerate(confs):
        log_detection_event(
            temp_db,
            file_name=f"file_{i:02d}.exe",
            sha256='a' * 64,
            file_format='PE',
            file_size=1024,
            predicted_family=families[i % 3],
            confidence=conf,
            device_used='cpu',
        )
    return temp_db


@pytest.fixture
def fake_dataset(tmp_path):
    """Fake Malimg-style directory: 3 families × 5 PNGs each."""
    for family in ['Allaple.A', 'Rbot!gen', 'VB.AT']:
        d = tmp_path / family
        d.mkdir()
        for i in range(5):
            img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            cv2.imwrite(str(d / f'img_{i:03d}.png'), img)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# get_filtered_events
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFilteredEvents:
    def test_returns_all_when_no_filters(self, populated_db):
        events = get_filtered_events(populated_db)
        assert len(events) == 10

    def test_returns_empty_for_missing_db(self, tmp_path):
        events = get_filtered_events(tmp_path / "missing.db")
        assert events == []

    def test_family_filter_works(self, populated_db):
        events = get_filtered_events(populated_db, family_filter='Allaple.A')
        assert all(e['predicted_family'] == 'Allaple.A' for e in events)
        assert len(events) > 0

    def test_family_filter_all_families_returns_everything(self, populated_db):
        events = get_filtered_events(populated_db, family_filter='All Families')
        assert len(events) == 10

    def test_confidence_filter_excludes_low_confidence(self, populated_db):
        events = get_filtered_events(populated_db, min_confidence=0.80)
        assert all(e['confidence'] >= 0.80 for e in events)

    def test_confidence_filter_zero_returns_all(self, populated_db):
        events = get_filtered_events(populated_db, min_confidence=0.0)
        assert len(events) == 10

    def test_limit_is_respected(self, populated_db):
        events = get_filtered_events(populated_db, limit=3)
        assert len(events) == 3

    def test_sort_desc_newest_first(self, populated_db):
        events = get_filtered_events(populated_db, sort_desc=True)
        ids = [e['id'] for e in events]
        assert ids == sorted(ids, reverse=True)

    def test_sort_asc_oldest_first(self, populated_db):
        events = get_filtered_events(populated_db, sort_desc=False)
        ids = [e['id'] for e in events]
        assert ids == sorted(ids)

    def test_days_back_filters_old_events(self, temp_db):
        """Events from now should appear; simulate via days_back=7."""
        log_detection_event(
            temp_db, 'recent.exe', 'b' * 64, 'PE',
            512, 'Allaple.A', 0.9, 'cpu',
        )
        events = get_filtered_events(temp_db, days_back=7)
        assert len(events) == 1
        assert events[0]['file_name'] == 'recent.exe'

    def test_combined_filters(self, populated_db):
        """Family + confidence filter should AND together."""
        events = get_filtered_events(
            populated_db,
            family_filter='Allaple.A',
            min_confidence=0.90,
        )
        for e in events:
            assert e['predicted_family'] == 'Allaple.A'
            assert e['confidence'] >= 0.90

    def test_returns_list_of_dicts(self, populated_db):
        events = get_filtered_events(populated_db)
        assert isinstance(events, list)
        assert isinstance(events[0], dict)

    def test_rows_contain_all_schema_columns(self, populated_db):
        events = get_filtered_events(populated_db, limit=1)
        required = {
            'id', 'timestamp', 'file_name', 'sha256',
            'file_format', 'file_size', 'predicted_family',
            'confidence', 'device_used',
        }
        assert required.issubset(events[0].keys())

    def test_no_sql_injection_via_family_filter(self, populated_db):
        """Malicious family_filter string must not crash or return wrong rows."""
        events = get_filtered_events(
            populated_db,
            family_filter="'; DROP TABLE detection_events; --",
        )
        # Should return empty list (no such family), not crash
        assert isinstance(events, list)
        # Table must still exist
        remaining = get_filtered_events(populated_db)
        assert len(remaining) == 10


# ─────────────────────────────────────────────────────────────────────────────
# get_family_list
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFamilyList:
    def test_returns_list_with_all_families_prepended(self, populated_db):
        families = get_family_list(populated_db)
        assert families[0] == 'All Families'

    def test_returns_only_all_families_for_empty_db(self, temp_db):
        families = get_family_list(temp_db)
        assert families == ['All Families']

    def test_returns_all_families_for_missing_db(self, tmp_path):
        families = get_family_list(tmp_path / 'missing.db')
        assert families == ['All Families']

    def test_families_are_sorted_alphabetically(self, populated_db):
        families = get_family_list(populated_db)[1:]   # skip 'All Families'
        assert families == sorted(families)

    def test_no_duplicates(self, populated_db):
        families = get_family_list(populated_db)
        assert len(families) == len(set(families))

    def test_detected_families_present(self, populated_db):
        families = get_family_list(populated_db)
        assert 'Allaple.A' in families
        assert 'Rbot!gen' in families
        assert 'VB.AT' in families

    def test_count_matches_distinct_families(self, populated_db):
        # 3 distinct families + 'All Families' header
        families = get_family_list(populated_db)
        assert len(families) == 4


# ─────────────────────────────────────────────────────────────────────────────
# gallery helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestGalleryHelpers:
    def test_load_family_names_returns_sorted_list(self, fake_dataset):
        names = _load_family_names(str(fake_dataset))
        assert names == sorted(names)

    def test_load_family_names_all_three_present(self, fake_dataset):
        names = _load_family_names(str(fake_dataset))
        assert 'Allaple.A' in names
        assert 'Rbot!gen' in names
        assert 'VB.AT' in names

    def test_load_family_names_missing_dir_returns_empty(self, tmp_path):
        names = _load_family_names(str(tmp_path / 'nonexistent'))
        assert names == []

    def test_load_sample_images_returns_arrays(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=5)
        assert len(images) == 5
        assert all(isinstance(img, np.ndarray) for img in images)

    def test_load_sample_images_respects_max_images(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=2)
        assert len(images) == 2

    def test_load_sample_images_missing_family_returns_empty(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'NonexistentFamily', max_images=5)
        assert images == []

    def test_load_sample_images_arrays_are_2d(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=3)
        for img in images:
            assert img.ndim == 2   # grayscale — no channel dimension

    def test_load_sample_images_dtype_uint8(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=3)
        for img in images:
            assert img.dtype == np.uint8

    def test_count_family_images_correct(self, fake_dataset):
        count = _count_family_images(str(fake_dataset), 'Allaple.A')
        assert count == 5

    def test_count_family_images_missing_family_returns_zero(self, fake_dataset):
        count = _count_family_images(str(fake_dataset), 'NoSuchFamily')
        assert count == 0

    def test_corrupt_image_skipped_gracefully(self, fake_dataset):
        """A corrupt PNG in the family dir should be skipped, not crash."""
        corrupt = fake_dataset / 'Allaple.A' / 'corrupt.png'
        corrupt.write_bytes(b'not a real image')
        # Should load the 5 valid images, skip the corrupt one
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=10)
        assert len(images) == 5   # corrupt skipped, 5 valid remain
```

---

## Definition of Done

```bash
# Step 3 tests
pytest tests/test_gallery.py -v
# Expected: all tests pass, 0 failures

# Verify the SQL injection test specifically
pytest tests/test_gallery.py::TestGetFilteredEvents::test_no_sql_injection_via_family_filter -v

# Full regression suite
pytest tests/ -v -m "not integration"
# Expected: 0 failures across all phases + all steps

# Import smoke tests
python -c "from modules.dashboard.pages.gallery import render"
python -c "from modules.dashboard.db import get_filtered_events, get_family_list"

# Dashboard launch — verify 5 pages load
streamlit run modules/dashboard/app.py --server.port 8501
# Navigate to each page and verify:
#   ✓ 🏠 Dashboard — history table with filter expander renders
#   ✓ 🖼️ Dataset Gallery — appears in sidebar
#   ✓ Gallery shows info message when dataset missing (no crash)
#   ✓ Gallery shows overview strip and detail grid when dataset present
```

### Checklist

- [ ] `pytest tests/test_gallery.py -v` — 0 failures
- [ ] All earlier tests still pass
- [ ] `modules/dashboard/pages/gallery.py` exists
- [ ] `modules/dashboard/db.py` has `get_filtered_events()` and `get_family_list()`
- [ ] `app.py` sidebar has 5 navigation options including `"🖼️ Dataset Gallery"`
- [ ] `app.py` routes `"🖼️ Dataset Gallery"` to `gallery.render()`
- [ ] `home.py` shows filterable history table (not just last-5 static rows)
- [ ] History table has Family, Min Confidence, Time Range, Sort Order, Max Rows controls
- [ ] History table has "Export to CSV" download button
- [ ] Gallery page shows info message when `config.DATA_DIR` is missing (no crash)
- [ ] Gallery uses `@st.cache_data(ttl=300)` on disk-read functions
- [ ] Gallery uses `PIL.Image.open().convert('L')` — not cv2, not PyTorch
- [ ] `get_filtered_events()` uses parameterised SQL (`?` placeholders) — no f-strings in SQL
- [ ] `get_filtered_events()` returns empty list on any error — never raises
- [ ] `get_family_list()` returns `['All Families']` when DB empty or missing

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|---|---|---|
| f-string in SQL query | SQL injection vulnerability; also breaks on special chars in family names | Always use `?` placeholders and `params` list |
| `@st.cache_data` on a function that returns PIL Images | Streamlit cache serialisation error | Cache returns numpy arrays or strings only — convert PIL → numpy before returning |
| `img.ndim == 3` for grayscale PIL images | Shape is `(H, W, 1)` instead of `(H, W)` | Use `.convert('L')` then `np.array()` → always `(H, W)` |
| Gallery crashing when dataset missing | `FileNotFoundError` propagates to Streamlit | Check `config.DATA_DIR.exists()` first, show `st.info()` and `return` |
| `_load_family_names` called inside a cached function with `Path` arg | Cache miss every call (Path not hashable consistently) | Pass `str(config.DATA_DIR)` to cached functions, not `Path` objects |
| `st.columns()` count mismatch with image count | IndexError on last row | Use `zip(cols, row_images)` — stops at shorter iterable automatically |
| History filter widget keys colliding with other pages | Widget state bleeds across page navigations | Use unique `key=` strings like `"history_family_filter"` per widget |
| CSV export encoding for non-ASCII family names | Garbled characters in Excel | Use `.encode('utf-8')` and `mime="text/csv"` — works for all family names |

---

*Step 3 complete → move to Step 4: Dynamic module status + FR1.1 live checks + FR1.3 navigation gating.*
