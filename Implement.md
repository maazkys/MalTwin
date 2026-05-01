# MalTwin — Implementation Step 4: Dynamic Module Status + Navigation Gating + System Stats
### SRS refs: FR1.1, FR1.3, FR2.3 (partial), NFR REL-1, NFR USE-2

> Complete Steps 1–3 first. Full regression suite must be green before starting here.
> This step touches `home.py`, `app.py`, and `state.py` — all three are shared
> infrastructure. Be careful not to break existing page routing.

---

## What This Step Delivers

| Item | Status before | Status after |
|---|---|---|
| `modules/dashboard/health.py` | Does not exist | Module health checker — dynamic status for all 8 modules |
| `modules/dashboard/pages/home.py` | Static hardcoded status strings | Live status badges + real system stats (CPU/mem/uptime) |
| `modules/dashboard/app.py` | No gating — all pages always selectable | Navigation gating — unavailable pages show warning, not crash |
| `modules/dashboard/state.py` | No health/uptime keys | `KEY_APP_START_TIME` added for uptime tracking |
| `tests/test_health.py` | Does not exist | Full health checker test suite |

---

## Mandatory Rules

- `health.py` functions **never raise** — every check is wrapped in try/except, returns a status dict.
- Status values are exactly one of: `"active"`, `"inactive"`, `"error"` — no other strings.
- `psutil` is used for CPU/memory — add to `requirements.txt` if not present.
- App start time is stored in `session_state[KEY_APP_START_TIME]` as a `datetime` object set once at startup.
- Navigation gating uses `st.sidebar.radio` with `disabled` entries replaced by a custom rendering approach (Streamlit does not natively support disabled radio options — use `st.sidebar.selectbox` with a note, or render unavailable pages as greyed caption items).
- The status refresh interval is **30 seconds** using `st.cache_data(ttl=30)` — not 5 seconds (FR1.1 says 5s but that would cause constant reruns degrading UX; 30s is the practical implementation).
- System stats (CPU, memory, uptime) replace the "Not Configured" digital twin placeholder on the home page.
- All health checks are **non-blocking** — if a check takes more than 2 seconds it is considered `"error"`.

---

## New `requirements.txt` entry

Add if not already present:
```
psutil>=5.9.0
```

Verify:
```bash
python -c "import psutil; print(psutil.__version__)"
# If missing:
pip install psutil --break-system-packages
```

---

## File 1: `modules/dashboard/health.py`

```python
# modules/dashboard/health.py
"""
Dynamic health checker for all 8 MalTwin modules.

SRS ref: FR1.1 — display operational status of all eight modules,
         refreshed automatically without manual page reload.

Each check returns:
    {
        'status':  'active' | 'inactive' | 'error',
        'detail':  str,   # one-line human-readable explanation
        'emoji':   str,   # ✅ | ⚠️ | 🔴
    }

get_all_module_statuses() is cached with ttl=30 seconds.
Individual check functions never raise.
"""
import sys
import time
import importlib
import streamlit as st
from datetime import datetime
from pathlib import Path

import config


# ── Individual module checks ──────────────────────────────────────────────────

def _check_module1_digital_twin() -> dict:
    """M1: Check if Docker is reachable on the host."""
    try:
        import subprocess
        result = subprocess.run(
            ['docker', 'info'],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            return {
                'status': 'inactive',
                'detail': 'Docker available but Digital Twin not deployed',
                'emoji':  '⚠️',
            }
        return {
            'status': 'inactive',
            'detail': 'Docker not running — Digital Twin unavailable',
            'emoji':  '⚠️',
        }
    except FileNotFoundError:
        return {
            'status': 'inactive',
            'detail': 'Docker not installed — Digital Twin deferred',
            'emoji':  '⚠️',
        }
    except Exception as e:
        return {
            'status': 'inactive',
            'detail': f'Digital Twin check failed: {e}',
            'emoji':  '⚠️',
        }


def _check_module2_binary_to_image() -> dict:
    """M2: Verify the binary_to_image module imports and BinaryConverter is usable."""
    try:
        from modules.binary_to_image.converter import BinaryConverter
        from modules.binary_to_image.utils import validate_binary_format
        # Minimal functional test — convert a tiny byte sequence
        converter = BinaryConverter(img_size=config.IMG_SIZE)
        dummy     = b'MZ' + b'\x00' * 62 + b'\x00' * 900   # minimal PE skeleton
        arr       = converter.convert(dummy)
        assert arr.shape == (config.IMG_SIZE, config.IMG_SIZE)
        return {'status': 'active', 'detail': 'Binary-to-image conversion ready', 'emoji': '✅'}
    except Exception as e:
        return {'status': 'error', 'detail': f'M2 error: {e}', 'emoji': '🔴'}


def _check_module3_dataset() -> dict:
    """M3: Check dataset directory exists and has at least one family subfolder."""
    try:
        if not config.DATA_DIR.exists():
            return {
                'status': 'inactive',
                'detail': f'Dataset not found at {config.DATA_DIR}',
                'emoji':  '⚠️',
            }
        families = [p for p in config.DATA_DIR.iterdir() if p.is_dir()]
        if not families:
            return {
                'status': 'inactive',
                'detail': 'Dataset directory empty — download Malimg',
                'emoji':  '⚠️',
            }
        return {
            'status': 'active',
            'detail': f'Malimg dataset: {len(families)} families found',
            'emoji':  '✅',
        }
    except Exception as e:
        return {'status': 'error', 'detail': f'M3 error: {e}', 'emoji': '🔴'}


def _check_module4_enhancement() -> dict:
    """M4: Verify augmentor and balancer import and transforms build correctly."""
    try:
        from modules.enhancement.augmentor import get_train_transforms, get_val_transforms
        from modules.enhancement.balancer import ClassAwareOversampler
        t = get_train_transforms(config.IMG_SIZE)
        v = get_val_transforms(config.IMG_SIZE)
        assert t is not None and v is not None
        return {'status': 'active', 'detail': 'Enhancement & balancing ready', 'emoji': '✅'}
    except Exception as e:
        return {'status': 'error', 'detail': f'M4 error: {e}', 'emoji': '🔴'}


def _check_module5_detection() -> dict:
    """M5: Check model file exists and loads correctly."""
    try:
        if not config.BEST_MODEL_PATH.exists():
            return {
                'status': 'inactive',
                'detail': 'No trained model — run scripts/train.py',
                'emoji':  '⚠️',
            }
        if not config.CLASS_NAMES_PATH.exists():
            return {
                'status': 'inactive',
                'detail': 'class_names.json missing — run scripts/train.py',
                'emoji':  '⚠️',
            }
        # Check file size is non-trivial (>10KB means a real model, not an empty file)
        size_kb = config.BEST_MODEL_PATH.stat().st_size / 1024
        if size_kb < 10:
            return {
                'status': 'error',
                'detail': f'Model file suspiciously small ({size_kb:.0f} KB)',
                'emoji':  '🔴',
            }
        return {
            'status': 'active',
            'detail': f'Model ready ({size_kb / 1024:.1f} MB)',
            'emoji':  '✅',
        }
    except Exception as e:
        return {'status': 'error', 'detail': f'M5 error: {e}', 'emoji': '🔴'}


def _check_module6_dashboard() -> dict:
    """M6: Dashboard is running if this function is executing — always active."""
    try:
        import streamlit
        return {
            'status': 'active',
            'detail': f'Dashboard running (Streamlit {streamlit.__version__})',
            'emoji':  '✅',
        }
    except Exception as e:
        return {'status': 'error', 'detail': f'M6 error: {e}', 'emoji': '🔴'}


def _check_module7_gradcam() -> dict:
    """M7: Verify Captum is installed and generate_gradcam is importable."""
    try:
        from modules.detection.gradcam import generate_gradcam
        import captum
        if not config.BEST_MODEL_PATH.exists():
            return {
                'status': 'inactive',
                'detail': f'Grad-CAM ready (Captum {captum.__version__}) — awaiting trained model',
                'emoji':  '⚠️',
            }
        return {
            'status': 'active',
            'detail': f'Grad-CAM XAI ready (Captum {captum.__version__})',
            'emoji':  '✅',
        }
    except ImportError:
        return {
            'status': 'error',
            'detail': 'Captum not installed — run: pip install captum',
            'emoji':  '🔴',
        }
    except Exception as e:
        return {'status': 'error', 'detail': f'M7 error: {e}', 'emoji': '🔴'}


def _check_module8_reporting() -> dict:
    """M8: Verify fpdf2 and MITRE JSON are available."""
    try:
        from modules.reporting import generate_pdf_report, generate_json_report
        from fpdf import FPDF

        mitre_ok = config.MITRE_JSON_PATH.exists()
        if not mitre_ok:
            return {
                'status': 'inactive',
                'detail': 'MITRE mapping JSON missing — create data/mitre_ics_mapping.json',
                'emoji':  '⚠️',
            }
        import json
        with open(config.MITRE_JSON_PATH) as f:
            db = json.load(f)
        n = len(db)
        return {
            'status': 'active',
            'detail': f'Reporting ready — MITRE DB: {n}/25 families mapped',
            'emoji':  '✅' if n == 25 else '⚠️',
        }
    except ImportError:
        return {
            'status': 'error',
            'detail': 'fpdf2 not installed — run: pip install fpdf2',
            'emoji':  '🔴',
        }
    except Exception as e:
        return {'status': 'error', 'detail': f'M8 error: {e}', 'emoji': '🔴'}


# ── Aggregate ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_all_module_statuses() -> list[dict]:
    """
    Run all 8 module health checks and return a list of status dicts.
    Cached for 30 seconds — avoids hammering the filesystem on every rerun.

    Returns:
        list of 8 dicts, each with keys:
            'id':     str  e.g. 'M1'
            'name':   str  e.g. 'Digital Twin Simulation'
            'status': str  'active' | 'inactive' | 'error'
            'detail': str
            'emoji':  str
    """
    checks = [
        ('M1', 'Digital Twin Simulation',      _check_module1_digital_twin),
        ('M2', 'Binary-to-Image Conversion',   _check_module2_binary_to_image),
        ('M3', 'Dataset & Preprocessing',      _check_module3_dataset),
        ('M4', 'Data Enhancement & Balancing', _check_module4_enhancement),
        ('M5', 'Intelligent Malware Detection', _check_module5_detection),
        ('M6', 'Dashboard & Visualization',    _check_module6_dashboard),
        ('M7', 'Explainable AI (Grad-CAM)',    _check_module7_gradcam),
        ('M8', 'Automated Threat Reporting',   _check_module8_reporting),
    ]

    results = []
    for mod_id, name, check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = {
                'status': 'error',
                'detail': f'Health check crashed: {e}',
                'emoji':  '🔴',
            }
        results.append({
            'id':     mod_id,
            'name':   name,
            'status': result['status'],
            'detail': result['detail'],
            'emoji':  result['emoji'],
        })

    return results


def get_system_stats() -> dict:
    """
    Return current system resource usage.

    Returns:
        {
            'cpu_pct':    float,  # CPU usage percentage 0–100
            'mem_pct':    float,  # RAM usage percentage 0–100
            'mem_used_gb':float,
            'mem_total_gb':float,
            'uptime_str': str,    # e.g. "2h 34m"
            'device':     str,    # 'cpu' or 'cuda:0'
            'error':      bool,   # True if psutil failed
        }
    """
    try:
        import psutil
        cpu  = psutil.cpu_percent(interval=0.1)
        mem  = psutil.virtual_memory()
        return {
            'cpu_pct':     cpu,
            'mem_pct':     mem.percent,
            'mem_used_gb': mem.used  / (1024 ** 3),
            'mem_total_gb':mem.total / (1024 ** 3),
            'uptime_str':  _format_uptime(),
            'device':      str(config.DEVICE),
            'error':       False,
        }
    except Exception as e:
        return {
            'cpu_pct':     0.0,
            'mem_pct':     0.0,
            'mem_used_gb': 0.0,
            'mem_total_gb':0.0,
            'uptime_str':  'unknown',
            'device':      str(config.DEVICE),
            'error':       True,
        }


def _format_uptime() -> str:
    """
    Compute dashboard uptime from session_state[KEY_APP_START_TIME].
    Returns formatted string e.g. '2h 34m' or '45s'.
    Falls back to 'unknown' if start time not set.
    """
    try:
        from modules.dashboard.state import KEY_APP_START_TIME
        import streamlit as st
        start = st.session_state.get(KEY_APP_START_TIME)
        if start is None:
            return 'unknown'
        delta = datetime.utcnow() - start
        total_seconds = int(delta.total_seconds())
        hours, rem    = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
    except Exception:
        return 'unknown'
```

---

## File 2: Update `modules/dashboard/state.py`

Add the app start time key and update `init_session_state()`:

**Add constant** (after `KEY_DEVICE_INFO`):
```python
KEY_APP_START_TIME = 'app_start_time'   # datetime — set once at first run
```

**Add to `init_session_state()` defaults dict**:
```python
KEY_APP_START_TIME: None,
```

**Add initialisation logic** at the end of `init_session_state()` (after the defaults loop):
```python
    # Set start time once — only on very first run of the session
    if st.session_state[KEY_APP_START_TIME] is None:
        from datetime import datetime
        st.session_state[KEY_APP_START_TIME] = datetime.utcnow()
```

---

## File 3: Update `modules/dashboard/pages/home.py`

### 3a — Replace `_render_module_status()` with dynamic version

Find and remove the old `_render_module_status()` function entirely. Replace it with:

```python
def _render_module_status() -> None:
    """
    Render the live module status table using health.py checks.
    SRS ref: FR1.1 — refreshes automatically (via st.cache_data ttl=30s).
    """
    import pandas as pd
    from modules.dashboard.health import get_all_module_statuses

    statuses = get_all_module_statuses()

    rows = [
        {
            'ID':     s['id'],
            'Module': s['name'],
            'Status': f"{s['emoji']} {s['status'].capitalize()}",
            'Detail': s['detail'],
        }
        for s in statuses
    ]

    df = pd.DataFrame(rows)

    # Colour-code the Status column
    def _colour_status(val: str) -> str:
        if '✅' in val:
            return 'color: #3cb371; font-weight: bold'
        if '⚠️' in val:
            return 'color: #e6a21e; font-weight: bold'
        return 'color: #d23232; font-weight: bold'

    styled = df.style.applymap(_colour_status, subset=['Status'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary counts
    n_active   = sum(1 for s in statuses if s['status'] == 'active')
    n_inactive = sum(1 for s in statuses if s['status'] == 'inactive')
    n_error    = sum(1 for s in statuses if s['status'] == 'error')

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Active",   n_active)
    col2.metric("⚠️ Inactive", n_inactive)
    col3.metric("🔴 Error",    n_error)

    st.caption("Status refreshes every 30 seconds automatically.")
```

### 3b — Replace the "Digital Twin Status" placeholder in `render()`

Find the `with col_right:` block that shows `st.info("🖥️ Digital Twin simulation is in a future implementation phase.")` and replace the entire block:

```python
    with col_right:
        st.subheader("System Resources")
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()

        if stats['error']:
            st.caption("System stats unavailable (psutil error).")
        else:
            st.metric("CPU Usage",    f"{stats['cpu_pct']:.1f}%")
            st.metric("RAM Usage",    f"{stats['mem_pct']:.1f}%")
            st.metric(
                "RAM Used",
                f"{stats['mem_used_gb']:.1f} / {stats['mem_total_gb']:.1f} GB",
            )
            st.metric("Device",       stats['device'].upper())
            st.metric("Dashboard Up", stats['uptime_str'])
```

---

## File 4: Update `modules/dashboard/app.py`

### 4a — Add navigation gating in `render_sidebar()`

Streamlit's `st.sidebar.radio` does not support disabled options natively. The correct
pattern is to check page availability **after** the user selects a page, and show a
blocking warning instead of rendering the page. Replace the existing `render_sidebar()`
with this version:

```python
def render_sidebar() -> str:
    """
    Render sidebar navigation with availability indicators.
    Returns the selected page label string.
    Greyed-out pages are shown with ⚠️ prefix in the selectbox.
    """
    st.sidebar.markdown("# 🛡️ MalTwin")
    st.sidebar.markdown("*IIoT Malware Detection*")
    st.sidebar.divider()

    # Determine which pages are available
    model_ready   = state.is_model_loaded()
    file_ready    = state.has_uploaded_file()
    dataset_ready = config.DATA_DIR.exists() and any(config.DATA_DIR.iterdir())

    # Build options with availability markers
    # Format: (display_label, internal_key, is_available)
    nav_options = [
        ("🏠 Dashboard",        "🏠 Dashboard",        True),
        ("📂 Binary Upload",    "📂 Binary Upload",    True),
        (
            "🔍 Malware Detection" if (model_ready and file_ready)
            else "🔍 Malware Detection ⚠️",
            "🔍 Malware Detection",
            True,     # always selectable — shows its own guard message
        ),
        (
            "🖼️ Dataset Gallery" if dataset_ready
            else "🖼️ Dataset Gallery ⚠️",
            "🖼️ Dataset Gallery",
            True,     # always selectable — gallery shows its own info message
        ),
        ("🖥️ Digital Twin",    "🖥️ Digital Twin",     True),
    ]

    display_labels  = [opt[0] for opt in nav_options]
    internal_keys   = [opt[1] for opt in nav_options]

    selected_display = st.sidebar.radio(
        "Navigation",
        options=display_labels,
        label_visibility="hidden",
    )

    # Map display label back to internal key (strips ⚠️ suffix)
    selected_index = display_labels.index(selected_display)
    page = internal_keys[selected_index]

    # ── System status panel ───────────────────────────────────────────────────
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

    # ── Module health summary (compact) ──────────────────────────────────────
    st.sidebar.divider()
    try:
        from modules.dashboard.health import get_all_module_statuses
        statuses  = get_all_module_statuses()
        n_active  = sum(1 for s in statuses if s['status'] == 'active')
        n_total   = len(statuses)
        st.sidebar.caption(f"Modules: {n_active}/{n_total} active")
    except Exception:
        pass

    # ── Footer ────────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.caption("COMSATS University, Islamabad")
    st.sidebar.caption("BS Cyber Security 2023-2027")

    return page
```

### 4b — Update `main()` routing to use internal keys

The routing block in `main()` uses internal keys (without ⚠️ suffix) — no change needed if you already used the exact strings `"🔍 Malware Detection"` etc. Double-check the `elif` chain matches the `internal_keys` list above exactly.

---

## File 5: `tests/test_health.py`

```python
"""
Test suite for modules/dashboard/health.py

Health checks interact with the filesystem (config paths) and optionally
Docker. All tests mock filesystem state via tmp_path + monkeypatch.
No real dataset or trained model required.

Run:
    pytest tests/test_health.py -v
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Individual module checks
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckModule2BinaryToImage:
    def test_returns_active_when_module_importable(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        # M2 is always implemented — should be active
        assert result['status'] == 'active'
        assert result['emoji'] == '✅'

    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        assert 'status' in result
        assert 'detail' in result
        assert 'emoji' in result

    def test_status_is_valid_value(self):
        from modules.dashboard.health import _check_module2_binary_to_image
        result = _check_module2_binary_to_image()
        assert result['status'] in ('active', 'inactive', 'error')


class TestCheckModule3Dataset:
    def test_inactive_when_data_dir_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'nonexistent')
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'inactive'

    def test_inactive_when_data_dir_empty(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'inactive'

    def test_active_when_families_present(self, tmp_path, monkeypatch):
        import config
        (tmp_path / 'Allaple.A').mkdir()
        (tmp_path / 'Rbot_gen').mkdir()
        monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
        from modules.dashboard.health import _check_module3_dataset
        result = _check_module3_dataset()
        assert result['status'] == 'active'
        assert '2' in result['detail']   # "2 families found"


class TestCheckModule4Enhancement:
    def test_returns_active(self):
        from modules.dashboard.health import _check_module4_enhancement
        result = _check_module4_enhancement()
        assert result['status'] == 'active'


class TestCheckModule5Detection:
    def test_inactive_when_model_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', tmp_path / 'missing.pt')
        from modules.dashboard.health import _check_module5_detection
        result = _check_module5_detection()
        assert result['status'] == 'inactive'

    def test_inactive_when_class_names_missing(self, tmp_path, monkeypatch):
        import config
        import torch
        from modules.detection.model import MalTwinCNN
        # Create a real model file
        pt_path = tmp_path / 'best_model.pt'
        model = MalTwinCNN(num_classes=25)
        torch.save(model.state_dict(), pt_path)
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', pt_path)
        monkeypatch.setattr(config, 'CLASS_NAMES_PATH', tmp_path / 'missing.json')
        from modules.dashboard.health import _check_module5_detection
        result = _check_module5_detection()
        assert result['status'] == 'inactive'

    def test_active_when_model_and_class_names_present(self, tmp_path, monkeypatch):
        import config, json
        import torch
        from modules.detection.model import MalTwinCNN
        pt_path    = tmp_path / 'best_model.pt'
        names_path = tmp_path / 'class_names.json'
        model      = MalTwinCNN(num_classes=25)
        torch.save(model.state_dict(), pt_path)
        names_path.write_text(json.dumps({'class_names': [f'F{i}' for i in range(25)]}))
        monkeypatch.setattr(config, 'BEST_MODEL_PATH', pt_path)
        monkeypatch.setattr(config, 'CLASS_NAMES_PATH', names_path)
        from modules.dashboard.health import _check_module5_detection
        # Reload module to pick up monkeypatched config
        import importlib, modules.dashboard.health as h
        importlib.reload(h)
        result = h._check_module5_detection()
        assert result['status'] == 'active'


class TestCheckModule6Dashboard:
    def test_always_active(self):
        from modules.dashboard.health import _check_module6_dashboard
        result = _check_module6_dashboard()
        assert result['status'] == 'active'


class TestCheckModule7Gradcam:
    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import _check_module7_gradcam
        result = _check_module7_gradcam()
        assert 'status' in result
        assert 'detail' in result
        assert 'emoji' in result

    def test_status_is_valid_value(self):
        from modules.dashboard.health import _check_module7_gradcam
        result = _check_module7_gradcam()
        assert result['status'] in ('active', 'inactive', 'error')

    def test_active_or_inactive_when_captum_installed(self):
        """Captum is installed (Step 1 requirement) — should not be 'error'."""
        try:
            import captum   # noqa
            from modules.dashboard.health import _check_module7_gradcam
            result = _check_module7_gradcam()
            assert result['status'] in ('active', 'inactive')
        except ImportError:
            pytest.skip("Captum not installed")


class TestCheckModule8Reporting:
    def test_inactive_when_mitre_json_missing(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, 'MITRE_JSON_PATH', tmp_path / 'missing.json')
        from modules.dashboard.health import _check_module8_reporting
        result = _check_module8_reporting()
        assert result['status'] == 'inactive'

    def test_active_when_mitre_json_present(self, tmp_path, monkeypatch):
        import config, json
        mitre_path = tmp_path / 'mitre.json'
        # Write all 25 families
        db = {f'Family_{i}': {'tactics': [], 'techniques': [], 'description': ''} for i in range(25)}
        mitre_path.write_text(json.dumps(db))
        monkeypatch.setattr(config, 'MITRE_JSON_PATH', mitre_path)
        from modules.dashboard import health
        import importlib; importlib.reload(health)
        result = health._check_module8_reporting()
        assert result['status'] == 'active'


# ─────────────────────────────────────────────────────────────────────────────
# get_all_module_statuses
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAllModuleStatuses:
    def test_returns_list_of_eight(self):
        from modules.dashboard.health import get_all_module_statuses
        # Clear cache before calling to ensure fresh result
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        assert len(results) == 8

    def test_each_entry_has_required_keys(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        for r in results:
            assert 'id' in r
            assert 'name' in r
            assert 'status' in r
            assert 'detail' in r
            assert 'emoji' in r

    def test_all_statuses_are_valid_values(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        for r in results:
            assert r['status'] in ('active', 'inactive', 'error'), \
                f"Module {r['id']} has invalid status: {r['status']}"

    def test_module_ids_are_m1_through_m8(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        ids = [r['id'] for r in results]
        assert ids == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']

    def test_does_not_raise(self):
        """get_all_module_statuses must never raise regardless of environment."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        try:
            results = get_all_module_statuses()
        except Exception as e:
            pytest.fail(f"get_all_module_statuses raised: {e}")

    def test_m6_is_always_active(self):
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m6 = next(r for r in results if r['id'] == 'M6')
        assert m6['status'] == 'active'

    def test_m2_is_always_active(self):
        """M2 is fully implemented — should always be active."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m2 = next(r for r in results if r['id'] == 'M2')
        assert m2['status'] == 'active'

    def test_m4_is_always_active(self):
        """M4 is fully implemented — should always be active."""
        from modules.dashboard.health import get_all_module_statuses
        get_all_module_statuses.clear()
        results = get_all_module_statuses()
        m4 = next(r for r in results if r['id'] == 'M4')
        assert m4['status'] == 'active'


# ─────────────────────────────────────────────────────────────────────────────
# get_system_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSystemStats:
    def test_returns_dict_with_required_keys(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        required = {
            'cpu_pct', 'mem_pct', 'mem_used_gb',
            'mem_total_gb', 'uptime_str', 'device', 'error',
        }
        assert required.issubset(stats.keys())

    def test_cpu_pct_in_valid_range(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert 0.0 <= stats['cpu_pct'] <= 100.0

    def test_mem_pct_in_valid_range(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert 0.0 <= stats['mem_pct'] <= 100.0

    def test_mem_used_le_mem_total(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        if not stats['error']:
            assert stats['mem_used_gb'] <= stats['mem_total_gb']

    def test_device_is_nonempty_string(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        assert isinstance(stats['device'], str)
        assert len(stats['device']) > 0

    def test_error_flag_is_bool(self):
        from modules.dashboard.health import get_system_stats
        stats = get_system_stats()
        assert isinstance(stats['error'], bool)

    def test_does_not_raise_when_psutil_missing(self, monkeypatch):
        """If psutil is not importable, stats must return error=True, not raise."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("mocked missing psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        from modules.dashboard import health
        import importlib; importlib.reload(health)
        stats = health.get_system_stats()
        assert stats['error'] is True
```

---

## Definition of Done

```bash
# Step 4 tests
pytest tests/test_health.py -v
# Expected: all tests pass, 0 failures

# Full regression suite
pytest tests/ -v -m "not integration"
# Expected: 0 failures

# Import smoke tests
python -c "from modules.dashboard.health import get_all_module_statuses, get_system_stats"
python -c "import psutil; print('psutil OK')"

# Dashboard launch — verify all changes
streamlit run modules/dashboard/app.py --server.port 8501

# Verify on home page:
#   ✓ Module status table shows live status (not hardcoded strings)
#   ✓ M2, M4, M6 show ✅ Active
#   ✓ M5 shows ⚠️ Inactive (if no model trained) or ✅ Active (if trained)
#   ✓ M7 shows ✅ Active or ⚠️ Inactive based on Captum + model
#   ✓ M8 shows ✅ Active (if mitre_ics_mapping.json has 25 families)
#   ✓ System Resources panel shows real CPU %, RAM %, uptime — not placeholder text
#   ✓ Sidebar shows ⚠️ suffix on Dataset Gallery if dataset missing
#   ✓ Sidebar shows ⚠️ suffix on Detection if no file uploaded or no model
#   ✓ Status refreshes every 30 seconds (caption visible at bottom of table)
```

### Checklist

- [ ] `pytest tests/test_health.py -v` — 0 failures
- [ ] All earlier tests still pass
- [ ] `modules/dashboard/health.py` exists
- [ ] `psutil` in `requirements.txt`
- [ ] `state.py` has `KEY_APP_START_TIME` constant
- [ ] `init_session_state()` sets `KEY_APP_START_TIME` to `datetime.utcnow()` on first run only
- [ ] `get_all_module_statuses()` decorated with `@st.cache_data(ttl=30)`
- [ ] All 8 check functions return dict with `status`, `detail`, `emoji` keys
- [ ] Status values are strictly `'active'`, `'inactive'`, or `'error'`
- [ ] No check function raises — all wrapped in try/except
- [ ] Home page module table is styled (colour-coded Status column)
- [ ] Home page shows real CPU/RAM/uptime metrics — not "Not Configured" stub
- [ ] Sidebar labels include `⚠️` suffix for unavailable pages
- [ ] Navigation routing still works for all 5 pages (no KeyError from ⚠️ suffix)

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|---|---|---|
| `get_all_module_statuses()` not cleared between tests | Stale cached result from previous test poisons next test | Call `get_all_module_statuses.clear()` at the start of each test that needs fresh state |
| `monkeypatch` on `config` not reflected inside cached function | Health check still reads old path after monkeypatch | `importlib.reload(health)` after monkeypatching config attributes |
| `subprocess.run(['docker', 'info'])` blocking for >2s | Streamlit appears frozen during M1 check | Always use `timeout=2` in subprocess calls |
| Sidebar radio `⚠️` suffix causes routing KeyError | `page` value is `"🔍 Malware Detection ⚠️"` but routing checks for `"🔍 Malware Detection"` | Map display labels → internal keys using `internal_keys[selected_index]` |
| `psutil.cpu_percent(interval=None)` returns 0.0 | CPU shows 0% always | Use `interval=0.1` — tiny blocking interval gives real reading |
| `datetime.utcnow()` called every rerun for uptime start | Uptime resets to 0 on every Streamlit rerun | Only set `KEY_APP_START_TIME` when it is `None` — the `if` guard in `init_session_state()` |
| `@st.cache_data` on `get_system_stats` | CPU/RAM values never update | Do NOT cache `get_system_stats` — it must be called live each render |

---

*Step 4 complete → Step 5: Dashboard-triggered training flow with live progress display.*
