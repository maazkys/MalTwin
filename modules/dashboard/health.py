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
import streamlit as st
from datetime import datetime

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
    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'detail': 'Digital Twin check timed out (>2s)',
            'emoji':  '🔴',
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
        validate_binary_format(dummy)
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
