# config.py
"""
Central configuration for MalTwin.
All modules import constants from here. Never hardcode paths or hyperparameters elsewhere.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Repository root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(os.getenv("MALTWIN_DATA_DIR",        str(BASE_DIR / "data/malimg")))
PROCESSED_DIR   = Path(os.getenv("MALTWIN_PROCESSED_DIR",   str(BASE_DIR / "data/processed")))
MODEL_DIR       = Path(os.getenv("MALTWIN_MODEL_DIR",       str(BASE_DIR / "models")))
CHECKPOINT_DIR  = MODEL_DIR / "checkpoints"
LOG_DIR         = Path(os.getenv("MALTWIN_LOG_DIR",         str(BASE_DIR / "logs")))
REPORTS_DIR     = Path(os.getenv("MALTWIN_REPORTS_DIR",     str(BASE_DIR / "reports")))
MITRE_JSON_PATH  = BASE_DIR / "data" / "mitre_ics_mapping.json"
CLASS_NAMES_PATH = PROCESSED_DIR / "class_names.json"
DB_PATH          = LOG_DIR / "maltwin.db"
BEST_MODEL_PATH  = MODEL_DIR / "best_model.pt"

# Create directories at import time (safe to call repeatedly)
for _dir in [PROCESSED_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE = int(os.getenv("MALTWIN_IMG_SIZE", "128"))

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE   = int(os.getenv("MALTWIN_BATCH_SIZE",   "32"))
EPOCHS       = int(os.getenv("MALTWIN_EPOCHS",        "30"))
LR           = float(os.getenv("MALTWIN_LR",          "0.001"))
WEIGHT_DECAY = float(os.getenv("MALTWIN_WEIGHT_DECAY","0.0001"))
LR_PATIENCE  = int(os.getenv("MALTWIN_LR_PATIENCE",   "5"))
NUM_WORKERS  = int(os.getenv("MALTWIN_NUM_WORKERS",    "4"))

# ── Dataset split ratios ───────────────────────────────────────────────────────
TRAIN_RATIO = float(os.getenv("MALTWIN_TRAIN_RATIO", "0.70"))
VAL_RATIO   = float(os.getenv("MALTWIN_VAL_RATIO",   "0.15"))
TEST_RATIO  = float(os.getenv("MALTWIN_TEST_RATIO",  "0.15"))
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "MALTWIN_TRAIN_RATIO + MALTWIN_VAL_RATIO + MALTWIN_TEST_RATIO must equal 1.0"

# ── Oversampler ────────────────────────────────────────────────────────────────
OVERSAMPLE_STRATEGY = os.getenv("MALTWIN_OVERSAMPLE_STRATEGY", "oversample_minority")
assert OVERSAMPLE_STRATEGY in {"oversample_minority", "sqrt_inverse", "uniform"}, \
    f"MALTWIN_OVERSAMPLE_STRATEGY must be one of: oversample_minority, sqrt_inverse, uniform"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("MALTWIN_RANDOM_SEED", "42"))

# ── Device ─────────────────────────────────────────────────────────────────────
_device_env = os.getenv("MALTWIN_DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# ── Upload limits ──────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES    = 50 * 1024 * 1024   # 50 MB
ACCEPTED_EXTENSIONS = {".exe", ".dll", ".elf", ""}

# ── Confidence thresholds for UI color coding ──────────────────────────────────
CONFIDENCE_GREEN = 0.80
CONFIDENCE_AMBER = 0.50

# ── Dashboard ─────────────────────────────────────────────────────────────────
STREAMLIT_PORT  = 8501
DASHBOARD_TITLE = "MalTwin — IIoT Malware Detection"

# ── Malimg dataset metadata ────────────────────────────────────────────────────
MALIMG_EXPECTED_FAMILIES = 25
MALIMG_TOTAL_SAMPLES     = 9339
