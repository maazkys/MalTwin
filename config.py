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

# ── Repository root ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(os.getenv("MALTWIN_DATA_DIR",        str(BASE_DIR / "data/malimg")))
PROCESSED_DIR   = Path(os.getenv("MALTWIN_PROCESSED_DIR",   str(BASE_DIR / "data/processed")))
MODEL_DIR       = Path(os.getenv("MALTWIN_MODEL_DIR",       str(BASE_DIR / "models")))
CHECKPOINT_DIR  = MODEL_DIR / "checkpoints"
LOG_DIR         = Path(os.getenv("MALTWIN_LOG_DIR",         str(BASE_DIR / "logs")))
REPORTS_DIR     = Path(os.getenv("MALTWIN_REPORTS_DIR",     str(BASE_DIR / "data" / "reports")))
MITRE_JSON_PATH  = BASE_DIR / "data" / "mitre_ics_mapping.json"
CLASS_NAMES_PATH = PROCESSED_DIR / "class_names.json"
DB_PATH          = LOG_DIR / "maltwin.db"
BEST_MODEL_PATH       = MODEL_DIR / "best_model.pt"
EVAL_METRICS_PATH     = PROCESSED_DIR / "eval_metrics.json"
CONFUSION_MATRIX_PATH = PROCESSED_DIR / "confusion_matrix.png"
ROC_CURVE_PATH        = PROCESSED_DIR / "roc_curves.png"
TRAINING_CURVES_PATH  = PROCESSED_DIR / "training_curves.png"
TRAINING_HISTORY_PATH = PROCESSED_DIR / "training_history.json"
UNSEEN_EVAL_PATH      = PROCESSED_DIR / "unseen_family_eval.json"

# ── Dynamic analysis (behavioural, IoT-23 / Digital Twin conn.log) ─────────────
DYNAMIC_DATA_DIR            = Path(os.getenv("MALTWIN_DYNAMIC_DATA_DIR", str(BASE_DIR / "data/iot23")))
DYNAMIC_MODEL_PATH          = MODEL_DIR / "dynamic_model.joblib"
DYNAMIC_METRICS_PATH        = PROCESSED_DIR / "dynamic_metrics.json"
DYNAMIC_CM_PATH             = PROCESSED_DIR / "dynamic_confusion_matrix.png"
DYNAMIC_IMPORTANCE_PATH     = PROCESSED_DIR / "dynamic_feature_importance.png"
MITRE_BEHAVIOUR_JSON_PATH   = BASE_DIR / "data" / "mitre_ics_behaviour_mapping.json"

# ── Hybrid fusion ──────────────────────────────────────────────────────────────
FUSION_STATIC_WEIGHT  = float(os.getenv("MALTWIN_FUSION_STATIC_WEIGHT",  "0.5"))
FUSION_DYNAMIC_WEIGHT = float(os.getenv("MALTWIN_FUSION_DYNAMIC_WEIGHT", "0.5"))
FUSION_MALICIOUS_THRESHOLD  = float(os.getenv("MALTWIN_FUSION_MALICIOUS",  "0.70"))
FUSION_SUSPICIOUS_THRESHOLD = float(os.getenv("MALTWIN_FUSION_SUSPICIOUS", "0.40"))

# Create directories at import time (safe to call repeatedly)
for _dir in [PROCESSED_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE = int(os.getenv("MALTWIN_IMG_SIZE", "128"))
# 2-D byte layout before resizing: 'malimg' matches how the training images were
# built (Nataraj et al. 2011 width table); 'square' is the earlier sqrt layout.
IMAGE_LAYOUT = os.getenv("MALTWIN_IMAGE_LAYOUT", "malimg")

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

# ── Oversampler ───────────────────────────────────────────────────────────────
OVERSAMPLE_STRATEGY = os.getenv("MALTWIN_OVERSAMPLE_STRATEGY", "oversample_minority")
assert OVERSAMPLE_STRATEGY in {"oversample_minority", "sqrt_inverse", "uniform"}, \
    f"MALTWIN_OVERSAMPLE_STRATEGY must be one of: oversample_minority, sqrt_inverse, uniform"

# ── Reproducibility ───────────────────────────────────────────────────────────
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
CONFIDENCE_GREEN = float(os.getenv('MALTWIN_CONFIDENCE_GREEN', '0.80'))
CONFIDENCE_AMBER = float(os.getenv('MALTWIN_CONFIDENCE_AMBER', '0.50'))

# ── Dashboard ──────────────────────────────────────────────────────────────────
STREAMLIT_PORT  = 8501
DASHBOARD_TITLE = "MalTwin — IIoT Malware Detection"

# ── Malimg dataset metadata ────────────────────────────────────────────────────
MALIMG_EXPECTED_FAMILIES = 25
MALIMG_TOTAL_SAMPLES     = 9339
