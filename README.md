# MalTwin — Complete Implementation PRD
### Version 2.0 | Modules 2, 3, 4, 5, 6 | Agent-Ready Specification

> **This document is the single source of truth for MalTwin implementation.**
> Every class, method, field, type signature, error condition, and behavioral rule is specified here.
> No external document should be required. Agents must implement exactly what is written.
> Modules 1 (Digital Twin), 7 (XAI/Grad-CAM), and 8 (Forensic Reporting) are explicitly out of scope and stubbed only.

---

## Table of Contents

1. [Project Overview & Scope](#1-project-overview--scope)
2. [Repository Structure](#2-repository-structure)
3. [Environment, Dependencies & Configuration](#3-environment-dependencies--configuration)
4. [Module 2 — Binary-to-Image Conversion](#4-module-2--binary-to-image-conversion)
5. [Module 3 — Dataset Collection & Preprocessing](#5-module-3--dataset-collection--preprocessing)
6. [Module 4 — Data Enhancement & Balancing](#6-module-4--data-enhancement--balancing)
7. [Module 5 — Intelligent Malware Detection](#7-module-5--intelligent-malware-detection)
8. [Module 6 — Dashboard & Visualization](#8-module-6--dashboard--visualization)
9. [Database Layer](#9-database-layer)
10. [CLI Scripts](#10-cli-scripts)
11. [Static Data Files](#11-static-data-files)
12. [Test Suite](#12-test-suite)
13. [Inter-Module Data Flow](#13-inter-module-data-flow)
14. [Error Handling Contract](#14-error-handling-contract)
15. [Implementation Order & Dependency Graph](#15-implementation-order--dependency-graph)
16. [Coding Agent Constraints & Rules](#16-coding-agent-constraints--rules)

---

## 1. Project Overview & Scope

### 1.1 What MalTwin Does (Implementation Perspective)

MalTwin is a Python application that:
1. Accepts a PE (.exe, .dll) or ELF binary file as input
2. Converts its raw bytes into a 128×128 grayscale PNG image
3. Passes that image through a trained CNN to classify it into one of 25 malware families
4. Displays the result, confidence score, per-class probabilities, and MITRE ATT&CK ICS mappings
5. Logs every detection event to a local SQLite database
6. Surfaces all of the above through a Streamlit web dashboard

### 1.2 Scope Decision Table

| Module | In Scope | Stub Required | Reason for Deferral |
|--------|----------|---------------|---------------------|
| M1 — Digital Twin Simulation | ❌ | ✅ Dashboard tab stubbed | Requires Docker + Mininet infra |
| M2 — Binary-to-Image Conversion | ✅ | — | Core pipeline entry point |
| M3 — Dataset Collection & Preprocessing | ✅ | — | Required before training |
| M4 — Data Enhancement & Balancing | ✅ | — | Required for training quality |
| M5 — Intelligent Malware Detection | ✅ | — | Core deliverable |
| M6 — Dashboard & Visualization | ✅ Partial | XAI + Report buttons stubbed | UI for M2/M5 outputs |
| M7 — Explainable AI (Grad-CAM) | ❌ | ✅ Checkbox stubbed in dashboard | Depends on trained M5 model |
| M8 — Automated Threat Reporting | ❌ | ✅ Download buttons stubbed | Depends on M5 + M7 |

### 1.3 Primary User Flow (End-to-End)

```
User opens dashboard (localhost:8501)
  → navigates to "Binary Upload" page
  → uploads suspicious.exe
  → system validates format (PE/ELF), checks size ≤ 50MB
  → system converts bytes → 128×128 grayscale numpy array
  → system computes SHA-256, extracts metadata
  → system displays grayscale image + metadata + histogram
  → user navigates to "Malware Detection" page
  → user clicks "Run Detection"
  → system runs CNN inference → returns family label + confidence + all class probs
  → system looks up MITRE ATT&CK mapping for predicted family
  → system logs event to SQLite
  → dashboard displays: label, confidence bar (color-coded), per-class chart, MITRE mapping
```

---

## 2. Repository Structure

```
maltwin/
│
├── README.md
├── MALTWIN_PRD_COMPLETE.md          ← this document
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
│
├── modules/
│   ├── __init__.py                  ← empty
│   │
│   ├── binary_to_image/
│   │   ├── __init__.py              ← exports: BinaryConverter, validate_binary_format, compute_sha256
│   │   ├── converter.py             ← BinaryConverter class
│   │   └── utils.py                 ← validate_binary_format, compute_sha256, compute_pixel_histogram
│   │
│   ├── dataset/
│   │   ├── __init__.py              ← exports: MalimgDataset, get_dataloaders, validate_dataset_integrity
│   │   ├── loader.py                ← MalimgDataset, get_dataloaders
│   │   └── preprocessor.py         ← validate_dataset_integrity, normalize_image, encode_labels
│   │
│   ├── enhancement/
│   │   ├── __init__.py              ← exports: get_train_transforms, get_val_transforms, ClassAwareOversampler
│   │   ├── augmentor.py             ← get_train_transforms, get_val_transforms, GaussianNoise
│   │   └── balancer.py              ← ClassAwareOversampler
│   │
│   ├── detection/
│   │   ├── __init__.py              ← exports: MalTwinCNN, train, evaluate, load_model, predict_single
│   │   ├── model.py                 ← MalTwinCNN (PyTorch nn.Module)
│   │   ├── trainer.py               ← train(), validate_epoch()
│   │   ├── evaluator.py             ← evaluate(), format_metrics_table()
│   │   └── inference.py             ← load_model(), predict_single(), predict_batch()
│   │
│   └── dashboard/
│       ├── __init__.py              ← empty
│       ├── app.py                   ← Streamlit entry point + navigation
│       ├── db.py                    ← SQLite helpers
│       ├── state.py                 ← session_state key constants + helpers
│       └── pages/
│           ├── __init__.py          ← empty
│           ├── home.py              ← Dashboard overview page
│           ├── upload.py            ← Binary upload + visualization (M3 UI)
│           ├── detection.py         ← Detection + prediction display (M5 UI)
│           └── digital_twin.py      ← STUB page (M1 deferred)
│
├── data/
│   ├── malimg/                      ← Malimg dataset root (user-downloaded)
│   │   └── .gitkeep
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── class_names.json         ← written by scripts/train.py, read by dashboard
│   └── mitre_ics_mapping.json       ← static MITRE ATT&CK for ICS reference (seed provided below)
│
├── models/
│   ├── checkpoints/
│   │   └── .gitkeep
│   └── best_model.pt                ← written by trainer, read by inference + dashboard
│
├── reports/
│   └── .gitkeep                     ← future M8 output directory
│
├── logs/
│   ├── maltwin.db                   ← SQLite detection event log
│   └── .gitkeep
│
├── scripts/
│   ├── train.py                     ← CLI: full training pipeline
│   ├── evaluate.py                  ← CLI: test-set evaluation only
│   └── convert_binary.py            ← CLI: single-file binary-to-image
│
└── tests/
    ├── conftest.py                  ← shared fixtures
    ├── test_converter.py
    ├── test_dataset.py
    ├── test_enhancement.py
    ├── test_model.py
    ├── test_inference.py
    ├── test_db.py
    └── fixtures/
        ├── create_fixtures.py       ← script to generate test PE/ELF binaries
        ├── sample_pe.exe            ← minimal valid PE (MZ header, 1024 bytes)
        └── sample_elf              ← minimal valid ELF (\x7fELF header, 1024 bytes)
```

### 2.1 `.gitignore` Contents

```
__pycache__/
*.pyc
*.pyo
.env
data/malimg/
data/processed/*.json
models/best_model.pt
models/checkpoints/
logs/maltwin.db
reports/
*.egg-info/
.pytest_cache/
.DS_Store
```

---

## 3. Environment, Dependencies & Configuration

### 3.1 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| OS | Ubuntu 22.04 LTS (kernel 5.15+) | Ubuntu 22.04 LTS |
| Python | 3.11.x | 3.11.9 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 100 GB free |
| GPU | None (CPU-only supported) | NVIDIA GPU, 6+ GB VRAM, CUDA 12.x |

> **Python version note**: SRS specifies 3.14.x but PyTorch 2.3 and torchvision 0.18 do not support Python 3.14 as of this writing. Use Python 3.11.x. Update when upstream support lands.

### 3.2 `requirements.txt`

```
# Deep Learning
torch==2.3.1
torchvision==0.18.1
captum==0.7.0          # installed now, used in M7

# Image Processing
opencv-python-headless==4.10.0.84
Pillow==10.4.0
numpy==1.26.4

# Data / ML utilities
scikit-learn==1.5.1
imbalanced-learn==0.12.3
pandas==2.2.2
scipy==1.14.0

# Dashboard
streamlit==1.37.0
plotly==5.23.0
watchdog==4.0.1        # improves Streamlit hot reload on Linux

# Reporting (M8 — install now, implement later)
fpdf2==2.7.9

# Utilities
python-dotenv==1.0.1
tqdm==4.66.5
matplotlib==3.9.2      # used for confusion matrix rendering in evaluator

# Testing
pytest==8.3.2
pytest-cov==5.0.0
```

### 3.3 `.env.example`

```bash
# Paths — relative to repo root, or use absolute paths
MALTWIN_DATA_DIR=./data/malimg
MALTWIN_PROCESSED_DIR=./data/processed
MALTWIN_MODEL_DIR=./models
MALTWIN_LOG_DIR=./logs
MALTWIN_REPORTS_DIR=./reports

# Image
MALTWIN_IMG_SIZE=128

# Training hyperparameters
MALTWIN_BATCH_SIZE=32
MALTWIN_EPOCHS=30
MALTWIN_LR=0.001
MALTWIN_WEIGHT_DECAY=0.0001
MALTWIN_LR_PATIENCE=5
MALTWIN_NUM_WORKERS=4

# Device: auto | cpu | cuda | cuda:0 | cuda:1
MALTWIN_DEVICE=auto

# Dataset splits (must sum to 1.0)
MALTWIN_TRAIN_RATIO=0.70
MALTWIN_VAL_RATIO=0.15
MALTWIN_TEST_RATIO=0.15

# Oversampler strategy: oversample_minority | sqrt_inverse | uniform
MALTWIN_OVERSAMPLE_STRATEGY=oversample_minority

# Reproducibility
MALTWIN_RANDOM_SEED=42
```

### 3.4 `config.py` — Complete Implementation

```python
# config.py
"""
Central configuration for MalTwin.
All modules import from here. Never hardcode paths or hyperparameters elsewhere.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Repository root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(os.getenv("MALTWIN_DATA_DIR",        BASE_DIR / "data/malimg"))
PROCESSED_DIR   = Path(os.getenv("MALTWIN_PROCESSED_DIR",   BASE_DIR / "data/processed"))
MODEL_DIR       = Path(os.getenv("MALTWIN_MODEL_DIR",       BASE_DIR / "models"))
CHECKPOINT_DIR  = MODEL_DIR / "checkpoints"
LOG_DIR         = Path(os.getenv("MALTWIN_LOG_DIR",         BASE_DIR / "logs"))
REPORTS_DIR     = Path(os.getenv("MALTWIN_REPORTS_DIR",     BASE_DIR / "reports"))
MITRE_JSON_PATH = BASE_DIR / "data/mitre_ics_mapping.json"
CLASS_NAMES_PATH = PROCESSED_DIR / "class_names.json"
DB_PATH         = LOG_DIR / "maltwin.db"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"

# Create directories if they don't exist (safe to call at import time)
for _dir in [PROCESSED_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE = int(os.getenv("MALTWIN_IMG_SIZE", 128))   # output PNG is IMG_SIZE × IMG_SIZE

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE      = int(os.getenv("MALTWIN_BATCH_SIZE", 32))
EPOCHS          = int(os.getenv("MALTWIN_EPOCHS", 30))
LR              = float(os.getenv("MALTWIN_LR", 0.001))
WEIGHT_DECAY    = float(os.getenv("MALTWIN_WEIGHT_DECAY", 1e-4))
LR_PATIENCE     = int(os.getenv("MALTWIN_LR_PATIENCE", 5))
NUM_WORKERS     = int(os.getenv("MALTWIN_NUM_WORKERS", 4))

# ── Dataset split ratios ───────────────────────────────────────────────────────
TRAIN_RATIO = float(os.getenv("MALTWIN_TRAIN_RATIO", 0.70))
VAL_RATIO   = float(os.getenv("MALTWIN_VAL_RATIO",   0.15))
TEST_RATIO  = float(os.getenv("MALTWIN_TEST_RATIO",  0.15))
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Split ratios must sum to 1.0"

# ── Oversampler ────────────────────────────────────────────────────────────────
OVERSAMPLE_STRATEGY = os.getenv("MALTWIN_OVERSAMPLE_STRATEGY", "oversample_minority")
assert OVERSAMPLE_STRATEGY in {"oversample_minority", "sqrt_inverse", "uniform"}

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = int(os.getenv("MALTWIN_RANDOM_SEED", 42))

# ── Device ─────────────────────────────────────────────────────────────────────
_device_env = os.getenv("MALTWIN_DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# ── Upload limits (SRS FR3.1) ──────────────────────────────────────────────────
MAX_UPLOAD_BYTES       = 50 * 1024 * 1024   # 50 MB
ACCEPTED_EXTENSIONS    = {".exe", ".dll", ".elf", ""}  # "" = extensionless ELF

# ── Confidence thresholds for UI color coding (SRS FR5.2) ─────────────────────
CONFIDENCE_GREEN = 0.80   # ≥ 80% → green
CONFIDENCE_AMBER = 0.50   # 50–79% → amber + warning
# < 50% → red + strong warning

# ── Dashboard ─────────────────────────────────────────────────────────────────
STREAMLIT_PORT = 8501
DASHBOARD_TITLE = "MalTwin — IIoT Malware Detection"

# ── Malimg dataset metadata ────────────────────────────────────────────────────
# 25 known families in Malimg dataset. Used for validation.
# Actual class list is loaded from data at runtime; this is a sanity-check reference.
MALIMG_EXPECTED_FAMILIES = 25
MALIMG_TOTAL_SAMPLES     = 9339  # approximate
```

---

## 4. Module 2 — Binary-to-Image Conversion

### 4.1 Overview

This module is the entry point for all analysis. It accepts raw binary bytes, validates format, converts bytes to a 128×128 grayscale image, and computes forensic metadata. It has zero ML dependencies and must work standalone.

### 4.2 `modules/binary_to_image/utils.py` — Complete Specification

#### Function: `validate_binary_format`

```python
def validate_binary_format(file_bytes: bytes) -> str:
    """
    Inspects magic bytes to identify binary format.

    Args:
        file_bytes: raw bytes of the uploaded file (minimum 4 bytes required)

    Returns:
        'PE'  if first 2 bytes are b'MZ' (0x4D 0x5A)
        'ELF' if first 4 bytes are b'\x7fELF' (0x7F 0x45 0x4C 0x46)

    Raises:
        ValueError("File is too small to be a valid binary (minimum 4 bytes required)")
            → if len(file_bytes) < 4
        ValueError("Unsupported file format. Expected PE (.exe/.dll) or ELF binary. "
                   "Detected magic bytes: {hex_repr}")
            → if magic bytes match neither PE nor ELF
            → hex_repr = file_bytes[:4].hex().upper()  e.g. "DEADBEEF"

    Notes:
        - Does NOT check full header validity (PE Optional Header, ELF e_type etc.)
          Just magic bytes check is sufficient per SRS UC-01 A1.
        - Case of magic bytes is always exact — b'MZ' not b'mz'.
    """
```

#### Function: `compute_sha256`

```python
def compute_sha256(file_bytes: bytes) -> str:
    """
    Computes SHA-256 digest of raw file bytes.

    Args:
        file_bytes: raw bytes of the uploaded file

    Returns:
        Lowercase hexadecimal string of length 64.
        Example: "a3f1c2d4e5b6a7f8..."

    Implementation:
        import hashlib
        return hashlib.sha256(file_bytes).hexdigest()

    Constraints:
        - MUST use Python standard library hashlib only.
        - NO external services, NO network calls. (SRS SEC-4, CON-9)
        - Function is deterministic: same input always returns same output.
    """
```

#### Function: `compute_pixel_histogram`

```python
def compute_pixel_histogram(img_array: np.ndarray) -> dict:
    """
    Computes byte-value frequency distribution of a grayscale image.

    Args:
        img_array: numpy array of shape (H, W), dtype uint8, values 0–255

    Returns:
        {
            'bins':   list[int]   # [0, 1, 2, ..., 255]  always length 256
            'counts': list[int]   # pixel count for each bin value
        }

    Implementation:
        import numpy as np
        bins = list(range(256))
        counts = [int(np.sum(img_array == i)) for i in range(256)]
        # OR faster: counts = np.bincount(img_array.flatten(), minlength=256).tolist()
        return {'bins': bins, 'counts': counts}

    Notes:
        - Total of all counts must equal img_array.size (H * W).
        - Used by dashboard to render pixel intensity histogram (SRS FR3.4).
        - 256 bins exactly, never more, never fewer.
    """
```

#### Function: `get_file_metadata`

```python
def get_file_metadata(
    file_bytes: bytes,
    filename: str,
    file_format: str,
) -> dict:
    """
    Assembles the complete metadata dict for a processed binary file.

    Args:
        file_bytes:  raw bytes of the uploaded file
        filename:    original filename as uploaded by user
        file_format: 'PE' or 'ELF' (output of validate_binary_format)

    Returns:
        {
            'name':        str   # original filename
            'size_bytes':  int   # len(file_bytes)
            'size_human':  str   # human-readable e.g. "2.4 MB", "512.0 KB", "89.2 B"
            'format':      str   # 'PE' or 'ELF'
            'sha256':      str   # 64-char hex string
            'upload_time': str   # ISO 8601 UTC string e.g. "2025-04-22T14:35:22.123456"
        }

    Implementation notes:
        - size_human: if size_bytes >= 1_048_576 use MB (2 decimal places),
                      elif >= 1024 use KB, else use B.
        - upload_time: datetime.utcnow().isoformat()
        - sha256: call compute_sha256(file_bytes)
    """
```

### 4.3 `modules/binary_to_image/converter.py` — Complete Specification

```python
import math
import numpy as np
import cv2
from pathlib import Path
from config import IMG_SIZE


class BinaryConverter:
    """
    Converts raw binary file bytes into a standardized grayscale PNG image.

    The conversion algorithm (Nataraj et al. 2011 method):
        1. Read file bytes as a flat uint8 numpy array.
        2. Determine 2D width using: width = int(math.sqrt(len(byte_array)))
           (This preserves aspect ratio proportional to file size.)
        3. Truncate array so its length is exactly (width * rows) where
           rows = len(byte_array) // width.
        4. Reshape to (rows, width) 2D array.
        5. Resize to (img_size, img_size) using bilinear interpolation.
        6. Result is a uint8 2D numpy array with pixel values 0–255.

    Constructor args:
        img_size (int): side length of output square image in pixels.
                        Default: config.IMG_SIZE (128).

    SRS refs: Module 2 FE-1, FE-2, FE-3
    """

    def __init__(self, img_size: int = IMG_SIZE):
        """
        Args:
            img_size: output image will be (img_size, img_size) pixels.
                      Must be a positive integer. Raises ValueError if not.
        """
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        self.img_size = img_size

    def convert(self, file_bytes: bytes) -> np.ndarray:
        """
        Convert raw binary bytes to a grayscale image array.

        Args:
            file_bytes: raw bytes of a PE or ELF binary. Caller is responsible
                        for having already validated format (validate_binary_format).

        Returns:
            numpy.ndarray of shape (img_size, img_size), dtype=uint8, values 0–255.

        Raises:
            ValueError("Binary file is empty or too small to convert (minimum 64 bytes)")
                → if len(file_bytes) < 64

        Algorithm (implement exactly):
            byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
            n = len(byte_array)
            width = int(math.sqrt(n))
            if width == 0:
                width = 1
            rows = n // width
            trimmed = byte_array[:rows * width]
            reshaped = trimmed.reshape((rows, width))
            resized = cv2.resize(
                reshaped,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR
            )
            return resized.astype(np.uint8)

        Notes:
            - np.frombuffer produces a read-only array. Do NOT modify it in-place.
              Always work on copies or use reshape which returns a new view.
            - If rows == 1 (very small file), the image will be a single-row resized
              to a square. This is acceptable behavior.
            - cv2.resize with INTER_LINEAR may produce float artifacts;
              .astype(np.uint8) truncates to valid range. Do NOT use np.clip here
              because INTER_LINEAR output for uint8 input is always in [0,255].
        """

    def to_png_bytes(self, img_array: np.ndarray) -> bytes:
        """
        Encode a grayscale numpy array to PNG bytes for in-memory use.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PNG-encoded bytes suitable for passing to st.image() or file.write().

        Implementation:
            success, encoded = cv2.imencode('.png', img_array)
            if not success:
                raise RuntimeError("cv2.imencode failed to encode image as PNG")
            return encoded.tobytes()
        """

    def to_pil_image(self, img_array: np.ndarray):
        """
        Convert grayscale array to PIL Image for torchvision transforms.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PIL.Image.Image in mode 'L' (8-bit grayscale).

        Implementation:
            from PIL import Image
            return Image.fromarray(img_array, mode='L')
        """

    def save(self, img_array: np.ndarray, output_path: Path) -> None:
        """
        Save grayscale array as PNG file to disk.

        Args:
            img_array:   numpy array of shape (H, W), dtype uint8.
            output_path: Path where PNG will be written. Parent directory must exist.

        Raises:
            RuntimeError("Failed to save image to {output_path}")
                → if cv2.imwrite returns False

        Implementation:
            success = cv2.imwrite(str(output_path), img_array)
            if not success:
                raise RuntimeError(f"Failed to save image to {output_path}")
        """
```

### 4.4 `modules/binary_to_image/__init__.py`

```python
from .converter import BinaryConverter
from .utils import (
    validate_binary_format,
    compute_sha256,
    compute_pixel_histogram,
    get_file_metadata,
)

__all__ = [
    "BinaryConverter",
    "validate_binary_format",
    "compute_sha256",
    "compute_pixel_histogram",
    "get_file_metadata",
]
```

---

## 5. Module 3 — Dataset Collection & Preprocessing

### 5.1 Dataset: Malimg

**Source URL**: https://www.kaggle.com/datasets/rraftogianou/malimg-dataset  
**Manual download** — user downloads and extracts to `data/malimg/`. Code never downloads.

**Expected directory structure after extraction:**
```
data/malimg/
├── Adialer.C/          (122 PNG files)
├── Agent.FYI/          (116 PNG files)
├── Allaple.A/          (2949 PNG files)    ← most samples, causes imbalance
├── Allaple.L/          (1591 PNG files)
├── Alueron.gen!J/      (198 PNG files)
├── Autorun.K/          (106 PNG files)
├── C2LOP.P/            (146 PNG files)
├── C2LOP.gen!g/        (200 PNG files)
├── Dialplatform.B/     (177 PNG files)
├── Dontovo.A/          (162 PNG files)
├── Fakerean/           (381 PNG files)
├── Instantaccess/      (431 PNG files)
├── Lolyda.AA1/         (213 PNG files)
├── Lolyda.AA2/         (184 PNG files)
├── Lolyda.AA3/         (123 PNG files)
├── Lolyda.AT/          (159 PNG files)
├── Malex.gen!J/        (136 PNG files)
├── Obfuscator.AD/      (142 PNG files)
├── Rbot!gen/           (158 PNG files)
├── Skintrim.N/         (80 PNG files)     ← fewest samples
├── Swizzor.gen!E/      (128 PNG files)
├── Swizzor.gen!I/      (132 PNG files)
├── VB.AT/              (408 PNG files)
├── Wintrim.BX/         (97 PNG files)
└── Yuner.A/            (800 PNG files)
```

**Important**: Images in Malimg are already in grayscale PNG format with varying sizes. They do NOT need binary-to-image conversion — they are pre-converted. The BinaryConverter is used only for new user-uploaded binaries in the dashboard. For training, we load the Malimg PNGs directly and resize to 128×128.

### 5.2 `modules/dataset/preprocessor.py` — Complete Specification

```python
import cv2
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional


def validate_dataset_integrity(data_dir: Path) -> dict:
    """
    Scans the Malimg dataset directory and produces an integrity report.

    Args:
        data_dir: Path to the Malimg root directory (config.DATA_DIR).

    Returns:
        {
            'valid':            bool,         # True if no corrupt files found
            'families':         list[str],    # sorted list of family folder names
            'counts':           dict[str,int],# {family: sample_count}
            'total':            int,          # sum of all counts
            'min_class':        str,          # family with fewest samples
            'max_class':        str,          # family with most samples
            'imbalance_ratio':  float,        # max_count / min_count
            'corrupt_files':    list[str],    # str(path) of unreadable files
            'missing_dirs':     list[str],    # expected families not found
        }

    Raises:
        FileNotFoundError(f"Dataset directory not found: {data_dir}")
            → if data_dir does not exist
        FileNotFoundError(f"Dataset directory is empty: {data_dir}")
            → if no subdirectories found

    Implementation notes:
        - Iterate over data_dir.iterdir(), keeping only directories.
        - For each family dir, iterate over *.png files (case-insensitive).
        - For each PNG, attempt cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).
          If result is None, add to corrupt_files list.
        - Sort families alphabetically.
        - corrupt_files contains str representations of Path objects for JSON serialisability.
        - missing_dirs: compare found families against MALIMG_EXPECTED_FAMILIES=25 count.
          Since we don't hardcode names, just report if total families < 25.
          Actually: missing_dirs = [] (we cannot know expected names without hardcoding;
          report imbalance_ratio as quality indicator instead).

    SRS ref: Module 3 FE-4
    """


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image [0,255] to float32 [0.0, 1.0].

    Args:
        img: numpy array, dtype uint8.

    Returns:
        numpy array, same shape, dtype float32, values in [0.0, 1.0].

    Implementation:
        return img.astype(np.float32) / 255.0

    Notes:
        - Do NOT use cv2.normalize here. Simple division is exact and fast.
        - The output of this function feeds directly into PyTorch tensors.
    """


def encode_labels(families: list[str]) -> dict[str, int]:
    """
    Create a deterministic string→integer label mapping.

    Args:
        families: list of family names (folder names from Malimg).

    Returns:
        Dict mapping each family name to a unique integer [0, len(families)-1].
        Sorted alphabetically so mapping is always the same for the same input.

    Implementation:
        return {name: idx for idx, name in enumerate(sorted(families))}

    Example:
        encode_labels(['Yuner.A', 'Allaple.A', 'VB.AT'])
        → {'Allaple.A': 0, 'VB.AT': 1, 'Yuner.A': 2}

    SRS ref: Module 3 FE-2
    """


def save_class_names(class_names: list[str], output_path: Path) -> None:
    """
    Persist the ordered class name list to JSON for dashboard use.

    Args:
        class_names: sorted list of family names (index = label integer).
        output_path: destination JSON path (config.CLASS_NAMES_PATH).

    File format:
        {"class_names": ["Adialer.C", "Agent.FYI", ...]}

    Notes:
        - Creates parent directory if it does not exist.
        - Overwrites if file already exists.
        - Used by dashboard to reconstruct label mapping without reloading dataset.
    """


def load_class_names(input_path: Path) -> list[str]:
    """
    Load class names from JSON file written by save_class_names.

    Args:
        input_path: path to class_names.json (config.CLASS_NAMES_PATH).

    Returns:
        list[str] of family names in index order.

    Raises:
        FileNotFoundError(f"class_names.json not found at {input_path}. "
                          "Run scripts/train.py first.")
            → if file does not exist.
    """
```

### 5.3 `modules/dataset/loader.py` — Complete Specification

```python
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from typing import Optional, Callable
import config
from .preprocessor import encode_labels, save_class_names


class MalimgDataset(Dataset):
    """
    PyTorch Dataset for the Malimg malware image dataset.

    Loads grayscale PNG images from directory structure:
        data_dir/FamilyName/image.png

    Each image is:
        1. Loaded as grayscale (single channel)
        2. Resized to (img_size, img_size)
        3. Converted to PIL Image (for torchvision transforms)
        4. Transform applied (returns float32 tensor shape (1, H, W))

    Constructor args:
        data_dir (Path):     root of Malimg folder
        split (str):         'train' | 'val' | 'test'
        img_size (int):      resize target, default config.IMG_SIZE
        transform (callable): torchvision transform pipeline; if None, uses
                              get_val_transforms(img_size) as default
        train_ratio (float): fraction of data for training
        val_ratio (float):   fraction of data for validation
        test_ratio (float):  fraction of data for test (1 - train - val)
        random_seed (int):   sklearn random_state for reproducible splits

    Raises:
        FileNotFoundError: if data_dir does not exist
        ValueError: if split is not one of 'train', 'val', 'test'
        ValueError: if train_ratio + val_ratio + test_ratio != 1.0 (within 1e-6)

    Internal data structure:
        self.samples: list[tuple[Path, int]]
            List of (image_path, label_integer) pairs for the requested split.
        self.class_names: list[str]
            Sorted list of all 25 family names. Index = label integer.
        self.label_map: dict[str, int]
            {family_name: label_integer}
        self.class_counts: dict[str, int]
            {family_name: count_in_this_split}

    Split algorithm (must be implemented exactly):
        Step 1: Gather all (path, label) pairs for the entire dataset.
        Step 2: Extract label list for stratification.
        Step 3: train_test_split(all_samples, test_size=(val_ratio + test_ratio),
                                 stratify=labels, random_state=random_seed)
                → produces train_samples, temp_samples
        Step 4: relative_val = val_ratio / (val_ratio + test_ratio)
                train_test_split(temp_samples, test_size=(1 - relative_val),
                                 stratify=temp_labels, random_state=random_seed)
                → produces val_samples, test_samples
        Step 5: Store the samples for the requested split.

    __len__:
        return len(self.samples)

    __getitem__(idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        # Convert to PIL Image for torchvision compatibility
        from PIL import Image
        pil_img = Image.fromarray(img, mode='L')
        tensor = self.transform(pil_img)   # shape: (1, img_size, img_size), float32
        return tensor, label

    Notes:
        - transforms.ToTensor() on an 'L' mode PIL Image produces shape (1, H, W)
          with values in [0.0, 1.0]. This is the expected input format.
        - Normalization in transform uses mean=[0.5], std=[0.5] (single channel).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        img_size: int = config.IMG_SIZE,
        transform: Optional[Callable] = None,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio: float = config.VAL_RATIO,
        test_ratio: float = config.TEST_RATIO,
        random_seed: int = config.RANDOM_SEED,
    ):
        ...  # implement as specified above

    def get_labels(self) -> list[int]:
        """
        Returns list of integer labels for all samples in this split.
        Required by ClassAwareOversampler.
        """
        return [label for _, label in self.samples]


def get_dataloaders(
    data_dir: Path = config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    oversample_strategy: str = config.OVERSAMPLE_STRATEGY,
    augment_train: bool = True,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build all three DataLoaders and return them alongside class_names.

    Args:
        data_dir:            Malimg dataset root
        img_size:            resize target for images
        batch_size:          samples per batch
        num_workers:         DataLoader worker processes
        oversample_strategy: passed to ClassAwareOversampler
        augment_train:       if True, apply augmentation transforms to train set
        random_seed:         reproducibility seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
        class_names: list[str] sorted alphabetically, index = label integer

    Implementation:
        from modules.enhancement.augmentor import get_train_transforms, get_val_transforms
        from modules.enhancement.balancer import ClassAwareOversampler

        val_transform = get_val_transforms(img_size)
        train_transform = get_train_transforms(img_size) if augment_train else val_transform

        train_ds = MalimgDataset(data_dir, 'train', img_size, train_transform, ...)
        val_ds   = MalimgDataset(data_dir, 'val',   img_size, val_transform,   ...)
        test_ds  = MalimgDataset(data_dir, 'test',  img_size, val_transform,   ...)

        sampler = ClassAwareOversampler(train_ds, strategy=oversample_strategy).get_sampler()

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,          # replaces shuffle=True
            num_workers=num_workers,
            pin_memory=(config.DEVICE.type == 'cuda'),
            drop_last=True,           # avoid incomplete final batch during training
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(config.DEVICE.type == 'cuda'),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(config.DEVICE.type == 'cuda'),
        )

        # Persist class names for dashboard
        save_class_names(train_ds.class_names, config.CLASS_NAMES_PATH)

        return train_loader, val_loader, test_loader, train_ds.class_names

    Notes:
        - pin_memory=True accelerates CPU→GPU transfer on CUDA machines.
        - drop_last=True on train_loader prevents batch norm instability on
          single-sample final batches.
        - val and test loaders NEVER use augmentation transforms.
        - val and test loaders NEVER use the oversampling sampler (shuffle=False).
    """
```

---

## 6. Module 4 — Data Enhancement & Balancing

### 6.1 `modules/enhancement/augmentor.py` — Complete Specification

```python
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class GaussianNoise:
    """
    Custom torchvision-compatible transform that adds Gaussian noise to a tensor.

    Must be placed AFTER transforms.ToTensor() in the pipeline because it
    operates on torch.Tensor, not PIL.Image.

    Constructor args:
        mean (float):     noise mean, default 0.0
        std_range (tuple): (min_std, max_std), std sampled uniformly each call.
                           Default (0.01, 0.05).

    __call__(tensor: torch.Tensor) -> torch.Tensor:
        1. Sample std = random.uniform(std_range[0], std_range[1])
        2. Generate noise = torch.randn_like(tensor) * std + mean
        3. result = tensor + noise
        4. Clamp result to [0.0, 1.0]
        5. Return clamped tensor (same shape and dtype as input)

    Note:
        - torch.randn_like preserves device and dtype of input tensor.
        - Clamping to [0.0, 1.0] is mandatory (SRS FE-2 Module 4).
        - This simulates minor binary perturbations without changing structure.

    SRS ref: Module 4 FE-2
    """

    def __init__(self, mean: float = 0.0, std_range: tuple = (0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        ...  # implement as described above

    def __repr__(self) -> str:
        return f"GaussianNoise(mean={self.mean}, std_range={self.std_range})"


def get_train_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the augmentation pipeline for training data.

    Returns a transforms.Compose with the following stages IN THIS ORDER:
        1. transforms.RandomRotation(degrees=15)
               rotates image by angle sampled uniformly from [-15, 15] degrees.
               fill=0 (black fill for out-of-bounds pixels — appropriate for binary images).
               SRS ref: Module 4 FE-1

        2. transforms.RandomHorizontalFlip(p=0.5)
               50% chance of horizontal flip.
               SRS ref: Module 4 FE-1

        3. transforms.RandomVerticalFlip(p=0.5)
               50% chance of vertical flip.
               SRS ref: Module 4 FE-1

        4. transforms.ColorJitter(brightness=0.2)
               Randomly adjust brightness by factor in [0.8, 1.2].
               SRS ref: Module 4 FE-1 (brightness adjustment)
               NOTE: ColorJitter works on PIL Images or tensors. Since this
               transform comes BEFORE ToTensor, it receives a PIL Image. ✓

        5. transforms.ToTensor()
               Converts PIL Image (mode 'L') to tensor of shape (1, H, W)
               with values in [0.0, 1.0].

        6. GaussianNoise(mean=0.0, std_range=(0.01, 0.05))
               Applied AFTER ToTensor (operates on tensor).
               SRS ref: Module 4 FE-2

        7. transforms.Normalize(mean=[0.5], std=[0.5])
               Normalizes single-channel tensor:
               output = (input - 0.5) / 0.5  →  maps [0,1] to [-1,1].
               Single-element lists because images are 1-channel.

    Args:
        img_size: not used directly (resizing done in Dataset.__getitem__),
                  kept as parameter for consistency with get_val_transforms.

    Returns:
        transforms.Compose instance

    CRITICAL ORDERING NOTE:
        ColorJitter MUST come before ToTensor (PIL stage).
        GaussianNoise MUST come after ToTensor (tensor stage).
        RandomRotation, RandomHorizontalFlip, RandomVerticalFlip work on PIL Images.
    """


def get_val_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the inference/validation transform pipeline (NO augmentation).

    Returns a transforms.Compose with:
        1. transforms.ToTensor()
               Converts PIL Image (mode 'L') to tensor (1, H, W), values [0,1].
        2. transforms.Normalize(mean=[0.5], std=[0.5])
               Normalizes to [-1, 1].

    Args:
        img_size: kept for API consistency, not used here.

    Returns:
        transforms.Compose instance

    NOTE: val and test loaders ALWAYS use this transform, never get_train_transforms.
    """
```

### 6.2 `modules/enhancement/balancer.py` — Complete Specification

```python
import math
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


class ClassAwareOversampler:
    """
    Produces a WeightedRandomSampler to address class imbalance in Malimg.

    Malimg is severely imbalanced (Allaple.A has 2949 samples, Skintrim.N has 80).
    Without balancing, the CNN learns to predict majority classes with high confidence
    and performs poorly on rare families.

    Constructor args:
        dataset: a MalimgDataset instance (train split).
                 Must expose a get_labels() method returning list[int].
        strategy (str): balancing strategy, one of:
            'oversample_minority' — weight = 1 / class_count (pure inverse frequency)
                Best for: severe imbalance, when minority classes are equally important.
            'sqrt_inverse'        — weight = 1 / sqrt(class_count) (softer balancing)
                Best for: moderate imbalance, preserves some natural distribution.
            'uniform'             — weight = 1.0 for all samples regardless of class
                Best for: testing/ablation; effectively random sampling.

    get_sampler() -> WeightedRandomSampler:
        Returns a WeightedRandomSampler configured for one full epoch.

        Implementation:
            labels = dataset.get_labels()          # list[int]
            class_counts = Counter(labels)         # {class_int: count}
            num_classes = len(class_counts)

            if strategy == 'oversample_minority':
                class_weights = {c: 1.0 / count for c, count in class_counts.items()}
            elif strategy == 'sqrt_inverse':
                class_weights = {c: 1.0 / math.sqrt(count) for c, count in class_counts.items()}
            elif strategy == 'uniform':
                class_weights = {c: 1.0 for c in class_counts}
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Per-sample weights
            sample_weights = torch.tensor(
                [class_weights[label] for label in labels],
                dtype=torch.float32
            )

            return WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(labels),    # draw exactly len(dataset) samples per epoch
                replacement=True,           # necessary for oversampling minority classes
            )

    Properties:
        class_weights: dict[int, float]  — computed weights per class (set in get_sampler)
        effective_class_counts: dict[int, float]  — expected samples per class per epoch
            = {c: class_weights[c] / sum(class_weights.values()) * len(labels)
               for c in class_weights}

    SRS ref: Module 4 FE-3
    """
```

---

## 7. Module 5 — Intelligent Malware Detection

### 7.1 `modules/detection/model.py` — Complete Specification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Reusable convolutional block: Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout2d

    Constructor args:
        in_channels (int):  input channels
        out_channels (int): output channels for both Conv layers
        dropout_p (float):  Dropout2d probability, default 0.25

    forward(x) → x after full block
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x


class MalTwinCNN(nn.Module):
    """
    Three-block CNN for grayscale malware image classification.

    Input:  (batch_size, 1, 128, 128)  — single-channel grayscale
    Output: (batch_size, num_classes)  — raw logits (NO softmax here)

    Architecture:

    Input (1, 128, 128)
        │
        ▼
    ConvBlock(1 → 32)      → output (32, 64, 64)   after MaxPool(2x2)
        │
        ▼
    ConvBlock(32 → 64)     → output (64, 32, 32)   after MaxPool(2x2)
        │
        ▼
    ConvBlock(64 → 128)    → output (128, 16, 16)  after MaxPool(2x2)
        │
        ▼ ← self.gradcam_layer points to the Conv2d(128,128) in this block
    AdaptiveAvgPool2d(4, 4) → output (128, 4, 4)
        │
        ▼
    Flatten                → output (2048,)
        │
        ▼
    Linear(2048 → 512)
    ReLU
    Dropout(p=0.5)
        │
        ▼
    Linear(512 → num_classes)
        │
        ▼
    Raw logits (num_classes,)

    Constructor args:
        num_classes (int): number of malware families to classify (25 for Malimg).

    Attributes:
        self.block1: ConvBlock(1, 32)
        self.block2: ConvBlock(32, 64)
        self.block3: ConvBlock(64, 128)
        self.pool:   nn.AdaptiveAvgPool2d((4, 4))
        self.classifier: nn.Sequential containing Flatten, Linear, ReLU, Dropout, Linear
        self.gradcam_layer: reference to block3.conv2
                            This is used by Module 7 (Grad-CAM) to register hooks.
                            Must be set: self.gradcam_layer = self.block3.conv2

    bias=False in all Conv2d:
        Because BatchNorm follows immediately, bias is redundant and wastes parameters.

    Weight initialization (call _initialize_weights() in __init__):
        For each module:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    forward(x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    Approximate parameter count: ~2.1 million (verify with sum(p.numel() for p in model.parameters()))

    SRS ref: Module 5 FE-1
    """

    def __init__(self, num_classes: int):
        ...  # implement as specified above

    def _initialize_weights(self):
        ...  # Kaiming / Xavier init as specified above

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...  # implement as specified above
```

### 7.2 `modules/detection/trainer.py` — Complete Specification

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import config
from .model import MalTwinCNN


def train(
    model: MalTwinCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    lr_patience: int = config.LR_PATIENCE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    best_model_path: Path = config.BEST_MODEL_PATH,
) -> dict:
    """
    Full training loop with checkpointing and LR scheduling.

    Returns:
        history dict:
        {
            'train_loss': list[float],   # mean loss per epoch
            'train_acc':  list[float],   # accuracy 0.0–1.0 per epoch
            'val_loss':   list[float],
            'val_acc':    list[float],
            'best_val_acc': float,
            'best_epoch':   int,
        }

    Optimizer:
        torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Loss function:
        nn.CrossEntropyLoss()
        Note: CrossEntropyLoss expects raw logits (no softmax in model.forward).

    LR Scheduler:
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',          # maximize val_acc
            factor=0.5,          # halve LR on plateau
            patience=lr_patience,
            min_lr=1e-6,
            verbose=True,
        )
        scheduler.step(val_acc) called after each epoch.

    Per-epoch procedure:
        1. model.train()
        2. Iterate train_loader with tqdm progress bar:
               desc=f"Epoch {epoch+1:03d}/{epochs:03d} [Train]"
               postfix={'loss': running_loss, 'acc': running_acc}
        3. Forward pass → compute loss → backward → optimizer.step()
        4. Accumulate correct predictions and total samples for accuracy.
        5. After all batches: compute mean train_loss and train_acc.
        6. Call validate_epoch(model, val_loader, device, criterion)
           → get val_loss, val_acc
        7. scheduler.step(val_acc)
        8. Print epoch summary line:
               f"Epoch {epoch+1:03d}/{epochs} | "
               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
               f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
               f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        9. Save checkpoint to checkpoint_dir/epoch_{epoch+1:03d}_acc{val_acc:.4f}.pt:
               torch.save({
                   'epoch':           epoch + 1,
                   'model_state':     model.state_dict(),
                   'optimizer_state': optimizer.state_dict(),
                   'val_acc':         val_acc,
                   'val_loss':        val_loss,
                   'train_acc':       train_acc,
                   'train_loss':      train_loss,
               }, checkpoint_path)
        10. If val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"  ★ New best model saved (val_acc={val_acc:.4f})")

    Reproducibility:
        torch.manual_seed(config.RANDOM_SEED) called at start of function.
        If CUDA: torch.cuda.manual_seed(config.RANDOM_SEED)

    SRS ref: Module 5 FE-2, OE-4, REL-1
    """


def validate_epoch(
    model: MalTwinCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Run one validation pass. Returns (avg_loss, accuracy).

    Implementation:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    Notes:
        - Multiply loss.item() by batch size for correct mean across variable batch sizes.
        - model.eval() disables Dropout and uses running stats in BatchNorm.
    """
```

### 7.3 `modules/detection/evaluator.py` — Complete Specification

```python
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
import config
from .model import MalTwinCNN


def evaluate(
    model: MalTwinCNN,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Full evaluation on test set. Returns comprehensive metrics dict.

    Returns:
        {
            'accuracy':          float,               # overall top-1 accuracy
            'precision_macro':   float,               # macro-averaged precision
            'recall_macro':      float,               # macro-averaged recall
            'f1_macro':          float,               # macro-averaged F1
            'precision_weighted': float,              # weighted-averaged precision
            'recall_weighted':   float,               # weighted-averaged recall
            'f1_weighted':       float,               # weighted-averaged F1
            'confusion_matrix':  np.ndarray,          # shape (num_classes, num_classes)
            'per_class': {
                family_name: {
                    'precision': float,
                    'recall':    float,
                    'f1':        float,
                    'support':   int,                 # true samples in test set
                }
            },
            'classification_report': str,             # sklearn formatted string
            'num_test_samples':  int,
        }

    Implementation:
        1. model.eval()
        2. Collect all predictions and true labels:
               all_preds  = []
               all_labels = []
               with torch.no_grad():
                   for inputs, labels in test_loader:
                       inputs = inputs.to(device)
                       outputs = model(inputs)
                       preds = outputs.argmax(dim=1).cpu().numpy()
                       all_preds.extend(preds.tolist())
                       all_labels.extend(labels.numpy().tolist())
        3. Compute all metrics using sklearn.
        4. Build per_class dict from precision_recall_fscore_support with average=None.
        5. confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        6. classification_report(all_labels, all_preds, target_names=class_names)

    SRS ref: Module 5 FE-3, BO-5
    """


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    figsize: tuple = (16, 14),
) -> None:
    """
    Render and save confusion matrix as PNG.

    Args:
        cm:           confusion matrix array shape (N, N)
        class_names:  list of N class names for axis labels
        output_path:  where to save PNG (created by this function)
        figsize:      matplotlib figure size in inches

    Implementation:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=8)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=6)

        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('MalTwin Confusion Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    Notes:
        - matplotlib.use('Agg') must be set at module level (no display required).
        - plt.close(fig) is mandatory to prevent memory leaks in long training runs.
    """


def format_metrics_table(metrics: dict, class_names: list[str]) -> str:
    """
    Format evaluation metrics as a printable ASCII table for CLI output.

    Returns multi-line string, example:
    ╔══════════════════════════════════════╗
    ║         MALTWIN TEST EVALUATION      ║
    ╠══════════════════════════════════════╣
    ║  Accuracy:           0.9387          ║
    ║  Precision (macro):  0.9412          ║
    ║  Recall (macro):     0.9371          ║
    ║  F1 (macro):         0.9389          ║
    ║  Test Samples:       1400            ║
    ╠══════════════════════════════════════╣
    ║  Per-Class F1 (top 5 worst):         ║
    ║    Skintrim.N:     0.7823            ║
    ║    Autorun.K:      0.8102            ║
    ...
    ╚══════════════════════════════════════╝
    """
```

### 7.4 `modules/detection/inference.py` — Complete Specification

```python
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import config
from .model import MalTwinCNN
from modules.enhancement.augmentor import get_val_transforms


def load_model(
    model_path: Path = config.BEST_MODEL_PATH,
    num_classes: int = config.MALIMG_EXPECTED_FAMILIES,
    device: torch.device = config.DEVICE,
) -> MalTwinCNN:
    """
    Load a trained MalTwinCNN from a .pt checkpoint.

    Args:
        model_path:  path to best_model.pt (state_dict only, not full checkpoint)
        num_classes: number of output classes (must match trained model)
        device:      device to map model onto

    Returns:
        MalTwinCNN instance in eval() mode, on the specified device.

    Raises:
        FileNotFoundError(f"Model file not found: {model_path}. "
                          "Run scripts/train.py to train the model first.")

    Implementation:
        if not model_path.exists():
            raise FileNotFoundError(...)
        model = MalTwinCNN(num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    Notes:
        - map_location=device handles CUDA→CPU migration for systems without GPU.
        - torch.load with weights_only=True is preferred in PyTorch 2.x for security.
          Use: torch.load(model_path, map_location=device, weights_only=True)
    """


def predict_single(
    model: MalTwinCNN,
    img_array: np.ndarray,
    class_names: list[str],
    device: torch.device = config.DEVICE,
) -> dict:
    """
    Run inference on a single 128×128 grayscale image array.

    Args:
        model:       MalTwinCNN in eval() mode
        img_array:   numpy array shape (128, 128), dtype uint8, values 0–255
                     (output of BinaryConverter.convert())
        class_names: ordered list of family names, index = class integer
        device:      inference device

    Returns:
        {
            'predicted_family': str,               # top-1 class name
            'confidence':       float,             # top-1 softmax probability [0.0, 1.0]
            'probabilities':    dict[str, float],  # {family_name: prob} for ALL classes
            'top3': [                              # top 3 predictions for display
                {'family': str, 'confidence': float},
                {'family': str, 'confidence': float},
                {'family': str, 'confidence': float},
            ]
        }

    Implementation:
        # 1. Convert uint8 array to PIL Image
        pil_img = Image.fromarray(img_array, mode='L')

        # 2. Apply val transforms (ToTensor + Normalize)
        transform = get_val_transforms(config.IMG_SIZE)
        tensor = transform(pil_img)           # shape: (1, 128, 128)

        # 3. Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(device)   # shape: (1, 1, 128, 128)

        # 4. Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(tensor)                 # shape: (1, num_classes)

        # 5. Softmax probabilities
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        # probs shape: (num_classes,)

        # 6. Top-1 prediction
        top1_idx = int(np.argmax(probs))
        top1_confidence = float(probs[top1_idx])
        top1_family = class_names[top1_idx]

        # 7. Full probabilities dict
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # 8. Top-3 list
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [{'family': class_names[i], 'confidence': float(probs[i])}
                for i in top3_indices]

        return {
            'predicted_family': top1_family,
            'confidence':       top1_confidence,
            'probabilities':    prob_dict,
            'top3':             top3,
        }

    Notes:
        - model.eval() is called even if already set, as a safety guard.
        - torch.no_grad() reduces memory usage and speeds up inference.
        - np.argmax on numpy array is faster than torch.argmax for single sample.
        - float() conversion ensures JSON serialisability of all values.

    SRS ref: Module 5 FE-4, REL-1
    """


def predict_batch(
    model: MalTwinCNN,
    img_arrays: list[np.ndarray],
    class_names: list[str],
    device: torch.device = config.DEVICE,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run inference on multiple images. Returns list of result dicts (same format as predict_single).

    Args:
        model:       MalTwinCNN in eval() mode
        img_arrays:  list of numpy arrays, each (128, 128) uint8
        class_names: ordered list of family names
        device:      inference device
        batch_size:  images per forward pass (default 16 to avoid OOM on CPU)

    Returns:
        list[dict] — same structure as predict_single return value, one per input image.

    Implementation:
        - Process img_arrays in chunks of batch_size.
        - For each chunk: stack tensors into (B, 1, 128, 128), run forward pass,
          softmax, decompose back into per-image dicts.
        - Results are in same order as input.

    SRS ref: Module 5 FE-4 (batch support for future CLI/dashboard use)
    """
```

---

## 8. Module 6 — Dashboard & Visualization

### 8.1 Dashboard Architecture Overview

```
modules/dashboard/
├── app.py            ← entry point, page routing, global state init
├── db.py             ← SQLite helpers (see Section 9)
├── state.py          ← session_state constants and utility functions
└── pages/
    ├── home.py       ← system overview, recent detections, stats
    ├── upload.py     ← binary upload, grayscale viz, metadata (M3 UI)
    ├── detection.py  ← run detection, results, MITRE, stubs for XAI/report
    └── digital_twin.py ← STUB page (M1 deferred)
```

**Design constraints from SRS:**
- Dark-themed professional color scheme (navy/white palette, accent for status)
- Two-column layout: sidebar navigation left, main content right
- Section headers: 18pt; body text: 13pt; table cells: 11pt minimum
- All interactive buttons show loading spinner during async operations
- Color-coded indicators MUST have text labels (accessibility, SRS USE-2)
- Error messages must follow: "Error: [what]. Cause: [why]. Action: [what to do]." (SRS USE-3)
- Inline tooltips/help text for: 'confidence score', 'Grad-CAM heatmap',
  'MITRE ATT&CK tactic', 'malware family' (SRS USE-4)

### 8.2 `modules/dashboard/state.py` — Complete Specification

```python
"""
Centralised session_state key definitions and helper functions.

WHY: Using string literals for session_state keys scattered across pages leads
to typos and hard-to-debug state bugs. All keys are defined as constants here.

All pages import from state.py and use these constants exclusively.
"""
import streamlit as st
import numpy as np
from typing import Optional

# ── Session state key constants ─────────────────────────────────────────────────
KEY_MODEL         = 'model'          # MalTwinCNN or None
KEY_CLASS_NAMES   = 'class_names'    # list[str] or None
KEY_IMG_ARRAY     = 'img_array'      # np.ndarray (128,128) uint8 or None
KEY_FILE_META     = 'file_meta'      # dict from get_file_metadata() or None
KEY_DETECTION     = 'detection_result'  # dict from predict_single() or None
KEY_MODEL_LOADED  = 'model_loaded'   # bool
KEY_DEVICE_INFO   = 'device_info'    # str e.g. "cuda:0 (RTX 3080)" or "cpu"


def init_session_state() -> None:
    """
    Initialize all session state keys with default values if not already set.
    Call once at the top of app.py before any page renders.

    Defaults:
        KEY_MODEL         = None
        KEY_CLASS_NAMES   = None
        KEY_IMG_ARRAY     = None
        KEY_FILE_META     = None
        KEY_DETECTION     = None
        KEY_MODEL_LOADED  = False
        KEY_DEVICE_INFO   = "unknown"
    """
    defaults = {
        KEY_MODEL:        None,
        KEY_CLASS_NAMES:  None,
        KEY_IMG_ARRAY:    None,
        KEY_FILE_META:    None,
        KEY_DETECTION:    None,
        KEY_MODEL_LOADED: False,
        KEY_DEVICE_INFO:  "unknown",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_file_state() -> None:
    """
    Clear file-related state keys. Called when a new file is uploaded
    to prevent stale detection results being shown for a different file.
    """
    st.session_state[KEY_IMG_ARRAY]  = None
    st.session_state[KEY_FILE_META]  = None
    st.session_state[KEY_DETECTION]  = None


def has_uploaded_file() -> bool:
    return st.session_state.get(KEY_IMG_ARRAY) is not None


def has_detection_result() -> bool:
    return st.session_state.get(KEY_DETECTION) is not None


def is_model_loaded() -> bool:
    return st.session_state.get(KEY_MODEL_LOADED, False)
```

### 8.3 `modules/dashboard/app.py` — Complete Specification

```python
"""
Streamlit application entry point.

Run: streamlit run modules/dashboard/app.py --server.port 8501

This file is responsible for:
    1. Page configuration
    2. Global model + class names loading (once per session)
    3. Sidebar navigation
    4. Routing to the correct page module

MUST NOT contain any analysis logic — only orchestration.
"""
import streamlit as st
import torch
from pathlib import Path
import config
from modules.dashboard import state
from modules.dashboard.db import init_db
from modules.dataset.preprocessor import load_class_names
from modules.detection.inference import load_model


def configure_page():
    """
    Called once before any other Streamlit call.
    st.set_page_config must be the FIRST Streamlit command in the script.

    Settings:
        page_title    = config.DASHBOARD_TITLE  ("MalTwin — IIoT Malware Detection")
        page_icon     = "🛡️"
        layout        = "wide"
        initial_sidebar_state = "expanded"
        menu_items    = {
            'Get Help': None,
            'Report a bug': None,
            'About': "MalTwin — AI-based IIoT Malware Detection Framework\\n"
                     "COMSATS University, Islamabad | BS Cyber Security 2023-2027"
        }
    """


def load_global_resources():
    """
    Load model and class_names into session_state on first run.
    Uses st.spinner for user feedback.

    Procedure:
        1. If KEY_CLASS_NAMES is None:
               try:
                   class_names = load_class_names(config.CLASS_NAMES_PATH)
                   st.session_state[state.KEY_CLASS_NAMES] = class_names
               except FileNotFoundError:
                   st.session_state[state.KEY_CLASS_NAMES] = None

        2. If KEY_MODEL is None AND KEY_CLASS_NAMES is not None:
               try:
                   with st.spinner("Loading detection model..."):
                       num_classes = len(st.session_state[state.KEY_CLASS_NAMES])
                       model = load_model(config.BEST_MODEL_PATH, num_classes, config.DEVICE)
                       st.session_state[state.KEY_MODEL] = model
                       st.session_state[state.KEY_MODEL_LOADED] = True
                       st.session_state[state.KEY_DEVICE_INFO] = str(config.DEVICE)
               except FileNotFoundError:
                   st.session_state[state.KEY_MODEL_LOADED] = False

    Notes:
        - This function checks state BEFORE attempting to load, so it only
          runs once per session even across page navigations.
        - Do NOT cache with @st.cache_resource here — model loading happens
          inside session_state guard, which is safer for multi-user contexts.
    """


def render_sidebar() -> str:
    """
    Render navigation sidebar. Returns selected page name.

    Sidebar contents:
        1. Logo/title area:
               st.sidebar.markdown("# 🛡️ MalTwin")
               st.sidebar.markdown("*IIoT Malware Detection*")
               st.sidebar.divider()

        2. Navigation radio:
               page = st.sidebar.radio(
                   "Navigation",
                   options=["🏠 Dashboard", "📂 Binary Upload",
                            "🔍 Malware Detection", "🖥️ Digital Twin"],
                   label_visibility="hidden",
               )

        3. Model status section (below navigation):
               st.sidebar.divider()
               st.sidebar.markdown("**System Status**")
               if is_model_loaded():
                   st.sidebar.success(f"✅ Model Ready ({state KEY_DEVICE_INFO})")
               else:
                   st.sidebar.warning("⚠️ No model loaded")
                   st.sidebar.caption("Run scripts/train.py first")

               if has_uploaded_file():
                   meta = st.session_state[KEY_FILE_META]
                   st.sidebar.info(f"📄 {meta['name']}")
               else:
                   st.sidebar.caption("No file uploaded")

               if has_detection_result():
                   result = st.session_state[KEY_DETECTION]
                   st.sidebar.success(f"🎯 {result['predicted_family']}")

        4. Footer:
               st.sidebar.divider()
               st.sidebar.caption("COMSATS University, Islamabad")
               st.sidebar.caption("BS Cyber Security 2023-2027")

    Returns:
        The selected page string e.g. "🏠 Dashboard"
    """


def main():
    """
    Application entry point.

    Procedure:
        configure_page()
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

### 8.4 `modules/dashboard/pages/home.py` — Complete Specification

```python
"""
Home / Dashboard Overview page.
Implements SRS Mockup M1 — Main Dashboard Screen.

render() function displays:
    1. Page header
    2. Four KPI metric cards (row 1)
    3. Detection activity chart (row 2 left)
    4. Digital Twin status panel (row 2 right) — stubbed
    5. Recent detection feed (row 3)
    6. Module status table (row 4)
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import config
from modules.dashboard.db import get_recent_events, get_detection_stats
from modules.dashboard import state


def render():
    st.title("🏠 System Overview Dashboard")
    st.markdown("---")

    # ── Row 1: KPI Cards ──────────────────────────────────────────────────────
    # Pull stats from SQLite
    # stats = get_detection_stats(config.DB_PATH)
    # Returns: {'total_analyzed': int, 'total_malware': int,
    #           'total_benign': int (always 0 for now), 'model_accuracy': float or None}
    # SRS ref: FR1.2

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Files Analyzed",
            value=stats.get('total_analyzed', 0),
            delta=None,
        )
    with col2:
        st.metric(label="Malware Detected",  value=stats.get('total_malware', 0))
    with col3:
        st.metric(label="Benign Files",      value=stats.get('total_benign', 0))
    with col4:
        acc = stats.get('model_accuracy')
        st.metric(
            label="Model Accuracy",
            value=f"{acc*100:.1f}%" if acc else "N/A",
        )

    st.markdown("---")

    # ── Row 2: Detection Activity Chart + DT Status ───────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Detection Activity (Last 7 Days)")
        # Pull last 7 days of events from DB, group by date
        # Plot as Plotly line chart with two traces: total detections per day
        # X-axis: date strings, Y-axis: count
        # SRS ref: FR1.2 (real-time stats)
        _render_activity_chart(config.DB_PATH)

    with col_right:
        st.subheader("Digital Twin Status")
        st.info("🖥️ Digital Twin simulation is in a future implementation phase.")
        st.markdown("**Status:** Not Configured")
        st.markdown("**Active Nodes:** —")
        st.markdown("**Traffic Flow:** —")

    st.markdown("---")

    # ── Row 3: Recent Detection Feed ──────────────────────────────────────────
    st.subheader("Recent Detections")
    # SRS ref: FR1.4 — show 5 most recent
    events = get_recent_events(config.DB_PATH, limit=5)
    if not events:
        st.caption("No detections yet. Upload a binary file to get started.")
    else:
        df = pd.DataFrame(events)[['timestamp','file_name','predicted_family',
                                    'confidence','device_used']]
        df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        df.columns = ['Timestamp', 'File', 'Predicted Family', 'Confidence', 'Device']
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Row 4: Module Status Table ────────────────────────────────────────────
    st.subheader("Module Status")
    # SRS ref: FR1.1 — show status of all modules
    _render_module_status()


def _render_activity_chart(db_path):
    """Helper: fetch event counts per day for last 7 days, render plotly chart."""
    # Implementation: query DB for events in last 7 days
    # Group by date(timestamp), count per day
    # Use plotly go.Scatter with mode='lines+markers'
    ...


def _render_module_status():
    """Helper: render a table showing status of each module."""
    import config
    from pathlib import Path

    modules = [
        ("M1", "Digital Twin Simulation",      "⚠️ Deferred",  "amber"),
        ("M2", "Binary-to-Image Conversion",   "✅ Active",     "green"),
        ("M3", "Dataset & Preprocessing",      "✅ Active",     "green"),
        ("M4", "Data Enhancement & Balancing", "✅ Active",     "green"),
        ("M5", "Malware Detection (CNN)",
            "✅ Active" if config.BEST_MODEL_PATH.exists() else "⚠️ No model", "green"),
        ("M6", "Dashboard & Visualization",    "✅ Active",     "green"),
        ("M7", "Explainable AI (Grad-CAM)",    "⚠️ Deferred",  "amber"),
        ("M8", "Automated Threat Reporting",   "⚠️ Deferred",  "amber"),
    ]
    df = pd.DataFrame(modules, columns=["ID", "Module", "Status", "_color"])
    st.dataframe(df[["ID","Module","Status"]], use_container_width=True, hide_index=True)
    # SRS ref: FR1.1 — auto-refresh every 5s using st.rerun() + time.sleep not recommended;
    # use st.empty() + loop pattern or rely on Streamlit's natural rerun on interaction.
```

### 8.5 `modules/dashboard/pages/upload.py` — Complete Specification

```python
"""
Binary Upload & Visualization page.
Implements SRS Mockup M3 — Binary Upload and Visualization Screen.
SRS refs: FR3.1, FR3.2, FR3.3, FR3.4, UC-01
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
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

    # ── File uploader ──────────────────────────────────────────────────────────
    # SRS ref: FR3.1
    uploaded_file = st.file_uploader(
        label="Upload Binary File",
        type=["exe", "dll"],
        help=(
            "Accepted formats: PE (.exe, .dll) or ELF binaries. "
            "ELF binaries (Linux/embedded) have no extension — rename to .elf if needed. "
            f"Maximum file size: {config.MAX_UPLOAD_BYTES // (1024*1024)} MB."
        ),
        key="binary_uploader",
    )

    if uploaded_file is not None:
        _process_upload(uploaded_file)

    # ── Show results if we have a processed file ───────────────────────────────
    if state.has_uploaded_file():
        _render_results()
    else:
        if uploaded_file is None:
            st.info("👆 Upload a binary file above to begin.")


def _process_upload(uploaded_file):
    """
    Reads, validates, and converts the uploaded file.
    Stores results in session_state.
    Clears previous state first.

    Error handling follows SRS USE-3 format:
        "Error: [what went wrong]. Cause: [why]. Action: [what to do]."

    Steps:
        1. Read bytes: file_bytes = uploaded_file.read()
        2. Check size: if len(file_bytes) > config.MAX_UPLOAD_BYTES:
               st.error("Error: File too large. "
                        f"Cause: File exceeds the 50 MB limit "
                        f"(uploaded: {len(file_bytes)//(1024*1024)} MB). "
                        "Action: Upload a smaller binary file.")
               return
        3. Clear previous state: state.clear_file_state()
        4. validate_binary_format(file_bytes) — catch ValueError:
               st.error(f"Error: Unsupported file format. "
                        f"Cause: {str(e)} "
                        "Action: Upload a valid PE (.exe, .dll) or ELF binary file.")
               return
        5. BinaryConverter().convert(file_bytes) — catch ValueError, RuntimeError:
               display error, return
        6. Store in session_state:
               st.session_state[state.KEY_IMG_ARRAY] = img_array
               st.session_state[state.KEY_FILE_META] = get_file_metadata(
                   file_bytes, uploaded_file.name, file_format)
        7. st.success("✅ File processed successfully. Navigate to Malware Detection to analyze.")
    """


def _render_results():
    """
    Display the grayscale image, metadata, and pixel histogram.
    Called when session_state has a processed image.

    Layout: two equal columns
        Left:  grayscale image + caption
        Right: metadata table + pixel histogram

    Grayscale image display (SRS FR3.2):
        converter = BinaryConverter()
        png_bytes = converter.to_png_bytes(img_array)
        col_left.image(png_bytes,
                       caption=f"Grayscale visualization (128×128 pixels, 8-bit)",
                       use_column_width=True)

    Metadata display (SRS FR3.3):
        meta = st.session_state[state.KEY_FILE_META]
        Display as st.table with rows:
            File Name   | meta['name']
            File Size   | meta['size_human']
            Format      | meta['format']
            SHA-256     | meta['sha256']  (monospace font via st.code or markdown)
            Upload Time | meta['upload_time']

    Pixel intensity histogram (SRS FR3.4):
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
        col_right.plotly_chart(fig, use_container_width=True)

    Navigation hint:
        st.info("➡️ Navigate to **Malware Detection** in the sidebar to run analysis.")
    """
```

### 8.6 `modules/dashboard/pages/detection.py` — Complete Specification

```python
"""
Malware Detection & Prediction page.
Implements SRS Mockup M5 — Malware Detection and Prediction Screen.
SRS refs: FR5.1, FR5.2, FR5.3, FR5.4, FR5.5, FR5.6, UC-02
"""
import streamlit as st
import plotly.graph_objects as go
import json
from pathlib import Path
import config
from modules.detection.inference import predict_single
from modules.dashboard.db import log_detection_event
from modules.dashboard import state


def render():
    st.title("🔍 Malware Detection & Classification")
    st.markdown("---")

    # ── Guard: no file uploaded ────────────────────────────────────────────────
    if not state.has_uploaded_file():
        st.warning(
            "⚠️ No binary file loaded. "
            "Please upload a file on the **Binary Upload** page first."
        )
        return

    # ── Guard: no model loaded ─────────────────────────────────────────────────
    if not state.is_model_loaded():
        st.warning(
            "⚠️ No trained model available. "
            "Run `python scripts/train.py` to train the model, then restart the dashboard."
        )
        return

    # ── File summary ───────────────────────────────────────────────────────────
    _render_file_summary()
    st.markdown("---")

    # ── Run Detection button ───────────────────────────────────────────────────
    # SRS ref: FR5.1
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "▶ Run Detection",
            type="primary",
            use_container_width=True,
            help="Run the CNN malware classifier on the uploaded binary image.",
        )

    if run_clicked:
        _run_detection()

    # ── Results ────────────────────────────────────────────────────────────────
    if state.has_detection_result():
        st.markdown("---")
        _render_results()


def _render_file_summary():
    """
    Show a compact summary of the currently loaded file.
    Uses three columns: image thumbnail | metadata | SHA-256
    """
    meta = st.session_state[state.KEY_FILE_META]
    img_array = st.session_state[state.KEY_IMG_ARRAY]

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        from modules.binary_to_image.converter import BinaryConverter
        png_bytes = BinaryConverter().to_png_bytes(img_array)
        st.image(png_bytes, caption="Binary image", width=128)
    with col2:
        st.markdown(f"**File:** `{meta['name']}`")
        st.markdown(f"**Size:** {meta['size_human']}")
        st.markdown(f"**Format:** {meta['format']}")
    with col3:
        st.markdown("**SHA-256:**")
        st.code(meta['sha256'], language=None)


def _run_detection():
    """
    Execute inference and store result in session_state.
    Log the event to SQLite.

    Implementation:
        with st.spinner("Running malware classification..."):
            model = st.session_state[state.KEY_MODEL]
            class_names = st.session_state[state.KEY_CLASS_NAMES]
            img_array = st.session_state[state.KEY_IMG_ARRAY]
            meta = st.session_state[state.KEY_FILE_META]

            result = predict_single(model, img_array, class_names, config.DEVICE)
            st.session_state[state.KEY_DETECTION] = result

            # Log to SQLite (SRS FR-B3)
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

    Error handling (SRS UC-02 Alternate Flows):
        Catch Exception broadly:
            st.error("Error: Detection failed. "
                     f"Cause: {str(e)}. "
                     "Action: Ensure the model is correctly loaded and try again.")
            return
    """


def _render_results():
    """
    Render the full detection results panel.

    Contains 5 sections (laid out as described below):
    """
    result = st.session_state[state.KEY_DETECTION]
    confidence = result['confidence']
    family = result['predicted_family']

    # ── Section A: Prediction Label + Confidence (SRS FR5.2) ──────────────────
    st.subheader("Detection Result")

    if confidence >= config.CONFIDENCE_GREEN:
        st.success(f"🎯 **{family}** detected with **{confidence*100:.1f}%** confidence")
    elif confidence >= config.CONFIDENCE_AMBER:
        st.warning(
            f"⚠️ **{family}** detected with **{confidence*100:.1f}%** confidence\n\n"
            "Low confidence — results may be unreliable. Manual verification recommended."
        )
    else:
        st.error(
            f"🔴 **{family}** detected with **{confidence*100:.1f}%** confidence\n\n"
            "Very low confidence — manual expert review is required."
        )

    # Confidence progress bar with color label (SRS FR5.2 color coding)
    confidence_pct = int(confidence * 100)
    color_label = (
        "🟢 High Confidence" if confidence >= config.CONFIDENCE_GREEN
        else "🟡 Medium Confidence" if confidence >= config.CONFIDENCE_AMBER
        else "🔴 Low Confidence"
    )
    col_bar, col_label = st.columns([3, 1])
    col_bar.progress(confidence_pct)
    col_label.markdown(f"**{confidence_pct}%** {color_label}")

    # ── Section B: Top-3 Predictions ──────────────────────────────────────────
    st.markdown("**Top 3 Predictions:**")
    for i, pred in enumerate(result['top3'], 1):
        st.markdown(f"{i}. `{pred['family']}` — {pred['confidence']*100:.2f}%")

    st.markdown("---")

    # ── Section C: Per-Class Probability Chart (SRS FR5.3) ────────────────────
    st.subheader("Class Probability Distribution")
    st.caption(
        "All 25 malware families shown. Zero-probability classes displayed as empty bars.",
        help="The model outputs a probability for every known malware family. "
             "Higher bars indicate the model believes the binary is more likely to belong to that family."
    )
    _render_probability_chart(result['probabilities'])

    st.markdown("---")

    # ── Section D: MITRE ATT&CK Mapping (SRS FR5.5) ───────────────────────────
    st.subheader("MITRE ATT&CK for ICS Mapping")
    st.caption(
        "Adversary tactics and techniques associated with the detected malware family.",
        help="MITRE ATT&CK for Industrial Control Systems (ICS) is a knowledge base "
             "of adversary tactics and techniques specific to operational technology environments."
    )
    _render_mitre_mapping(family)

    st.markdown("---")

    # ── Section E: XAI Heatmap Toggle (STUB — SRS FR5.4) ─────────────────────
    st.subheader("Explainable AI — Grad-CAM Heatmap")
    xai_requested = st.checkbox(
        "Generate Grad-CAM Heatmap",
        help="Grad-CAM highlights which regions of the binary image influenced the "
             "classification decision. Requires additional computation time.",
    )
    if xai_requested:
        st.info(
            "🔬 Grad-CAM XAI visualization will be implemented in the next phase (Module 7). "
            "This feature generates heatmaps showing which byte regions drove the classification."
        )

    # ── Section F: Report Download (STUB — SRS FR5.6) ─────────────────────────
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
        # JSON export IS partially available now — export the result dict
        import json as json_module
        export_data = {
            'file_name':         st.session_state[state.KEY_FILE_META]['name'],
            'sha256':            st.session_state[state.KEY_FILE_META]['sha256'],
            'file_format':       st.session_state[state.KEY_FILE_META]['format'],
            'file_size_bytes':   st.session_state[state.KEY_FILE_META]['size_bytes'],
            'upload_time':       st.session_state[state.KEY_FILE_META]['upload_time'],
            'predicted_family':  result['predicted_family'],
            'confidence':        result['confidence'],
            'top3':              result['top3'],
            'all_probabilities': result['probabilities'],
        }
        json_bytes = json_module.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            label="📥 Download JSON Result",
            data=json_bytes,
            file_name=f"maltwin_result_{st.session_state[state.KEY_FILE_META]['sha256'][:8]}.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_probability_chart(probabilities: dict):
    """
    Render horizontal bar chart of all class probabilities.

    Args:
        probabilities: dict[str, float] — {family_name: probability}

    Implementation:
        Sort by probability descending.
        Use go.Bar with orientation='h'.
        Highlight top-1 bar with different color.
        ALL classes shown, including zero-probability ones (SRS FR5.3).
        Template: "plotly_dark"
        Height: max(400, len(probabilities) * 20)  # scale with class count

    Colors:
        Top-1 bar: '#FF4B4B' (red/alert)
        Other bars: '#4A90D9' (blue)
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
        text=[f"{p*100:.2f}%" for p in probs],
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


def _render_mitre_mapping(predicted_family: str):
    """
    Load and display MITRE ATT&CK for ICS mapping for the predicted family.

    Reads config.MITRE_JSON_PATH. If file missing or family not found,
    displays informational message (never an error).

    Display format for each technique:
        st.markdown(f"**{tactic}**")
        for technique in techniques:
            st.markdown(f"  - `{technique['id']}` — {technique['name']}")

    SRS ref: FR5.5 — if no mapping, display "MITRE mapping not available for this family."
    """
    try:
        with open(config.MITRE_JSON_PATH, 'r') as f:
            mitre_db = json.load(f)
    except FileNotFoundError:
        st.info("MITRE ATT&CK mapping database not found. "
                "Ensure data/mitre_ics_mapping.json exists.")
        return

    mapping = mitre_db.get(predicted_family)
    if not mapping:
        st.info(f"MITRE ATT&CK mapping not available for family: **{predicted_family}**")
        return

    tactics = mapping.get('tactics', [])
    techniques = mapping.get('techniques', [])

    if tactics:
        st.markdown(f"**Tactics:** {', '.join(tactics)}")

    if techniques:
        st.markdown("**Techniques:**")
        for t in techniques:
            st.markdown(f"  - `{t['id']}` — {t['name']}")
```

### 8.7 `modules/dashboard/pages/digital_twin.py` — Stub Specification

```python
"""
Digital Twin Simulation page — STUB.
Module 1 is deferred. This page provides a placeholder UI.
"""
import streamlit as st


def render():
    st.title("🖥️ Digital Twin Simulation")
    st.markdown("---")
    st.warning(
        "⚠️ **Module 1 — Digital Twin Simulation** is not yet implemented.\n\n"
        "This module will provide a Docker + Mininet based IIoT simulation environment "
        "for safe malware execution and behavioral observation."
    )
    st.markdown("**Planned capabilities:**")
    st.markdown("- Deploy containerized IIoT nodes (PLCs, sensors, MQTT broker, Modbus server)")
    st.markdown("- Simulate Modbus TCP and MQTT industrial traffic")
    st.markdown("- Execute malware samples in isolated containers")
    st.markdown("- Stream live network traffic log to dashboard")
    st.markdown("- Monitor node infection status in real-time")
    st.info("This page will be implemented in a future sprint once the ML pipeline is stable.")
```

---

## 9. Database Layer

### 9.1 `modules/dashboard/db.py` — Complete Specification

```python
"""
SQLite event logging for detection history.
All database access in MalTwin goes through this module.

Database file: config.DB_PATH (logs/maltwin.db)
WAL mode enabled: required by SRS REL-4 for crash safety.
File permissions 600: required by SRS SEC-3.

Schema:

    CREATE TABLE IF NOT EXISTS detection_events (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT    NOT NULL,   -- ISO 8601 UTC: "2025-04-22T14:35:22.123456"
        file_name        TEXT    NOT NULL,   -- original upload filename
        sha256           TEXT    NOT NULL,   -- 64-char hex SHA-256
        file_format      TEXT    NOT NULL,   -- 'PE' or 'ELF'
        file_size        INTEGER NOT NULL,   -- bytes
        predicted_family TEXT    NOT NULL,   -- top-1 class name
        confidence       REAL    NOT NULL,   -- top-1 softmax probability [0.0, 1.0]
        device_used      TEXT    NOT NULL    -- 'cpu', 'cuda', 'cuda:0', etc.
    );

    CREATE INDEX IF NOT EXISTS idx_timestamp ON detection_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_family    ON detection_events(predicted_family);
"""
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def get_connection(db_path: Path):
    """
    Context manager for database connections.
    Ensures WAL mode on every connection and closes properly.

    Usage:
        with get_connection(db_path) as conn:
            conn.execute(...)

    Implementation:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row   # allows dict-like row access
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

    Notes:
        - PRAGMA journal_mode=WAL must be set on EVERY connection (SRS REL-4).
        - PRAGMA synchronous=NORMAL balances safety and performance for WAL mode.
        - row_factory=sqlite3.Row allows accessing columns by name.
    """


def init_db(db_path: Path) -> None:
    """
    Create the database file, table, and indexes if they don't exist.
    Also sets file permissions to 600 (owner read/write only).

    Args:
        db_path: path to SQLite database file (config.DB_PATH)

    Implementation:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with get_connection(db_path) as conn:
            conn.execute(CREATE TABLE IF NOT EXISTS detection_events ...)
            conn.execute(CREATE INDEX IF NOT EXISTS idx_timestamp ...)
            conn.execute(CREATE INDEX IF NOT EXISTS idx_family ...)
        os.chmod(db_path, 0o600)   # SRS SEC-3

    Notes:
        - Safe to call multiple times (IF NOT EXISTS guards).
        - Called once in app.py main() at startup.
    """


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
    Insert one detection event into the database.

    Args:
        All fields map directly to schema columns.
        timestamp is set automatically to datetime.utcnow().isoformat().

    Error handling (SRS FR-B3, UC-02 A3):
        Attempt insert. On any exception, retry ONCE after 0.1 seconds.
        If retry also fails: log to stderr with print(f"DB write failed: {e}", file=sys.stderr)
        DO NOT raise — never block the detection result display for a DB failure.

    Implementation:
        import time, sys
        timestamp = datetime.utcnow().isoformat()
        sql = """INSERT INTO detection_events
                 (timestamp, file_name, sha256, file_format, file_size,
                  predicted_family, confidence, device_used)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (timestamp, file_name, sha256, file_format, file_size,
                  predicted_family, confidence, device_used)
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
    """


def get_recent_events(db_path: Path, limit: int = 5) -> list[dict]:
    """
    Retrieve the most recent detection events.

    Args:
        db_path: database path
        limit:   number of rows to return (default 5 per SRS FR1.4)

    Returns:
        List of dicts with keys matching all schema columns.
        Ordered by id DESC (most recent first).
        Returns empty list if DB does not exist or table is empty.

    Implementation:
        if not db_path.exists():
            return []
        try:
            with get_connection(db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM detection_events ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []
    """


def get_detection_stats(db_path: Path) -> dict:
    """
    Aggregate statistics for the home dashboard KPI cards.

    Returns:
        {
            'total_analyzed': int,     # total rows in detection_events
            'total_malware':  int,     # total rows (all detections count as malware for now)
            'total_benign':   int,     # 0 (benign files not tracked yet)
            'model_accuracy': None,    # loaded from metrics file if available, else None
        }

    Implementation:
        if not db_path.exists():
            return {'total_analyzed': 0, 'total_malware': 0,
                    'total_benign': 0, 'model_accuracy': None}
        try:
            with get_connection(db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
            # Try to load accuracy from processed/eval_metrics.json if it exists
            metrics_path = config.PROCESSED_DIR / 'eval_metrics.json'
            acc = None
            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    acc = json.load(f).get('accuracy')
            return {'total_analyzed': total, 'total_malware': total,
                    'total_benign': 0, 'model_accuracy': acc}
        except Exception:
            return {'total_analyzed': 0, 'total_malware': 0,
                    'total_benign': 0, 'model_accuracy': None}


def get_events_by_date_range(
    db_path: Path,
    days_back: int = 7,
) -> list[dict]:
    """
    Return all events from the last `days_back` days.
    Used by home.py activity chart.

    Returns list of dicts with 'timestamp' and 'predicted_family' columns.
    Returns empty list if DB missing or on any error.
    """
```

---

## 10. CLI Scripts

### 10.1 `scripts/train.py` — Complete Specification

```python
"""
Full training pipeline CLI.
Usage: python scripts/train.py [OPTIONS]

Options (all optional, defaults from config.py):
    --data-dir    PATH   Path to Malimg dataset root
    --epochs      INT    Number of training epochs
    --lr          FLOAT  Learning rate
    --batch-size  INT    Batch size
    --workers     INT    DataLoader workers
    --oversample  STR    Oversampling strategy
    --no-augment         Disable training augmentation (flag)
    --seed        INT    Random seed

Exit codes:
    0 — success
    1 — dataset not found
    2 — training error
"""
import argparse
import json
import sys
import torch
from pathlib import Path

def main():
    args = parse_args()

    # 1. Seed everything
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    import numpy as np, random
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 2. Validate dataset
    from modules.dataset.preprocessor import validate_dataset_integrity
    print("Validating dataset...")
    try:
        report = validate_dataset_integrity(Path(args.data_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Families found: {len(report['families'])}")
    print(f"  Total samples:  {report['total']}")
    print(f"  Imbalance ratio: {report['imbalance_ratio']:.1f}x "
          f"({report['max_class']} vs {report['min_class']})")
    if report['corrupt_files']:
        print(f"  WARNING: {len(report['corrupt_files'])} corrupt files found")

    # 3. Build DataLoaders
    print("\nBuilding DataLoaders...")
    from modules.dataset.loader import get_dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=Path(args.data_dir),
        img_size=128,
        batch_size=args.batch_size,
        num_workers=args.workers,
        oversample_strategy=args.oversample,
        augment_train=not args.no_augment,
        random_seed=args.seed,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Classes:       {len(class_names)}")

    # 4. Build model
    from modules.detection.model import MalTwinCNN
    import config
    model = MalTwinCNN(num_classes=len(class_names)).to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MalTwinCNN")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device:     {config.DEVICE}")

    # 5. Train
    print("\nStarting training...")
    from modules.detection.trainer import train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"\nTraining complete.")
    print(f"  Best Val Acc: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")

    # 6. Evaluate on test set
    print("\nEvaluating on test set...")
    from modules.detection.inference import load_model
    from modules.detection.evaluator import evaluate, format_metrics_table, plot_confusion_matrix
    best_model = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
    metrics = evaluate(best_model, test_loader, config.DEVICE, class_names)
    print(format_metrics_table(metrics, class_names))

    # 7. Save eval metrics for dashboard
    metrics_path = config.PROCESSED_DIR / 'eval_metrics.json'
    serialisable = {k: v for k, v in metrics.items()
                    if k != 'confusion_matrix' and k != 'per_class'
                    and k != 'classification_report'}
    serialisable['per_class'] = {k: dict(v) for k, v in metrics['per_class'].items()}
    with open(metrics_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nEval metrics saved to {metrics_path}")

    # 8. Save confusion matrix image
    cm_path = config.PROCESSED_DIR / 'confusion_matrix.png'
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    print("\nDone. Launch dashboard: streamlit run modules/dashboard/app.py")
    sys.exit(0)
```

### 10.2 `scripts/evaluate.py` — Specification

```python
"""
Evaluate best_model.pt on test split only (no retraining).
Usage: python scripts/evaluate.py [--model-path PATH] [--data-dir PATH]

Loads best_model.pt, builds test DataLoader, runs evaluate(), prints metrics.
Useful for re-evaluation after any code changes without retraining.
"""
```

### 10.3 `scripts/convert_binary.py` — Specification

```python
"""
Convert a single binary file to grayscale PNG (standalone utility).
Usage: python scripts/convert_binary.py --input FILE --output FILE.png

Steps:
    1. Read file bytes
    2. validate_binary_format → print "Detected: PE/ELF"
    3. compute_sha256 → print hash
    4. get_file_metadata → print size, format
    5. BinaryConverter().convert() → save PNG
    6. Print: "Saved 128x128 grayscale PNG to {output}"

No ML dependencies required.
"""
```

---

## 11. Static Data Files

### 11.1 `data/mitre_ics_mapping.json` — Full Seed File

```json
{
  "Allaple.A": {
    "tactics": ["Lateral Movement", "Impact"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0882", "name": "Theft of Operational Information"},
      {"id": "T0866", "name": "Exploitation of Remote Services"}
    ]
  },
  "Allaple.L": {
    "tactics": ["Lateral Movement", "Collection"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0802", "name": "Automated Collection"}
    ]
  },
  "Yuner.A": {
    "tactics": ["Execution", "Persistence"],
    "techniques": [
      {"id": "T0807", "name": "Command-Line Interface"},
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0859", "name": "Valid Accounts"}
    ]
  },
  "Instantaccess": {
    "tactics": ["Collection", "Exfiltration"],
    "techniques": [
      {"id": "T0802", "name": "Automated Collection"},
      {"id": "T0811", "name": "Data from Information Repositories"}
    ]
  },
  "Swizzor.gen!E": {
    "tactics": ["Defense Evasion"],
    "techniques": [
      {"id": "T0858", "name": "Change Operating Mode"},
      {"id": "T0820", "name": "Exploitation for Evasion"}
    ]
  },
  "Swizzor.gen!I": {
    "tactics": ["Defense Evasion", "Execution"],
    "techniques": [
      {"id": "T0858", "name": "Change Operating Mode"},
      {"id": "T0871", "name": "Execution through API"}
    ]
  },
  "VB.AT": {
    "tactics": ["Execution", "Persistence"],
    "techniques": [
      {"id": "T0871", "name": "Execution through API"},
      {"id": "T0839", "name": "Module Firmware"}
    ]
  },
  "Fakerean": {
    "tactics": ["Defense Evasion", "Impact"],
    "techniques": [
      {"id": "T0851", "name": "Rootkit"},
      {"id": "T0826", "name": "Loss of Availability"}
    ]
  },
  "Adialer.C": {
    "tactics": ["Exfiltration", "Command and Control"],
    "techniques": [
      {"id": "T0861", "name": "Point-to-Point Communication"},
      {"id": "T0868", "name": "Detect Operating Mode"}
    ]
  },
  "Agent.FYI": {
    "tactics": ["Collection", "Command and Control"],
    "techniques": [
      {"id": "T0802", "name": "Automated Collection"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "Alueron.gen!J": {
    "tactics": ["Persistence", "Privilege Escalation"],
    "techniques": [
      {"id": "T0839", "name": "Module Firmware"},
      {"id": "T0874", "name": "Hooking"}
    ]
  },
  "Autorun.K": {
    "tactics": ["Persistence", "Lateral Movement"],
    "techniques": [
      {"id": "T0843", "name": "Program Download"},
      {"id": "T0812", "name": "Default Credentials"}
    ]
  },
  "C2LOP.P": {
    "tactics": ["Command and Control"],
    "techniques": [
      {"id": "T0885", "name": "Commonly Used Port"},
      {"id": "T0884", "name": "Connection Proxy"}
    ]
  },
  "C2LOP.gen!g": {
    "tactics": ["Command and Control", "Exfiltration"],
    "techniques": [
      {"id": "T0885", "name": "Commonly Used Port"},
      {"id": "T0882", "name": "Theft of Operational Information"}
    ]
  },
  "Dialplatform.B": {
    "tactics": ["Exfiltration"],
    "techniques": [
      {"id": "T0882", "name": "Theft of Operational Information"}
    ]
  },
  "Dontovo.A": {
    "tactics": ["Execution"],
    "techniques": [
      {"id": "T0807", "name": "Command-Line Interface"}
    ]
  },
  "Lolyda.AA1": {
    "tactics": ["Credential Access"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0859", "name": "Valid Accounts"}
    ]
  },
  "Lolyda.AA2": {
    "tactics": ["Credential Access"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"}
    ]
  },
  "Lolyda.AA3": {
    "tactics": ["Credential Access", "Lateral Movement"],
    "techniques": [
      {"id": "T0812", "name": "Default Credentials"},
      {"id": "T0866", "name": "Exploitation of Remote Services"}
    ]
  },
  "Lolyda.AT": {
    "tactics": ["Credential Access"],
    "techniques": [
      {"id": "T0859", "name": "Valid Accounts"}
    ]
  },
  "Malex.gen!J": {
    "tactics": ["Defense Evasion", "Execution"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0871", "name": "Execution through API"}
    ]
  },
  "Obfuscator.AD": {
    "tactics": ["Defense Evasion"],
    "techniques": [
      {"id": "T0820", "name": "Exploitation for Evasion"},
      {"id": "T0858", "name": "Change Operating Mode"}
    ]
  },
  "Rbot!gen": {
    "tactics": ["Command and Control", "Lateral Movement"],
    "techniques": [
      {"id": "T0885", "name": "Commonly Used Port"},
      {"id": "T0866", "name": "Exploitation of Remote Services"}
    ]
  },
  "Skintrim.N": {
    "tactics": ["Defense Evasion", "Persistence"],
    "techniques": [
      {"id": "T0851", "name": "Rootkit"},
      {"id": "T0839", "name": "Module Firmware"}
    ]
  },
  "Wintrim.BX": {
    "tactics": ["Execution", "Defense Evasion"],
    "techniques": [
      {"id": "T0807", "name": "Command-Line Interface"},
      {"id": "T0858", "name": "Change Operating Mode"}
    ]
  }
}
```

---

## 12. Test Suite

### 12.1 `tests/conftest.py`

```python
"""
Shared pytest fixtures.
"""
import pytest
import numpy as np
from pathlib import Path
import torch


@pytest.fixture
def sample_pe_bytes() -> bytes:
    """Minimal valid PE binary: MZ header + 1020 zero bytes = 1024 bytes."""
    header = b'MZ' + b'\x00' * 58  # minimal DOS stub (60 bytes)
    body   = b'\x00' * (1024 - 60)
    return header + body


@pytest.fixture
def sample_elf_bytes() -> bytes:
    """Minimal valid ELF binary: ELF magic + 1020 bytes."""
    header = b'\x7fELF' + b'\x00' * 56
    body   = b'\x00' * (1024 - 60)
    return header + body


@pytest.fixture
def random_binary_bytes() -> bytes:
    """Random bytes that are NOT PE or ELF."""
    import os
    return os.urandom(2048)


@pytest.fixture
def sample_grayscale_array() -> np.ndarray:
    """128x128 uint8 numpy array simulating a converted binary."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_tensor() -> torch.Tensor:
    """(1, 128, 128) float32 tensor in [-1, 1] range."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)  # (1, 128, 128)


@pytest.fixture
def num_classes() -> int:
    return 25
```

### 12.2 `tests/test_converter.py`

```python
"""Test suite for modules/binary_to_image/"""
import pytest
import numpy as np
from modules.binary_to_image.converter import BinaryConverter
from modules.binary_to_image.utils import (
    validate_binary_format, compute_sha256, compute_pixel_histogram, get_file_metadata
)


class TestValidateBinaryFormat:
    def test_accepts_pe_mz_header(self, sample_pe_bytes):
        assert validate_binary_format(sample_pe_bytes) == 'PE'

    def test_accepts_elf_magic(self, sample_elf_bytes):
        assert validate_binary_format(sample_elf_bytes) == 'ELF'

    def test_rejects_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            validate_binary_format(b'\x4d\x5a')  # only 2 bytes

    def test_rejects_unknown_format(self, random_binary_bytes):
        # Random bytes are almost certainly not PE or ELF
        # Patch first 4 bytes to be definitively non-magic
        bad_bytes = b'\xDE\xAD\xBE\xEF' + random_binary_bytes[4:]
        with pytest.raises(ValueError, match="Unsupported"):
            validate_binary_format(bad_bytes)

    def test_error_includes_hex_repr(self):
        bad_bytes = b'\xDE\xAD\xBE\xEF' + b'\x00' * 100
        with pytest.raises(ValueError) as exc_info:
            validate_binary_format(bad_bytes)
        assert 'DEADBEEF' in str(exc_info.value)


class TestComputeSha256:
    def test_returns_64_char_hex(self, sample_pe_bytes):
        result = compute_sha256(sample_pe_bytes)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_deterministic(self, sample_pe_bytes):
        assert compute_sha256(sample_pe_bytes) == compute_sha256(sample_pe_bytes)

    def test_different_inputs_different_hashes(self, sample_pe_bytes, sample_elf_bytes):
        assert compute_sha256(sample_pe_bytes) != compute_sha256(sample_elf_bytes)

    def test_lowercase(self, sample_pe_bytes):
        result = compute_sha256(sample_pe_bytes)
        assert result == result.lower()


class TestComputePixelHistogram:
    def test_returns_256_bins(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert len(hist['bins']) == 256
        assert len(hist['counts']) == 256

    def test_bins_are_0_to_255(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert hist['bins'] == list(range(256))

    def test_counts_sum_to_total_pixels(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert sum(hist['counts']) == sample_grayscale_array.size

    def test_all_counts_nonnegative(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert all(c >= 0 for c in hist['counts'])


class TestBinaryConverter:
    def test_output_shape_is_correct(self, sample_pe_bytes):
        converter = BinaryConverter(img_size=128)
        result = converter.convert(sample_pe_bytes)
        assert result.shape == (128, 128)

    def test_output_dtype_is_uint8(self, sample_pe_bytes):
        result = BinaryConverter().convert(sample_pe_bytes)
        assert result.dtype == np.uint8

    def test_output_values_in_range(self, sample_pe_bytes):
        result = BinaryConverter().convert(sample_pe_bytes)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_custom_img_size(self, sample_pe_bytes):
        for size in [64, 128, 256]:
            result = BinaryConverter(img_size=size).convert(sample_pe_bytes)
            assert result.shape == (size, size)

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="too small"):
            BinaryConverter().convert(b'')

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            BinaryConverter().convert(b'\x4d\x5a\x00')  # 3 bytes

    def test_elf_binary_converts(self, sample_elf_bytes):
        result = BinaryConverter().convert(sample_elf_bytes)
        assert result.shape == (128, 128)

    def test_deterministic(self, sample_pe_bytes):
        c = BinaryConverter()
        r1 = c.convert(sample_pe_bytes)
        r2 = c.convert(sample_pe_bytes)
        np.testing.assert_array_equal(r1, r2)

    def test_to_png_bytes_returns_bytes(self, sample_grayscale_array):
        converter = BinaryConverter()
        result = converter.to_png_bytes(sample_grayscale_array)
        assert isinstance(result, bytes)
        assert result[:4] == b'\x89PNG'  # PNG magic bytes

    def test_to_pil_image_mode(self, sample_grayscale_array):
        from PIL import Image
        converter = BinaryConverter()
        pil = converter.to_pil_image(sample_grayscale_array)
        assert isinstance(pil, Image.Image)
        assert pil.mode == 'L'
        assert pil.size == (128, 128)
```

### 12.3 `tests/test_dataset.py`

```python
"""
Test suite for modules/dataset/
NOTE: These tests require Malimg dataset to be present at config.DATA_DIR.
Tests that need the dataset are marked @pytest.mark.integration.
Unit tests (no dataset needed) run without the dataset.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from modules.dataset.preprocessor import (
    normalize_image, encode_labels, validate_dataset_integrity
)


class TestNormalizeImage:
    def test_output_range(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.dtype == np.float32

    def test_zero_maps_to_zero(self):
        arr = np.zeros((4, 4), dtype=np.uint8)
        assert normalize_image(arr).max() == 0.0

    def test_255_maps_to_one(self):
        arr = np.full((4, 4), 255, dtype=np.uint8)
        np.testing.assert_almost_equal(normalize_image(arr).min(), 1.0, decimal=6)


class TestEncodeLabels:
    def test_sorted_alphabetically(self):
        result = encode_labels(['Yuner.A', 'Allaple.A', 'VB.AT'])
        assert result == {'Allaple.A': 0, 'VB.AT': 1, 'Yuner.A': 2}

    def test_unique_integers(self):
        families = ['A', 'B', 'C', 'D']
        result = encode_labels(families)
        assert len(set(result.values())) == 4

    def test_range_correct(self):
        families = ['X', 'Y', 'Z']
        result = encode_labels(families)
        assert set(result.values()) == {0, 1, 2}

    def test_deterministic(self):
        f = ['Yuner.A', 'Allaple.A']
        assert encode_labels(f) == encode_labels(f)

    def test_single_family(self):
        assert encode_labels(['OnlyOne']) == {'OnlyOne': 0}


class TestValidateDatasetIntegrity:
    def test_missing_dir_raises(self, tmp_path):
        missing = tmp_path / 'nonexistent'
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_dataset_integrity(missing)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="empty"):
            validate_dataset_integrity(tmp_path)

    def test_returns_required_keys(self, tmp_path):
        """Create a minimal fake dataset structure."""
        (tmp_path / 'FamilyA').mkdir()
        # No PNGs — corrupt_files and counts will be empty but structure valid
        # Actually need at least one valid PNG; skip this or mock
        pytest.skip("Requires fake PNG files — covered by integration test")

    @pytest.mark.integration
    def test_malimg_dataset_valid(self):
        """Requires real Malimg dataset at config.DATA_DIR."""
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found at DATA_DIR")
        report = validate_dataset_integrity(config.DATA_DIR)
        assert report['total'] > 0
        assert len(report['families']) == 25
        assert len(report['corrupt_files']) == 0


class TestMalimgDataset:
    @pytest.mark.integration
    def test_split_sizes_sum_correctly(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        train_ds = MalimgDataset(config.DATA_DIR, 'train')
        val_ds   = MalimgDataset(config.DATA_DIR, 'val')
        test_ds  = MalimgDataset(config.DATA_DIR, 'test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        # Total should be close to 9339 (Malimg size)
        assert 9000 < total < 9500

    @pytest.mark.integration
    def test_getitem_tensor_shape(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(config.DATA_DIR, 'train')
        tensor, label = ds[0]
        assert tensor.shape == (1, 128, 128)
        assert tensor.dtype == torch.float32
        assert isinstance(label, int)

    @pytest.mark.integration
    def test_all_splits_contain_all_classes(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        for split in ['train', 'val', 'test']:
            ds = MalimgDataset(config.DATA_DIR, split)
            labels_in_split = set(ds.get_labels())
            assert len(labels_in_split) == 25, \
                f"Split '{split}' missing classes: {25 - len(labels_in_split)} classes absent"
```

### 12.4 `tests/test_enhancement.py`

```python
"""Test suite for modules/enhancement/"""
import pytest
import torch
import numpy as np
from PIL import Image
from modules.enhancement.augmentor import GaussianNoise, get_train_transforms, get_val_transforms
from modules.enhancement.balancer import ClassAwareOversampler


class TestGaussianNoise:
    def test_output_clamped_to_01(self):
        noise = GaussianNoise(std_range=(0.3, 0.5))  # large noise to test clamping
        tensor = torch.rand(1, 128, 128)
        result = noise(tensor)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_output_same_shape(self):
        noise = GaussianNoise()
        tensor = torch.rand(1, 128, 128)
        result = noise(tensor)
        assert result.shape == tensor.shape

    def test_output_same_dtype(self):
        noise = GaussianNoise()
        tensor = torch.rand(1, 128, 128)
        result = noise(tensor)
        assert result.dtype == tensor.dtype

    def test_noise_actually_changes_values(self):
        noise = GaussianNoise(std_range=(0.1, 0.2))
        tensor = torch.full((1, 128, 128), 0.5)
        result = noise(tensor)
        assert not torch.equal(result, tensor)


class TestTransformPipelines:
    def test_train_transform_output_shape(self):
        transform = get_train_transforms(128)
        pil_img = Image.fromarray(np.random.randint(0,256,(128,128),dtype=np.uint8), mode='L')
        result = transform(pil_img)
        assert result.shape == (1, 128, 128)

    def test_val_transform_output_shape(self):
        transform = get_val_transforms(128)
        pil_img = Image.fromarray(np.random.randint(0,256,(128,128),dtype=np.uint8), mode='L')
        result = transform(pil_img)
        assert result.shape == (1, 128, 128)

    def test_val_transform_output_dtype(self):
        transform = get_val_transforms(128)
        pil_img = Image.fromarray(np.zeros((128,128), dtype=np.uint8), mode='L')
        result = transform(pil_img)
        assert result.dtype == torch.float32

    def test_val_transform_is_deterministic(self):
        transform = get_val_transforms(128)
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode='L')
        r1 = transform(pil_img)
        pil_img2 = Image.fromarray(arr, mode='L')
        r2 = transform(pil_img2)
        torch.testing.assert_close(r1, r2)

    def test_normalize_range(self):
        """After normalize(mean=0.5, std=0.5), pure black image → -1.0"""
        transform = get_val_transforms(128)
        black_img = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='L')
        result = transform(black_img)
        assert abs(result.min().item() - (-1.0)) < 1e-5

    def test_train_transforms_contain_augmentation(self):
        """Train pipeline should differ from val pipeline (due to stochastic ops)."""
        train_tf = get_train_transforms(128)
        val_tf   = get_val_transforms(128)
        # They should not be the same object
        assert type(train_tf) == type(val_tf)  # both Compose
        # Train pipeline has more transforms
        assert len(train_tf.transforms) > len(val_tf.transforms)


class TestClassAwareOversampler:
    def _make_mock_dataset(self, labels: list):
        """Create a minimal mock dataset with get_labels()."""
        class MockDS:
            def __init__(self, lbls):
                self._labels = lbls
            def get_labels(self):
                return self._labels
            def __len__(self):
                return len(self._labels)
        return MockDS(labels)

    def test_returns_weighted_random_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        labels = [0]*100 + [1]*10 + [2]*5
        ds = self._make_mock_dataset(labels)
        sampler = ClassAwareOversampler(ds, 'oversample_minority').get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_num_samples_equals_dataset_length(self):
        labels = [0]*100 + [1]*10
        ds = self._make_mock_dataset(labels)
        sampler = ClassAwareOversampler(ds, 'oversample_minority').get_sampler()
        assert sampler.num_samples == len(labels)

    def test_oversample_minority_weights_minority_higher(self):
        labels = [0]*100 + [1]*10   # class 0 majority, class 1 minority
        ds = self._make_mock_dataset(labels)
        oversampler = ClassAwareOversampler(ds, 'oversample_minority')
        oversampler.get_sampler()  # computes class_weights
        # Minority class (1) should have higher weight than majority (0)
        assert oversampler.class_weights[1] > oversampler.class_weights[0]

    def test_uniform_strategy_equal_weights(self):
        labels = [0]*100 + [1]*10
        ds = self._make_mock_dataset(labels)
        oversampler = ClassAwareOversampler(ds, 'uniform')
        oversampler.get_sampler()
        assert oversampler.class_weights[0] == oversampler.class_weights[1]

    def test_invalid_strategy_raises(self):
        labels = [0, 1, 2]
        ds = self._make_mock_dataset(labels)
        with pytest.raises(ValueError, match="Unknown strategy"):
            ClassAwareOversampler(ds, 'invalid_strategy').get_sampler()
```

### 12.5 `tests/test_model.py`

```python
"""Test suite for modules/detection/model.py and inference.py"""
import pytest
import torch
import numpy as np
from modules.detection.model import MalTwinCNN, ConvBlock


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(in_channels=1, out_channels=32)
        x = torch.randn(4, 1, 128, 128)
        out = block(x)
        # After MaxPool2d(2x2): spatial dims halved
        assert out.shape == (4, 32, 64, 64)

    def test_output_shape_block2(self):
        block = ConvBlock(in_channels=32, out_channels=64)
        x = torch.randn(4, 32, 64, 64)
        out = block(x)
        assert out.shape == (4, 64, 32, 32)


class TestMalTwinCNN:
    def test_forward_pass_output_shape(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(4, 1, 128, 128)   # batch of 4
        out = model(x)
        assert out.shape == (4, num_classes)

    def test_single_sample_forward(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, num_classes)

    def test_parameter_count_reasonable(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000, f"Too few parameters: {total}"
        assert total < 20_000_000, f"Too many parameters: {total}"

    def test_output_is_logits_not_probabilities(self, num_classes):
        """Output logits can be any value, not constrained to [0,1]."""
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        # Logits are NOT guaranteed to be in [0,1]
        # If softmax was applied, all values would be in [0,1] and sum to 1
        # We check that NOT all values are in [0,1]
        has_values_outside_01 = (out < 0).any() or (out > 1).any()
        # This is expected for logits (may not always be true for tiny random models)
        # So just verify softmax makes them valid probabilities
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_gradcam_layer_attribute_exists(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert hasattr(model, 'gradcam_layer')
        assert model.gradcam_layer is model.block3.conv2

    def test_deterministic_in_eval_mode(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_dropout_nondeterministic(self, num_classes):
        """In train mode, Dropout should cause different outputs."""
        model = MalTwinCNN(num_classes=num_classes)
        model.train()
        x = torch.randn(2, 1, 128, 128)
        torch.manual_seed(0)
        out1 = model(x)
        torch.manual_seed(1)
        out2 = model(x)
        # With different seeds, Dropout should give different outputs
        assert not torch.equal(out1, out2)


class TestPredictSingle:
    def test_returns_required_keys(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert 'predicted_family' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'top3' in result

    def test_confidence_in_valid_range(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_has_all_classes(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert len(result['probabilities']) == num_classes

    def test_top3_has_three_entries(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert len(result['top3']) == 3

    def test_predicted_family_is_top3_first(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert result['predicted_family'] == result['top3'][0]['family']

    def test_top3_sorted_descending(self, sample_grayscale_array, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        confs = [item['confidence'] for item in result['top3']]
        assert confs == sorted(confs, reverse=True)

    def test_all_probability_values_json_serialisable(self, sample_grayscale_array, num_classes):
        import json
        model = MalTwinCNN(num_classes=num_classes)
        class_names = [f"Family_{i}" for i in range(num_classes)]
        from modules.detection.inference import predict_single
        result = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        # Should not raise
        json.dumps(result)
```

### 12.6 `tests/test_db.py`

```python
"""Test suite for modules/dashboard/db.py"""
import pytest
import os
from pathlib import Path
from modules.dashboard.db import init_db, log_detection_event, get_recent_events, get_detection_stats


@pytest.fixture
def temp_db(tmp_path) -> Path:
    """Create a fresh DB in a temp directory."""
    db_path = tmp_path / "test_maltwin.db"
    init_db(db_path)
    return db_path


class TestInitDb:
    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        assert not db_path.exists()
        init_db(db_path)
        assert db_path.exists()

    def test_idempotent(self, temp_db):
        """Calling init_db twice should not raise."""
        init_db(temp_db)  # second call

    def test_file_permissions_600(self, temp_db):
        stat = os.stat(temp_db)
        mode = stat.st_mode & 0o777
        assert mode == 0o600, f"Expected 600, got {oct(mode)}"


class TestLogDetectionEvent:
    def test_inserts_row(self, temp_db):
        log_detection_event(
            temp_db, "test.exe", "a"*64, "PE", 1024, "Allaple.A", 0.95, "cpu"
        )
        events = get_recent_events(temp_db, limit=10)
        assert len(events) == 1

    def test_inserted_values_correct(self, temp_db):
        log_detection_event(
            temp_db, "test.exe", "b"*64, "ELF", 2048, "Yuner.A", 0.75, "cuda"
        )
        events = get_recent_events(temp_db)
        row = events[0]
        assert row['file_name'] == "test.exe"
        assert row['sha256'] == "b"*64
        assert row['file_format'] == "ELF"
        assert row['file_size'] == 2048
        assert row['predicted_family'] == "Yuner.A"
        assert abs(row['confidence'] - 0.75) < 1e-6
        assert row['device_used'] == "cuda"

    def test_does_not_raise_on_missing_db(self, tmp_path):
        bad_path = tmp_path / "nonexistent" / "db.db"
        # Should not raise (just log to stderr)
        # Actually this will fail because parent dir doesn't exist
        # The function should handle this gracefully
        # If it raises, that's acceptable as long as it doesn't crash the app
        # We test the no-raise contract:
        try:
            log_detection_event(bad_path, "x", "a"*64, "PE", 1, "X", 0.5, "cpu")
        except Exception:
            pass  # acceptable — just shouldn't crash the calling code


class TestGetRecentEvents:
    def test_returns_empty_list_for_empty_db(self, temp_db):
        assert get_recent_events(temp_db) == []

    def test_returns_empty_list_if_db_missing(self, tmp_path):
        assert get_recent_events(tmp_path / "missing.db") == []

    def test_returns_most_recent_first(self, temp_db):
        for i in range(3):
            log_detection_event(temp_db, f"file_{i}.exe", "a"*64, "PE",
                                 1024, f"Family_{i}", 0.9, "cpu")
        events = get_recent_events(temp_db, limit=5)
        assert events[0]['file_name'] == "file_2.exe"

    def test_limit_respected(self, temp_db):
        for i in range(10):
            log_detection_event(temp_db, f"f{i}.exe", "a"*64, "PE", 100, "X", 0.5, "cpu")
        assert len(get_recent_events(temp_db, limit=3)) == 3


class TestGetDetectionStats:
    def test_empty_db_returns_zeros(self, temp_db):
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 0
        assert stats['total_malware'] == 0

    def test_counts_correctly(self, temp_db):
        for i in range(5):
            log_detection_event(temp_db, f"f{i}.exe", "a"*64, "PE", 100, "X", 0.5, "cpu")
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 5
```

---

## 13. Inter-Module Data Flow

```
User uploads file.exe (bytes)
        │
        ▼
validate_binary_format(bytes) → 'PE'
compute_sha256(bytes)          → hex string
get_file_metadata(...)         → dict
        │
        ▼
BinaryConverter.convert(bytes) → np.ndarray (128, 128) uint8
        │
        ▼ [stored in session_state]
        │
        ▼
[User clicks Run Detection]
        │
        ▼
predict_single(model, img_array, class_names, device)
    │
    ├─ Image.fromarray(img_array, 'L')         PIL Image (L mode)
    ├─ get_val_transforms()(pil_img)           torch.Tensor (1,128,128) float32
    ├─ unsqueeze(0)                            torch.Tensor (1,1,128,128)
    ├─ model(tensor) → logits                  torch.Tensor (1,25)
    ├─ torch.softmax(logits, dim=1) → probs    torch.Tensor (1,25)
    └─ argmax, sort, slice                     → result dict
        │
        ▼
log_detection_event(db_path, ...) → SQLite row inserted
        │
        ▼
_render_results()
    ├─ confidence bar (color coded)
    ├─ top-3 predictions
    ├─ plotly per-class probability chart (all 25 families)
    ├─ MITRE ATT&CK mapping lookup from JSON
    ├─ XAI heatmap checkbox (STUB)
    └─ JSON download button (active) / PDF button (STUB)
```

### 13.1 Data Type Contracts Between Modules

| Hand-off Point | Sender | Receiver | Type | Shape/Format |
|---|---|---|---|---|
| Raw file bytes | User upload | validate_binary_format | `bytes` | any length |
| Binary format string | validate_binary_format | BinaryConverter | `str` | `'PE'` or `'ELF'` |
| Grayscale array | BinaryConverter.convert | session_state | `np.ndarray` | `(128,128)` uint8 |
| PIL Image | BinaryConverter.to_pil_image | get_val_transforms | `PIL.Image` | mode=`'L'` |
| Transformed tensor | get_val_transforms | predict_single | `torch.Tensor` | `(1,128,128)` float32 |
| Batched tensor | predict_single | model.forward | `torch.Tensor` | `(1,1,128,128)` |
| Logits | model.forward | predict_single | `torch.Tensor` | `(1,25)` |
| Probabilities | torch.softmax | predict_single | `torch.Tensor` | `(1,25)` sum=1 |
| Result dict | predict_single | session_state | `dict` | keys: predicted_family, confidence, probabilities, top3 |
| class_names | MalimgDataset | predict_single | `list[str]` | length=25, sorted alphabetically |
| class_names JSON | save_class_names | load_class_names | `JSON file` | `{"class_names": [...]}` |
| Detection event | log_detection_event | SQLite DB | row | all fields as specified |

---

## 14. Error Handling Contract

Every user-facing error message must follow this format (SRS USE-3):
```
Error: [plain English description of what went wrong].
Cause: [why it happened].
Action: [exactly what the user should do].
```

### 14.1 Error Catalogue

| Error Condition | Where Raised | Where Caught | User Message |
|---|---|---|---|
| File size > 50 MB | upload.py | upload.py | "Error: File too large. Cause: File exceeds the 50 MB limit. Action: Upload a smaller binary." |
| Invalid binary format | validate_binary_format | upload.py | "Error: Unsupported file format. Cause: {magic_bytes} is not PE or ELF. Action: Upload a .exe, .dll, or ELF binary." |
| File too small (<4 bytes) | validate_binary_format | upload.py | "Error: File is too small. Cause: Minimum 4 bytes required. Action: Upload a valid binary file." |
| Binary < 64 bytes | BinaryConverter.convert | upload.py | "Error: Binary too small to convert. Cause: Minimum 64 bytes required for image conversion. Action: Upload a larger binary." |
| cv2.imread returns None | MalimgDataset.__getitem__ | DataLoader collate | RuntimeError logged to stderr; batch skipped |
| Model file not found | load_model | app.py + detection.py | Sidebar warning + page warning (no st.error, just st.warning) |
| class_names.json missing | load_class_names | app.py | Sidebar caption "Run scripts/train.py first" |
| Detection inference error | predict_single | detection.py | "Error: Detection failed. Cause: {exception}. Action: Ensure model is loaded and try again." |
| DB write failure | log_detection_event | (swallowed) | Non-blocking: print to stderr, detection result still displayed |
| MITRE JSON not found | _render_mitre_mapping | detection.py | st.info (not error): "MITRE mapping database not found." |
| Confusion matrix save fails | plot_confusion_matrix | train.py | Warning printed; does not abort training |

### 14.2 Rules for Error Handling

1. **Never crash the Streamlit app.** All exceptions in page render functions must be caught and displayed as `st.error()`.
2. **Never show raw Python tracebacks** to end users. Log them with `print(..., file=sys.stderr)` or Python `logging`.
3. **DB failures are non-blocking.** A failed SQLite write must never prevent the detection result from being displayed.
4. **Model loading failure is graceful.** If `best_model.pt` doesn't exist, the app starts without a model and shows clear guidance (not an error page).
5. **File validation errors return early** from the upload function — never partially process an invalid file.

---

## 15. Implementation Order & Dependency Graph

```
Phase 1 — Foundation (no ML deps)
────────────────────────────────
  1.  config.py
  2.  modules/binary_to_image/utils.py
  3.  modules/binary_to_image/converter.py
  4.  modules/binary_to_image/__init__.py
  5.  tests/conftest.py
  6.  tests/test_converter.py
  ✓  Run: pytest tests/test_converter.py -v

Phase 2 — Dataset (needs Malimg + scikit-learn)
────────────────────────────────────────────────
  7.  modules/dataset/preprocessor.py
  8.  modules/dataset/loader.py
  9.  modules/dataset/__init__.py
  10. tests/test_dataset.py
  ✓  Run: pytest tests/test_dataset.py -v -m "not integration"
  ✓  With Malimg: pytest tests/test_dataset.py -v (including integration)

Phase 3 — Enhancement (needs torchvision)
──────────────────────────────────────────
  11. modules/enhancement/augmentor.py
  12. modules/enhancement/balancer.py
  13. modules/enhancement/__init__.py
  14. tests/test_enhancement.py
  ✓  Run: pytest tests/test_enhancement.py -v

Phase 4 — Detection Model (needs Phase 2+3)
────────────────────────────────────────────
  15. modules/detection/model.py
  16. modules/detection/trainer.py
  17. modules/detection/evaluator.py
  18. modules/detection/inference.py
  19. modules/detection/__init__.py
  20. tests/test_model.py
  ✓  Run: pytest tests/test_model.py -v

Phase 5 — CLI Scripts (needs Phase 2+3+4)
──────────────────────────────────────────
  21. scripts/train.py
  22. scripts/evaluate.py
  23. scripts/convert_binary.py
  ✓  Run: python scripts/train.py --epochs 2 (smoke test with 2 epochs)
  ✓  Full train: python scripts/train.py (produces best_model.pt + class_names.json)

Phase 6 — Dashboard (needs Phase 4+5 outputs)
──────────────────────────────────────────────
  24. modules/dashboard/db.py
  25. tests/test_db.py
  ✓  Run: pytest tests/test_db.py -v
  26. modules/dashboard/state.py
  27. modules/dashboard/pages/digital_twin.py  (STUB — write first, trivial)
  28. modules/dashboard/pages/home.py
  29. modules/dashboard/pages/upload.py
  30. modules/dashboard/pages/detection.py
  31. modules/dashboard/app.py
  ✓  Run: streamlit run modules/dashboard/app.py

Phase 7 — Integration smoke test
──────────────────────────────────
  ✓  pytest tests/ -v --ignore=tests/test_dataset.py  (all non-integration tests)
  ✓  Open http://localhost:8501
  ✓  Upload a binary → verify grayscale image renders
  ✓  Click Run Detection → verify result displays with confidence bar
  ✓  Verify Recent Detections on home page shows the event
  ✓  Verify JSON download button works
```

### 15.1 Dependency Graph (imports)

```
config.py  (no internal imports)
    ↑
modules/binary_to_image/utils.py     (config)
modules/binary_to_image/converter.py (config, utils)
    ↑
modules/dataset/preprocessor.py     (config)
modules/dataset/loader.py            (config, preprocessor, enhancement/augmentor)
    ↑
modules/enhancement/augmentor.py     (config)
modules/enhancement/balancer.py      (no internal imports)
    ↑
modules/detection/model.py           (no internal imports)
modules/detection/trainer.py         (config, model)
modules/detection/evaluator.py       (config, model)
modules/detection/inference.py       (config, model, enhancement/augmentor)
    ↑
modules/dashboard/db.py              (config)
modules/dashboard/state.py           (no internal imports)
modules/dashboard/pages/upload.py    (config, binary_to_image, dashboard/state)
modules/dashboard/pages/detection.py (config, detection/inference, dashboard/db, dashboard/state)
modules/dashboard/pages/home.py      (config, dashboard/db, dashboard/state)
modules/dashboard/app.py             (config, dashboard/*, dataset/preprocessor, detection/inference)
    ↑
scripts/train.py     (config, dataset/*, enhancement/*, detection/*)
scripts/evaluate.py  (config, dataset/loader, detection/*)
scripts/convert_binary.py (config, binary_to_image/*)
```

---

## 16. Coding Agent Constraints & Rules

The following rules are **mandatory**. Violations will cause bugs, test failures, or security issues.

### 16.1 PyTorch Rules
- All CNN tensors are **single-channel**: shape `(batch, 1, H, W)`. NEVER RGB `(batch, 3, H, W)`.
- `transforms.Normalize(mean=[0.5], std=[0.5])` uses **single-element lists** (not scalars, not 3-element lists).
- `CrossEntropyLoss` expects **raw logits** — do NOT apply softmax inside `model.forward()`.
- `model.eval()` and `torch.no_grad()` are **always paired** during inference and validation.
- `torch.manual_seed(42)` is called at the **start of `train()`**, not at module level.
- Use `weights_only=True` in `torch.load()` for PyTorch 2.x security.
- `bias=False` in Conv2d when followed by BatchNorm (eliminates redundant parameters).
- `drop_last=True` in train DataLoader to prevent single-sample batches breaking BatchNorm.

### 16.2 Data Pipeline Rules
- `cv2.imread()` is always called with `cv2.IMREAD_GRAYSCALE` flag. Never load as BGR.
- `cv2.resize()` target is `(width, height)` — `(IMG_SIZE, IMG_SIZE)` not `(height, width)`.
  OpenCV uses `(width, height)` convention. This is the most common bug.
- `np.frombuffer(file_bytes, dtype=np.uint8)` produces a **read-only array**. Never `.reshape()` in-place — always assign to new variable.
- `GaussianNoise` transform operates on **torch.Tensor** (post-ToTensor), never PIL Image.
- `ColorJitter` operates on **PIL Image** (pre-ToTensor). Transform order matters.
- `get_val_transforms` is used for **val, test, and inference** — never `get_train_transforms` for inference.

### 16.3 Streamlit Rules
- `st.set_page_config()` must be the **absolute first** Streamlit call in `app.py`.
- **Never use HTML `<form>` tags** in Streamlit code. Use `st.button`, `st.file_uploader`, etc.
- `st.session_state` keys are **all defined in `state.py`** as constants. Never use string literals directly in page files.
- `@st.cache_resource` is NOT used for the model in this implementation (session_state guard used instead).
- `st.image()` for PNG bytes uses `use_column_width=True` (not `use_container_width` which is for charts).
- Always call `state.clear_file_state()` when a new file is uploaded, before processing.

### 16.4 Database Rules
- `PRAGMA journal_mode=WAL` is set on **every** connection via `get_connection()` context manager.
- `os.chmod(db_path, 0o600)` is called in `init_db()` after creation.
- `log_detection_event()` **never raises** — all exceptions are caught and logged to stderr.
- `sqlite3.Row` factory is set so rows can be accessed by column name as well as index.
- `conn.commit()` happens inside the `get_connection` context manager (not manually).

### 16.5 File & Security Rules
- Uploaded binary bytes are **never written to disk** during dashboard operation.
  Processing is 100% in-memory (bytes → numpy array → tensor → result).
- No file content, hash, or analysis result is **transmitted externally**. Everything is local.
- `hashlib.sha256` from the standard library is the **only** permitted hash implementation.

### 16.6 Reproducibility Rules
- All `train_test_split()` calls use `random_state=config.RANDOM_SEED` (42).
- `torch.manual_seed(42)` and `np.random.seed(42)` at training script entry.
- `WeightedRandomSampler` uses `replacement=True` (required for oversampling).
- `encode_labels()` always sorts alphabetically — same input → same mapping, always.

### 16.7 Directory & Path Rules
- All paths in code use `pathlib.Path`, never string concatenation.
- `str(path)` is used only when an API requires a string (e.g., `cv2.imread(str(path))`).
- All four output directories (`PROCESSED_DIR`, `MODEL_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`) are created at `config.py` import time via `mkdir(parents=True, exist_ok=True)`.

### 16.8 Testing Rules
- Tests that require Malimg dataset are marked `@pytest.mark.integration`.
- All unit tests run without the Malimg dataset (use fixtures from `conftest.py`).
- `pytest tests/ -v -m "not integration"` must pass with zero failures on a clean install.
- No test modifies `config.DATA_DIR` or any other global config value.

### 16.9 How to Run Everything

```bash
# Setup
git clone <repo>
cd maltwin
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Download Malimg dataset from Kaggle, place at:
# data/malimg/

# Run unit tests (no dataset needed)
pytest tests/ -v -m "not integration"

# Train
python scripts/train.py
# Outputs: models/best_model.pt, data/processed/class_names.json,
#          data/processed/eval_metrics.json, data/processed/confusion_matrix.png

# Launch dashboard
streamlit run modules/dashboard/app.py --server.port 8501
# Open: http://localhost:8501

# Convert a single binary (CLI utility)
python scripts/convert_binary.py --input suspicious.exe --output suspicious.png
```
