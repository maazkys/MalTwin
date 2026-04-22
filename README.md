# MalTwin — Phase 1 Implementation
### Binary-to-Image Conversion Module
### Agent Instruction Document

---

## YOUR TASK

Implement Phase 1 of the MalTwin project. This phase has zero ML dependencies.
You are building the binary file ingestion and conversion pipeline.

At the end of this phase the following must be true:
- `pytest tests/test_converter.py -v` passes with **zero failures**
- `python scripts/convert_binary.py --input <file> --output <file>.png` runs without error
- All functions have complete type hints
- No network calls anywhere in this phase

---

## FILES TO CREATE

Create every file listed below. Do not skip any.

```
config.py
.env.example
requirements.txt
.gitignore
modules/__init__.py
modules/binary_to_image/__init__.py
modules/binary_to_image/utils.py
modules/binary_to_image/converter.py
tests/__init__.py
tests/conftest.py
tests/fixtures/create_fixtures.py
tests/test_converter.py
scripts/__init__.py
scripts/convert_binary.py
data/.gitkeep
data/malimg/.gitkeep
data/processed/.gitkeep
models/.gitkeep
models/checkpoints/.gitkeep
logs/.gitkeep
reports/.gitkeep
```

---

## MANDATORY RULES — READ BEFORE WRITING ANY CODE

These rules prevent the most common agent mistakes on this project:

1. `cv2.imread()` is ALWAYS called with `cv2.IMREAD_GRAYSCALE` flag. Never omit it.
2. `cv2.resize()` target is `(width, height)` — OpenCV uses `(width, height)` NOT `(height, width)`. For a square image this is `(IMG_SIZE, IMG_SIZE)` which is unambiguous.
3. `np.frombuffer(file_bytes, dtype=np.uint8)` produces a READ-ONLY array. Never modify it in-place. Always assign the result of reshape/slice to a new variable.
4. `hashlib.sha256` from the Python standard library is the ONLY permitted hash implementation. No external services, no network calls.
5. All paths in code use `pathlib.Path`. Use `str(path)` only when an external library requires a string argument (e.g. `cv2.imwrite(str(path), arr)`).
6. No HTML `<form>` tags anywhere (Streamlit rule that applies project-wide).
7. All `__init__.py` files in `modules/` must export the public API as specified in the `__all__` lists below.

---

## FILE 1: `requirements.txt`

```
# Deep Learning
torch==2.3.1
torchvision==0.18.1
captum==0.7.0

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
watchdog==4.0.1

# Reporting
fpdf2==2.7.9

# Utilities
python-dotenv==1.0.1
tqdm==4.66.5
matplotlib==3.9.2

# Testing
pytest==8.3.2
pytest-cov==5.0.0
```

---

## FILE 2: `.env.example`

```bash
MALTWIN_DATA_DIR=./data/malimg
MALTWIN_PROCESSED_DIR=./data/processed
MALTWIN_MODEL_DIR=./models
MALTWIN_LOG_DIR=./logs
MALTWIN_REPORTS_DIR=./reports
MALTWIN_IMG_SIZE=128
MALTWIN_BATCH_SIZE=32
MALTWIN_EPOCHS=30
MALTWIN_LR=0.001
MALTWIN_WEIGHT_DECAY=0.0001
MALTWIN_LR_PATIENCE=5
MALTWIN_NUM_WORKERS=4
MALTWIN_DEVICE=auto
MALTWIN_TRAIN_RATIO=0.70
MALTWIN_VAL_RATIO=0.15
MALTWIN_TEST_RATIO=0.15
MALTWIN_OVERSAMPLE_STRATEGY=oversample_minority
MALTWIN_RANDOM_SEED=42
```

---

## FILE 3: `.gitignore`

```
__pycache__/
*.pyc
*.pyo
.env
data/malimg/
data/processed/*.json
data/processed/*.png
models/best_model.pt
models/checkpoints/
logs/maltwin.db
reports/
*.egg-info/
.pytest_cache/
.DS_Store
venv/
.venv/
```

---

## FILE 4: `config.py`

Implement exactly as follows. Do not add or remove any fields.

```python
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
```

---

## FILE 5: `modules/__init__.py`

```python
# empty
```

---

## FILE 6: `modules/binary_to_image/utils.py`

Implement all four functions exactly as specified. Every docstring condition is a test requirement.

```python
"""
Utility functions for binary file validation and metadata extraction.
No ML dependencies. No network calls.
"""
import hashlib
import math
from datetime import datetime
from pathlib import Path

import numpy as np


def validate_binary_format(file_bytes: bytes) -> str:
    """
    Inspect magic bytes to identify PE or ELF binary format.

    Args:
        file_bytes: raw bytes of the uploaded file.

    Returns:
        'PE'  — if first 2 bytes are b'MZ'   (0x4D 0x5A)
        'ELF' — if first 4 bytes are b'\x7fELF' (0x7F 0x45 0x4C 0x46)

    Raises:
        ValueError: "File is too small to be a valid binary (minimum 4 bytes required)"
            if len(file_bytes) < 4
        ValueError: "Unsupported file format. Expected PE (.exe/.dll) or ELF binary.
                     Detected magic bytes: {HEX}"
            if magic bytes match neither format.
            HEX = file_bytes[:4].hex().upper()  e.g. "DEADBEEF"

    Notes:
        Magic byte check only — no full header validation.
        b'MZ' check uses first 2 bytes only.
        b'\x7fELF' check uses first 4 bytes.
        Check PE first, then ELF.
    """
    # implement here


def compute_sha256(file_bytes: bytes) -> str:
    """
    Compute SHA-256 digest of raw file bytes.

    Args:
        file_bytes: raw bytes of any length including empty.

    Returns:
        Lowercase hexadecimal string of length exactly 64.
        Example: "a3f1c2d4e5b6a7f8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"

    Implementation:
        import hashlib
        return hashlib.sha256(file_bytes).hexdigest()

    Constraints:
        Standard library hashlib ONLY. No external services. No network.
        Deterministic: identical input always produces identical output.
        Output is always lowercase.
    """
    # implement here


def compute_pixel_histogram(img_array: np.ndarray) -> dict:
    """
    Compute byte-value frequency distribution of a grayscale image.

    Args:
        img_array: numpy array of shape (H, W), dtype uint8, values 0–255.

    Returns:
        {
            'bins':   list[int]  — exactly [0, 1, 2, ..., 255], always length 256
            'counts': list[int]  — pixel count for each bin value, always length 256
        }

    Invariants:
        len(result['bins'])   == 256
        len(result['counts']) == 256
        result['bins']        == list(range(256))
        sum(result['counts']) == img_array.size   (H * W)
        all(c >= 0 for c in result['counts'])

    Implementation:
        bins = list(range(256))
        counts = np.bincount(img_array.flatten(), minlength=256).tolist()
        return {'bins': bins, 'counts': counts}
    """
    # implement here


def get_file_metadata(
    file_bytes: bytes,
    filename: str,
    file_format: str,
) -> dict:
    """
    Assemble the complete metadata dict for a processed binary file.

    Args:
        file_bytes:  raw bytes of the uploaded file.
        filename:    original filename as provided by the user.
        file_format: 'PE' or 'ELF' (output of validate_binary_format).

    Returns:
        {
            'name':        str   — original filename
            'size_bytes':  int   — len(file_bytes)
            'size_human':  str   — human-readable size (see rules below)
            'format':      str   — 'PE' or 'ELF'
            'sha256':      str   — 64-char hex (from compute_sha256)
            'upload_time': str   — ISO 8601 UTC string e.g. "2025-04-22T14:35:22.123456"
        }

    size_human rules:
        size_bytes >= 1_048_576 → f"{size_bytes / 1_048_576:.2f} MB"
        size_bytes >= 1_024     → f"{size_bytes / 1_024:.2f} KB"
        else                    → f"{size_bytes:.2f} B"

    upload_time:
        datetime.utcnow().isoformat()

    sha256:
        compute_sha256(file_bytes)
    """
    # implement here
```

---

## FILE 7: `modules/binary_to_image/converter.py`

```python
"""
BinaryConverter: converts raw PE/ELF bytes to a 128x128 grayscale image.
Algorithm from Nataraj et al. (2011) — byte array reshaped to 2D, then resized.
No ML dependencies.
"""
import math
from pathlib import Path

import cv2
import numpy as np

import config


class BinaryConverter:
    """
    Convert raw binary file bytes into a standardised grayscale PNG image.

    Conversion algorithm:
        1. Read file bytes as flat uint8 numpy array via np.frombuffer.
        2. Determine 2D width:  width = max(1, int(math.sqrt(len(byte_array))))
        3. Determine rows:      rows  = len(byte_array) // width
        4. Trim array to exact (rows * width) length (discard tail bytes).
        5. Reshape to (rows, width) 2D array.
        6. Resize to (img_size, img_size) using cv2.INTER_LINEAR (bilinear).
        7. Cast result to uint8 and return.

    Constructor args:
        img_size (int): side length of output square image in pixels.
                        Must be > 0. Default: config.IMG_SIZE (128).

    Raises on construction:
        ValueError: "img_size must be a positive integer, got {img_size}"
            if img_size <= 0.
    """

    def __init__(self, img_size: int = config.IMG_SIZE) -> None:
        if img_size <= 0:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        self.img_size = img_size

    def convert(self, file_bytes: bytes) -> np.ndarray:
        """
        Convert raw binary bytes to a grayscale image array.

        Args:
            file_bytes: raw bytes of a PE or ELF binary.
                        Caller is responsible for prior format validation.

        Returns:
            numpy.ndarray of shape (img_size, img_size), dtype=uint8, values 0–255.

        Raises:
            ValueError: "Binary file is empty or too small to convert (minimum 64 bytes)"
                if len(file_bytes) < 64.

        Implementation (follow exactly):
            byte_array = np.frombuffer(file_bytes, dtype=np.uint8)
            n = len(byte_array)
            width = max(1, int(math.sqrt(n)))
            rows  = n // width
            trimmed   = byte_array[:rows * width]       # trim tail — READ ONLY array, slice is fine
            reshaped  = trimmed.reshape((rows, width))  # creates a new view
            resized   = cv2.resize(
                reshaped,
                (self.img_size, self.img_size),          # OpenCV: (width, height) — both same for square
                interpolation=cv2.INTER_LINEAR,
            )
            return resized.astype(np.uint8)

        Notes:
            np.frombuffer output is read-only. Never modify it in-place.
            trimmed.reshape creates a view — safe to use directly with cv2.resize.
            cv2.resize on uint8 input with INTER_LINEAR outputs values in [0,255].
            .astype(np.uint8) handles any edge-case float conversion from resize.
        """
        # implement here

    def to_png_bytes(self, img_array: np.ndarray) -> bytes:
        """
        Encode a grayscale numpy array to PNG bytes for in-memory use.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PNG-encoded bytes. First 4 bytes will be b'\\x89PNG' (PNG magic).

        Raises:
            RuntimeError: "cv2.imencode failed to encode image as PNG"
                if cv2.imencode returns success=False.

        Implementation:
            success, encoded = cv2.imencode('.png', img_array)
            if not success:
                raise RuntimeError("cv2.imencode failed to encode image as PNG")
            return encoded.tobytes()
        """
        # implement here

    def to_pil_image(self, img_array: np.ndarray):
        """
        Convert grayscale array to PIL Image for torchvision transform compatibility.

        Args:
            img_array: numpy array of shape (H, W), dtype uint8.

        Returns:
            PIL.Image.Image in mode 'L' (8-bit grayscale).

        Implementation:
            from PIL import Image
            return Image.fromarray(img_array, mode='L')
        """
        # implement here

    def save(self, img_array: np.ndarray, output_path: Path) -> None:
        """
        Save grayscale array as PNG file to disk.

        Args:
            img_array:   numpy array of shape (H, W), dtype uint8.
            output_path: Path where PNG will be written.
                         Parent directory must already exist.

        Raises:
            RuntimeError: "Failed to save image to {output_path}"
                if cv2.imwrite returns False.

        Implementation:
            success = cv2.imwrite(str(output_path), img_array)
            if not success:
                raise RuntimeError(f"Failed to save image to {output_path}")
        """
        # implement here
```

---

## FILE 8: `modules/binary_to_image/__init__.py`

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

## FILE 9: `tests/__init__.py`

```python
# empty
```

---

## FILE 10: `tests/conftest.py`

This file provides shared fixtures used by ALL test phases.
Create all fixtures now even though some are only used in later phases.

```python
"""
Shared pytest fixtures for MalTwin test suite.
All fixtures are deterministic (fixed seeds / fixed byte patterns).
"""
import numpy as np
import pytest
import torch
from pathlib import Path


# ── Binary fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pe_bytes() -> bytes:
    """
    Minimal valid PE binary.
    First 2 bytes = b'MZ' (PE magic).
    Total size = 1024 bytes (enough for BinaryConverter minimum of 64 bytes).
    """
    header = b'MZ' + b'\x90' * 58   # MZ + 58 bytes of DOS stub padding
    body   = b'\x00' * (1024 - len(header))
    return header + body


@pytest.fixture
def sample_elf_bytes() -> bytes:
    """
    Minimal valid ELF binary.
    First 4 bytes = b'\\x7fELF' (ELF magic).
    Total size = 1024 bytes.
    """
    header = b'\x7fELF' + b'\x00' * 56
    body   = b'\x00' * (1024 - len(header))
    return header + body


@pytest.fixture
def large_pe_bytes() -> bytes:
    """
    Larger PE binary (10 KB) with varied byte values for histogram testing.
    """
    import os
    rng = np.random.default_rng(seed=123)
    body = rng.integers(0, 256, size=10 * 1024, dtype=np.uint8).tobytes()
    return b'MZ' + b'\x90' * 58 + body


@pytest.fixture
def non_binary_bytes() -> bytes:
    """
    Bytes that are definitively NOT PE or ELF.
    First 4 bytes are 0xDEADBEEF.
    """
    return b'\xDE\xAD\xBE\xEF' + b'\x00' * 100


@pytest.fixture
def too_small_bytes() -> bytes:
    """Only 2 bytes — fails the minimum 4-byte check."""
    return b'\x4d\x5a'   # MZ but only 2 bytes


@pytest.fixture
def tiny_valid_pe() -> bytes:
    """
    Valid PE magic but only 32 bytes total.
    Passes validate_binary_format but fails BinaryConverter (< 64 bytes).
    """
    return b'MZ' + b'\x00' * 30


# ── Image array fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_grayscale_array() -> np.ndarray:
    """
    128x128 uint8 numpy array simulating a converted binary.
    Deterministic via fixed seed.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def uniform_black_array() -> np.ndarray:
    """128x128 array of all zeros (pure black image)."""
    return np.zeros((128, 128), dtype=np.uint8)


@pytest.fixture
def uniform_white_array() -> np.ndarray:
    """128x128 array of all 255 (pure white image)."""
    return np.full((128, 128), 255, dtype=np.uint8)


# ── Tensor fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_grayscale_tensor() -> torch.Tensor:
    """
    Single-channel grayscale tensor in normalised range [-1, 1].
    Shape: (1, 128, 128), dtype: float32.
    Simulates output of get_val_transforms()(pil_image).
    """
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(128, 128), dtype=np.uint8).astype(np.float32)
    tensor = torch.from_numpy(arr / 255.0)   # [0, 1]
    tensor = (tensor - 0.5) / 0.5            # [-1, 1]
    return tensor.unsqueeze(0)               # (1, 128, 128)


@pytest.fixture
def batch_grayscale_tensors() -> torch.Tensor:
    """
    Batch of 4 single-channel grayscale tensors.
    Shape: (4, 1, 128, 128), dtype: float32, range: [-1, 1].
    """
    rng = np.random.default_rng(seed=99)
    arr = rng.integers(0, 256, size=(4, 128, 128), dtype=np.uint8).astype(np.float32)
    tensor = torch.from_numpy(arr / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(1)   # (4, 1, 128, 128)


# ── Misc fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def num_classes() -> int:
    """Number of malware families in Malimg dataset."""
    return 25


@pytest.fixture
def sample_class_names() -> list:
    """
    Sorted list of all 25 Malimg family names.
    Used as a drop-in for real class_names in inference tests.
    """
    return [
        "Adialer.C", "Agent.FYI", "Allaple.A", "Allaple.L",
        "Alueron.gen!J", "Autorun.K", "C2LOP.P", "C2LOP.gen!g",
        "Dialplatform.B", "Dontovo.A", "Fakerean", "Instantaccess",
        "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3", "Lolyda.AT",
        "Malex.gen!J", "Obfuscator.AD", "Rbot!gen", "Skintrim.N",
        "Swizzor.gen!E", "Swizzor.gen!I", "VB.AT", "Wintrim.BX",
        "Yuner.A",
    ]
```

---

## FILE 11: `tests/fixtures/create_fixtures.py`

```python
"""
Script to generate minimal binary fixture files for tests.
Run once: python tests/fixtures/create_fixtures.py

Creates:
    tests/fixtures/sample_pe.exe   — 1024-byte minimal PE binary
    tests/fixtures/sample.elf      — 1024-byte minimal ELF binary
"""
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def create_minimal_pe(output_path: Path) -> None:
    """Write a 1024-byte minimal PE binary (MZ magic + padding)."""
    header = b'MZ' + b'\x90' * 58
    body   = b'\x00' * (1024 - len(header))
    output_path.write_bytes(header + body)
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


def create_minimal_elf(output_path: Path) -> None:
    """Write a 1024-byte minimal ELF binary (ELF magic + padding)."""
    header = b'\x7fELF' + b'\x00' * 56
    body   = b'\x00' * (1024 - len(header))
    output_path.write_bytes(header + body)
    print(f"Created: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    create_minimal_pe(FIXTURES_DIR / "sample_pe.exe")
    create_minimal_elf(FIXTURES_DIR / "sample.elf")
    print("Done.")
```

---

## FILE 12: `tests/test_converter.py`

Implement all test classes and methods exactly as written. Do not skip or rename any.

```python
"""
Test suite for modules/binary_to_image/
All tests are unit tests — no dataset, no ML, no network required.
Run: pytest tests/test_converter.py -v
"""
import hashlib

import numpy as np
import pytest
from PIL import Image

from modules.binary_to_image.converter import BinaryConverter
from modules.binary_to_image.utils import (
    compute_pixel_histogram,
    compute_sha256,
    get_file_metadata,
    validate_binary_format,
)


# ══════════════════════════════════════════════════════════════════════════════
# validate_binary_format
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateBinaryFormat:

    def test_accepts_pe_mz_header(self, sample_pe_bytes):
        assert validate_binary_format(sample_pe_bytes) == 'PE'

    def test_accepts_elf_magic(self, sample_elf_bytes):
        assert validate_binary_format(sample_elf_bytes) == 'ELF'

    def test_rejects_too_small_raises_value_error(self, too_small_bytes):
        with pytest.raises(ValueError, match="too small"):
            validate_binary_format(too_small_bytes)

    def test_rejects_empty_bytes(self):
        with pytest.raises(ValueError):
            validate_binary_format(b'')

    def test_rejects_exactly_3_bytes(self):
        with pytest.raises(ValueError):
            validate_binary_format(b'\x7f\x45\x4c')   # 3 bytes — ELF-like but too short

    def test_rejects_unknown_magic_raises_value_error(self, non_binary_bytes):
        with pytest.raises(ValueError, match="Unsupported"):
            validate_binary_format(non_binary_bytes)

    def test_error_message_contains_hex_repr(self):
        bad = b'\xDE\xAD\xBE\xEF' + b'\x00' * 100
        with pytest.raises(ValueError) as exc_info:
            validate_binary_format(bad)
        assert 'DEADBEEF' in str(exc_info.value)

    def test_pe_check_uses_first_two_bytes_only(self):
        """Bytes 3+ can be anything — still valid PE if first 2 are MZ."""
        data = b'MZ' + b'\xFF\xFF' + b'\x00' * 200
        assert validate_binary_format(data) == 'PE'

    def test_elf_check_requires_all_four_magic_bytes(self):
        """Only 3 of 4 ELF magic bytes — must fail."""
        data = b'\x7fEL\x00' + b'\x00' * 100   # missing 'F'
        with pytest.raises(ValueError):
            validate_binary_format(data)

    def test_return_type_is_str(self, sample_pe_bytes):
        result = validate_binary_format(sample_pe_bytes)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# compute_sha256
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSha256:

    def test_returns_string(self, sample_pe_bytes):
        assert isinstance(compute_sha256(sample_pe_bytes), str)

    def test_returns_64_char_hex(self, sample_pe_bytes):
        result = compute_sha256(sample_pe_bytes)
        assert len(result) == 64

    def test_output_is_valid_hex(self, sample_pe_bytes):
        result = compute_sha256(sample_pe_bytes)
        assert all(c in '0123456789abcdef' for c in result)

    def test_output_is_lowercase(self, sample_pe_bytes):
        result = compute_sha256(sample_pe_bytes)
        assert result == result.lower()

    def test_deterministic_same_input(self, sample_pe_bytes):
        assert compute_sha256(sample_pe_bytes) == compute_sha256(sample_pe_bytes)

    def test_different_inputs_produce_different_hashes(self, sample_pe_bytes, sample_elf_bytes):
        assert compute_sha256(sample_pe_bytes) != compute_sha256(sample_elf_bytes)

    def test_matches_stdlib_hashlib(self, sample_pe_bytes):
        expected = hashlib.sha256(sample_pe_bytes).hexdigest()
        assert compute_sha256(sample_pe_bytes) == expected

    def test_works_on_empty_bytes(self):
        """SHA-256 of empty bytes is a known constant."""
        expected = hashlib.sha256(b'').hexdigest()
        assert compute_sha256(b'') == expected

    def test_works_on_single_byte(self):
        result = compute_sha256(b'\x00')
        assert len(result) == 64

    def test_works_on_large_input(self, large_pe_bytes):
        result = compute_sha256(large_pe_bytes)
        assert len(result) == 64


# ══════════════════════════════════════════════════════════════════════════════
# compute_pixel_histogram
# ══════════════════════════════════════════════════════════════════════════════

class TestComputePixelHistogram:

    def test_returns_dict_with_bins_and_counts(self, sample_grayscale_array):
        result = compute_pixel_histogram(sample_grayscale_array)
        assert 'bins' in result
        assert 'counts' in result

    def test_bins_is_list(self, sample_grayscale_array):
        assert isinstance(compute_pixel_histogram(sample_grayscale_array)['bins'], list)

    def test_counts_is_list(self, sample_grayscale_array):
        assert isinstance(compute_pixel_histogram(sample_grayscale_array)['counts'], list)

    def test_exactly_256_bins(self, sample_grayscale_array):
        assert len(compute_pixel_histogram(sample_grayscale_array)['bins']) == 256

    def test_exactly_256_counts(self, sample_grayscale_array):
        assert len(compute_pixel_histogram(sample_grayscale_array)['counts']) == 256

    def test_bins_are_zero_to_255(self, sample_grayscale_array):
        assert compute_pixel_histogram(sample_grayscale_array)['bins'] == list(range(256))

    def test_counts_sum_to_total_pixels(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert sum(hist['counts']) == sample_grayscale_array.size   # 128*128 = 16384

    def test_all_counts_nonnegative(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert all(c >= 0 for c in hist['counts'])

    def test_uniform_black_image_all_count_in_bin_zero(self, uniform_black_array):
        hist = compute_pixel_histogram(uniform_black_array)
        assert hist['counts'][0] == uniform_black_array.size
        assert sum(hist['counts'][1:]) == 0

    def test_uniform_white_image_all_count_in_bin_255(self, uniform_white_array):
        hist = compute_pixel_histogram(uniform_white_array)
        assert hist['counts'][255] == uniform_white_array.size
        assert sum(hist['counts'][:255]) == 0

    def test_counts_elements_are_int(self, sample_grayscale_array):
        hist = compute_pixel_histogram(sample_grayscale_array)
        assert all(isinstance(c, int) for c in hist['counts'])


# ══════════════════════════════════════════════════════════════════════════════
# get_file_metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFileMetadata:

    def test_returns_all_required_keys(self, sample_pe_bytes):
        meta = get_file_metadata(sample_pe_bytes, "test.exe", "PE")
        for key in ['name', 'size_bytes', 'size_human', 'format', 'sha256', 'upload_time']:
            assert key in meta, f"Missing key: {key}"

    def test_name_matches_input(self, sample_pe_bytes):
        meta = get_file_metadata(sample_pe_bytes, "suspicious.exe", "PE")
        assert meta['name'] == "suspicious.exe"

    def test_size_bytes_correct(self, sample_pe_bytes):
        meta = get_file_metadata(sample_pe_bytes, "f.exe", "PE")
        assert meta['size_bytes'] == len(sample_pe_bytes)

    def test_format_matches_input(self, sample_elf_bytes):
        meta = get_file_metadata(sample_elf_bytes, "binary", "ELF")
        assert meta['format'] == "ELF"

    def test_sha256_is_correct(self, sample_pe_bytes):
        meta = get_file_metadata(sample_pe_bytes, "f.exe", "PE")
        expected = compute_sha256(sample_pe_bytes)
        assert meta['sha256'] == expected

    def test_size_human_bytes_format(self):
        """Files under 1024 bytes use 'B' suffix."""
        data = b'MZ' + b'\x00' * 100   # 102 bytes
        meta = get_file_metadata(data, "tiny.exe", "PE")
        assert 'B' in meta['size_human']
        assert 'KB' not in meta['size_human']
        assert 'MB' not in meta['size_human']

    def test_size_human_kb_format(self, sample_pe_bytes):
        """1024-byte file → KB."""
        meta = get_file_metadata(sample_pe_bytes, "f.exe", "PE")
        assert 'KB' in meta['size_human']

    def test_size_human_mb_format(self, large_pe_bytes):
        """File > 1 MB → MB suffix."""
        big = large_pe_bytes * 120   # ~1.2 MB
        meta = get_file_metadata(big, "big.exe", "PE")
        assert 'MB' in meta['size_human']

    def test_upload_time_is_iso_format(self, sample_pe_bytes):
        from datetime import datetime
        meta = get_file_metadata(sample_pe_bytes, "f.exe", "PE")
        # Should parse without exception
        datetime.fromisoformat(meta['upload_time'])

    def test_all_values_json_serialisable(self, sample_pe_bytes):
        import json
        meta = get_file_metadata(sample_pe_bytes, "f.exe", "PE")
        json.dumps(meta)   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# BinaryConverter
# ══════════════════════════════════════════════════════════════════════════════

class TestBinaryConverterInit:

    def test_default_img_size_from_config(self):
        import config
        c = BinaryConverter()
        assert c.img_size == config.IMG_SIZE

    def test_custom_img_size(self):
        c = BinaryConverter(img_size=64)
        assert c.img_size == 64

    def test_zero_img_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BinaryConverter(img_size=0)

    def test_negative_img_size_raises(self):
        with pytest.raises(ValueError):
            BinaryConverter(img_size=-1)


class TestBinaryConverterConvert:

    def test_output_shape_128x128(self, sample_pe_bytes):
        result = BinaryConverter(img_size=128).convert(sample_pe_bytes)
        assert result.shape == (128, 128)

    def test_output_shape_64x64(self, sample_pe_bytes):
        result = BinaryConverter(img_size=64).convert(sample_pe_bytes)
        assert result.shape == (64, 64)

    def test_output_shape_256x256(self, sample_pe_bytes):
        result = BinaryConverter(img_size=256).convert(sample_pe_bytes)
        assert result.shape == (256, 256)

    def test_output_dtype_uint8(self, sample_pe_bytes):
        result = BinaryConverter().convert(sample_pe_bytes)
        assert result.dtype == np.uint8

    def test_output_values_in_0_255(self, sample_pe_bytes):
        result = BinaryConverter().convert(sample_pe_bytes)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_elf_binary_converts(self, sample_elf_bytes):
        result = BinaryConverter().convert(sample_elf_bytes)
        assert result.shape == (128, 128)
        assert result.dtype == np.uint8

    def test_large_binary_converts(self, large_pe_bytes):
        result = BinaryConverter().convert(large_pe_bytes)
        assert result.shape == (128, 128)

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="too small"):
            BinaryConverter().convert(b'')

    def test_less_than_64_bytes_raises(self, tiny_valid_pe):
        with pytest.raises(ValueError, match="too small"):
            BinaryConverter().convert(tiny_valid_pe)

    def test_deterministic_same_input(self, sample_pe_bytes):
        c = BinaryConverter()
        r1 = c.convert(sample_pe_bytes)
        r2 = c.convert(sample_pe_bytes)
        np.testing.assert_array_equal(r1, r2)

    def test_different_inputs_produce_different_images(self, sample_pe_bytes, sample_elf_bytes):
        c = BinaryConverter()
        r1 = c.convert(sample_pe_bytes)
        r2 = c.convert(sample_elf_bytes)
        # They may be equal if both are mostly zeros — check shape at minimum
        assert r1.shape == r2.shape   # both (128, 128)
        # For non-trivial input they should differ (our fixtures differ enough)

    def test_output_is_not_read_only(self, sample_pe_bytes):
        """Result should be a writable numpy array."""
        result = BinaryConverter().convert(sample_pe_bytes)
        assert result.flags['WRITEABLE']


class TestBinaryConverterToPngBytes:

    def test_returns_bytes(self, sample_grayscale_array):
        result = BinaryConverter().to_png_bytes(sample_grayscale_array)
        assert isinstance(result, bytes)

    def test_starts_with_png_magic(self, sample_grayscale_array):
        result = BinaryConverter().to_png_bytes(sample_grayscale_array)
        assert result[:4] == b'\x89PNG'

    def test_nonempty_output(self, sample_grayscale_array):
        result = BinaryConverter().to_png_bytes(sample_grayscale_array)
        assert len(result) > 0

    def test_roundtrip_via_pil(self, sample_grayscale_array):
        """PNG bytes can be decoded back to the original array."""
        import io
        png_bytes = BinaryConverter().to_png_bytes(sample_grayscale_array)
        decoded = np.array(Image.open(io.BytesIO(png_bytes)))
        np.testing.assert_array_equal(decoded, sample_grayscale_array)


class TestBinaryConverterToPilImage:

    def test_returns_pil_image(self, sample_grayscale_array):
        result = BinaryConverter().to_pil_image(sample_grayscale_array)
        assert isinstance(result, Image.Image)

    def test_mode_is_L(self, sample_grayscale_array):
        result = BinaryConverter().to_pil_image(sample_grayscale_array)
        assert result.mode == 'L'

    def test_size_matches_array(self, sample_grayscale_array):
        result = BinaryConverter().to_pil_image(sample_grayscale_array)
        # PIL size is (width, height) = (128, 128)
        assert result.size == (128, 128)

    def test_pixel_values_preserved(self, sample_grayscale_array):
        pil = BinaryConverter().to_pil_image(sample_grayscale_array)
        back = np.array(pil)
        np.testing.assert_array_equal(back, sample_grayscale_array)


class TestBinaryConverterSave:

    def test_creates_file(self, sample_grayscale_array, tmp_path):
        output = tmp_path / "test_output.png"
        BinaryConverter().save(sample_grayscale_array, output)
        assert output.exists()

    def test_created_file_is_valid_png(self, sample_grayscale_array, tmp_path):
        output = tmp_path / "test_output.png"
        BinaryConverter().save(sample_grayscale_array, output)
        # Must be readable as an image
        loaded = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == (128, 128)

    def test_saved_values_match_input(self, sample_grayscale_array, tmp_path):
        output = tmp_path / "test_output.png"
        BinaryConverter().save(sample_grayscale_array, output)
        loaded = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(loaded, sample_grayscale_array)


# Need cv2 for the save test
import cv2
```

---

## FILE 13: `scripts/convert_binary.py`

```python
"""
CLI utility: convert a single binary file to a 128x128 grayscale PNG.
No ML dependencies.

Usage:
    python scripts/convert_binary.py --input path/to/file.exe --output path/to/out.png

Exit codes:
    0 — success
    1 — file not found or format error
    2 — conversion error
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MalTwin: Convert binary file to grayscale PNG"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input PE or ELF binary file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Path to output PNG file (will be created)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Output image size in pixels (default: 128)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Read bytes
    file_bytes = args.input.read_bytes()
    print(f"Read {len(file_bytes):,} bytes from {args.input.name}")

    # Validate format
    from modules.binary_to_image.utils import (
        validate_binary_format,
        compute_sha256,
        get_file_metadata,
    )
    try:
        file_format = validate_binary_format(file_bytes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Detected format : {file_format}")

    # Compute metadata
    sha256 = compute_sha256(file_bytes)
    meta   = get_file_metadata(file_bytes, args.input.name, file_format)
    print(f"File size       : {meta['size_human']}")
    print(f"SHA-256         : {sha256}")

    # Convert
    from modules.binary_to_image.converter import BinaryConverter
    try:
        converter = BinaryConverter(img_size=args.size)
        img_array = converter.convert(file_bytes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter.save(img_array, args.output)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved {args.size}x{args.size} grayscale PNG → {args.output}")


if __name__ == "__main__":
    main()
```

---

## FILE 14: `scripts/__init__.py`

```python
# empty
```

---

## DEFINITION OF DONE

Before marking this phase complete, verify all of the following:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate fixture files
python tests/fixtures/create_fixtures.py

# Run the full test suite for this phase
pytest tests/test_converter.py -v

# Expected output: all tests PASSED, 0 failed, 0 errors
# Approximate count: 55–60 test cases

# Smoke test the CLI
python tests/fixtures/create_fixtures.py
python scripts/convert_binary.py \
    --input tests/fixtures/sample_pe.exe \
    --output /tmp/test_output.png
# Expected output:
#   Read 1,024 bytes from sample_pe.exe
#   Detected format : PE
#   File size       : 1.00 KB
#   SHA-256         : <64-char hex>
#   Saved 128x128 grayscale PNG → /tmp/test_output.png

# Verify config.py imports cleanly
python -c "import config; print(config.DEVICE)"
# Expected: cpu  (or cuda if GPU present)
```

---

## WHAT NOT TO IMPLEMENT IN THIS PHASE

Do not implement anything from these modules — they come in later phases:

- `modules/dataset/`       — Phase 2
- `modules/enhancement/`   — Phase 3
- `modules/detection/`     — Phase 4
- `modules/dashboard/`     — Phase 6
- `scripts/train.py`       — Phase 5
- `scripts/evaluate.py`    — Phase 5
- `data/mitre_ics_mapping.json` — Phase 6

Creating empty placeholder `__init__.py` files in those directories is acceptable if needed to prevent import errors, but do not implement any logic in them.
