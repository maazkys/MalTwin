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
    if len(file_bytes) < 4:
        raise ValueError(
            "File is too small to be a valid binary (minimum 4 bytes required)"
        )
    if file_bytes[:2] == b'MZ':
        return 'PE'
    if file_bytes[:4] == b'\x7fELF':
        return 'ELF'
    hex_repr = file_bytes[:4].hex().upper()
    raise ValueError(
        f"Unsupported file format. Expected PE (.exe/.dll) or ELF binary. "
        f"Detected magic bytes: {hex_repr}"
    )


def compute_sha256(file_bytes: bytes) -> str:
    """
    Compute SHA-256 digest of raw file bytes.

    Args:
        file_bytes: raw bytes of any length including empty.

    Returns:
        Lowercase hexadecimal string of length exactly 64.
        Example: "a3f1c2d4e5b6a7f8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"

    Constraints:
        Standard library hashlib ONLY. No external services. No network.
        Deterministic: identical input always produces identical output.
        Output is always lowercase.
    """
    return hashlib.sha256(file_bytes).hexdigest()


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
    """
    bins = list(range(256))
    counts = np.bincount(img_array.flatten(), minlength=256).tolist()
    return {'bins': bins, 'counts': counts}


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
            'size_human':  str   — human-readable size
            'format':      str   — 'PE' or 'ELF'
            'sha256':      str   — 64-char hex (from compute_sha256)
            'upload_time': str   — ISO 8601 UTC string
        }

    size_human rules:
        size_bytes >= 1_048_576 → f"{size_bytes / 1_048_576:.2f} MB"
        size_bytes >= 1_024     → f"{size_bytes / 1_024:.2f} KB"
        else                    → f"{size_bytes:.2f} B"

    upload_time:
        datetime.utcnow().isoformat()
    """
    size_bytes = len(file_bytes)
    if size_bytes >= 1_048_576:
        size_human = f"{size_bytes / 1_048_576:.2f} MB"
    elif size_bytes >= 1_024:
        size_human = f"{size_bytes / 1_024:.2f} KB"
    else:
        size_human = f"{size_bytes:.2f} B"

    return {
        'name': filename,
        'size_bytes': size_bytes,
        'size_human': size_human,
        'format': file_format,
        'sha256': compute_sha256(file_bytes),
        'upload_time': datetime.utcnow().isoformat(),
    }
