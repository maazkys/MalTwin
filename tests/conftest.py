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
