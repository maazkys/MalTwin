"""
test_converter.py — matches actual MalTwin signatures.

Key facts from source:
- BinaryConverter.convert() expects raw bytes (not a Path)
- validate_binary_format() expects raw bytes (not a Path)
- compute_sha256() expects raw bytes (not a Path)
- compute_pixel_histogram() takes a numpy array, returns {'bins': [...], 'counts': [...]}
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Binary builders
# ---------------------------------------------------------------------------

def _make_pe_bytes(size: int = 1024) -> bytes:
    data = bytearray(size)
    data[0] = 0x4D; data[1] = 0x5A
    return bytes(data)

def _make_elf_bytes(size: int = 1024) -> bytes:
    data = bytearray(size)
    data[0] = 0x7F; data[1] = 0x45; data[2] = 0x4C; data[3] = 0x46
    return bytes(data)

def _make_invalid_bytes(size: int = 1024) -> bytes:
    return b'\x00' * size


@pytest.fixture(scope="module")
def converter_mod():
    from modules.binary_to_image import converter, utils
    return converter, utils


# ===========================================================================
# BinaryConverter — output shape and dtype
# ===========================================================================

class TestBinaryConverterOutputShape:

    def test_pe_returns_128x128(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        arr = bc.convert(_make_pe_bytes())
        assert arr.shape == (128, 128), f"Expected (128,128), got {arr.shape}"

    def test_elf_returns_128x128(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        arr = bc.convert(_make_elf_bytes())
        assert arr.shape == (128, 128)

    def test_output_dtype_uint8(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        arr = bc.convert(_make_pe_bytes())
        assert arr.dtype == np.uint8

    def test_pixel_values_in_range(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        arr = bc.convert(_make_pe_bytes())
        assert arr.min() >= 0 and arr.max() <= 255


# ===========================================================================
# BinaryConverter — determinism
# ===========================================================================

class TestBinaryConverterDeterminism:

    def test_same_bytes_same_output(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        b = _make_pe_bytes()
        arr1 = bc.convert(b)
        arr2 = bc.convert(b)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_bytes_different_output(self, converter_mod):
        converter, _ = converter_mod
        bc = converter.BinaryConverter()
        b1 = _make_pe_bytes(512)
        b2 = bytearray(_make_pe_bytes(512))
        b2[100:200] = bytes(range(100))
        arr1 = bc.convert(b1)
        arr2 = bc.convert(bytes(b2))
        assert not np.array_equal(arr1, arr2)


# ===========================================================================
# validate_binary_format
# ===========================================================================

class TestValidateBinaryFormat:

    def test_accepts_pe(self, converter_mod):
        _, utils = converter_mod
        result = utils.validate_binary_format(_make_pe_bytes())
        assert result in ("PE", "pe", True, "valid")

    def test_accepts_elf(self, converter_mod):
        _, utils = converter_mod
        result = utils.validate_binary_format(_make_elf_bytes())
        assert result in ("ELF", "elf", True, "valid")

    def test_rejects_invalid(self, converter_mod):
        _, utils = converter_mod
        with pytest.raises(Exception):
            utils.validate_binary_format(_make_invalid_bytes())

    def test_rejects_empty_bytes(self, converter_mod):
        _, utils = converter_mod
        with pytest.raises(Exception):
            utils.validate_binary_format(b"")


# ===========================================================================
# compute_sha256 — takes raw bytes
# ===========================================================================

class TestComputeSha256:

    def test_output_is_64_hex_chars(self, converter_mod):
        _, utils = converter_mod
        digest = utils.compute_sha256(_make_pe_bytes())
        assert len(digest) == 64
        assert all(c in "0123456789abcdefABCDEF" for c in digest)

    def test_matches_reference_hash(self, converter_mod):
        _, utils = converter_mod
        content = b"MZ" + b"\x00" * 62
        expected = hashlib.sha256(content).hexdigest()
        actual = utils.compute_sha256(content)
        assert actual.lower() == expected.lower()

    def test_different_bytes_different_hashes(self, converter_mod):
        _, utils = converter_mod
        h1 = utils.compute_sha256(_make_pe_bytes(256))
        h2 = utils.compute_sha256(_make_pe_bytes(512))
        assert h1 != h2

    def test_same_bytes_same_hash(self, converter_mod):
        _, utils = converter_mod
        b = _make_pe_bytes()
        assert utils.compute_sha256(b) == utils.compute_sha256(b)


# ===========================================================================
# histogram — takes numpy array, returns dict with 'bins' and 'counts'
# ===========================================================================

class TestHistogram:

    def test_returns_dict_with_bins_and_counts(self, converter_mod):
        _, utils = converter_mod
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = utils.compute_pixel_histogram(arr)
        assert isinstance(result, dict)
        assert "bins" in result and "counts" in result

    def test_bins_length_256(self, converter_mod):
        _, utils = converter_mod
        arr = np.zeros((128, 128), dtype=np.uint8)
        result = utils.compute_pixel_histogram(arr)
        assert len(result["bins"]) == 256

    def test_counts_length_256(self, converter_mod):
        _, utils = converter_mod
        arr = np.zeros((128, 128), dtype=np.uint8)
        result = utils.compute_pixel_histogram(arr)
        assert len(result["counts"]) == 256

    def test_counts_sum_to_total_pixels(self, converter_mod):
        _, utils = converter_mod
        arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        result = utils.compute_pixel_histogram(arr)
        assert sum(result["counts"]) == arr.size

    def test_bins_are_0_to_255(self, converter_mod):
        _, utils = converter_mod
        arr = np.zeros((64, 64), dtype=np.uint8)
        result = utils.compute_pixel_histogram(arr)
        assert result["bins"] == list(range(256))

# ---------------------------------------------------------------------------
# Malimg-consistent layout (train/inference consistency)
# ---------------------------------------------------------------------------

class TestMalimgLayout:
    @pytest.mark.parametrize('kb,width', [
        (5, 32), (20, 64), (45, 128), (80, 256), (150, 384),
        (300, 512), (800, 768), (2000, 1024),
    ])
    def test_width_table(self, kb, width):
        from modules.binary_to_image.converter import BinaryConverter
        assert BinaryConverter.malimg_width(kb * 1024) == width

    def test_default_layout_is_malimg(self):
        from modules.binary_to_image.converter import BinaryConverter
        assert BinaryConverter().layout == 'malimg'

    def test_malimg_layout_matches_reference_pipeline(self):
        """Uploaded bytes → same pixels as a Malimg-style PNG loaded by the dataset loader."""
        import cv2
        import numpy as np
        from modules.binary_to_image.converter import BinaryConverter
        rng = np.random.default_rng(1)
        data = rng.integers(0, 256, 50 * 1024, dtype=np.uint8).tobytes()   # 50 KB → width 128
        w = 128
        ref = np.frombuffer(data, dtype=np.uint8)[: (len(data) // w) * w].reshape(-1, w)
        ref = cv2.resize(ref, (128, 128), interpolation=cv2.INTER_LINEAR)
        out = BinaryConverter(128, layout='malimg').convert(data)
        assert np.array_equal(out, ref)

    def test_square_layout_still_available(self):
        from modules.binary_to_image.converter import BinaryConverter
        import numpy as np
        out = BinaryConverter(128, layout='square').convert(bytes(range(256)) * 64)
        assert out.shape == (128, 128) and out.dtype == np.uint8

    def test_tiny_file_does_not_crash(self):
        from modules.binary_to_image.converter import BinaryConverter
        out = BinaryConverter(128).convert(b'MZ' + b'\x00' * 62)   # 64 bytes, minimum
        assert out.shape == (128, 128)

    def test_invalid_layout(self):
        from modules.binary_to_image.converter import BinaryConverter
        with pytest.raises(ValueError):
            BinaryConverter(128, layout='diagonal')
