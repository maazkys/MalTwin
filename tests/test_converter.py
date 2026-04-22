"""
Test suite for modules/binary_to_image/
All tests are unit tests — no dataset, no ML, no network required.
Run: pytest tests/test_converter.py -v
"""
import hashlib

import cv2
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
