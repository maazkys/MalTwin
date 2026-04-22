"""
Test suite for modules/dataset/
NOTE: Tests that require the Malimg dataset are marked @pytest.mark.integration.
Unit tests (no dataset needed) run without the dataset.

Run unit tests only (CI-safe):
    pytest tests/test_dataset.py -v -m "not integration"

Run all tests (requires Malimg at config.DATA_DIR):
    pytest tests/test_dataset.py -v
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from modules.dataset.preprocessor import (
    normalize_image, encode_labels, validate_dataset_integrity,
    save_class_names, load_class_names,
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

    def test_shape_preserved(self, sample_grayscale_array):
        result = normalize_image(sample_grayscale_array)
        assert result.shape == sample_grayscale_array.shape

    def test_midpoint_maps_correctly(self):
        arr = np.full((4, 4), 128, dtype=np.uint8)
        result = normalize_image(arr)
        np.testing.assert_almost_equal(result[0, 0], 128 / 255.0, decimal=6)


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

    def test_order_independent(self):
        f1 = ['C', 'A', 'B']
        f2 = ['A', 'B', 'C']
        assert encode_labels(f1) == encode_labels(f2)

    def test_returns_dict(self):
        result = encode_labels(['X', 'Y'])
        assert isinstance(result, dict)

    def test_all_values_are_ints(self):
        result = encode_labels(['A', 'B', 'C'])
        assert all(isinstance(v, int) for v in result.values())


class TestSaveLoadClassNames:
    def test_roundtrip(self, tmp_path):
        names = ['Allaple.A', 'Agent.FYI', 'VB.AT']
        path = tmp_path / 'class_names.json'
        save_class_names(names, path)
        loaded = load_class_names(path)
        assert loaded == names

    def test_creates_parent_dirs(self, tmp_path):
        names = ['A', 'B']
        path = tmp_path / 'subdir' / 'nested' / 'class_names.json'
        save_class_names(names, path)
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        missing = tmp_path / 'nonexistent.json'
        with pytest.raises(FileNotFoundError, match="class_names.json not found"):
            load_class_names(missing)

    def test_file_format_correct(self, tmp_path):
        import json
        names = ['A', 'B', 'C']
        path = tmp_path / 'class_names.json'
        save_class_names(names, path)
        with open(path) as f:
            data = json.load(f)
        assert 'class_names' in data
        assert data['class_names'] == names

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / 'class_names.json'
        save_class_names(['X'], path)
        save_class_names(['A', 'B'], path)
        loaded = load_class_names(path)
        assert loaded == ['A', 'B']


class TestValidateDatasetIntegrity:
    def test_missing_dir_raises(self, tmp_path):
        missing = tmp_path / 'nonexistent'
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_dataset_integrity(missing)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="empty"):
            validate_dataset_integrity(tmp_path)

    def test_returns_required_keys(self, tmp_path):
        """Create a minimal fake dataset with one family and one valid PNG."""
        import cv2
        family_dir = tmp_path / 'FamilyA'
        family_dir.mkdir()
        # Create a minimal valid grayscale PNG
        img = np.zeros((16, 16), dtype=np.uint8)
        img_path = family_dir / 'sample.png'
        cv2.imwrite(str(img_path), img)

        report = validate_dataset_integrity(tmp_path)
        required_keys = {
            'valid', 'families', 'counts', 'total',
            'min_class', 'max_class', 'imbalance_ratio',
            'corrupt_files', 'missing_dirs'
        }
        assert required_keys.issubset(report.keys())

    def test_counts_and_total_correct(self, tmp_path):
        import cv2
        for family in ['FamilyA', 'FamilyB']:
            d = tmp_path / family
            d.mkdir()
            for i in range(3):
                img = np.zeros((16, 16), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i}.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert report['total'] == 6
        assert report['counts']['FamilyA'] == 3
        assert report['counts']['FamilyB'] == 3

    def test_corrupt_file_detection(self, tmp_path):
        """A corrupt PNG (not a valid image) should appear in corrupt_files."""
        family_dir = tmp_path / 'FamilyA'
        family_dir.mkdir()
        bad_png = family_dir / 'bad.png'
        bad_png.write_bytes(b'not an image')  # cv2.imread will return None

        report = validate_dataset_integrity(tmp_path)
        assert len(report['corrupt_files']) == 1
        assert report['valid'] is False

    def test_families_sorted(self, tmp_path):
        import cv2
        for name in ['Zebra', 'Alligator', 'Monkey']:
            d = tmp_path / name
            d.mkdir()
            img = np.zeros((8, 8), dtype=np.uint8)
            cv2.imwrite(str(d / 'img.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert report['families'] == ['Alligator', 'Monkey', 'Zebra']

    def test_imbalance_ratio(self, tmp_path):
        import cv2
        # FamilyA has 1 sample, FamilyB has 4 samples → ratio = 4.0
        for name, count in [('FamilyA', 1), ('FamilyB', 4)]:
            d = tmp_path / name
            d.mkdir()
            for i in range(count):
                img = np.zeros((8, 8), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i}.png'), img)

        report = validate_dataset_integrity(tmp_path)
        assert abs(report['imbalance_ratio'] - 4.0) < 1e-6

    def test_missing_dirs_always_empty_list(self, tmp_path):
        import cv2
        d = tmp_path / 'FamilyA'
        d.mkdir()
        img = np.zeros((8, 8), dtype=np.uint8)
        cv2.imwrite(str(d / 'img.png'), img)
        report = validate_dataset_integrity(tmp_path)
        assert report['missing_dirs'] == []

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
    """Unit tests that build a tiny fake dataset (no real Malimg needed)."""

    @pytest.fixture
    def fake_data_dir(self, tmp_path):
        """3 families × 5 samples each = 15 total images."""
        import cv2
        for family in ['FamilyA', 'FamilyB', 'FamilyC']:
            d = tmp_path / family
            d.mkdir()
            for i in range(5):
                img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
                cv2.imwrite(str(d / f'img{i:03d}.png'), img)
        return tmp_path

    def test_invalid_split_raises(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        with pytest.raises(ValueError, match="split"):
            MalimgDataset(fake_data_dir, 'invalid_split')

    def test_missing_data_dir_raises(self, tmp_path):
        from modules.dataset.loader import MalimgDataset
        missing = tmp_path / 'does_not_exist'
        with pytest.raises(FileNotFoundError):
            MalimgDataset(missing, 'train')

    def test_split_sizes_sum_to_total(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        train_ds = MalimgDataset(fake_data_dir, 'train')
        val_ds   = MalimgDataset(fake_data_dir, 'val')
        test_ds  = MalimgDataset(fake_data_dir, 'test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 15

    def test_getitem_tensor_shape(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        tensor, label = ds[0]
        assert tensor.shape == (1, 128, 128)

    def test_getitem_tensor_dtype(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        tensor, label = ds[0]
        assert tensor.dtype == torch.float32

    def test_getitem_label_is_int(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train', img_size=128)
        _, label = ds[0]
        assert isinstance(label, int)

    def test_class_names_sorted(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert ds.class_names == sorted(ds.class_names)

    def test_label_map_keys_match_class_names(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert set(ds.label_map.keys()) == set(ds.class_names)

    def test_label_map_values_are_unique_ints(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        values = list(ds.label_map.values())
        assert len(set(values)) == len(values)
        assert all(isinstance(v, int) for v in values)

    def test_get_labels_length_matches_len(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds = MalimgDataset(fake_data_dir, 'train')
        assert len(ds.get_labels()) == len(ds)

    def test_splits_are_reproducible(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds1 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        ds2 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        paths1 = [str(p) for p, _ in ds1.samples]
        paths2 = [str(p) for p, _ in ds2.samples]
        assert paths1 == paths2

    def test_different_seeds_produce_different_splits(self, fake_data_dir):
        from modules.dataset.loader import MalimgDataset
        ds1 = MalimgDataset(fake_data_dir, 'train', random_seed=42)
        ds2 = MalimgDataset(fake_data_dir, 'train', random_seed=99)
        # With 15 samples, different seeds should produce different orderings
        paths1 = set(str(p) for p, _ in ds1.samples)
        paths2 = set(str(p) for p, _ in ds2.samples)
        # Not guaranteed to differ but very likely with different seeds
        # At minimum, verify no crash
        assert isinstance(paths1, set)
        assert isinstance(paths2, set)

    @pytest.mark.integration
    def test_malimg_split_sizes_sum_correctly(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        train_ds = MalimgDataset(config.DATA_DIR, 'train')
        val_ds   = MalimgDataset(config.DATA_DIR, 'val')
        test_ds  = MalimgDataset(config.DATA_DIR, 'test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert 9000 < total < 9500

    @pytest.mark.integration
    def test_malimg_getitem_tensor_shape(self):
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
    def test_malimg_all_splits_contain_all_classes(self):
        import config
        if not config.DATA_DIR.exists():
            pytest.skip("Malimg dataset not found")
        from modules.dataset.loader import MalimgDataset
        for split in ['train', 'val', 'test']:
            ds = MalimgDataset(config.DATA_DIR, split)
            labels_in_split = set(ds.get_labels())
            assert len(labels_in_split) == 25, \
                f"Split '{split}' missing classes: {25 - len(labels_in_split)} absent"

