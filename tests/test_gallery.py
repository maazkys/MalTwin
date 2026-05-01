"""
Test suite for modules/dashboard/pages/gallery.py
and the new db.py functions: get_filtered_events, get_family_list.

All tests use tmp_path — no real Malimg dataset required.

Run:
    pytest tests/test_gallery.py -v
"""
import os
import cv2
import pytest
import numpy as np
from pathlib import Path

from modules.dashboard.db import (
    init_db,
    log_detection_event,
    get_filtered_events,
    get_family_list,
)
from modules.dashboard.pages.gallery import (
    _load_family_names,
    _load_sample_images,
    _count_family_images,
)


@pytest.fixture
def temp_db(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    return db


@pytest.fixture
def populated_db(temp_db):
    """DB with 10 events across 3 families."""
    families = ['Allaple.A', 'Rbot!gen', 'VB.AT']
    confs    = [0.95, 0.72, 0.45, 0.88, 0.61, 0.90, 0.55, 0.83, 0.40, 0.77]
    for i, conf in enumerate(confs):
        log_detection_event(
            temp_db,
            file_name=f"file_{i:02d}.exe",
            sha256='a' * 64,
            file_format='PE',
            file_size=1024,
            predicted_family=families[i % 3],
            confidence=conf,
            device_used='cpu',
        )
    return temp_db


@pytest.fixture
def fake_dataset(tmp_path):
    """Fake Malimg-style directory: 3 families × 5 PNGs each."""
    for family in ['Allaple.A', 'Rbot!gen', 'VB.AT']:
        d = tmp_path / family
        d.mkdir()
        for i in range(5):
            img = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
            cv2.imwrite(str(d / f'img_{i:03d}.png'), img)
    return tmp_path


class TestGetFilteredEvents:
    def test_returns_all_when_no_filters(self, populated_db):
        events = get_filtered_events(populated_db)
        assert len(events) == 10

    def test_returns_empty_for_missing_db(self, tmp_path):
        events = get_filtered_events(tmp_path / "missing.db")
        assert events == []

    def test_family_filter_works(self, populated_db):
        events = get_filtered_events(populated_db, family_filter='Allaple.A')
        assert all(e['predicted_family'] == 'Allaple.A' for e in events)
        assert len(events) > 0

    def test_family_filter_all_families_returns_everything(self, populated_db):
        events = get_filtered_events(populated_db, family_filter='All Families')
        assert len(events) == 10

    def test_confidence_filter_excludes_low_confidence(self, populated_db):
        events = get_filtered_events(populated_db, min_confidence=0.80)
        assert all(e['confidence'] >= 0.80 for e in events)

    def test_confidence_filter_zero_returns_all(self, populated_db):
        events = get_filtered_events(populated_db, min_confidence=0.0)
        assert len(events) == 10

    def test_limit_is_respected(self, populated_db):
        events = get_filtered_events(populated_db, limit=3)
        assert len(events) == 3

    def test_sort_desc_newest_first(self, populated_db):
        events = get_filtered_events(populated_db, sort_desc=True)
        ids = [e['id'] for e in events]
        assert ids == sorted(ids, reverse=True)

    def test_sort_asc_oldest_first(self, populated_db):
        events = get_filtered_events(populated_db, sort_desc=False)
        ids = [e['id'] for e in events]
        assert ids == sorted(ids)

    def test_days_back_filters_old_events(self, temp_db):
        """Events from now should appear; simulate via days_back=7."""
        log_detection_event(
            temp_db, 'recent.exe', 'b' * 64, 'PE',
            512, 'Allaple.A', 0.9, 'cpu',
        )
        events = get_filtered_events(temp_db, days_back=7)
        assert len(events) == 1
        assert events[0]['file_name'] == 'recent.exe'

    def test_combined_filters(self, populated_db):
        """Family + confidence filter should AND together."""
        events = get_filtered_events(
            populated_db,
            family_filter='Allaple.A',
            min_confidence=0.90,
        )
        for e in events:
            assert e['predicted_family'] == 'Allaple.A'
            assert e['confidence'] >= 0.90

    def test_returns_list_of_dicts(self, populated_db):
        events = get_filtered_events(populated_db)
        assert isinstance(events, list)
        assert isinstance(events[0], dict)

    def test_rows_contain_all_schema_columns(self, populated_db):
        events = get_filtered_events(populated_db, limit=1)
        required = {
            'id', 'timestamp', 'file_name', 'sha256',
            'file_format', 'file_size', 'predicted_family',
            'confidence', 'device_used',
        }
        assert required.issubset(events[0].keys())

    def test_no_sql_injection_via_family_filter(self, populated_db):
        """Malicious family_filter string must not crash or return wrong rows."""
        events = get_filtered_events(
            populated_db,
            family_filter="'; DROP TABLE detection_events; --",
        )
        assert isinstance(events, list)
        remaining = get_filtered_events(populated_db)
        assert len(remaining) == 10


class TestGetFamilyList:
    def test_returns_list_with_all_families_prepended(self, populated_db):
        families = get_family_list(populated_db)
        assert families[0] == 'All Families'

    def test_returns_only_all_families_for_empty_db(self, temp_db):
        families = get_family_list(temp_db)
        assert families == ['All Families']

    def test_returns_all_families_for_missing_db(self, tmp_path):
        families = get_family_list(tmp_path / 'missing.db')
        assert families == ['All Families']

    def test_families_are_sorted_alphabetically(self, populated_db):
        families = get_family_list(populated_db)[1:]
        assert families == sorted(families)

    def test_no_duplicates(self, populated_db):
        families = get_family_list(populated_db)
        assert len(families) == len(set(families))

    def test_detected_families_present(self, populated_db):
        families = get_family_list(populated_db)
        assert 'Allaple.A' in families
        assert 'Rbot!gen' in families
        assert 'VB.AT' in families

    def test_count_matches_distinct_families(self, populated_db):
        families = get_family_list(populated_db)
        assert len(families) == 4


class TestGalleryHelpers:
    def test_load_family_names_returns_sorted_list(self, fake_dataset):
        names = _load_family_names(str(fake_dataset))
        assert names == sorted(names)

    def test_load_family_names_all_three_present(self, fake_dataset):
        names = _load_family_names(str(fake_dataset))
        assert 'Allaple.A' in names
        assert 'Rbot!gen' in names
        assert 'VB.AT' in names

    def test_load_family_names_missing_dir_returns_empty(self, tmp_path):
        names = _load_family_names(str(tmp_path / 'nonexistent'))
        assert names == []

    def test_load_sample_images_returns_arrays(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=5)
        assert len(images) == 5
        assert all(isinstance(img, np.ndarray) for img in images)

    def test_load_sample_images_respects_max_images(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=2)
        assert len(images) == 2

    def test_load_sample_images_missing_family_returns_empty(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'NonexistentFamily', max_images=5)
        assert images == []

    def test_load_sample_images_arrays_are_2d(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=3)
        for img in images:
            assert img.ndim == 2

    def test_load_sample_images_dtype_uint8(self, fake_dataset):
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=3)
        for img in images:
            assert img.dtype == np.uint8

    def test_count_family_images_correct(self, fake_dataset):
        count = _count_family_images(str(fake_dataset), 'Allaple.A')
        assert count == 5

    def test_count_family_images_missing_family_returns_zero(self, fake_dataset):
        count = _count_family_images(str(fake_dataset), 'NoSuchFamily')
        assert count == 0

    def test_corrupt_image_skipped_gracefully(self, fake_dataset):
        """A corrupt PNG in the family dir should be skipped, not crash."""
        corrupt = fake_dataset / 'Allaple.A' / 'corrupt.png'
        corrupt.write_bytes(b'not a real image')
        images = _load_sample_images(str(fake_dataset), 'Allaple.A', max_images=10)
        assert len(images) == 5
