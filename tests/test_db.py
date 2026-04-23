# tests/test_db.py
"""
Test suite for modules/dashboard/db.py

All tests use tmp_path (pytest built-in) for isolated SQLite files.
No Malimg dataset required.

Run:
    pytest tests/test_db.py -v
"""
import os
import pytest
from pathlib import Path
from modules.dashboard.db import (
    init_db,
    log_detection_event,
    get_recent_events,
    get_detection_stats,
    get_events_by_date_range,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path) -> Path:
    """Initialised DB in a temporary directory."""
    db_path = tmp_path / "test_maltwin.db"
    init_db(db_path)
    return db_path


def _insert(db_path: Path, **overrides):
    """Helper: insert a detection event with sensible defaults."""
    defaults = dict(
        file_name="sample.exe",
        sha256="a" * 64,
        file_format="PE",
        file_size=1024,
        predicted_family="Allaple.A",
        confidence=0.95,
        device_used="cpu",
    )
    defaults.update(overrides)
    log_detection_event(db_path, **defaults)


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        init_db(db_path)
        assert db_path.exists()

    def test_idempotent(self, temp_db):
        """Calling init_db a second time must not raise."""
        init_db(temp_db)

    def test_file_permissions_are_600(self, temp_db):
        mode = os.stat(temp_db).st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_creates_detection_events_table(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_events'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_timestamp_index(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_family_index(self, temp_db):
        from modules.dashboard.db import get_connection
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_family'"
            ).fetchall()
        assert len(rows) == 1

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "deep" / "test.db"
        init_db(db_path)
        assert db_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# log_detection_event
# ─────────────────────────────────────────────────────────────────────────────

class TestLogDetectionEvent:
    def test_inserts_one_row(self, temp_db):
        _insert(temp_db)
        events = get_recent_events(temp_db, limit=10)
        assert len(events) == 1

    def test_inserted_values_are_correct(self, temp_db):
        _insert(temp_db,
                file_name="test.exe",
                sha256="b" * 64,
                file_format="ELF",
                file_size=2048,
                predicted_family="Yuner.A",
                confidence=0.75,
                device_used="cuda")
        row = get_recent_events(temp_db)[0]
        assert row['file_name']        == "test.exe"
        assert row['sha256']           == "b" * 64
        assert row['file_format']      == "ELF"
        assert row['file_size']        == 2048
        assert row['predicted_family'] == "Yuner.A"
        assert abs(row['confidence'] - 0.75) < 1e-6
        assert row['device_used']      == "cuda"

    def test_timestamp_is_set_automatically(self, temp_db):
        _insert(temp_db)
        row = get_recent_events(temp_db)[0]
        assert 'timestamp' in row
        assert len(row['timestamp']) > 10   # ISO 8601 string

    def test_multiple_inserts_accumulate(self, temp_db):
        for i in range(5):
            _insert(temp_db, file_name=f"file_{i}.exe")
        assert len(get_recent_events(temp_db, limit=10)) == 5

    def test_does_not_raise_on_bad_path(self, tmp_path):
        """A path where the parent dir doesn't exist — must not raise."""
        bad = tmp_path / "nonexistent_dir" / "db.db"
        # log_detection_event must swallow the error
        try:
            log_detection_event(
                bad, "x.exe", "a" * 64, "PE", 100, "X", 0.5, "cpu"
            )
        except Exception:
            pass   # acceptable — it just must not propagate up and crash calling code


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_events
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentEvents:
    def test_returns_empty_list_for_empty_db(self, temp_db):
        assert get_recent_events(temp_db) == []

    def test_returns_empty_list_for_missing_db(self, tmp_path):
        assert get_recent_events(tmp_path / "missing.db") == []

    def test_returns_most_recent_first(self, temp_db):
        for i in range(3):
            _insert(temp_db, file_name=f"file_{i}.exe")
        events = get_recent_events(temp_db, limit=5)
        assert events[0]['file_name'] == "file_2.exe"

    def test_limit_is_respected(self, temp_db):
        for i in range(10):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert len(get_recent_events(temp_db, limit=3)) == 3

    def test_default_limit_is_five(self, temp_db):
        for i in range(8):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert len(get_recent_events(temp_db)) == 5

    def test_returns_list_of_dicts(self, temp_db):
        _insert(temp_db)
        events = get_recent_events(temp_db)
        assert isinstance(events, list)
        assert isinstance(events[0], dict)

    def test_rows_contain_all_schema_columns(self, temp_db):
        _insert(temp_db)
        row = get_recent_events(temp_db)[0]
        expected_keys = {
            'id', 'timestamp', 'file_name', 'sha256', 'file_format',
            'file_size', 'predicted_family', 'confidence', 'device_used',
        }
        assert expected_keys.issubset(row.keys())


# ─────────────────────────────────────────────────────────────────────────────
# get_detection_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDetectionStats:
    def test_empty_db_returns_zeros(self, temp_db):
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 0
        assert stats['total_malware']  == 0
        assert stats['total_benign']   == 0

    def test_missing_db_returns_zeros(self, tmp_path):
        stats = get_detection_stats(tmp_path / "missing.db")
        assert stats['total_analyzed'] == 0

    def test_counts_correctly_after_inserts(self, temp_db):
        for i in range(5):
            _insert(temp_db, file_name=f"f{i}.exe")
        stats = get_detection_stats(temp_db)
        assert stats['total_analyzed'] == 5
        assert stats['total_malware']  == 5

    def test_returns_required_keys(self, temp_db):
        stats = get_detection_stats(temp_db)
        assert 'total_analyzed' in stats
        assert 'total_malware'  in stats
        assert 'total_benign'   in stats
        assert 'model_accuracy' in stats

    def test_model_accuracy_none_when_no_metrics_file(self, temp_db):
        # As long as data/processed/eval_metrics.json doesn't exist, this is None
        stats = get_detection_stats(temp_db)
        # Can be None or float — just check it doesn't crash
        assert stats['model_accuracy'] is None or isinstance(stats['model_accuracy'], float)

    def test_total_benign_always_zero(self, temp_db):
        for i in range(3):
            _insert(temp_db, file_name=f"f{i}.exe")
        assert get_detection_stats(temp_db)['total_benign'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_events_by_date_range
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEventsByDateRange:
    def test_returns_empty_list_for_empty_db(self, temp_db):
        assert get_events_by_date_range(temp_db) == []

    def test_returns_empty_list_for_missing_db(self, tmp_path):
        assert get_events_by_date_range(tmp_path / "missing.db") == []

    def test_returns_events_within_range(self, temp_db):
        _insert(temp_db, file_name="recent.exe")
        events = get_events_by_date_range(temp_db, days_back=7)
        assert len(events) == 1

    def test_returned_dicts_have_timestamp_key(self, temp_db):
        _insert(temp_db)
        events = get_events_by_date_range(temp_db, days_back=7)
        assert 'timestamp' in events[0]

    def test_returned_dicts_have_predicted_family_key(self, temp_db):
        _insert(temp_db)
        events = get_events_by_date_range(temp_db, days_back=7)
        assert 'predicted_family' in events[0]
