"""
test_db.py — matches actual MalTwin db.py signatures.

Key facts from source:
- Table name:  detection_events
- init_db(db_path: Path)
- log_detection_event(db_path, file_name, sha256, file_format,
                      file_size, predicted_family, confidence, device_used)
- get_recent_events(db_path, limit=5) -> list[dict]
- get_detection_stats(db_path) -> dict
- get_events_by_date_range(db_path, days_back=7) -> list[dict]
"""

import sqlite3
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def db_mod():
    from modules.dashboard import db
    return db


@pytest.fixture()
def tmp_db(tmp_path, db_mod):
    db_path = tmp_path / "test_maltwin.db"
    db_mod.init_db(db_path)
    yield db_path


def _log(db_mod, db_path, **overrides):
    defaults = dict(
        file_name="sample.exe",
        sha256="a" * 64,
        file_format="PE",
        file_size=48640,
        predicted_family="Allaple.A",
        confidence=0.97,
        device_used="cpu",
    )
    defaults.update(overrides)
    db_mod.log_detection_event(db_path, **defaults)


# ===========================================================================
# init_db
# ===========================================================================

class TestInitDb:

    def test_creates_file(self, db_mod, tmp_path):
        db_path = tmp_path / "init_test.db"
        db_mod.init_db(db_path)
        assert db_path.exists()

    def test_creates_detection_events_table(self, db_mod, tmp_path):
        db_path = tmp_path / "table_test.db"
        db_mod.init_db(db_path)
        con = sqlite3.connect(db_path)
        tables = [t[0] for t in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        con.close()
        assert "detection_events" in tables

    def test_wal_mode_enabled(self, db_mod, tmp_path):
        db_path = tmp_path / "wal_test.db"
        db_mod.init_db(db_path)
        con = sqlite3.connect(db_path)
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
        assert mode.lower() == "wal"

    def test_idempotent_double_init(self, db_mod, tmp_path):
        db_path = tmp_path / "double_init.db"
        db_mod.init_db(db_path)
        db_mod.init_db(db_path)  # must not raise

    def test_schema_has_required_columns(self, db_mod, tmp_path):
        db_path = tmp_path / "schema_test.db"
        db_mod.init_db(db_path)
        con = sqlite3.connect(db_path)
        cols = [row[1].lower() for row in con.execute(
            "PRAGMA table_info(detection_events)"
        ).fetchall()]
        con.close()
        for col in ("sha256", "confidence", "predicted_family", "file_name"):
            assert col in cols, f"Column '{col}' missing. Found: {cols}"


# ===========================================================================
# log_detection_event
# ===========================================================================

class TestLogDetectionEvent:

    def test_inserts_one_row(self, db_mod, tmp_db):
        _log(db_mod, tmp_db)
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        con.close()
        assert count == 1

    def test_inserts_multiple_rows(self, db_mod, tmp_db):
        for i in range(5):
            _log(db_mod, tmp_db, sha256="b" * 63 + str(i))
        con = sqlite3.connect(tmp_db)
        count = con.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        con.close()
        assert count == 5

    def test_stored_sha256_matches(self, db_mod, tmp_db):
        sha = "c" * 64
        _log(db_mod, tmp_db, sha256=sha)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT sha256 FROM detection_events").fetchone()
        con.close()
        assert row[0] == sha

    def test_stored_confidence_matches(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, confidence=0.831)
        con = sqlite3.connect(tmp_db)
        row = con.execute("SELECT confidence FROM detection_events").fetchone()
        con.close()
        assert abs(row[0] - 0.831) < 1e-4

    def test_stored_family_matches(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, predicted_family="Yuner.A")
        con = sqlite3.connect(tmp_db)
        row = con.execute(
            "SELECT predicted_family FROM detection_events"
        ).fetchone()
        con.close()
        assert row[0] == "Yuner.A"

    def test_never_raises_on_bad_confidence(self, db_mod, tmp_db):
        """log_detection_event must not raise — it retries silently."""
        try:
            _log(db_mod, tmp_db, confidence=9999.0)
        except Exception as e:
            pytest.fail(f"log_detection_event raised unexpectedly: {e}")


# ===========================================================================
# get_recent_events
# ===========================================================================

class TestGetRecentEvents:

    def test_empty_db_returns_empty_list(self, db_mod, tmp_db):
        results = db_mod.get_recent_events(tmp_db)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_returns_inserted_rows(self, db_mod, tmp_db):
        _log(db_mod, tmp_db)
        results = db_mod.get_recent_events(tmp_db)
        assert len(results) == 1

    def test_limit_parameter(self, db_mod, tmp_db):
        for i in range(10):
            _log(db_mod, tmp_db, sha256="d" * 63 + str(i))
        results = db_mod.get_recent_events(tmp_db, limit=5)
        assert len(results) <= 5

    def test_returns_list_of_dicts(self, db_mod, tmp_db):
        _log(db_mod, tmp_db)
        results = db_mod.get_recent_events(tmp_db)
        assert isinstance(results[0], dict)

    def test_dict_has_sha256_key(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, sha256="e" * 64)
        results = db_mod.get_recent_events(tmp_db)
        assert "sha256" in results[0]

    def test_same_sha256_logged_twice_gives_two_rows(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, sha256="f" * 64)
        _log(db_mod, tmp_db, sha256="f" * 64)
        results = db_mod.get_recent_events(tmp_db, limit=100)
        assert len(results) == 2

    def test_missing_db_returns_empty_list(self, db_mod, tmp_path):
        missing = tmp_path / "no_db.db"
        results = db_mod.get_recent_events(missing)
        assert results == []


# ===========================================================================
# get_detection_stats
# ===========================================================================

class TestGetDetectionStats:

    def test_returns_dict(self, db_mod, tmp_db):
        result = db_mod.get_detection_stats(tmp_db)
        assert isinstance(result, dict)

    def test_has_required_keys(self, db_mod, tmp_db):
        result = db_mod.get_detection_stats(tmp_db)
        for key in ("total_analyzed", "total_malware", "total_benign"):
            assert key in result, f"Missing key: {key}"

    def test_total_analyzed_increments(self, db_mod, tmp_db):
        before = db_mod.get_detection_stats(tmp_db)["total_analyzed"]
        _log(db_mod, tmp_db)
        after = db_mod.get_detection_stats(tmp_db)["total_analyzed"]
        assert after == before + 1

    def test_missing_db_returns_zeros(self, db_mod, tmp_path):
        missing = tmp_path / "no_stats.db"
        result = db_mod.get_detection_stats(missing)
        assert result["total_analyzed"] == 0


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_confidence_zero(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, confidence=0.0)
        assert len(db_mod.get_recent_events(tmp_db)) == 1

    def test_confidence_one(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, confidence=1.0)
        assert len(db_mod.get_recent_events(tmp_db)) == 1

    def test_unicode_filename(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, file_name="مالویر.exe", sha256="g" * 64)
        results = db_mod.get_recent_events(tmp_db)
        assert any("exe" in r.get("file_name", "") for r in results)

    def test_elf_format_stored(self, db_mod, tmp_db):
        _log(db_mod, tmp_db, file_format="ELF", sha256="h" * 64)
        results = db_mod.get_recent_events(tmp_db)
        assert any(r.get("file_format") == "ELF" for r in results)


# ===========================================================================
# log_report_event
# ===========================================================================

class TestLogReportEvent:
    def test_does_not_raise(self, db_mod, tmp_db):
        from modules.dashboard.db import log_report_event
        log_report_event(tmp_db, None, 'a' * 64, 'PDF', False)

    def test_creates_report_events_table(self, db_mod, tmp_db):
        from modules.dashboard.db import log_report_event, get_connection
        log_report_event(tmp_db, None, 'a' * 64, 'JSON', True)
        with get_connection(tmp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='report_events'"
            ).fetchall()
        assert len(rows) == 1

    def test_inserted_values_correct(self, db_mod, tmp_db):
        from modules.dashboard.db import log_report_event, get_connection
        log_report_event(tmp_db, 42, 'b' * 64, 'PDF', True)
        with get_connection(tmp_db) as conn:
            row = conn.execute("SELECT * FROM report_events").fetchone()
        assert dict(row)['sha256'] == 'b' * 64
        assert dict(row)['report_format'] == 'PDF'
        assert dict(row)['gradcam_included'] == 1

    def test_does_not_raise_on_missing_db(self, db_mod, tmp_path):
        from modules.dashboard.db import log_report_event
        bad_path = tmp_path / "nonexistent_dir" / "db.db"
        log_report_event(bad_path, None, 'a' * 64, 'JSON', False)