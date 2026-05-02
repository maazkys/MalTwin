"""
Test suite for modules/training_manager.py

Tests verify the TrainingJob lifecycle using a real subprocess running
a simple Python script (not scripts/train.py — no dataset needed).

Run:
    pytest tests/test_training_manager.py -v
"""
import sys
import time
import textwrap
from pathlib import Path

import pytest

from modules.training_manager import TrainingJob, TrainingJobState


# ── Helper: create a tiny fake training script ────────────────────────────────

@pytest.fixture
def fake_train_script(tmp_path) -> Path:
    """
    Creates a minimal Python script that mimics train.py output format,
    runs for ~0.5 seconds, then exits 0.
    """
    script = tmp_path / "fake_train.py"
    script.write_text(textwrap.dedent("""
        import time, sys
        print("MalTwin Training Pipeline", flush=True)
        print("[1/6] Validating dataset...", flush=True)
        print("  Families found:   25", flush=True)
        print("[2/6] Building DataLoaders...", flush=True)
        print("[3/6] Initialising model...", flush=True)
        print("[4/6] Training for 3 epochs...", flush=True)
        for epoch in range(1, 4):
            time.sleep(0.1)
            print(f"Epoch {epoch:03d}/003 | Train Loss: 1.2345 | Val Acc: 0.{epoch*30:04d}", flush=True)
            print(f"  ★ New best model saved (val_acc=0.{epoch*30:04d})", flush=True)
        print("[5/6] Evaluating...", flush=True)
        print("[6/6] Saving outputs...", flush=True)
        print("Done!", flush=True)
        sys.exit(0)
    """))
    return script


@pytest.fixture
def fail_train_script(tmp_path) -> Path:
    """Script that exits with code 1 immediately."""
    script = tmp_path / "fail_train.py"
    script.write_text("import sys; print('ERROR: Dataset not found', flush=True); sys.exit(1)")
    return script


# ─────────────────────────────────────────────────────────────────────────────
# TrainingJobState dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingJobState:
    def test_default_status_is_idle(self):
        s = TrainingJobState()
        assert s.status == 'idle'

    def test_default_log_lines_is_empty_list(self):
        s = TrainingJobState()
        assert s.log_lines == []

    def test_default_return_code_is_none(self):
        s = TrainingJobState()
        assert s.return_code is None

    def test_default_args_used_is_empty_dict(self):
        s = TrainingJobState()
        assert s.args_used == {}


# ─────────────────────────────────────────────────────────────────────────────
# TrainingJob lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingJobLifecycle:
    def _run_to_completion(self, job: TrainingJob, timeout: float = 10.0) -> None:
        """Poll until job finishes or timeout expires."""
        start = time.time()
        while job.is_running():
            job.poll()
            if time.time() - start > timeout:
                job.stop()
                pytest.fail("Job did not complete within timeout")
            time.sleep(0.1)
        # Final poll to flush remaining lines
        job.poll()

    def test_is_not_running_before_start(self):
        job = TrainingJob()
        assert job.is_running() is False

    def test_is_running_after_start(self, fake_train_script):
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        assert job.is_running() is True
        job.stop()

    def test_start_raises_if_already_running(self, fake_train_script):
        """Starting a second job while one is running must raise RuntimeError."""
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        with pytest.raises(RuntimeError, match="already running"):
            job.start({'_script': str(fake_train_script)})
        job.stop()

    def test_poll_returns_tuple_of_three(self, fake_train_script):
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        result = job.poll()
        assert len(result) == 3
        job.stop()

    def test_state_status_running_while_process_alive(self, fake_train_script):
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        assert job.state.status == 'running'
        job.stop()

    def test_log_lines_accumulate_across_polls(self, fake_train_script):
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        self._run_to_completion(job)
        assert len(job.state.log_lines) > 0

    def test_state_status_completed_on_exit_zero(self, fake_train_script):
        job = TrainingJob()
        job.start({'_script': str(fake_train_script)})
        self._run_to_completion(job)
        assert job.state.status == 'completed'
        assert job.state.return_code == 0

    def test_state_status_failed_on_nonzero_exit(self, fail_train_script):
        job = TrainingJob()
        job.start({'_script': str(fail_train_script)})
        self._run_to_completion(job)
        assert job.state.status == 'failed'
        assert job.state.return_code == 1

    def test_stop_sets_status_stopped(self, tmp_path):
        script = tmp_path / "sleep.py"
        script.write_text("import time; time.sleep(30)")
        job = TrainingJob()
        job.start({'_script': str(script)})
        job.stop()
        assert job.state.status == 'stopped'
        assert job.is_running() is False

    def test_stop_on_non_running_job_does_not_raise(self):
        job = TrainingJob()
        job.stop()   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    def test_estimate_progress_zero_when_no_epoch_lines(self):
        from modules.dashboard.pages.training import _estimate_progress
        assert _estimate_progress([], 30) == pytest.approx(0.05)

    def test_estimate_progress_parses_epoch_line(self):
        from modules.dashboard.pages.training import _estimate_progress
        lines = ["Epoch 015/030 | Train Loss: 0.123 | Val Acc: 0.8500"]
        assert _estimate_progress(lines, 30) == pytest.approx(0.5)

    def test_estimate_progress_capped_at_one(self):
        from modules.dashboard.pages.training import _estimate_progress
        lines = ["Epoch 030/030 | Train Loss: 0.050 | Val Acc: 0.9700"]
        assert _estimate_progress(lines, 30) == pytest.approx(1.0)

    def test_estimate_progress_handles_malformed_lines(self):
        from modules.dashboard.pages.training import _estimate_progress
        lines = ["Epoch GARBAGE | blah blah", "no epoch here"]
        result = _estimate_progress(lines, 30)
        assert 0.0 <= result <= 1.0

    def test_parse_best_val_acc_returns_none_when_no_lines(self):
        from modules.dashboard.pages.training import _parse_best_val_acc
        assert _parse_best_val_acc([]) is None

    def test_parse_best_val_acc_finds_val_acc(self):
        from modules.dashboard.pages.training import _parse_best_val_acc
        lines = [
            "Epoch 001/030 | Val Acc: 0.5500",
            "  ★ New best model saved (val_acc=0.5500)",
            "Epoch 002/030 | Val Acc: 0.7200",
            "  ★ New best model saved (val_acc=0.7200)",
        ]
        result = _parse_best_val_acc(lines)
        assert result == pytest.approx(0.72)

    def test_parse_best_val_acc_returns_highest(self):
        from modules.dashboard.pages.training import _parse_best_val_acc
        lines = [
            "★ New best model saved (val_acc=0.4500)",
            "★ New best model saved (val_acc=0.8900)",
            "★ New best model saved (val_acc=0.7200)",
        ]
        result = _parse_best_val_acc(lines)
        assert result == pytest.approx(0.89)

    def test_parse_best_val_acc_handles_malformed(self):
        from modules.dashboard.pages.training import _parse_best_val_acc
        lines = ["val_acc=NOTANUMBER", "val_acc="]
        result = _parse_best_val_acc(lines)
        assert result is None
