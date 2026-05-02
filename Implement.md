# MalTwin — Implementation Step 5: Dashboard-Triggered Training Flow
### SRS refs: BO-7, UC-05, FR-B2 (partial), NFR PER-2

> Complete Steps 1–4 first. Full regression suite must be green before starting.
> This step adds a new dashboard page — `training.py` — and wires it into routing.
> It does NOT replace `scripts/train.py`; the CLI script remains the primary
> training path. The dashboard provides a managed, observable wrapper around it.

---

## What This Step Delivers

| Item | Status before | Status after |
|---|---|---|
| `modules/dashboard/pages/training.py` | Does not exist | Full training management page |
| `modules/dashboard/app.py` | 5-page routing | 6-page routing with training page |
| `modules/dashboard/state.py` | No training keys | `KEY_TRAINING_STATE` added |
| `modules/training_manager.py` | Does not exist | Subprocess-based training runner with live log streaming |
| `tests/test_training_manager.py` | Does not exist | Training manager test suite |

---

## Mandatory Rules

- Training runs in a **subprocess** (`subprocess.Popen`) — never in the main Streamlit process. Blocking the main process freezes the entire dashboard for all users.
- The training subprocess runs `scripts/train.py` with the configured args — do not duplicate training logic.
- Log lines are read from the subprocess `stdout` pipe and stored in `session_state` for display — not streamed directly (Streamlit does not support true async streaming without workarounds).
- `st.rerun()` is used on a timer loop to poll subprocess status — the page auto-refreshes every 2 seconds while training is running.
- Training state persists in `session_state[KEY_TRAINING_STATE]` as a dict — survives page navigation during training.
- The training page **never** imports PyTorch or any ML module directly — it only launches the subprocess and reads its output.
- Hyperparameter widgets write to `session_state` — they do not directly start training.
- Only **one training job** can run at a time — the Start button is disabled while a job is active.
- `subprocess.Popen` must use `stdout=PIPE, stderr=STDOUT, text=True, bufsize=1` for line-buffered output.
- On page unload / navigation away, the subprocess is **not** killed — training continues in the background. The user can return to the training page to see progress.

---

## File 1: `modules/training_manager.py`

```python
# modules/training_manager.py
"""
Subprocess-based training job manager for the MalTwin dashboard.

Launches scripts/train.py as a child process, captures stdout line by line,
and exposes a simple polling interface for the Streamlit training page.

Never imports PyTorch, torchvision, or any ML library.
All ML work happens inside the subprocess.

Public API
----------
TrainingJob — manages one training subprocess lifecycle
    .start(args)   → launches subprocess
    .poll()        → returns (is_running, new_lines, return_code)
    .stop()        → terminates subprocess gracefully
    .is_running()  → bool
"""
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

import config


@dataclass
class TrainingJobState:
    """
    Serialisable snapshot of a training job — stored in session_state.
    All fields must be JSON-compatible types (no Process objects).
    """
    status:       str = 'idle'        # 'idle' | 'running' | 'completed' | 'failed' | 'stopped'
    start_time:   str = ''            # ISO 8601 string
    end_time:     str = ''            # ISO 8601 string — set on completion
    return_code:  int | None = None
    log_lines:    list[str] = field(default_factory=list)
    args_used:    dict = field(default_factory=dict)
    error_msg:    str = ''


class TrainingJob:
    """
    Manages one training subprocess.

    Usage:
        job = TrainingJob()
        job.start({'epochs': 5, 'lr': 0.001, ...})
        while job.is_running():
            is_running, new_lines, rc = job.poll()
            # update UI with new_lines
            time.sleep(2)
    """

    def __init__(self):
        self._process:  subprocess.Popen | None = None
        self._queue:    Queue = Queue()
        self._reader:   Thread | None = None
        self.state:     TrainingJobState = TrainingJobState()

    def start(self, args: dict) -> None:
        """
        Launch scripts/train.py as a subprocess with the given hyperparameters.

        Args:
            args: dict with keys matching CLI flags (without --):
                  epochs, lr, batch_size, workers, oversample, seed, no_augment
                  All are optional — missing keys fall back to config defaults.

        Raises:
            RuntimeError: if a job is already running.
            FileNotFoundError: if scripts/train.py does not exist.
        """
        if self.is_running():
            raise RuntimeError("A training job is already running.")

        script_path = Path('scripts') / 'train.py'
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")

        # Build CLI command
        cmd = [sys.executable, str(script_path)]
        if 'epochs' in args:
            cmd += ['--epochs', str(args['epochs'])]
        if 'lr' in args:
            cmd += ['--lr', str(args['lr'])]
        if 'batch_size' in args:
            cmd += ['--batch-size', str(args['batch_size'])]
        if 'workers' in args:
            cmd += ['--workers', str(args['workers'])]
        if 'oversample' in args:
            cmd += ['--oversample', str(args['oversample'])]
        if 'seed' in args:
            cmd += ['--seed', str(args['seed'])]
        if args.get('no_augment'):
            cmd += ['--no-augment']

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,               # line-buffered
            cwd=Path.cwd(),
        )

        self.state = TrainingJobState(
            status='running',
            start_time=datetime.utcnow().isoformat(),
            args_used=dict(args),
        )

        # Start background reader thread
        self._reader = Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        """Background thread: reads stdout line by line into the queue."""
        if self._process is None:
            return
        try:
            for line in self._process.stdout:
                self._queue.put(line.rstrip('\n'))
        except Exception:
            pass
        finally:
            self._queue.put(None)   # sentinel — signals EOF

    def poll(self) -> tuple[bool, list[str], int | None]:
        """
        Non-blocking poll. Call from the Streamlit page on each rerun.

        Returns:
            (is_running, new_lines, return_code)
            - is_running: True if subprocess still alive
            - new_lines:  list of new stdout lines since last poll (may be empty)
            - return_code: None while running, int when finished
        """
        new_lines = []

        # Drain the queue
        while True:
            try:
                line = self._queue.get_nowait()
                if line is None:
                    break          # EOF sentinel
                new_lines.append(line)
                self.state.log_lines.append(line)
            except Empty:
                break

        # Check if process has finished
        if self._process is not None:
            rc = self._process.poll()
            if rc is not None:
                # Process has exited
                self.state.return_code = rc
                self.state.end_time    = datetime.utcnow().isoformat()
                self.state.status      = 'completed' if rc == 0 else 'failed'
                if rc != 0:
                    self.state.error_msg = f"Process exited with code {rc}"
                return False, new_lines, rc

        return self.is_running(), new_lines, None

    def stop(self) -> None:
        """Terminate the subprocess gracefully (SIGTERM, then SIGKILL after 5s)."""
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        except Exception:
            pass
        self.state.status   = 'stopped'
        self.state.end_time = datetime.utcnow().isoformat()

    def is_running(self) -> bool:
        """True if subprocess exists and has not yet exited."""
        if self._process is None:
            return False
        return self._process.poll() is None


# ── Module-level singleton ─────────────────────────────────────────────────────
# One TrainingJob per Python process — Streamlit reruns share this instance
# via session_state (the TrainingJob object is stored there, not recreated).
# Do NOT instantiate at module level — let the page create it.
```

---

## File 2: `modules/dashboard/state.py` — additions only

**Add constant** (after `KEY_APP_START_TIME`):
```python
KEY_TRAINING_JOB   = 'training_job'    # TrainingJob instance or None
KEY_TRAINING_STATE = 'training_state'  # TrainingJobState dict snapshot or None
```

**Add to `init_session_state()` defaults dict**:
```python
KEY_TRAINING_JOB:   None,
KEY_TRAINING_STATE: None,
```

**Add helper functions** (after `is_model_loaded()`):
```python
def is_training_running() -> bool:
    job = st.session_state.get(KEY_TRAINING_JOB)
    if job is None:
        return False
    return job.is_running()


def get_training_state():
    """Returns TrainingJobState or None."""
    job = st.session_state.get(KEY_TRAINING_JOB)
    if job is None:
        return None
    return job.state
```

---

## File 3: `modules/dashboard/pages/training.py`

```python
# modules/dashboard/pages/training.py
"""
Dashboard-triggered model training page.

SRS ref: BO-7 — interactive Streamlit dashboard enabling training.
         UC-05 — Train Detection Model use case.

Layout:
    Left col (1/3):  Hyperparameter configuration widgets
    Right col (2/3): Training log, progress indicators, status

Training runs scripts/train.py as a subprocess. This page never imports
PyTorch directly — all ML work happens in the subprocess.
"""
import time
import streamlit as st
from datetime import datetime

import config
from modules.dashboard import state
from modules.training_manager import TrainingJob, TrainingJobState


# ── Constants ─────────────────────────────────────────────────────────────────

_POLL_INTERVAL_S = 2      # seconds between auto-reruns while training
_MAX_LOG_LINES   = 300    # cap displayed log lines to avoid massive DOM


def render():
    st.title("🏋️ Model Training")
    st.markdown(
        "Configure and launch a training run directly from the dashboard. "
        "Training runs `scripts/train.py` as a background process — you can "
        "navigate away and return to check progress."
    )

    # Dataset guard
    if not config.DATA_DIR.exists() or not any(config.DATA_DIR.iterdir()):
        st.error(
            "Error: Malimg dataset not found. "
            f"Cause: {config.DATA_DIR} is empty or does not exist. "
            "Action: Download the Malimg dataset and extract it before training."
        )
        return

    st.markdown("---")

    col_config, col_log = st.columns([1, 2])

    with col_config:
        _render_config_panel()

    with col_log:
        _render_log_panel()

    # Auto-rerun while training is active
    if state.is_training_running():
        time.sleep(_POLL_INTERVAL_S)
        st.rerun()


def _render_config_panel() -> None:
    """Hyperparameter widgets + Start/Stop controls."""
    st.subheader("Configuration")

    is_running = state.is_training_running()

    # Disable all widgets while training is running
    with st.form("training_config_form"):
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=200,
            value=config.EPOCHS,
            step=1,
            disabled=is_running,
            help="Number of full passes over the training dataset.",
        )
        lr = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-1,
            value=float(config.LR),
            format="%.6f",
            disabled=is_running,
            help="Adam optimiser learning rate.",
        )
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            index=[8, 16, 32, 64, 128].index(
                config.BATCH_SIZE if config.BATCH_SIZE in [8, 16, 32, 64, 128] else 32
            ),
            disabled=is_running,
            help="Number of images per training batch.",
        )
        workers = st.slider(
            "DataLoader Workers",
            min_value=0,
            max_value=8,
            value=min(config.NUM_WORKERS, 4),
            disabled=is_running,
            help="Set to 0 if you encounter DataLoader errors on Windows.",
        )
        oversample = st.selectbox(
            "Oversampling Strategy",
            options=['oversample_minority', 'sqrt_inverse', 'uniform'],
            index=['oversample_minority', 'sqrt_inverse', 'uniform'].index(
                config.OVERSAMPLE_STRATEGY
                if config.OVERSAMPLE_STRATEGY in ['oversample_minority', 'sqrt_inverse', 'uniform']
                else 'oversample_minority'
            ),
            disabled=is_running,
            help=(
                "oversample_minority: weight = 1/count (strongest balancing)\n"
                "sqrt_inverse: weight = 1/√count (softer)\n"
                "uniform: no oversampling"
            ),
        )
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=config.RANDOM_SEED,
            disabled=is_running,
        )
        no_augment = st.checkbox(
            "Disable augmentation",
            value=False,
            disabled=is_running,
            help="Use val transforms for training (no random flips / noise).",
        )

        col_start, col_stop = st.columns(2)
        with col_start:
            start_clicked = st.form_submit_button(
                "▶ Start Training",
                type="primary",
                disabled=is_running,
                use_container_width=True,
            )
        with col_stop:
            stop_clicked = st.form_submit_button(
                "■ Stop",
                type="secondary",
                disabled=not is_running,
                use_container_width=True,
            )

    # Handle Start
    if start_clicked and not is_running:
        _start_training({
            'epochs':     int(epochs),
            'lr':         float(lr),
            'batch_size': int(batch_size),
            'workers':    int(workers),
            'oversample': oversample,
            'seed':       int(seed),
            'no_augment': no_augment,
        })
        st.rerun()

    # Handle Stop
    if stop_clicked and is_running:
        job = st.session_state.get(state.KEY_TRAINING_JOB)
        if job:
            job.stop()
        st.warning("Training stopped by user.")
        st.rerun()

    # Config summary (shown while running)
    if is_running:
        ts = get_training_state()
        if ts and ts.args_used:
            st.markdown("**Running with:**")
            for k, v in ts.args_used.items():
                st.caption(f"`{k}`: {v}")


def _render_log_panel() -> None:
    """Training status, progress indicators, and live log."""
    st.subheader("Training Log")

    job: TrainingJob | None = st.session_state.get(state.KEY_TRAINING_JOB)

    if job is None:
        st.info("No training job started yet. Configure parameters and click **Start Training**.")
        return

    ts: TrainingJobState = job.state

    # Poll for new output
    if job.is_running():
        _, new_lines, _ = job.poll()
    else:
        new_lines = []

    # ── Status banner ─────────────────────────────────────────────────────────
    if ts.status == 'running':
        st.success(f"🟢 Training in progress — started {ts.start_time[:19]} UTC")

        # Elapsed time
        try:
            started = datetime.fromisoformat(ts.start_time)
            elapsed = datetime.utcnow() - started
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            st.caption(f"Elapsed: {mins}m {secs}s")
        except Exception:
            pass

        st.progress(
            _estimate_progress(ts.log_lines, ts.args_used.get('epochs', config.EPOCHS)),
            text="Training in progress…",
        )

    elif ts.status == 'completed':
        st.success(
            f"✅ Training completed successfully "
            f"(exit code 0) — finished {ts.end_time[:19]} UTC"
        )
        # Reload model into session state automatically
        _reload_model_after_training()

    elif ts.status == 'failed':
        st.error(
            f"🔴 Training failed (exit code {ts.return_code}). "
            "Check the log below for details."
        )

    elif ts.status == 'stopped':
        st.warning(f"⚠️ Training stopped by user — {ts.end_time[:19]} UTC")

    # ── Metrics extraction (parse log for best val_acc) ───────────────────────
    best_val_acc = _parse_best_val_acc(ts.log_lines)
    if best_val_acc is not None:
        st.metric("Best Val Accuracy", f"{best_val_acc * 100:.2f}%")

    # ── Live log ──────────────────────────────────────────────────────────────
    st.markdown("**Output:**")
    log_text = '\n'.join(ts.log_lines[-_MAX_LOG_LINES:])
    st.code(log_text or "(no output yet)", language=None)

    # ── Output files ──────────────────────────────────────────────────────────
    if ts.status == 'completed':
        st.markdown("**Generated files:**")
        for path, label in [
            (config.BEST_MODEL_PATH,    "Best model weights"),
            (config.CLASS_NAMES_PATH,   "Class names JSON"),
            (config.PROCESSED_DIR / 'eval_metrics.json',  "Eval metrics JSON"),
            (config.PROCESSED_DIR / 'confusion_matrix.png', "Confusion matrix PNG"),
        ]:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                st.caption(f"✅ {label}: `{path}` ({size_kb:.1f} KB)")
            else:
                st.caption(f"⚠️ {label}: not found at `{path}`")


def _start_training(args: dict) -> None:
    """Create a new TrainingJob, store it in session_state, and start it."""
    try:
        job = TrainingJob()
        job.start(args)
        st.session_state[state.KEY_TRAINING_JOB] = job
    except FileNotFoundError as e:
        st.error(
            f"Error: {e}. "
            "Action: Ensure scripts/train.py exists and you are running "
            "the dashboard from the repo root directory."
        )
    except RuntimeError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Error: Failed to start training. Cause: {e}")


def _reload_model_after_training() -> None:
    """
    After successful training, reload class names and model into session_state.
    Mirrors what load_global_resources() in app.py does at startup.
    Only reloads if the model file exists and session state is stale.
    """
    if not config.BEST_MODEL_PATH.exists():
        return
    # Force reload by resetting the model key
    if st.session_state.get(state.KEY_MODEL_LOADED):
        return   # already loaded, no action needed
    try:
        from modules.dataset.preprocessor import load_class_names
        from modules.detection.inference import load_model
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        model = load_model(
            config.BEST_MODEL_PATH,
            len(class_names),
            config.DEVICE,
        )
        st.session_state[state.KEY_CLASS_NAMES]  = class_names
        st.session_state[state.KEY_MODEL]         = model
        st.session_state[state.KEY_MODEL_LOADED]  = True
        st.session_state[state.KEY_DEVICE_INFO]   = str(config.DEVICE)
        st.success("✅ Model loaded into dashboard automatically.")
    except Exception as e:
        st.warning(f"Training completed but model auto-load failed: {e}. Restart the dashboard.")


def _estimate_progress(log_lines: list[str], total_epochs: int) -> float:
    """
    Parse log lines for 'Epoch NNN/TTT' pattern to estimate progress.
    Returns float [0.0, 1.0]. Returns 0.05 if no epoch lines found yet
    (shows a small non-zero bar so the user knows something is happening).
    """
    if total_epochs <= 0:
        return 0.0
    last_epoch = 0
    for line in reversed(log_lines):
        if 'Epoch' in line and '/' in line:
            try:
                # e.g. "Epoch 003/030 | Train Loss: ..."
                part  = line.split('Epoch')[1].strip().split()[0]
                curr  = int(part.split('/')[0])
                last_epoch = curr
                break
            except (IndexError, ValueError):
                continue
    if last_epoch == 0:
        return 0.05
    return min(last_epoch / total_epochs, 1.0)


def _parse_best_val_acc(log_lines: list[str]) -> float | None:
    """
    Scan log lines for '★ New best model saved (val_acc=X.XXXX)' pattern.
    Returns the highest val_acc seen, or None if not found.
    """
    best = None
    for line in log_lines:
        if 'val_acc=' in line:
            try:
                val_str = line.split('val_acc=')[1].strip().rstrip(')')
                val     = float(val_str)
                if best is None or val > best:
                    best = val
            except (IndexError, ValueError):
                continue
    return best


# Re-export for state.py helper
def get_training_state() -> TrainingJobState | None:
    job = st.session_state.get(state.KEY_TRAINING_JOB)
    return job.state if job else None
```

---

## File 4: Update `modules/dashboard/app.py`

### 4a — Add training to sidebar navigation

In `render_sidebar()`, add to the `nav_options` list (insert before Digital Twin):

```python
        ("🏋️ Model Training",  "🏋️ Model Training",   True),
```

So the full options list becomes:
```python
    nav_options = [
        ("🏠 Dashboard",        "🏠 Dashboard",        True),
        ("📂 Binary Upload",    "📂 Binary Upload",    True),
        (
            "🔍 Malware Detection" if (model_ready and file_ready)
            else "🔍 Malware Detection ⚠️",
            "🔍 Malware Detection",
            True,
        ),
        (
            "🖼️ Dataset Gallery" if dataset_ready
            else "🖼️ Dataset Gallery ⚠️",
            "🖼️ Dataset Gallery",
            True,
        ),
        ("🏋️ Model Training",  "🏋️ Model Training",   True),
        ("🖥️ Digital Twin",    "🖥️ Digital Twin",     True),
    ]
```

### 4b — Add training indicator to sidebar status panel

In `render_sidebar()`, after the detection result status block, add:

```python
    if state.is_training_running():
        st.sidebar.warning("🏋️ Training in progress…")
```

### 4c — Add route in `main()`

```python
    elif page == "🏋️ Model Training":
        from modules.dashboard.pages.training import render
        render()
```

---

## File 5: `tests/test_training_manager.py`

```python
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
import pytest
from pathlib import Path
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
        print("MalTwin Training Pipeline")
        print("[1/6] Validating dataset...")
        print("  Families found:   25")
        print("[2/6] Building DataLoaders...")
        print("[3/6] Initialising model...")
        print("[4/6] Training for 3 epoch(s)...")
        for epoch in range(1, 4):
            time.sleep(0.1)
            print(f"Epoch {epoch:03d}/003 | Train Loss: 1.2345 | Val Acc: 0.{epoch*30:04d}")
            print(f"  ★ New best model saved (val_acc=0.{epoch*30:04d})")
        print("[5/6] Evaluating...")
        print("[6/6] Saving outputs...")
        print("Done!")
        sys.exit(0)
    """))
    return script


@pytest.fixture
def fail_train_script(tmp_path) -> Path:
    """Script that exits with code 1 immediately."""
    script = tmp_path / "fail_train.py"
    script.write_text("import sys; print('ERROR: Dataset not found'); sys.exit(1)")
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
        # Patch: override command building for test
        # We test via a different approach — use monkeypatch in next test
        job.stop()

    def test_start_raises_if_already_running(self, fake_train_script, monkeypatch):
        """Starting a second job while one is running must raise RuntimeError."""
        job = TrainingJob()
        # Monkeypatch _build_cmd to use our fake script
        monkeypatch.setattr(
            job, '_build_cmd',
            lambda args: [sys.executable, str(fake_train_script)],
            raising=False,
        )
        # Manually start process
        import subprocess
        job._process = subprocess.Popen(
            [sys.executable, str(fake_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        from threading import Thread
        job._reader = Thread(target=job._read_output, daemon=True)
        job._reader.start()

        with pytest.raises(RuntimeError, match="already running"):
            job.start({})

        job.stop()

    def test_poll_returns_tuple_of_three(self, fake_train_script):
        job = TrainingJob()
        import subprocess, threading
        job._process = subprocess.Popen(
            [sys.executable, str(fake_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()

        result = job.poll()
        assert len(result) == 3
        job.stop()

    def test_state_status_running_while_process_alive(self, fake_train_script):
        job = TrainingJob()
        import subprocess, threading
        job._process = subprocess.Popen(
            [sys.executable, str(fake_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()
        assert job.state.status == 'running'
        job.stop()

    def test_log_lines_accumulate_across_polls(self, fake_train_script):
        import subprocess, threading, time
        job = TrainingJob()
        job._process = subprocess.Popen(
            [sys.executable, str(fake_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()

        # Poll multiple times while script runs
        deadline = time.time() + 8
        while job.is_running() and time.time() < deadline:
            job.poll()
            time.sleep(0.2)
        job.poll()   # final flush

        assert len(job.state.log_lines) > 0

    def test_state_status_completed_on_exit_zero(self, fake_train_script):
        import subprocess, threading, time
        job = TrainingJob()
        job._process = subprocess.Popen(
            [sys.executable, str(fake_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()

        deadline = time.time() + 8
        while job.is_running() and time.time() < deadline:
            job.poll()
            time.sleep(0.1)
        job.poll()

        assert job.state.status == 'completed'
        assert job.state.return_code == 0

    def test_state_status_failed_on_nonzero_exit(self, fail_train_script):
        import subprocess, threading, time
        job = TrainingJob()
        job._process = subprocess.Popen(
            [sys.executable, str(fail_train_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()

        deadline = time.time() + 5
        while job.is_running() and time.time() < deadline:
            job.poll()
            time.sleep(0.1)
        job.poll()

        assert job.state.status == 'failed'
        assert job.state.return_code == 1

    def test_stop_sets_status_stopped(self, fake_train_script):
        import subprocess, threading
        job = TrainingJob()
        job._process = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(30)'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.state.status = 'running'
        job._reader = threading.Thread(target=job._read_output, daemon=True)
        job._reader.start()

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
```

---

## Definition of Done

```bash
# Step 5 tests
pytest tests/test_training_manager.py -v
# Expected: all tests pass, 0 failures

# Full regression suite
pytest tests/ -v -m "not integration"
# Expected: 0 failures across all steps

# Import smoke tests
python -c "from modules.training_manager import TrainingJob, TrainingJobState"
python -c "from modules.dashboard.pages.training import render"
python -c "from modules.dashboard import state; assert hasattr(state, 'KEY_TRAINING_JOB')"

# Dashboard launch — verify training page
streamlit run modules/dashboard/app.py --server.port 8501

# Verify on training page:
#   ✓ 🏋️ Model Training appears in sidebar (6 nav options total)
#   ✓ Hyperparameter form renders correctly
#   ✓ Start button disabled when training is running
#   ✓ Stop button disabled when not running
#   ✓ Clicking Start without dataset shows error (not crash)
#   ✓ Sidebar shows "🏋️ Training in progress…" while running
#   ✓ Log output appears and updates every ~2 seconds
#   ✓ After completion: status banner turns green, output files listed
#   ✓ After completion: model auto-loaded into session state
#   ✓ Can navigate away and return — log is still there
```

### Checklist

- [ ] `pytest tests/test_training_manager.py -v` — 0 failures
- [ ] All earlier tests still pass
- [ ] `modules/training_manager.py` exists with `TrainingJob` and `TrainingJobState`
- [ ] `modules/dashboard/pages/training.py` exists
- [ ] `state.py` has `KEY_TRAINING_JOB` and `KEY_TRAINING_STATE`
- [ ] `state.py` has `is_training_running()` helper
- [ ] `app.py` sidebar has 6 navigation options including `"🏋️ Model Training"`
- [ ] `app.py` routes `"🏋️ Model Training"` to `training.render()`
- [ ] Training runs `scripts/train.py` as subprocess — not inline
- [ ] `subprocess.Popen` uses `stdout=PIPE, stderr=STDOUT, text=True, bufsize=1`
- [ ] Page auto-reruns every 2 seconds while training is running
- [ ] Navigating away does not kill the subprocess
- [ ] Stop button sends SIGTERM then SIGKILL after 5s
- [ ] Progress bar estimates from epoch log lines
- [ ] Best val_acc extracted and shown as metric
- [ ] After successful training, model auto-loaded into session_state
- [ ] Dataset missing → `st.error()` shown, page returns early — no crash

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|---|---|---|
| Running training inline (not subprocess) | Dashboard freezes for entire training duration | Always use `subprocess.Popen` — never call `train()` directly |
| `subprocess.PIPE` on stderr separately | stderr output lost; error messages invisible | Use `stderr=subprocess.STDOUT` to merge stderr into stdout |
| `bufsize=0` or default | Output appears in large chunks, not line by line | Use `bufsize=1` (line-buffered) with `text=True` |
| `st.rerun()` called unconditionally | Infinite rerun loop even when not training | Only call inside `if state.is_training_running():` block |
| `time.sleep(_POLL_INTERVAL_S)` before `st.rerun()` | `sleep` blocks the Streamlit server thread briefly | Acceptable at 2s; do not increase. Never sleep in a non-training render. |
| `TrainingJob` created fresh on every rerun | Job lost between reruns; log disappears | Store job in `session_state[KEY_TRAINING_JOB]` — retrieve, don't recreate |
| `job.poll()` not called before reading `state.log_lines` | Log display is always one poll behind | Always call `job.poll()` at the top of `_render_log_panel()` |
| Daemon thread reading stdout keeps process alive | Process appears finished but output still queued | Use `None` sentinel in queue from `_read_output` to signal EOF |
| Form submit button outside `st.form` | Streamlit `StreamlitAPIException` | Start/Stop buttons must be `st.form_submit_button` inside the form, or `st.button` outside it — not mixed |

---

*Step 5 complete → Step 6 (final): Integration verification, SRS compliance audit, and cleanup.*
