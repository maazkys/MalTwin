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
from dataclasses import asdict
from datetime import datetime, timezone

import streamlit as st

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
            _update_training_snapshot(job)
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
    job.poll()
    _update_training_snapshot(job)

    # ── Status banner ─────────────────────────────────────────────────────────
    if ts.status == 'running':
        st.success(f"🟢 Training in progress — started {ts.start_time[:19]} UTC")

        # Elapsed time
        try:
            started = datetime.fromisoformat(ts.start_time)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed = datetime.now(timezone.utc) - started
            mins, secs = divmod(int(elapsed.total_seconds()), 60)
            st.caption(f"Elapsed: {mins}m {secs}s")
        except (ValueError, TypeError):
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
        _update_training_snapshot(job)
    except FileNotFoundError as e:
        st.error(
            f"Error: {e}. "
            "Action: Ensure scripts/train.py exists and you are running "
            "the dashboard from the repo root directory."
        )
    except RuntimeError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(
            "Error: Failed to start training. "
            f"Cause: {e}. "
            "Action: Verify hyperparameters, dependencies, and file permissions, then try again."
        )


def _reload_model_after_training() -> None:
    """
    After successful training, reload class names and model into session_state.
    Mirrors what load_global_resources() in app.py does at startup.
    Only reloads if the model file exists and session state is stale.
    """
    if not config.BEST_MODEL_PATH.exists():
        return
    job = st.session_state.get(state.KEY_TRAINING_JOB)
    if job and job.state.model_reloaded:
        return
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
        _mark_model_reloaded()
        st.success("✅ Model loaded into dashboard automatically.")
    except Exception as e:
        st.warning(
            f"Training completed but model auto-load failed: {e}. "
            "You may need to restart the dashboard or manually reload from the Home page."
        )


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
                part = line.split('Epoch')[1].strip().split()[0]
                curr = int(part.split('/')[0])
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
                val = float(val_str)
                if best is None or val > best:
                    best = val
            except (IndexError, ValueError):
                continue
    return best


def _update_training_snapshot(job: TrainingJob) -> None:
    st.session_state[state.KEY_TRAINING_STATE] = asdict(job.state)


def _mark_model_reloaded() -> None:
    job = st.session_state.get(state.KEY_TRAINING_JOB)
    if job:
        job.state.model_reloaded = True
        st.session_state[state.KEY_TRAINING_STATE] = asdict(job.state)


# Re-export for state.py helper
def get_training_state() -> TrainingJobState | None:
    job = st.session_state.get(state.KEY_TRAINING_JOB)
    return job.state if job else None
