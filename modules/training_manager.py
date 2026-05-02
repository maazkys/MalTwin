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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread

import config


@dataclass
class TrainingJobState:
    """
    Serialisable snapshot of a training job — stored in session_state.
    All fields must be JSON-compatible types (no Process objects).
    """
    status:      str = 'idle'        # 'idle' | 'running' | 'completed' | 'failed' | 'stopped'
    start_time:  str = ''            # ISO 8601 string
    end_time:    str = ''            # ISO 8601 string — set on completion
    return_code: int | None = None
    log_lines:   list[str] = field(default_factory=list)
    args_used:   dict = field(default_factory=dict)
    error_msg:   str = ''


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
        self._process: subprocess.Popen | None = None
        self._queue: Queue = Queue()
        self._reader: Thread | None = None
        self.state: TrainingJobState = TrainingJobState()

    def _build_cmd(self, args: dict) -> list[str]:
        script_override = args.get('_script')
        script_path = Path(script_override) if script_override else Path('scripts') / 'train.py'
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

        return cmd

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

        cmd = self._build_cmd(args)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,               # line-buffered
            cwd=Path.cwd(),
        )

        args_used = {k: v for k, v in args.items() if k != '_script'}
        self.state = TrainingJobState(
            status='running',
            start_time=datetime.utcnow().isoformat(),
            args_used=args_used,
        )

        # Start background reader thread
        self._reader = Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        """Background thread: reads stdout line by line into the queue."""
        if self._process is None or self._process.stdout is None:
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
                self.state.end_time = datetime.utcnow().isoformat()
                self.state.status = 'completed' if rc == 0 else 'failed'
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
        self.state.status = 'stopped'
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
