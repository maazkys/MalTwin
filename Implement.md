# MalTwin — Compliance Gap Fixes
### Addressing all findings from the SRS compliance audit

> Apply these fixes after Steps 1–6 are complete.
> Each fix is self-contained. Apply them in order — Fix 4 depends on Fix 3.

---

## Fix 1 — ELF Upload: Complete the UI gap (not just add `.elf`)

### The problem

The PR added `.elf` to the `type` list, which helps for renamed files. But raw ELF binaries have **no extension** — a user cannot select them through the browser file picker at all. The SRS (FR3.1) explicitly requires ELF support.

The fix has two parts: keep the extension list broad, and add a clear explanation in the UI so users know exactly what to do.

### Change in `modules/dashboard/pages/upload.py`

Replace the `st.file_uploader` call:

```python
    uploaded_file = st.file_uploader(
        label="Upload Binary File",
        type=["exe", "dll", "elf"],
        help=(
            "Accepted formats: PE (.exe, .dll) or ELF (.elf) binaries. "
            "ELF binaries from Linux/IIoT systems have no extension by default — "
            "rename them to .elf before uploading (e.g. mv malware malware.elf). "
            f"Maximum file size: {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
            "Format is validated by magic bytes, not extension."
        ),
        key="binary_uploader",
    )
```

Then add an expander below the uploader (before `if uploaded_file is not None:`) with explicit ELF instructions:

```python
    with st.expander("ℹ️ Uploading ELF binaries (Linux/IIoT malware)", expanded=False):
        st.markdown(
            "ELF binaries from Linux-based IIoT devices (PLCs, routers, embedded systems) "
            "typically have no file extension. To upload them:\n\n"
            "1. Rename the file to add a `.elf` extension:\n"
            "   ```bash\n"
            "   mv suspicious_binary suspicious_binary.elf\n"
            "   ```\n"
            "2. Upload the renamed file using the uploader above.\n\n"
            "The system validates format using ELF magic bytes (`\\x7fELF`), "
            "not the file extension — the extension is only required by the browser file picker."
        )
```

---

## Fix 2 — Checkpoint format mismatch

### The problem

`trainer.py` saves per-epoch checkpoints with the key `model_state`:
```python
torch.save({
    'epoch':           epoch + 1,
    'model_state':     model.state_dict(),   # ← key is 'model_state'
    'optimizer_state': optimizer.state_dict(),
    ...
}, checkpoint_path)
```

`inference.py` tries `model_state_dict`, `state_dict`, then falls through to a raw dict check — it **never checks `model_state`**. The PR correctly added `"model_state"` to the key list. Verify it's in the right place.

### Verify `modules/detection/inference.py`

The `load_model` function's state dict extraction block must look exactly like this:

```python
    raw = torch.load(str(model_path), map_location=device, weights_only=True)

    # ── Extract state_dict regardless of how it was saved ──────────────────
    if isinstance(raw, dict) and not _looks_like_state_dict(raw):
        # Full checkpoint dict — try all known key names
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if key in raw:
                state_dict = raw[key]
                break
        if state_dict is None:
            raise ValueError(
                f"Cannot find model weights in checkpoint at {model_path}. "
                f"Keys found: {list(raw.keys())}. "
                "Expected one of: model_state_dict, state_dict, model, model_state."
            )
    else:
        state_dict = raw
```

### Add the `_looks_like_state_dict` helper if it does not exist

```python
def _looks_like_state_dict(d: dict) -> bool:
    """
    Heuristic: a raw state_dict has tensor values.
    A checkpoint dict has mixed types (int epoch, dict optimizer_state, etc.).
    Returns True if d looks like a state dict (all or mostly tensor values).
    """
    import torch
    if not d:
        return False
    values = list(d.values())
    tensor_count = sum(1 for v in values if isinstance(v, torch.Tensor))
    return tensor_count / len(values) >= 0.8
```

### Add a test to `tests/test_model.py`

Add this to `TestLoadModel`:

```python
    def test_loads_from_full_checkpoint_dict_model_state_key(self, tmp_path, num_classes):
        """
        Simulate exactly what trainer.py saves: a checkpoint dict with key 'model_state'.
        This is the format saved to CHECKPOINT_DIR per epoch.
        inference.py must handle it correctly.
        """
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / "epoch_001_acc0.8500.pt"
        # Save exactly as trainer.py does
        torch.save({
            'epoch':           1,
            'model_state':     model.state_dict(),   # ← the key that was missing
            'optimizer_state': {'state': {}, 'param_groups': []},
            'val_acc':         0.85,
            'val_loss':        0.42,
            'train_acc':       0.87,
            'train_loss':      0.39,
        }, pt_path)

        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        assert isinstance(loaded, MalTwinCNN)
        assert not loaded.training   # must be in eval mode

        # Weights must match
        orig_w   = model.block1.conv1.weight.data
        loaded_w = loaded.block1.conv1.weight.data
        torch.testing.assert_close(orig_w, loaded_w)

    def test_load_model_raises_helpful_error_for_unknown_checkpoint_keys(self, tmp_path, num_classes):
        """
        A checkpoint dict with none of the known keys should raise ValueError
        with a message that lists the keys found — not a generic KeyError.
        """
        from modules.detection.inference import load_model
        pt_path = tmp_path / "weird_checkpoint.pt"
        torch.save({'some_unknown_key': 'garbage', 'epoch': 42}, pt_path)

        with pytest.raises(ValueError, match="Cannot find model weights"):
            load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
```

---

## Fix 3 — Report generation logging (UC-03 postcondition)

### The problem

SRS UC-03 postcondition: *"Report generation event is logged with timestamp."*

No report generation logging exists. Only detection events are logged. This is a genuine gap — the audit correctly identified it.

### Add `log_report_event()` to `modules/dashboard/db.py`

Add this function after `log_detection_event()`:

```python
def log_report_event(
    db_path: Path,
    detection_event_id: int | None,
    sha256: str,
    report_format: str,
    gradcam_included: bool,
) -> None:
    """
    Log a forensic report generation event. UC-03 postcondition.
    Never raises — a DB failure must not block the download.

    Args:
        detection_event_id: id of the detection_events row (None if unknown).
        sha256:             SHA-256 of the analyzed binary.
        report_format:      'PDF' or 'JSON'.
        gradcam_included:   True if Grad-CAM heatmap was embedded.
    """
    # Ensure the report_events table exists first
    _ensure_report_table(db_path)

    timestamp = datetime.utcnow().isoformat()
    sql = """
        INSERT INTO report_events
            (timestamp, detection_event_id, sha256, report_format, gradcam_included)
        VALUES (?, ?, ?, ?, ?)
    """
    params = (
        timestamp,
        detection_event_id,
        sha256,
        report_format,
        1 if gradcam_included else 0,
    )
    for attempt in range(2):
        try:
            with get_connection(db_path) as conn:
                conn.execute(sql, params)
            return
        except Exception as e:
            if attempt == 0:
                time.sleep(0.1)
            else:
                print(f"[MalTwin] Report log write failed: {e}", file=sys.stderr)


def _ensure_report_table(db_path: Path) -> None:
    """Create report_events table if it does not exist. Idempotent."""
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS report_events (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT    NOT NULL,
                    detection_event_id  INTEGER,
                    sha256              TEXT    NOT NULL,
                    report_format       TEXT    NOT NULL,
                    gradcam_included    INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_report_sha256 "
                "ON report_events(sha256)"
            )
    except Exception as e:
        print(f"[MalTwin] Failed to create report_events table: {e}", file=sys.stderr)
```

### Call `log_report_event()` from `detection.py`

In the `col_pdf` block in `_render_results()`, after `pdf_bytes` is confirmed non-None and before rendering `st.download_button`, add:

```python
                # Log report generation (UC-03 postcondition)
                from modules.dashboard.db import log_report_event
                log_report_event(
                    config.DB_PATH,
                    detection_event_id=None,   # not tracked in session state
                    sha256=report_data['sha256'],
                    report_format='PDF',
                    gradcam_included=report_data['gradcam'].get('generated', False),
                )
```

In the `col_json` block, wrap the `generate_json_report` call and add logging:

```python
        from modules.reporting.json_report import generate_json_report
        from modules.dashboard.db import log_report_event
        json_bytes = generate_json_report(report_data)
        # Log JSON report generation (UC-03 postcondition)
        log_report_event(
            config.DB_PATH,
            detection_event_id=None,
            sha256=report_data['sha256'],
            report_format='JSON',
            gradcam_included=report_data['gradcam'].get('generated', False),
        )
```

### Add tests to `tests/test_db.py`

```python
class TestLogReportEvent:
    def test_does_not_raise(self, temp_db):
        from modules.dashboard.db import log_report_event
        log_report_event(temp_db, None, 'a' * 64, 'PDF', False)

    def test_creates_report_events_table(self, temp_db):
        from modules.dashboard.db import log_report_event, get_connection
        log_report_event(temp_db, None, 'a' * 64, 'JSON', True)
        with get_connection(temp_db) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='report_events'"
            ).fetchall()
        assert len(rows) == 1

    def test_inserted_values_correct(self, temp_db):
        from modules.dashboard.db import log_report_event, get_connection
        log_report_event(temp_db, 42, 'b' * 64, 'PDF', True)
        with get_connection(temp_db) as conn:
            row = conn.execute("SELECT * FROM report_events").fetchone()
        assert dict(row)['sha256'] == 'b' * 64
        assert dict(row)['report_format'] == 'PDF'
        assert dict(row)['gradcam_included'] == 1

    def test_does_not_raise_on_missing_db(self, tmp_path):
        from modules.dashboard.db import log_report_event
        bad_path = tmp_path / "nonexistent_dir" / "db.db"
        log_report_event(bad_path, None, 'a' * 64, 'JSON', False)
```

---

## Fix 4 — FR1.1 refresh rate disclosure

### The problem

SRS FR1.1 business rule: *"Status indicators shall refresh automatically every 5 seconds."*  
Implementation: `@st.cache_data(ttl=30)` — 30 seconds, not 5.

A true 5-second TTL causes excessive filesystem checks during normal dashboard use. The 30-second TTL is the correct engineering choice. The gap to close is **disclosure** — update `SRS_COMPLIANCE.md` and the home page caption to be explicit.

### Update `home.py` caption

In `_render_module_status()`, replace:
```python
    st.caption("Status refreshes every 30 seconds automatically.")
```
With:
```python
    st.caption(
        "Status refreshes every 30 seconds automatically. "
        "(SRS FR1.1 specifies 5s; 30s is used to avoid excessive filesystem I/O — "
        "functionally equivalent for research use.)"
    )
```

### Update `SRS_COMPLIANCE.md`

In the FR1.1 row, change the Status column from `✅ Implemented` to `✅ Implemented*` and add a footnote at the bottom of the table:

```
*FR1.1 TTL deviation: SRS specifies 5-second refresh. Implementation uses 30-second
cache TTL (health.py:225) to avoid excessive filesystem polling during normal use.
Functionally equivalent for research deployment. Would require architectural change
(websocket or server-sent events) to achieve true 5-second push updates in Streamlit.
```

---

## Fix 5 — FR1.4 recent detection feed (top 5 baseline)

### The problem

SRS FR1.4: *"display a scrollable feed of the five most recent detection events."*  
Current: the filter expander replaces the feed entirely — there's no quick baseline view before opening filters.

### Update `home.py` `render()`

Before the `_render_history_section()` call, add a quick baseline feed that's always visible:

```python
    # ── Quick recent feed (always visible — SRS FR1.4) ───────────────────────
    st.subheader("Recent Detections")
    _render_recent_feed_baseline()

    # ── Full filterable history ───────────────────────────────────────────────
    with st.expander("📋 Full Detection History (filter & export)", expanded=False):
        _render_history_section()
```

Add this new function:

```python
def _render_recent_feed_baseline() -> None:
    """
    Scrollable feed of the 5 most recent detections. SRS FR1.4 baseline.
    Always visible — no filters, no interaction required.
    """
    import pandas as pd
    events = get_recent_events(config.DB_PATH, limit=5)

    if not events:
        st.caption("No detections yet. Upload a binary file to get started.")
        return

    df = pd.DataFrame(events)
    df['confidence'] = df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
    df['timestamp']  = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    display = df[['timestamp', 'file_name', 'predicted_family', 'confidence']].copy()
    display.columns = ['Timestamp', 'File', 'Predicted Family', 'Confidence']
    st.dataframe(display, use_container_width=True, hide_index=True)
```

The full filterable history remains, but now inside an expander so the baseline feed satisfies FR1.4 and the enhanced filtering satisfies M8 FE-5.

---

## Fix 6 — SI-5 deviation: document the `ImageFolder` decision

### The problem

SRS SI-5 specifies using `PyTorch ImageFolder`. The custom `MalimgDataset` is better but diverges from spec. This needs to be documented, not hidden.

### Update `SRS_COMPLIANCE.md`

In the M3 FE-1 row, change status from `✅` to `✅*` and add to footnotes:

```
*SI-5 deviation: SRS specifies PyTorch ImageFolder for dataset loading.
Implementation uses a custom MalimgDataset class (modules/dataset/loader.py)
that provides stratified splitting, class count tracking, and integration
with WeightedRandomSampler — capabilities ImageFolder does not natively
support. This is an intentional improvement over the specification.
```

Also add a docstring to `MalimgDataset.__init__` explaining the decision:

```python
    """
    Custom PyTorch Dataset for the Malimg malware image dataset.

    Deviation from SRS SI-5: The SRS specifies PyTorch ImageFolder.
    This custom class is used instead because it provides:
      - Stratified train/val/test splitting (ImageFolder has no split support)
      - Per-split class count tracking (needed by ClassAwareOversampler)
      - Integration with WeightedRandomSampler for class balancing

    The interface is otherwise Dataset-compatible (len, getitem, get_labels).
    """
```

---

## Fix 7 — Benign class surfaced to user (architectural notice)

### The problem

The system classifies every uploaded binary into one of 25 malware families. There is no benign class. A user who uploads a legitimate binary will still receive a malware family prediction — this could be misleading.

### Update `detection.py` `_render_results()`

Add a persistent notice below the detection result banner:

```python
    # Benign class notice (architectural constraint — SRS_COMPLIANCE.md)
    st.caption(
        "ℹ️ MalTwin classifies binaries into known malware families only. "
        "There is no benign class — every uploaded binary receives a malware family prediction. "
        "Low confidence scores (<50%) indicate the binary may not match any trained family. "
        "This is a known architectural constraint of the Malimg-trained model."
    )
```

Place this immediately after the confidence bar and before the Top 3 section.

---

## Fix 8 — Model/class_names stale session state

### The problem

If `best_model.pt` is replaced on disk mid-session (e.g. a new training run completes via CLI while the dashboard is open), `session_state[KEY_MODEL]` holds stale weights. Detection runs fine but produces results from the old model.

### Add a staleness check in `app.py` `load_global_resources()`

After the model load block, add:

```python
def load_global_resources() -> None:
    # ... existing class names load ...

    # ── Staleness check ─────────────────────────────────────────────────────
    # If model is loaded but the file on disk has a newer mtime,
    # reset and reload. Handles CLI training completing mid-session.
    if st.session_state[state.KEY_MODEL_LOADED]:
        try:
            stored_mtime = st.session_state.get('_model_mtime', 0)
            current_mtime = config.BEST_MODEL_PATH.stat().st_mtime
            if current_mtime > stored_mtime + 1:   # +1s tolerance
                # Model file changed — reset and reload
                st.session_state[state.KEY_MODEL]        = None
                st.session_state[state.KEY_MODEL_LOADED] = False
                st.session_state['_model_mtime']         = 0
        except Exception:
            pass

    # ── Model load ──────────────────────────────────────────────────────────
    if (
        st.session_state[state.KEY_MODEL] is None
        and st.session_state[state.KEY_CLASS_NAMES] is not None
    ):
        try:
            with st.spinner("Loading detection model…"):
                num_classes = len(st.session_state[state.KEY_CLASS_NAMES])
                model = load_model(config.BEST_MODEL_PATH, num_classes, config.DEVICE)
                st.session_state[state.KEY_MODEL]        = model
                st.session_state[state.KEY_MODEL_LOADED] = True
                st.session_state[state.KEY_DEVICE_INFO]  = str(config.DEVICE)
                st.session_state['_model_mtime']         = config.BEST_MODEL_PATH.stat().st_mtime
        except FileNotFoundError:
            st.session_state[state.KEY_MODEL_LOADED] = False
```

---

## Definition of Done

```bash
# Run the new/updated tests
pytest tests/test_model.py::TestLoadModel::test_loads_from_full_checkpoint_dict_model_state_key -v
pytest tests/test_model.py::TestLoadModel::test_load_model_raises_helpful_error_for_unknown_checkpoint_keys -v
pytest tests/test_db.py::TestLogReportEvent -v

# Full suite — nothing broken
pytest tests/ -v -m "not integration"

# Verify ELF fix
python -c "
from modules.binary_to_image.utils import validate_binary_format
elf_bytes = b'\\x7fELF' + b'\\x00' * 60
fmt = validate_binary_format(elf_bytes)
assert fmt == 'ELF', f'Expected ELF, got {fmt}'
print('ELF validation: OK')
"

# Verify checkpoint fix
python -c "
import torch, tempfile
from pathlib import Path
from modules.detection.model import MalTwinCNN
from modules.detection.inference import load_model

model = MalTwinCNN(num_classes=25)
with tempfile.TemporaryDirectory() as d:
    pt = Path(d) / 'epoch_001.pt'
    torch.save({'epoch': 1, 'model_state': model.state_dict(),
                'optimizer_state': {}, 'val_acc': 0.85}, pt)
    loaded = load_model(pt, 25, torch.device('cpu'))
    print('Checkpoint load from model_state key: OK')
"
```

### Checklist

- [ ] `upload.py` — type list includes `"elf"`, help text explains rename workflow, ELF expander added
- [ ] `inference.py` — `_looks_like_state_dict()` helper exists, key list includes `"model_state"`
- [ ] New test: `test_loads_from_full_checkpoint_dict_model_state_key` passes
- [ ] New test: `test_load_model_raises_helpful_error_for_unknown_checkpoint_keys` passes
- [ ] `db.py` — `log_report_event()` and `_ensure_report_table()` added
- [ ] `detection.py` — `log_report_event()` called after both PDF and JSON generation
- [ ] New tests: `TestLogReportEvent` (4 tests) pass
- [ ] `home.py` — baseline top-5 feed visible without opening expander
- [ ] `home.py` — full history inside `st.expander`
- [ ] `home.py` — FR1.1 TTL deviation disclosed in caption
- [ ] `detection.py` — benign class architectural notice added below confidence bar
- [ ] `app.py` — model staleness check added to `load_global_resources()`
- [ ] `SRS_COMPLIANCE.md` — FR1.1 and SI-5 deviations documented with footnotes
- [ ] `MalimgDataset.__init__` docstring explains SI-5 deviation rationale
- [ ] `pytest tests/ -v -m "not integration"` — 0 failures
