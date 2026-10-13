# MalTwin — SRS Compliance Matrix
**Project:** MalTwin — IIoT Malware Detection Framework  
**Authors:** Iman Fatima (CIIT/SP23-BCT-021/ISB), Maaz Malik (CIIT/SP23-BCT-025/ISB)  
**Supervisor:** Ms. Najla Raza  
**Institution:** COMSATS University Islamabad | BS Cyber Security 2023-2027  
**SRS Version:** 1.0 | **Implementation Version:** 1.1 (Evaluation 3, October 2026)

> Performance figures below were measured with `scripts/benchmark.py` on a 2-core CPU-only machine (no GPU), using an untrained network of identical architecture (timings do not depend on the weights). Re-run on the demo machine after training to refresh them.

---

## Functional Requirements

### Mockup M1 — Main Dashboard Screen

| FR-ID | Requirement | Implementation | Status |
|-------|-------------|----------------|--------|
| FR1.1 | Display operational status of all modules | `modules/dashboard/health.py` → `get_all_module_statuses()` (M1–M8 + M9 dynamic + M10 fusion) cached 30s; styled table in `home.py` | ✅ Implemented* |
| FR1.2 | Display cumulative detection statistics | `db.py` `get_detection_stats()` reads SQLite + `eval_metrics.json`; rendered as `st.metric` in `home.py` | ✅ Implemented |
| FR1.3 | Persistent sidebar navigation | `app.py` `render_sidebar()` — 6-page radio nav; ⚠️ suffix on unavailable pages | ✅ Implemented |
| FR1.4 | Scrollable feed of 5 most recent detections | `home.py` `_render_history_section()` — filterable table with sort, family, confidence, date range, CSV export | ✅ Implemented (exceeds spec) |

### Mockup M2 — Digital Twin Monitor Screen

| FR-ID | Requirement | Implementation | Status |
|-------|-------------|----------------|--------|
| FR2.1 | Start/Stop simulation via dashboard | `digital_twin.py` stub page; Docker check in `health.py` `_check_module1_digital_twin()` | ⚠️ Deferred (M1 future sprint) |
| FR2.2 | Live traffic log | Planned for Digital Twin sprint | ⚠️ Deferred |
| FR2.3 | Node status panel | Planned for Digital Twin sprint | ⚠️ Deferred |
| FR2.4 | Protocol traffic pie chart | Planned for Digital Twin sprint | ⚠️ Deferred |

### Mockup M3 — Binary Upload and Visualization Screen

| FR-ID | Requirement | Implementation | Status |
|-------|-------------|----------------|--------|
| FR3.1 | Binary file upload (PE/ELF, max 50 MB) | `upload.py` size + magic-byte validation; `.streamlit/config.toml` caps uploads at 50 MB | ✅ Implemented |
| FR3.2 | Grayscale image display within 3 seconds | `upload.py` `_process_upload()` → `BinaryConverter` → `st.image()` | ✅ Implemented |
| FR3.3 | File metadata display (name, size, format, SHA-256) | `upload.py` `_render_results()` metadata table + monospace SHA-256 | ✅ Implemented |
| FR3.4 | Pixel intensity histogram (256 bins) | `upload.py` `_render_results()` Plotly bar chart via `compute_pixel_histogram()` | ✅ Implemented |

### Mockup M5 — Malware Detection and Prediction Screen

| FR-ID | Requirement | Implementation | Status |
|-------|-------------|----------------|--------|
| FR5.1 | Run detection on uploaded image | `detection.py` `_run_detection()` via `predict_single()` | ✅ Implemented |
| FR5.2 | Top-1 label + colour-coded confidence bar | `detection.py` `_render_results()` — green ≥80%, amber 50–79%, red <50% | ✅ Implemented |
| FR5.3 | Per-class probability chart (all 25 classes) | `detection.py` `_render_probability_chart()` horizontal Plotly bar chart | ✅ Implemented |
| FR5.4 | Grad-CAM XAI heatmap toggle | `detection.py` checkbox → `_run_gradcam()` → `generate_gradcam()` via Captum | ✅ Implemented (Step 1) |
| FR5.5 | MITRE ATT&CK for ICS mapping | `detection.py` `_render_mitre_mapping()` reads `mitre_ics_mapping.json` | ✅ Implemented |
| FR5.6 | PDF + JSON forensic report download | `detection.py` → `pdf_report.py` + `json_report.py` via FPDF2 | ✅ Implemented (Step 2) |

### Backend Process Requirements (Event-Response Table)

| FR-ID | Event | Implementation | Status |
|-------|-------|----------------|--------|
| FR-B1 | Binary uploaded | `upload.py` → `validate_binary_format()` → `BinaryConverter` → `compute_sha256()` | ✅ Implemented |
| FR-B2 | Detection run requested | `detection.py` → `predict_single()` → softmax probabilities | ✅ Implemented |
| FR-B3 | Detection result available → DB log | `detection.py` → `log_detection_event()`; failure never blocks the result. Deviation: retries once after 0.1 s (SRS: 2 s) | ✅ Implemented (minor deviation) |
| FR-B4 | Forensic report requested | `detection.py` → `generate_pdf_report()` / `generate_json_report()` + MITRE query | ✅ Implemented (Step 2) |
| FR-B5 | Digital Twin start requested | Deferred — Docker/Mininet not yet deployed | ⚠️ Deferred |
| FR-B6 | Grad-CAM heatmap requested | `detection.py` → `generate_gradcam()` via Captum `LayerGradCam` | ✅ Implemented (Step 1) |

---

## Non-Functional Requirements

### Reliability

| NFR-ID | Requirement | Implementation | Status |
|--------|-------------|----------------|--------|
| REL-1 | Identical results across 10 runs (variance < 0.5%) | `model.eval()` + `torch.no_grad()`; benchmark: max difference over 10 runs = 0.0 | ✅ Met (measured) |
| REL-2 | Corrupt files handled without crash | `upload.py` wraps parsing in try/except with USE-3 messages. Eval 3 fixed a crash (`NameError: mono`) that occurred after *every* successful upload | ✅ Met (after fix) |
| REL-3 | Digital Twin MTBF ≥ 4 hours | Deferred with M1 | ⚠️ Deferred |
| REL-4 | SQLite WAL mode — no records lost on unclean shutdown | `db.py` `get_connection()` sets `PRAGMA journal_mode=WAL` on every connection | ✅ Met |
| REL-5 | Handle 2 concurrent sessions without collision | Streamlit session isolation via `st.session_state` — each session has independent state | ✅ Met |

### Usability

| NFR-ID | Requirement | Implementation | Status |
|--------|-------------|----------------|--------|
| USE-1 | First-time user completes upload + detection in < 5 minutes | Sidebar flow with inline instructions; Eval 3 fixed a navigation bug that discarded the user's click after upload | 🟡 Implemented — timed user trial pending |
| USE-2 | Visual indicators interpretable without ML background | Colour-coded confidence bar, emoji status, labelled MITRE table | ✅ Met |
| USE-3 | All errors: what + cause + action | All `st.error()` calls follow "Error: X. Cause: Y. Action: Z." format | ✅ Met |
| USE-4 | Tooltips on technical terms | `help=` parameter on all technical widgets (confidence, Grad-CAM, MITRE) | ✅ Met |
| USE-5 | Report download < 10 seconds, opens in standard viewers | FPDF2 PDF + JSON; generation measured at ~9 ms (PER-5) | ✅ Met (measured) |

### Performance

| NFR-ID | Requirement | Implementation | Status |
|--------|-------------|----------------|--------|
| PER-1 | Conversion < 3s for ≤10 MB, < 10s for ≤50 MB | `BinaryConverter` (zero-copy `np.frombuffer` + one `cv2.resize`): 10 MB ≈ 0.2 ms, 50 MB ≈ 0.4 ms (max of 5 runs) | ✅ Met (measured) |
| PER-2 | Inference < 5s CPU, < 1s GPU | `predict_single()`: mean 14 ms, max 26 ms on CPU | ✅ Met (measured, CPU) |
| PER-3 | Dashboard renders within 4s on local network | Streamlit on localhost | 🟡 Implemented — not yet timed |
| PER-4 | Grad-CAM < 8s CPU, < 3s GPU | Captum `LayerGradCam`: mean 0.26 s, max 0.49 s on CPU | ✅ Met (measured, CPU) |
| PER-5 | Report generation < 10s | PDF + JSON together: mean 9 ms | ✅ Met (measured) |
| PER-6 | DB supports 100k records, query < 500ms | 100,000-row table, filtered GROUP BY: mean 94 ms, max 105 ms | ✅ Met (measured) |

### Security

| NFR-ID | Requirement | Implementation | Status |
|--------|-------------|----------------|--------|
| SEC-1 | Uploaded files processed in Docker only | Files are read as bytes in memory and never executed or written to disk, but the dashboard itself runs on the host. Container isolation arrives with the Digital Twin sandbox node | ⚠️ Partial |
| SEC-2 | Digital Twin network isolated | Planned for M1 sprint with `--internal` Docker bridge network | ⚠️ Deferred |
| SEC-3 | SQLite file permissions 600 | `db.py` `init_db()` calls `os.chmod(db_path, 0o600)` after creation | ✅ Met |
| SEC-4 | SHA-256 computed locally via hashlib | `utils.py` `compute_sha256()` uses `hashlib.sha256()` — no external calls | ✅ Met |
| SEC-5 | Dashboard bound to localhost by default | `.streamlit/config.toml` sets `server.address = 127.0.0.1`; `app.py` warns on non-localhost binding | ✅ Implemented |

---

## Use Cases

| UC-ID | Use Case | Implementation | Status |
|-------|----------|----------------|--------|
| UC-01 | Upload Binary File and Convert | `upload.py` full flow — validate → convert → display → store in state | ✅ Implemented |
| UC-02 | Run Malware Detection and View Prediction | `detection.py` full flow — guard → run → display → log | ✅ Implemented |
| UC-03 | Generate and Download Forensic Report | `detection.py` → `pdf_report.py` + `json_report.py` → `st.download_button` | ✅ Implemented |
| UC-04 | Simulate IIoT Environment via Digital Twin | `digital_twin.py` stub — full implementation deferred | ⚠️ Deferred |
| UC-05 | Train Detection Model | `training.py` → `training_manager.py` → `scripts/train.py` subprocess | ✅ Implemented (Step 5) |

---

## Business Objectives

| BO-ID | Objective | Implementation | Status |
|-------|-----------|----------------|--------|
| BO-1 | Digital Twin IIoT simulation | `digital_twin.py` stub; Docker check in health.py | ⚠️ Deferred |
| BO-2 | Binary-to-image conversion pipeline | `modules/binary_to_image/` — PE/ELF → 128×128 grayscale | ✅ Implemented |
| BO-3 | Data enhancement + class balancing | `modules/enhancement/` — augmentation + `ClassAwareOversampler` | ✅ Implemented |
| BO-4 | CNN malware classification | `MalTwinCNN` — 3-block CNN, 25-class, 1.35 M parameters | ✅ Implemented |
| BO-5 | Evaluation metrics (acc/prec/rec/F1/CM) on seen *and unseen* samples | `evaluator.py` adds ROC-AUC, per-class FPR, MCC, top-3; `scripts/evaluate_unseen.py` leave-families-out experiment | ✅ Implemented — final numbers require the Malimg training run |
| BO-6 | Grad-CAM XAI | `gradcam.py` via Captum `LayerGradCam`, jet overlay, heatmap export | ✅ Implemented |
| BO-7 | Interactive Streamlit dashboard | 6-page dashboard with training, gallery, history, XAI, reporting | ✅ Implemented |
| BO-8 | Automated forensic reporting (PDF + JSON + MITRE) | `modules/reporting/` — FPDF2 PDF, structured JSON, 25-family MITRE DB | ✅ Implemented |

---

## Module Feature Coverage

| Module | Feature | SRS FE-ID | Status |
|--------|---------|-----------|--------|
| M2 | Accept PE/ELF, validate header | FE-1 | ✅ |
| M2 | Read bytes, reshape to 2D array | FE-2 | ✅ (Eval 3: Malimg width table so uploads match training images) |
| M2 | Render 128×128 grayscale PNG | FE-3 | ✅ |
| M2 | Compute SHA-256 | FE-4 | ✅ |
| M3 | Source from Malimg dataset | FE-1 | ✅* (Malimg only; VirusShare/IoT-23 deferred) |
| M3 | Normalise pixels, encode labels | FE-2 | ✅ |
| M3 | Stratified train/val/test split | FE-3 | ✅ |
| M3 | Validate dataset integrity | FE-4 | ✅ |
| M4 | Rotation, flip, brightness augmentation | FE-1 | ✅ |
| M4 | Gaussian noise injection | FE-2 | ✅ |
| M4 | Class-aware oversampling | FE-3 | ✅ |
| M4 | Dataset gallery visualisation | FE-4 | ✅ (Step 3) |
| M5 | CNN architecture | FE-1 | ✅ |
| M5 | Configurable training | FE-2 | ✅ |
| M5 | Acc/prec/rec/F1/CM evaluation | FE-3 | ✅ |
| M5 | Per-class probability + top-1 label | FE-4 | ✅ |
| M5 | PyTorch .pt serialisation | FE-5 | ✅ |
| M6 | Main dashboard + module status | FE-1 | ✅ |
| M6 | Binary upload + image display | FE-2 | ✅ |
| M6 | Detection result + confidence | FE-3 | ✅ |
| M6 | Grad-CAM heatmap display | FE-4 | ✅ (Step 1) |
| M6 | Dataset gallery | FE-5 | ✅ (Step 3) |
| M6 | Digital Twin status | FE-6 | 🟡 Design page with live prerequisite checks (Docker, Zeek, dynamic model) |
| M7 | Captum LayerGradCam | FE-1 | ✅ (Step 1) |
| M7 | Jet colormap overlay on binary image | FE-2 | ✅ (Step 1) |
| M7 | Textual interpretation annotations | FE-3 | ✅ (Step 1) |
| M7 | XAI heatmap export | FE-4 | ✅ (Step 1) |
| M8 | PDF + JSON report generation | FE-1 | ✅ (Step 2) |
| M8 | Report content (hash, family, conf, heatmap) | FE-2 | ✅ (Step 2) |
| M8 | MITRE ATT&CK for ICS mapping | FE-3 | ✅ (Step 2) |
| M8 | SQLite detection event logging | FE-4 | ✅ |
| M8 | Detection history view with filtering | FE-5 | ✅ (Step 3) |

---

## Evaluation 3 Additions (from the proposal objectives)

The proposal presentation (objectives 03 *Dynamic Feature Extraction* and 04 *Hybrid Detection Models*) promised behavioural analysis and static–dynamic fusion. These were not in SRS v1.0; they are tracked here as modules M9 and M10.

| ID | Feature | Implementation | Status |
|----|---------|----------------|--------|
| M9-1 | Ingest network behaviour logs | `modules/dynamic/zeek_parser.py` — Zeek conn.log / IoT-23 `conn.log.labeled` (handles IoT-23's space-separated label columns, reservoir sampling for multi-GB files) | ✅ |
| M9-2 | Behavioural feature extraction | `modules/dynamic/features.py` — 61 flow features, no IP addresses (avoids host-identity leakage), IIoT port indicators (Modbus 502, MQTT 1883/8883, Telnet 23/2323) | ✅ |
| M9-3 | Behaviour classification | `modules/dynamic/detector.py` — random forest, `class_weight='balanced_subsample'`; classes Benign, PortScan, C&C, DDoS, Attack, FileDownload, Okiru | ✅ (needs IoT-23 training run) |
| M9-4 | Capture-level verdict | Mean P(malicious) of the top 5% flows → MALICIOUS / SUSPICIOUS / BENIGN | ✅ |
| M9-5 | Feature-importance explanation | Global RF importance + per-capture deviation from the benign baseline | ✅ |
| M9-6 | MITRE ATT&CK for ICS per behaviour | `data/mitre_ics_behaviour_mapping.json` | ✅ |
| M9-7 | Dashboard page | `pages/dynamic.py` | ✅ |
| M10-1 | Decision-level fusion | `modules/fusion/fusion.py` — weighted score + agreement category; NOVEL_VARIANT escalation rule | ✅ |
| M10-2 | Combined MITRE mapping | Techniques tagged by evidence source, corroborated first | ✅ |
| M10-3 | Dashboard page + what-if simulator | `pages/hybrid.py` | ✅ |
| M10-4 | Paired evaluation (same sample, both views) | Requires Digital Twin to execute samples and capture their traffic | ⏳ Final evaluation |

## Defects Found and Fixed in Evaluation 3

| # | Defect | Impact | Fix |
|---|--------|--------|-----|
| D1 | `NameError: mono` in `upload.py` and `detection.py` | Upload page crashed after every upload; detection page crashed on render | Module-level import |
| D2 | Sidebar labels changed after upload ("(no file)" suffix) | Streamlit re-created the widget and discarded the user's navigation click | Fixed labels, stable widget key, status shown in System panel |
| D3 | Streamlit auto-discovered `pages/` | Second, broken navigation list exposing blank pages | `client.showSidebarNavigation = false` |
| D4 | `ReduceLROnPlateau(verbose=True)` | Training crashed immediately on newer PyTorch | Argument removed (LR already logged per epoch) |
| D5 | Upload conversion used a square layout; Malimg images use the Nataraj width table | Uploaded binaries differed from the training distribution | `layout='malimg'` default |
| D6 | `st.image(use_column_width=…)` removed in newer Streamlit | Warning/no effect | Version-safe `image_full_width()` helper |
| D7 | Flaky unit test (random data, dropout on) | Intermittent CI failure | Seeded, dropout disabled for the overfit check |
| D8 | Histogram unit tests skipped (function renamed) | Feature silently untested | Tests repointed to `compute_pixel_histogram` |
| D9 | Upload page re-processed the file on every rerun; sidebar drawn before upload handled | Detection result wiped when revisiting Upload; stale sidebar status | Skip already-processed SHA-256; refresh once after upload |
| D10 | CNN documented as ~3.2 M parameters | Incorrect documentation | Corrected to 1,348,793 (≈1.35 M) |


---

## Summary

| Category | Total | Implemented | Deferred | 
|----------|-------|-------------|----------|
| Functional Requirements | 18 | 14 | 4 (all M1/Digital Twin) |
| Non-Functional Requirements | 21 (REL 5, USE 5, PER 6, SEC 5) | 16 met · 2 implemented, validation pending (USE-1, PER-3) · 1 partial (SEC-1) | 2 (REL-3, SEC-2 — Digital Twin) |
| Eval-3 additions (M9, M10) | 11 | 10 | 1 (M10-4 paired evaluation — Digital Twin) |
| Use Cases | 5 | 4 | 1 (UC-04 Digital Twin) |
| Business Objectives | 8 | 7 | 1 (BO-1 Digital Twin) |
| Module Features | 30 | 28 | 2 |

**All deferred items depend on the Digital Twin (M1).** Model-quality figures (accuracy, F1, ROC-AUC, unseen-family AUROC, dynamic detection rate) are produced by the training scripts and are reported in the Evaluation 3 document once the Malimg and IoT-23 runs complete.  
The ML pipeline, dashboard, XAI, and reporting are fully implemented.

---

## Deviation Footnotes

*FR1.1 TTL deviation: SRS specifies 5-second refresh. Implementation uses 30-second
cache TTL (health.py:225) to avoid excessive filesystem polling during normal use.
Functionally equivalent for research deployment. Would require architectural change
(websocket or server-sent events) to achieve true 5-second push updates in Streamlit.

*SI-5 deviation: SRS specifies PyTorch ImageFolder for dataset loading.
Implementation uses a custom MalimgDataset class (modules/dataset/loader.py)
that provides stratified splitting, class count tracking, and integration
with WeightedRandomSampler — capabilities ImageFolder does not natively
support. This is an intentional improvement over the specification.
