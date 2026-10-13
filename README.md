# MalTwin — IIoT Malware Detection Framework

> Hybrid static + dynamic malware detection for Industrial IoT: binary visualisation with a custom CNN, behavioural analysis of network traffic, and explainable fusion of the two.  
> BS Cyber Security Capstone Project — COMSATS University Islamabad, 2023–2027

---

## What is MalTwin?

MalTwin detects and classifies malware targeting Industrial IoT (IIoT) systems. It takes a PE or ELF binary file, converts its raw bytes into a grayscale image that captures the binary's structural patterns, then passes that image through a trained Convolutional Neural Network to classify it into one of **25 known malware families** from the Malimg dataset.

Alongside this **static** view, a **dynamic** module classifies the network behaviour a sample produces (Zeek `conn.log`, trained on IoT-23), and a **fusion** module combines both into one verdict — flagging the important case where a binary matches no known family but behaves maliciously (a likely new variant).

Results are displayed through a Streamlit web dashboard with confidence scores, per-class probability charts, and MITRE ATT&CK for ICS technique mappings. Every detection is logged to a local SQLite database.

### How it works

```
Binary (.exe / .dll / ELF)
        ↓
 Validate format (PE / ELF magic bytes)
        ↓
 Convert raw bytes → 128×128 grayscale PNG
        ↓
 Run CNN inference → malware family label + confidence
        ↓
 Look up MITRE ATT&CK ICS techniques
        ↓
 Log to SQLite + display on dashboard
```

---

## Features

- **Binary visualisation** — converts any PE/ELF binary to a structural grayscale image with no ML dependencies
- **Upload validation & metadata** — 50 MB cap, magic-byte validation, SHA-256 hashing, and pixel histogram
- **25-family CNN classifier** — 3-block ConvNet trained on the Malimg dataset with weighted oversampling to handle class imbalance
- **Confidence-coded results** — green / amber / red UI based on prediction confidence thresholds
- **MITRE ATT&CK for ICS mapping** — tactics and techniques for each detected malware family
- **Detection history** — SQLite log with CSV export and filtering
- **Grad-CAM XAI** — heatmap overlays that highlight which byte regions drove a prediction
- **Forensic reporting** — PDF + JSON exports with MITRE mappings and optional Grad-CAM embeds
- **Dataset gallery** — per-family image grids with MITRE context
- **Training manager** — run model training from the dashboard with live logs and progress
- **System health checks** — module status + CPU/RAM metrics in the dashboard
- **Custom dashboard theme** — dark industrial Streamlit theme with CSS overrides and chart styling
- **Dynamic behavioural analysis (M9)** — Zeek conn.log parser (IoT-23 compatible), IIoT-aware flow features, random-forest behaviour classifier (PortScan, C&C, DDoS, Okiru, …) with per-capture explanations and MITRE ICS mapping
- **Hybrid fusion (M10)** — decision-level fusion of CNN and behavioural scores with agreement categories (CONFIRMED / STATIC_ONLY / NOVEL_VARIANT / NO_EVIDENCE) and a what-if simulator
- **Extended evaluation** — ROC-AUC, per-class false positive rate, MCC, top-3 accuracy, ROC/training curves, and a leave-families-out *unseen variant* experiment
- **Performance benchmark** — measures the SRS PER-x requirements on the current machine
- **Streamlit dashboard** — eight-page web UI (overview, upload, detection, gallery, dynamic analysis, hybrid verdict, training, digital twin design)
- **CLI & utilities** — training, evaluation, binary conversion, synthetic data generation, MITRE verification

---

## Repository Structure

```
MalTwin/
├── README.md
├── config.py                        # Central config — all paths and hyperparameters
├── conftest.py                      # Pytest path setup for project root
├── requirements.txt
├── .env.example                     # Copy to .env and edit before running
├── Implement.md                     # Implementation walkthrough + test notes
├── SRS_COMPLIANCE.md                # SRS traceability matrix
├── MALTWIN_SRS.md                   # SRS document (Markdown)
├── MalTwin_Testing_Documentation.docx
├── verify_mitre.py                  # Validate MITRE mapping vs class_names.json
│
├── modules/
│   ├── __init__.py
│   ├── binary_to_image/
│   │   ├── converter.py             # BinaryConverter — bytes → 128×128 numpy array
│   │   └── utils.py                 # validate_binary_format, compute_sha256, histogram
│   │
│   ├── dataset/
│   │   ├── loader.py                # MalimgDataset (PyTorch), get_dataloaders()
│   │   └── preprocessor.py          # validate_dataset_integrity, encode_labels, etc.
│   │
│   ├── enhancement/
│   │   ├── augmentor.py             # get_train_transforms, get_val_transforms, GaussianNoise
│   │   └── balancer.py              # ClassAwareOversampler → WeightedRandomSampler
│   │
│   ├── detection/
│   │   ├── model.py                 # MalTwinCNN (ConvBlock × 3 + classifier)
│   │   ├── trainer.py               # train(), validate_epoch()
│   │   ├── evaluator.py             # evaluate(), plot_confusion_matrix()
│   │   ├── inference.py             # load_model(), predict_single(), predict_batch()
│   │   ├── gradcam.py               # Grad-CAM heatmap generator
│   │   └── openset.py               # Unseen-family (open-set) metrics
│   │
│   ├── dynamic/                     # M9 — behavioural analysis
│   │   ├── zeek_parser.py           # Zeek / IoT-23 conn.log parser
│   │   ├── features.py              # IP-free, IIoT-aware flow features
│   │   ├── detector.py              # Random-forest detector + explanations
│   │   └── synthetic.py             # Synthetic flow generator (tests/demo)
│   │
│   ├── fusion/                      # M10 — hybrid static+dynamic fusion
│   │   └── fusion.py
│   │
│   ├── reporting/
│   │   ├── json_report.py           # JSON forensic report builder
│   │   ├── pdf_report.py            # PDF forensic report (FPDF2)
│   │   └── mitre_mapper.py          # MITRE mapping loader
│   │
│   ├── dashboard/
│   │   ├── app.py                   # Streamlit entry point + navigation routing
│   │   ├── db.py                    # SQLite helpers (WAL mode, init, log, query)
│   │   ├── health.py                # Module health + system stats
│   │   ├── theme.py                 # Custom Streamlit theme + CSS helpers
│   │   ├── state.py                 # session_state key constants + helpers
│   │   └── pages/
│   │       ├── home.py              # KPI cards, activity chart, module status
│   │       ├── upload.py            # File uploader, visualisation, histogram
│   │       ├── detection.py         # Run inference, probability chart, MITRE mapping
│   │       ├── gallery.py           # Dataset gallery and MITRE context
│   │       ├── training.py          # Model training UI
│   │       ├── dynamic.py           # Dynamic Analysis page
│   │       ├── hybrid.py            # Hybrid Verdict page
│   │       └── digital_twin.py      # Digital Twin design + prerequisites
│   │
│   └── training_manager.py          # Subprocess-based training runner
│
├── scripts/
│   ├── __init__.py
│   ├── train.py                     # Full training pipeline (validate → train → evaluate)
│   ├── evaluate.py                  # Test-set evaluation only (no retraining)
│   ├── convert_binary.py            # Convert a single binary to PNG via CLI
│   ├── generate_data.py             # Synthetic binary generator
│   ├── train_dynamic.py             # Train the M9 behavioural model
│   ├── generate_flows.py            # Synthetic Zeek conn.log for demos
│   ├── evaluate_unseen.py           # Leave-families-out variant experiment
│   └── benchmark.py                 # SRS performance measurements
│
├── data/
│   ├── malimg/                      # ← Download dataset here (not in git)
│   ├── processed/                   # class_names.json, eval_metrics.json, confusion_matrix.png (generated)
│   ├── logs/                        # Sample SQLite DB snapshot (optional)
│   │   └── maltwin.db
│   ├── iot23/                       # ← IoT-23 conn.log files (not in git)
│   ├── mitre_ics_mapping.json       # MITRE ATT&CK ICS per malware family
│   └── mitre_ics_behaviour_mapping.json  # MITRE ATT&CK ICS per behaviour
│
├── models/
│   ├── best_model.pt                # Best checkpoint by val accuracy (generated)
│   └── checkpoints/                 # Per-epoch checkpoints (generated)
│
├── logs/                            # Runtime SQLite detection log (configurable)
├── reports/                         # Generated PDF/JSON reports (configurable)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Shared pytest fixtures
    ├── test_converter.py
    ├── test_dataset.py
    ├── test_enhancement.py
    ├── test_model.py
    ├── test_db.py
    ├── test_gallery.py
    ├── test_gradcam.py
    ├── test_health.py
    ├── test_reporting.py
    ├── test_training_manager.py
    ├── test_integration.py
    ├── test_evaluation_extended.py
    ├── test_dynamic.py
    ├── test_fusion.py
    └── fixtures/
        ├── sample_pe.exe            # Minimal valid PE (1024 bytes, MZ header)
        └── sample_elf               # Minimal valid ELF (1024 bytes, \x7fELF header)
```

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Python | 3.11.x | 3.11.9 |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 100 GB free |
| GPU | None (CPU works) | NVIDIA GPU, 6+ GB VRAM, CUDA 12.x |

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/maltwin.git
cd maltwin
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training with CUDA 12.x, replace the PyTorch lines with the CUDA-enabled wheel:

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change paths, batch size, epochs, etc.
# The defaults work out of the box for most setups.
```

### 5. Download the Malimg dataset

The Malimg dataset is not included in this repository. Download it from Kaggle:

1. Go to [Malimg Dataset on Kaggle](https://www.kaggle.com/datasets/ang3loliveira/malimg-dataset-25-malware-families)
2. Download and extract so the folder structure looks like this:

```
data/malimg/
├── Adialer.C/
│   ├── 00a5c6a6c2a56253e1d1cbfe14561c78.png
│   └── ...
├── Agent.FYI/
├── Allaple.A/
├── ...
└── Yuner.A/
```

Each subfolder name is a malware family. Each file is a greyscale PNG.

---

## Quickstart

### Train the model

```bash
python scripts/train.py
```

This validates the dataset, builds DataLoaders with weighted oversampling, trains MalTwinCNN for 30 epochs, evaluates on the test split, and saves:

- `models/best_model.pt` — best checkpoint by validation accuracy
- `data/processed/class_names.json` — class index mapping for the dashboard
- `data/processed/eval_metrics.json` — test metrics for the dashboard KPI cards
- `data/processed/confusion_matrix.png` — 25×25 confusion matrix heatmap

A 2-epoch smoke run to verify everything works before committing to a full training run:

```bash
python scripts/train.py --epochs 2 --workers 0
```

### Launch the dashboard

```bash
streamlit run modules/dashboard/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Evaluate a trained model (without retraining)

```bash
python scripts/evaluate.py
```

Use `--save-metrics` to overwrite `eval_metrics.json`:

```bash
python scripts/evaluate.py --save-metrics
```

### Dynamic analysis model (IoT-23)

Download the IoT-23 *lighter* archive (labelled `conn.log.labeled` files) from the Stratosphere Lab and extract it under `data/iot23/`. Any subset of scenarios works.

```bash
python scripts/train_dynamic.py                 # random stratified split
python scripts/train_dynamic.py --split capture # hold out whole captures (harder)
python scripts/train_dynamic.py --synthetic 6000  # no IoT-23: demo model, clearly labelled as synthetic
```

### Extra evaluation

```bash
python scripts/evaluate_unseen.py --holdout 5   # unseen-family (variant) experiment
python scripts/benchmark.py                     # SRS performance requirements
```

### Convert a binary to a PNG image

```bash
python scripts/convert_binary.py --input suspicious.exe --output output.png
```

### Utility scripts

Generate synthetic test binaries into `MALTWIN_DATA_DIR`:

```bash
python scripts/generate_data.py --families Mirai_Synthetic Gafgyt_Synthetic Benign_IoT --count 50
```

Validate MITRE mappings against `data/processed/class_names.json`:

```bash
python verify_mitre.py
```

---

## Training Options

All training hyperparameters can be set via `.env` or CLI flags:

```
python scripts/train.py --help

Options:
  --data-dir    PATH   Path to Malimg dataset root      [default: data/malimg]
  --epochs      INT    Number of training epochs         [default: 30]
  --lr          FLOAT  Adam learning rate                [default: 0.001]
  --batch-size  INT    Batch size                        [default: 32]
  --workers     INT    DataLoader workers                [default: 4]
  --oversample  STR    Oversampling strategy             [default: oversample_minority]
                       Choices: oversample_minority | sqrt_inverse | uniform
  --no-augment         Disable training augmentation     [flag]
  --seed        INT    Random seed                       [default: 42]
```

---

## Model Architecture — MalTwinCNN

```
Input: (batch, 1, 128, 128)   ← single-channel grayscale

ConvBlock 1:  Conv(1→32)  → BN → ReLU → Conv(32→32)  → BN → ReLU → MaxPool → Dropout2d
ConvBlock 2:  Conv(32→64) → BN → ReLU → Conv(64→64)  → BN → ReLU → MaxPool → Dropout2d
ConvBlock 3:  Conv(64→128)→ BN → ReLU → Conv(128→128)→ BN → ReLU → MaxPool → Dropout2d
              ↑ self.gradcam_layer = block3.conv2 (Grad-CAM hook target)

AdaptiveAvgPool2d(4, 4)
Flatten → Linear(2048→512) → ReLU → Dropout(0.5) → Linear(512→25)

Output: (batch, 25)   ← raw logits, softmax applied at inference only
```

**1,348,793 parameters (≈1.35 M).** Trained with Adam + ReduceLROnPlateau, CrossEntropyLoss, weighted random oversampling to handle the 36:1 class imbalance in Malimg (Allaple.A has 2,949 samples; Skintrim.N has 80).

---

## Running Tests

```bash
# All unit tests (no dataset or trained model required)
pytest tests/ -v -m "not integration"

# A specific phase
pytest tests/test_converter.py -v
pytest tests/test_dataset.py -v -m "not integration"
pytest tests/test_model.py -v
pytest tests/test_db.py -v

# Full suite including integration tests (requires Malimg at data/malimg/)
pytest tests/ -v

# With coverage
pytest tests/ -m "not integration" --cov=modules --cov-report=term-missing
```

Integration tests are marked `@pytest.mark.integration` and require the Malimg dataset. All other tests use synthetic data and run in CI without any dataset.

---

## Documentation & References

- **Implement.md** — implementation walkthrough, design notes, and test execution logs
- **SRS_COMPLIANCE.md** — requirements traceability matrix against the SRS
- **MALTWIN_SRS.md** — full Software Requirements Specification document
- **MalTwin_Testing_Documentation.docx** — formal testing documentation and results

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards (total analyzed, malware count, model accuracy), 7-day activity chart, detection history with filters + CSV export, system stats, module status |
| 📂 Binary Upload | Upload `.exe`, `.dll`, or `.elf`, validates magic bytes, 50 MB limit, shows metadata table + pixel histogram |
| 🔍 Malware Detection | Run CNN inference on the uploaded binary, shows predicted family with confidence bar (green/amber/red), top-3 predictions, full 25-class probability chart, MITRE ATT&CK for ICS mapping, PDF/JSON report export |
| 🖼️ Dataset Gallery | Per-family gallery with MITRE context expander |
| 📡 Dynamic Analysis | Upload a Zeek `conn.log` (or generate a labelled synthetic demo capture); capture verdict, behaviour profile, explanation, MITRE ICS techniques, most suspicious flows, JSON export |
| ⚖️ Hybrid Verdict | Fuse the latest static detection with the latest dynamic report; agreement category, combined MITRE table, JSON export; what-if simulator |
| 🏋️ Model Training | Configure and run training from the dashboard with live logs and progress |
| 🖥️ Digital Twin | Architecture, planned nodes, interface contract and live prerequisite checks — implementation scheduled for the final evaluation |

**ELF upload note:** Linux/IIoT ELF binaries often have no extension. Rename them to `.elf` before uploading so the browser file picker will accept them. Validation still uses magic bytes, not the extension.

---

## Configuration Reference

Environment variables in `.env` (copied from `.env.example`) control paths and training settings. `config.py` reads them at startup and also defines runtime constants.

| Variable | Default | Description |
|----------|---------|-------------|
| `MALTWIN_DATA_DIR` | `./data/malimg` | Malimg dataset root |
| `MALTWIN_PROCESSED_DIR` | `./data/processed` | Processed artifacts (class_names, metrics, confusion matrix) |
| `MALTWIN_MODEL_DIR` | `./models` | Model checkpoints and best model path |
| `MALTWIN_LOG_DIR` | `./logs` | SQLite detection database directory |
| `MALTWIN_REPORTS_DIR` | `./reports` | PDF/JSON report output directory (config.py fallback: `./data/reports`) |
| `MALTWIN_IMG_SIZE` | `128` | Output image size (N×N) |
| `MALTWIN_IMAGE_LAYOUT` | `malimg` | Byte layout before resize: `malimg` (Nataraj width table, matches training data) or `square` |
| `MALTWIN_DYNAMIC_DATA_DIR` | `./data/iot23` | IoT-23 conn.log root for the dynamic model |
| `MALTWIN_FUSION_STATIC_WEIGHT` | `0.5` | Fusion weight for the CNN |
| `MALTWIN_FUSION_DYNAMIC_WEIGHT` | `0.5` | Fusion weight for the behavioural model |
| `MALTWIN_FUSION_MALICIOUS` | `0.70` | Fused score for a MALICIOUS verdict |
| `MALTWIN_FUSION_SUSPICIOUS` | `0.40` | Fused score for a SUSPICIOUS verdict |
| `MALTWIN_BATCH_SIZE` | `32` | Training batch size |
| `MALTWIN_EPOCHS` | `30` | Training epochs |
| `MALTWIN_LR` | `0.001` | Adam learning rate |
| `MALTWIN_WEIGHT_DECAY` | `0.0001` | Adam weight decay |
| `MALTWIN_LR_PATIENCE` | `5` | ReduceLROnPlateau patience |
| `MALTWIN_NUM_WORKERS` | `4` | DataLoader workers |
| `MALTWIN_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0` |
| `MALTWIN_TRAIN_RATIO` | `0.70` | Train split fraction |
| `MALTWIN_VAL_RATIO` | `0.15` | Validation split fraction |
| `MALTWIN_TEST_RATIO` | `0.15` | Test split fraction |
| `MALTWIN_OVERSAMPLE_STRATEGY` | `oversample_minority` | `oversample_minority`, `sqrt_inverse`, `uniform` |
| `MALTWIN_RANDOM_SEED` | `42` | Global random seed |
| `MALTWIN_CONFIDENCE_GREEN` | `0.80` | Green confidence threshold for UI |
| `MALTWIN_CONFIDENCE_AMBER` | `0.50` | Amber confidence threshold for UI |

Additional runtime constants (defined in `config.py`):

- `MAX_UPLOAD_BYTES` — 50 MB upload limit
- `ACCEPTED_EXTENSIONS` — `.exe`, `.dll`, `.elf`
- `STREAMLIT_PORT` — `8501`
- `DASHBOARD_TITLE` — `MalTwin — IIoT Malware Detection`
- `MALIMG_EXPECTED_FAMILIES` — `25`
- `MALIMG_TOTAL_SAMPLES` — `9339`

---

## Project Status

| Module | Status | Notes |
|--------|--------|-------|
| M2 — Binary-to-Image Conversion | ✅ Complete | |
| M3 — Dataset & Preprocessing | ✅ Complete | Requires Malimg download |
| M4 — Data Enhancement & Balancing | ✅ Complete | |
| M5 — Malware Detection CNN | ✅ Complete | Requires training run |
| M6 — Dashboard & Visualization | ✅ Complete | |
| M1 — Digital Twin Simulation | 🟡 Designed | Architecture and interface fixed; implementation for final evaluation (Docker networking; Mininet optional) |
| M7 — Explainable AI (Grad-CAM) | ✅ Complete | Captum heatmap pipeline with overlay export |
| M8 — Automated Threat Reporting | ✅ Complete | PDF + JSON export with MITRE mapping |
| M9 — Dynamic Behavioural Analysis | ✅ Complete | Requires IoT-23 conn.log files for a real (non-demo) model |
| M10 — Hybrid Static–Dynamic Fusion | ✅ Logic complete | Paired evaluation needs the Digital Twin to run samples and capture their traffic |

---

## Acknowledgements

- **Malimg Dataset** — Nataraj et al., "Malware Images: Visualization and Automatic Classification" (2011)
- **IoT-23 Dataset** — Garcia, Parmisano & Erquiaga, Stratosphere Laboratory, CTU (2020)
- **MITRE ATT&CK for ICS** — MITRE Corporation

---

*COMSATS University Islamabad | BS Cyber Security 2023–2027*
