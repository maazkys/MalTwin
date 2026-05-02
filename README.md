# MalTwin — IIoT Malware Detection Framework

> AI-powered malware classification for Industrial IoT environments using binary visualisation and a custom CNN.  
> BS Cyber Security Capstone Project — COMSATS University Islamabad, 2023–2027

---

## What is MalTwin?

MalTwin detects and classifies malware targeting Industrial IoT (IIoT) systems. It takes a PE or ELF binary file, converts its raw bytes into a grayscale image that captures the binary's structural patterns, then passes that image through a trained Convolutional Neural Network to classify it into one of **25 known malware families** from the Malimg dataset.

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
- **25-family CNN classifier** — 3-block ConvNet trained on the Malimg dataset with weighted oversampling to handle class imbalance
- **Confidence-coded results** — green / amber / red UI based on prediction confidence thresholds
- **MITRE ATT&CK for ICS mapping** — tactics and techniques for each detected malware family
- **Detection history** — all events logged to SQLite with timestamp, SHA-256, confidence, and device used
- **Grad-CAM XAI** — heatmap overlays that highlight which byte regions drove a prediction
- **Forensic reporting** — PDF + JSON report downloads with MITRE mappings
- **Streamlit dashboard** — six-page web UI (overview, upload, detection, gallery, training, digital twin stub)
- **CLI tools** — standalone scripts for training, evaluation, and binary conversion

---

## Repository Structure

```
maltwin/
├── config.py                        # Central config — all paths and hyperparameters
├── requirements.txt
├── .env.example                     # Copy to .env and edit before running
│
├── modules/
│   ├── binary_to_image/
│   │   ├── converter.py             # BinaryConverter — bytes → 128×128 numpy array
│   │   └── utils.py                 # validate_binary_format, compute_sha256, histogram
│   │
│   ├── dataset/
│   │   ├── loader.py                # MalimgDataset (PyTorch), get_dataloaders()
│   │   └── preprocessor.py         # validate_dataset_integrity, encode_labels, etc.
│   │
│   ├── enhancement/
│   │   ├── augmentor.py             # get_train_transforms, get_val_transforms, GaussianNoise
│   │   └── balancer.py              # ClassAwareOversampler → WeightedRandomSampler
│   │
│   ├── detection/
│   │   ├── model.py                 # MalTwinCNN (ConvBlock × 3 + classifier)
│   │   ├── trainer.py               # train(), validate_epoch()
│   │   ├── evaluator.py             # evaluate(), plot_confusion_matrix()
│   │   └── inference.py             # load_model(), predict_single(), predict_batch()
│   │
│   └── dashboard/
│       ├── app.py                   # Streamlit entry point + navigation routing
│       ├── db.py                    # SQLite helpers (WAL mode, init, log, query)
│       ├── state.py                 # session_state key constants + helpers
│       └── pages/
│           ├── home.py              # KPI cards, activity chart, module status
│           ├── upload.py            # File uploader, visualisation, histogram
│           ├── detection.py         # Run inference, probability chart, MITRE mapping
│           ├── gallery.py           # Dataset gallery and MITRE context
│           ├── training.py          # Model training UI
│           └── digital_twin.py      # Stub (Module 1 deferred)
│
├── scripts/
│   ├── train.py                     # Full training pipeline (validate → train → evaluate)
│   ├── evaluate.py                  # Test-set evaluation only (no retraining)
│   └── convert_binary.py            # Convert a single binary to PNG via CLI
│
├── data/
│   ├── malimg/                      # ← Download dataset here (not in git)
│   ├── processed/                   # class_names.json, eval_metrics.json (generated)
│   └── mitre_ics_mapping.json       # Static MITRE ATT&CK ICS reference data
│
├── models/
│   ├── best_model.pt                # Best checkpoint by val accuracy (generated)
│   └── checkpoints/                 # Per-epoch checkpoints (generated)
│
├── logs/
│   └── maltwin.db                   # SQLite detection event log (generated)
│
└── tests/
    ├── conftest.py                  # Shared pytest fixtures
    ├── test_converter.py
    ├── test_dataset.py
    ├── test_enhancement.py
    ├── test_model.py
    ├── test_db.py
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

### Convert a binary to a PNG image

```bash
python scripts/convert_binary.py --input suspicious.exe --output output.png
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

**~3.2M parameters.** Trained with Adam + ReduceLROnPlateau, CrossEntropyLoss, weighted random oversampling to handle the 36:1 class imbalance in Malimg (Allaple.A has 2,949 samples; Skintrim.N has 80).

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

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Dashboard | KPI cards (total analyzed, malware count, model accuracy), 7-day activity chart, recent detections feed, module status table |
| 📂 Binary Upload | Upload `.exe` or `.dll`, validates format, converts to 128×128 greyscale image, displays metadata table and pixel intensity histogram |
| 🔍 Malware Detection | Run CNN inference on the uploaded binary, shows predicted family with confidence bar (green/amber/red), top-3 predictions, full 25-class probability chart, MITRE ATT&CK for ICS mapping, PDF/JSON report export |
| 🖼️ Dataset Gallery | Per-family gallery with MITRE context expander |
| 🏋️ Model Training | Configure and run training from the dashboard |
| 🖥️ Digital Twin | Stub page — Module 1 (Docker/Mininet IIoT simulation) is deferred to a future sprint |

---

## Configuration Reference

All settings live in `.env` (copied from `.env.example`). `config.py` reads them at startup.

| Variable | Default | Description |
|----------|---------|-------------|
| `MALTWIN_DATA_DIR` | `./data/malimg` | Malimg dataset root |
| `MALTWIN_IMG_SIZE` | `128` | Output image size (N×N) |
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

---

## Project Status

| Module | Status | Notes |
|--------|--------|-------|
| M2 — Binary-to-Image Conversion | ✅ Complete | |
| M3 — Dataset & Preprocessing | ✅ Complete | Requires Malimg download |
| M4 — Data Enhancement & Balancing | ✅ Complete | |
| M5 — Malware Detection CNN | ✅ Complete | Requires training run |
| M6 — Dashboard & Visualization | ✅ Complete | |
| M1 — Digital Twin Simulation | ⚠️ Deferred | Requires Docker + Mininet infrastructure |
| M7 — Explainable AI (Grad-CAM) | ✅ Complete | Captum heatmap pipeline with overlay export |
| M8 — Automated Threat Reporting | ✅ Complete | PDF + JSON export with MITRE mapping |

---

## Acknowledgements

- **Malimg Dataset** — Nataraj et al., "Malware Images: Visualization and Automatic Classification" (2011)
- **MITRE ATT&CK for ICS** — MITRE Corporation

---

*COMSATS University Islamabad | BS Cyber Security 2023–2027*
