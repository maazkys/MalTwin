# Evaluation 3 Runbook

Everything you need to run before Sunday 18 October (document deadline) and for the demo on Monday 19 October.

---

## 0. Setup (once, ~15 min)

```bash
git clone https://github.com/imanfatimadocx/MalTwin.git
cd MalTwin
git checkout feature/eval3-60pct        # until the PR is merged
python3.11 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -m "not integration" -q   # expect: all passed, a few skipped
```

> If `pip install` fails on your Python version, install the latest compatible versions instead:
> `pip install torch torchvision captum opencv-python-headless pillow numpy scikit-learn pandas scipy streamlit plotly psutil fpdf2 python-dotenv tqdm matplotlib pytest joblib`.
> The code works on both the pinned and newer versions.

---

## 1. Train the CNN on Malimg (the most important step)

Put the dataset at `data/malimg/<Family>/*.png` (25 folders), then:

```bash
python scripts/train.py --epochs 2 --workers 0     # 2-minute smoke run first
python scripts/train.py                            # full run, 30 epochs
```

Time: roughly 1–2 hours on a laptop CPU, a few minutes on a GPU. Google Colab or Kaggle Notebooks (free GPU) work too: upload the repo and dataset, then run the same command.

Produces in `data/processed/`:
`eval_metrics.json`, `confusion_matrix.png`, `roc_curves.png`, `training_curves.png`, `training_history.json`, `class_names.json`, plus `models/best_model.pt`.

## 2. Unseen-variant experiment (the "variant detection" result)

```bash
python scripts/evaluate_unseen.py --holdout 5 --epochs 15
```

This trains a separate model on 20 families and holds 5 back completely. It reports whether confidence drops on families the model has never seen. It does **not** overwrite `best_model.pt`.

Produces `data/processed/unseen_family_eval.json` and `unseen_confidence_hist.png`.

## 3. Dynamic model on IoT-23

1. Download the **IoT-23 lighter version** (labelled `conn.log.labeled` files only) from the Stratosphere Laboratory website.
2. Extract it under `data/iot23/`. A few scenario folders are enough if bandwidth is limited. Pick ones with different malware (e.g. Mirai, Okiru, Torii, Hide-and-Seek) plus the honeypot (benign) captures.
3. Run:

```bash
python scripts/train_dynamic.py                     # random split
python scripts/train_dynamic.py --split capture     # harder: test on captures never seen in training
```

Run the `--split capture` one **last**, so its model and metrics are the ones kept. Copy the numbers from the first run before running the second, because both write the same files. Report both in the document.

Produces `models/dynamic_model.joblib`, `data/processed/dynamic_metrics.json`, `dynamic_confusion_matrix.png`, `dynamic_feature_importance.png`.

If IoT-23 isn't ready in time: `python scripts/train_dynamic.py --synthetic 6000` gives a **demo** model. The dashboard labels it as synthetic. Do not report its numbers as results.

## 4. Performance benchmark (on the demo laptop)

```bash
python scripts/benchmark.py
```

Produces `data/processed/performance.json` (SRS PER-1 … PER-6, REL-1).

---

## 5. Send these files back for Document 4

Zip and send everything in `data/processed/` **except** large files. The `.json` and `.png` files are all that's needed.

---

## 6. Screenshots for the document

Run `streamlit run modules/dashboard/app.py` and capture:

| # | Page | What should be visible |
|---|------|------------------------|
| 1 | Dashboard | KPI cards + module status table (10 modules) |
| 2 | Binary Upload | A binary uploaded: grayscale image, metadata, histogram |
| 3 | Malware Detection | Result card with confidence bar + probability chart |
| 4 | Malware Detection | Grad-CAM heatmap enabled |
| 5 | Malware Detection | MITRE table + report download buttons |
| 6 | Dynamic Analysis | A real IoT-23 capture analysed: verdict + behaviour profile + explanation |
| 7 | Hybrid Verdict | Static + dynamic fused result |
| 8 | Hybrid Verdict | What-if simulator showing NOVEL VARIANT |
| 9 | Digital Twin | Architecture diagram |
| 10 | Terminal | `pytest tests/ -m "not integration" -v` final summary line |
| 11 | Terminal | `python scripts/train.py` final metrics table |

For page 6, upload one of the IoT-23 `conn.log.labeled` files directly. The page shows ground-truth accuracy for labelled files.

---

## 7. Demo order for the evaluation (≈7 minutes)

1. **Dashboard:** 10 modules, what's active, what's scheduled (Digital Twin).
2. **Binary Upload → Malware Detection:** upload a sample, show family + confidence, Grad-CAM, MITRE, download the PDF.
3. **Unseen-variant result:** show `unseen_confidence_hist.png`. Point: "On families the model never saw, confidence drops, so the amber/red bar warns the analyst instead of a confident wrong answer."
4. **Dynamic Analysis:** upload an IoT-23 capture, explain the behaviour profile and *why* it was flagged.
5. **Hybrid Verdict:** fuse both, then switch to the what-if simulator: static low + dynamic high gives **NOVEL VARIANT**. That is the core idea of MalTwin.
6. **Digital Twin:** the design, its interface (Zeek conn.log, same as IoT-23, so the dynamic model plugs straight in), and the plan for the final evaluation.

### Likely questions and honest answers

- *"Is the fusion evaluated?"* The logic is unit-tested against a decision table. A paired evaluation, where the same binary's image and its traffic are scored together, needs the Digital Twin to run samples. That is the final-evaluation milestone.
- *"Malimg has no benign class. What happens with a clean file?"* The CNN always names a family. Low confidence is the signal, and the unseen-family experiment quantifies how well that works. Dynamic analysis adds the benign/malicious view.
- *"Why are IP addresses not features?"* Each IoT-23 capture has its own infected IP. Using IPs would teach the model "this address is bad" instead of "this behaviour is bad", and that wouldn't transfer to the Digital Twin's network.
- *"Why Docker instead of Mininet?"* Mininet needs a native Linux host with root. Docker internal networks give the same isolation and run on Windows too. Mininet remains an optional extension.
