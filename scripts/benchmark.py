#!/usr/bin/env python3
"""
Measure MalTwin's SRS performance requirements on the current machine.

    PER-1  binary → image conversion   < 3 s (≤10 MB), < 10 s (≤50 MB)
    PER-2  CNN inference               < 5 s CPU
    PER-4  Grad-CAM heatmap            < 8 s CPU
    PER-5  PDF + JSON report           < 10 s
    PER-6  SQLite query over 100k rows < 500 ms
    (+)    dynamic analysis of a capture and fusion — new modules M9/M10
    REL-1  identical inference results across 10 runs

Timings depend on model weights only marginally (same compute graph), so an
untrained model is used if models/best_model.pt does not exist.

Usage:
    python scripts/benchmark.py            # writes data/processed/performance.json
    python scripts/benchmark.py --runs 10
"""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def timeit(fn, runs):
    times = []
    out = None
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return out, {'mean_s': statistics.mean(times), 'max_s': max(times), 'runs': runs}


def fake_pe(size: int) -> bytes:
    rng = np.random.default_rng(0)
    return b'MZ' + rng.integers(0, 256, size - 2, dtype=np.uint8).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    args = ap.parse_args()
    R = args.runs
    res = {'machine': {'platform': platform.platform(), 'python': platform.python_version(),
                       'torch': torch.__version__, 'device': str(config.DEVICE),
                       'cpu_count': os.cpu_count()}}

    from modules.binary_to_image.converter import BinaryConverter
    conv = BinaryConverter(config.IMG_SIZE)
    for mb in (1, 10, 50):
        data = fake_pe(mb * 1024 * 1024)
        img, t = timeit(lambda: conv.convert(data), R)
        res[f'PER-1_convert_{mb}MB'] = t

    from modules.detection.model import MalTwinCNN
    from modules.detection.inference import predict_single, load_model
    names = [f'class_{i}' for i in range(25)]
    trained = config.BEST_MODEL_PATH.exists() and config.CLASS_NAMES_PATH.exists()
    if trained:
        from modules.dataset.preprocessor import load_class_names
        names = load_class_names(config.CLASS_NAMES_PATH)
        model = load_model(config.BEST_MODEL_PATH, len(names), config.DEVICE)
    else:
        torch.manual_seed(0)
        model = MalTwinCNN(num_classes=25).to(config.DEVICE).eval()
    res['model'] = 'trained best_model.pt' if trained else 'untrained (timing only)'

    pred, t = timeit(lambda: predict_single(model, img, names, config.DEVICE), max(R, 10))
    res['PER-2_inference'] = t
    confs = [predict_single(model, img, names, config.DEVICE)['confidence'] for _ in range(10)]
    res['REL-1_inference_max_abs_diff_10_runs'] = float(max(confs) - min(confs))

    from modules.detection.gradcam import generate_gradcam
    target = names.index(pred['predicted_family'])
    gc, t = timeit(lambda: generate_gradcam(model, img, target, config.DEVICE), R)
    res['PER-4_gradcam'] = t
    res['PER-4_gradcam_ok'] = gc is not None

    from modules.reporting import generate_pdf_report, generate_json_report
    report = {
        'file_name': 'bench.exe', 'sha256': '0' * 64, 'file_format': 'PE',
        'file_size_bytes': 1024, 'upload_time': '2026-01-01T00:00:00',
        'predicted_family': pred['predicted_family'], 'confidence': pred['confidence'],
        'top3': pred['top3'], 'all_probabilities': pred['probabilities'],
        'gradcam': {'generated': False}, 'mitre': {'tactics': [], 'techniques': [], 'description': ''},
    }
    _, t = timeit(lambda: (generate_pdf_report(report), generate_json_report(report)), R)
    res['PER-5_reports_pdf_json'] = t

    from modules.dashboard.db import init_db, get_connection
    with tempfile.TemporaryDirectory() as d:
        dbp = Path(d) / 'bench.db'
        init_db(dbp)
        with get_connection(dbp) as conn:
            conn.executemany(
                "INSERT INTO detection_events (timestamp, file_name, sha256, file_format, "
                "file_size, predicted_family, confidence, device_used) VALUES (?,?,?,?,?,?,?,?)",
                [(f'2026-01-{1 + i % 28:02d}T00:00:{i % 60:02d}', f'f{i}', f'{i:064d}', 'PE', 1024,
                  names[i % len(names)], 0.5, 'cpu') for i in range(100_000)])
            conn.commit()
        def q():
            with get_connection(dbp) as conn:
                return conn.execute(
                    "SELECT predicted_family, COUNT(*) FROM detection_events "
                    "WHERE timestamp >= '2026-01-15' GROUP BY predicted_family").fetchall()
        _, t = timeit(q, R)
        res['PER-6_sqlite_query_100k'] = t

    from modules.dynamic import DynamicDetector
    from modules.dynamic.synthetic import generate_flows
    from modules.dynamic.zeek_parser import map_behaviour
    train = generate_flows({'Benign': 1500, 'PortScan': 400, 'C&C': 150, 'DDoS': 150}, seed=1)
    train['behaviour'] = [map_behaviour(a, b) for a, b in zip(train['label'], train['detailed-label'])]
    det = DynamicDetector(n_estimators=100).fit(train)
    cap = generate_flows({'Benign': 9000, 'PortScan': 1000}, seed=2)
    _, t = timeit(lambda: det.analyse_capture(cap), R)
    res['M9_dynamic_capture_10k_flows'] = t

    from modules.fusion import fuse
    _, t = timeit(lambda: fuse(static_conf=0.7, dynamic_score=0.9), 1000)
    res['M10_fusion'] = t

    limits = {'PER-1_convert_10MB': 3, 'PER-1_convert_50MB': 10, 'PER-2_inference': 5,
              'PER-4_gradcam': 8, 'PER-5_reports_pdf_json': 10, 'PER-6_sqlite_query_100k': 0.5}
    res['verdicts'] = {k: ('PASS' if res[k]['max_s'] < v else 'FAIL') for k, v in limits.items()}
    res['verdicts']['REL-1'] = 'PASS' if res['REL-1_inference_max_abs_diff_10_runs'] < 0.005 else 'FAIL'

    out = config.PROCESSED_DIR / 'performance.json'
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    for k, v in res.items():
        if isinstance(v, dict) and 'mean_s' in v:
            print(f"{k:<34} mean {v['mean_s']*1000:9.1f} ms   max {v['max_s']*1000:9.1f} ms")
    print("\n".join(f"{k:<34} {v}" for k, v in res['verdicts'].items()))
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
