#!/usr/bin/env python3
"""
Train the dynamic (behavioural) detector on Zeek conn.log files.

Data
  IoT-23 (Stratosphere Lab, CTU): download the "lighter" archive
  (labelled conn.log files only) and extract it so that
      data/iot23/**/conn.log.labeled
  exist. Any subset of scenarios works; more scenarios = more behaviours.

Usage
  python scripts/train_dynamic.py                         # all files under data/iot23
  python scripts/train_dynamic.py --logs path/a/conn.log.labeled path/b/conn.log.labeled
  python scripts/train_dynamic.py --split capture         # test on unseen captures
  python scripts/train_dynamic.py --synthetic 6000        # no IoT-23: synthetic demo model

Outputs
  models/dynamic_model.joblib
  data/processed/dynamic_metrics.json
  data/processed/dynamic_confusion_matrix.png
  data/processed/dynamic_feature_importance.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train MalTwin dynamic detector",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--logs', nargs='*', default=None,
                   help='conn.log(.labeled) files; default: search MALTWIN_DYNAMIC_DATA_DIR')
    p.add_argument('--data-dir', default=str(config.DYNAMIC_DATA_DIR))
    p.add_argument('--max-rows-per-file', type=int, default=200_000,
                   help='Uniform sample per file (IoT-23 files can hold millions of flows)')
    p.add_argument('--split', choices=['random', 'capture'], default='random',
                   help="'capture' holds out whole captures — a harder, more honest test")
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--min-class-flows', type=int, default=10,
                   help='Drop behaviour classes with fewer flows than this')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--synthetic', type=int, default=0,
                   help='Train on N synthetic flows instead of IoT-23 (demo only)')
    p.add_argument('--seed', type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def _plot_cm(cm, labels, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    t = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > t else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Dynamic detector — behaviour confusion matrix')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def _plot_importance(rows, path, top=15):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = rows[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([r['feature'] for r in rows], [r['importance'] for r in rows], color='#388BFD')
    ax.set_xlabel('Random-forest importance (mean impurity decrease)')
    ax.set_title(f'Top {top} behavioural features')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    args = parse_args()
    from modules.dynamic import load_conn_logs, map_behaviour, DynamicDetector
    from modules.dynamic.synthetic import generate_flows

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.synthetic:
        n = args.synthetic
        counts = {'Benign': int(n * .5), 'PortScan': int(n * .15), 'C&C': int(n * .08),
                  'DDoS': int(n * .08), 'Attack': int(n * .07), 'FileDownload': int(n * .05),
                  'Okiru': int(n * .07)}
        df = generate_flows(counts, seed=args.seed)
        df['behaviour'] = [map_behaviour(a, b) for a, b in zip(df['label'], df['detailed-label'])]
        df['capture'] = 'synthetic'
        source = 'synthetic'
        print(f"[1/4] Generated {len(df):,} SYNTHETIC flows (demo model, not IoT-23)")
    else:
        paths = args.logs or sorted(
            str(p) for p in Path(args.data_dir).rglob('conn.log*') if p.is_file())
        if not paths:
            print(f"ERROR: no conn.log files found under {args.data_dir}.\n"
                  f"Download IoT-23 (lighter version) and extract it there, or pass --logs, "
                  f"or use --synthetic N for a demo model.", file=sys.stderr)
            sys.exit(1)
        print(f"[1/4] Loading {len(paths)} conn.log file(s) "
              f"(≤{args.max_rows_per_file:,} flows each)...")
        df = load_conn_logs(paths, max_rows_per_file=args.max_rows_per_file, seed=args.seed)
        if 'behaviour' not in df.columns:
            print("ERROR: files are not labelled (no label/detailed-label columns)", file=sys.stderr)
            sys.exit(1)
        source = 'IoT-23'

    counts = df['behaviour'].value_counts()
    keep = counts[counts >= args.min_class_flows].index
    dropped = counts[counts < args.min_class_flows]
    df = df[df['behaviour'].isin(keep)].reset_index(drop=True)
    print("  Behaviour distribution:")
    for k, v in df['behaviour'].value_counts().items():
        print(f"    {k:<14} {v:>9,}")
    if len(dropped):
        print(f"  Dropped rare classes (<{args.min_class_flows} flows): {dict(dropped)}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    captures = df['capture'].unique()
    if args.split == 'capture' and len(captures) >= 2:
        rng = np.random.default_rng(args.seed)
        n_test = max(1, int(round(len(captures) * args.test_size)))
        test_caps = set(rng.choice(captures, n_test, replace=False))
        test_mask = df['capture'].isin(test_caps)
        train_df, test_df = df[~test_mask], df[test_mask]
        print(f"[2/4] Capture split — test captures: {sorted(test_caps)}")
    else:
        if args.split == 'capture':
            print("  (only one capture available — falling back to random split)")
        train_df, test_df = train_test_split(df, test_size=args.test_size,
                                             stratify=df['behaviour'], random_state=args.seed)
        print(f"[2/4] Stratified random split")
    print(f"  Train flows: {len(train_df):,} | Test flows: {len(test_df):,}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("[3/4] Training random forest...")
    det = DynamicDetector(n_estimators=args.n_estimators, random_state=args.seed)
    det.fit(train_df)

    # ── 4. Evaluate + save ────────────────────────────────────────────────────
    print("[4/4] Evaluating...")
    m = det.evaluate(test_df)
    print(m['classification_report'])
    b = m['binary']
    fmt = lambda v: 'n/a' if v is None else f"{v:.4f}"  # noqa: E731
    print(f"  Malicious-vs-benign: detection rate {fmt(b['detection_rate'])}, "
          f"FPR {fmt(b['false_positive_rate'])}, ROC-AUC {fmt(b['roc_auc'])}")

    meta = {'data_source': source, 'split': args.split, 'n_train': len(train_df),
            'n_test': len(test_df), 'classes': det.classes_, 'seed': args.seed}
    det.save(config.DYNAMIC_MODEL_PATH, metadata=meta)

    importance = det.feature_importance()
    out = {k: v for k, v in m.items() if k != 'classification_report'}
    out.update({'metadata': meta, 'feature_importance': importance[:20]})
    with open(config.DYNAMIC_METRICS_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    _plot_cm(m['confusion_matrix'], m['labels'], config.DYNAMIC_CM_PATH)
    _plot_importance(importance, config.DYNAMIC_IMPORTANCE_PATH)

    print(f"\nSaved model   → {config.DYNAMIC_MODEL_PATH}")
    print(f"Saved metrics → {config.DYNAMIC_METRICS_PATH}")
    print(f"Saved plots   → {config.DYNAMIC_CM_PATH.name}, {config.DYNAMIC_IMPORTANCE_PATH.name}")


if __name__ == '__main__':
    main()
