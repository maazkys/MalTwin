#!/usr/bin/env python3
"""
Evaluate best_model.pt on the test split only (no retraining).

Usage:
    python scripts/evaluate.py [OPTIONS]

Options:
    --model-path  PATH   Path to trained model .pt file   [default: config.BEST_MODEL_PATH]
    --data-dir    PATH   Path to Malimg dataset root       [default: config.DATA_DIR]
    --batch-size  INT    Batch size for test DataLoader    [default: config.BATCH_SIZE]
    --workers     INT    DataLoader worker processes       [default: config.NUM_WORKERS]
    --seed        INT    Random seed (affects split)       [default: config.RANDOM_SEED]
    --save-metrics       Save eval_metrics.json to data/processed/ [flag]

Exit codes:
    0  success
    1  model file or dataset not found
    2  evaluation error

Use case:
    Re-evaluate after code changes or hyperparameter review without retraining.
    Produces the same test split as training (same seed and split ratios).
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MalTwin model on the Malimg test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model-path',   type=str, default=str(config.BEST_MODEL_PATH),
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--data-dir',     type=str, default=str(config.DATA_DIR),
                        help='Path to Malimg dataset root directory')
    parser.add_argument('--batch-size',   type=int, default=config.BATCH_SIZE,
                        help='Batch size for test DataLoader')
    parser.add_argument('--workers',      type=int, default=config.NUM_WORKERS,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--seed',         type=int, default=config.RANDOM_SEED,
                        help='Random seed (must match training seed for same test split)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save eval_metrics.json to data/processed/')
    return parser.parse_args()


def main():
    args = parse_args()
    import config

    model_path = Path(args.model_path)
    data_dir   = Path(args.data_dir)

    # ── 1. Validate model file exists ─────────────────────────────────────────
    if not model_path.exists():
        print(
            f"ERROR: Model file not found: {model_path}\n"
            "Run scripts/train.py first to produce best_model.pt",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 2. Validate dataset exists ────────────────────────────────────────────
    if not data_dir.exists():
        print(
            f"ERROR: Dataset directory not found: {data_dir}\n"
            "Download Malimg from Kaggle and extract to data/malimg/",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 3. Load class names ───────────────────────────────────────────────────
    print("Loading class names...")
    try:
        from modules.dataset.preprocessor import load_class_names
        class_names = load_class_names(config.CLASS_NAMES_PATH)
        print(f"  {len(class_names)} classes loaded from {config.CLASS_NAMES_PATH}")
    except FileNotFoundError:
        # Fall back to scanning the dataset directory
        print("  class_names.json not found — scanning dataset directory...")
        from modules.dataset.preprocessor import encode_labels
        subdirs = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
        label_map = encode_labels(subdirs)
        class_names = sorted(label_map.keys())
        print(f"  {len(class_names)} classes found in dataset")

    # ── 4. Build test DataLoader ──────────────────────────────────────────────
    print("Building test DataLoader...")
    try:
        from modules.dataset.loader import MalimgDataset
        from modules.enhancement.augmentor import get_val_transforms
        from torch.utils.data import DataLoader

        test_ds = MalimgDataset(
            data_dir=data_dir,
            split='test',
            img_size=config.IMG_SIZE,
            transform=get_val_transforms(config.IMG_SIZE),
            random_seed=args.seed,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        print(f"  Test samples:  {len(test_ds)}")
        print(f"  Test batches:  {len(test_loader)}")
    except Exception as e:
        print(f"ERROR building test DataLoader: {e}", file=sys.stderr)
        sys.exit(2)

    # ── 5. Load model ─────────────────────────────────────────────────────────
    print(f"Loading model from {model_path}...")
    try:
        from modules.detection.inference import load_model
        model = load_model(
            model_path=model_path,
            num_classes=len(class_names),
            device=config.DEVICE,
        )
        print(f"  Device: {config.DEVICE}")
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    try:
        from modules.detection.evaluator import evaluate, format_metrics_table
        metrics = evaluate(model, test_loader, config.DEVICE, class_names)
    except Exception as e:
        print(f"ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(2)

    # ── 7. Print results ──────────────────────────────────────────────────────
    print(format_metrics_table(metrics, class_names))
    print("\nClassification Report:")
    print(metrics['classification_report'])

    # ── 8. Optionally save metrics ────────────────────────────────────────────
    if args.save_metrics:
        try:
            metrics_path = config.EVAL_METRICS_PATH
            serialisable = {
                k: v for k, v in metrics.items()
                if k not in ('confusion_matrix', 'per_class', 'classification_report')
            }
            serialisable['per_class'] = {
                family: dict(stats) for family, stats in metrics['per_class'].items()
            }
            with open(metrics_path, 'w') as f:
                json.dump(serialisable, f, indent=2)
            print(f"Eval metrics saved to {metrics_path}")
        except Exception as e:
            print(f"WARNING: Could not save eval metrics: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == '__main__':
    main()
