#!/usr/bin/env python3
"""
Full MalTwin training pipeline.

Usage:
    python scripts/train.py [OPTIONS]

Options (all optional — defaults come from config.py / .env):
    --data-dir    PATH   Path to Malimg dataset root     [default: config.DATA_DIR]
    --epochs      INT    Number of training epochs        [default: config.EPOCHS]
    --lr          FLOAT  Learning rate                    [default: config.LR]
    --batch-size  INT    Batch size                       [default: config.BATCH_SIZE]
    --workers     INT    DataLoader worker processes      [default: config.NUM_WORKERS]
    --oversample  STR    Oversampling strategy            [default: config.OVERSAMPLE_STRATEGY]
                         Choices: oversample_minority | sqrt_inverse | uniform
    --no-augment         Disable training augmentation    [flag, default: augmentation ON]
    --seed        INT    Random seed                      [default: config.RANDOM_SEED]

Exit codes:
    0  success
    1  dataset not found or invalid
    2  training or evaluation error

Outputs (written to disk on success):
    models/best_model.pt                   ← best checkpoint by val_acc
    models/checkpoints/epoch_NNN_accX.pt   ← per-epoch checkpoints
    data/processed/class_names.json        ← ordered class name list for dashboard
    data/processed/eval_metrics.json       ← test-set metrics for dashboard KPI cards
    data/processed/confusion_matrix.png    ← confusion matrix heatmap
"""
import argparse
import json
import sys
import random
import torch
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
    parser = argparse.ArgumentParser(
        description="Train MalTwin CNN on the Malimg malware image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir',   type=str,   default=str(config.DATA_DIR),
                        help='Path to Malimg dataset root directory')
    parser.add_argument('--epochs',     type=int,   default=config.EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr',         type=float, default=config.LR,
                        help='Adam learning rate')
    parser.add_argument('--batch-size', type=int,   default=config.BATCH_SIZE,
                        help='Batch size for DataLoaders')
    parser.add_argument('--workers',    type=int,   default=config.NUM_WORKERS,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--oversample', type=str,   default=config.OVERSAMPLE_STRATEGY,
                        choices=['oversample_minority', 'sqrt_inverse', 'uniform'],
                        help='Class oversampling strategy for training loader')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable training augmentation (use val transforms for train)')
    parser.add_argument('--seed',       type=int,   default=config.RANDOM_SEED,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Seed everything ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    import config

    # ── 2. Validate dataset ────────────────────────────────────────────────────
    from modules.dataset.preprocessor import validate_dataset_integrity
    print("=" * 55)
    print("MalTwin Training Pipeline")
    print("=" * 55)
    print("\n[1/6] Validating dataset...")
    try:
        report = validate_dataset_integrity(Path(args.data_dir))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Families found:   {len(report['families'])}")
    print(f"  Total samples:    {report['total']}")
    print(f"  Imbalance ratio:  {report['imbalance_ratio']:.1f}x "
          f"({report['max_class']} vs {report['min_class']})")
    if report['corrupt_files']:
        print(f"  WARNING: {len(report['corrupt_files'])} corrupt file(s) found — skipping")

    # ── 3. Build DataLoaders ───────────────────────────────────────────────────
    print("\n[2/6] Building DataLoaders...")
    try:
        from modules.dataset.loader import get_dataloaders
        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            data_dir=Path(args.data_dir),
            img_size=config.IMG_SIZE,
            batch_size=args.batch_size,
            num_workers=args.workers,
            oversample_strategy=args.oversample,
            augment_train=not args.no_augment,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"ERROR building DataLoaders: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"  Train batches:  {len(train_loader)}")
    print(f"  Val batches:    {len(val_loader)}")
    print(f"  Test batches:   {len(test_loader)}")
    print(f"  Classes:        {len(class_names)}")
    print(f"  Augmentation:   {'OFF (--no-augment)' if args.no_augment else 'ON'}")
    print(f"  Oversample:     {args.oversample}")

    # ── 4. Build model ─────────────────────────────────────────────────────────
    print("\n[3/6] Initialising model...")
    try:
        from modules.detection.model import MalTwinCNN
        model = MalTwinCNN(num_classes=len(class_names)).to(config.DEVICE)
    except Exception as e:
        print(f"ERROR initialising model: {e}", file=sys.stderr)
        sys.exit(2)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture:       MalTwinCNN")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Device:             {config.DEVICE}")

    # ── 5. Train ───────────────────────────────────────────────────────────────
    print(f"\n[4/6] Training for {args.epochs} epoch(s)...")
    try:
        from modules.detection.trainer import train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.DEVICE,
            epochs=args.epochs,
            lr=args.lr,
        )
    except Exception as e:
        print(f"ERROR during training: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"\nTraining complete.")
    print(f"  Best val accuracy: {history['best_val_acc']:.4f} at epoch {history['best_epoch']}")
    print(f"  Model saved to:    {config.BEST_MODEL_PATH}")

    # ── 6. Evaluate on test set ────────────────────────────────────────────────
    print("\n[5/6] Evaluating best model on test set...")
    try:
        from modules.detection.inference import load_model
        from modules.detection.evaluator import evaluate, format_metrics_table, plot_confusion_matrix
        best_model = load_model(config.BEST_MODEL_PATH, len(class_names), config.DEVICE)
        metrics = evaluate(best_model, test_loader, config.DEVICE, class_names)
    except Exception as e:
        print(f"ERROR during evaluation: {e}", file=sys.stderr)
        sys.exit(2)

    print(format_metrics_table(metrics, class_names))

    # ── 7. Save eval metrics JSON ──────────────────────────────────────────────
    print("\n[6/6] Saving outputs...")
    try:
        metrics_path = config.EVAL_METRICS_PATH
        # confusion_matrix is np.ndarray — not JSON serialisable; exclude it
        # classification_report is a string — exclude (too verbose for dashboard)
        serialisable = {
            k: v for k, v in metrics.items()
            if k not in ('confusion_matrix', 'per_class', 'classification_report')
        }
        # per_class contains nested dicts with Python floats/ints — safe to serialise
        serialisable['per_class'] = {
            family: dict(stats) for family, stats in metrics['per_class'].items()
        }
        with open(metrics_path, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"  Eval metrics → {metrics_path}")
    except Exception as e:
        # Non-fatal: metrics file is nice-to-have for dashboard
        print(f"  WARNING: Could not save eval metrics: {e}", file=sys.stderr)

    # ── 8. Save confusion matrix PNG ───────────────────────────────────────────
    try:
        cm_path = config.CONFUSION_MATRIX_PATH
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
        print(f"  Confusion matrix → {cm_path}")
    except Exception as e:
        print(f"  WARNING: Could not save confusion matrix: {e}", file=sys.stderr)

    print("\n" + "=" * 55)
    print("Done!")
    print(f"  Launch dashboard: streamlit run modules/dashboard/app.py")
    print("=" * 55)
    sys.exit(0)


if __name__ == '__main__':
    main()
