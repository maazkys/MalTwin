#!/usr/bin/env python3
"""
Unseen-variant (leave-families-out) evaluation.

Trains a fresh MalTwinCNN on all but N families, then measures whether the
model's confidence drops on the N families it never saw. See
modules/detection/openset.py for the metric definitions.

This does NOT touch models/best_model.pt — the experiment model is written to
models/openset/ so the dashboard's main model is never overwritten.

Usage:
    python scripts/evaluate_unseen.py                      # 5 random families held out
    python scripts/evaluate_unseen.py --holdout 3 --epochs 15
    python scripts/evaluate_unseen.py --families Skintrim.N Wintrim.BX

Outputs:
    data/processed/unseen_family_eval.json
    data/processed/unseen_confidence_hist.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Leave-families-out evaluation of MalTwin on unseen variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-dir',   type=str, default=str(config.DATA_DIR))
    p.add_argument('--holdout',    type=int, default=5, help='Number of families to hold out')
    p.add_argument('--families',   nargs='*', default=None,
                   help='Explicit families to hold out (overrides --holdout)')
    p.add_argument('--epochs',     type=int, default=config.EPOCHS)
    p.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--workers',    type=int, default=config.NUM_WORKERS)
    p.add_argument('--seed',       type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def _max_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    maxp, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            maxp.append(prob.max(axis=1))
            preds.append(prob.argmax(axis=1))
            labels.append(y.numpy())
    return np.concatenate(maxp), np.concatenate(preds), np.concatenate(labels)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    from torch.utils.data import DataLoader, ConcatDataset
    from modules.dataset.loader import MalimgDataset
    from modules.enhancement.augmentor import get_train_transforms, get_val_transforms
    from modules.enhancement.balancer import ClassAwareOversampler
    from modules.detection.model import MalTwinCNN
    from modules.detection.trainer import train
    from modules.detection.openset import (
        select_holdout_families, openset_metrics, plot_confidence_histogram,
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: dataset not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_families = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    holdout = (sorted(args.families) if args.families
               else select_holdout_families(all_families, args.holdout, args.seed))
    known = [f for f in all_families if f not in holdout]
    print(f"Known families  ({len(known)}): {', '.join(known)}")
    print(f"Unseen families ({len(holdout)}): {', '.join(holdout)}")

    img = config.IMG_SIZE
    tr_t, va_t = get_train_transforms(img), get_val_transforms(img)
    mk = lambda split, t, fams: MalimgDataset(  # noqa: E731
        data_dir, split, img, t, random_seed=args.seed, include_families=fams)

    train_ds = mk('train', tr_t, known)
    val_ds   = mk('val',   va_t, known)
    test_ds  = mk('test',  va_t, known)
    unseen_ds = ConcatDataset([mk(s, va_t, holdout) for s in ('train', 'val', 'test')])

    sampler = ClassAwareOversampler(train_ds, strategy=config.OVERSAMPLE_STRATEGY).get_sampler()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, drop_last=True)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)
    unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size, num_workers=args.workers)

    out_dir = config.MODEL_DIR / 'openset'
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MalTwinCNN(num_classes=len(known)).to(config.DEVICE)
    history = train(model, train_loader, val_loader, config.DEVICE,
                    epochs=args.epochs,
                    checkpoint_dir=out_dir / 'checkpoints',
                    best_model_path=out_dir / 'best_model_openset.pt')

    state = torch.load(out_dir / 'best_model_openset.pt', map_location=config.DEVICE)
    model.load_state_dict(state)

    k_conf, k_pred, k_true = _max_probs(model, test_loader, config.DEVICE)
    u_conf, _, _ = _max_probs(model, unseen_loader, config.DEVICE)

    result = openset_metrics(k_conf, u_conf,
                             thresholds=(config.CONFIDENCE_AMBER, config.CONFIDENCE_GREEN))
    result.update({
        'known_families':      known,
        'unseen_families':     holdout,
        'known_test_accuracy': float((k_pred == k_true).mean()),
        'best_val_acc':        float(history['best_val_acc']),
        'epochs':              args.epochs,
        'seed':                args.seed,
    })

    with open(config.UNSEEN_EVAL_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    hist_path = config.PROCESSED_DIR / 'unseen_confidence_hist.png'
    plot_confidence_histogram(k_conf, u_conf, hist_path,
                              thresholds=(config.CONFIDENCE_AMBER, config.CONFIDENCE_GREEN))

    print("\n=== Unseen-variant evaluation ===")
    print(f"Known-family test accuracy : {result['known_test_accuracy']:.4f}")
    print(f"AUROC known vs unseen      : {result['auroc_known_vs_unknown']:.4f}")
    print(f"FPR @ 95% TPR              : {result['fpr_at_95_tpr']:.4f}")
    print(f"Mean confidence known/unseen: {result['mean_conf_known']:.3f} / "
          f"{result['mean_conf_unknown']:.3f}")
    for t, row in result['thresholds'].items():
        print(f"  threshold {t}: unseen flagged {row['unknown_flagged_rate']:.1%}, "
              f"known wrongly flagged {row['known_flagged_rate']:.1%}")
    print(f"\nSaved → {config.UNSEEN_EVAL_PATH}\n        {hist_path}")


if __name__ == '__main__':
    main()
