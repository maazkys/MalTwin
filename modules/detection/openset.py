# modules/detection/openset.py
"""
Unseen-variant (open-set) evaluation helpers.

Motivation
----------
A closed-set classifier trained on 25 Malimg families will always answer with
one of those 25 names, even for a binary from a family it has never seen.
For IIoT defence the useful question is therefore not only "which family?"
but "does the model *know* it doesn't know?".

We measure this with a leave-families-out experiment:

    1. Hold out K families entirely (never seen in train/val).
    2. Train on the remaining families.
    3. Compare the model's top-1 confidence (max softmax probability) on
       - known-family test images   → should be high
       - held-out-family images     → should be low
    4. Treat "confidence below threshold" as an UNKNOWN-VARIANT flag.

Reported metrics
----------------
    auroc_known_vs_unknown : how well max-softmax separates known from unseen
                             (0.5 = random, 1.0 = perfect)
    fpr_at_95_tpr          : fraction of unseen samples still accepted as
                             "known" when 95% of known samples are accepted
    per-threshold table    : at the dashboard's amber (0.50) and green (0.80)
                             thresholds — % unseen flagged, % known wrongly flagged

These thresholds are the same ones that colour the confidence bar on the
detection page, so the experiment directly measures what an analyst sees.
"""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def select_holdout_families(
    families: Iterable[str],
    n_holdout: int,
    seed: int = 42,
) -> list[str]:
    """Deterministically choose `n_holdout` families to keep unseen."""
    fams = sorted(families)
    if not 0 < n_holdout < len(fams):
        raise ValueError(
            f"n_holdout must be between 1 and {len(fams) - 1}, got {n_holdout}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(fams, n_holdout))


def openset_metrics(
    known_max_prob: np.ndarray,
    unknown_max_prob: np.ndarray,
    thresholds: Iterable[float] = (0.5, 0.8),
) -> dict:
    """
    Compute known-vs-unknown separation metrics from top-1 confidences.

    Args:
        known_max_prob:   max softmax prob for each known-family test sample
        unknown_max_prob: max softmax prob for each held-out-family sample
        thresholds:       confidence cut-offs; below → flagged UNKNOWN

    Returns a JSON-serialisable dict.
    """
    known = np.asarray(known_max_prob, dtype=float)
    unknown = np.asarray(unknown_max_prob, dtype=float)
    if known.size == 0 or unknown.size == 0:
        raise ValueError("Both known and unknown confidence arrays must be non-empty")

    # Label unknown as the positive class; score = 1 - confidence
    y = np.concatenate([np.zeros_like(known), np.ones_like(unknown)])
    score = 1.0 - np.concatenate([known, unknown])
    auroc = float(roc_auc_score(y, score))

    # FPR@95TPR from the "known = positive" viewpoint (standard OOD metric):
    # threshold that keeps 95% of known samples, how many unknown slip through?
    fpr, tpr, _ = roc_curve(1 - y, -score)
    idx = np.searchsorted(tpr, 0.95, side='left')
    idx = min(idx, len(fpr) - 1)
    fpr95 = float(fpr[idx])

    per_threshold = {}
    for t in thresholds:
        per_threshold[f"{t:.2f}"] = {
            'unknown_flagged_rate': float((unknown < t).mean()),
            'known_flagged_rate':   float((known < t).mean()),
        }

    return {
        'auroc_known_vs_unknown': auroc,
        'fpr_at_95_tpr':          fpr95,
        'mean_conf_known':        float(known.mean()),
        'mean_conf_unknown':      float(unknown.mean()),
        'n_known':                int(known.size),
        'n_unknown':              int(unknown.size),
        'thresholds':             per_threshold,
    }


def plot_confidence_histogram(
    known_max_prob: np.ndarray,
    unknown_max_prob: np.ndarray,
    output_path,
    thresholds: Iterable[float] = (0.5, 0.8),
) -> None:
    """Overlayed histograms of top-1 confidence for known vs unseen families."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(known_max_prob, bins=bins, alpha=0.6, density=True, label='known families')
    ax.hist(unknown_max_prob, bins=bins, alpha=0.6, density=True, label='unseen families')
    for t in thresholds:
        ax.axvline(t, ls='--', lw=1, color='grey')
        ax.text(t, ax.get_ylim()[1] * 0.95, f' {t:.2f}', fontsize=8, color='grey')
    ax.set_xlabel('Top-1 confidence (max softmax probability)')
    ax.set_ylabel('Density')
    ax.set_title('Confidence on known vs unseen malware families')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
