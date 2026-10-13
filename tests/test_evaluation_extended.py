# tests/test_evaluation_extended.py
"""Tests for extended evaluation metrics and open-set (unseen-variant) helpers."""
import json

import numpy as np
import pytest

from modules.detection.evaluator import (
    compute_extended_metrics, plot_roc_curves, plot_training_curves,
)
from modules.detection.openset import (
    select_holdout_families, openset_metrics, plot_confidence_histogram,
)

NAMES = ['A', 'B', 'C']


def _onehot_probs(labels, n=3, conf=0.9):
    p = np.full((len(labels), n), (1 - conf) / (n - 1))
    p[np.arange(len(labels)), labels] = conf
    return p


class TestExtendedMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        cm = np.diag([2, 2, 2])
        m = compute_extended_metrics(y, _onehot_probs(y), cm, NAMES)
        assert m['roc_auc_macro'] == pytest.approx(1.0)
        assert m['fpr_macro'] == pytest.approx(0.0)
        assert m['mcc'] == pytest.approx(1.0)
        assert m['top3_accuracy'] is None or m['top3_accuracy'] == pytest.approx(1.0)

    def test_per_class_fpr_matches_hand_calculation(self):
        # true: A A B B C C ; pred: A B B B C A
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        cm = np.zeros((3, 3), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        m = compute_extended_metrics(y_true, _onehot_probs(y_pred), cm, NAMES)
        # A: FP=1 (the C predicted as A), TN = 4 non-A rows minus FP = 3 → 1/4
        assert m['per_class_fpr']['A'] == pytest.approx(0.25)
        # B: FP=1 (A predicted as B), negatives=4 → 1/4
        assert m['per_class_fpr']['B'] == pytest.approx(0.25)
        assert m['per_class_fpr']['C'] == pytest.approx(0.0)

    def test_missing_class_does_not_crash(self):
        y = np.array([0, 0, 1, 1])  # class C absent
        cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
        m = compute_extended_metrics(y, _onehot_probs(y), cm, NAMES)
        assert 'C' not in m['per_class_auc']
        assert m['roc_auc_macro'] == pytest.approx(1.0)

    def test_single_class_auc_is_none(self):
        y = np.array([1, 1, 1])
        cm = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
        m = compute_extended_metrics(y, _onehot_probs(y), cm, NAMES)
        assert m['roc_auc_macro'] is None

    def test_result_is_json_serialisable(self):
        y = np.array([0, 1, 2])
        m = compute_extended_metrics(y, _onehot_probs(y), np.eye(3, dtype=int), NAMES)
        json.dumps(m)

    def test_plots_write_files(self, tmp_path):
        y = np.array([0, 1, 2, 0, 1, 2])
        plot_roc_curves(y, _onehot_probs(y, conf=0.6), NAMES, tmp_path / 'roc.png')
        plot_training_curves(
            {'train_loss': [1, .5], 'val_loss': [1.1, .6],
             'train_acc': [.5, .8], 'val_acc': [.4, .7]},
            tmp_path / 'curves.png')
        assert (tmp_path / 'roc.png').stat().st_size > 0
        assert (tmp_path / 'curves.png').stat().st_size > 0


class TestOpenSet:
    def test_holdout_is_deterministic_and_sized(self):
        fams = [f'F{i}' for i in range(10)]
        a = select_holdout_families(fams, 3, seed=1)
        b = select_holdout_families(fams, 3, seed=1)
        assert a == b and len(a) == 3 and set(a) <= set(fams)

    @pytest.mark.parametrize('n', [0, 10, 11])
    def test_holdout_rejects_bad_sizes(self, n):
        with pytest.raises(ValueError):
            select_holdout_families([f'F{i}' for i in range(10)], n)

    def test_perfect_separation(self):
        m = openset_metrics(np.array([0.95, 0.9, 0.99]), np.array([0.2, 0.3, 0.4]))
        assert m['auroc_known_vs_unknown'] == pytest.approx(1.0)
        assert m['fpr_at_95_tpr'] == pytest.approx(0.0)
        assert m['thresholds']['0.50']['unknown_flagged_rate'] == pytest.approx(1.0)
        assert m['thresholds']['0.50']['known_flagged_rate'] == pytest.approx(0.0)

    def test_no_separation_is_half(self):
        conf = np.linspace(0.1, 0.9, 50)
        m = openset_metrics(conf, conf.copy())
        assert m['auroc_known_vs_unknown'] == pytest.approx(0.5)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            openset_metrics(np.array([]), np.array([0.5]))

    def test_histogram_written(self, tmp_path):
        plot_confidence_histogram(np.array([.9, .8]), np.array([.3, .4]), tmp_path / 'h.png')
        assert (tmp_path / 'h.png').stat().st_size > 0
