# modules/detection/evaluator.py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # MUST be at module level — no display required in server env
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    cohen_kappa_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import config
from .model import MalTwinCNN


def evaluate(
    model: MalTwinCNN,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Full evaluation on test set. Returns comprehensive metrics dict.

    Returns:
        {
            'accuracy':              float,
            'precision_macro':       float,
            'recall_macro':          float,
            'f1_macro':              float,
            'precision_weighted':    float,
            'recall_weighted':       float,
            'f1_weighted':           float,
            'confusion_matrix':      np.ndarray,   # shape (num_classes, num_classes)
            'per_class': {
                family_name: {
                    'precision': float,
                    'recall':    float,
                    'f1':        float,
                    'support':   int,
                }
            },
            'classification_report': str,
            'num_test_samples':      int,
        }

    Implementation:
        1. model.eval()
        2. Collect all predictions and true labels with torch.no_grad()
        3. Compute all metrics using sklearn
        4. Build per_class dict from precision_recall_fscore_support(average=None)
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_labels.extend(labels.numpy().tolist())

    num_classes = len(class_names)
    probs_arr = (np.concatenate(all_probs, axis=0) if all_probs
                 else np.zeros((0, num_classes)))

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Per-class metrics
    prec_per, rec_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        class_names[i]: {
            'precision': float(prec_per[i]),
            'recall':    float(rec_per[i]),
            'f1':        float(f1_per[i]),
            'support':   int(support_per[i]),
        }
        for i in range(num_classes)
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Full classification report string
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )

    extended = compute_extended_metrics(
        np.asarray(all_labels), probs_arr, cm, class_names
    )

    return {
        **extended,
        'accuracy':              float(accuracy),
        'precision_macro':       float(prec_macro),
        'recall_macro':          float(rec_macro),
        'f1_macro':              float(f1_macro),
        'precision_weighted':    float(prec_weighted),
        'recall_weighted':       float(rec_weighted),
        'f1_weighted':           float(f1_weighted),
        'confusion_matrix':      cm,
        'per_class':             per_class,
        'classification_report': report,
        'num_test_samples':      len(all_labels),
        # Raw arrays kept for plotting (not JSON-serialisable; stripped on save)
        'y_true':                np.asarray(all_labels),
        'y_prob':                probs_arr,
    }


def compute_extended_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Metrics beyond accuracy/P/R/F1, as listed in the project evaluation plan:

      - roc_auc_macro / roc_auc_weighted : one-vs-rest ROC-AUC over all classes
      - fpr_macro                        : mean per-class false positive rate
      - per_class_fpr                    : FP / (FP + TN) for each family
      - top3_accuracy                    : true family is in the top-3 predictions
      - mcc                              : Matthews correlation coefficient
      - cohen_kappa                      : agreement beyond chance

    FPR matters for IIoT: a false alarm can stop a production line, so we
    report it explicitly rather than leaving it implied by precision.

    Any metric that is undefined for the given data (e.g. ROC-AUC when a
    class is missing from the test split) is returned as None instead of
    raising, so evaluation of small or partial datasets still completes.
    """
    num_classes = len(class_names)
    labels = list(range(num_classes))
    out: dict = {}

    # Per-class FPR from the confusion matrix
    cm = np.asarray(cm, dtype=np.int64)
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    with np.errstate(divide='ignore', invalid='ignore'):
        fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
    out['per_class_fpr'] = {class_names[i]: float(fpr[i]) for i in range(num_classes)}
    out['fpr_macro'] = float(fpr.mean()) if num_classes else 0.0

    y_pred = y_prob.argmax(axis=1) if len(y_prob) else np.array([], dtype=int)

    try:
        out['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out['mcc'] = None
    try:
        out['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        out['cohen_kappa'] = None

    try:
        k = min(3, num_classes - 1) if num_classes > 2 else 1
        out['top3_accuracy'] = float(
            top_k_accuracy_score(y_true, y_prob, k=k, labels=labels)
        ) if num_classes > 2 else None
    except Exception:
        out['top3_accuracy'] = None

    # One-vs-rest ROC-AUC — only defined for classes present in y_true
    present = sorted(set(int(c) for c in np.unique(y_true)))
    out['roc_auc_macro'] = None
    out['roc_auc_weighted'] = None
    out['per_class_auc'] = {}
    if len(present) >= 2 and len(y_prob):
        y_bin = label_binarize(y_true, classes=labels)
        aucs, weights = [], []
        for c in present:
            pos = y_bin[:, c]
            if pos.min() == pos.max():
                continue
            try:
                a = float(roc_auc_score(pos, y_prob[:, c]))
            except ValueError:
                continue
            out['per_class_auc'][class_names[c]] = a
            aucs.append(a)
            weights.append(int(pos.sum()))
        if aucs:
            out['roc_auc_macro'] = float(np.mean(aucs))
            out['roc_auc_weighted'] = float(np.average(aucs, weights=weights))

    return out


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    One-vs-rest ROC curves plus the micro-average curve, saved as PNG.
    Classes absent from y_true are skipped.
    """
    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.shape[1] == 1:  # binary edge case from label_binarize
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap('tab20')
    for c in labels:
        pos = y_bin[:, c]
        if pos.min() == pos.max():
            continue
        f, t, _ = roc_curve(pos, y_prob[:, c])
        ax.plot(f, t, lw=0.9, alpha=0.75, color=cmap(c % 20), label=class_names[c])

    f, t, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    ax.plot(f, t, lw=2.5, color='black', label='micro-average')
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('MalTwin One-vs-Rest ROC Curves')
    ax.legend(fontsize=6, ncol=2, loc='lower right')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history: dict, output_path: Path) -> None:
    """Train/val loss and accuracy per epoch, saved as a two-panel PNG."""
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='validation')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='train')
    ax2.plot(epochs, history['val_acc'], label='validation')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    figsize: tuple = (16, 14),
) -> None:
    """
    Render and save confusion matrix as PNG.

    matplotlib.use('Agg') is set at module level — never call plt.show().
    plt.close(fig) is mandatory — prevents memory leaks in long training runs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    # Cell annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=6,
            )

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('MalTwin Confusion Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)  # MANDATORY — prevent memory leak


def format_metrics_table(metrics: dict, class_names: list[str]) -> str:
    """
    Format evaluation metrics as a printable ASCII table for CLI output.

    Includes overall metrics and the 5 worst per-class F1 scores.
    """
    width = 44

    def row(label: str, value) -> str:
        if isinstance(value, float):
            return f"║  {label:<22} {value:.4f}          ║"
        return f"║  {label:<22} {value:<14}║"

    border_top    = '╔' + '═' * width + '╗'
    border_mid    = '╠' + '═' * width + '╣'
    border_bot    = '╚' + '═' * width + '╝'
    title_line    = f"║{'MALTWIN TEST EVALUATION':^{width}}║"

    lines = [
        border_top,
        title_line,
        border_mid,
        row('Accuracy:', metrics['accuracy']),
        row('Precision (macro):', metrics['precision_macro']),
        row('Recall (macro):', metrics['recall_macro']),
        row('F1 (macro):', metrics['f1_macro']),
        row('F1 (weighted):', metrics['f1_weighted']),
        *( [row('ROC-AUC (macro):', metrics['roc_auc_macro'])]
           if metrics.get('roc_auc_macro') is not None else [] ),
        *( [row('FPR (macro):', metrics['fpr_macro'])]
           if metrics.get('fpr_macro') is not None else [] ),
        *( [row('MCC:', metrics['mcc'])]
           if metrics.get('mcc') is not None else [] ),
        row('Test Samples:', metrics['num_test_samples']),
        border_mid,
        f"║{'Per-Class F1 (5 worst):':^{width}}║",
    ]

    # Sort by F1 ascending to find worst
    per_class = metrics.get('per_class', {})
    worst5 = sorted(per_class.items(), key=lambda kv: kv[1]['f1'])[:5]
    for name, vals in worst5:
        short_name = name[:20] if len(name) > 20 else name
        lines.append(f"║    {short_name:<20} {vals['f1']:.4f}          ║")

    lines.append(border_bot)
    return '\n'.join(lines)

