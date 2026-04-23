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
)
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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    num_classes = len(class_names)

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

    return {
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
    }


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

