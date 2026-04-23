# MalTwin — Phase 4: Detection Module
### Agent Instruction Document | `modules/detection/` + `tests/test_model.py`

> **Read this entire document before writing a single line of code.**
> Every class, method, signature, and behavioral rule is fully specified.
> Do not infer, guess, or deviate from what is written here.

---

## Mandatory Rules (from PRD Section 16)

These are the most commonly hallucinated bugs. Violating any of them causes test failures or silent training errors.

- **Read `MALTWIN_PRD_COMPLETE.md`** before writing any code.
- All CNN tensors are **single-channel**: shape `(batch, 1, H, W)`. NEVER `(batch, 3, H, W)`.
- `CrossEntropyLoss` expects **raw logits** — do NOT apply softmax inside `model.forward()`.
- `model.eval()` and `torch.no_grad()` are **always paired** during inference and validation.
- `torch.manual_seed(42)` is called at the **start of `train()`**, not at module level.
- Use `weights_only=True` in `torch.load()` for PyTorch 2.x security.
- `bias=False` in Conv2d when followed by BatchNorm (eliminates redundant parameters).
- `drop_last=True` in train DataLoader to prevent single-sample batches breaking BatchNorm.
- `self.gradcam_layer = self.block3.conv2` **must be set** in `MalTwinCNN.__init__`.
  Agents forget this constantly even when it is in the spec. The test will catch it.
- `matplotlib.use('Agg')` must be at the **module level** of `evaluator.py` (before any plt calls).
- `plt.close(fig)` is **mandatory** after saving — prevents memory leaks.
- `get_val_transforms` is used inside `predict_single` — NEVER `get_train_transforms`.
- All paths use `pathlib.Path`, never string concatenation.

---

## Phase 4 Scope

Phase 4 implements the full detection module: model architecture, training loop, evaluation, and inference. It depends on Phases 1–3 (binary_to_image, dataset, enhancement must already exist).

### Files to create

| File | Description |
|------|-------------|
| `modules/detection/__init__.py` | Package exports |
| `modules/detection/model.py` | `ConvBlock`, `MalTwinCNN` |
| `modules/detection/trainer.py` | `train()`, `validate_epoch()` |
| `modules/detection/evaluator.py` | `evaluate()`, `plot_confusion_matrix()`, `format_metrics_table()` |
| `modules/detection/inference.py` | `load_model()`, `predict_single()`, `predict_batch()` |
| `tests/test_model.py` | Full test suite (exactly as specified below) |

---

## File 1: `modules/detection/__init__.py`

```python
# modules/detection/__init__.py
from .model import MalTwinCNN
from .trainer import train
from .evaluator import evaluate
from .inference import load_model, predict_single
```

---

## File 2: `modules/detection/model.py`

```python
# modules/detection/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Reusable convolutional block:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d → Dropout2d

    Constructor args:
        in_channels  (int):   input channels
        out_channels (int):   output channels for BOTH Conv layers
        dropout_p    (float): Dropout2d probability, default 0.25

    bias=False in all Conv2d because BatchNorm follows immediately.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop  = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x


class MalTwinCNN(nn.Module):
    """
    Three-block CNN for grayscale malware image classification.

    Input:  (batch_size, 1, 128, 128)  — single-channel grayscale
    Output: (batch_size, num_classes)  — raw logits (NO softmax)

    Architecture:
        Input (1, 128, 128)
            ↓
        block1: ConvBlock(1 → 32)      → (32, 64, 64)   after MaxPool
            ↓
        block2: ConvBlock(32 → 64)     → (64, 32, 32)   after MaxPool
            ↓
        block3: ConvBlock(64 → 128)    → (128, 16, 16)  after MaxPool
            ↓                            ← self.gradcam_layer = self.block3.conv2
        pool:   AdaptiveAvgPool2d(4,4) → (128, 4, 4)
            ↓
        flatten                        → (2048,)
            ↓
        classifier:
            Linear(2048 → 512)
            ReLU
            Dropout(p=0.5)
            Linear(512 → num_classes)
            ↓
        raw logits (num_classes,)

    CRITICAL:
        self.gradcam_layer = self.block3.conv2
        This MUST be set — it is tested explicitly.
        It is used by Module 7 (Grad-CAM) to register backward hooks.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool   = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # Grad-CAM hook target — MUST be the second conv of block3
        self.gradcam_layer = self.block3.conv2

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming normal for Conv2d, constant init for BatchNorm, Xavier for Linear.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # raw logits — NO softmax here
```

---

## File 3: `modules/detection/trainer.py`

```python
# modules/detection/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import config
from .model import MalTwinCNN


def train(
    model: MalTwinCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    lr_patience: int = config.LR_PATIENCE,
    checkpoint_dir: Path = config.CHECKPOINT_DIR,
    best_model_path: Path = config.BEST_MODEL_PATH,
) -> dict:
    """
    Full training loop with per-epoch checkpointing and LR scheduling.

    Returns history dict:
        {
            'train_loss':   list[float],   # mean loss per epoch
            'train_acc':    list[float],   # accuracy 0.0–1.0 per epoch
            'val_loss':     list[float],
            'val_acc':      list[float],
            'best_val_acc': float,
            'best_epoch':   int,
        }

    Optimizer:   Adam(lr=lr, weight_decay=weight_decay)
    Loss:        CrossEntropyLoss() — expects raw logits, no softmax in model
    Scheduler:   ReduceLROnPlateau(mode='max', factor=0.5, patience=lr_patience, min_lr=1e-6)
                 scheduler.step(val_acc) called after each epoch.

    Per-epoch:
        1. model.train()
        2. Iterate train_loader with tqdm (desc="Epoch NNN/NNN [Train]")
        3. Forward → loss → backward → optimizer.step()
        4. Accumulate correct predictions for accuracy
        5. validate_epoch() → val_loss, val_acc
        6. scheduler.step(val_acc)
        7. Print epoch summary
        8. Save checkpoint: checkpoint_dir/epoch_{N:03d}_acc{val_acc:.4f}.pt
        9. If val_acc > best so far: save model.state_dict() to best_model_path

    Reproducibility:
        torch.manual_seed(config.RANDOM_SEED) at top of function.
        torch.cuda.manual_seed(config.RANDOM_SEED) if CUDA.
    """
    # Reproducibility — seed at start of train(), not at module level
    torch.manual_seed(config.RANDOM_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.RANDOM_SEED)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-6,
        verbose=True,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss':   [],
        'train_acc':    [],
        'val_loss':     [],
        'val_acc':      [],
        'best_val_acc': 0.0,
        'best_epoch':   0,
    }
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ── Training phase ──────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:03d}/{epochs:03d} [Train]",
            leave=False,
        )
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total:.4f}"})

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validation phase ─────────────────────────────────────────────────────
        val_loss, val_acc = validate_epoch(model, val_loader, device, criterion)

        # ── Scheduler step ───────────────────────────────────────────────────────
        scheduler.step(val_acc)

        # ── Logging ──────────────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────────
        checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:03d}_acc{val_acc:.4f}.pt"
        torch.save({
            'epoch':           epoch + 1,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc':         val_acc,
            'val_loss':        val_loss,
            'train_acc':       train_acc,
            'train_loss':      train_loss,
        }, checkpoint_path)

        # ── Best model save ───────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch']   = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ New best model saved (val_acc={val_acc:.4f})")

    return history


def validate_epoch(
    model: MalTwinCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Run one full validation pass. Returns (avg_loss, accuracy).

    model.eval() and torch.no_grad() are always used together here.
    Loss is accumulated weighted by batch size for correctness with variable batch sizes.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Multiply by batch size for correct mean across variable-size final batch
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
```

---

## File 4: `modules/detection/evaluator.py`

```python
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
```

---

## File 5: `modules/detection/inference.py`

```python
# modules/detection/inference.py
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import config
from .model import MalTwinCNN
from modules.enhancement.augmentor import get_val_transforms


def load_model(
    model_path: Path = config.BEST_MODEL_PATH,
    num_classes: int = config.MALIMG_EXPECTED_FAMILIES,
    device: torch.device = config.DEVICE,
) -> MalTwinCNN:
    """
    Load a trained MalTwinCNN from a .pt file containing state_dict only.

    Args:
        model_path:  path to best_model.pt
        num_classes: must match the trained model's output layer (25 for Malimg)
        device:      target device; handles CUDA→CPU migration automatically

    Returns:
        MalTwinCNN in eval() mode on the specified device.

    Raises:
        FileNotFoundError: if model_path does not exist.

    Notes:
        - weights_only=True is the PyTorch 2.x secure loading flag.
        - map_location=device handles CUDA-trained models on CPU-only machines.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run scripts/train.py to train the model first."
        )
    model = MalTwinCNN(num_classes=num_classes)
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_single(
    model: MalTwinCNN,
    img_array: np.ndarray,
    class_names: list[str],
    device: torch.device = config.DEVICE,
) -> dict:
    """
    Run inference on a single 128×128 grayscale image array.

    Args:
        model:       MalTwinCNN instance (eval mode recommended but enforced here)
        img_array:   numpy array shape (128, 128), dtype uint8, values 0–255
        class_names: ordered list of family names (index = class integer)
        device:      inference device

    Returns:
        {
            'predicted_family': str,               # top-1 class name
            'confidence':       float,             # top-1 softmax probability [0.0, 1.0]
            'probabilities':    dict[str, float],  # {family: prob} for ALL classes
            'top3': [
                {'family': str, 'confidence': float},  # rank 1
                {'family': str, 'confidence': float},  # rank 2
                {'family': str, 'confidence': float},  # rank 3
            ]
        }

    All float values are Python float (JSON-serialisable, not numpy float32).

    Pipeline:
        1. PIL Image from uint8 array (mode='L')
        2. get_val_transforms()(pil_img) → tensor (1, 128, 128) float32
        3. unsqueeze(0) → (1, 1, 128, 128)
        4. model.eval() + torch.no_grad() → logits (1, num_classes)
        5. softmax → probs (num_classes,)
        6. argmax → top-1; argsort descending → top-3
    """
    # 1. PIL Image
    pil_img = Image.fromarray(img_array, mode='L')

    # 2. Val transforms (ToTensor + Normalize) — NEVER train transforms for inference
    transform = get_val_transforms(config.IMG_SIZE)
    tensor = transform(pil_img)            # (1, 128, 128)

    # 3. Batch dimension + device
    tensor = tensor.unsqueeze(0).to(device)  # (1, 1, 128, 128)

    # 4. Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(tensor)             # (1, num_classes)

    # 5. Softmax probabilities
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    # probs shape: (num_classes,)

    # 6. Top-1
    top1_idx        = int(np.argmax(probs))
    top1_confidence = float(probs[top1_idx])
    top1_family     = class_names[top1_idx]

    # 7. All probabilities dict
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    # 8. Top-3 (descending by confidence)
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {'family': class_names[int(i)], 'confidence': float(probs[i])}
        for i in top3_indices
    ]

    return {
        'predicted_family': top1_family,
        'confidence':       top1_confidence,
        'probabilities':    prob_dict,
        'top3':             top3,
    }


def predict_batch(
    model: MalTwinCNN,
    img_arrays: list[np.ndarray],
    class_names: list[str],
    device: torch.device = config.DEVICE,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run inference on multiple images. Returns list of result dicts (same format as predict_single).

    Processes img_arrays in chunks of batch_size to avoid OOM on CPU.
    Results are in the same order as input.

    Implementation:
        - For each chunk: stack transforms into (B, 1, 128, 128), forward, softmax.
        - Decompose back into per-image dicts.
    """
    transform = get_val_transforms(config.IMG_SIZE)
    results = []

    model.eval()
    with torch.no_grad():
        for chunk_start in range(0, len(img_arrays), batch_size):
            chunk = img_arrays[chunk_start:chunk_start + batch_size]

            tensors = []
            for arr in chunk:
                pil_img = Image.fromarray(arr, mode='L')
                tensors.append(transform(pil_img))

            batch_tensor = torch.stack(tensors).to(device)   # (B, 1, 128, 128)
            logits = model(batch_tensor)                      # (B, num_classes)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()  # (B, num_classes)

            for probs in probs_batch:
                top1_idx    = int(np.argmax(probs))
                prob_dict   = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                top3_idx    = np.argsort(probs)[::-1][:3]
                top3        = [{'family': class_names[int(i)], 'confidence': float(probs[i])}
                               for i in top3_idx]
                results.append({
                    'predicted_family': class_names[top1_idx],
                    'confidence':       float(probs[top1_idx]),
                    'probabilities':    prob_dict,
                    'top3':             top3,
                })

    return results
```

---

## File 6: `tests/test_model.py`

Write this file **exactly** as shown. Do not add, remove, or rename any test.

```python
"""
Test suite for modules/detection/model.py and modules/detection/inference.py

All tests run without the Malimg dataset or a trained model.
No @pytest.mark.integration tests are needed here — the model can be instantiated
and inference can be run on random tensors without any dataset.

Run:
    pytest tests/test_model.py -v
"""
import pytest
import torch
import numpy as np
import json
from modules.detection.model import MalTwinCNN, ConvBlock


# ─────────────────────────────────────────────────────────────────────────────
# ConvBlock tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConvBlock:
    def test_output_shape_block1(self):
        """block1: (B, 1, 128, 128) → (B, 32, 64, 64) after MaxPool."""
        block = ConvBlock(in_channels=1, out_channels=32)
        x = torch.randn(4, 1, 128, 128)
        out = block(x)
        assert out.shape == (4, 32, 64, 64)

    def test_output_shape_block2(self):
        """block2: (B, 32, 64, 64) → (B, 64, 32, 32) after MaxPool."""
        block = ConvBlock(in_channels=32, out_channels=64)
        x = torch.randn(4, 32, 64, 64)
        out = block(x)
        assert out.shape == (4, 64, 32, 32)

    def test_output_shape_block3(self):
        """block3: (B, 64, 32, 32) → (B, 128, 16, 16) after MaxPool."""
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(4, 64, 32, 32)
        out = block(x)
        assert out.shape == (4, 128, 16, 16)

    def test_conv1_bias_false(self):
        block = ConvBlock(1, 32)
        assert block.conv1.bias is None, "Conv2d bias should be None when bias=False"

    def test_conv2_bias_false(self):
        block = ConvBlock(1, 32)
        assert block.conv2.bias is None, "Conv2d bias should be None when bias=False"

    def test_has_batchnorm(self):
        import torch.nn as nn
        block = ConvBlock(1, 32)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)

    def test_has_maxpool(self):
        import torch.nn as nn
        block = ConvBlock(1, 32)
        assert isinstance(block.pool, nn.MaxPool2d)

    def test_custom_dropout(self):
        import torch.nn as nn
        block = ConvBlock(1, 32, dropout_p=0.1)
        assert isinstance(block.drop, nn.Dropout2d)


# ─────────────────────────────────────────────────────────────────────────────
# MalTwinCNN architecture tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMalTwinCNN:
    def test_forward_pass_output_shape(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert out.shape == (4, num_classes)

    def test_single_sample_forward(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, num_classes)

    def test_parameter_count_reasonable(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000, f"Too few parameters: {total}"
        assert total < 20_000_000, f"Too many parameters: {total}"

    def test_output_is_raw_logits_no_softmax(self, num_classes):
        """
        Verify softmax was NOT applied in forward().
        If softmax was applied, all outputs would be in [0,1] and sum to 1.
        We verify that softmax of the output produces valid probabilities —
        this tests the contract (logits in, probs after softmax), not the values.
        """
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        # Applying softmax to raw logits must yield probabilities that sum to 1
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_gradcam_layer_attribute_exists(self, num_classes):
        """
        CRITICAL: self.gradcam_layer must be set and must point to block3.conv2.
        This is required by Module 7 (Grad-CAM).
        """
        model = MalTwinCNN(num_classes=num_classes)
        assert hasattr(model, 'gradcam_layer'), \
            "MalTwinCNN must have self.gradcam_layer attribute"
        assert model.gradcam_layer is model.block3.conv2, \
            "gradcam_layer must be self.block3.conv2 (the second conv of block3)"

    def test_block_attributes_exist(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert hasattr(model, 'block1')
        assert hasattr(model, 'block2')
        assert hasattr(model, 'block3')
        assert hasattr(model, 'pool')
        assert hasattr(model, 'classifier')

    def test_block1_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block1.conv1.in_channels == 1
        assert model.block1.conv1.out_channels == 32

    def test_block2_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block2.conv1.in_channels == 32
        assert model.block2.conv1.out_channels == 64

    def test_block3_channels(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        assert model.block3.conv1.in_channels == 64
        assert model.block3.conv1.out_channels == 128

    def test_deterministic_in_eval_mode(self, num_classes):
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_train_mode_dropout_nondeterministic(self, num_classes):
        """In train mode, Dropout causes different outputs for different seeds."""
        model = MalTwinCNN(num_classes=num_classes)
        model.train()
        x = torch.randn(4, 1, 128, 128)
        torch.manual_seed(0)
        out1 = model(x)
        torch.manual_seed(1)
        out2 = model(x)
        assert not torch.equal(out1, out2)

    def test_weight_initialization_conv_no_zeros(self, num_classes):
        """Kaiming init should produce non-zero weights."""
        model = MalTwinCNN(num_classes=num_classes)
        # At least some weights should be nonzero after Kaiming init
        conv_weights = model.block1.conv1.weight.data
        assert conv_weights.abs().sum() > 0

    def test_batchnorm_initialized_correctly(self, num_classes):
        """BatchNorm weight=1, bias=0 after _initialize_weights."""
        model = MalTwinCNN(num_classes=num_classes)
        # Check block1's BN
        assert torch.allclose(model.block1.bn1.weight, torch.ones_like(model.block1.bn1.weight))
        assert torch.allclose(model.block1.bn1.bias, torch.zeros_like(model.block1.bn1.bias))

    def test_adaptive_avg_pool_output(self, num_classes):
        """AdaptiveAvgPool2d should output (B, 128, 4, 4) before flatten."""
        import torch.nn as nn
        model = MalTwinCNN(num_classes=num_classes)
        # Run up to just before classifier
        x = torch.randn(2, 1, 128, 128)
        x = model.block1(x)   # (2, 32, 64, 64)
        x = model.block2(x)   # (2, 64, 32, 32)
        x = model.block3(x)   # (2, 128, 16, 16)
        x = model.pool(x)     # (2, 128, 4, 4)
        assert x.shape == (2, 128, 4, 4)

    def test_classifier_output_size(self, num_classes):
        """Classifier input should be 128*4*4=2048."""
        model = MalTwinCNN(num_classes=num_classes)
        # Find the first Linear layer in classifier
        import torch.nn as nn
        first_linear = next(m for m in model.classifier.modules() if isinstance(m, nn.Linear))
        assert first_linear.in_features == 2048


# ─────────────────────────────────────────────────────────────────────────────
# Inference tests (predict_single, load_model)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSingle:
    def _make_result(self, num_classes, sample_grayscale_array):
        from modules.detection.inference import predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        return predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))

    def test_returns_required_keys(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert 'predicted_family' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'top3' in result

    def test_confidence_in_valid_range(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_has_all_classes(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert len(result['probabilities']) == num_classes

    def test_all_probabilities_nonnegative(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert all(v >= 0.0 for v in result['probabilities'].values())

    def test_all_probabilities_at_most_one(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert all(v <= 1.0 for v in result['probabilities'].values())

    def test_top3_has_three_entries(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert len(result['top3']) == 3

    def test_top3_entries_have_required_keys(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        for entry in result['top3']:
            assert 'family' in entry
            assert 'confidence' in entry

    def test_predicted_family_matches_top3_first(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert result['predicted_family'] == result['top3'][0]['family']

    def test_predicted_family_confidence_matches_top3_first(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert abs(result['confidence'] - result['top3'][0]['confidence']) < 1e-6

    def test_top3_sorted_descending(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        confs = [item['confidence'] for item in result['top3']]
        assert confs == sorted(confs, reverse=True)

    def test_predicted_family_is_valid_class(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        assert result['predicted_family'] in class_names

    def test_all_values_json_serialisable(self, sample_grayscale_array, num_classes):
        """All float values must be Python float, not numpy float32."""
        result = self._make_result(num_classes, sample_grayscale_array)
        # Should not raise TypeError
        json.dumps(result)

    def test_confidence_is_python_float(self, sample_grayscale_array, num_classes):
        result = self._make_result(num_classes, sample_grayscale_array)
        assert isinstance(result['confidence'], float)

    def test_uses_val_transforms_not_train(self, sample_grayscale_array, num_classes):
        """
        Calling predict_single twice on the same image should return identical results
        (val transforms are deterministic; train transforms are not).
        """
        from modules.detection.inference import predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        r1 = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        r2 = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        assert r1['predicted_family'] == r2['predicted_family']
        assert abs(r1['confidence'] - r2['confidence']) < 1e-6


class TestLoadModel:
    def test_raises_on_missing_file(self, tmp_path):
        from modules.detection.inference import load_model
        missing = tmp_path / 'nonexistent.pt'
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(model_path=missing, num_classes=25, device=torch.device('cpu'))

    def test_loads_saved_state_dict(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        # Save a freshly initialised model's state dict
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)

        # Load it back
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        assert isinstance(loaded, MalTwinCNN)

    def test_loaded_model_is_in_eval_mode(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        assert not loaded.training, "load_model() must return model in eval() mode"

    def test_loaded_model_produces_correct_output_shape(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = loaded(x)
        assert out.shape == (1, num_classes)

    def test_loaded_weights_match_original(self, tmp_path, num_classes):
        from modules.detection.inference import load_model
        model = MalTwinCNN(num_classes=num_classes)
        pt_path = tmp_path / 'test_model.pt'
        torch.save(model.state_dict(), pt_path)
        loaded = load_model(model_path=pt_path, num_classes=num_classes, device=torch.device('cpu'))

        original_w = model.block1.conv1.weight.data
        loaded_w   = loaded.block1.conv1.weight.data
        torch.testing.assert_close(original_w, loaded_w)


class TestPredictBatch:
    def test_returns_list_of_dicts(self, sample_grayscale_array, num_classes):
        from modules.detection.inference import predict_batch
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]
        results = predict_batch(model, [sample_grayscale_array, sample_grayscale_array],
                                class_names, torch.device('cpu'))
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_results_match_single(self, sample_grayscale_array, num_classes):
        """predict_batch on one image should match predict_single on same image."""
        from modules.detection.inference import predict_batch, predict_single
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]

        single = predict_single(model, sample_grayscale_array, class_names, torch.device('cpu'))
        batch  = predict_batch(model, [sample_grayscale_array], class_names, torch.device('cpu'))

        assert single['predicted_family'] == batch[0]['predicted_family']
        assert abs(single['confidence'] - batch[0]['confidence']) < 1e-5

    def test_result_order_preserved(self, num_classes):
        """Results must be in same order as inputs."""
        from modules.detection.inference import predict_batch
        import cv2
        model = MalTwinCNN(num_classes=num_classes)
        model.eval()
        class_names = [f"Family_{i:02d}" for i in range(num_classes)]

        # Create two distinct arrays: all-zeros and all-255
        arr_black = np.zeros((128, 128), dtype=np.uint8)
        arr_white = np.full((128, 128), 255, dtype=np.uint8)

        results = predict_batch(model, [arr_black, arr_white], class_names, torch.device('cpu'))
        assert len(results) == 2
        # Both should return valid results (can't predict which family, but structure valid)
        for r in results:
            assert 'predicted_family' in r
            assert 'confidence' in r
```

---

## Definition of Done

Run these commands. All must pass before Phase 4 is complete.

```bash
# Phase 4 tests — no dataset, no trained model required
pytest tests/test_model.py -v

# Expected: all tests pass with zero failures
# ====== 40+ passed ======

# Regression check — phases 1, 2, 3 must still pass
pytest tests/test_converter.py tests/test_dataset.py tests/test_enhancement.py \
       -v -m "not integration"
```

### Checklist

- [ ] `pytest tests/test_model.py -v` passes with zero failures
- [ ] No regressions in earlier test files
- [ ] `MalTwinCNN.__init__` sets `self.gradcam_layer = self.block3.conv2`
- [ ] `model.forward()` returns raw logits — NO `torch.softmax()` inside `forward()`
- [ ] `CrossEntropyLoss` is used in `trainer.py` (not NLLLoss + softmax)
- [ ] `torch.manual_seed(config.RANDOM_SEED)` is at the **top of `train()`**, not module level
- [ ] `validate_epoch()` uses `model.eval()` + `torch.no_grad()` together
- [ ] `matplotlib.use('Agg')` is at **module level** in `evaluator.py` (before any plt import)
- [ ] `plt.close(fig)` is called after `plt.savefig()` in `plot_confusion_matrix()`
- [ ] `torch.load(..., weights_only=True)` is used in `load_model()`
- [ ] `predict_single()` calls `get_val_transforms`, not `get_train_transforms`
- [ ] All probability values returned by `predict_single()` are Python `float`, not `np.float32`
- [ ] `load_model()` returns model in `eval()` mode
- [ ] `bias=False` in all Conv2d layers in `ConvBlock`

---

## Common Bugs to Avoid

| Bug | Symptom | Fix |
|-----|---------|-----|
| `torch.softmax()` inside `forward()` | CrossEntropyLoss produces wrong gradients; model trains poorly | Remove softmax from `forward()`. Apply in `predict_single()` only. |
| Forgetting `self.gradcam_layer = self.block3.conv2` | `test_gradcam_layer_attribute_exists` fails | Set it explicitly in `__init__` after `self.block3` is created |
| `matplotlib.use('Agg')` inside a function | Fails if any plt has already been imported elsewhere | Must be at **module level**, line 3–4 of evaluator.py |
| Missing `plt.close(fig)` | Memory leak during long training runs | Always `plt.close(fig)` after `plt.savefig()` |
| `torch.load()` without `weights_only=True` | PyTorch 2.x security warning; potential attack surface | Use `torch.load(path, map_location=device, weights_only=True)` |
| `get_train_transforms` in `predict_single` | Results non-deterministic (GaussianNoise, random flips) | Use `get_val_transforms` — always for inference |
| `np.float32` values in result dict | `json.dumps(result)` raises `TypeError` | Wrap all values: `float(probs[i])` not `probs[i]` |
| `torch.manual_seed()` at module level | Seed applied at import time, not at training start | Call inside `train()` as first line |
| `bias=True` in Conv2d before BatchNorm | Extra parameters, redundant computation | `bias=False` whenever BatchNorm follows |
| `validate_epoch` without `model.eval()` | Dropout active during validation → noisy val_acc | Always call `model.eval()` before the val loop |

---

*Phase 4 complete → proceed to Phase 5: CLI scripts (`scripts/train.py`, `scripts/evaluate.py`, `scripts/convert_binary.py`).*
