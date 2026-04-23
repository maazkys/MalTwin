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

