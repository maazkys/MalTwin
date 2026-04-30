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
    Load a trained MalTwinCNN from a .pt file.

    Handles the following saved formats automatically:
        1. Raw state_dict (standard torch.save(model.state_dict(), path))
        2. Checkpoint dict with 'model_state_dict' or 'state_dict' key
        3. DataParallel-wrapped weights (module. prefix)

    Args:
        model_path:  path to best_model.pt
        num_classes: must match the trained model's output layer (25 for Malimg)
        device:      target device; handles CUDA→CPU migration automatically

    Returns:
        MalTwinCNN in eval() mode on the specified device.

    Raises:
        FileNotFoundError: if model_path does not exist.
        RuntimeError:      if weights cannot be loaded after all recovery attempts,
                           with a detailed diagnostic message.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run scripts/train.py to train the model first."
        )

    # ── 1. Load raw checkpoint ───────────────────────────────────────────────
    raw = torch.load(str(model_path), map_location=device, weights_only=True)

    # ── 2. Extract state_dict regardless of how it was saved ────────────────
    if isinstance(raw, dict) and not _looks_like_state_dict(raw):
        # Likely a full checkpoint dict — try common key names
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw:
                state_dict = raw[key]
                break
        else:
            # Fallback: assume it is the state_dict after all
            state_dict = raw
    else:
        state_dict = raw

    # ── 3. Strip DataParallel 'module.' prefix if present ───────────────────
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # ── 4. Build model ───────────────────────────────────────────────────────
    model = MalTwinCNN(num_classes=num_classes)

    # ── 5. Diagnose mismatches before attempting strict load ─────────────────
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state_dict.keys())
    missing    = model_keys - ckpt_keys
    unexpected = ckpt_keys  - model_keys

    if missing or unexpected:
        # Try to infer the num_classes the checkpoint was trained with
        ckpt_num_classes = _infer_num_classes(state_dict)
        msg = (
            f"\n[load_model] State dict mismatch detected.\n"
            f"  model_path           : {model_path}\n"
            f"  num_classes (passed) : {num_classes}\n"
            f"  num_classes (ckpt)   : {ckpt_num_classes if ckpt_num_classes else 'unknown'}\n"
            f"  Missing  keys ({len(missing)}): {missing}\n"
            f"  Unexpected keys ({len(unexpected)}): {unexpected}\n"
        )

        if ckpt_num_classes and ckpt_num_classes != num_classes:
            msg += (
                f"\n  → num_classes mismatch is the likely cause.\n"
                f"    Rebuilding model with num_classes={ckpt_num_classes} and retrying.\n"
            )
            # Rebuild with the correct num_classes and retry
            model = MalTwinCNN(num_classes=ckpt_num_classes)
            missing    = set(model.state_dict().keys()) - ckpt_keys
            unexpected = ckpt_keys - set(model.state_dict().keys())
            if not missing and not unexpected:
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                print(msg)  # still print so developer is aware
                return model

        raise RuntimeError(msg)

    # ── 6. Strict load (clean path) ──────────────────────────────────────────
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _looks_like_state_dict(d: dict) -> bool:
    """
    Heuristic: a raw state_dict has tensor values.
    A checkpoint dict typically has non-tensor values (epoch, loss, etc.).
    """
    return all(isinstance(v, torch.Tensor) for v in d.values())


def _infer_num_classes(state_dict: dict) -> int | None:
    """
    Inspect the final Linear layer weight shape to recover num_classes.
    MalTwinCNN's last layer key: 'classifier.4.weight' → shape (num_classes, 512)
    """
    # classifier is nn.Sequential; index 4 is the final Linear
    for key in ("classifier.4.weight", "classifier.3.weight", "fc.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return None


# ── Inference ─────────────────────────────────────────────────────────────────

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