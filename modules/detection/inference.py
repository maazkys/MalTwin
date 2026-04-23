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

