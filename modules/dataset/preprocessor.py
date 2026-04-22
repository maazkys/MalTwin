# modules/dataset/preprocessor.py
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional


def validate_dataset_integrity(data_dir: Path) -> dict:
    """
    Scans the Malimg dataset directory and produces an integrity report.

    Args:
        data_dir: Path to the Malimg root directory (config.DATA_DIR).

    Returns:
        {
            'valid':            bool,         # True if no corrupt files found
            'families':         list[str],    # sorted list of family folder names
            'counts':           dict[str,int],# {family: sample_count}
            'total':            int,          # sum of all counts
            'min_class':        str,          # family with fewest samples
            'max_class':        str,          # family with most samples
            'imbalance_ratio':  float,        # max_count / min_count
            'corrupt_files':    list[str],    # str(path) of unreadable files
            'missing_dirs':     list[str],    # always [] — see notes
        }

    Raises:
        FileNotFoundError: if data_dir does not exist
        FileNotFoundError: if data_dir has no subdirectories

    Implementation notes:
        - Iterate over data_dir.iterdir(), keeping only directories.
        - For each family dir, iterate over *.png files (case-insensitive via glob('*.png') + glob('*.PNG')).
        - For each PNG, attempt cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).
          If result is None, add str(path) to corrupt_files list.
        - Sort families alphabetically.
        - corrupt_files contains str representations of Path objects.
        - missing_dirs = [] (we cannot know expected names without hardcoding).
        - imbalance_ratio = max_count / min_count. Handle divide-by-zero if min_count == 0.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"Dataset directory is empty: {data_dir}")

    families = sorted([d.name for d in subdirs])
    counts = {}
    corrupt_files = []

    for family_dir in sorted(subdirs, key=lambda d: d.name):
        family = family_dir.name
        png_files = list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG'))
        # Deduplicate (glob may overlap on case-insensitive filesystems)
        png_files = list({str(p): p for p in png_files}.values())
        count = 0
        for path in png_files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                corrupt_files.append(str(path))
            else:
                count += 1
        counts[family] = count

    total = sum(counts.values())
    max_class = max(counts, key=lambda k: counts[k]) if counts else ''
    min_class = min(counts, key=lambda k: counts[k]) if counts else ''
    max_count = counts.get(max_class, 0)
    min_count = counts.get(min_class, 1)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    return {
        'valid':           len(corrupt_files) == 0,
        'families':        families,
        'counts':          counts,
        'total':           total,
        'min_class':       min_class,
        'max_class':       max_class,
        'imbalance_ratio': imbalance_ratio,
        'corrupt_files':   corrupt_files,
        'missing_dirs':    [],
    }


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert uint8 image [0, 255] to float32 [0.0, 1.0].

    Args:
        img: numpy array, dtype uint8.

    Returns:
        numpy array, same shape, dtype float32, values in [0.0, 1.0].

    Implementation:
        return img.astype(np.float32) / 255.0

    Notes:
        - Do NOT use cv2.normalize here. Simple division is exact and fast.
        - The output of this function feeds directly into PyTorch tensors.
    """
    return img.astype(np.float32) / 255.0


def encode_labels(families: list[str]) -> dict[str, int]:
    """
    Create a deterministic string→integer label mapping.

    Args:
        families: list of family names.

    Returns:
        Dict mapping each family name to a unique integer [0, len(families)-1].
        Sorted alphabetically so the mapping is always the same for the same input.

    Implementation:
        return {name: idx for idx, name in enumerate(sorted(families))}

    Example:
        encode_labels(['Yuner.A', 'Allaple.A', 'VB.AT'])
        → {'Allaple.A': 0, 'VB.AT': 1, 'Yuner.A': 2}
    """
    return {name: idx for idx, name in enumerate(sorted(families))}


def save_class_names(class_names: list[str], output_path: Path) -> None:
    """
    Persist the ordered class name list to JSON for dashboard use.

    Args:
        class_names: sorted list of family names (index = label integer).
        output_path: destination JSON path (config.CLASS_NAMES_PATH).

    File format:
        {"class_names": ["Adialer.C", "Agent.FYI", ...]}

    Notes:
        - Creates parent directory if it does not exist.
        - Overwrites if file already exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)


def load_class_names(input_path: Path) -> list[str]:
    """
    Load class names from JSON file written by save_class_names.

    Args:
        input_path: path to class_names.json (config.CLASS_NAMES_PATH).

    Returns:
        list[str] of family names in index order.

    Raises:
        FileNotFoundError: if file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"class_names.json not found at {input_path}. "
            "Run scripts/train.py first."
        )
    with open(input_path) as f:
        return json.load(f)['class_names']

