# modules/dataset/loader.py
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from typing import Optional, Callable
from PIL import Image

import config
from .preprocessor import encode_labels, save_class_names


class MalimgDataset(Dataset):
    """
    PyTorch Dataset for the Malimg malware image dataset.

    Loads grayscale PNG images from directory structure:
        data_dir/FamilyName/image.png

    Each image is:
        1. Loaded as grayscale (single channel) with cv2.IMREAD_GRAYSCALE
        2. Resized to (img_size, img_size) using cv2.resize(img, (img_size, img_size))
           NOTE: cv2.resize takes (width, height), so (img_size, img_size) is correct for square images.
        3. Converted to PIL Image mode 'L'
        4. Transform applied (returns float32 tensor shape (1, H, W))

    Internal data structure:
        self.samples: list[tuple[Path, int]]
        self.class_names: list[str]  — sorted alphabetically, index = label integer
        self.label_map: dict[str, int]
        self.class_counts: dict[str, int]  — counts for THIS split only

    Split algorithm (must be implemented exactly as specified):
        Step 1: Gather all (path, label) pairs for entire dataset across all families.
        Step 2: Extract label list for stratification.
        Step 3: train_test_split(all_samples, test_size=(val_ratio + test_ratio),
                                 stratify=labels, random_state=random_seed)
                → produces train_samples, temp_samples
        Step 4: relative_val = val_ratio / (val_ratio + test_ratio)
                train_test_split(temp_samples, test_size=(1 - relative_val),
                                 stratify=temp_labels, random_state=random_seed)
                → produces val_samples, test_samples
        Step 5: self.samples = train_samples / val_samples / test_samples per requested split.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        img_size: int = config.IMG_SIZE,
        transform: Optional[Callable] = None,
        train_ratio: float = config.TRAIN_RATIO,
        val_ratio: float = config.VAL_RATIO,
        test_ratio: float = config.TEST_RATIO,
        random_seed: int = config.RANDOM_SEED,
    ):
        """
        Custom PyTorch Dataset for the Malimg malware image dataset.

        Deviation from SRS SI-5: The SRS specifies PyTorch ImageFolder.
        This custom class is used instead because it provides:
          - Stratified train/val/test splitting (ImageFolder has no split support)
          - Per-split class count tracking (needed by ClassAwareOversampler)
          - Integration with WeightedRandomSampler for class balancing

        The interface is otherwise Dataset-compatible (len, getitem, get_labels).
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.random_seed = random_seed

        # Build label map from all family subdirectories
        family_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        all_families = [d.name for d in family_dirs]
        self.label_map = encode_labels(all_families)
        self.class_names = sorted(all_families)  # sorted alphabetically

        # Gather all (path, label) pairs
        all_samples: list[tuple[Path, int]] = []
        for family_dir in family_dirs:
            label = self.label_map[family_dir.name]
            png_files = sorted(list(family_dir.glob('*.png')) + list(family_dir.glob('*.PNG')))
            # Deduplicate on case-insensitive filesystems
            seen = set()
            deduped = []
            for p in png_files:
                key = str(p).lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(p)
            for path in deduped:
                all_samples.append((path, label))

        # Stratified split
        labels = [s[1] for s in all_samples]
        train_samples, temp_samples = train_test_split(
            all_samples,
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=random_seed,
        )
        temp_labels = [s[1] for s in temp_samples]
        relative_val = val_ratio / (val_ratio + test_ratio)
        try:
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1.0 - relative_val),
                stratify=temp_labels,
                random_state=random_seed,
            )
        except ValueError:
            # Fall back to unstratified when temp set is too small to stratify
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1.0 - relative_val),
                stratify=None,
                random_state=random_seed,
            )

        split_map = {'train': train_samples, 'val': val_samples, 'test': test_samples}
        self.samples = split_map[split]

        # Compute class counts for this split
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.samples)
        self.class_counts = {self.class_names[lbl]: cnt.get(lbl, 0) for lbl in range(len(self.class_names))}

        # Default transform: val transforms (no augmentation)
        if transform is None:
            from modules.enhancement.augmentor import get_val_transforms
            self.transform = get_val_transforms(img_size)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        # cv2.resize takes (width, height) — for square images this is (img_size, img_size)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(img, mode='L')
        tensor = self.transform(pil_img)   # shape: (1, img_size, img_size), float32
        return tensor, label

    def get_labels(self) -> list[int]:
        """Returns list of integer labels for all samples in this split."""
        return [label for _, label in self.samples]


def get_dataloaders(
    data_dir: Path = config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    oversample_strategy: str = config.OVERSAMPLE_STRATEGY,
    augment_train: bool = True,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build all three DataLoaders and return (train_loader, val_loader, test_loader, class_names).

    - Train loader uses oversampling sampler + optional augmentation.
    - Val and test loaders use val transforms, shuffle=False, no sampler.
    - Persists class_names to config.CLASS_NAMES_PATH for dashboard use.
    - drop_last=True on train loader prevents incomplete final batches.
    - pin_memory=True when CUDA is available.
    """
    from modules.enhancement.augmentor import get_train_transforms, get_val_transforms
    from modules.enhancement.balancer import ClassAwareOversampler

    val_transform = get_val_transforms(img_size)
    train_transform = get_train_transforms(img_size) if augment_train else val_transform

    train_ds = MalimgDataset(data_dir, 'train', img_size, train_transform,
                              random_seed=random_seed)
    val_ds   = MalimgDataset(data_dir, 'val',   img_size, val_transform,
                              random_seed=random_seed)
    test_ds  = MalimgDataset(data_dir, 'test',  img_size, val_transform,
                              random_seed=random_seed)

    sampler = ClassAwareOversampler(train_ds, strategy=oversample_strategy).get_sampler()
    use_pin = (config.DEVICE.type == 'cuda')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,         # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,          # avoid incomplete final batch during training
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
    )

    # Persist class names for dashboard
    save_class_names(train_ds.class_names, config.CLASS_NAMES_PATH)

    return train_loader, val_loader, test_loader, train_ds.class_names

