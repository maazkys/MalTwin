# modules/enhancement/augmentor.py
import torch
from torchvision import transforms


class GaussianNoise:
    """
    Custom torchvision-compatible transform that adds Gaussian noise to a tensor.

    MUST be placed AFTER transforms.ToTensor() in the pipeline.
    Operates on torch.Tensor, not PIL.Image.

    Constructor args:
        mean (float):      noise mean, default 0.0
        std_range (tuple): (min_std, max_std), std sampled uniformly each call.
                           Default (0.01, 0.05).

    __call__:
        1. Sample std = torch.empty(1).uniform_(std_range[0], std_range[1]).item()
        2. Generate noise = torch.randn_like(tensor) * std + mean
        3. result = tensor + noise
        4. Clamp result to [0.0, 1.0]
        5. Return clamped tensor (same shape and dtype as input)
    """

    def __init__(self, mean: float = 0.0, std_range: tuple = (0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussianNoise(mean={self.mean}, std_range={self.std_range})"


def get_train_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the augmentation pipeline for training data.

    Transform order (CRITICAL — do not reorder):
        1. RandomRotation(degrees=15, fill=0, interpolation=BILINEAR) ← PIL stage
        2. RandomHorizontalFlip(p=0.5)             ← PIL stage
        3. RandomVerticalFlip(p=0.5)               ← PIL stage
        4. ColorJitter(brightness=0.2)             ← PIL stage (MUST be before ToTensor)
        5. ToTensor()                              ← converts PIL 'L' → (1, H, W) float32
        6. GaussianNoise(mean=0.0, std=(0.01,0.05))← Tensor stage (MUST be after ToTensor)
        7. Normalize(mean=[0.5], std=[0.5])        ← Tensor stage (single-element lists!)

    Args:
        img_size: not used directly (resizing done in Dataset.__getitem__).

    Returns:
        transforms.Compose instance
    """
    return transforms.Compose([
        transforms.RandomRotation(
            degrees=15,
            fill=0,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),    # PIL stage — before ToTensor
        transforms.ToTensor(),
        GaussianNoise(mean=0.0, std_range=(0.01, 0.05)),  # Tensor stage — after ToTensor
        transforms.Normalize(mean=[0.5], std=[0.5]),       # single-element lists
    ])


def get_val_transforms(img_size: int = 128) -> transforms.Compose:
    """
    Build the inference/validation transform pipeline (NO augmentation).

    Transform order:
        1. ToTensor()                       ← PIL 'L' → (1, H, W) float32
        2. Normalize(mean=[0.5], std=[0.5]) ← maps [0,1] to [-1,1]

    Used for val, test, and inference. NEVER use get_train_transforms for inference.

    Args:
        img_size: kept for API consistency, not used here.

    Returns:
        transforms.Compose instance
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

