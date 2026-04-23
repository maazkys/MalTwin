"""
Test suite for modules/enhancement/
All tests are unit tests — no Malimg dataset required.
Run: pytest tests/test_enhancement.py -v
"""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from modules.enhancement.augmentor import (
    GaussianNoise,
    get_train_transforms,
    get_val_transforms,
)
from modules.enhancement.balancer import ClassAwareOversampler


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_pil_grayscale(size: int = 128, seed: int = 0) -> Image.Image:
    """Create a deterministic grayscale PIL Image in mode 'L'."""
    rng = np.random.default_rng(seed=seed)
    arr = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


def make_tensor(size: int = 128, seed: int = 0) -> torch.Tensor:
    """Create a deterministic float32 tensor in [0, 1], shape (1, size, size)."""
    rng = np.random.default_rng(seed=seed)
    arr = rng.random(size=(1, size, size)).astype(np.float32)
    return torch.from_numpy(arr)


class MockDataset:
    """
    Minimal mock dataset for balancer tests.
    Exposes get_labels() and __len__() only.
    """
    def __init__(self, labels: list[int]) -> None:
        self._labels = labels

    def get_labels(self) -> list[int]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._labels)


# ══════════════════════════════════════════════════════════════════════════════
# GaussianNoise
# ══════════════════════════════════════════════════════════════════════════════

class TestGaussianNoise:

    def test_output_same_shape_as_input(self):
        noise = GaussianNoise()
        t = make_tensor(128)
        assert noise(t).shape == t.shape

    def test_output_same_dtype_as_input(self):
        noise = GaussianNoise()
        t = make_tensor()
        assert noise(t).dtype == t.dtype

    def test_output_clamped_min_zero(self):
        """Output must never be below 0.0."""
        noise = GaussianNoise(std_range=(0.5, 1.0))   # large noise to stress-test clamping
        t = torch.zeros(1, 128, 128)
        result = noise(t)
        assert result.min().item() >= 0.0

    def test_output_clamped_max_one(self):
        """Output must never exceed 1.0."""
        noise = GaussianNoise(std_range=(0.5, 1.0))
        t = torch.ones(1, 128, 128)
        result = noise(t)
        assert result.max().item() <= 1.0

    def test_clamping_with_extreme_noise(self):
        """Very high std should still produce valid output."""
        noise = GaussianNoise(std_range=(10.0, 20.0))
        t = make_tensor()
        result = noise(t)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_noise_actually_changes_values(self):
        """Noise must actually modify the tensor (not a no-op)."""
        noise = GaussianNoise(std_range=(0.1, 0.2))
        t = torch.full((1, 128, 128), 0.5)
        result = noise(t)
        assert not torch.equal(result, t)

    def test_different_calls_produce_different_outputs(self):
        """Each call samples a fresh noise tensor — outputs must differ."""
        noise = GaussianNoise(std_range=(0.1, 0.2))
        t = torch.full((1, 128, 128), 0.5)
        r1 = noise(t)
        r2 = noise(t)
        # Two independent calls should differ (probability of exact equality is ~0)
        assert not torch.equal(r1, r2)

    def test_zero_std_range_is_identity(self):
        """With std_range=(0, 0), mean=0: adds zero noise → output equals input."""
        noise = GaussianNoise(mean=0.0, std_range=(0.0, 0.0))
        t = make_tensor()
        result = noise(t)
        torch.testing.assert_close(result, t)

    def test_output_on_zero_tensor_clamped_correctly(self):
        """Black image + noise → some values rise, none fall below 0."""
        noise = GaussianNoise(std_range=(0.05, 0.1))
        t = torch.zeros(1, 128, 128)
        result = noise(t)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_repr_contains_class_name(self):
        noise = GaussianNoise(mean=0.0, std_range=(0.01, 0.05))
        assert "GaussianNoise" in repr(noise)

    def test_repr_contains_mean(self):
        noise = GaussianNoise(mean=0.0, std_range=(0.01, 0.05))
        assert "0.0" in repr(noise)

    def test_output_is_float32(self):
        noise = GaussianNoise()
        t = torch.rand(1, 64, 64, dtype=torch.float32)
        result = noise(t)
        assert result.dtype == torch.float32

    def test_works_on_batch_tensor(self):
        """Should work on (B, 1, H, W) tensors too (not just (1, H, W))."""
        noise = GaussianNoise()
        t = torch.rand(4, 1, 128, 128)
        result = noise(t)
        assert result.shape == (4, 1, 128, 128)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# get_train_transforms
# ══════════════════════════════════════════════════════════════════════════════

class TestGetTrainTransforms:

    def test_returns_compose(self):
        from torchvision.transforms import Compose
        assert isinstance(get_train_transforms(128), Compose)

    def test_output_tensor_shape(self):
        transform = get_train_transforms(128)
        pil = make_pil_grayscale(128)
        result = transform(pil)
        assert result.shape == (1, 128, 128)

    def test_output_dtype_float32(self):
        transform = get_train_transforms(128)
        result = transform(make_pil_grayscale(128))
        assert result.dtype == torch.float32

    def test_output_is_tensor(self):
        transform = get_train_transforms(128)
        result = transform(make_pil_grayscale(128))
        assert isinstance(result, torch.Tensor)

    def test_output_range_after_normalize(self):
        """
        After Normalize(mean=[0.5], std=[0.5]) the theoretical range is [-1, 1].
        With GaussianNoise applied before Normalize, the values could be slightly
        outside [-1, 1] but GaussianNoise clamps to [0, 1] first, so after Normalize
        output is in [-1, 1].
        We check a relaxed range to account for float precision.
        """
        transform = get_train_transforms(128)
        for seed in range(5):
            pil = make_pil_grayscale(128, seed=seed)
            result = transform(pil)
            assert result.min().item() >= -1.05, f"seed={seed}: min={result.min().item()}"
            assert result.max().item() <= 1.05,  f"seed={seed}: max={result.max().item()}"

    def test_pure_black_image_normalizes_to_minus_one(self):
        """
        Black image (all zeros) → ToTensor → [0.0] → GaussianNoise adds small noise
        → clamped back → Normalize(0.5, 0.5): (0 - 0.5) / 0.5 = -1.0
        With noise the result is close to -1.0 but not exact.
        """
        transform = get_train_transforms(128)
        black = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='L')
        # Run multiple times; all should be close to -1
        for _ in range(3):
            result = transform(black)
            assert result.mean().item() > -1.1   # with noise it won't be exactly -1

    def test_accepts_pil_l_mode_input(self):
        """Transform must work on PIL Image mode 'L' without raising."""
        transform = get_train_transforms(128)
        pil = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='L')
        result = transform(pil)   # must not raise
        assert result is not None

    def test_stochastic_produces_different_results_on_same_input(self):
        """
        Training transforms are stochastic. Two calls on identical input
        should produce different tensors (with overwhelming probability).
        """
        transform = get_train_transforms(128)
        pil = make_pil_grayscale(128, seed=7)
        outputs = [transform(pil) for _ in range(5)]
        # At least two outputs should differ
        all_equal = all(torch.equal(outputs[0], o) for o in outputs[1:])
        assert not all_equal, "Training transforms appear to be deterministic — check stochastic stages"

    def test_contains_gaussian_noise(self):
        """Pipeline must include GaussianNoise after ToTensor."""
        transform = get_train_transforms(128)
        has_gaussian = any(isinstance(t, GaussianNoise) for t in transform.transforms)
        assert has_gaussian, "get_train_transforms must include GaussianNoise"

    def test_contains_to_tensor(self):
        from torchvision.transforms import ToTensor
        transform = get_train_transforms(128)
        has_to_tensor = any(isinstance(t, ToTensor) for t in transform.transforms)
        assert has_to_tensor, "get_train_transforms must include ToTensor"

    def test_contains_normalize(self):
        from torchvision.transforms import Normalize
        transform = get_train_transforms(128)
        has_normalize = any(isinstance(t, Normalize) for t in transform.transforms)
        assert has_normalize, "get_train_transforms must include Normalize"

    def test_gaussian_noise_comes_after_to_tensor(self):
        """
        GaussianNoise must be positioned after ToTensor in the pipeline.
        Verify by checking index positions.
        """
        from torchvision.transforms import ToTensor
        transform = get_train_transforms(128)
        ts = transform.transforms
        to_tensor_idx   = next(i for i, t in enumerate(ts) if isinstance(t, ToTensor))
        gaussian_idx    = next(i for i, t in enumerate(ts) if isinstance(t, GaussianNoise))
        assert gaussian_idx > to_tensor_idx, (
            f"GaussianNoise (idx={gaussian_idx}) must come after "
            f"ToTensor (idx={to_tensor_idx})"
        )

    def test_normalize_is_last_transform(self):
        from torchvision.transforms import Normalize
        transform = get_train_transforms(128)
        last = transform.transforms[-1]
        assert isinstance(last, Normalize), (
            f"Last transform must be Normalize, got {type(last).__name__}"
        )

    def test_color_jitter_comes_before_to_tensor(self):
        """ColorJitter operates on PIL — must be before ToTensor."""
        from torchvision.transforms import ColorJitter, ToTensor
        transform = get_train_transforms(128)
        ts = transform.transforms
        to_tensor_idx = next(i for i, t in enumerate(ts) if isinstance(t, ToTensor))
        jitter_indices = [i for i, t in enumerate(ts) if isinstance(t, ColorJitter)]
        assert len(jitter_indices) > 0, "get_train_transforms must include ColorJitter"
        assert all(idx < to_tensor_idx for idx in jitter_indices), (
            "ColorJitter must come before ToTensor"
        )

    def test_more_transforms_than_val(self):
        """Train pipeline has more transforms than val pipeline."""
        train_tf = get_train_transforms(128)
        val_tf   = get_val_transforms(128)
        assert len(train_tf.transforms) > len(val_tf.transforms)

    def test_normalize_uses_single_channel_params(self):
        """Normalize must use mean=[0.5], std=[0.5] — single-element lists."""
        from torchvision.transforms import Normalize
        transform = get_train_transforms(128)
        normalize = next(t for t in transform.transforms if isinstance(t, Normalize))
        assert list(normalize.mean) == [0.5], f"Expected mean=[0.5], got {normalize.mean}"
        assert list(normalize.std)  == [0.5], f"Expected std=[0.5],  got {normalize.std}"


# ══════════════════════════════════════════════════════════════════════════════
# get_val_transforms
# ══════════════════════════════════════════════════════════════════════════════

class TestGetValTransforms:

    def test_returns_compose(self):
        from torchvision.transforms import Compose
        assert isinstance(get_val_transforms(128), Compose)

    def test_output_tensor_shape(self):
        transform = get_val_transforms(128)
        result = transform(make_pil_grayscale(128))
        assert result.shape == (1, 128, 128)

    def test_output_dtype_float32(self):
        transform = get_val_transforms(128)
        result = transform(make_pil_grayscale(128))
        assert result.dtype == torch.float32

    def test_is_deterministic(self):
        """Same input must always produce exactly the same tensor."""
        transform = get_val_transforms(128)
        arr = np.random.default_rng(seed=42).integers(0,256,(128,128),dtype=np.uint8)
        pil1 = Image.fromarray(arr, mode='L')
        pil2 = Image.fromarray(arr, mode='L')
        r1 = transform(pil1)
        r2 = transform(pil2)
        torch.testing.assert_close(r1, r2)

    def test_pure_black_image_outputs_minus_one(self):
        """Black image → ToTensor → 0.0 → Normalize(0.5,0.5) → -1.0"""
        transform = get_val_transforms(128)
        black = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='L')
        result = transform(black)
        torch.testing.assert_close(result, torch.full((1, 128, 128), -1.0))

    def test_pure_white_image_outputs_plus_one(self):
        """White image → ToTensor → 1.0 → Normalize(0.5,0.5) → 1.0"""
        transform = get_val_transforms(128)
        white = Image.fromarray(np.full((128, 128), 255, dtype=np.uint8), mode='L')
        result = transform(white)
        torch.testing.assert_close(result, torch.full((1, 128, 128), 1.0), atol=1e-5, rtol=0)

    def test_output_range_is_minus_one_to_plus_one(self):
        transform = get_val_transforms(128)
        for seed in range(5):
            result = transform(make_pil_grayscale(128, seed=seed))
            assert result.min().item() >= -1.0 - 1e-5
            assert result.max().item() <= 1.0  + 1e-5

    def test_does_not_contain_gaussian_noise(self):
        transform = get_val_transforms(128)
        has_gaussian = any(isinstance(t, GaussianNoise) for t in transform.transforms)
        assert not has_gaussian, "get_val_transforms must NOT include GaussianNoise"

    def test_does_not_contain_random_flip(self):
        from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
        transform = get_val_transforms(128)
        for t in transform.transforms:
            assert not isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip)), \
                "get_val_transforms must not contain random flips"

    def test_does_not_contain_random_rotation(self):
        from torchvision.transforms import RandomRotation
        transform = get_val_transforms(128)
        for t in transform.transforms:
            assert not isinstance(t, RandomRotation), \
                "get_val_transforms must not contain RandomRotation"

    def test_does_not_contain_color_jitter(self):
        from torchvision.transforms import ColorJitter
        transform = get_val_transforms(128)
        for t in transform.transforms:
            assert not isinstance(t, ColorJitter), \
                "get_val_transforms must not contain ColorJitter"

    def test_exactly_two_transforms(self):
        """Val pipeline must have exactly: ToTensor, Normalize."""
        transform = get_val_transforms(128)
        assert len(transform.transforms) == 2, (
            f"Expected exactly 2 transforms, got {len(transform.transforms)}: "
            f"{[type(t).__name__ for t in transform.transforms]}"
        )

    def test_normalize_uses_single_channel_params(self):
        from torchvision.transforms import Normalize
        transform = get_val_transforms(128)
        normalize = next(t for t in transform.transforms if isinstance(t, Normalize))
        assert list(normalize.mean) == [0.5]
        assert list(normalize.std)  == [0.5]

    def test_accepts_pil_l_mode_input(self):
        transform = get_val_transforms(128)
        pil = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='L')
        result = transform(pil)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# ClassAwareOversampler
# ══════════════════════════════════════════════════════════════════════════════

class TestClassAwareOversamplerInit:

    def test_invalid_strategy_raises_value_error(self):
        ds = MockDataset([0, 1, 2])
        with pytest.raises(ValueError, match="Unknown oversampling strategy"):
            ClassAwareOversampler(ds, strategy="nonexistent")

    def test_valid_strategies_do_not_raise(self):
        ds = MockDataset([0, 1, 2])
        for strategy in ["oversample_minority", "sqrt_inverse", "uniform"]:
            ClassAwareOversampler(ds, strategy=strategy)   # must not raise

    def test_stores_strategy(self):
        ds = MockDataset([0, 1, 2])
        o = ClassAwareOversampler(ds, strategy="uniform")
        assert o.strategy == "uniform"

    def test_default_strategy_is_oversample_minority(self):
        ds = MockDataset([0, 1, 2])
        o = ClassAwareOversampler(ds)
        assert o.strategy == "oversample_minority"


class TestClassAwareOversamplerGetSampler:

    def test_returns_weighted_random_sampler(self):
        ds = MockDataset([0] * 100 + [1] * 10 + [2] * 5)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_num_samples_equals_dataset_length(self):
        labels = [0] * 100 + [1] * 10 + [2] * 5
        ds = MockDataset(labels)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert sampler.num_samples == len(labels)

    def test_replacement_is_true(self):
        ds = MockDataset([0] * 50 + [1] * 5)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert sampler.replacement is True

    def test_class_weights_set_as_attribute(self):
        """class_weights dict must be accessible after get_sampler() is called."""
        ds = MockDataset([0] * 100 + [1] * 10)
        o = ClassAwareOversampler(ds)
        o.get_sampler()
        assert hasattr(o, 'class_weights')
        assert isinstance(o.class_weights, dict)
        assert 0 in o.class_weights
        assert 1 in o.class_weights

    def test_oversample_minority_weights_minority_higher(self):
        """Class 1 (10 samples) must get higher weight than class 0 (100 samples)."""
        ds = MockDataset([0] * 100 + [1] * 10)
        o = ClassAwareOversampler(ds, strategy="oversample_minority")
        o.get_sampler()
        assert o.class_weights[1] > o.class_weights[0], (
            f"Minority class weight {o.class_weights[1]} should be > "
            f"majority class weight {o.class_weights[0]}"
        )

    def test_oversample_minority_weight_formula(self):
        """Weight must be exactly 1.0 / count for each class."""
        labels = [0] * 100 + [1] * 20 + [2] * 5
        ds = MockDataset(labels)
        o = ClassAwareOversampler(ds, strategy="oversample_minority")
        o.get_sampler()
        assert abs(o.class_weights[0] - 1.0 / 100) < 1e-9
        assert abs(o.class_weights[1] - 1.0 / 20)  < 1e-9
        assert abs(o.class_weights[2] - 1.0 / 5)   < 1e-9

    def test_sqrt_inverse_weight_formula(self):
        """Weight must be exactly 1.0 / sqrt(count) for each class."""
        import math
        labels = [0] * 100 + [1] * 25
        ds = MockDataset(labels)
        o = ClassAwareOversampler(ds, strategy="sqrt_inverse")
        o.get_sampler()
        assert abs(o.class_weights[0] - 1.0 / math.sqrt(100)) < 1e-9
        assert abs(o.class_weights[1] - 1.0 / math.sqrt(25))  < 1e-9

    def test_sqrt_inverse_minority_still_weighted_higher(self):
        """Even with sqrt_inverse, minority should weigh more than majority."""
        ds = MockDataset([0] * 100 + [1] * 10)
        o = ClassAwareOversampler(ds, strategy="sqrt_inverse")
        o.get_sampler()
        assert o.class_weights[1] > o.class_weights[0]

    def test_uniform_all_weights_equal_one(self):
        labels = [0] * 100 + [1] * 10 + [2] * 5
        ds = MockDataset(labels)
        o = ClassAwareOversampler(ds, strategy="uniform")
        o.get_sampler()
        for cls, weight in o.class_weights.items():
            assert abs(weight - 1.0) < 1e-9, f"Class {cls} weight should be 1.0, got {weight}"

    def test_uniform_sampler_weights_per_sample_all_equal(self):
        """With uniform strategy, every sample has the same weight."""
        labels = [0] * 100 + [1] * 10
        ds = MockDataset(labels)
        sampler = ClassAwareOversampler(ds, strategy="uniform").get_sampler()
        weights = sampler.weights
        assert torch.allclose(weights, torch.ones_like(weights)), \
            "Uniform strategy should give all samples weight 1.0"

    def test_sample_weights_tensor_is_float32(self):
        ds = MockDataset([0] * 50 + [1] * 50)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert sampler.weights.dtype == torch.float32

    def test_sample_weights_length_equals_dataset(self):
        labels = [0] * 80 + [1] * 20
        ds = MockDataset(labels)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert len(sampler.weights) == len(labels)

    def test_all_sample_weights_positive(self):
        """No sample should have zero or negative weight."""
        labels = [0] * 100 + [1] * 10 + [2] * 5
        ds = MockDataset(labels)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert (sampler.weights > 0).all()

    def test_works_with_single_class(self):
        """Edge case: all samples belong to one class."""
        ds = MockDataset([0] * 50)
        sampler = ClassAwareOversampler(ds).get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == 50

    def test_works_with_25_classes(self, sample_class_names):
        """Stress test with Malimg-like 25-class imbalanced distribution."""
        rng = np.random.default_rng(seed=0)
        # Simulate Malimg imbalance: classes 0-24 with varying counts
        labels = []
        counts = [2949, 1591, 800, 431, 408, 381, 213, 200, 198, 184,
                  177, 162, 159, 158, 146, 142, 136, 132, 128, 123,
                  122, 116, 106, 97, 80]
        for cls, count in enumerate(counts):
            labels.extend([cls] * count)
        ds = MockDataset(labels)
        sampler = ClassAwareOversampler(ds, strategy="oversample_minority").get_sampler()
        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == len(labels)

    def test_minority_class_has_highest_weight_in_25_class(self, sample_class_names):
        """Skintrim.N (class 24, 80 samples) should have highest weight."""
        counts = [2949, 1591, 800, 431, 408, 381, 213, 200, 198, 184,
                  177, 162, 159, 158, 146, 142, 136, 132, 128, 123,
                  122, 116, 106, 97, 80]
        labels = []
        for cls, count in enumerate(counts):
            labels.extend([cls] * count)
        ds = MockDataset(labels)
        o = ClassAwareOversampler(ds, strategy="oversample_minority")
        o.get_sampler()
        max_weight_class = max(o.class_weights, key=o.class_weights.get)
        # Class 24 has 80 samples (fewest) → highest weight
        assert max_weight_class == 24, (
            f"Expected class 24 (80 samples) to have highest weight, "
            f"got class {max_weight_class}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Integration: transforms + oversampler work together
# ══════════════════════════════════════════════════════════════════════════════

class TestTransformOversamplerIntegration:

    def test_train_transforms_output_compatible_with_model_input(self):
        """
        Output of get_train_transforms must be directly usable as CNN input.
        Shape: (1, 128, 128) — single channel, 128x128.
        dtype: float32.
        """
        transform = get_train_transforms(128)
        pil = make_pil_grayscale(128)
        tensor = transform(pil)
        assert tensor.shape == (1, 128, 128)
        assert tensor.dtype == torch.float32

    def test_val_transforms_output_compatible_with_model_input(self):
        transform = get_val_transforms(128)
        pil = make_pil_grayscale(128)
        tensor = transform(pil)
        assert tensor.shape == (1, 128, 128)
        assert tensor.dtype == torch.float32

    def test_train_and_val_differ_on_same_input(self):
        """
        Training transforms are stochastic. They should (almost certainly)
        produce different output than val transforms on the same input.
        Test across multiple seeds for robustness.
        """
        train_tf = get_train_transforms(128)
        val_tf   = get_val_transforms(128)
        found_difference = False
        for seed in range(10):
            pil_for_train = make_pil_grayscale(128, seed=seed)
            pil_for_val   = make_pil_grayscale(128, seed=seed)   # identical image
            t_out = train_tf(pil_for_train)
            v_out = val_tf(pil_for_val)
            if not torch.equal(t_out, v_out):
                found_difference = True
                break
        assert found_difference, (
            "train and val transforms produced identical output for 10 seeds — "
            "training transforms appear non-stochastic"
        )

    def test_oversampler_can_be_used_with_dataloader(self):
        """
        WeightedRandomSampler from ClassAwareOversampler must be accepted by
        torch.utils.data.DataLoader without error.
        """
        from torch.utils.data import DataLoader, TensorDataset

        # Build a tiny fake dataset with 3 classes (20 + 5 + 2 = 27 samples)
        labels = [0] * 20 + [1] * 5 + [2] * 2

        class FakeDataset:
            def __init__(self, labels):
                self._labels = labels
                self._data = [
                    (torch.zeros(1, 128, 128), lbl)
                    for lbl in labels
                ]
            def get_labels(self):
                return list(self._labels)
            def __len__(self):
                return len(self._labels)
            def __getitem__(self, idx):
                return self._data[idx]

        ds = FakeDataset(labels)
        sampler = ClassAwareOversampler(ds, strategy="oversample_minority").get_sampler()
        loader = DataLoader(ds, batch_size=4, sampler=sampler)

        # Draw one batch — must not raise
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (4, 1, 128, 128)
        assert batch_labels.shape == (4,)

