# modules/enhancement/balancer.py
import math
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


class ClassAwareOversampler:
    """
    Produces a WeightedRandomSampler to address class imbalance in Malimg.

    Malimg is severely imbalanced (Allaple.A has ~2949 samples, Skintrim.N has ~80).
    Without balancing, the CNN learns to predict majority classes and performs
    poorly on rare families.

    Constructor args:
        dataset:  a MalimgDataset instance (train split).
                  Must expose a get_labels() method returning list[int].
        strategy: one of 'oversample_minority', 'sqrt_inverse', 'uniform'.

    Strategies:
        'oversample_minority' — weight = 1 / class_count (pure inverse frequency)
        'sqrt_inverse'        — weight = 1 / sqrt(class_count) (softer balancing)
        'uniform'             — weight = 1.0 for all samples (effectively random sampling)

    Properties set after get_sampler() call:
        self.class_weights: dict[int, float]
        self.effective_class_counts: dict[int, float]
    """

    def __init__(self, dataset, strategy: str = 'oversample_minority'):
        self.dataset = dataset
        self.strategy = strategy
        self.class_weights: dict[int, float] = {}
        self.effective_class_counts: dict[int, float] = {}

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Compute per-sample weights and return a WeightedRandomSampler.
        """
        labels = self.dataset.get_labels()
        class_counts = Counter(labels)

        if self.strategy == 'oversample_minority':
            self.class_weights = {c: 1.0 / count for c, count in class_counts.items()}
        elif self.strategy == 'sqrt_inverse':
            self.class_weights = {c: 1.0 / math.sqrt(count) for c, count in class_counts.items()}
        elif self.strategy == 'uniform':
            self.class_weights = {c: 1.0 for c in class_counts}
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. "
                             "Choose from: oversample_minority, sqrt_inverse, uniform")

        total_weight = sum(self.class_weights.values())
        n = len(labels)
        self.effective_class_counts = {
            c: self.class_weights[c] / total_weight * n
            for c in self.class_weights
        }

        sample_weights = torch.tensor(
            [self.class_weights[label] for label in labels],
            dtype=torch.float32,
        )

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

