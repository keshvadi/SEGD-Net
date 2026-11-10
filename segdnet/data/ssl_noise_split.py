"""Stratified label-budget selection and train-only label corruption."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSLPartition:
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray
    observed_labels: np.ndarray
    clean_labels: np.ndarray


def create_ssl_partition(
    labels: np.ndarray,
    labeled_fraction: float = 0.30,
    noise_rate: float = 0.0,
    noise_mode: str = "symmetric",
    seed: int = 17,
) -> SSLPartition:
    """Stratify labeled selection, then corrupt only selected training labels."""
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if labels.ndim != 1 or len(classes) < 2:
        raise ValueError(
            "labels must be a one-dimensional array with at least two classes"
        )
    if not 0 < labeled_fraction <= 1:
        raise ValueError("labeled_fraction must be in (0, 1]")
    if not 0 <= noise_rate < 1:
        raise ValueError("noise_rate must be in [0, 1)")
    if noise_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("noise_mode must be 'symmetric' or 'asymmetric'")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in classes:
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        count = max(1, round(len(class_indices) * labeled_fraction))
        selected.extend(class_indices[:count].tolist())
    labeled = np.asarray(sorted(selected), dtype=np.int64)
    unlabeled = np.setdiff1d(np.arange(len(labels)), labeled, assume_unique=True)
    observed = labels.copy()
    corrupt_count = round(len(labeled) * noise_rate)
    if corrupt_count:
        corrupt_indices = rng.choice(labeled, size=corrupt_count, replace=False)
        for index in corrupt_indices:
            alternatives = classes[classes != labels[index]]
            if noise_mode == "asymmetric" and len(classes) > 2:
                target = classes[
                    (np.where(classes == labels[index])[0][0] + 1) % len(classes)
                ]
                if target == labels[index]:
                    target = alternatives[0]
            else:
                target = rng.choice(alternatives)
            observed[index] = target
    return SSLPartition(labeled, unlabeled, observed, labels.copy())
