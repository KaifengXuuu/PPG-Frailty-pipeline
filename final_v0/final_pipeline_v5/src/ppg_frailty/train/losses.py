"""训练标签权重工具 / Training-label weight utilities."""

import numpy as np

def inverse_frequency_class_weights(labels: np.ndarray, *, n_classes: int = 3) -> np.ndarray:
    """仅由 outer-train 标签计算权重 / Compute weights from outer-train labels only."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values >= n_classes)):
        raise ValueError("labels must be a non-empty registered class vector")
    counts = np.bincount(values, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("every class must occur in the outer-training partition")
    weights = values.size / (n_classes * counts.astype(np.float64))
    return weights


__all__ = ["inverse_frequency_class_weights"]
