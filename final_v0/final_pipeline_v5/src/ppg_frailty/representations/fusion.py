"""文件级融合池化 / File-level fusion pooling."""

import numpy as np

def masked_file_mean(embeddings: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
    """每文件只池化一次 / Pool each file exactly once with a window mask."""

    values = np.asarray(embeddings, dtype=np.float64)
    mask = np.asarray(window_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != (values.shape[0],) or not np.any(mask):
        raise ValueError("embeddings [window,feature] require a non-empty window mask")
    if not np.isfinite(values[mask]).all():
        raise ValueError("valid window embeddings must be finite")
    return np.mean(values[mask], axis=0)


__all__ = ["masked_file_mean"]
