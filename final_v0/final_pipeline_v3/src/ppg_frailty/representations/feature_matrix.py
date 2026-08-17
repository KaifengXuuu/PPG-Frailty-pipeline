"""OrderedFeatureMatrixV1 合同检查 / OrderedFeatureMatrixV1 validation."""

import numpy as np

from ..contracts import OrderedFeatureMatrixV1


def validate_feature_matrix(matrix: OrderedFeatureMatrixV1) -> OrderedFeatureMatrixV1:
    """强制 D×32、mask 和 neutral padding / Enforce D-by-32 and neutral padding."""

    values = np.asarray(matrix.values, dtype=np.float64)
    mask = np.asarray(matrix.row_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 32 or mask.shape != (32,):
        raise ValueError("OrderedFeatureMatrixV1 must be [D,32] with row_mask[32]")
    if len(matrix.channel_schema) != values.shape[0] or len(set(matrix.channel_schema)) != values.shape[0]:
        raise ValueError("matrix channel schema must uniquely name D channels")
    if not np.isfinite(values).all() or not np.any(mask):
        raise ValueError("matrix must be finite with at least one valid position")
    if np.any(values[:, ~mask] != 0.0):
        raise ValueError("post-transform padded positions must be standardized neutral zero")
    return matrix


__all__ = ["validate_feature_matrix"]
