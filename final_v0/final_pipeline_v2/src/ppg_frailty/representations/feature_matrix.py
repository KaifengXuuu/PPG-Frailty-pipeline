"""OrderedFeatureMatrixV1 合同检查 / OrderedFeatureMatrixV1 validation."""

import hashlib
import numpy as np

from ..contracts import OrderedFeatureMatrixV1


def validate_feature_matrix(matrix: OrderedFeatureMatrixV1) -> OrderedFeatureMatrixV1:
    """强制 D×K、mask 和 neutral padding / Enforce D-by-K and neutral padding."""

    from ..features.engineering import (
        ENGINEERING_SCHEMA_VERSION,
        engineering_feature_names,
    )
    from ..features.registry import ordered_matrix_schema_version

    values = np.asarray(matrix.values, dtype=np.float64)
    mask = np.asarray(matrix.row_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] <= 0 or mask.shape != (values.shape[1],):
        raise ValueError("OrderedFeatureMatrixV1 must be [D,K] with row_mask[K]")
    matrix_k = int(values.shape[1])
    if len(matrix.channel_schema) != values.shape[0] or len(set(matrix.channel_schema)) != values.shape[0]:
        raise ValueError("matrix channel schema must uniquely name D channels")
    if not np.isfinite(values).all() or not np.any(mask):
        raise ValueError("matrix must be finite with at least one valid position")
    if np.any(values[:, ~mask] != 0.0):
        raise ValueError("post-transform padded positions must be standardized neutral zero")
    if matrix_k != 150 or tuple(matrix.context_schema):
        raise ValueError("formal feature matrix must be 115-by-150 without context channels")
    expected_schema_version = ordered_matrix_schema_version(matrix_k)
    engineering_names = engineering_feature_names()
    expected_channel_schema = engineering_names
    channel_hash = hashlib.sha256(
        "\n".join(expected_channel_schema).encode("utf-8")
    ).hexdigest()
    provenance = matrix.provenance
    if (
        matrix.schema_version != expected_schema_version
        or provenance.get("matrix_k") != matrix_k
        or provenance.get("matrix_schema_version") != expected_schema_version
        or tuple(matrix.channel_schema) != expected_channel_schema
        or tuple(matrix.context_schema)
        or provenance.get("matrix_channel_schema_sha256") != channel_hash
        or provenance.get("validity_encoding")
        != "provenance_only_not_predictor_channels_v1"
        or provenance.get("engineering_transform_version")
        != ENGINEERING_SCHEMA_VERSION + "+fold_robust_v1"
    ):
        raise ValueError(
            "OrderedFeatureMatrixV1 uses a stale or inconsistent formal schema"
        )
    return matrix


__all__ = ["validate_feature_matrix"]
