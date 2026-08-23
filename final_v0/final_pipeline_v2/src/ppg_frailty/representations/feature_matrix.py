"""OrderedFeatureMatrixV1 合同检查 / OrderedFeatureMatrixV1 validation."""

import hashlib
import numpy as np

from ..contracts import OrderedFeatureMatrixV1


def validate_feature_matrix(matrix: OrderedFeatureMatrixV1) -> OrderedFeatureMatrixV1:
    """强制 D×K、mask 和 neutral padding / Enforce D-by-K and neutral padding."""

    from ..features.window_matrix import (
        ORDERED_WINDOW_MATRIX_SCHEMA_VERSION,
        window_feature_names,
    )

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
    if values.shape[0] != 146 or tuple(matrix.context_schema):
        raise ValueError("formal feature matrix must be 146-by-variable-K without context")
    expected_schema_version = ORDERED_WINDOW_MATRIX_SCHEMA_VERSION
    expected_channel_schema = window_feature_names()
    channel_hash = hashlib.sha256(
        "\n".join(expected_channel_schema).encode("utf-8")
    ).hexdigest()
    provenance = matrix.provenance
    if (
        matrix.schema_version != expected_schema_version
        or provenance.get("matrix_k") != matrix_k
        or provenance.get("matrix_length_policy")
        != "all_complete_windows_variable_k"
        or provenance.get("matrix_schema_version") != expected_schema_version
        or tuple(matrix.channel_schema) != expected_channel_schema
        or tuple(matrix.context_schema)
        or provenance.get("matrix_channel_schema_sha256") != channel_hash
        or provenance.get("validity_encoding")
        != "provenance_only_not_predictor_channels_v1"
        or provenance.get("padding_policy")
        != "none_at_record_storage_batch_only"
        or provenance.get("unavailable_after_transform")
        != "outer_train_center_zero"
    ):
        raise ValueError(
            "OrderedFeatureMatrixV1 uses a stale or inconsistent formal schema"
        )
    return matrix


__all__ = ["validate_feature_matrix"]
