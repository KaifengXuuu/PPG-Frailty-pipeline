"""OrderedFeatureMatrixV1 合同检查 / OrderedFeatureMatrixV1 validation."""

import hashlib
import numpy as np

from ..contracts import OrderedFeatureMatrixV1


def validate_feature_matrix(matrix: OrderedFeatureMatrixV1) -> OrderedFeatureMatrixV1:
    """强制 D×32、mask 和 neutral padding / Enforce D-by-32 and neutral padding."""

    from ..features.engineering import (
        ENGINEERING_SCHEMA_VERSION,
        engineering_feature_names,
    )
    from ..features.registry import (
        ORDERED_MATRIX_SCHEMA_VERSION,
        default_registry,
    )

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
    registry = default_registry()
    engineering_names = engineering_feature_names()
    context_schema = registry.names + tuple(
        f"{name}.validity" for name in registry.names
    )
    expected_channel_schema = (
        engineering_names
        + tuple(f"{name}.validity" for name in engineering_names)
        + context_schema
    )
    channel_hash = hashlib.sha256(
        "\n".join(expected_channel_schema).encode("utf-8")
    ).hexdigest()
    context_hash = hashlib.sha256(
        "\n".join(context_schema).encode("utf-8")
    ).hexdigest()
    provenance = matrix.provenance
    if (
        matrix.schema_version != ORDERED_MATRIX_SCHEMA_VERSION
        or tuple(matrix.channel_schema) != expected_channel_schema
        or tuple(matrix.context_schema) != context_schema
        or provenance.get("matrix_channel_schema_sha256") != channel_hash
        or provenance.get("context_schema_sha256") != context_hash
        or provenance.get("context_registry_sha256") != registry.sha256
        or provenance.get("engineering_transform_version")
        != ENGINEERING_SCHEMA_VERSION + "+fold_robust_v1"
    ):
        raise ValueError(
            "OrderedFeatureMatrixV1 uses a stale or inconsistent formal schema"
        )
    return matrix


__all__ = ["validate_feature_matrix"]
